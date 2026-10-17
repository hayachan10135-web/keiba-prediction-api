# -*- coding: utf-8 -*-
"""
app/schedule/races.py

出走スケジュール登録・リアルタイム発走時刻同期・レース一覧バッジ付与（仮想枠/実戦枠調停）エンジン
"""

import re
import time
import datetime
from typing import Optional, Callable, Dict, Any, List, Set, Tuple
from fastapi import Query

from app.core.database import get_supabase_client
from app.core.timezone import now_jst_naive, today_jst_str
from app.scrapers import (
    fetch_venues_for_date_raw,
    register_crawler_post_time_handler,
    register_venues_loaded_handler,
    register_post_time_callback,
    register_schedule_callback,
)
from app.schedule.period import (
    get_current_kaisai_period,
    discover_period_kaisai_dates,
    _latest_post_times,
)

# レース一覧キャッシュ
_today_races_cache: Dict[str, Tuple[float, dict]] = {}

# 外部DI
_ensure_daily_single_real_bet_fn: Optional[Callable] = None


def set_races_dependencies(
    ensure_daily_single_real_bet: Callable,
):
    """レーススケジュール管理が必要とする外部依存を登録"""
    global _ensure_daily_single_real_bet_fn
    _ensure_daily_single_real_bet_fn = ensure_daily_single_real_bet


def register_race_schedule(race_id: str, race_date: str, race_name: str, post_time_str: str):
    """出走スケジュールを keiba_scheduled_races に登録"""
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        m = re.search(r"(\d{1,2}):(\d{2})", post_time_str)
        if not m:
            return
        hour, minute = int(m.group(1)), int(m.group(2))

        d_parts = [int(p) for p in race_date.split("-")]
        post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
        post_dt_str = post_dt.strftime("%Y-%m-%d %H:%M:%S")

        existing = supabase.table("keiba_scheduled_races").select("race_id, is_settled, is_predicted").eq("race_id", race_id).execute()
        if existing.data and len(existing.data) > 0:
            supabase.table("keiba_scheduled_races").update({
                "race_date": race_date,
                "race_name": race_name,
                "post_datetime": post_dt_str
            }).eq("race_id", race_id).execute()
        else:
            supabase.table("keiba_scheduled_races").insert({
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "post_datetime": post_dt_str,
                "is_settled": 0,
                "is_predicted": 0
            }).execute()
    except Exception as e:
        print(f"Supabaseスケジュール登録エラー ({race_id}): {e}")


def ensure_schedules_registered_for_venues(formatted_date: str, venues: List[dict]):
    """レース一覧取得時に実際の最新発走時刻を keiba_scheduled_races および _latest_post_times に同期（発走時刻変更対応）"""
    if not venues:
        return
    try:
        d_parts = [int(p) for p in formatted_date.split("-")]
        updates_for_schedule = []
        for v in venues:
            for r in v.get("races", []):
                r_id = r.get("race_id")
                if not r_id:
                    continue
                r_info = r.get("race_info") or r.get("race_name") or ""
                m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                if m:
                    hour, minute = int(m.group(1)), int(m.group(2))
                    post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
                    post_dt_str = post_dt.strftime("%Y-%m-%d %H:%M:%S")
                    post_time_str = f"{hour:02d}:{minute:02d}発走"
                    _latest_post_times[r_id] = post_time_str
                    updates_for_schedule.append({
                        "race_id": r_id,
                        "race_date": formatted_date,
                        "race_name": r.get("race_name", r.get("race_no", "")),
                        "post_datetime": post_dt_str,
                        "post_time_str": post_time_str
                    })

        supabase = get_supabase_client()
        if supabase and updates_for_schedule:
            rids_list = [row["race_id"] for row in updates_for_schedule]
            existing_rids = set()
            try:
                ex_res = supabase.table("keiba_scheduled_races").select("race_id").in_("race_id", rids_list).execute()
                existing_rids = {er["race_id"] for er in (ex_res.data or []) if "race_id" in er}
            except Exception:
                pass
            for row in updates_for_schedule:
                try:
                    if row["race_id"] in existing_rids:
                        supabase.table("keiba_scheduled_races").update({
                            "race_date": row["race_date"],
                            "post_datetime": row["post_datetime"],
                            "race_name": row["race_name"]
                        }).eq("race_id", row["race_id"]).execute()
                    else:
                        supabase.table("keiba_scheduled_races").insert({
                            "race_id": row["race_id"],
                            "race_date": row["race_date"],
                            "race_name": row["race_name"],
                            "post_datetime": row["post_datetime"],
                            "is_predicted": 0,
                            "is_settled": 0,
                            "retry_count": 0
                        }).execute()
                except Exception:
                    pass
    except Exception as e:
        print(f"スケジュール一括同期エラー ({formatted_date}): {e}")


def _on_crawler_post_time(r_id: str, pt_str: str):
    _latest_post_times[r_id] = pt_str

def _on_crawler_schedule_registered(r_id: str, target_date: str, r_name: str, pt_str: str):
    register_race_schedule(r_id, target_date, r_name, pt_str)

def _on_crawler_venues_loaded(formatted_date: str, venues: List[dict]):
    ensure_schedules_registered_for_venues(formatted_date, venues)

register_crawler_post_time_handler(_on_crawler_post_time)
register_venues_loaded_handler(_on_crawler_venues_loaded)
register_post_time_callback(_on_crawler_post_time)
register_schedule_callback(_on_crawler_schedule_registered)


def attach_prediction_badges_to_venues(formatted_date: str, venues: List[dict]) -> List[dict]:
    """レース一覧の各レースにSupabase上の予想・実戦推奨バッジ情報および確定・的中結果を付与"""
    if not venues:
        return []
    cached_preds_map = {}
    raw_preds_cache = {}
    settled_map = {}
    supabase = get_supabase_client()
    if supabase:
        try:
            p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
            for row in (p_res.data or []):
                d_json = row.get("data_json") or {}
                raw_preds_cache[row["race_id"]] = d_json
                h_rec = d_json.get("hybrid_recommendation")
                r_name = d_json.get("race_name", "")
                if h_rec:
                    cached_preds_map[row["race_id"]] = {
                        "is_real_bet_target": h_rec.get("is_real_bet_target", False),
                        "real_bet_badge": h_rec.get("real_bet_badge"),
                        "real_bet_level": h_rec.get("real_bet_level", "none"),
                        "edge_ratio": h_rec.get("edge_ratio"),
                        "is_final_odds": h_rec.get("is_final_odds", False),
                        "confidence_score": float(h_rec.get("confidence_score") or 0.0),
                        "is_virtual_bet_target": h_rec.get("is_virtual_bet_target", False),
                        "virtual_bet_slot": h_rec.get("virtual_bet_slot"),
                        "virtual_badge": h_rec.get("virtual_badge"),
                        "race_name": r_name
                    }
        except Exception as e:
            print(f"レース一覧キャッシュ付与エラー: {e}")

        try:
            s_res = supabase.table("keiba_race_settlements").select("race_id, is_hit, is_pass, total_payout").eq("race_date", formatted_date).execute()
            for s_row in (s_res.data or []):
                is_hit_val = 1 if (s_row.get("is_hit") == 1 and s_row.get("is_pass") == 0) else 0
                settled_map[s_row["race_id"]] = {
                    "is_settled": 1,
                    "is_hit": is_hit_val,
                    "is_pass": s_row.get("is_pass", 0),
                    "total_payout": s_row.get("total_payout", 0)
                }
        except Exception as se:
            print(f"レース一覧確定情報付与エラー: {se}")

        natural_real_count = sum(1 for v in cached_preds_map.values() if v.get("is_real_bet_target"))
        has_downgraded_ref = any("参考推奨" in str(v.get("real_bet_badge") or "") for v in cached_preds_map.values())
        if (natural_real_count == 0 or has_downgraded_ref) and formatted_date >= today_jst_str() and cached_preds_map:
            if _ensure_daily_single_real_bet_fn:
                _ensure_daily_single_real_bet_fn(formatted_date)
            try:
                p_res2 = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
                for row in (p_res2.data or []):
                    d_json = row.get("data_json") or {}
                    h_rec = d_json.get("hybrid_recommendation")
                    if h_rec and row["race_id"] in cached_preds_map:
                        cached_preds_map[row["race_id"]]["is_real_bet_target"] = h_rec.get("is_real_bet_target", False)
                        cached_preds_map[row["race_id"]]["real_bet_badge"] = h_rec.get("real_bet_badge")
                        cached_preds_map[row["race_id"]]["real_bet_level"] = h_rec.get("real_bet_level", "none")
            except Exception:
                pass

    for venue in venues:
        races_in_venue = venue.get("races", [])
        for race in races_in_venue:
            info = cached_preds_map.get(race["race_id"])
            if info:
                race["is_real_bet_target"] = info["is_real_bet_target"]
                badge_str = info["real_bet_badge"]
                if badge_str and isinstance(badge_str, str):
                    badge_str = badge_str.replace("🎯", "⭐")
                race["real_bet_badge"] = badge_str
                race["real_bet_level"] = info.get("real_bet_level", "none")
                race["edge_ratio"] = info["edge_ratio"]
                race["is_final_odds"] = info["is_final_odds"]
                race["confidence_score"] = info.get("confidence_score", 0.0)
                race["is_virtual_bet_target"] = info.get("is_virtual_bet_target", False)
                race["virtual_bet_slot"] = info.get("virtual_bet_slot")
            else:
                race["is_real_bet_target"] = False
                race["real_bet_badge"] = None
                race["real_bet_level"] = "none"
                race["edge_ratio"] = None
                race["is_final_odds"] = False
                race["confidence_score"] = 0.0
                race["is_virtual_bet_target"] = False
                race["virtual_bet_slot"] = None

            s_info = settled_map.get(race["race_id"])
            if s_info:
                race["is_settled"] = s_info.get("is_settled", 0)
                race["is_hit"] = s_info.get("is_hit", 0)
                race["is_pass"] = s_info.get("is_pass", 0)
            else:
                race["is_settled"] = 0
                race["is_hit"] = 0
                race["is_pass"] = 0

        # 各会場ごとに確信度上位3レースを「仮想馬券枠 (各場3R)」として保証
        eligible_for_virtual = []
        for r in races_in_venue:
            if float(r.get("confidence_score") or 0.0) <= 0.0:
                continue
            r_nm = r.get("race_name", "")
            r_info = r.get("race_info", "")
            is_shinba = ("新馬" in r_nm) or ("メイクデビュー" in r_nm)
            is_obstacle = ("障害" in r_nm) or ("障" in r_info)
            if not is_shinba and not is_obstacle:
                eligible_for_virtual.append(r)

        eligible_for_virtual.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        top3_virtual = eligible_for_virtual[:3]
        top3_virtual_ids = set()
        for slot_idx, v_race in enumerate(top3_virtual, start=1):
            v_race["is_virtual_bet_target"] = True
            v_race["virtual_bet_slot"] = slot_idx
            vr_id = v_race.get("race_id")
            if vr_id:
                top3_virtual_ids.add(vr_id)
            if not v_race.get("is_real_bet_target"):
                v_race["real_bet_level"] = "virtual"
                v_race["real_bet_badge"] = f"🎮【仮想観戦】第{slot_idx}推奨"
                if vr_id and vr_id in raw_preds_cache:
                    c_d = raw_preds_cache[vr_id]
                    c_hrec = c_d.get("hybrid_recommendation")
                    if c_hrec and (
                        c_hrec.get("real_bet_level") != "virtual"
                        or c_hrec.get("real_bet_badge") != v_race["real_bet_badge"]
                        or c_hrec.get("virtual_badge") != v_race["real_bet_badge"]
                    ):
                        c_hrec["is_virtual_bet_target"] = True
                        c_hrec["virtual_bet_slot"] = slot_idx
                        c_hrec["real_bet_level"] = "virtual"
                        c_hrec["real_bet_badge"] = v_race["real_bet_badge"]
                        c_hrec["virtual_badge"] = v_race["real_bet_badge"]
                        if supabase:
                            try:
                                supabase.table("keiba_predictions_cache").update({"data_json": c_d}).eq("race_id", vr_id).execute()
                            except Exception as ue:
                                print(f"仮想枠キャッシュ同期エラー ({vr_id}): {ue}")

        # 会場内でトップ3から漏れた全レース（新馬・障害含む）の仮想枠ステータスを「見送り(none)」に完全整合
        for other_r in races_in_venue:
            or_id = other_r.get("race_id")
            if or_id and or_id not in top3_virtual_ids and not other_r.get("is_real_bet_target"):
                other_r["is_virtual_bet_target"] = False
                other_r["virtual_bet_slot"] = None
                other_r["real_bet_level"] = "none"
                other_r["real_bet_badge"] = None
                if or_id in raw_preds_cache:
                    c_d = raw_preds_cache[or_id]
                    c_hrec = c_d.get("hybrid_recommendation")
                    if c_hrec and (
                        c_hrec.get("real_bet_level") == "virtual"
                        or c_hrec.get("is_virtual_bet_target")
                        or c_hrec.get("virtual_badge") is not None
                        or c_hrec.get("real_bet_badge") is not None
                    ):
                        c_hrec["is_virtual_bet_target"] = False
                        c_hrec["virtual_bet_slot"] = None
                        c_hrec["real_bet_level"] = "none"
                        c_hrec["real_bet_badge"] = None
                        c_hrec["virtual_badge"] = None
                        if supabase:
                            try:
                                supabase.table("keiba_predictions_cache").update({"data_json": c_d}).eq("race_id", or_id).execute()
                            except Exception:
                                pass

    return venues


def get_today_races(
    date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)"),
    force_refresh: bool = Query(False, description="手動でレース一覧・発走時刻を再取得する場合True")
):
    """指定日または最新開催日のレース一覧・会場一覧を返却"""
    now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)

    if date:
        clean_d = date.replace("-", "")
        target_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else date
    else:
        target_date = p_info["today_str"]

    now_ts = time.time()
    cache_ttl = 30.0 if target_date == p_info["today_str"] else 3600.0
    if not force_refresh:
        c_val = _today_races_cache.get(target_date)
        if c_val and (now_ts - c_val[0] < cache_ttl):
            return c_val[1]

    available_dates = discover_period_kaisai_dates(now_dt)

    if not date:
        if not available_dates:
            return {
                "date": "",
                "available_dates": [],
                "is_waiting_period": False,
                "message": "現在表示可能な開催データがありません。",
                "venues": []
            }
        if p_info["today_str"] in available_dates:
            target_date = p_info["today_str"]
        else:
            target_date = available_dates[-1]

    venues = fetch_venues_for_date_raw(target_date, force_refresh=force_refresh)
    if venues:
        if target_date not in available_dates:
            available_dates = sorted(list(set(available_dates + [target_date])))
        venues = attach_prediction_badges_to_venues(target_date, venues)
        resp = {
            "date": target_date,
            "available_dates": available_dates[-6:],
            "is_waiting_period": False,
            "venues": venues
        }
        _today_races_cache[target_date] = (now_ts, resp)
        return resp

    resp_empty = {
        "date": target_date,
        "available_dates": available_dates[-6:],
        "is_waiting_period": False,
        "message": "指定された開催日のレース情報が見つかりませんでした。",
        "venues": []
    }
    _today_races_cache[target_date] = (now_ts, resp_empty)
    return resp_empty

