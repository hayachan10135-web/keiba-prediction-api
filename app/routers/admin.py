# -*- coding: utf-8 -*-
"""
app/routers/admin.py

管理者向けメンテナンス・診断・データ修復・ヘルスチェック・運用一括処理 API ルーター
"""

import os
import re
import time
import requests
import datetime
import traceback
from typing import Optional, Dict, Any, List, Tuple, Callable
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Query

from app.core.database import get_supabase_client
from app.core.timezone import now_jst, now_jst_naive, today_jst_str, JST
from app.settlement import invalidate_summary_caches
from app.schedule import discover_period_kaisai_dates, _latest_post_times
from app.scrapers import fetch_venues_for_date_raw
from app.maintenance import (
    CURRENT_DB_VERSION,
    clear_all_database,
    fix_mislabeled_20261011_races,
    normalize_mislabeled_dates,
    sync_and_process_target_dates,
    heal_settlement_marks,
    heal_today_weights_and_results,
)
from app.strategies.daily_promotion import ensure_daily_single_real_bet, _daily_top_race_id_cache
from app.ml.track_bias import compute_track_bias_and_matches
from app.ml.predictor import apply_odds_anomaly_indicators
from app.ml.model_loader import (
    load_stats_overlay_from_supabase,
    stats_dict,
    _stats_overlay,
    update_stats_from_race_settlement,
)

router = APIRouter(tags=["Admin"])

DICT_PATH = "stats_dict.json"

# 外部DI
_get_cached_prediction_fn: Optional[Callable] = None
_all_predictions_cache_ref: Optional[Dict[str, Any]] = None


def set_admin_dependencies(
    get_cached_prediction: Callable,
    all_predictions_cache: Dict[str, Any],
):
    """管理者ルーターが必要とする外部依存を登録"""
    global _get_cached_prediction_fn, _all_predictions_cache_ref
    _get_cached_prediction_fn = get_cached_prediction
    _all_predictions_cache_ref = all_predictions_cache


@router.get("/admin/test-umaca-connection")
def endpoint_test_umaca_connection():
    """Renderコンテナ上からJRA（UMACAスマートおよび照会サービス）へのネットワーク疎通テスト"""
    urls = {
        "voting_site": "https://www.ipat.jra.go.jp/sp/umaca/",
        "inquiry_site": "https://www.nvinq.jra.go.jp/jrasp/SMP000400",
        "jra_top": "https://www.jra.go.jp/"
    }
    results = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    }

    for name, url in urls.items():
        try:
            start_t = time.time()
            resp = requests.get(url, headers=headers, timeout=10)
            elapsed = round(time.time() - start_t, 2)
            html_snippet = resp.text[:300].replace("\n", " ").strip()
            is_blocked = resp.status_code in (403, 401, 503) or "アクセスが制限されています" in resp.text
            results[name] = {
                "status_code": resp.status_code,
                "elapsed_sec": elapsed,
                "is_blocked": is_blocked,
                "content_bytes": len(resp.content),
                "snippet": html_snippet[:120]
            }
        except Exception as e:
            results[name] = {
                "status_code": None,
                "error": str(e)
            }

    can_reach_all = all(r.get("status_code") == 200 for r in results.values())
    return {
        "status": "ok",
        "can_reach_all": can_reach_all,
        "render_ip_check": results,
        "summary": "【接続成功】RenderからJRAへの通信はブロックされていません。クラウド化が可能です。" if can_reach_all else "【接続制限あり】一部の通信がブロックされています。"
    }


def fix_cancelled_races_2026() -> dict:
    """2026年の中止レース（2/7 東京8-12R、2/8 小倉4R等）を正規の開催中止（返還・投資0/払戻0/不参戦）に是正"""
    supabase = get_supabase_client()
    if not supabase:
        print("[FixCancelled] Supabase未接続")
        return {"status": "skipped", "message": "Supabase not connected"}

    target_race_ids = [
        "202605010308",  # 2026-02-07 東京8R (積雪中止)
        "202605010309",  # 2026-02-07 東京9R (積雪中止)
        "202605010310",  # 2026-02-07 東京10R (積雪中止)
        "202605010311",  # 2026-02-07 東京11R (積雪中止)
        "202605010312",  # 2026-02-07 東京12R (積雪中止)
        "202610010604",  # 2026-02-08 小倉4R (中止)
    ]

    updated_count = 0
    try:
        for r_id in target_race_ids:
            row_data = {
                "strategy_type": "cancelled",
                "target_umaban": None,
                "horse_name": None,
                "grade": "C",
                "is_pass": 1,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": 0,
                "actual_order": None,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": 0,
                "is_hit": 0,
                "hit_type": "開催中止",
                "hypothetical_bet": 0,
                "hypothetical_payout": 0,
                "is_real_bet_target": False,
                "real_bet_tier": "none",
                "real_bet_badge": None,
                "confidence_score": 0.0,
                "edge_ratio": 0.0,
                "bet_details": {
                    "reason": "悪天候によるレース開催中止・全額返還",
                    "real_bet_meta": {
                        "is_real_bet_target": False,
                        "real_bet_badge": None,
                        "real_bet_level": "none",
                        "mark_orders": [],
                        "top3_orders": []
                    }
                },
                "payout_details": None,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "tansho_payout": 0,
                "fukusho_payout": 0
            }
            supabase.table("keiba_race_settlements").update(row_data).eq("race_id", r_id).execute()
            updated_count += 1
            print(f"[FixCancelled] {r_id} を開催中止（全額返還）に更新しました。")

        msg = f"中止レース是正完了: {updated_count}件を全額返還（投資0/払戻0）に修正しました。"
        print(f"[FixCancelled] {msg}")
        return {"status": "ok", "message": msg, "updated_count": updated_count}
    except Exception as e:
        err_msg = f"[FixCancelled Error] {e}"
        print(err_msg)
        return {"status": "error", "message": err_msg}


@router.get("/admin/fix-cancelled-races")
@router.post("/admin/fix-cancelled-races")
def endpoint_fix_cancelled_races():
    return fix_cancelled_races_2026()


@router.get("/admin/stats-dict-info")
def endpoint_stats_dict_info():
    """統計辞書（stats_dict.json）のファイル存在状態・容量およびSupabase接続ステータス情報"""
    supabase = get_supabase_client()
    file_exists = os.path.exists(DICT_PATH)
    file_size_mb = round(os.path.getsize(DICT_PATH) / (1024 * 1024), 2) if file_exists else 0.0

    return {
        "status": "ok",
        "file_exists": file_exists,
        "file_size_mb": file_size_mb,
        "supabase_connected": supabase is not None,
        "server_time_jst": now_jst().strftime("%Y-%m-%d %H:%M:%S")
    }


@router.post("/admin/clear-db")
def endpoint_clear_db(sync_today: bool = False):
    supabase = get_supabase_client()
    res = clear_all_database(sync_today=sync_today)
    if supabase and res.get("status") == "ok":
        try:
            supabase.table("keiba_app_meta").upsert({
                "key": "schema_version",
                "value": {"version": CURRENT_DB_VERSION, "updated_at": today_jst_str()}
            }).execute()
        except Exception:
            pass
        try:
            supabase.table("keiba_predictions_cache").upsert({
                "race_id": "SCHEMA_VERSION",
                "race_date": "9999-12-31",
                "data_json": {"version": CURRENT_DB_VERSION, "updated_at": today_jst_str()}
            }).execute()
        except Exception:
            pass
    return res


@router.post("/admin/fix-20261011")
def endpoint_fix_20261011():
    return fix_mislabeled_20261011_races()


def sync_and_process_1003_1004():
    return sync_and_process_target_dates(["2026-10-03", "2026-10-04"])


def sync_and_process_0926_0927():
    return sync_and_process_target_dates(["2026-09-26", "2026-09-27"])


@router.post("/admin/normalize-dates")
def endpoint_normalize_dates():
    return normalize_mislabeled_dates()


@router.post("/admin/process-1003-1004")
def endpoint_process_1003_1004():
    return sync_and_process_1003_1004()


@router.post("/admin/process-0926-0927")
def endpoint_process_0926_0927():
    return sync_and_process_0926_0927()


@router.post("/admin/process-date")
def endpoint_process_date(date: str = Query(..., description="対象日付 (YYYY-MM-DD)")):
    return sync_and_process_target_dates([date])


class BatchUpsertPayload(BaseModel):
    predictions: Optional[List[dict]] = None
    settlements: Optional[List[dict]] = None
    scheduled: Optional[List[dict]] = None


@router.post("/admin/upsert-verified-data")
def endpoint_upsert_verified_data(payload: BatchUpsertPayload):
    """検証済みの予想・収支結果・スケジュールデータをSupabaseに一括アップサート"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase未接続")

    # バリデーション: 不完全データ、ダミー馬名、出走馬0頭のアップサートを厳格に遮断
    if payload.predictions:
        for p in payload.predictions:
            r_id = p.get("race_id", "")
            d_json = p.get("data_json", {})
            horses = d_json.get("horses", [])
            if not horses or len(horses) == 0:
                raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} の出走馬データが0頭です。")
            for h in horses:
                h_name = h.get("horse_name", "")
                if not h_name or re.match(r"^馬\d+$", h_name):
                    raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} にダミー馬名 '{h_name}' (馬番 {h.get('umaban')}) が含まれています。")

    if payload.settlements:
        for s in payload.settlements:
            r_id = s.get("race_id", "")
            h_name = s.get("horse_name", "")
            if not s.get("is_cancelled") and s.get("strategy_type") != "cancelled":
                if not h_name or re.match(r"^馬\d+$", h_name):
                    raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} の本命馬名がダミー '{h_name}' です。")

    pred_count = 0
    settle_count = 0
    sched_count = 0
    try:
        if payload.predictions:
            supabase.table("keiba_predictions_cache").upsert(payload.predictions, on_conflict="race_id").execute()
            pred_count = len(payload.predictions)
        if payload.settlements:
            supabase.table("keiba_race_settlements").upsert(payload.settlements, on_conflict="race_id").execute()
            settle_count = len(payload.settlements)
        if payload.scheduled:
            supabase.table("keiba_scheduled_races").upsert(payload.scheduled, on_conflict="race_id").execute()
            sched_count = len(payload.scheduled)
        invalidate_summary_caches()
        return {
            "status": "ok",
            "predictions_count": pred_count,
            "settlements_count": settle_count,
            "scheduled_count": sched_count
        }
    except Exception as e:
        print(f"[UpsertVerifiedData Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/sync-active-post-times")
def endpoint_sync_active_post_times(dates: Optional[str] = Query(None, description="カンマ区切りの対象日 (YYYY-MM-DD)")):
    """現在メイン画面に表示される開催期間の全レースについて、実際の出走時刻をスクレイピングしてDBに反映（変更対応）"""
    supabase = get_supabase_client()
    if dates:
        target_dates = [d.strip() for d in dates.split(",") if d.strip()]
    else:
        now_dt = now_jst_naive()
        target_dates = discover_period_kaisai_dates(now_dt)
        if not target_dates:
            target_dates = ["2026-10-12", "2026-10-11", "2026-10-10"]

    results = {}
    for d_str in target_dates:
        venues = fetch_venues_for_date_raw(d_str, force_refresh=True)
        count = 0
        for v in venues:
            for r in v.get("races", []):
                r_id = r.get("race_id")
                r_info = r.get("race_info", "")
                m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                if r_id and m:
                    hour_int, min_int = int(m.group(1)), int(m.group(2))
                    pt_str = f"{hour_int:02d}:{min_int:02d}発走"
                    _latest_post_times[r_id] = pt_str
                    # DB更新
                    if supabase:
                        try:
                            # 1. scheduled_races
                            d_parts = [int(p) for p in d_str.split("-")]
                            post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour_int, min_int, 0)
                            supabase.table("keiba_scheduled_races").update({
                                "post_datetime": post_dt.strftime("%Y-%m-%d %H:%M:%S")
                            }).eq("race_id", r_id).execute()
                            # 2. predictions_cache
                            p_res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", r_id).execute()
                            if p_res.data and p_res.data[0].get("data_json"):
                                dj = p_res.data[0]["data_json"]
                                dj_details = dj.get("race_details") or {}
                                if dj_details.get("post_time") != pt_str:
                                    dj_details["post_time"] = pt_str
                                    dj["race_details"] = dj_details
                                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", r_id).execute()
                        except Exception:
                            pass
                    count += 1
        results[d_str] = count

    return {"status": "ok", "updated_races_by_date": results}


@router.post("/admin/refresh-stats-dict")
def endpoint_refresh_stats_dict(rebuild_from_cached_results: bool = False):
    """
    統計辞書オーバーレイをSupabaseから再読み込み・マージ、
    または過去の確定キャッシュ（keiba_race_results_cache）から動的集計して最新化する。
    """
    supabase = get_supabase_client()
    global stats_dict, _stats_overlay
    # 1. Supabaseから永続化オーバーレイをリロード
    loaded = load_stats_overlay_from_supabase()
    rebuilt_races = 0
    updated_horses = 0

    if rebuild_from_cached_results and supabase:
        try:
            # 確定レース結果キャッシュを取得して順次適用
            res = supabase.table("keiba_race_results_cache").select("race_id, race_date, result_json").order("race_date", desc=False).execute()
            if res.data:
                for row in res.data:
                    rid = row.get("race_id")
                    rdate = row.get("race_date")
                    r_res = row.get("result_json")
                    if rid and rdate and r_res:
                        pred_row = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", rid).execute()
                        if pred_row.data and pred_row.data[0].get("data_json"):
                            p_data = pred_row.data[0]["data_json"]
                            cnt = update_stats_from_race_settlement(rid, rdate, p_data, r_res)
                            if cnt > 0:
                                rebuilt_races += 1
                                updated_horses += cnt
        except Exception as e:
            print(f"[RefreshStatsDict] キャッシュからの再構築エラー: {e}")

    return {
        "status": "ok",
        "loaded_from_supabase": loaded,
        "rebuilt_races": rebuilt_races,
        "updated_horses": updated_horses,
        "total_horses": len(stats_dict.get("horse_stats", {})),
        "overlay_horses_count": len(_stats_overlay.get("horse_stats", {})),
        "overlay_last_updated": _stats_overlay.get("updated_at")
    }


@router.post("/admin/rebuild-predictions")
def endpoint_rebuild_predictions(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)。省略時は直近10開催期間内のキャッシュ全レース"),
    sync_settlements: bool = Query(True, description="成績台帳(keiba_race_settlements)にもトラックバイアスとオッズ情報を反映する場合True")
):
    """
    最新のオッズ異常検知ロジックおよびトラックバイアス分析エンジンを
    既存のキャッシュおよび成績台帳へ再適用 (delete/upsert) して完全同期
    """
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not connected")

    try:
        query = supabase.table("keiba_predictions_cache").select("race_id, race_date").neq("race_id", "SCHEMA_VERSION")
        if date:
            query = query.eq("race_date", date)
        else:
            cutoff = (datetime.datetime.now(JST).date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
            query = query.gte("race_date", cutoff)
        
        res = query.execute()
        rows = res.data or []
        updated_predictions = 0
        updated_settlements = 0

        for row in rows:
            r_id = row.get("race_id")
            r_date = row.get("race_date") or today_jst_str()
            if not r_id:
                continue

            try:
                p_item = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", r_id).execute()
                if not p_item.data:
                    continue
                d_json = p_item.data[0].get("data_json") or {}
                horses = d_json.get("horses", [])
                if not horses:
                    continue

                # 1. オッズ変動・異常検知を再適用
                apply_odds_anomaly_indicators(horses)

                # 2. トラックバイアスを再計算
                c_details = (d_json.get("race_details") or {}).get("course_details", "")
                surf_str = "芝" if "芝" in c_details else ("ダート" if "ダ" in c_details else "障害")
                tb_info = compute_track_bias_and_matches(r_id, r_date, surf_str, horses)
                if tb_info:
                    d_json["track_bias"] = tb_info

                # 3. keiba_predictions_cache を最新JSONで更新 (delete/upsert)
                supabase.table("keiba_predictions_cache").upsert({
                    "race_id": r_id,
                    "race_date": r_date,
                    "data_json": d_json
                }).execute()
                updated_predictions += 1

                # 4. 成績台帳 (keiba_race_settlements) にも反映
                if sync_settlements:
                    try:
                        s_res = supabase.table("keiba_race_settlements").select("race_id, bet_details").eq("race_id", r_id).execute()
                        if s_res.data and len(s_res.data) > 0:
                            o_alerts = [
                                {
                                    "umaban": h.get("umaban"),
                                    "horse_name": h.get("horse_name"),
                                    "odds": h.get("odds"),
                                    "odds_trend": h.get("odds_trend"),
                                    "odds_alert": h.get("odds_alert"),
                                    "bias_match": h.get("bias_match"),
                                    "odds_history": h.get("odds_history")
                                }
                                for h in horses
                                if h.get("odds_alert") or h.get("odds_trend") or h.get("bias_match") or h.get("odds_history")
                            ]
                            patch_dict = {}
                            if tb_info:
                                patch_dict["track_bias"] = tb_info
                            if o_alerts:
                                patch_dict["odds_alerts"] = o_alerts
                            
                            b_det = s_res.data[0].get("bet_details") or {}
                            t3 = b_det.get("real_bet_meta", {}).get("top3_orders")
                            if t3:
                                patch_dict["top3_orders"] = t3
                            
                            try:
                                supabase.table("keiba_race_settlements").update(patch_dict).eq("race_id", r_id).execute()
                                updated_settlements += 1
                            except Exception:
                                if tb_info:
                                    b_det["track_bias"] = tb_info
                                if o_alerts:
                                    b_det["odds_alerts"] = o_alerts
                                supabase.table("keiba_race_settlements").update({"bet_details": b_det}).eq("race_id", r_id).execute()
                                updated_settlements += 1
                    except Exception as se:
                        pass

                time.sleep(0.08)
            except Exception as row_err:
                print(f"[Rebuild race warning] {r_id}: {row_err}")
                continue

        invalidate_summary_caches()
        return {
            "status": "ok",
            "updated_predictions": updated_predictions,
            "updated_settlements": updated_settlements,
            "message": f"{updated_predictions}件の予想キャッシュおよび{updated_settlements}件の成績台帳を最新ロジックで完全再同期しました。"
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Rebuild Fatal Error] {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Rebuild error: {e}")


@router.get("/debug/track-bias/{race_id}")
def endpoint_debug_track_bias(race_id: str):
    """トラックバイアス計算の内部動作を詳細に診断して返す"""
    supabase = get_supabase_client()
    if not supabase:
        return {"error": "Supabase not connected"}
    
    day_prefix = race_id[:10] if len(race_id) >= 10 else race_id[:6]
    start_rid = f"{day_prefix}01"
    
    res_cache = supabase.table("keiba_race_results_cache")\
        .select("race_id, result_json")\
        .gte("race_id", start_rid)\
        .lt("race_id", race_id)\
        .execute()
    results_rows = res_cache.data or []

    past_rids = [r["race_id"] for r in results_rows if r.get("race_id")]
    pred_map = {}
    if past_rids:
        try:
            p_res = supabase.table("keiba_predictions_cache")\
                .select("race_id, data_json")\
                .in_("race_id", past_rids)\
                .execute()
            pred_map = {row["race_id"]: (row.get("data_json") or {}) for row in (p_res.data or [])}
        except Exception as pe:
            print(f"[Debug] pred_map fetch err: {pe}")

    this_p = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", race_id).execute()
    this_dj = (this_p.data[0].get("data_json") or {}) if this_p.data else {}
    this_course = (this_dj.get("race_details") or {}).get("course_details", "")
    
    eval_list = []
    for r in results_rows:
        rid = str(r.get("race_id") or "")
        rj = r.get("result_json") or {}
        orders = rj.get("orders") or {}
        dj = pred_map.get(rid, {})
        c_details = (dj.get("race_details") or {}).get("course_details", "")
        
        eval_list.append({
            "race_id": rid,
            "has_orders": bool(orders),
            "orders_count": len(orders),
            "course_details": c_details,
            "dj_found": bool(dj)
        })

    return {
        "target_race_id": race_id,
        "day_prefix": day_prefix,
        "start_rid": start_rid,
        "this_course_details": this_course,
        "results_count": len(results_rows),
        "eval_list": eval_list
    }


@router.post("/admin/heal-settlement-marks")
@router.get("/admin/heal-settlement-marks")
def endpoint_heal_settlement_marks(date: Optional[str] = Query(None, description="修復対象日付 (YYYY-MM-DD, 省略時は全期間)")):
    return heal_settlement_marks(date=date)


@router.post("/admin/heal-today")
@router.get("/admin/heal-today")
def endpoint_heal_today(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)"),
    force_resettle: bool = Query(False, description="全レースの結果照合・確定を強制再計算するか")
):
    return heal_today_weights_and_results(date, force_resettle=force_resettle)


@router.post("/admin/reselect-daily-real-bet")
def endpoint_reselect_daily_real_bet(target_date: Optional[str] = Query(None)):
    """当日の実戦厳選レースを再選定し、未発走レースへ移行する"""
    t_date = target_date or today_jst_str()
    _daily_top_race_id_cache.pop(t_date, None)
    if _all_predictions_cache_ref is not None:
        _all_predictions_cache_ref.pop(t_date, None)
    top_rid = ensure_daily_single_real_bet(t_date)
    pred = _get_cached_prediction_fn(top_rid) if (_get_cached_prediction_fn and top_rid) else None
    return {
        "status": "ok",
        "date": t_date,
        "selected_race_id": top_rid,
        "prediction": pred
    }
