# -*- coding: utf-8 -*-
"""
app/ml/track_bias.py

リアルタイムトラックバイアス分析エンジン
当日同競馬場・同トラック種別の前半レース結果から、内枠/外枠・先行/差し有利度を動的算出し、各馬のマッチ度を判定する
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple, Callable

from app.core.database import get_supabase_client

GLOBAL_VENUE_MAP = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"
}

_track_bias_cache: Dict[str, Tuple[float, dict]] = {}
_day_results_cache: Dict[str, Tuple[float, List[dict]]] = {}
_past_pred_cache: Dict[str, dict] = {}
_get_stats_dict_handler: Optional[Callable[[], dict]] = None


def set_track_bias_dependencies(get_stats_dict_fn: Optional[Callable[[], dict]] = None):
    """統計辞書取得ハンドラーの登録 (DI)"""
    global _get_stats_dict_handler
    if get_stats_dict_fn is not None:
        _get_stats_dict_handler = get_stats_dict_fn


def evaluate_horse_bias_match(
    horse: Dict[str, Any],
    bias_info: Dict[str, Any]
) -> str:
    """
    出走馬の枠番・脚質と馬場バイアス指数とのマッチ度 (bias_match) を判定
    戻り値: "絶好バイアス", "有利", "合致", "逆風", ""
    """
    if not bias_info:
        return ""

    wakuban = int(horse.get("wakuban", 0))
    running_style = str(horse.get("running_style", ""))
    inner_fav = bool(bias_info.get("is_inner_favorable", False))
    outer_fav = bool(bias_info.get("is_outer_favorable", False))
    front_fav = bool(bias_info.get("is_front_favorable", False))

    is_inner = (1 <= wakuban <= 3)
    is_outer = (6 <= wakuban <= 8)
    is_front = any(s in running_style for s in ("逃げ", "先行"))

    if inner_fav and is_inner and front_fav and is_front:
        return "絶好バイアス (内前有利)"
    elif outer_fav and is_outer:
        return "絶好バイアス (外枠有利)"
    elif (inner_fav and is_inner) or (outer_fav and is_outer):
        return "有利 (枠順合致)"
    elif inner_fav and is_outer:
        return "逆風 (外枠逆風)"
    elif outer_fav and is_inner:
        return "逆風 (内枠逆風)"

    return ""


def compute_track_bias_and_matches(
    race_id: str,
    race_date: str,
    surface_name: str,
    horses: List[dict],
    prefetched_settled_races: Optional[List[dict]] = None,
    prefetched_pred_map: Optional[Dict[str, dict]] = None,
    stats_dict: Optional[dict] = None
) -> Optional[dict]:
    """
    【Step 5】トラックバイアス（当日の馬場傾向：内枠/外枠・先行/差し有利）のリアルタイム分析
    1. 当日(race_date)・同競馬場(race_id[4:6])・同トラック種別(芝/ダート)の確定済みレース(1〜3着)を集計。
    2. 内枠(1〜3枠)・外枠(6〜8枠)の複勝シェア率、および先行逃げ(4角4番手以内)の勝率から馬場バイアス指数を算出。
    3. 出走各馬の枠番・脚質とのマッチ度(bias_match)を判定。
    ※ モデル推論確率(prob_win/prob_top3)自体は変更しない（ゼロリーク・ベースライン厳守）。
    """
    if not surface_name or surface_name in ("障害", ""):
        return None

    venue_code_str = race_id[4:6] if len(race_id) >= 6 else "00"
    try:
        venue_code_int = int(venue_code_str)
    except Exception:
        venue_code_int = 0
    venue_name = GLOBAL_VENUE_MAP.get(venue_code_int, "競馬場")

    cache_key = f"{race_id}_{surface_name}"
    now_ts = time.time()
    cached_entry = _track_bias_cache.get(cache_key)

    bias_info = None
    if cached_entry and (now_ts - cached_entry[0] < 120.0):
        bias_info = cached_entry[1]
    else:
        supabase = get_supabase_client()
        if supabase:
            try:
                day_prefix = race_id[:10] if len(race_id) >= 10 else race_id[:6]
                settled_races = []

                if prefetched_settled_races is not None:
                    for r in prefetched_settled_races:
                        rid = str(r.get("race_id") or "")
                        if rid[:10] == day_prefix and rid < race_id:
                            settled_races.append(r)
                else:
                    c_day_entry = _day_results_cache.get(day_prefix)
                    if c_day_entry and (now_ts - c_day_entry[0] < 60.0):
                        for r in c_day_entry[1]:
                            if str(r.get("race_id") or "") < race_id:
                                settled_races.append(r)
                    else:
                        start_rid = f"{day_prefix}01"
                        day_settled = []
                        try:
                            res_cache = supabase.table("keiba_race_results_cache")\
                                .select("race_id, result_json")\
                                .gte("race_id", start_rid)\
                                .lte("race_id", f"{day_prefix}12")\
                                .execute()
                            for r in (res_cache.data or []):
                                rid = r.get("race_id")
                                rj = r.get("result_json") or {}
                                orders = rj.get("orders") or {}
                                if orders:
                                    day_settled.append({"race_id": rid, "orders": orders})
                        except Exception as e:
                            print(f"[TrackBias] results_cache query warning: {e}")

                        if not day_settled:
                            try:
                                s_res = supabase.table("keiba_race_settlements")\
                                    .select("race_id, bet_details")\
                                    .gte("race_id", start_rid)\
                                    .lte("race_id", f"{day_prefix}12")\
                                    .execute()
                                for r in (s_res.data or []):
                                    rid = r.get("race_id")
                                    b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
                                    top3 = b_det.get("real_bet_meta", {}).get("top3_orders") or []
                                    if top3:
                                        day_settled.append({"race_id": rid, "top3_orders": top3})
                            except Exception as e:
                                print(f"[TrackBias] settlements query warning: {e}")

                        _day_results_cache[day_prefix] = (now_ts, day_settled)
                        for r in day_settled:
                            if str(r.get("race_id") or "") < race_id:
                                settled_races.append(r)

                sample_races = 0
                inside_count = 0
                middle_count = 0
                outside_count = 0
                front_wins = 0

                active_stats = stats_dict or (_get_stats_dict_handler() if _get_stats_dict_handler else {})

                if settled_races:
                    past_rids = [str(r["race_id"]) for r in settled_races if r.get("race_id")]
                    pred_map = {}
                    if prefetched_pred_map is not None:
                        pred_map = prefetched_pred_map
                    elif past_rids:
                        missing_rids = [r for r in past_rids if r not in _past_pred_cache]
                        if missing_rids:
                            try:
                                p_res = supabase.table("keiba_predictions_cache")\
                                    .select("race_id, data_json")\
                                    .in_("race_id", missing_rids)\
                                    .execute()
                                for row in (p_res.data or []):
                                    if row.get("race_id"):
                                        _past_pred_cache[row["race_id"]] = (row.get("data_json") or {})
                            except Exception as pe:
                                print(f"[TrackBias] predictions_cache fetch warning: {pe}")
                        pred_map = {r: _past_pred_cache.get(r, {}) for r in past_rids}

                    is_target_turf = "芝" in surface_name
                    is_target_dirt = ("ダ" in surface_name) or ("ダート" in surface_name)

                    for s_item in settled_races:
                        r_id = str(s_item.get("race_id") or "")
                        dj = pred_map.get(r_id, {})
                        c_details = (dj.get("race_details") or {}).get("course_details", "")
                        past_is_turf = "芝" in c_details
                        past_is_dirt = ("ダ" in c_details) or ("ダート" in c_details)
                        if is_target_turf and not past_is_turf:
                            continue
                        if is_target_dirt and not past_is_dirt:
                            continue
                        if not is_target_turf and not is_target_dirt:
                            continue

                        top3_list = []
                        if "orders" in s_item:
                            for u_k, o_v in (s_item.get("orders") or {}).items():
                                try:
                                    u_int = int(u_k)
                                    o_int = int(o_v)
                                    if o_int in (1, 2, 3):
                                        top3_list.append({"umaban": u_int, "order": o_int})
                                except Exception:
                                    pass
                            top3_list.sort(key=lambda x: x["order"])
                        elif "top3_orders" in s_item:
                            for it in (s_item.get("top3_orders") or []):
                                if isinstance(it, dict):
                                    top3_list.append({
                                        "umaban": int(it.get("umaban") or 0),
                                        "order": int(it.get("order") or 1),
                                        "horse_name": it.get("horse_name", "")
                                    })
                                else:
                                    try:
                                        top3_list.append({"umaban": int(it), "order": 1, "horse_name": ""})
                                    except Exception:
                                        pass

                        if len(top3_list) < 3:
                            continue

                        sample_races += 1
                        r_horses = dj.get("horses") or []
                        w_map = {int(h["umaban"]): int(h.get("wakuban") or 0) for h in r_horses if "umaban" in h and str(h["umaban"]).isdigit()}
                        name_map = {int(h["umaban"]): h.get("horse_name", "") for h in r_horses if "umaban" in h and str(h["umaban"]).isdigit()}

                        for item in top3_list[:3]:
                            u_num = item["umaban"]
                            order_num = item["order"]
                            h_name = item.get("horse_name") or name_map.get(u_num, "")
                            h_clean = re.sub(r"[\s\u3000]", "", str(h_name))

                            waku_num = w_map.get(u_num, 0)
                            if waku_num == 0 and u_num > 0:
                                waku_num = min(8, max(1, (u_num - 1) // 2 + 1))

                            if waku_num in (1, 2, 3):
                                inside_count += 1
                            elif waku_num in (4, 5):
                                middle_count += 1
                            elif waku_num in (6, 7, 8):
                                outside_count += 1

                            if order_num == 1 and h_clean:
                                st = active_stats.get("horse_stats", {}).get(h_clean, {})
                                if float(st.get("last_corner_final", 8.0)) <= 4.0:
                                    front_wins += 1

                if sample_races < 2:
                    bias_info = {
                        "venue_name": venue_name,
                        "surface_name": surface_name,
                        "sample_races": sample_races,
                        "bias_summary": f"⚖️ 当日{surface_name}分析中 (確定{sample_races}R)",
                        "waku_bias": "flat",
                        "running_style_bias": "flat",
                        "inside_top3_ratio": 0.0,
                        "outside_top3_ratio": 0.0,
                        "detail_text": f"本日{surface_name}: 確定{sample_races}レースのためフラット判定"
                    }
                else:
                    total_top3 = sample_races * 3
                    in_pct = round((inside_count / total_top3) * 100.0, 1)
                    out_pct = round((outside_count / total_top3) * 100.0, 1)
                    front_pct = round((front_wins / sample_races) * 100.0, 1)

                    waku_bias = "inside" if in_pct >= 50.0 else ("outside" if out_pct >= 50.0 else "flat")
                    style_bias = "front" if front_pct >= 60.0 else ("closer" if front_pct <= 25.0 else "flat")

                    if waku_bias == "inside" and style_bias == "front":
                        sum_txt = "🔥内枠先行超有利 (前残りバイアス)"
                    elif waku_bias == "inside" and style_bias == "closer":
                        sum_txt = "🟢内枠有利 (ロスなし差し)"
                    elif waku_bias == "inside":
                        sum_txt = "🟢内枠有利 (経済コース優位)"
                    elif waku_bias == "outside" and style_bias == "closer":
                        sum_txt = "⚡外枠外差し有利 (上がり最速優位)"
                    elif waku_bias == "outside" and style_bias == "front":
                        sum_txt = "🔵外枠有利 (スムーズ先行)"
                    elif waku_bias == "outside":
                        sum_txt = "🔵外枠有利 (馬場外伸び)"
                    elif style_bias == "front":
                        sum_txt = "🏃先行逃げ有利 (前残り馬場)"
                    elif style_bias == "closer":
                        sum_txt = "🌪️差し追込有利 (直線外伸び)"
                    else:
                        sum_txt = "⚖️内外・脚質フラット"

                    detail = f"本日{venue_name}{surface_name}{sample_races}R確定: 内枠(1-3枠) 複勝率{in_pct:.0f}% / 外枠 {out_pct:.0f}% / 先行逃げ {front_wins}勝"
                    bias_info = {
                        "venue_name": venue_name,
                        "surface_name": surface_name,
                        "sample_races": sample_races,
                        "bias_summary": sum_txt,
                        "waku_bias": waku_bias,
                        "running_style_bias": style_bias,
                        "inside_top3_ratio": in_pct,
                        "outside_top3_ratio": out_pct,
                        "detail_text": detail
                    }

                _track_bias_cache[cache_key] = (now_ts, bias_info)
            except Exception as be:
                print(f"[TrackBias Warning] 集計エラー: {be}")

    # 各出走馬へのバイアスマッチング (bias_match)
    if bias_info and horses:
        waku_b = bias_info.get("waku_bias", "flat")
        style_b = bias_info.get("running_style_bias", "flat")
        active_stats = stats_dict or (_get_stats_dict_handler() if _get_stats_dict_handler else {})
        h_stats = active_stats.get("horse_stats", {})

        for h in horses:
            w = int(h.get("wakuban") or 0)
            h_clean = re.sub(r"[\s\u3000]", "", str(h.get("horse_name") or ""))
            st = h_stats.get(h_clean, {})
            is_front = float(st.get("last_corner_final", 8.0)) <= 4.0

            match_tag = None
            if waku_b == "inside" and w in (1, 2, 3):
                if is_front and style_b == "front":
                    match_tag = "🎯絶好バイアス (内先行)"
                else:
                    match_tag = "🟢内枠有利"
            elif waku_b == "inside" and w in (7, 8):
                match_tag = "⚠️外枠逆風"
            elif waku_b == "outside" and w in (6, 7, 8):
                if not is_front and style_b == "closer":
                    match_tag = "🎯絶好バイアス (外差し)"
                else:
                    match_tag = "🔵外枠有利"
            elif waku_b == "outside" and w in (1, 2):
                match_tag = "⚠️内枠逆風"
            elif style_b == "front" and is_front:
                match_tag = "🏃前残り合致"
            elif style_b == "closer" and not is_front:
                match_tag = "🌪️差し合致"

            h["bias_match"] = match_tag

    return bias_info
