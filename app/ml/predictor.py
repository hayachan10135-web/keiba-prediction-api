# -*- coding: utf-8 -*-
"""
app/ml/predictor.py

コア推論オーケストレーター
- 出走馬特徴量マトリクス生成 (48 / 50 / 54特徴量)
- LightGBM カスケード推論 (Win / Top3 / Maiden / Gachi / Himo)
- AI印決定 (◎, ◯, ▲, △, ☆)
- オッズ異常・乖離インジケーター (直前急落、妙味穴、過剰人気)
- 買い目戦略調停レイヤ連携
- レース推論ライフサイクル管理 (predict_race_core)
"""

import math
import re
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from fastapi import HTTPException

from app.core.database import get_supabase_client
from app.core.timezone import today_jst_str, now_jst, now_jst_naive, is_before_morning_prediction
from app.ml.model_loader import (
    model,
    model_win,
    model_top3,
    model_gachi,
    model_himo,
    model_cb_win,
    model_cb_top3,
    model_xgb_win,
    model_xgb_top3,
    model_class_a_win,
    model_class_a_top3,
    model_class_b_win,
    model_class_b_top3,
    model_class_c_win,
    model_class_c_top3,
    calibration_table,
    ensemble_config,
    stats_dict,
)
from app.ml.track_bias import compute_track_bias_and_matches
from app.scrapers.netkeiba_client import fetch_real_odds, fetch_shutuba_table
from app.strategies import (
    StrategyArbiter,
    get_or_select_daily_top_race_id,
    ensure_daily_single_real_bet,
)

_strategy_arbiter = StrategyArbiter()

DEFAULT_FEATURES_48 = [
    "wakuban", "umaban", "kinryo",
    "venue_code", "surface_type", "distance", "distance_diff", "condition_code",
    "career_races", "career_top3_rate",
    "prev1_order", "prev1_pop", "prev1_odds", "prev2_order", "days_since_prev",
    "jockey_top3_rate", "jockey_course_top3_rate",
    "trainer_top3_rate", "trainer_course_top3_rate",
    "head_count", "relative_umaban", "kinryo_diff_from_mean",
    "jockey_rank_in_race", "trainer_rank_in_race", "prev1_order_rank_in_race",
    "is_blinker", "is_first_blinker",
    "is_turf_to_dirt", "is_dirt_to_turf",
    "is_dist_shorten_200m", "is_dist_lengthen_200m",
    "is_tataki_2nd",
    "prev1_last_3f", "prev1_corner_final",
    "prev1_is_front", "prev1_is_closer",
    "horse_weight", "weight_diff", "abs_weight_diff", "weight_ratio",
    "turn_type", "is_turn_change",
    "race_num", "race_class_code", "is_female_only", "kinryo_std_in_race",
    "sex_code", "age"
]

ODDS_STAGE_ORDER = ["0830", "60m", "30m", "15m", "05m"]
ODDS_STAGE_RANK = {"0830": 1, "60m": 2, "30m": 3, "15m": 4, "05m": 5}


def infer_odds_stage_from_minutes(mins_to_post: Optional[float]) -> str:
    """発走までの残り分数から現在のオッズ取得ステージを判定"""
    if mins_to_post is None or mins_to_post > 65.0:
        return "0830"
    elif mins_to_post > 33.0:
        return "60m"
    elif mins_to_post > 17.0:
        return "30m"
    elif mins_to_post > 6.5:
        return "15m"
    else:
        return "05m"

# --- DI (依存性注入) ハンドラー群 ---
_get_cached_pred_handler: Optional[Callable[[str], Optional[dict]]] = None
_save_cached_pred_handler: Optional[Callable[[str, str, dict], None]] = None
_sync_virtual_badge_handler: Optional[Callable[[str, dict, Optional[str]], dict]] = None
_resolve_race_date_handler: Optional[Callable[[str], Optional[str]]] = None
_latest_post_times_ref: Optional[Dict[str, str]] = None


def set_predictor_dependencies(
    get_cached_pred: Optional[Callable[[str], Optional[dict]]] = None,
    save_cached_pred: Optional[Callable[[str, str, dict], None]] = None,
    sync_virtual_badge: Optional[Callable[[str, dict, Optional[str]], dict]] = None,
    resolve_race_date: Optional[Callable[[str], Optional[str]]] = None,
    latest_post_times: Optional[Dict[str, str]] = None,
):
    """外部キャッシュ・日付解決ハンドラーの登録"""
    global _get_cached_pred_handler, _save_cached_pred_handler
    global _sync_virtual_badge_handler, _resolve_race_date_handler, _latest_post_times_ref

    if get_cached_pred is not None:
        _get_cached_pred_handler = get_cached_pred
    if save_cached_pred is not None:
        _save_cached_pred_handler = save_cached_pred
    if sync_virtual_badge is not None:
        _sync_virtual_badge_handler = sync_virtual_badge
    if resolve_race_date is not None:
        _resolve_race_date_handler = resolve_race_date
    if latest_post_times is not None:
        _latest_post_times_ref = latest_post_times


def rank_min_desc(values: List[float]) -> List[int]:
    """降順ランク (値が大きいほど1位、同率は同率最小順位: pandas rank(ascending=False, method='min') と同等)"""
    sorted_unique = sorted(list(set(values)), reverse=True)
    val_to_rank = {}
    current_rank = 1
    for v in sorted_unique:
        val_to_rank[v] = current_rank
        current_rank += values.count(v)
    return [val_to_rank[v] for v in values]


def rank_min_asc(values: List[float]) -> List[int]:
    """昇順ランク (値が小さいほど1位、同率は同率最小順位: pandas rank(ascending=True, method='min') と同等)"""
    sorted_unique = sorted(list(set(values)), reverse=False)
    val_to_rank = {}
    current_rank = 1
    for v in sorted_unique:
        val_to_rank[v] = current_rank
        current_rank += values.count(v)
    return [val_to_rank[v] for v in values]


def calculate_dynamic_bet(score: float, odds: Optional[float] = None):
    """スコア・オッズに基づく単複動的推奨度判定"""
    if odds is not None and odds < 2.5:
        return "D", "見送り推奨(低オッズ)", True, 0, 0

    if score >= 0.650:
        return "S", "鉄板・大勝負", False, 300, 700
    elif score >= 0.550:
        return "A", "勝負レース", False, 200, 400
    elif score >= 0.400:
        return "B", "標準推奨", False, 100, 200
    elif score >= 0.330:
        return "C", "少額推奨", False, 100, 100
    else:
        return "D", "見送り推奨", True, 0, 0


def calc_trio_odds(o1: float, o2: float, o3: float) -> float:
    """Harville確率モデルに基づく三連複推定オッズ計算"""
    p1 = 0.80 / max(1.1, float(o1))
    p2 = 0.80 / max(1.1, float(o2))
    p3 = 0.80 / max(1.1, float(o3))
    p_sum = 0.0
    for a, b, c in [(p1, p2, p3), (p1, p3, p2), (p2, p1, p3), (p2, p3, p1), (p3, p1, p2), (p3, p2, p1)]:
        denom1 = max(0.01, 1.0 - a)
        denom2 = max(0.01, 1.0 - a - b)
        p_sum += a * (b / denom1) * (c / denom2)
    if p_sum <= 0:
        return 10.0
    return round(max(2.0, min(1000.0, 0.75 / p_sum)), 1)


def compute_hybrid_rec_from_inputs(
    race_name: str,
    race_details: Optional[dict],
    horses: List[dict],
    surface_type: int,
    prob_gachi: float,
    prob_himo: float,
    opponents: List[tuple],
    u0: int,
    honmei_horse: Optional[dict],
    race_date_str: str,
    force_final_odds: bool = False,
    real_trio_odds: Optional[dict] = None,
    track_bias_data: Optional[dict] = None,
    is_daily_top_race: bool = False,
    is_real_guarantee: bool = False,
    is_virtual_guarantee: bool = False,
    race_class_code: Optional[int] = None,
    real_umaren_odds: Optional[dict] = None,
) -> dict:
    """買い目戦略・調停・最新オッズ再推論判定を app/strategies/arbiter.py へ委任"""
    return _strategy_arbiter.evaluate_race(
        race_name=race_name,
        race_details=race_details,
        horses=horses,
        surface_type=surface_type,
        prob_gachi=prob_gachi,
        prob_himo=prob_himo,
        opponents=opponents,
        u0=u0,
        honmei_horse=honmei_horse,
        race_date_str=race_date_str,
        force_final_odds=force_final_odds,
        real_trio_odds=real_trio_odds,
        track_bias_data=track_bias_data,
        is_daily_top_race=is_daily_top_race,
        is_real_guarantee=is_real_guarantee,
        is_virtual_guarantee=is_virtual_guarantee,
        race_class_code=race_class_code,
        real_umaren_odds=real_umaren_odds,
    )


def apply_odds_anomaly_indicators(
    horses: List[dict],
    prev_horses: Optional[List[dict]] = None,
    odds_stage: Optional[str] = None
):
    """
    リアルタイムオッズの時系列異常・支持急増およびAI評価乖離のインジケーターを算出・付与
    1. オッズ推移トレンド判定 (odds_trend / odds_alert):
       - 朝イチ(08:30) または 馬体重発表時(60m)比で 25%以上の急落: 🔥直前支持急増 (plunge)
       - 連続3回以上のオッズ下落: 📉連続支持 (continuous_drop)
       - 基準比30%以上の上昇（かつ1.5倍以上上昇）: 📈人気軟化 (drift)
    2. AI評価×市場オッズの異常乖離シグナル:
       - AI印あり(◎◯▲△☆) かつ 5番人気以下(または単勝10倍以上) かつ 期待値(prob_win * odds) >= 1.15: 💎妙味穴 (value_gap)
       - AI無印(-) かつ 1〜2番人気(または単勝3.5倍以下) かつ 期待値(prob_win * odds) < 0.45: ⚠️過剰人気 (overbet)
    """
    if not horses:
        return
    prev_map = {}
    if prev_horses:
        for ph in prev_horses:
            if "umaban" in ph and str(ph["umaban"]).isdigit():
                prev_map[int(ph["umaban"])] = ph

    for h in horses:
        try:
            u = int(h.get("umaban", 0))
        except Exception:
            continue
        curr_raw = h.get("odds")
        curr_odds = round(float(curr_raw), 1) if (curr_raw is not None and float(curr_raw) > 0) else None
        prev_h = prev_map.get(u, {})

        hist: Dict[str, float] = {}
        for src in (prev_h.get("odds_history"), h.get("odds_history")):
            if isinstance(src, dict):
                for k, v in src.items():
                    if k in ODDS_STAGE_RANK and v is not None:
                        try:
                            fv = round(float(v), 1)
                            if fv > 0:
                                hist[k] = fv
                        except Exception:
                            pass

        base_raw = hist.get("0830") or h.get("base_odds") or prev_h.get("base_odds") or prev_h.get("odds") or curr_odds
        base_odds = round(float(base_raw), 1) if (base_raw is not None and float(base_raw) > 0) else None
        if base_odds and "0830" not in hist:
            hist["0830"] = base_odds

        if curr_odds:
            if odds_stage in ODDS_STAGE_RANK:
                hist[odds_stage] = curr_odds
            elif not hist:
                hist["0830"] = curr_odds

        ordered_hist = {k: hist[k] for k in ODDS_STAGE_ORDER if k in hist}
        h["odds_history"] = ordered_hist
        h["base_odds"] = base_odds

        weight_base = ordered_hist.get("60m") or base_odds
        stage_vals = [ordered_hist[k] for k in ODDS_STAGE_ORDER if k in ordered_hist]
        if curr_odds and (not stage_vals or stage_vals[-1] != curr_odds):
            stage_vals.append(curr_odds)

        compact_vals: List[float] = []
        for v in stage_vals:
            if not compact_vals or compact_vals[-1] != v:
                compact_vals.append(v)

        is_continuous_drop = len(compact_vals) >= 3 and all(
            compact_vals[i] > compact_vals[i + 1] for i in range(len(compact_vals) - 1)
        )

        if curr_odds and base_odds and base_odds > 0:
            pct = round(((curr_odds - base_odds) / base_odds) * 100.0, 1)
        else:
            pct = 0.0
        h["odds_change_pct"] = pct

        late_pct = 0.0
        is_late_plunge = False
        if curr_odds and weight_base and weight_base > 0 and "60m" in ordered_hist:
            late_pct = round(((curr_odds - weight_base) / weight_base) * 100.0, 1)
            if curr_odds <= weight_base * 0.80 and (weight_base - curr_odds) >= 1.0:
                is_late_plunge = True

        if len(compact_vals) >= 3:
            traj_str = "→".join(f"{v:.1f}" for v in compact_vals[-3:]) + "倍"
        elif base_odds and curr_odds:
            traj_str = f"{base_odds:.1f}→{curr_odds:.1f}倍"
        else:
            traj_str = f"{curr_odds:.1f}倍" if curr_odds else ""

        mark = str(h.get("mark") or "-")
        pop_raw = h.get("popularity")
        pop = int(pop_raw) if (pop_raw is not None and str(pop_raw).isdigit() and int(pop_raw) > 0) else None
        pw_raw = float(h.get("prob_win") or h.get("score") or 0.0)
        prob_win = pw_raw / 100.0 if pw_raw > 1.0 else pw_raw
        ev = round(prob_win * curr_odds, 2) if curr_odds else 0.0

        trend = None
        alert = None
        if curr_odds and base_odds and (
            (curr_odds <= base_odds * 0.75 and (base_odds - curr_odds) >= 1.0)
            or is_late_plunge
            or (is_continuous_drop and curr_odds <= base_odds * 0.82 and (base_odds - curr_odds) >= 1.0)
        ):
            trend = "plunge"
            ref_pct = late_pct if is_late_plunge else pct
            alert = f"🔥直前支持急増 {traj_str} ({ref_pct:+.0f}%)"
        elif mark in ("◎", "◯", "▲", "△", "☆") and (
            (pop is not None and pop >= 5) or (curr_odds and curr_odds >= 10.0)
        ) and ev >= 1.15:
            trend = "value_gap"
            pop_str = f"{pop}番人気" if pop else f"{curr_odds:.1f}倍"
            alert = f"💎妙味穴 ({pop_str} / EV {ev:.2f})"
        elif mark == "-" and (
            (pop is not None and pop in (1, 2)) or (curr_odds and curr_odds <= 3.5)
        ) and ev < 0.45:
            trend = "overbet"
            pop_str = f"{pop}番人気" if pop else f"{curr_odds:.1f}倍"
            alert = f"⚠️過剰人気 ({pop_str} / AI無印)"
        elif curr_odds and base_odds and curr_odds >= base_odds * 1.30 and (curr_odds - base_odds) >= 1.5:
            trend = "drift"
            alert = f"📈人気軟化 {traj_str} ({pct:+.0f}%)"

        h["odds_trend"] = trend
        h["odds_alert"] = alert


def build_prediction_and_recs(race_name: str, race_details: dict, race_id: str, horses: List[dict], course_meta: dict) -> dict:
    """特徴量構築・LightGBM推論・印決定・買い目生成コア"""
    empty_recommendation = {
        "target_umaban": None,
        "horse_name": None,
        "grade": "D",
        "confidence_label": "見送り推奨",
        "is_pass": True,
        "tansho_amount": 0,
        "fukusho_amount": 0,
        "total_amount": 0,
        "summary": "推奨馬なし (見送り)"
    }

    if not horses or len(horses) == 0:
        raise ValueError(f"レース {race_id} の出走馬データが空のため予想を生成できません。ダミー値による補完は禁止されています。")

    expected_num_features = 48
    if model_class_b_win:
        try:
            expected_num_features = model_class_b_win.num_feature()
        except Exception:
            expected_num_features = 48
    elif model_win:
        try:
            expected_num_features = model_win.num_feature()
        except Exception:
            expected_num_features = 48

    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    distance = course_meta.get("distance", 1600.0)
    surface_type = course_meta.get("surface_type", 0)
    condition_code = course_meta.get("condition_code", 0)
    turn_type = int(course_meta.get("turn_type", 2 if (venue_code == 4 and surface_type == 0 and int(distance) == 1000) else (1 if venue_code in (4, 5, 7) else 0)))
    race_num_int = int(course_meta.get("race_num_int", int(race_id[10:12]) if len(race_id) == 12 and race_id[10:12].isdigit() else 6))
    is_daily_top_race = bool(course_meta.get("is_daily_top_race", False))

    course_key = f"{venue_code}_{surface_type}"

    j_stats = stats_dict.get("jockey_stats", {})
    jc_stats = stats_dict.get("jockey_course_stats", {})
    t_stats = stats_dict.get("trainer_stats", {})
    tc_stats = stats_dict.get("trainer_course_stats", {})
    h_stats = stats_dict.get("horse_stats", {})

    head_count = len(horses)
    kinryo_list = [float(h.get("kinryo", 55.0)) for h in horses]
    mean_kinryo = sum(kinryo_list) / max(head_count, 1)
    if head_count > 1:
        kinryo_var = sum((x - mean_kinryo) ** 2 for x in kinryo_list) / (head_count - 1)
        kinryo_std_in_race = math.sqrt(max(0.0, kinryo_var))
    else:
        kinryo_std_in_race = 0.0

    all_female = head_count > 0 and all(int(h.get("sex_code", 0)) == 1 for h in horses)
    is_female_only = 1 if (course_meta.get("is_female_only_text") or all_female) else 0

    today_date = now_jst().date()
    r_date_meta = course_meta.get("race_date")
    if r_date_meta and isinstance(r_date_meta, str):
        try:
            today_date = datetime.date.fromisoformat(r_date_meta[:10])
        except Exception:
            pass

    horse_feature_data = []
    jc_rates = []
    tc_rates = []
    p1_orders = []
    career_list = []
    wins_list = []

    for h in horses:
        u = int(h.get("umaban") or 1)
        w_raw = h.get("wakuban")
        if w_raw is not None and str(w_raw).isdigit() and int(w_raw) > 0:
            w = int(w_raw)
        else:
            w = min(max(((u - 1) * 8 // max(head_count, 1)) + 1, 1), 8)
        k = float(h.get("kinryo", 55.0))

        raw_name = h.get("horse_name", "")
        h_clean = re.sub(r"[\s\u3000]", "", raw_name)

        stat = h_stats.get(h_clean)
        if stat:
            career_races = float(stat.get("career_races", 0.0))
            career_wins = float(stat.get("career_wins", round(career_races * float(stat.get("career_top3_rate", 0.25)) * 0.33)))
            career_top3_rate = float(stat.get("career_top3_rate", 0.25))
            prev1_order = float(stat.get("prev1_order", 99.0))
            prev1_pop = float(stat.get("prev1_pop", 10.0))
            prev1_odds = float(stat.get("prev1_odds", 50.0))
            prev2_order = float(stat.get("prev2_order", 99.0))

            last_date_str = stat.get("last_date", "")
            if last_date_str:
                try:
                    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    days_since_prev = float(max(0, (today_date - last_date).days))
                except Exception:
                    days_since_prev = 35.0
            else:
                days_since_prev = 35.0

            last_dist = float(stat.get("last_dist", distance))
            dist_diff = float(distance - last_dist)
        else:
            career_races = 0.0
            career_wins = 0.0
            career_top3_rate = 0.25
            prev1_order = 99.0
            prev1_pop = 10.0
            prev1_odds = 50.0
            prev2_order = 99.0
            days_since_prev = 999.0
            dist_diff = 0.0

        career_list.append(career_races)
        wins_list.append(career_wins)

        j_clean = re.sub(r"[▲△◇☆\s\u3000]", "", h.get("jockey", ""))
        t_clean = re.sub(r"\[.*?\]|[\s\u3000]", "", h.get("trainer", "") or "")

        j_rate = j_stats.get(j_clean, 0.25)
        jc_rate = jc_stats.get(f"{j_clean}_{course_key}", j_rate)
        t_rate = t_stats.get(t_clean, 0.25)
        tc_rate = tc_stats.get(f"{t_clean}_{course_key}", t_rate)

        rel_umaban = u / max(head_count, 1)
        kinryo_diff = k - mean_kinryo

        jc_rates.append(jc_rate)
        tc_rates.append(tc_rate)
        p1_orders.append(prev1_order)

        horse_feature_data.append({
            "w": w, "u": u, "k": k, "dist_diff": dist_diff,
            "career_races": career_races, "career_wins": career_wins, "career_top3_rate": career_top3_rate,
            "prev1_order": prev1_order, "prev1_pop": prev1_pop, "prev1_odds": prev1_odds, "prev2_order": prev2_order,
            "days_since_prev": days_since_prev, "j_rate": j_rate, "jc_rate": jc_rate, "t_rate": t_rate, "tc_rate": tc_rate,
            "rel_umaban": rel_umaban, "kinryo_diff": kinryo_diff,
            "base_feats": [
                w, u, k,
                venue_code, surface_type, distance, dist_diff, condition_code,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev,
                j_rate, jc_rate, t_rate, tc_rate,
                float(head_count), float(rel_umaban), float(kinryo_diff)
            ]
        })

    race_class_code = course_meta.get("race_class_code")
    if race_class_code is None:
        max_c = max(career_list) if career_list else 0.0
        max_w = max(wins_list) if wins_list else 0.0
        sorted_w = sorted(wins_list)
        q75_w = sorted_w[min(len(sorted_w) - 1, int(len(sorted_w) * 0.75))] if sorted_w else 0.0
        if max_c == 0:
            race_class_code = 0
        elif max_w == 0 and race_num_int <= 6:
            race_class_code = 1
        elif q75_w <= 1.0:
            race_class_code = 2
        elif q75_w <= 2.0:
            race_class_code = 3
        elif q75_w <= 3.0 and not (race_num_int == 11 and max_w >= 4):
            race_class_code = 4
        else:
            race_class_code = 5

    j_ranks = rank_min_desc(jc_rates)
    t_ranks = rank_min_desc(tc_rates)
    p1_ranks = rank_min_asc(p1_orders)

    feature_rows = []
    for idx, fd in enumerate(horse_feature_data):
        h = horses[idx]
        h_clean = re.sub(r"[\s\u3000]", "", h.get("horse_name", ""))
        stat = h_stats.get(h_clean)

        is_blinker_val = float(1 if h.get("is_blinker") else 0)
        is_first_blinker_val = float(1 if h.get("is_first_blinker") else 0)

        last_surface = int(stat.get("last_surface", surface_type)) if stat else surface_type
        is_turf_to_dirt_val = float(1 if (last_surface == 0 and surface_type == 1) else 0)
        is_dirt_to_turf_val = float(1 if (last_surface == 1 and surface_type == 0) else 0)

        last_dist = float(stat.get("last_dist", distance)) if stat else distance
        h_dist_diff = float(distance - last_dist)
        is_dist_shorten_val = float(1 if h_dist_diff <= -200.0 else 0)
        is_dist_lengthen_val = float(1 if h_dist_diff >= 200.0 else 0)

        h_days_since = 35.0
        if stat and stat.get("last_date"):
            try:
                ld = datetime.datetime.strptime(stat["last_date"], "%Y-%m-%d").date()
                h_days_since = float(max(0, (today_date - ld).days))
            except Exception:
                pass
        h_career = float(stat.get("career_races", 0.0)) if stat else 0.0
        was_last_rest = bool(stat.get("was_last_rest", 0)) if stat else False
        is_tataki_2nd_val = float(1 if (was_last_rest and 7.0 <= h_days_since <= 35.0 and h_career >= 2) else 0)

        new_feats = [
            is_blinker_val, is_first_blinker_val,
            is_turf_to_dirt_val, is_dirt_to_turf_val,
            is_dist_shorten_val, is_dist_lengthen_val,
            is_tataki_2nd_val
        ]

        prev1_last_3f_val = float(stat.get("last_3f", 36.0)) if stat and stat.get("last_3f") is not None else 36.0
        prev1_corner_final_val = float(stat.get("last_corner_final", 8.0)) if stat and stat.get("last_corner_final") is not None else 8.0
        prev1_is_front_val = 1.0 if prev1_corner_final_val <= 3.0 else 0.0
        prev1_is_closer_val = 1.0 if prev1_corner_final_val >= 8.0 else 0.0

        agari_feats = [
            prev1_last_3f_val,
            prev1_corner_final_val,
            prev1_is_front_val,
            prev1_is_closer_val
        ]

        raw_hw = h.get("horse_weight")
        if raw_hw is not None and float(raw_hw) > 0:
            hw_val = float(raw_hw)
        elif stat and stat.get("last_weight") is not None and float(stat.get("last_weight") or 0) > 0:
            hw_val = float(stat["last_weight"])
        else:
            hw_val = 470.0

        raw_wd = h.get("weight_diff")
        wd_val = float(raw_wd) if raw_wd is not None else 0.0
        abs_wd_val = abs(wd_val)
        k_val = float(h.get("kinryo", 55.0))
        weight_ratio_val = k_val / max(hw_val, 350.0)

        last_turn = int(stat.get("last_turn", turn_type)) if stat and stat.get("last_turn") is not None else turn_type
        is_turn_change_val = float(1 if (h_career > 0 and turn_type != last_turn) else 0)

        sex_code_val = float(h.get("sex_code", 0))
        age_val = float(h.get("age", 4.0))

        extra12_feats = [
            hw_val, wd_val, abs_wd_val, weight_ratio_val,
            float(turn_type), is_turn_change_val,
            float(race_num_int), float(race_class_code), float(is_female_only), float(kinryo_std_in_race),
            sex_code_val, age_val
        ]

        row25 = fd["base_feats"] + [float(j_ranks[idx]), float(t_ranks[idx]), float(p1_ranks[idx])]
        row32 = row25 + new_feats
        row36 = row32 + agari_feats
        row48 = row36 + extra12_feats

        # 50特徴量の構築
        c_races = float(fd.get("career_races", 0.0))
        c_wins = float(fd.get("career_wins", 0.0))
        c_win_rate = float(c_wins / max(c_races, 1.0)) if c_races > 0 else 0.08
        j_win_rate = round(float(fd.get("j_rate", 0.25)) * 0.33, 4)
        jc_win_rate = round(float(fd.get("jc_rate", 0.25)) * 0.33, 4)
        t_win_rate = round(float(fd.get("t_rate", 0.25)) * 0.33, 4)
        tc_win_rate = round(float(fd.get("tc_rate", 0.25)) * 0.33, 4)

        j_clean_str = re.sub(r"[▲△◇☆\s\u3000]", "", h.get("jockey", ""))
        t_clean_str = re.sub(r"\[.*?\]|[\s\u3000]", "", h.get("trainer", "") or "")
        jt_key = f"{j_clean_str}_{t_clean_str}"
        jt_rate_val = float(stats_dict.get("jockey_trainer_stats", {}).get(jt_key, 0.25))

        k_val = float(fd.get("k", 55.0))
        last_k = float(stat.get("last_kinryo", k_val)) if stat else k_val
        kinryo_diff_prev_val = float(k_val - last_k)

        p1_order_val = float(fd.get("prev1_order", 99.0))
        p1_pop_val = float(fd.get("prev1_pop", 10.0))
        p1_time_diff_val = float(stat.get("prev1_time_diff", 3.0)) if stat else 3.0
        p1_pop_order_diff_val = float(p1_pop_val - p1_order_val)
        p1_l3f_rank_val = float(stat.get("prev1_last_3f_rank", 10.0)) if stat else 10.0

        hs_key = f"{h_clean}_{surface_type}"
        horse_surf_val = float(stats_dict.get("horse_surface_stats", {}).get(hs_key, 0.25))

        prev_time_diff_min_val = float(stat.get("prev_time_diff_min", 3.0)) if stat else 3.0

        js_key = f"{j_clean_str}_{surface_type}"
        jockey_surf_val = float(stats_dict.get("jockey_surface_stats", {}).get(js_key, 0.25))
        p1_class_diff_val = float(stat.get("prev1_class_diff", 0.0)) if stat else 0.0

        # 前走道中押し上げ幅 (position_gain)
        last_c1 = float(stat.get("last_corner_first", 8.0)) if stat else 8.0
        last_cf = float(stat.get("last_corner_final", 8.0)) if stat else 8.0
        prev1_pos_gain_val = float(last_c1 - last_cf)

        extra10_feats = [
            float(jt_rate_val),
            float(kinryo_diff_prev_val),
            float(p1_time_diff_val),
            float(prev_time_diff_min_val),
            float(p1_pop_order_diff_val),
            float(p1_l3f_rank_val),
            float(prev1_pos_gain_val),
            float(horse_surf_val),
            float(jockey_surf_val),
            float(p1_class_diff_val),
        ]
        row58 = row48 + extra10_feats

        row50 = [
            float(fd["w"]), float(fd["u"]), float(k_val), float(venue_code), float(surface_type), float(distance), float(fd["dist_diff"]), float(condition_code),
            float(c_races), float(c_win_rate), float(fd["career_top3_rate"]),
            float(p1_order_val), float(p1_pop_val), float(fd["prev1_odds"]), float(fd["prev2_order"]), float(fd["days_since_prev"]),
            float(j_win_rate), float(fd["j_rate"]), float(jc_win_rate), float(fd["jc_rate"]),
            float(t_win_rate), float(fd["t_rate"]), float(tc_win_rate), float(fd["tc_rate"]),
            float(head_count), float(fd["rel_umaban"]), float(fd["kinryo_diff"]),
            float(j_ranks[idx]), float(t_ranks[idx]), float(p1_ranks[idx]),
            float(is_blinker_val), float(is_first_blinker_val), float(is_turf_to_dirt_val), float(is_dirt_to_turf_val),
            float(is_dist_shorten_val), float(is_dist_lengthen_val), float(is_tataki_2nd_val),
            float(prev1_last_3f_val), float(prev1_corner_final_val), float(prev1_is_front_val), float(prev1_is_closer_val),
            float(jt_rate_val), float(kinryo_diff_prev_val), float(p1_time_diff_val), float(p1_pop_order_diff_val), float(p1_l3f_rank_val),
            float(horse_surf_val), float(prev_time_diff_min_val), float(jockey_surf_val), float(p1_class_diff_val)
        ]

        if expected_num_features == 58:
            feature_rows.append(row58)
        elif expected_num_features == 50:
            feature_rows.append(row50)
        elif expected_num_features == 48:
            feature_rows.append(row48)
        elif expected_num_features == 36:
            feature_rows.append(row36)
        elif expected_num_features == 32:
            feature_rows.append(row32)
        elif expected_num_features == 25:
            feature_rows.append(row25)
        elif expected_num_features == 19:
            feature_rows.append(row25[:19])
        else:
            feature_rows.append(row58[:expected_num_features] if len(row58) >= expected_num_features else row48[:expected_num_features])

    w_scores = None
    t_scores = None

    # 1. クラス別特化モデル (A/B/C) が利用可能な場合は最優先で推論（不要な旧6モデル推論を完全排除し3〜5倍高速化）
    target_class_code = race_class_code
    m_cls_w, m_cls_t, cls_key = None, None, None
    if target_class_code in (0, 1) and model_class_a_win and model_class_a_top3:
        m_cls_w, m_cls_t, cls_key = model_class_a_win, model_class_a_top3, "class_a"
    elif target_class_code in (2, 3) and model_class_b_win and model_class_b_top3:
        m_cls_w, m_cls_t, cls_key = model_class_b_win, model_class_b_top3, "class_b"
    elif target_class_code in (4, 5) and model_class_c_win and model_class_c_top3:
        m_cls_w, m_cls_t, cls_key = model_class_c_win, model_class_c_top3, "class_c"

    if m_cls_w and m_cls_t:
        try:
            n_cw = m_cls_w.num_feature()
            n_ct = m_cls_t.num_feature()
            inp_cw = [r[:n_cw] for r in feature_rows]
            inp_ct = [r[:n_ct] for r in feature_rows]
            raw_cw = np.array(m_cls_w.predict(inp_cw, num_threads=4))
            raw_ct = np.array(m_cls_t.predict(inp_ct, num_threads=4))

            # キャリブレーション補正 (FR-ML-03: 区分線形補間 np.interp)
            cal_info = (calibration_table or {}).get(cls_key, {})
            w_cal_tab = cal_info.get("win_calib", [])
            t_cal_tab = cal_info.get("top3_calib", [])

            def _calib_linear(vals, tab):
                if not tab:
                    return vals
                xp = [float(e.get("pred", 0.0)) for e in tab]
                fp = [float(e.get("calib", 0.0)) for e in tab]
                return np.interp(vals, xp, fp)

            w_scores = _calib_linear(raw_cw, w_cal_tab)
            t_scores = _calib_linear(raw_ct, t_cal_tab)
        except Exception as ex_cls:
            print(f"クラス別モデル推論フォールバック ({cls_key}): {ex_cls}")
            w_scores = None
            t_scores = None

    # 2. クラス別モデル未該当レースまたはエラー時のみ、旧3大GBDTアンサンブル (LightGBM × CatBoost × XGBoost) にフォールバック
    if (w_scores is None or t_scores is None):
        from app.ml.model_loader import ensure_fallback_ensemble_loaded
        ensure_fallback_ensemble_loaded()
    if (w_scores is None or t_scores is None) and model_win and model_top3:
        try:
            n_win = model_win.num_feature()
            n_top3 = model_top3.num_feature()
            inp_win = [r[:n_win] for r in feature_rows]
            inp_top3 = [r[:n_top3] for r in feature_rows]

            # 1. LightGBM 推論
            w_lgb = np.array(model_win.predict(inp_win, num_threads=4))
            t_lgb = np.array(model_top3.predict(inp_top3, num_threads=4))

            # 重み設定 (ensemble_config または デフォルト)
            cfg_win = (ensemble_config or {}).get("models", {}).get("win", {})
            wt_lgb_w = float(cfg_win.get("lightgbm", {}).get("weight", 0.30))
            wt_cb_w = float(cfg_win.get("catboost", {}).get("weight", 0.20))
            wt_xgb_w = float(cfg_win.get("xgboost", {}).get("weight", 0.50))

            cfg_top3 = (ensemble_config or {}).get("models", {}).get("top3", {})
            wt_lgb_t = float(cfg_top3.get("lightgbm", {}).get("weight", 0.20))
            wt_cb_t = float(cfg_top3.get("catboost", {}).get("weight", 0.30))
            wt_xgb_t = float(cfg_top3.get("xgboost", {}).get("weight", 0.50))

            # 2. CatBoost 推論
            if model_cb_win and model_cb_top3:
                w_cb = np.array(model_cb_win.predict_proba(inp_win)[:, 1])
                t_cb = np.array(model_cb_top3.predict_proba(inp_top3)[:, 1])
            else:
                w_cb, t_cb = w_lgb, t_lgb
                wt_lgb_w += wt_cb_w
                wt_lgb_t += wt_cb_t
                wt_cb_w = 0.0
                wt_cb_t = 0.0

            # 3. XGBoost 推論
            if model_xgb_win and model_xgb_top3:
                import xgboost as xgb
                feat_names = (ensemble_config or {}).get("features", DEFAULT_FEATURES_48)
                df_win = pd.DataFrame(inp_win, columns=feat_names[:n_win])
                df_top3 = pd.DataFrame(inp_top3, columns=feat_names[:n_top3])
                d_win = xgb.DMatrix(df_win)
                d_top3 = xgb.DMatrix(df_top3)
                w_xgb = np.array(model_xgb_win.predict(d_win))
                t_xgb = np.array(model_xgb_top3.predict(d_top3))
            else:
                w_xgb, t_xgb = w_lgb, t_lgb
                wt_lgb_w += wt_xgb_w
                wt_lgb_t += wt_xgb_t
                wt_xgb_w = 0.0
                wt_xgb_t = 0.0

            w_scores = wt_lgb_w * w_lgb + wt_cb_w * w_cb + wt_xgb_w * w_xgb
            t_scores = wt_lgb_t * t_lgb + wt_cb_t * t_cb + wt_xgb_t * t_xgb
        except Exception as e:
            raise RuntimeError(f"AIアンサンブル推論エラー ({race_id}): {e}")

    # 3. 推論スコアを出走各馬へ反映
    if w_scores is not None and t_scores is not None:
        for idx, h in enumerate(horses):
            h["prob_win"] = round(float(w_scores[idx]), 4)
            h["prob_top3"] = round(float(t_scores[idx]), 4)
            h["score"] = h["prob_win"]
    elif model:
        try:
            n_mod = model.num_feature()
            scores = model.predict([r[:n_mod] for r in feature_rows], num_threads=4)
            for idx, h in enumerate(horses):
                h["prob_win"] = round(float(scores[idx]), 4)
                h["prob_top3"] = round(float(scores[idx]), 4)
                h["score"] = round(float(scores[idx]), 4)
        except Exception as e:
            raise RuntimeError(f"AIモデル推論エラー ({race_id}): {e}")
    else:
        raise RuntimeError(f"AIモデルがロードされていないため、レース {race_id} の推論を実行できません。")

    sorted_by_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    for h in horses:
        h["mark"] = "-"

    if len(sorted_by_win) >= 1:
        sorted_by_win[0]["mark"] = "◎"
    if len(sorted_by_win) >= 2:
        sorted_by_win[1]["mark"] = "◯"

    remaining_horses = [h for h in sorted_by_win if h["mark"] == "-"]
    sorted_by_top3 = sorted(remaining_horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
    if len(sorted_by_top3) >= 1:
        sorted_by_top3[0]["mark"] = "▲"
    if len(sorted_by_top3) >= 2:
        sorted_by_top3[1]["mark"] = "△"
    if len(sorted_by_top3) >= 3:
        sorted_by_top3[2]["mark"] = "☆"

    sorted_horses = sorted_by_win
    honmei_horse = next((h for h in sorted_horses if h["mark"] == "◎"), None)

    prev_h_list = (course_meta or {}).get("prev_horses")
    stg = (course_meta or {}).get("odds_stage")
    apply_odds_anomaly_indicators(horses, prev_horses=prev_h_list, odds_stage=stg)

    if honmei_horse:
        u_num = honmei_horse["umaban"]
        h_name = honmei_horse["horse_name"]
        h_score = honmei_horse["score"]
        h_odds = honmei_horse.get("odds")

        grade, conf_label, is_pass, b_tan, b_fuku = calculate_dynamic_bet(h_score, h_odds)
        total_amt = b_tan + b_fuku

        if is_pass:
            summary_text = f"{u_num}番 {h_name} [{conf_label}] (期待値不足のため見送り推奨)"
        else:
            summary_text = f"{u_num}番 {h_name} [{conf_label}] (単勝{b_tan}円 + 複勝{b_fuku}円 / 計{total_amt}円)"

        recommendation = {
            "target_umaban": u_num,
            "horse_name": h_name,
            "grade": grade,
            "confidence_label": conf_label,
            "is_pass": is_pass,
            "tansho_amount": b_tan,
            "fukusho_amount": b_fuku,
            "total_amount": total_amt,
            "summary": summary_text
        }
    else:
        raise ValueError(f"レース {race_id} の本命馬(◎)を特定できませんでした。ダミー買い目の生成は禁止されています。")

    hybrid_rec: Optional[dict] = None
    track_bias_data: Optional[dict] = None
    if len(horses) < 3:
        raise ValueError(f"レース {race_id} の出走頭数が少なすぎます ({len(horses)}頭)。ダミー予想の生成は禁止されています。")

    h_honmei = next((h for h in horses if h["mark"] == "◎"), sorted_horses[0])
    h_taiko = next((h for h in horses if h["mark"] == "◯"), (sorted_horses[1] if len(sorted_horses) > 1 else sorted_horses[0]))
    h_kuro = next((h for h in horses if h["mark"] == "▲"), (sorted_horses[2] if len(sorted_horses) > 2 else sorted_horses[0]))
    h_osae = next((h for h in horses if h["mark"] == "△"), (sorted_horses[3] if len(sorted_horses) > 3 else sorted_horses[0]))

    u0 = int(h_honmei["umaban"])
    u1 = int(h_taiko["umaban"])
    u2 = int(h_kuro["umaban"])
    u3 = int(h_osae["umaban"])

    s_odds = sorted(horses, key=lambda x: (x.get("odds") if x.get("odds") is not None and x.get("odds") > 0 else 999.0))
    o1 = float(s_odds[0].get("odds") or 50.0) if len(s_odds) >= 1 else 50.0
    o2 = float(s_odds[1].get("odds") or 50.0) if len(s_odds) >= 2 else 50.0
    o3 = float(s_odds[2].get("odds") or 50.0) if len(s_odds) >= 3 else 50.0

    inv_sum = sum(1.0 / max(float(h.get("odds") or 50.0), 1.01) for h in horses)
    shares = [(1.0 / max(float(h.get("odds") or 50.0), 1.01)) / inv_sum for h in horses]
    top3_share = sum(sorted(shares, reverse=True)[:3])
    entropy = -sum(p * math.log(max(p, 1e-6)) for p in shares)

    s_ai = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    ai_w1 = float(s_ai[0].get("prob_win", 0.0)) if len(s_ai) >= 1 else 0.0
    ai_w2 = float(s_ai[1].get("prob_win", 0.0)) if len(s_ai) >= 2 else 0.0
    ai_diff = ai_w1 - ai_w2

    front_runners = 0
    for h in horses:
        h_clean = re.sub(r"[\s\u3000]", "", h.get("horse_name", ""))
        st = h_stats.get(h_clean, {})
        if float(st.get("last_corner_final", 8.0)) <= 3.0:
            front_runners += 1
    front_ratio = front_runners / max(len(horses), 1)

    race_feat_row = [
        float(head_count), float(distance), float(surface_type), float(condition_code), float(venue_code),
        float(o1), float(o2), float(o3), float(o2 / max(o1, 0.5)),
        float(top3_share), float(entropy),
        float(ai_w1), float(ai_w2), float(ai_diff),
        float(front_runners), float(front_ratio),
        float(turn_type), float(race_class_code), float(is_female_only), float(kinryo_std_in_race)
    ]

    if not model_gachi:
        raise RuntimeError(f"model_gachi がロードされていないため、レース {race_id} のガチ度を推論できません。")
    try:
        n_g = model_gachi.num_feature()
        prob_gachi = round(float(model_gachi.predict([race_feat_row[:n_g]], num_threads=2)[0]), 4)
    except Exception as e:
        raise RuntimeError(f"model_gachi 推論エラー ({race_id}): {e}")

    if not model_himo:
        raise RuntimeError(f"model_himo がロードされていないため、レース {race_id} の混戦度を推論できません。")
    try:
        n_h = model_himo.num_feature()
        prob_himo = round(float(model_himo.predict([race_feat_row[:n_h]], num_threads=2)[0]), 4)
    except Exception as e:
        raise RuntimeError(f"model_himo 推論エラー ({race_id}): {e}")

    real_odds_data = fetch_real_odds(race_id)
    umaren_real_map = real_odds_data.get("umaren", {})
    trio_real_map = real_odds_data.get("trio", {})

    def get_umaren_odds_info(ua, ub):
        comb = (min(ua, ub), max(ua, ub))
        if comb in umaren_real_map:
            return umaren_real_map[comb], True
        else:
            h_a = next((h for h in horses if int(h.get("umaban", 0)) == ua), {})
            h_b = next((h for h in horses if int(h.get("umaban", 0)) == ub), {})
            oa = float(h_a.get("odds") or 0.0)
            ob = float(h_b.get("odds") or 0.0)
            est_o = round(max(2.5, 0.65 * oa * ob), 1) if (oa > 0 and ob > 0) else 0.0
            return est_o, False

    u01_odds, u01_is_real = get_umaren_odds_info(u0, u1) if (u0 and u1) else (0.0, False)
    u02_odds, u02_is_real = get_umaren_odds_info(u0, u2) if (u0 and u2) else (0.0, False)
    u03_odds, u03_is_real = get_umaren_odds_info(u0, u3) if (u0 and u3) else (0.0, False)

    opponents = []
    if u1: opponents.append((u1, u01_odds, u01_is_real, "◯"))
    if u2: opponents.append((u2, u02_odds, u02_is_real, "▲"))
    if u3: opponents.append((u3, u03_odds, u03_is_real, "△"))

    r_date_str = (course_meta or {}).get("race_date") or today_jst_str()
    force_final_flag = bool((course_meta or {}).get("force_final_odds", False))
    is_real_guarantee = bool((course_meta or {}).get("is_real_guarantee", False))
    is_virtual_guarantee = bool((course_meta or {}).get("is_virtual_guarantee", False))
    if not is_real_guarantee and not is_virtual_guarantee and race_id:
        try:
            from app.strategies.venue_guarantee import get_race_guarantee_status
            is_real_guarantee, is_virtual_guarantee = get_race_guarantee_status(race_id, r_date_str)
        except Exception:
            pass

    c_det = (race_details or {}).get("course_details", "") if isinstance(race_details, dict) else ""
    surf_str = "芝" if "芝" in c_det else ("ダート" if "ダ" in c_det else "障害")
    track_bias_data = compute_track_bias_and_matches(
        race_id=race_id,
        race_date=r_date_str,
        surface_name=surf_str,
        horses=sorted_horses
    )

    hybrid_rec = compute_hybrid_rec_from_inputs(
        race_name=race_name,
        race_details=race_details,
        horses=sorted_horses,
        surface_type=surface_type,
        prob_gachi=prob_gachi,
        prob_himo=prob_himo,
        opponents=opponents,
        u0=u0,
        honmei_horse=honmei_horse,
        race_date_str=r_date_str,
        force_final_odds=force_final_flag,
        real_trio_odds=trio_real_map,
        track_bias_data=track_bias_data,
        is_daily_top_race=is_daily_top_race,
        is_real_guarantee=is_real_guarantee,
        is_virtual_guarantee=is_virtual_guarantee,
        race_class_code=race_class_code,
        real_umaren_odds=umaren_real_map,
    )

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    if hybrid_rec:
        is_p = hybrid_rec.get("is_participate", False)
        t_amt = hybrid_rec.get("tansho_amount", 0)
        f_amt = hybrid_rec.get("fukusho_amount", 0)
        tot_amt = hybrid_rec.get("total_amount", 0)
        st_type = hybrid_rec.get("strategy_type", "skip")
        b_cat = hybrid_rec.get("bet_category", "skip")
        grd = "S" if b_cat == "super" else ("A" if b_cat == "real" else ("B" if b_cat == "virtual" else "D"))
        clabel = hybrid_rec.get("badge_text") or ("⭐勝負R" if b_cat == "super" else ("🎯実勝負" if b_cat == "real" else ("🎮仮想" if b_cat == "virtual" else "見送り推奨")))
        recommendation = {
            "target_umaban": u0 if is_p else (honmei_horse["umaban"] if honmei_horse else None),
            "horse_name": honmei_horse["horse_name"] if honmei_horse else None,
            "grade": grd,
            "confidence_label": clabel,
            "is_pass": not is_p,
            "tansho_amount": t_amt,
            "fukusho_amount": f_amt,
            "total_amount": tot_amt,
            "summary": hybrid_rec.get("summary", ""),
            "bet_category": b_cat,
            "is_virtual_bet": hybrid_rec.get("is_virtual_bet", False),
            "badge_text": clabel
        }

    return {
        "race_name": race_name,
        "race_details": race_details,
        "horses": horses_sorted_by_num,
        "recommendation": recommendation,
        "hybrid_recommendation": hybrid_rec,
        "track_bias": track_bias_data
    }


def predict_race_core(
    race_id: str,
    force_refresh: bool = False,
    is_auto_final: bool = False,
    odds_stage: Optional[str] = None
) -> dict:
    """レース予想実行オーケストレーター"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    cached_pred = _get_cached_pred_handler(race_id) if _get_cached_pred_handler else None

    # 発走後判定
    is_post_race = False
    if cached_pred:
        r_d_str = (_resolve_race_date_handler(race_id) if _resolve_race_date_handler else None) or today_jst_str()
        if r_d_str < today_jst_str():
            is_post_race = True
        elif r_d_str == today_jst_str():
            p_str = (cached_pred.get("race_details") or {}).get("post_time", "")
            m_t = re.search(r"(\d{1,2}):(\d{2})", p_str)
            if m_t:
                d_parts = [int(p) for p in r_d_str.split("-")]
                est_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], int(m_t.group(1)), int(m_t.group(2)))
                if now_jst_naive() >= est_dt:
                    is_post_race = True

    if cached_pred and (not force_refresh or is_post_race):
        for h in cached_pred.get("horses", []):
            if "score" not in h or h["score"] is None:
                h["score"] = round(float(h.get("prob_win") or 0.0) * 100.0, 1)

        latest_pt = _latest_post_times_ref.get(race_id) if _latest_post_times_ref is not None else None
        supabase = get_supabase_client()
        if supabase:
            try:
                s_res = supabase.table("keiba_scheduled_races").select("race_date, post_datetime").eq("race_id", race_id).execute()
                if s_res.data and len(s_res.data) > 0:
                    sched_post_dt_raw = s_res.data[0].get("post_datetime")
                    if not latest_pt and sched_post_dt_raw:
                        m_pt = re.search(r"(\d{1,2}):(\d{2})", sched_post_dt_raw)
                        if m_pt:
                            latest_pt = f"{int(m_pt.group(1)):02d}:{int(m_pt.group(2)):02d}発走"
                            if _latest_post_times_ref is not None:
                                _latest_post_times_ref[race_id] = latest_pt
            except Exception:
                pass

        if latest_pt:
            r_details = cached_pred.get("race_details") or {}
            cur_pt = r_details.get("post_time", "")
            if cur_pt != latest_pt:
                r_details["post_time"] = latest_pt
                cached_pred["race_details"] = r_details
                if supabase:
                    try:
                        supabase.table("keiba_predictions_cache").update({"data_json": cached_pred}).eq("race_id", race_id).execute()
                    except Exception as ue:
                        print(f"予想キャッシュ出走時刻更新エラー ({race_id}): {ue}")

        if is_before_morning_prediction(r_d_str):
            hr = cached_pred.get("hybrid_recommendation")
            if hr and isinstance(hr, dict):
                hr["is_real_bet_target"] = False
                hr["real_bet_badge"] = None
                hr["real_bet_level"] = "none"
                hr["is_virtual_bet_target"] = False
                hr["virtual_bet_slot"] = None
                hr["virtual_badge"] = None
                hr["is_promoted_today"] = False
            return cached_pred

        if _sync_virtual_badge_handler:
            cached_pred = _sync_virtual_badge_handler(race_id, cached_pred, r_d_str)

        # 【整合性絶対保証】見送りレース (strategy_type == 'skip' または total_amount == 0) は絶対に実馬券対象外にする
        hr_c = cached_pred.get("hybrid_recommendation")
        if hr_c and isinstance(hr_c, dict):
            if hr_c.get("strategy_type") == "skip" or (hr_c.get("total_amount") or 0) == 0:
                hr_c["is_real_bet_target"] = False
                hr_c["real_bet_badge"] = None
                hr_c["real_bet_level"] = "none"

        return cached_pred

    # 出馬表スクレイピング
    try:
        race_name, race_details, horses, course_meta = fetch_shutuba_table(race_id)
        target_race_date = (_resolve_race_date_handler(race_id) if _resolve_race_date_handler else None) or today_jst_str()
        if not target_race_date or target_race_date == today_jst_str():
            target_race_date = (_resolve_race_date_handler(race_id) if _resolve_race_date_handler else None) or today_jst_str()

        if force_refresh and target_race_date == today_jst_str():
            p_time_str = race_details.get("post_time", "")
            post_dt = None
            m_time = re.search(r"(\d{1,2}):(\d{2})", p_time_str)
            if m_time:
                hh, mm = int(m_time.group(1)), int(m_time.group(2))
                d_parts = [int(p) for p in target_race_date.split("-")]
                post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hh, mm)
            if post_dt and now_jst_naive() >= post_dt:
                raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")

        if course_meta is None:
            course_meta = {}
        course_meta["race_date"] = target_race_date
        course_meta["odds_stage"] = odds_stage
        prev_horses_list = cached_pred.get("horses", []) if isinstance(cached_pred, dict) else None
        course_meta["prev_horses"] = prev_horses_list
        if is_auto_final or odds_stage == "05m":
            course_meta["force_final_odds"] = True

        top_rid = get_or_select_daily_top_race_id(target_race_date)
        prev_was_daily_top = bool(
            isinstance(cached_pred, dict)
            and (cached_pred.get("hybrid_recommendation") or {}).get("strategy_type") == "daily_top"
            and (cached_pred.get("hybrid_recommendation") or {}).get("is_real_bet_target")
        )
        course_meta["is_daily_top_race"] = bool((top_rid and race_id == top_rid) or (prev_was_daily_top and top_rid is not None))

        result = build_prediction_and_recs(race_name, race_details, race_id, horses, course_meta)
        apply_odds_anomaly_indicators(result["horses"], prev_horses=prev_horses_list, odds_stage=odds_stage)

        resp_data = {
            "race_id": race_id,
            "race_name": result["race_name"],
            "race_details": result["race_details"],
            "horses": result["horses"],
            "recommendation": result["recommendation"],
            "hybrid_recommendation": result.get("hybrid_recommendation"),
            "track_bias": result.get("track_bias"),
            "last_odds_stage": odds_stage
        }

        if _save_cached_pred_handler:
            _save_cached_pred_handler(race_id, target_race_date, resp_data)

        if is_before_morning_prediction(target_race_date):
            hr = resp_data.get("hybrid_recommendation")
            if hr and isinstance(hr, dict):
                hr["is_real_bet_target"] = False
                hr["real_bet_badge"] = None
                hr["real_bet_level"] = "none"
                hr["is_virtual_bet_target"] = False
                hr["virtual_bet_slot"] = None
                hr["virtual_badge"] = None
                hr["is_promoted_today"] = False
        else:
            try:
                # 【重要軽量化】レース予想時に毎回36レース全件をSupabaseからフェッチする重い日次厳選処理を抑制
                top_rid = get_or_select_daily_top_race_id(target_race_date)
                if top_rid == race_id:
                    promoted_rid = ensure_daily_single_real_bet(target_race_date)
                    if promoted_rid == race_id and not (resp_data.get("hybrid_recommendation") or {}).get("is_real_bet_target"):
                        reloaded = _get_cached_pred_handler(race_id) if _get_cached_pred_handler else None
                        if reloaded:
                            resp_data = reloaded
            except Exception as de:
                print(f"[predict_race_core] daily_top同期警告 ({race_id}): {de}")
            finally:
                import gc
                gc.collect()

        supabase = get_supabase_client()
        if supabase:
            try:
                has_weights_now = any(h.get("horse_weight") is not None for h in result["horses"])
                if is_auto_final or odds_stage == "05m":
                    new_pred_flag = 5
                elif odds_stage == "15m":
                    new_pred_flag = 4
                elif odds_stage == "30m":
                    new_pred_flag = 3
                elif odds_stage == "60m":
                    new_pred_flag = 2 if has_weights_now else 1
                else:
                    new_pred_flag = 1
                supabase.table("keiba_scheduled_races").update({"is_predicted": new_pred_flag}).eq("race_id", race_id).lt("is_predicted", new_pred_flag).execute()
            except Exception:
                pass

        if not is_before_morning_prediction(target_race_date) and _sync_virtual_badge_handler:
            resp_data = _sync_virtual_badge_handler(race_id, resp_data, target_race_date)

        # 【整合性絶対保証】見送りレース (strategy_type == 'skip' または total_amount == 0) は絶対に実馬券対象外にする
        hr_final = resp_data.get("hybrid_recommendation")
        if hr_final and isinstance(hr_final, dict):
            if hr_final.get("strategy_type") == "skip" or (hr_final.get("total_amount") or 0) == 0:
                hr_final["is_real_bet_target"] = False
                hr_final["real_bet_badge"] = None
                hr_final["real_bet_level"] = "none"

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")
