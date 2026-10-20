# -*- coding: utf-8 -*-
"""
app/maintenance/recalculator.py

開催期間内のキャッシュ済み予想・確定成績への最新買い目・戦略ロジック一括再計算パイプライン
"""

from typing import Optional, Callable, Dict, Any, List, Set
from fastapi import HTTPException

from app.core.database import get_supabase_client
from app.ml import compute_hybrid_rec_from_inputs
from app.settlement import process_settlement_calculation
from app.scrapers import fetch_real_odds

# 外部DI
_today_jst_str_fn: Optional[Callable] = None
_save_cached_prediction_fn: Optional[Callable] = None
_period_venues_cache_ref: Optional[Dict[str, Any]] = None


def set_recalculator_dependencies(
    today_jst_str: Callable,
    save_cached_prediction: Callable,
    period_venues_cache: Dict[str, Any],
):
    """再計算エンジンが必要とする外部依存を登録"""
    global _today_jst_str_fn, _save_cached_prediction_fn, _period_venues_cache_ref
    _today_jst_str_fn = today_jst_str
    _save_cached_prediction_fn = save_cached_prediction
    _period_venues_cache_ref = period_venues_cache


def recalculate_period_records():
    """今開催期間のキャッシュ済み予想・確定成績に最新の買い目・実戦推奨ロジックを再適用して一括更新"""
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase未接続")

    preds_res = supabase.table("keiba_predictions_cache").select("race_id, race_date, data_json").execute()
    results_res = supabase.table("keiba_race_results_cache").select("race_id, race_date, result_json").execute()

    preds_list = preds_res.data or []
    res_map = {r["race_id"]: r.get("result_json") for r in (results_res.data or [])}
    valid_rids = set(res_map.keys())

    today_str = _today_jst_str_fn() if _today_jst_str_fn else ""
    updated_preds = 0
    updated_settlements = 0
    participating_count = 0
    total_bet_sum = 0
    total_pay_sum = 0
    total_pass_benefit = 0
    rb_count = 0
    rb_bet_sum = 0
    rb_pay_sum = 0

    for row in preds_list:
        rid = row["race_id"]
        rdate = row.get("race_date") or today_str
        pred_data = row.get("data_json") or {}
        if not pred_data:
            continue

        rname = pred_data.get("race_name", "")
        rdetails = pred_data.get("race_details") or {}
        horses = pred_data.get("horses") or []
        old_h = pred_data.get("hybrid_recommendation") or {}

        c_details = rdetails.get("course_details", "") if isinstance(rdetails, dict) else ""
        surf = 0
        if "障" in c_details or "障害" in rname:
            surf = 2
        elif "ダ" in c_details:
            surf = 1

        honmei = next((h for h in horses if h.get("mark") == "◎"), None)
        if not honmei and horses:
            honmei = sorted(horses, key=lambda x: x.get("score", 0), reverse=True)[0]
        u0 = int(honmei["umaban"]) if honmei else 0

        # 既存の馬連オッズ情報（実オッズ含む）を復元
        old_bets = old_h.get("hypothetical_umaren_bets") or old_h.get("umaren_bets") or []
        odds_by_opp = {}
        for ob in old_bets:
            comb = ob.get("combination", [])
            if len(comb) == 2:
                opp_u = comb[1] if comb[0] == u0 else comb[0]
                odds_by_opp[int(opp_u)] = (float(ob.get("odds") or 0.0), bool(ob.get("is_real_odds", True)))

        opponents = []
        for mark_sym in ["◯", "▲", "△"]:
            h_opp = next((h for h in horses if h.get("mark") == mark_sym or (mark_sym == "◯" and h.get("mark") in ("〇", "○"))), None)
            if h_opp:
                ou = int(h_opp["umaban"])
                if ou in odds_by_opp:
                    o_val, o_real = odds_by_opp[ou]
                else:
                    oa = float(honmei.get("odds") or 0.0) if honmei else 0.0
                    ob_val = float(h_opp.get("odds") or 0.0)
                    o_val = round(max(2.5, 0.65 * oa * ob_val), 1) if (oa > 0 and ob_val > 0) else 0.0
                    o_real = False
                opponents.append((ou, o_val, o_real, mark_sym))

        # 既存の三連複実オッズを復元
        old_t_bets = old_h.get("trio_bets") or ([old_h.get("trio_bet")] if old_h.get("trio_bet") else [])
        real_trio_map = {}
        for tb in old_t_bets:
            if isinstance(tb, dict):
                c = tb.get("combination") or []
                if len(c) == 3 and tb.get("is_real_odds"):
                    real_trio_map[tuple(sorted(c))] = float(tb.get("odds") or 0.0)

        new_h = compute_hybrid_rec_from_inputs(
            race_name=rname,
            race_details=rdetails,
            horses=horses,
            surface_type=surf,
            prob_gachi=float(old_h.get("prob_gachi") or 0.0),
            prob_himo=float(old_h.get("prob_himo") or 0.0),
            opponents=opponents,
            u0=u0,
            honmei_horse=honmei,
            race_date_str=rdate,
            force_final_odds=bool(old_h.get("is_final_odds", True)),
            real_trio_odds=real_trio_map
        )

        # 三連複対象レースで実オッズが未取得ならnetkeibaから取得して反映
        if new_h.get("trio_bets") and not real_trio_map:
            fetched_trio = fetch_real_odds(rid).get("trio", {})
            if fetched_trio:
                new_h = compute_hybrid_rec_from_inputs(
                    race_name=rname,
                    race_details=rdetails,
                    horses=horses,
                    surface_type=surf,
                    prob_gachi=float(old_h.get("prob_gachi") or 0.0),
                    prob_himo=float(old_h.get("prob_himo") or 0.0),
                    opponents=opponents,
                    u0=u0,
                    honmei_horse=honmei,
                    race_date_str=rdate,
                    force_final_odds=bool(old_h.get("is_final_odds", True)),
                    real_trio_odds=fetched_trio
                )

        pred_data["hybrid_recommendation"] = new_h

        if not new_h.get("is_participate") or surf == 2:
            rec_obj = pred_data.get("recommendation") or {}
            rec_obj["grade"] = "D"
            rec_obj["confidence_label"] = "見送り推奨"
            rec_obj["is_pass"] = True
            rec_obj["tansho_amount"] = 0
            rec_obj["fukusho_amount"] = 0
            rec_obj["total_amount"] = 0
            pred_data["recommendation"] = rec_obj

        if _save_cached_prediction_fn:
            _save_cached_prediction_fn(rid, rdate, pred_data)
        updated_preds += 1

        res_json = res_map.get(rid)
        if res_json:
            st_res = process_settlement_calculation(
                race_id=rid,
                race_date=rdate,
                race_name=rname,
                pred_data=pred_data,
                race_res=res_json,
                from_cache=True
            )
            updated_settlements += 1
            if not st_res.get("is_pass"):
                participating_count += 1
                total_bet_sum += st_res.get("total_bet", 0)
                total_pay_sum += st_res.get("total_payout", 0)
                if st_res.get("is_real_bet_target"):
                    rb_count += 1
                    rb_bet_sum += st_res.get("total_bet", 0)
                    rb_pay_sum += st_res.get("total_payout", 0)
            else:
                total_pass_benefit += (st_res.get("pass_benefit") or 0)

    if _period_venues_cache_ref is not None:
        _period_venues_cache_ref.clear()

    roi = round((total_pay_sum / total_bet_sum) * 100.0, 2) if total_bet_sum > 0 else 0.0
    rb_roi = round((rb_pay_sum / rb_bet_sum) * 100.0, 2) if rb_bet_sum > 0 else 0.0
    return {
        "status": "ok",
        "updated_predictions": updated_preds,
        "updated_settlements": updated_settlements,
        "participating_races": participating_count,
        "total_bet": total_bet_sum,
        "total_payout": total_pay_sum,
        "profit": total_pay_sum - total_bet_sum,
        "recovery_rate": roi,
        "real_bet_races": rb_count,
        "real_bet_bet": rb_bet_sum,
        "real_bet_payout": rb_pay_sum,
        "real_bet_profit": rb_pay_sum - rb_bet_sum,
        "real_bet_recovery_rate": rb_roi,
        "pass_benefit_total": total_pass_benefit
    }

