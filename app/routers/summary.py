# -*- coding: utf-8 -*-
"""
app/routers/summary.py

日次収支サマリー、累計収支サマリー、開催日一覧 API ルーター
"""

import gc
from typing import List, Optional
from fastapi import APIRouter, Query

from app.core.database import get_supabase_client
from app.core.timezone import today_jst_str
from app.settlement.summary_calculator import (
    DailySummary,
    AggregateSummaryResponse,
    get_summary_dates_list,
    calculate_aggregate_summary
)
from app.settlement import (
    extract_top3_orders,
    extract_mark_orders,
    sync_orders_marks
)

router = APIRouter(tags=["Summary"])

GLOBAL_VENUE_MAP = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"
}

CANCELLED_RACE_NAMES = {
    "202605010308": "8R 4歳以上1勝クラス (積雪中止)",
    "202605010309": "9R 春望S (積雪中止)",
    "202605010310": "10R 雲雀S (積雪中止)",
    "202605010311": "11R 早春S (積雪中止)",
    "202605010312": "12R 4歳以上2勝クラス (積雪中止)",
    "202610010604": "4R 障害4歳以上未勝利 (中止)"
}


@router.get("/summary/dates", response_model=List[str])
@router.get("/daily-summary/dates", response_model=List[str])
def get_summary_dates_endpoint(force_refresh: bool = False):
    """過去の予想・集計可能日付一覧を取得 (降順・キャッシュ付き)"""
    return get_summary_dates_list(force_refresh=force_refresh)


@router.get("/summary/aggregates", response_model=AggregateSummaryResponse)
def get_aggregate_summary_endpoint(force_refresh: bool = False):
    """生涯・年間・月間の総合収支サマリーを取得（キャッシュ付き）"""
    return calculate_aggregate_summary(force_refresh=force_refresh)


@router.get("/summary/today", response_model=DailySummary)
def get_daily_summary_endpoint(date: Optional[str] = Query(None, description="集計対象日付 (YYYY-MM-DD)")):
    """当日（または指定日）の収支・的中一覧・パス効果サマリーを取得"""
    target_date = date if date else today_jst_str()
    supabase = get_supabase_client()

    rows = []
    if supabase:
        try:
            res = supabase.table("keiba_race_settlements").select("*").eq("race_date", target_date).order("race_id", desc=False).execute()
            rows = res.data or []
            # 当日公式開催スケジュール（東京:05, 京都:08, 新潟:04）以外の誤登録データを厳密に除外
            if target_date == "2026-10-18":
                rows = [r for r in rows if str(r.get("race_id") or "")[4:6] in {"05", "08", "04"}]
            else:
                try:
                    sched_res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", target_date).execute()
                    valid_rids = {sr["race_id"] for sr in (sched_res.data or []) if sr.get("race_id")}
                    if valid_rids:
                        rows = [r for r in rows if r.get("race_id") in valid_rids]
                except Exception:
                    pass
        except Exception as e:
            print(f"Supabaseサマリー取得エラー: {e}")

    # 結果キャッシュと予想キャッシュを一括取得 (確定済みの rows に含まれる race_id のみに限定し、OOMと42703エラーを完全根絶)
    res_cache_map = {}
    pred_cache_map = {}
    sched_map = {}
    if supabase and rows:
        settled_rids = [r["race_id"] for r in rows if r.get("race_id")]
        if settled_rids:
            try:
                res_cache = supabase.table("keiba_race_results_cache").select("race_id, result_json").in_("race_id", settled_rids).execute()
                res_cache_map = {item["race_id"]: item.get("result_json", {}) for item in (res_cache.data or [])}
            except Exception as e:
                print(f"Supabaseサマリー用結果キャッシュ取得エラー: {e}")
            try:
                pred_cache = supabase.table("keiba_predictions_cache").select("race_id, data_json").in_("race_id", settled_rids).execute()
                pred_cache_map = {item["race_id"]: item.get("data_json", {}) for item in (pred_cache.data or [])}
            except Exception as e:
                print(f"Supabaseサマリー用予想キャッシュ取得エラー: {e}")

    pass_benefit_total = 0
    pass_avoided_losses = 0
    pass_missed_profits = 0

    for r in rows:
        p_data = pred_cache_map.get(r["race_id"], {})
        h_rec = p_data.get("hybrid_recommendation") or {}
        b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
        rb_meta = b_det.get("real_bet_meta") or {}
        sc_info = sched_map.get(r["race_id"], {})

        # 実戦推奨情報を付与
        is_real = bool(
            h_rec.get("is_real_bet_target")
            or rb_meta.get("is_real_bet_target")
            or sc_info.get("is_real_bet_target")
            or (rb_meta.get("real_bet_level") in ["high", "standard"])
            or (h_rec.get("real_bet_level") in ["high", "standard"])
        )
        r["is_real_bet_target"] = is_real

        badge_val = h_rec.get("real_bet_badge") or rb_meta.get("real_bet_badge") or sc_info.get("real_bet_badge")
        if badge_val and isinstance(badge_val, str):
            badge_val = badge_val.replace("🎯", "⭐")
        r["real_bet_badge"] = badge_val
        if r.get("strategy_name") and isinstance(r["strategy_name"], str):
            r["strategy_name"] = r["strategy_name"].replace("🎯", "⭐")
        r["real_bet_level"] = h_rec.get("real_bet_level") or rb_meta.get("real_bet_level") or sc_info.get("real_bet_level") or "none"
        r["edge_ratio"] = h_rec.get("edge_ratio") if h_rec.get("edge_ratio") is not None else rb_meta.get("edge_ratio")
        r["is_final_odds"] = h_rec.get("is_final_odds", rb_meta.get("is_final_odds", False))

        is_virt = bool(
            sc_info.get("is_virtual_bet_target")
            or h_rec.get("is_virtual_bet_target")
            or rb_meta.get("is_virtual_bet_target")
            or r.get("real_bet_level") == "virtual"
            or (r.get("real_bet_badge") and "仮想" in str(r.get("real_bet_badge")))
            or r.get("strategy_type") == "virtual"
            or (r.get("hit_type") and "仮想" in str(r.get("hit_type")))
        ) and not is_real
        r["is_virtual_bet_target"] = is_virt

        if r.get("strategy_type") == "cancelled" or r.get("hit_type") == "開催中止":
            r["is_real_bet_target"] = False
            r["is_virtual_bet_target"] = False
            r["real_bet_badge"] = "⛔【開催中止】全額返還"
            r["real_bet_level"] = "none"
            r["hypothetical_bet"] = 0
            r["hypothetical_payout"] = 0
            r["pass_benefit"] = 0
            r["mark_orders"] = []
            r["top3_orders"] = []
        elif is_virt:
            v_bet = r.get("total_bet") or r.get("hypothetical_bet") or 200
            v_pay = r.get("total_payout") or r.get("hypothetical_payout") or 0
            pb = r.get("pass_benefit") or 0
            r["hypothetical_bet"] = v_bet
            r["hypothetical_payout"] = v_pay
            r["pass_benefit"] = pb
            r["total_bet"] = v_bet
            r["total_payout"] = v_pay
            r["profit"] = v_pay - v_bet
            r["recovery_rate"] = round((v_pay / v_bet) * 100.0, 1) if v_bet > 0 else 0.0
            r["is_pass"] = 0
            r["is_hit"] = 1 if v_pay > 0 else 0
            r["hit_type"] = "仮想的中" if v_pay > 0 else "仮想不的中"
            r["strategy_type"] = "virtual"
            r["real_bet_level"] = "virtual"
            if not r.get("real_bet_badge"):
                slot_no = sc_info.get("virtual_bet_slot") or 1
                r["real_bet_badge"] = f"💡【仮想馬券】会場No.{slot_no}推奨 (単複)"
            if (r.get("tansho_bet") or 0) == 0:
                r["tansho_bet"] = 100
                r["fukusho_bet"] = 100
            if not r.get("bet_breakdown"):
                r["bet_breakdown"] = {"単勝": 100, "複勝": 100}
            p_bd = r.get("payout_breakdown") if isinstance(r.get("payout_breakdown"), dict) else {}
            if (r.get("tansho_payout") or 0) > 0: p_bd["単勝"] = r["tansho_payout"]
            if (r.get("fukusho_payout") or 0) > 0: p_bd["複勝"] = r["fukusho_payout"]
            r["payout_breakdown"] = p_bd
        elif r.get("is_pass") == 1:
            pb = r.get("pass_benefit") or 0
            r["hypothetical_bet"] = r.get("hypothetical_bet") or 0
            r["hypothetical_payout"] = r.get("hypothetical_payout") or 0
            r["pass_benefit"] = pb
            pass_benefit_total += pb
            if pb > 0:
                pass_avoided_losses += pb
            elif pb < 0:
                pass_missed_profits += abs(pb)

        # mark_orders / top3_orders の完全自己修復・出走表印同期
        raw_mo = r.get("mark_orders") or rb_meta.get("mark_orders") or b_det.get("mark_orders")
        raw_t3 = r.get("top3_orders") or rb_meta.get("top3_orders") or b_det.get("top3_orders")

        r_orders = res_cache_map.get(r["race_id"], {}).get("orders") or {}
        if p_data and r_orders:
            t3 = extract_top3_orders(p_data, r_orders)
            mo = extract_mark_orders(p_data, r_orders)
        else:
            t3 = raw_t3 if isinstance(raw_t3, list) else []
            mo = raw_mo if isinstance(raw_mo, list) else []
            if p_data:
                t3, mo = sync_orders_marks(t3, mo, p_data)

        r["mark_orders"] = mo if isinstance(mo, list) else []
        r["top3_orders"] = t3 if isinstance(t3, list) else []

        # 本命馬の人気(target_popularity)が未設定の場合、出走表から補完
        if r.get("target_popularity") is None and p_data and r.get("target_umaban") is not None:
            t_u = r.get("target_umaban")
            h_match = next((h for h in p_data.get("horses", []) if h.get("umaban") == t_u), None)
            if h_match and h_match.get("popularity"):
                r["target_popularity"] = h_match.get("popularity")

        if not isinstance(r.get("bet_breakdown"), dict):
            b_dict = {}
            if (r.get("umaren_bet") or 0) > 0: b_dict["馬連"] = r["umaren_bet"]
            if (r.get("trio_bet") or 0) > 0: b_dict["三連複"] = r["trio_bet"]
            r["bet_breakdown"] = b_dict

        if not isinstance(r.get("payout_breakdown"), dict):
            p_dict = {}
            if (r.get("umaren_payout") or 0) > 0: p_dict["馬連"] = r["umaren_payout"]
            if (r.get("trio_payout") or 0) > 0: p_dict["三連複"] = r["trio_payout"]
            r["payout_breakdown"] = p_dict

        r_id_str = str(r.get("race_id", ""))
        if len(r_id_str) >= 12 and not r.get("venue_name"):
            try:
                v_code = int(r_id_str[4:6])
                r["venue_name"] = GLOBAL_VENUE_MAP.get(v_code, "JRA")
            except Exception:
                pass
        if len(r_id_str) >= 12 and not r.get("race_num"):
            try:
                r_num_int = int(r_id_str[10:12])
                r["race_num"] = f"{r_num_int}R"
            except Exception:
                pass

        if r_id_str in CANCELLED_RACE_NAMES:
            r["race_name"] = CANCELLED_RACE_NAMES[r_id_str]

        r_bet = r.get("total_bet") or 0
        r_payout = r.get("total_payout") or 0
        if r.get("strategy_type") == "cancelled" or r.get("hit_type") == "開催中止":
            r["total_bet"] = 0
            r["total_payout"] = 0
            r["profit"] = 0
            r["recovery_rate"] = 0.0
            r["is_pass"] = 1
            r["is_hit"] = 0
        else:
            r["profit"] = r_payout - r_bet
            r["recovery_rate"] = round((r_payout / r_bet) * 100.0, 1) if r_bet > 0 else 0.0

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r.get("is_pass") == 0)
    pass_races = sum(1 for r in rows if r.get("is_pass") == 1)
    hit_races = sum(1 for r in rows if r.get("is_hit") == 1 and r.get("is_pass") == 0)

    total_bet = sum((r.get("total_bet") or 0) for r in rows)
    total_payout = sum((r.get("total_payout") or 0) for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

    # 実戦推奨レースのみの収支集計
    real_bet_rows = [r for r in rows if r.get("is_real_bet_target") and r.get("is_pass") == 0]
    real_bet_races = len(real_bet_rows)
    real_bet_hits = sum(1 for r in real_bet_rows if r.get("is_hit") == 1)
    real_bet_hit_rate = round((real_bet_hits / real_bet_races) * 100.0, 2) if real_bet_races > 0 else 0.0
    real_bet_amount = sum((r.get("total_bet") or 0) for r in real_bet_rows)
    real_bet_payout = sum((r.get("total_payout") or 0) for r in real_bet_rows)
    real_bet_profit = real_bet_payout - real_bet_amount
    real_bet_recovery_rate = round((real_bet_payout / real_bet_amount) * 100.0, 2) if real_bet_amount > 0 else 0.0

    # 汎用券種別内訳集計
    ticket_stats = {}
    for r in rows:
        if r.get("is_pass") == 0:
            b_map = r.get("bet_breakdown") or {}
            p_map = r.get("payout_breakdown") or {}
            for t_type, b_amt in b_map.items():
                if b_amt > 0:
                    st = ticket_stats.setdefault(t_type, {
                        "ticket_type": t_type,
                        "bet_amount": 0,
                        "payout_amount": 0,
                        "hit_count": 0,
                        "bet_count": 0,
                        "recovery_rate": 0.0
                    })
                    st["bet_amount"] += b_amt
                    st["bet_count"] += 1
                    p_amt = p_map.get(t_type, 0)
                    st["payout_amount"] += p_amt
                    if p_amt > 0:
                        st["hit_count"] += 1

    ticket_breakdowns = []
    for t_type, st in ticket_stats.items():
        st["recovery_rate"] = round((st["payout_amount"] / st["bet_amount"]) * 100.0, 2) if st["bet_amount"] > 0 else 0.0
        ticket_breakdowns.append(st)

    # 実戦推奨用の券種別内訳集計
    real_ticket_stats = {}
    for r in real_bet_rows:
        b_map = r.get("bet_breakdown") or {}
        p_map = r.get("payout_breakdown") or {}
        for t_type, b_amt in b_map.items():
            if b_amt > 0:
                st = real_ticket_stats.setdefault(t_type, {
                    "ticket_type": t_type,
                    "bet_amount": 0,
                    "payout_amount": 0,
                    "hit_count": 0,
                    "bet_count": 0,
                    "recovery_rate": 0.0
                })
                st["bet_amount"] += b_amt
                st["bet_count"] += 1
                p_amt = p_map.get(t_type, 0)
                st["payout_amount"] += p_amt
                if p_amt > 0:
                    st["hit_count"] += 1

    real_ticket_breakdowns = []
    for t_type, st in real_ticket_stats.items():
        st["recovery_rate"] = round((st["payout_amount"] / st["bet_amount"]) * 100.0, 2) if st["bet_amount"] > 0 else 0.0
        real_ticket_breakdowns.append(st)

    # 実戦推奨視点での見送り損益集計
    real_pass_rows = [r for r in rows if not (r.get("is_real_bet_target") and r.get("is_pass") == 0)]
    real_pass_races = len(real_pass_rows)
    real_pass_avoided_losses = 0
    real_pass_missed_profits = 0
    real_pass_benefit_total = 0

    for r in real_pass_rows:
        if r.get("is_pass") == 0:
            r_profit = (r.get("total_payout") or 0) - (r.get("total_bet") or 0)
            if r_profit < 0:
                loss_avoided = abs(r_profit)
                real_pass_avoided_losses += loss_avoided
                real_pass_benefit_total += loss_avoided
            elif r_profit > 0:
                real_pass_missed_profits += r_profit
                real_pass_benefit_total -= r_profit
        else:
            pb = r.get("pass_benefit")
            if pb is not None:
                real_pass_benefit_total += pb
                if pb > 0:
                    real_pass_avoided_losses += pb
                elif pb < 0:
                    real_pass_missed_profits += abs(pb)

    umaren_hit_races = sum(1 for r in rows if (r.get("umaren_payout") or 0) > 0 and r.get("is_pass") == 0)
    trio_hit_races = sum(1 for r in rows if (r.get("trio_payout") or 0) > 0 and r.get("is_pass") == 0)
    umaren_bet_total = sum((r.get("umaren_bet") or 0) for r in rows if r.get("is_pass") == 0)
    trio_bet_total = sum((r.get("trio_bet") or 0) for r in rows if r.get("is_pass") == 0)
    umaren_payout_total = sum((r.get("umaren_payout") or 0) for r in rows if r.get("is_pass") == 0)
    trio_payout_total = sum((r.get("trio_payout") or 0) for r in rows if r.get("is_pass") == 0)
    umaren_rec = round((umaren_payout_total / umaren_bet_total) * 100.0, 2) if umaren_bet_total > 0 else 0.0
    trio_rec = round((trio_payout_total / trio_bet_total) * 100.0, 2) if trio_bet_total > 0 else 0.0
    del res_cache_map
    del pred_cache_map
    del sched_map
    gc.collect()

    return DailySummary(
        target_date=target_date,
        total_races=total_races,
        bet_races=bet_races,
        pass_races=pass_races,
        hit_races=hit_races,
        hit_rate=hit_rate,
        total_bet_amount=total_bet,
        total_payout_amount=total_payout,
        total_profit=total_profit,
        recovery_rate=recovery_rate,
        settled_races=rows,
        pass_benefit_total=pass_benefit_total,
        pass_avoided_losses=pass_avoided_losses,
        pass_missed_profits=pass_missed_profits,
        ticket_breakdowns=ticket_breakdowns,
        real_ticket_breakdowns=real_ticket_breakdowns,
        real_pass_races=real_pass_races,
        real_pass_benefit_total=real_pass_benefit_total,
        real_pass_avoided_losses=real_pass_avoided_losses,
        real_pass_missed_profits=real_pass_missed_profits,
        umaren_hit_races=umaren_hit_races,
        trio_hit_races=trio_hit_races,
        umaren_bet_total=umaren_bet_total,
        trio_bet_total=trio_bet_total,
        umaren_payout_total=umaren_payout_total,
        trio_payout_total=trio_payout_total,
        umaren_recovery_rate=umaren_rec,
        trio_recovery_rate=trio_rec,
        real_bet_races=real_bet_races,
        real_bet_hits=real_bet_hits,
        real_bet_hit_rate=real_bet_hit_rate,
        real_bet_amount=real_bet_amount,
        real_bet_payout=real_bet_payout,
        real_bet_profit=real_bet_profit,
        real_bet_recovery_rate=real_bet_recovery_rate,
        cumulative_days=0,
        cumulative_total_races=0,
        cumulative_bet_races=0,
        cumulative_pass_races=0,
        cumulative_hit_races=0,
        cumulative_hit_rate=0.0,
        cumulative_bet_amount=0,
        cumulative_payout_amount=0,
        cumulative_profit=0,
        cumulative_recovery_rate=0.0,
        cumulative_pass_benefit=0,
        cumulative_real_bet_races=0,
        cumulative_real_bet_hits=0,
        cumulative_real_bet_hit_rate=0.0,
        cumulative_real_bet_amount=0,
        cumulative_real_bet_payout=0,
        cumulative_real_bet_profit=0,
        cumulative_real_bet_recovery_rate=0.0
    )

