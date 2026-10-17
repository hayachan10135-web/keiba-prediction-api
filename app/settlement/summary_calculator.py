# -*- coding: utf-8 -*-
"""
app/settlement/summary_calculator.py

日次収支サマリー、期間集計（生涯・年間・月間）、券種別内訳集計エンジン
"""

import time
from collections import defaultdict
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.core.database import get_supabase_client
from app.core.timezone import today_jst_str

# --- Pydantic レスポンスモデル ---

class TicketTypeSummary(BaseModel):
    ticket_type: str
    bet_amount: int
    payout_amount: int
    hit_count: int
    bet_count: int
    recovery_rate: float


class DailySummary(BaseModel):
    target_date: str
    total_races: int
    bet_races: int
    pass_races: int
    hit_races: int
    hit_rate: float
    total_bet_amount: int
    total_payout_amount: int
    total_profit: int
    recovery_rate: float
    settled_races: List[dict]
    pass_benefit_total: Optional[int] = 0
    pass_avoided_losses: Optional[int] = 0
    pass_missed_profits: Optional[int] = 0
    ticket_breakdowns: Optional[List[TicketTypeSummary]] = []
    real_ticket_breakdowns: Optional[List[TicketTypeSummary]] = []
    real_pass_races: Optional[int] = 0
    real_pass_benefit_total: Optional[int] = 0
    real_pass_avoided_losses: Optional[int] = 0
    real_pass_missed_profits: Optional[int] = 0
    umaren_hit_races: Optional[int] = 0
    trio_hit_races: Optional[int] = 0
    umaren_bet_total: Optional[int] = 0
    trio_bet_total: Optional[int] = 0
    umaren_payout_total: Optional[int] = 0
    trio_payout_total: Optional[int] = 0
    umaren_recovery_rate: Optional[float] = 0.0
    trio_recovery_rate: Optional[float] = 0.0
    real_bet_races: Optional[int] = 0
    real_bet_hits: Optional[int] = 0
    real_bet_hit_rate: Optional[float] = 0.0
    real_bet_amount: Optional[int] = 0
    real_bet_payout: Optional[int] = 0
    real_bet_profit: Optional[int] = 0
    real_bet_recovery_rate: Optional[float] = 0.0
    cumulative_days: Optional[int] = 0
    cumulative_total_races: Optional[int] = 0
    cumulative_bet_races: Optional[int] = 0
    cumulative_pass_races: Optional[int] = 0
    cumulative_hit_races: Optional[int] = 0
    cumulative_hit_rate: Optional[float] = 0.0
    cumulative_bet_amount: Optional[int] = 0
    cumulative_payout_amount: Optional[int] = 0
    cumulative_profit: Optional[int] = 0
    cumulative_recovery_rate: Optional[float] = 0.0
    cumulative_pass_benefit: Optional[int] = 0
    cumulative_real_bet_races: Optional[int] = 0
    cumulative_real_bet_hits: Optional[int] = 0
    cumulative_real_bet_hit_rate: Optional[float] = 0.0
    cumulative_real_bet_amount: Optional[int] = 0
    cumulative_real_bet_payout: Optional[int] = 0
    cumulative_real_bet_profit: Optional[int] = 0
    cumulative_real_bet_recovery_rate: Optional[float] = 0.0


class PeriodStatItem(BaseModel):
    title: str
    period_key: str
    days: int = 0
    total_races: int = 0
    bet_races: int = 0
    pass_races: int = 0
    hit_races: int = 0
    hit_rate: float = 0.0
    total_bet: int = 0
    total_payout: int = 0
    total_profit: int = 0
    recovery_rate: float = 0.0
    real_bet_races: int = 0
    real_bet_hits: int = 0
    real_bet_hit_rate: float = 0.0
    real_bet_amount: int = 0
    real_bet_payout: int = 0
    real_bet_profit: int = 0
    real_bet_recovery_rate: float = 0.0
    pass_benefit: int = 0


class AggregateSummaryResponse(BaseModel):
    lifetime: PeriodStatItem
    yearly: List[PeriodStatItem] = []
    monthly: List[PeriodStatItem] = []


# --- メモリキャッシュ管理 ---
_summary_dates_cache: Dict[str, Any] = {
    "dates": None,
    "updated_at": 0.0
}
_aggregate_summary_cache: Dict[str, Any] = {
    "data": None,
    "updated_at": 0.0
}

CACHE_TTL_SUMMARY_DATES = 600.0  # 10分
CACHE_TTL_AGGREGATE = 600.0      # 10分


def invalidate_summary_caches():
    """新レース確定や一括同期時にサマリーキャッシュを破棄"""
    global _summary_dates_cache, _aggregate_summary_cache
    _summary_dates_cache["updated_at"] = 0.0
    _aggregate_summary_cache["updated_at"] = 0.0


def calc_period_stat(title: str, key: str, rows: list) -> dict:
    """期間ごとの収支・回収率・実戦推奨成績を集計"""
    days = len(set(r.get("race_date") for r in rows if r.get("race_date")))
    total_races = len(rows)
    bet_races = sum(1 for r in rows if r.get("is_pass") == 0)
    pass_races = sum(1 for r in rows if r.get("is_pass") == 1)
    hit_races = sum(1 for r in rows if r.get("is_hit") == 1 and r.get("is_pass") == 0)
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0
    total_bet = sum((r.get("total_bet") or 0) for r in rows)
    total_payout = sum((r.get("total_payout") or 0) for r in rows)
    total_profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    pass_benefit = sum((r.get("pass_benefit") or 0) for r in rows if r.get("is_pass") == 1)

    rb_races = 0
    rb_hits = 0
    rb_amount = 0
    rb_payout = 0
    for r in rows:
        if r.get("is_pass") == 0:
            bd = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
            rbm = bd.get("real_bet_meta") or {}
            if rbm.get("is_real_bet_target"):
                rb_races += 1
                if r.get("is_hit") == 1:
                    rb_hits += 1
                rb_amount += (r.get("total_bet") or 0)
                rb_payout += (r.get("total_payout") or 0)
    rb_profit = rb_payout - rb_amount
    rb_recovery = round((rb_payout / rb_amount) * 100.0, 2) if rb_amount > 0 else 0.0
    rb_hit_rate = round((rb_hits / rb_races) * 100.0, 2) if rb_races > 0 else 0.0

    return {
        "title": title,
        "period_key": key,
        "days": days,
        "total_races": total_races,
        "bet_races": bet_races,
        "pass_races": pass_races,
        "hit_races": hit_races,
        "hit_rate": hit_rate,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "total_profit": total_profit,
        "recovery_rate": recovery_rate,
        "real_bet_races": rb_races,
        "real_bet_hits": rb_hits,
        "real_bet_hit_rate": rb_hit_rate,
        "real_bet_amount": rb_amount,
        "real_bet_payout": rb_payout,
        "real_bet_profit": rb_profit,
        "real_bet_recovery_rate": rb_recovery,
        "pass_benefit": pass_benefit
    }


def calculate_aggregate_summary(force_refresh: bool = False) -> dict:
    """生涯・年間・月間の総合収支サマリーを取得（キャッシュ付き）"""
    global _aggregate_summary_cache
    now = time.time()
    if not force_refresh and _aggregate_summary_cache["data"] and (now - _aggregate_summary_cache["updated_at"] < CACHE_TTL_AGGREGATE):
        return _aggregate_summary_cache["data"]

    supabase = get_supabase_client()
    cum_rows = []
    if supabase:
        try:
            page = 0
            page_size = 1000
            while True:
                cum_res = supabase.table("keiba_race_settlements").select(
                    "race_date, is_pass, is_hit, total_bet, total_payout, pass_benefit, bet_details"
                ).range(page * page_size, (page + 1) * page_size - 1).execute()
                batch = cum_res.data or []
                cum_rows.extend(batch)
                if len(batch) < page_size:
                    break
                page += 1
        except Exception as e:
            print(f"Supabase総合集計データ取得エラー: {e}")

    # 1. 生涯（全期間）
    lifetime = calc_period_stat("生涯（全期間）", "lifetime", cum_rows)

    # 2. 年別 & 月別グループ分け
    year_map = defaultdict(list)
    month_map = defaultdict(list)
    for r in cum_rows:
        rd = r.get("race_date") or ""
        if len(rd) >= 4:
            year_map[rd[:4]].append(r)
        if len(rd) >= 7:
            month_map[rd[:7]].append(r)

    yearly = []
    for y in sorted(year_map.keys(), reverse=True):
        yearly.append(calc_period_stat(f"{y}年", y, year_map[y]))

    monthly = []
    for ym in sorted(month_map.keys(), reverse=True):
        y_part = ym[:4]
        m_part = int(ym[5:7]) if ym[5:7].isdigit() else ym[5:7]
        monthly.append(calc_period_stat(f"{y_part}年{m_part}月", ym, month_map[ym]))

    result = {
        "lifetime": lifetime,
        "yearly": yearly,
        "monthly": monthly
    }
    _aggregate_summary_cache["data"] = result
    _aggregate_summary_cache["updated_at"] = now
    return result


def get_summary_dates_list(force_refresh: bool = False) -> List[str]:
    """過去の予想・集計可能日付一覧を取得 (降順・キャッシュ付き)"""
    global _summary_dates_cache
    now = time.time()
    if not force_refresh and _summary_dates_cache["dates"] and (now - _summary_dates_cache["updated_at"] < CACHE_TTL_SUMMARY_DATES):
        return _summary_dates_cache["dates"]

    today_str = today_jst_str()
    supabase = get_supabase_client()
    try:
        dates_set = set()
        if supabase:
            # 1. 確定・集計済みレコードがある日付
            try:
                page = 0
                while True:
                    res_settle = supabase.table("keiba_race_settlements").select("race_date").range(page * 1000, (page + 1) * 1000 - 1).execute()
                    batch = res_settle.data or []
                    for row in batch:
                        d = row.get("race_date")
                        if d and len(d) == 10 and d.startswith("20"):
                            dates_set.add(d)
                    if len(batch) < 1000:
                        break
                    page += 1
            except Exception as e:
                print(f"settlements dates error: {e}")

            # 2. レース開催日として登録された日付
            try:
                page = 0
                while True:
                    res_sched = supabase.table("keiba_scheduled_races").select("race_date").range(page * 1000, (page + 1) * 1000 - 1).execute()
                    batch = res_sched.data or []
                    for row in batch:
                        d = row.get("race_date")
                        if d and len(d) == 10 and d.startswith("20"):
                            dates_set.add(d)
                    if len(batch) < 1000:
                        break
                    page += 1
            except Exception as e:
                print(f"scheduled dates error: {e}")

        sorted_dates = sorted(list(dates_set), reverse=True)
        res = sorted_dates if sorted_dates else [today_str]
        _summary_dates_cache["dates"] = res
        _summary_dates_cache["updated_at"] = now
        return res
    except Exception as e:
        print(f"日付一覧取得エラー: {e}")
        return [today_str]

