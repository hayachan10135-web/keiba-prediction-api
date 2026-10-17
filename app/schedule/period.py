# -*- coding: utf-8 -*-
"""
app/schedule/period.py

開催期間（土曜日 〜 次週金曜日 20:00）管理および期間内開催日走査エンジン
"""

import time
import datetime
from typing import Optional, Callable, Dict, Any, List, Set, Tuple

from app.core.database import get_supabase_client
from app.core.timezone import JST, now_jst_naive, today_jst_str
from app.settlement import invalidate_summary_caches
from app.scrapers import fetch_venues_for_date_raw

# 開催期間インメモリキャッシュ
_period_venues_cache: Dict[str, Any] = {}
_period_checked_empty_dates: Set[str] = set()
_latest_post_times: Dict[str, str] = {}  # {race_id: "09:50発走"} (スクレイピングした実際の最新出走時刻)
_last_weight_check_ts: Dict[str, float] = {}  # {race_id: timestamp} (馬体重発表チェックのクールダウン管理)
_period_kaisai_dates_cache: Dict[str, Tuple[float, List[str]]] = {}


def get_current_kaisai_period(now_dt: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """
    1開催期間（土曜日 00:00 〜 次週金曜日 24:00）の境界を算出。
    ※金曜夜〜土曜朝の使用不可待機期間は完全撤廃し、常時24時間利用可能 (is_waiting_period=False)。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    today_d = now_dt.date()
    weekday = today_d.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    # 待機期間・使用不可期間は完全撤廃（常時利用可能）
    is_waiting = False

    days_since_sat = (weekday - 5) % 7
    period_start_d = today_d - datetime.timedelta(days=days_since_sat)
    period_end_d = period_start_d + datetime.timedelta(days=6)

    return {
        "period_start": period_start_d.strftime("%Y-%m-%d"),
        "period_end": period_end_d.strftime("%Y-%m-%d"),
        "period_start_date": period_start_d,
        "period_end_date": period_end_d,
        "today_str": today_d.strftime("%Y-%m-%d"),
        "today_date": today_d,
        "is_waiting_period": is_waiting
    }


def clear_kaisai_period_caches():
    """金曜20:00の開催期間キャッシュ整理（直近10開催期間＝約75日間は完全保持し、それ以前の過去分のみ安全破棄）"""
    global _period_venues_cache, _period_checked_empty_dates, _latest_post_times, _last_weight_check_ts
    _period_venues_cache.clear()
    _period_checked_empty_dates.clear()
    _latest_post_times.clear()
    _last_weight_check_ts.clear()
    supabase = get_supabase_client()
    if not supabase:
        return
    try:
        cutoff_date = (datetime.datetime.now(JST).date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
        supabase.table("keiba_predictions_cache").delete().neq("race_id", "SCHEMA_VERSION").lt("race_date", cutoff_date).execute()
        supabase.table("keiba_race_results_cache").delete().neq("race_id", "0").lt("race_date", cutoff_date).execute()
        supabase.table("keiba_scheduled_races").delete().neq("race_id", "0").lt("race_date", cutoff_date).execute()
        print(f"[Cleanup] 金曜20:00 キャッシュ整理完了: {cutoff_date} 以前の古いキャッシュのみ破棄しました（直近10開催期間分は完全保持）。")
    except Exception as e:
        print(f"[Cleanup] キャッシュ破棄エラー: {e}")
    invalidate_summary_caches()


def discover_period_kaisai_dates(now_dt: Optional[datetime.datetime] = None) -> List[str]:
    """
    現在の1開催期間（土曜日〜次週金曜日20:00）における、本日までのJRA開催日リスト(昇順)を返す。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)

    now_ts = time.time()
    today_k = p_info["today_str"]
    c_dates = _period_kaisai_dates_cache.get(today_k)
    if c_dates and (now_ts - c_dates[0] < 300.0):
        return list(c_dates[1])

    cutoff_d = (now_dt.date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
    past_db_dates = set()
    supabase = get_supabase_client()
    if supabase:
        try:
            s_res = supabase.table("keiba_scheduled_races").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).execute()
            for row in (s_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
            p_res = supabase.table("keiba_predictions_cache").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).neq("race_id", "SCHEMA_VERSION").execute()
            for row in (p_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
            st_res = supabase.table("keiba_race_settlements").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).execute()
            for row in (st_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
        except Exception:
            pass

    start_d = p_info["period_start_date"]
    today_d = p_info["today_date"]

    available_dates = list(past_db_dates)
    cur_d = start_d
    while cur_d <= today_d:
        d_str = cur_d.strftime("%Y-%m-%d")

        if d_str in _period_venues_cache:
            available_dates.append(d_str)
        elif d_str in _period_checked_empty_dates:
            pass
        elif cur_d.weekday() in (5, 6, 0) or d_str in past_db_dates:
            v = fetch_venues_for_date_raw(d_str)
            if v:
                available_dates.append(d_str)
        else:
            _period_checked_empty_dates.add(d_str)

        cur_d += datetime.timedelta(days=1)

    # 金曜夕方（17:00以降）であれば翌日（土曜）の出走表確定状況も先行走査
    if today_d.weekday() == 4 and now_dt.hour >= 17:
        tomorrow_d = today_d + datetime.timedelta(days=1)
        tom_str = tomorrow_d.strftime("%Y-%m-%d")
        if tom_str in _period_venues_cache:
            available_dates.append(tom_str)
        elif tom_str not in _period_checked_empty_dates:
            v = fetch_venues_for_date_raw(tom_str)
            if v:
                available_dates.append(tom_str)

    res_dates = sorted(list(set(available_dates)))[-6:]
    _period_kaisai_dates_cache[today_k] = (now_ts, res_dates)
    return res_dates

