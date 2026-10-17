# -*- coding: utf-8 -*-
"""
app/betting/settings.py

自動投票設定（即PAT / UMACAスマート / OFF）および残高・予定額の永続化管理
"""

import os
import datetime
from typing import Dict, Any, Optional, Tuple, List
from app.core.database import get_supabase_client
from app.core.timezone import now_jst, now_jst_naive, today_jst_str, parse_post_datetime

_betting_settings: Dict[str, Any] = {
    "betting_mode": os.getenv("DEFAULT_BETTING_MODE", "off").lower().strip(),
    "auto_bet_real_only": True,
    "last_balance": None,
    "balance_status": "unavailable",
    "balance_message": None,
    "updated_at": None
}


def load_betting_settings_from_supabase() -> dict:
    """Supabase keiba_app_meta から投票モード設定を取得"""
    global _betting_settings
    supabase = get_supabase_client()
    if not supabase:
        return _betting_settings
    try:
        res = supabase.table("keiba_app_meta").select("value").eq("key", "betting_settings").execute()
        if res.data and len(res.data) > 0 and res.data[0].get("value"):
            val = res.data[0]["value"]
            if isinstance(val, dict):
                _betting_settings.update(val)
    except Exception as e:
        print(f"[BettingSettings] ロードエラー: {e}")
    return _betting_settings


def save_betting_settings_to_supabase(new_settings: dict) -> bool:
    """Supabase keiba_app_meta に投票モード設定を永続化"""
    global _betting_settings
    _betting_settings.update(new_settings)
    _betting_settings["updated_at"] = f"{today_jst_str()} {now_jst().strftime('%H:%M:%S')}"
    supabase = get_supabase_client()
    if not supabase:
        return True
    try:
        supabase.table("keiba_app_meta").upsert({
            "key": "betting_settings",
            "value": _betting_settings
        }).execute()
        return True
    except Exception as e:
        print(f"[BettingSettings] 保存エラー: {e}")
        return False


def calculate_remaining_real_bet_amount(target_date: Optional[str] = None) -> Tuple[int, int, List[dict]]:
    """
    指定日（または本日/次回開催日）の未発走・未確定レースのうち、
    実馬券・勝負馬券に該当するレースの残り投票予定合計金額を計算する。
    戻り値: (残り合計金額, 該当レース数, 該当レース詳細リスト)
    """
    supabase = get_supabase_client()
    if not supabase:
        return 0, 0, []

    t_date = target_date or today_jst_str()
    now_d = now_jst_naive()

    try:
        s_res = supabase.table("keiba_scheduled_races")\
            .select("race_id, race_date, race_name, post_datetime, is_predicted")\
            .gte("race_date", t_date)\
            .order("race_date")\
            .order("race_id")\
            .execute()

        rows = s_res.data or []
        if not rows:
            return 0, 0, []

        active_date = rows[0]["race_date"]
        active_races = [r for r in rows if r["race_date"] == active_date]

        unrun_races = []
        for r in active_races:
            p_str = r.get("post_datetime")
            is_unrun = True
            p_dt = parse_post_datetime(p_str, active_date)
            if p_dt and now_d > p_dt + datetime.timedelta(minutes=10):
                is_unrun = False
            if is_unrun:
                unrun_races.append(r)

        if not unrun_races:
            return 0, 0, []

        unrun_rids = [r["race_id"] for r in unrun_races]

        p_res = supabase.table("keiba_predictions_cache")\
            .select("race_id, data_json")\
            .in_("race_id", unrun_rids)\
            .execute()

        total_remaining_amount = 0
        real_target_races = []

        for p_row in (p_res.data or []):
            dj = p_row.get("data_json") or {}
            h_rec = dj.get("hybrid_recommendation") or {}
            if h_rec.get("is_real_bet_target"):
                amt = int(h_rec.get("total_amount") or 0)
                if amt > 0:
                    total_remaining_amount += amt
                    real_target_races.append({
                        "race_id": p_row["race_id"],
                        "race_name": dj.get("race_name", p_row["race_id"]),
                        "amount": amt,
                        "badge": h_rec.get("real_bet_badge", "実戦推奨")
                    })

        return total_remaining_amount, len(real_target_races), real_target_races
    except Exception as e:
        print(f"[RemainingBetAmount] 計算エラー: {e}")
        return 0, 0, []
