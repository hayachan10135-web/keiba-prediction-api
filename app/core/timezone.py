# -*- coding: utf-8 -*-
"""
app/core/timezone.py

日本時間 (JST) タイムゾーン処理および発走時刻パースユーティリティ
"""

import datetime
import zoneinfo
import re
from typing import Optional

JST = zoneinfo.ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime.datetime:
    """日本時間の現在時刻を取得 (tz-aware)"""
    return datetime.datetime.now(JST)


def now_jst_naive() -> datetime.datetime:
    """比較用のタイムゾーンなしJST現在日時を取得 (naive)"""
    return now_jst().replace(tzinfo=None)


def today_jst_str() -> str:
    """日本時間の当日日付文字列 (YYYY-MM-DD)"""
    return now_jst().strftime("%Y-%m-%d")


def parse_post_datetime(p_dt_str: Optional[str], fallback_date_str: Optional[str] = None) -> Optional[datetime.datetime]:
    """
    Supabase等から返る発走日時文字列 ('YYYY-MM-DD HH:MM:SS', ISO形式 'T'/'Z'/'+09:00', 'HH:MM') を安全にnaive JST datetimeへ変換
    """
    if not p_dt_str or not isinstance(p_dt_str, str):
        return None
    s = p_dt_str.strip()
    m_full = re.search(r"(\d{4})-(\d{2})-(\d{2})[T\s]+(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m_full:
        try:
            return datetime.datetime(
                int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3)),
                int(m_full.group(4)), int(m_full.group(5)), int(m_full.group(6) or 0)
            )
        except Exception:
            pass
    if fallback_date_str:
        m_hm = re.search(r"(\d{1,2}):(\d{2})", s)
        m_d = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", fallback_date_str.strip())
        if m_hm and m_d:
            try:
                return datetime.datetime(
                    int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3)),
                    int(m_hm.group(1)), int(m_hm.group(2)), 0
                )
            except Exception:
                pass
    return None


def is_before_morning_prediction(target_date: Optional[str] = None) -> bool:
    """
    朝08:30の一括事前予想前であるかを判定。
    対象日付が当日以降（当日または未来日）かつ、現在時刻が朝08:30前であれば True を返す。
    ※ 過去日（昨日以前）の場合は既にレース確定済みのため常に False を返す。
    """
    now_dt = now_jst_naive()
    today_s = today_jst_str()
    t_date = target_date if target_date else today_s
    clean_t = t_date.replace("-", "")
    clean_today = today_s.replace("-", "")

    if clean_t < clean_today:
        return False
    if clean_t > clean_today:
        return True
    # 当日の場合: 朝08:30前かどうか (08:30以降であれば False)
    return (now_dt.hour < 8 or (now_dt.hour == 8 and now_dt.minute < 30))

