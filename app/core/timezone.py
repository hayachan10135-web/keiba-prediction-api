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

