# -*- coding: utf-8 -*-
"""
app/betting/factory.py

投票クライアント生成ファクトリ (BettingClientFactory)
"""

from typing import Optional
from app.betting.base_client import BaseBettingClient
from app.betting.ipat_client import IPATClient
from app.betting.umaca_client import UMACAClient


class BettingClientFactory:
    """投票モードに応じたクライアントを生成"""

    @classmethod
    def get_client(cls, mode: Optional[str] = "ipat") -> BaseBettingClient:
        mode_str = (mode or "ipat").strip().lower()
        if mode_str == "umaca":
            return UMACAClient()
        else:
            return IPATClient()

