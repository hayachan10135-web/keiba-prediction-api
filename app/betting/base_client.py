# -*- coding: utf-8 -*-
"""
app/betting/base_client.py

投票クライアント抽象基底クラス (BaseBettingClient)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseBettingClient(ABC):
    """JRA投票クライアント共通インターフェース"""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """プラットフォーム名 ('ipat' または 'umaca')"""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """認証設定（環境変数）が揃っているか"""
        pass

    @abstractmethod
    def login(self) -> Dict[str, Any]:
        """
        ログイン認証および購入限度額（残高）の取得
        戻り値: {"success": bool, "balance": int, "error": Optional[str], "message": Optional[str]}
        """
        pass

    @abstractmethod
    def execute_bets(
        self,
        venue_name: str,
        race_no: int,
        items: List[Dict[str, Any]],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        買い目を一括発注
        items: [{"ticket_type": "単勝", "combination": [7], "amount": 1000}, ...]
        戻り値: {"status": "success"|"failed"|"aborted"|"skipped", "receipt_no": str, ...}
        """
        pass

