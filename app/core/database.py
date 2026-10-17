# -*- coding: utf-8 -*-
"""
app/core/database.py

Supabase クライアント初期化および接続管理
Fail-Closed 原則に基づき、接続異常時は安全停止を促す
"""

import sys
from typing import Optional
from supabase import create_client, Client
from app.core.config import settings

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Optional[Client]:
    """Supabase クライアントシングルトンを取得"""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if settings.SUPABASE_URL and settings.SUPABASE_KEY:
        try:
            _supabase_client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            print("[Database] Supabase クライアントを正常に初期化しました。")
            return _supabase_client
        except Exception as e:
            print(f"[Database] ❌ Supabase 初期化エラー: {e}", file=sys.stderr)
            return None
    else:
        print("[Database] ⚠️ 警告: SUPABASE_URL または SUPABASE_KEY が環境変数に設定されていません。", file=sys.stderr)
        return None

