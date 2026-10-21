# -*- coding: utf-8 -*-
"""
app/core/security.py

API 認証・セキュリティ検証モジュール
- X-API-Key ヘッダーによる認証
- タイミング攻撃耐性のある secrets.compare_digest による安全なキー照合
- OpenAPI (Swagger UI) 連携用 APIKeyHeader
"""

import os
import secrets
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings

# Swagger UI / OpenAPI ドキュメント上でも X-API-Key 入力欄を表示するための定義
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_expected_api_key() -> Optional[str]:
    """環境変数または設定から期待される API キーを取得"""
    env_key = os.environ.get("API_KEY") or getattr(settings, "API_KEY", "")
    if env_key and env_key.strip():
        key = env_key.strip()
        if key.lower() not in {"your-secure-api-key-here", "changeme", "change-me"}:
            return key
    return None


def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """
    X-API-Key ヘッダーを検証する FastAPI 依存関係関数。
    不正または欠落している場合は 401 Unauthorized を送出。
    """
    expected_key = get_expected_api_key()

    if expected_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Please provide 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not secrets.compare_digest(api_key.strip(), expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key
