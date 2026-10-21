# -*- coding: utf-8 -*-
"""
tests/test_api_auth.py

API 認証 (FR-SEC-01 / X-API-Key) セキュリティ検証テスト:
- 外部公開エンドポイント保護
- キーなし / 不正キーでの 401 Unauthorized 遮断
- 正規キーでの 200 OK 疎通
- 公開ヘルスチェック (/health) の無認証アクセス
- 削除済みエンドポイントの 404 Not Found 確認
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app
from app.core import security
from unittest.mock import patch


class TestApiAuth(unittest.TestCase):
    """API 認証セキュリティのテストケース"""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_key = "test-api-key-for-auth-tests"
        cls.invalid_key = "invalid-token-12345"

    def setUp(self):
        self.settings_api_key = security.settings.API_KEY
        security.settings.API_KEY = self.valid_key
        self.api_key_env = patch.dict(os.environ, {"API_KEY": self.valid_key})
        self.api_key_env.start()

    def tearDown(self):
        self.api_key_env.stop()
        security.settings.API_KEY = self.settings_api_key

    def test_health_check_public(self):
        """/health は監視サービス用に認証なしで 200 OK が返却されること"""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")

    def test_protected_endpoint_without_key(self):
        """認証ヘッダーなしのリクエストが 401 Unauthorized で遮断されること"""
        resp = self.client.get("/settings/betting-mode")
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Missing API Key", resp.json().get("detail", ""))

    def test_protected_endpoint_with_invalid_key(self):
        """不正な API キーによるリクエストが 401 Unauthorized で遮断されること"""
        resp = self.client.get("/settings/betting-mode", headers={"X-API-Key": self.invalid_key})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("Invalid API Key", resp.json().get("detail", ""))

    def test_protected_endpoint_with_valid_key(self):
        """正しい API キーによるリクエストが 200 OK で正常に認可されること"""
        resp = self.client.get("/settings/betting-mode", headers={"X-API-Key": self.valid_key})
        self.assertEqual(resp.status_code, 200)

    def test_protected_endpoint_fails_closed_without_configured_key(self):
        """API_KEY 未設定時に固定キー等で認証を通さないこと"""
        security.settings.API_KEY = ""
        with patch.dict(os.environ, {"API_KEY": ""}):
            resp = self.client.get(
                "/settings/betting-mode",
                headers={"X-API-Key": self.valid_key},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json().get("detail"), "API authentication is not configured.")

    def test_deleted_admin_endpoints_return_404(self):
        """削除された一回限り管理エンドポイントが 404 Not Found となること"""
        headers = {"X-API-Key": self.valid_key}
        self.assertEqual(self.client.post("/admin/fix-20261011", headers=headers).status_code, 404)
        self.assertEqual(self.client.post("/admin/process-1003-1004", headers=headers).status_code, 404)
        self.assertEqual(self.client.post("/admin/process-0926-0927", headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
