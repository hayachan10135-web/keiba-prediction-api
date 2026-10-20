# -*- coding: utf-8 -*-
"""
backend-render/tests/test_config_integrity.py

環境設定・.env.example 整合性・FastAPI 構成の健全性自動検証テスト
"""

import os
import unittest
from app.core.config import Settings, settings
from main import app


class TestConfigIntegrity(unittest.TestCase):
    """設定ファイルおよびアプリケーション構成の検証"""

    def setUp(self):
        self.backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_example_path = os.path.join(self.backend_dir, ".env.example")

    def test_env_example_is_valid_utf8(self):
        """.env.example が UTF-8 で正常に読み込めること（文字コード破損防止）"""
        self.assertTrue(os.path.exists(self.env_example_path), ".env.example が存在すること")
        with open(self.env_example_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("API_KEY", content)
        self.assertIn("USE_LOCAL_SQLITE", content)
        self.assertIn("DAILY_INVESTMENT_LIMIT", content)

    def test_env_example_contains_all_critical_keys(self):
        """.env.example に本番および自動投票で必須の全キーが網羅されていること"""
        required_keys = [
            "ENV",
            "DEBUG",
            "PORT",
            "USE_LOCAL_SQLITE",
            "DEFAULT_BETTING_MODE",
            "DAILY_INVESTMENT_LIMIT",
            "API_KEY",
            "IPAT_USER_NUM",
            "IPAT_PIN",
            "IPAT_PARS",
            "UMACA_CARD_NUMBER",
            "UMACA_PIN",
            "UMACA_BIRTHDAY",
        ]
        with open(self.env_example_path, "r", encoding="utf-8") as f:
            content = f.read()

        for key in required_keys:
            self.assertTrue(
                any(line.startswith(key + "=") or line.startswith(f"#{key}=") for line in content.splitlines() if line.strip()),
                f"キー '{key}' が .env.example に定義されていること"
            )

    def test_settings_daily_limit_positive(self):
        """1日投資上限金額および1レース最大発注額が正の整数値として定義されていること"""
        from app.betting.settings import _betting_settings
        self.assertIsInstance(_betting_settings["daily_investment_limit"], int)
        self.assertGreater(_betting_settings["daily_investment_limit"], 0)
        self.assertIsInstance(settings.DEFAULT_MAX_BET_PER_RACE, int)
        self.assertGreater(settings.DEFAULT_MAX_BET_PER_RACE, 0)
        self.assertLessEqual(settings.DEFAULT_MAX_BET_PER_RACE, 5000)

    def test_fastapi_lifespan_is_configured(self):
        """FastAPI アプリケーションに最新の lifespan ハンドラーが設定されていること"""
        self.assertIsNotNone(app.router.lifespan_context)
