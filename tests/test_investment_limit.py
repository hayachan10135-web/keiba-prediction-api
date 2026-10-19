# -*- coding: utf-8 -*-
"""
tests/test_investment_limit.py

資金安全原則 (AGENT_RULES 第1章 / FR-BET-05 / FR-BET-06) 検証テスト:
- 1日最大投資上限額 (daily_investment_limit) ガードの判定
- 累計投資額 + 追加発注額が上限を超える場合の安全スキップ判定
- 1レース最大上限額 (DEFAULT_MAX_BET_PER_RACE) の制限
"""

import unittest
from unittest.mock import patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.betting.settings import check_investment_limit, _betting_settings
from app.core.config import settings


class TestInvestmentLimit(unittest.TestCase):
    """投資上限ガード機能の厳格検証テスト"""

    def setUp(self):
        # テスト用に日次上限を 30,000 円に設定
        _betting_settings["daily_investment_limit"] = 30000

    def test_within_daily_limit(self):
        """本日累計 + 今回発注額が上限以下のとき、超過と判定されないこと"""
        # 本日累計 20,000 円、追加 5,000 円 -> 合計 25,000 円 <= 30,000 円
        with patch("app.betting.settings.get_today_total_betted", return_value=20000):
            is_exceeded, current_betted, limit = check_investment_limit(5000)
            self.assertFalse(is_exceeded, "上限内のため False が返却されること")
            self.assertEqual(current_betted, 20000)
            self.assertEqual(limit, 30000)

    def test_exceeded_daily_limit(self):
        """本日累計 + 今回発注額が上限を超えるとき、超過 (is_exceeded=True) と判定されること"""
        # 本日累計 28,000 円、追加 5,000 円 -> 合計 33,000 円 > 30,000 円
        with patch("app.betting.settings.get_today_total_betted", return_value=28000):
            is_exceeded, current_betted, limit = check_investment_limit(5000)
            self.assertTrue(is_exceeded, "上限超過のため True が返却され安全スキップされること")
            self.assertEqual(current_betted, 28000)
            self.assertEqual(limit, 30000)

    def test_exact_daily_limit(self):
        """本日累計 + 今回発注額が上限ぴったりのとき、超過と判定されないこと"""
        # 本日累計 25,000 円、追加 5,000 円 -> 合計 30,000 円 == 30,000 円
        with patch("app.betting.settings.get_today_total_betted", return_value=25000):
            is_exceeded, current_betted, limit = check_investment_limit(5000)
            self.assertFalse(is_exceeded, "上限ちょうどは許容されること")

    def test_max_bet_per_race_constant(self):
        """1レースあたりの最大賭け金定数が安全基準（最大5000円以内）に収まっていること"""
        self.assertLessEqual(settings.DEFAULT_MAX_BET_PER_RACE, 5000)
        self.assertGreater(settings.DEFAULT_MAX_BET_PER_RACE, 0)


if __name__ == "__main__":
    unittest.main()
