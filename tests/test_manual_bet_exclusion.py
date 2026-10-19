# -*- coding: utf-8 -*-
"""
tests/test_manual_bet_exclusion.py

要件 FR-BET-07 / FR-SET-05 / AGENT_RULES 1.3 準拠検証テスト:
マニュアル（手動・実機テスト）投票レコード (is_manual_bet=True) が、
AIモデルの実戦成績集計および当日累計投資額集計から物理的に完全除外されていることを検証する。
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# テスト対象ディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.betting.settings import get_today_total_betted


class TestManualBetExclusion(unittest.TestCase):
    """マニュアル投票除外の厳格検証テスト"""

    def test_get_today_total_betted_excludes_manual_bets(self):
        """
        get_today_total_betted が is_manual_bet=True のレコードを除外し、
        is_manual_bet=False のレコードのみを集計することを検証
        """
        # モックSupabase / SQLite レスポンス
        # 1. 自動投票レコード (is_manual_bet=False, 2,500円) -> 集計対象
        # 2. 手動テスト投票レコード (is_manual_bet=True, 1,000円) -> 除外対象
        mock_data = [
            {"total_amount": 2500, "is_manual_bet": False},
        ]

        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_select = MagicMock()
        mock_gte = MagicMock()
        mock_lte = MagicMock()
        mock_eq_status = MagicMock()
        mock_eq_manual = MagicMock()

        mock_supabase.table.return_value = mock_table
        mock_table.select.return_value = mock_select
        mock_select.gte.return_value = mock_gte
        mock_gte.lte.return_value = mock_lte
        mock_lte.eq.return_value = mock_eq_status
        mock_eq_status.eq.return_value = mock_eq_manual
        mock_eq_manual.execute.return_value = MagicMock(data=mock_data)

        with patch("app.betting.settings.get_supabase_client", return_value=mock_supabase):
            total = get_today_total_betted("2026-10-19")

            # eq("is_manual_bet", False) がクエリチェーンで確実に呼ばれたことを検証
            mock_eq_status.eq.assert_called_with("is_manual_bet", False)
            # 自動投票の2,500円のみが集計され、マニュアル投票は含まれないこと
            self.assertEqual(total, 2500)

    def test_sqlite_bridge_filters_manual_bets(self):
        """
        実際のローカル SQLite クライアント (LocalSQLiteClient) において、
        eq('is_manual_bet', False) によるフィルタが正しく動作することを検証
        """
        from app.core.sqlite_bridge import LocalSQLiteClient, DEFAULT_DB_PATH

        if not os.path.exists(DEFAULT_DB_PATH):
            self.skipTest("ローカル SQLite DBが存在しないためスキップ")

        client = LocalSQLiteClient(DEFAULT_DB_PATH)
        # is_manual_bet=True を明示してクエリ
        res_manual = client.table("keiba_real_bets").select("id, is_manual_bet").eq("is_manual_bet", True).execute()
        for row in (res_manual.data or []):
            self.assertTrue(bool(row.get("is_manual_bet")), "is_manual_bet=True のみが取得されること")

        # is_manual_bet=False を明示してクエリ
        res_auto = client.table("keiba_real_bets").select("id, is_manual_bet").eq("is_manual_bet", False).execute()
        for row in (res_auto.data or []):
            self.assertFalse(bool(row.get("is_manual_bet")), "is_manual_bet=False のみが取得されること")

    def test_sqlite_bridge_distinct_query(self):
        """
        LocalSQLiteClient において SELECT DISTINCT <column> が正常に解釈され、
        重複のないカラム値リストが返却されることを検証
        """
        from app.core.sqlite_bridge import LocalSQLiteClient, DEFAULT_DB_PATH

        if not os.path.exists(DEFAULT_DB_PATH):
            self.skipTest("ローカル SQLite DBが存在しないためスキップ")

        client = LocalSQLiteClient(DEFAULT_DB_PATH)
        res_distinct = client.table("keiba_race_settlements").select("DISTINCT race_date").execute()
        self.assertGreater(len(res_distinct.data), 0)
        seen_dates = set()
        for row in res_distinct.data:
            self.assertIn("race_date", row, "race_date カラムが存在すること")
            d = row["race_date"]
            self.assertNotIn(d, seen_dates, f"重複日付が存在してはならない: {d}")
            seen_dates.add(d)


if __name__ == "__main__":
    unittest.main()
