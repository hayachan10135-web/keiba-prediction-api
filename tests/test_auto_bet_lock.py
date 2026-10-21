"""自動投票前の永続ロックと結果状態更新を検証する。"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.betting.settings import reserve_auto_bet_attempt, update_auto_bet_attempt
from app.core.sqlite_bridge import LocalSQLiteClient


class TestAutoBetLock(unittest.TestCase):
    def test_pending_reservation_blocks_duplicate_attempt_and_can_be_finalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "auto-bet-lock.db")
            client = LocalSQLiteClient(db_path)
            with patch("app.betting.settings.get_supabase_client", return_value=client):
                self.assertTrue(reserve_auto_bet_attempt(
                    "202610010501",
                    "ipat",
                    3000,
                    [{"ticket_type": "単勝", "combination": [3], "amount": 3000}],
                    "2026-10-01 12:00:00",
                ))
                self.assertFalse(reserve_auto_bet_attempt(
                    "202610010501",
                    "ipat",
                    3000,
                    [{"ticket_type": "単勝", "combination": [3], "amount": 3000}],
                    "2026-10-01 12:00:01",
                ))
                with self.assertRaises(sqlite3.IntegrityError):
                    client.table("keiba_real_bets").insert({
                        "race_id": "202610010501",
                        "platform": "ipat",
                        "receipt_no": "",
                        "total_amount": 3000,
                        "items": [],
                        "status": "pending",
                        "is_manual_bet": False,
                        "bet_time": "2026-10-01 12:00:02",
                    }).execute()

                update_auto_bet_attempt(
                    "202610010501",
                    {"status": "success", "receipt_no": "receipt-1"},
                )
                row = client.table("keiba_real_bets")\
                    .select("status, receipt_no")\
                    .eq("race_id", "202610010501")\
                    .execute().data[0]
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["receipt_no"], "receipt-1")

            client.pool.get_connection().close()


if __name__ == "__main__":
    unittest.main()
