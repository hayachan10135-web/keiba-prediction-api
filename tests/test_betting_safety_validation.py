"""投票設定永続化とマニュアル投票入力のフェイルクローズ検証。"""

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from app.betting.factory import BettingClientFactory
from app.betting.manual_service import validate_manual_bet
from app.betting.settings import (
    _betting_settings,
    load_betting_settings_from_supabase,
    save_betting_settings_to_supabase,
)
from app.routers.betting import (
    ManualBetItem,
    ManualBetRequest,
    UpdateBettingModeRequest,
    manual_bet_endpoint,
    update_betting_mode_endpoint,
)


class TestBettingSafetyValidation(unittest.TestCase):
    def setUp(self):
        self.original_settings = dict(_betting_settings)

    def tearDown(self):
        _betting_settings.clear()
        _betting_settings.update(self.original_settings)

    def test_settings_load_and_save_fail_closed_without_database(self):
        with patch("app.betting.settings.get_supabase_client", return_value=None):
            with self.assertRaises(RuntimeError):
                load_betting_settings_from_supabase()
            self.assertFalse(save_betting_settings_to_supabase({"betting_mode": "ipat"}))
        self.assertEqual(_betting_settings["betting_mode"], self.original_settings["betting_mode"])

    def test_failed_settings_persistence_returns_service_unavailable(self):
        with patch("app.routers.betting.load_betting_settings_from_supabase", return_value=dict(_betting_settings)):
            with patch("app.routers.betting.save_betting_settings_to_supabase", return_value=False):
                with self.assertRaises(HTTPException) as exc:
                    update_betting_mode_endpoint(UpdateBettingModeRequest(betting_mode="ipat"))
        self.assertEqual(exc.exception.status_code, 503)

    def test_daily_limit_can_be_updated_without_changing_mode(self):
        current = {**_betting_settings, "betting_mode": "off"}
        with patch("app.routers.betting.load_betting_settings_from_supabase", return_value=current):
            with patch("app.routers.betting.save_betting_settings_to_supabase", return_value=True):
                result = update_betting_mode_endpoint(UpdateBettingModeRequest(daily_investment_limit=25000))
        self.assertEqual(result["betting_mode"], "off")

    def test_manual_bet_rejects_non_unit_amounts_and_malformed_ticket(self):
        with self.assertRaisesRegex(ValueError, "100円単位"):
            validate_manual_bet(
                "manual_tokyo_11",
                "東京",
                11,
                [{"ticket_type": "単勝", "combination": [1], "amount": 150}],
            )
        with self.assertRaisesRegex(ValueError, "必要です"):
            validate_manual_bet(
                "manual_tokyo_11",
                "東京",
                11,
                [{"ticket_type": "馬連", "combination": [1], "amount": 100}],
            )

    def test_manual_endpoint_rejects_invalid_mode_before_client_creation(self):
        request = ManualBetRequest(
            race_id="manual_tokyo_11",
            venue_name="東京",
            race_no=11,
            items=[ManualBetItem(ticket_type="単勝", combination=[1], amount=100)],
            mode="invalid",
            dry_run=True,
        )
        with patch("app.routers.betting.load_betting_settings_from_supabase", return_value={"betting_mode": "ipat"}):
            with patch("app.routers.betting.execute_manual_bet") as execute:
                with self.assertRaises(HTTPException) as exc:
                    manual_bet_endpoint(request)
        self.assertEqual(exc.exception.status_code, 400)
        execute.assert_not_called()

    def test_factory_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            BettingClientFactory.get_client("invalid")


if __name__ == "__main__":
    unittest.main()
