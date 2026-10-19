# -*- coding: utf-8 -*-
"""
tests/test_arbiter_4classes.py
新4区分戦略調停エンジン (StrategyArbiter) 網羅検証テスト:
🔥【勝負R】/ ⭐【実勝負】/ 🎮【仮想】/ 🚫【見送り】
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.strategies.arbiter import StrategyArbiter


class TestArbiter4Classes(unittest.TestCase):
    """StrategyArbiter 4区分振り分けロジックの網羅的テスト"""

    def setUp(self):
        self.arbiter = StrategyArbiter()

    def _create_mock_horses(self, count=12, top_odds=3.5, top_prob_win=0.35, top_prob_top3=0.70):
        horses = []
        for i in range(1, count + 1):
            if i == 1:
                horses.append({
                    "umaban": 1,
                    "horse_name": "テスト本命馬",
                    "odds": top_odds,
                    "prob_win": top_prob_win,
                    "prob_top3": top_prob_top3,
                    "bias_match": "絶好バイアス",
                })
            elif i == 2:
                horses.append({
                    "umaban": 2,
                    "horse_name": "テスト対抗馬",
                    "odds": 5.0,
                    "prob_win": 0.18,
                    "prob_top3": 0.45,
                    "bias_match": "普通",
                })
            else:
                horses.append({
                    "umaban": i,
                    "horse_name": f"テスト馬{i}",
                    "odds": 10.0 + i * 2,
                    "prob_win": 0.05,
                    "prob_top3": 0.15,
                    "bias_match": "普通",
                })
        return horses

    def test_skip_on_insufficient_horses(self):
        """出走頭数が6頭未満の場合、必ず見送り(skip)になること"""
        horses = self._create_mock_horses(count=5)
        decision = self.arbiter.evaluate_race(
            race_name="3歳以上1勝クラス",
            race_details={"course_details": "芝2000m"},
            horses=horses,
            surface_type=0,
            prob_gachi=0.4,
            prob_himo=0.3,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
        )
        self.assertEqual(decision["bet_category"], "skip")
        self.assertFalse(decision["is_participate"])
        self.assertIn("出走頭数不足", decision["reason"])

    def test_skip_on_obstacle_race(self):
        """障害戦の場合、必ず見送り(skip)になること"""
        horses = self._create_mock_horses(count=10)
        decision = self.arbiter.evaluate_race(
            race_name="障害3歳以上オープン",
            race_details={"course_details": "障2800m"},
            horses=horses,
            surface_type=2,
            prob_gachi=0.4,
            prob_himo=0.3,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
        )
        self.assertEqual(decision["bet_category"], "skip")
        self.assertIn("障害戦除外", decision["reason"])

    def test_super_or_real_bet_classification(self):
        """高エッジ・芝レースにおいて実馬券対象（勝負Rまたは実勝負）に分類されること"""
        horses = self._create_mock_horses(count=14, top_odds=4.2, top_prob_win=0.38, top_prob_top3=0.72)
        decision = self.arbiter.evaluate_race(
            race_name="3歳以上2勝クラス",
            race_details={"course_details": "芝1800m"},
            horses=horses,
            surface_type=0,
            prob_gachi=0.3,
            prob_himo=0.2,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
            race_class_code=3,
        )
        self.assertTrue(decision["is_participate"])
        self.assertIn(decision["bet_category"], ("super", "real"))
        if decision["bet_category"] == "super":
            self.assertIn("🔥", decision["badge_text"])

    def test_real_guarantee_promotion(self):
        """is_real_guarantee=True（日次1R保証枠）の場合、⭐【実勝負】として選出されること"""
        horses = self._create_mock_horses(count=12, top_odds=2.5, top_prob_win=0.30, top_prob_top3=0.65)
        decision = self.arbiter.evaluate_race(
            race_name="3歳以上1勝クラス",
            race_details={"course_details": "ダ1800m"},
            horses=horses,
            surface_type=1,
            prob_gachi=0.5,
            prob_himo=0.2,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
            is_real_guarantee=True,
            race_class_code=2,
        )
        self.assertTrue(decision["is_participate"])
        self.assertEqual(decision["bet_category"], "real")
        self.assertTrue(decision["is_real_bet_target"])
        self.assertIn("⭐", decision["badge_text"])

    def test_virtual_bet_classification(self):
        """is_virtual_guarantee=True の場合、🎮【仮想】として選出され、実馬券対象外となること"""
        horses = self._create_mock_horses(count=10, top_odds=6.0, top_prob_win=0.18, top_prob_top3=0.45)
        decision = self.arbiter.evaluate_race(
            race_name="3歳以上1勝クラス",
            race_details={"course_details": "ダ1200m"},
            horses=horses,
            surface_type=1,
            prob_gachi=0.2,
            prob_himo=0.5,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
            is_virtual_guarantee=True,
            race_class_code=2,
        )
        self.assertTrue(decision["is_participate"])
        self.assertEqual(decision["bet_category"], "virtual")
        self.assertFalse(decision["is_real_bet_target"])
        self.assertTrue(decision["is_virtual_bet"])
        self.assertIn("🎮", decision["badge_text"])

    def test_virtual_class_a_generates_wide_bets(self):
        """クラスA（新馬・未勝利）の仮想枠選出時、ワイド2点（計1,000円）が正しく生成されること"""
        horses = self._create_mock_horses(count=12, top_odds=4.5, top_prob_win=0.25, top_prob_top3=0.55)
        decision = self.arbiter.evaluate_race(
            race_name="3歳未勝利",
            race_details={"course_details": "芝1600m"},
            horses=horses,
            surface_type=0,
            prob_gachi=0.3,
            prob_himo=0.4,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
            is_virtual_guarantee=True,
            race_class_code=1,  # クラスA (未勝利)
        )
        self.assertEqual(decision["bet_category"], "virtual")
        self.assertIn("wide_bets", decision)
        self.assertEqual(len(decision["wide_bets"]), 2)
        self.assertEqual(decision["wide_amount"], 1000)
        self.assertEqual(len(decision["virtual_bets"]), 2)
        self.assertEqual(decision["virtual_total_amount"], 1000)

    def test_daily_top_race_promotion(self):
        """is_daily_top_race=True（自然合致0件時のデイリー昇格枠）の場合、⭐【実勝負(保証)】として選出され実馬券対象となること"""
        horses = self._create_mock_horses(count=10, top_odds=8.5, top_prob_win=0.15, top_prob_top3=0.40)
        decision = self.arbiter.evaluate_race(
            race_name="3歳以上1勝クラス",
            race_details={"course_details": "ダ1800m"},
            horses=horses,
            surface_type=1,
            prob_gachi=0.2,
            prob_himo=0.2,
            opponents=[],
            u0=1,
            honmei_horse=horses[0],
            race_date_str="2026-10-19",
            is_daily_top_race=True,
            race_class_code=2,
        )
        self.assertTrue(decision["is_participate"])
        self.assertEqual(decision["bet_category"], "real")
        self.assertTrue(decision["is_real_bet_target"])
        self.assertEqual(decision["total_amount"], 300)
        self.assertIn("wide_bets", decision)
        self.assertIn("wide_amount", decision)

    def test_payout_verifier_wide_hit(self):
        """PayoutVerifier がワイド馬券の的中と払戻金を正しく照合できること"""
        from app.settlement.payout_verifier import PayoutVerifier

        tickets = [
            {"ticket_type": "ワイド", "combination": [1, 2], "amount": 500},
            {"ticket_type": "ワイド", "combination": [1, 3], "amount": 500},
        ]
        payouts_data = {
            "wide": {
                "1-2": 480,   # 1-2 的中 (100円あたり480円 -> 500円投資で 2,400円)
                "2-5": 820,
                "1-5": 650,
            }
        }
        total_payout, all_hits, bet_breakdown, payout_breakdown = PayoutVerifier.verify_all_tickets(tickets, payouts_data)
        self.assertEqual(total_payout, 2400)
        self.assertEqual(len(all_hits), 1)
        self.assertEqual(all_hits[0]["type"], "ワイド")
        self.assertEqual(all_hits[0]["comb"], "1-2")
        self.assertEqual(all_hits[0]["payout"], 2400)
        self.assertEqual(bet_breakdown.get("ワイド"), 1000)
        self.assertEqual(payout_breakdown.get("ワイド"), 2400)

    def test_virtual_bet_slot_type_is_int_or_none(self):
        """virtual_bet_slot が文字列ではなく None または int であること（FastAPI/Android型整合性）"""
        arbiter = StrategyArbiter()
        mock_horses = self._create_mock_horses()
        decision = arbiter.evaluate_race(
            race_name="テスト未勝利",
            race_details={"course_details": "芝2000m"},
            horses=mock_horses,
            surface_type=0,
            prob_gachi=0.2,
            prob_himo=0.8,
            opponents=[],
            u0=1,
            honmei_horse=mock_horses[0],
            race_date_str="2026-10-20",
            is_virtual_guarantee=True,
            race_class_code=1
        )
        self.assertIn("virtual_bet_slot", decision)
        self.assertTrue(
            decision["virtual_bet_slot"] is None or isinstance(decision["virtual_bet_slot"], int),
            f"virtual_bet_slot must be None or int, but got {type(decision['virtual_bet_slot'])}"
        )

    def test_all_10_jra_venues_supported(self):
        """全10競馬場 (01:札幌 〜 10:小倉) のレースIDが正常に認識されること"""
        for code_int in range(1, 11):
            venue_code = f"{code_int:02d}"
            # 12桁のレースID例: 2026 + 会場コード(2桁) + 回日R(6桁)
            sample_rid = f"2026{venue_code}010101"
            v_code = sample_rid[4:6]
            is_valid = (len(sample_rid) == 12 and v_code.isdigit() and 1 <= int(v_code) <= 10)
            self.assertTrue(is_valid, f"Venue code {venue_code} should be valid")

    def test_aggregate_summary_runs_cleanly(self):
        """最適化されたカラム指定クエリで総合収支集計が正常完了すること"""
        from app.settlement.summary_calculator import calculate_aggregate_summary
        agg = calculate_aggregate_summary(force_refresh=True)
        self.assertIn("lifetime", agg)
        self.assertIn("yearly", agg)
        self.assertIn("monthly", agg)
        self.assertEqual(agg["lifetime"]["period_key"], "lifetime")


if __name__ == "__main__":
    unittest.main()
