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


if __name__ == "__main__":
    unittest.main()
