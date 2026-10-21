"""保証枠は当日スケジュール全件の予想が揃ってから確定する。"""

import unittest
from unittest.mock import MagicMock

from app.strategies.venue_guarantee import (
    invalidate_venue_guarantee_cache,
    resolve_venue_guarantees_for_date,
)


def _race_row(race_id, top3_probability):
    horses = [
        {
            "umaban": 1,
            "prob_win": 0.2,
            "prob_top3": top3_probability,
            "odds": 4.0,
        }
    ] + [
        {"umaban": number, "prob_win": 0.1, "prob_top3": 0.2, "odds": 8.0}
        for number in range(2, 7)
    ]
    return {
        "race_id": race_id,
        "data_json": {
            "race_name": "3歳以上1勝クラス",
            "race_details": {"course_details": "芝1600m"},
            "horses": horses,
            "hybrid_recommendation": {
                "bet_category": "skip",
                "strategy_type": "skip",
                "confidence_score": 0.1,
                "edge_ratio": 1.0,
            },
        },
    }


class TestGuaranteeSlots(unittest.TestCase):
    def setUp(self):
        invalidate_venue_guarantee_cache()
        self.race_ids = ["202610010501", "202610010502", "202610010503"]
        self.scheduled = [{"race_id": race_id} for race_id in self.race_ids]

    def tearDown(self):
        invalidate_venue_guarantee_cache()

    def _db(self, predictions):
        db = MagicMock()
        scheduled_query = MagicMock()
        scheduled_query.select.return_value.eq.return_value.execute.return_value.data = self.scheduled
        prediction_query = MagicMock()
        prediction_query.select.return_value.eq.return_value.execute.return_value.data = predictions
        db.table.side_effect = lambda name: (
            scheduled_query if name == "keiba_scheduled_races" else prediction_query
        )
        return db

    def test_incomplete_predictions_are_not_cached_as_guarantee_result(self):
        partial = [_race_row(self.race_ids[0], 0.9)]
        self.assertEqual(resolve_venue_guarantees_for_date("2026-10-01", self._db(partial)), (set(), set()))

        complete = [
            _race_row(self.race_ids[0], 0.9),
            _race_row(self.race_ids[1], 0.8),
            _race_row(self.race_ids[2], 0.7),
        ]
        real_guarantees, virtual_guarantees = resolve_venue_guarantees_for_date(
            "2026-10-01", self._db(complete)
        )
        self.assertEqual(real_guarantees, {self.race_ids[0]})
        self.assertEqual(virtual_guarantees, {self.race_ids[1], self.race_ids[2]})

    def test_existing_real_and_virtual_races_count_toward_venue_minimum(self):
        predictions = [
            _race_row(self.race_ids[0], 0.9),
            _race_row(self.race_ids[1], 0.8),
            _race_row(self.race_ids[2], 0.7),
        ]
        predictions[0]["data_json"]["hybrid_recommendation"].update({
            "bet_category": "real",
            "strategy_type": "real_bet",
        })
        predictions[1]["data_json"]["hybrid_recommendation"].update({
            "bet_category": "virtual",
            "strategy_type": "virtual_bet",
        })
        predictions[2]["data_json"]["hybrid_recommendation"].update({
            "bet_category": "skip",
            "strategy_type": "skip",
        })

        real_guarantees, virtual_guarantees = resolve_venue_guarantees_for_date(
            "2026-10-01", self._db(predictions)
        )
        self.assertEqual(real_guarantees, set())
        self.assertEqual(virtual_guarantees, {self.race_ids[2]})


if __name__ == "__main__":
    unittest.main()
