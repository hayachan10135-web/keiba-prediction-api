"""本番キャリブレーション表の単調性と補間を検証する。"""

import json
import unittest

import numpy as np

from app.ml.predictor import calibrate_probabilities
from app.ml.model_loader import CALIBRATION_TABLE_PATH


class TestCalibration(unittest.TestCase):
    def test_production_calibration_tables_are_monotonic(self):
        with open(CALIBRATION_TABLE_PATH, "r", encoding="utf-8") as calibration_file:
            tables = json.load(calibration_file)

        self.assertEqual(set(tables), {"class_a", "class_b", "class_c"})
        for class_tables in tables.values():
            for table_name in ("win_calib", "top3_calib"):
                table = class_tables[table_name]
                predictions = [row["pred"] for row in table]
                calibrated = [row["calib"] for row in table]
                self.assertTrue(all(a < b for a, b in zip(predictions, predictions[1:])))
                self.assertTrue(all(a <= b for a, b in zip(calibrated, calibrated[1:])))
                calibrated_values = calibrate_probabilities(np.array([0.2, 0.5]), table)
                self.assertTrue(np.all(np.diff(calibrated_values) >= 0))

    def test_invalid_or_missing_calibration_table_is_rejected(self):
        with self.assertRaises(ValueError):
            calibrate_probabilities(np.array([0.5]), [])
        with self.assertRaises(ValueError):
            calibrate_probabilities(
                np.array([0.5]),
                [{"pred": 0.2, "calib": 0.4}, {"pred": 0.8, "calib": 0.3}],
            )


if __name__ == "__main__":
    unittest.main()
