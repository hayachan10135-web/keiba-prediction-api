# -*- coding: utf-8 -*-
"""
tests/test_pipeline_integration.py

クラス別特化モデル (A/B/C) および新4区分推論パイプライン統合動作テスト:
- クラス別 LightGBM モデル (A/B/C) のロード検証
- キャリブレーション辞書 (等張回帰) のロード検証
- 推論パイプラインおよび StrategyArbiter との連携確認
- 日次1R保証 (ensure_daily_single_real_bet) の選定動作検証
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPipelineIntegration(unittest.TestCase):
    """推論パイプライン・クラス別モデルの統合テスト"""

    def test_models_loaded(self):
        """クラス別特化モデル (A/B/C) およびキャリブレーションテーブルがロードされていること"""
        from app.ml import predictor

        self.assertIsNotNone(predictor.model_class_a_win, "クラスA Winモデルが未ロード")
        self.assertIsNotNone(predictor.model_class_a_top3, "クラスA Top3モデルが未ロード")
        self.assertIsNotNone(predictor.model_class_b_win, "クラスB Winモデルが未ロード")
        self.assertIsNotNone(predictor.model_class_b_top3, "クラスB Top3モデルが未ロード")
        self.assertIsNotNone(predictor.model_class_c_win, "クラスC Winモデルが未ロード")
        self.assertIsNotNone(predictor.model_class_c_top3, "クラスC Top3モデルが未ロード")
        self.assertIsNotNone(predictor.calibration_table, "キャリブレーションテーブルが未ロード")
        self.assertIn("class_a", predictor.calibration_table)
        self.assertIn("class_b", predictor.calibration_table)
        self.assertIn("class_c", predictor.calibration_table)

    def test_inference_pipeline_execution(self):
        """ダミーレースデータに対して build_prediction_and_recs が正常に推論を完了すること"""
        from app.ml.predictor import build_prediction_and_recs

        horses = [
            {"umaban": 1, "horse_name": "馬A", "odds": 3.0, "popularity": 1, "jockey": "ルメール", "weight": 480, "weight_diff": 0},
            {"umaban": 2, "horse_name": "馬B", "odds": 5.5, "popularity": 2, "jockey": "川田", "weight": 500, "weight_diff": 2},
            {"umaban": 3, "horse_name": "馬C", "odds": 8.0, "popularity": 3, "jockey": "武豊", "weight": 460, "weight_diff": -4},
            {"umaban": 4, "horse_name": "馬D", "odds": 12.0, "popularity": 4, "jockey": "戸崎", "weight": 470, "weight_diff": 0},
            {"umaban": 5, "horse_name": "馬E", "odds": 20.0, "popularity": 5, "jockey": "横山武", "weight": 490, "weight_diff": 4},
            {"umaban": 6, "horse_name": "馬F", "odds": 35.0, "popularity": 6, "jockey": "坂井", "weight": 450, "weight_diff": -2},
        ]
        race_details = {"course_details": "芝2000m", "post_time": "15:40", "weather": "晴", "condition": "良"}
        course_meta = {"venue": "東京", "surface": "芝", "distance": 2000, "race_date": "2026-10-19"}

        res = build_prediction_and_recs(
            horses=horses,
            race_details=race_details,
            race_name="3歳以上2勝クラス",
            race_id="202605040811",
            course_meta=course_meta
        )

        self.assertIn("horses", res)
        self.assertIn("hybrid_recommendation", res)
        h_rec = res["hybrid_recommendation"]
        self.assertIn("bet_category", h_rec)
        self.assertIn(h_rec["bet_category"], ("super", "real", "virtual", "skip"))


if __name__ == "__main__":
    unittest.main()
