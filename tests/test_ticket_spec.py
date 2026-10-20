# -*- coding: utf-8 -*-
"""
tests/test_ticket_spec.py

券種非依存の共通仕様定義 ＆ 券種レジストリ (Unified Ticket Architecture) の単体テスト
- 表記ゆれ正規化 ("三連複" -> "3連複", "ﾜｲﾄﾞ" -> "ワイド")
- 組番フォーマット (順序固定: 7→3, 順不同: 3-7)
- 払戻金自動照合 (全券種)
- レガシー個別フィールド自動生成 (umaren_amount, wide_amount 等)
- StrategyReal における本命馬自身の複勝率参照検証
"""

import unittest
from app.core.ticket_spec import TicketTypeRegistry, TicketTypeSpec
from app.settlement.payout_verifier import PayoutVerifier
from app.strategies.strategy_real import StrategyReal
from app.strategies.models import BetTicket


class TestTicketSpec(unittest.TestCase):
    """Unified Ticket Architecture の包括テスト"""

    def test_normalize_ticket_type_names(self):
        """表記ゆれが標準名に100%正規化されること"""
        self.assertEqual(TicketTypeRegistry.normalize_name("3連複"), "3連複")
        self.assertEqual(TicketTypeRegistry.normalize_name("３連複"), "3連複")
        self.assertEqual(TicketTypeRegistry.normalize_name("三連複"), "3連複")
        self.assertEqual(TicketTypeRegistry.normalize_name("trio"), "3連複")

        self.assertEqual(TicketTypeRegistry.normalize_name("ワイド"), "ワイド")
        self.assertEqual(TicketTypeRegistry.normalize_name("ﾜｲﾄﾞ"), "ワイド")
        self.assertEqual(TicketTypeRegistry.normalize_name("wide"), "ワイド")

        self.assertEqual(TicketTypeRegistry.normalize_name("単勝"), "単勝")
        self.assertEqual(TicketTypeRegistry.normalize_name("複勝"), "複勝")
        self.assertEqual(TicketTypeRegistry.normalize_name("馬連"), "馬連")
        self.assertEqual(TicketTypeRegistry.normalize_name("馬単"), "馬単")
        self.assertEqual(TicketTypeRegistry.normalize_name("3連単"), "3連単")
        self.assertEqual(TicketTypeRegistry.normalize_name("三連単"), "3連単")

    def test_format_combination_key(self):
        """順不同（自動ソート）および順序固定（順序維持）の組番フォーマット検証"""
        # 順不同: 自動昇順ソート
        self.assertEqual(TicketTypeRegistry.format_combination_key("馬連", [7, 3]), "3-7")
        self.assertEqual(TicketTypeRegistry.format_combination_key("ワイド", [9, 2]), "2-9")
        self.assertEqual(TicketTypeRegistry.format_combination_key("3連複", [12, 5, 1]), "1-5-12")

        # 順序固定: 順序維持かつ矢印区切り
        self.assertEqual(TicketTypeRegistry.format_combination_key("馬単", [7, 3]), "7→3")
        self.assertEqual(TicketTypeRegistry.format_combination_key("3連単", [7, 3, 11]), "7→3→11")

    def test_lookup_payout_with_various_types(self):
        """PayoutVerifier および TicketTypeRegistry による払戻金照合"""
        payouts_data = {
            "tansho": {"7": 450},
            "fukusho": {"7": 180, "3": 240},
            "umaren": {"3-7": 1200},
            "wide": {"3-7": 450, "7-11": 680},
            "umatan": {"7-3": 2100},
            "trio": {"3-7-11": 4800},
            "tierce": {"7-3-11": 18500}
        }

        # 単勝
        pay, unit_p, _ = TicketTypeRegistry.lookup_payout("単勝", [7], 2000, payouts_data)
        self.assertEqual(pay, 9000)
        self.assertEqual(unit_p, 450)

        # 複勝
        pay, unit_p, _ = TicketTypeRegistry.lookup_payout("複勝", [7], 1000, payouts_data)
        self.assertEqual(pay, 1800)

        # 馬連 (入力順が逆でも昇順照合)
        pay, unit_p, key = TicketTypeRegistry.lookup_payout("馬連", [7, 3], 500, payouts_data)
        self.assertEqual(pay, 6000)
        self.assertEqual(key, "3-7")

        # ワイド
        pay, unit_p, key = TicketTypeRegistry.lookup_payout("ワイド", [11, 7], 500, payouts_data)
        self.assertEqual(pay, 3400)
        self.assertEqual(key, "7-11")

        # 漢数字「三連複」の入力でも正常にヒットすること
        pay, unit_p, key = TicketTypeRegistry.lookup_payout("三連複", [11, 7, 3], 300, payouts_data)
        self.assertEqual(pay, 14400)
        self.assertEqual(unit_p, 4800)

        # 馬単 (着順が一致した時のみ)
        pay_hit, _, _ = TicketTypeRegistry.lookup_payout("馬単", [7, 3], 1000, payouts_data)
        self.assertEqual(pay_hit, 21000)
        pay_miss, _, _ = TicketTypeRegistry.lookup_payout("馬単", [3, 7], 1000, payouts_data)
        self.assertEqual(pay_miss, 0)

    def test_build_legacy_ticket_fields(self):
        """共通チケット配列から後方互換フィールドが一括生成されること"""
        tickets = [
            BetTicket(ticket_type="単勝", combination=[7], combination_str="7", amount=2000, odds=4.5, expected_payout=9000),
            BetTicket(ticket_type="馬連", combination=[3, 7], combination_str="3-7", amount=500, odds=12.0, expected_payout=6000),
            BetTicket(ticket_type="ワイド", combination=[7, 11], combination_str="7-11", amount=500, odds=6.8, expected_payout=3400),
        ]

        fields = TicketTypeRegistry.build_legacy_ticket_fields(tickets)

        self.assertEqual(fields["total_amount"], 3000)
        self.assertEqual(fields["tansho_amount"], 2000)
        self.assertEqual(fields["umaren_amount"], 500)
        self.assertEqual(fields["wide_amount"], 500)
        self.assertEqual(fields["fukusho_amount"], 0)

        self.assertEqual(len(fields["tansho_bets"]), 1)
        self.assertEqual(len(fields["umaren_bets"]), 1)
        self.assertEqual(len(fields["wide_bets"]), 1)
        self.assertEqual(len(fields["fukusho_bets"]), 0)
        self.assertEqual(len(fields["tickets"]), 3)

    def test_strategy_real_uses_honmei_top3_prob(self):
        """StrategyReal が別馬ではなく本命馬（honmei_horse）自身の複勝率を正しく評価すること"""
        strat = StrategyReal()
        race_info = {"is_shinba": False}

        # ケース1: 本命馬(馬番7)の複勝率が 0.70 (>= 0.65) の場合 -> 合格
        honmei_good = {"umaban": 7, "odds": 4.0, "prob_top3": 0.70}
        horses_1 = [
            {"umaban": 7, "prob_win": 0.35, "prob_top3": 0.70, "odds": 4.0},
            {"umaban": 3, "prob_win": 0.20, "prob_top3": 0.50, "odds": 6.0},
            {"umaban": 1, "prob_win": 0.15, "prob_top3": 0.40, "odds": 8.0},
            {"umaban": 2, "prob_win": 0.10, "prob_top3": 0.30, "odds": 10.0},
            {"umaban": 4, "prob_win": 0.10, "prob_top3": 0.30, "odds": 12.0},
            {"umaban": 5, "prob_win": 0.10, "prob_top3": 0.30, "odds": 15.0},
        ]
        ok_res = strat.evaluate_eligibility(
            race_info=race_info,
            honmei_horse=honmei_good,
            horses=horses_1,
            confidence_score=0.15,
            win_edge=1.40,
            prob_gachi=0.5,
            surface_type=0,
            is_summer=False
        )
        self.assertTrue(ok_res)

        # ケース2: 別馬(馬番3)の複勝率が0.80だが、本命馬(馬番7)自身の複勝率が0.50 (< 0.65) の場合 -> 不合格
        honmei_bad = {"umaban": 7, "odds": 4.0, "prob_top3": 0.50}
        horses_2 = [
            {"umaban": 7, "prob_win": 0.35, "prob_top3": 0.50, "odds": 4.0},
            {"umaban": 3, "prob_win": 0.20, "prob_top3": 0.80, "odds": 6.0},
            {"umaban": 1, "prob_win": 0.15, "prob_top3": 0.40, "odds": 8.0},
            {"umaban": 2, "prob_win": 0.10, "prob_top3": 0.30, "odds": 10.0},
            {"umaban": 4, "prob_win": 0.10, "prob_top3": 0.30, "odds": 12.0},
            {"umaban": 5, "prob_win": 0.10, "prob_top3": 0.30, "odds": 15.0},
        ]
        ng_res = strat.evaluate_eligibility(
            race_info=race_info,
            honmei_horse=honmei_bad,
            horses=horses_2,
            confidence_score=0.15,
            win_edge=1.40,
            prob_gachi=0.5,
            surface_type=0,
            is_summer=False
        )
        self.assertFalse(ng_res, "本命馬自身の複勝率が0.65未満の場合、別馬が高くても実勝負として選出されてはならない")


if __name__ == "__main__":
    unittest.main()

