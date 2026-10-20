# -*- coding: utf-8 -*-
"""
app/settlement/payout_verifier.py

券種非依存の確定照合＆払戻金計算エンジン (TicketTypeRegistry 連携版)
- 全券種（単勝・複勝・枠連・馬連・ワイド・馬単・3連複・3連単）の統一照合
- 表記ゆれの完全吸収と標準化
"""

from typing import Dict, Any, List, Tuple
from app.core.ticket_spec import TicketTypeRegistry


class PayoutVerifier:
    """払戻金照合エンジン (Unified Ticket Architecture)"""

    @classmethod
    def verify_all_tickets(
        cls,
        tickets: List[Dict[str, Any]],
        payouts_data: Dict[str, Any]
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
        """
        購入チケットリストとJRA公式払戻データ (payouts_data) を照合する
        戻り値: (総払戻金, 的中明細リスト, 券種別投資内訳, 券種別払戻内訳)
        """
        total_payout = 0
        all_hits: List[Dict[str, Any]] = []
        bet_breakdown: Dict[str, int] = {}
        payout_breakdown: Dict[str, int] = {}

        if not tickets or not payouts_data:
            return 0, [], {}, {}

        for t in tickets:
            raw_type = t.get("ticket_type", "")
            norm_type = TicketTypeRegistry.normalize_name(raw_type)
            comb = t.get("combination", [])
            amt = int(t.get("amount", 100) or 100)

            # 券種別投資内訳の集計 (標準名)
            bet_breakdown[norm_type] = bet_breakdown.get(norm_type, 0) + amt

            # 共通照合インターフェースによる払戻金算出
            pay, unit_p, matched_key = TicketTypeRegistry.lookup_payout(norm_type, comb, amt, payouts_data)

            if pay > 0:
                total_payout += pay
                payout_breakdown[norm_type] = payout_breakdown.get(norm_type, 0) + pay
                all_hits.append({
                    "type": norm_type,
                    "comb": matched_key,
                    "amount": amt,
                    "unit_payout": unit_p,
                    "payout": pay
                })

        # レガシー後方互換性: "3連複" と "三連複" の双方で参照可能なように同期
        if "3連複" in payout_breakdown and "三連複" not in payout_breakdown:
            payout_breakdown["三連複"] = payout_breakdown["3連複"]
        elif "三連複" in payout_breakdown and "3連複" not in payout_breakdown:
            payout_breakdown["3連複"] = payout_breakdown["三連複"]

        if "3連複" in bet_breakdown and "三連複" not in bet_breakdown:
            bet_breakdown["三連複"] = bet_breakdown["3連複"]
        elif "三連複" in bet_breakdown and "3連複" not in bet_breakdown:
            bet_breakdown["3連複"] = bet_breakdown["三連複"]

        return total_payout, all_hits, bet_breakdown, payout_breakdown


