# -*- coding: utf-8 -*-
"""
app/settlement/payout_verifier.py

券種別（単勝・複勝・馬単・馬連・三連複・3連単）の確定照合＆払戻金計算エンジン
"""

from typing import Dict, Any, List, Tuple


class PayoutVerifier:
    """払戻金照合エンジン"""

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

        tan_map = payouts_data.get("tansho", {})
        fuku_map = payouts_data.get("fukusho", {})
        utan_map = payouts_data.get("umatan", {})
        uren_map = payouts_data.get("umaren", {})
        tierce_map = payouts_data.get("tierce", {})  # 3連単
        trio_map = payouts_data.get("trio", {})      # 3連複

        for t in tickets:
            t_type = t.get("ticket_type", "")
            comb = t.get("combination", [])
            amt = int(t.get("amount", 100))
            bet_breakdown[t_type] = bet_breakdown.get(t_type, 0) + amt

            # 1. 単勝
            if t_type == "単勝" and len(comb) >= 1:
                u = comb[0]
                unit_p = tan_map.get(u, 0) or tan_map.get(str(u), 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["単勝"] = payout_breakdown.get("単勝", 0) + pay
                    all_hits.append({
                        "type": "単勝", "comb": f"{u}", "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

            # 2. 複勝
            elif t_type == "複勝" and len(comb) >= 1:
                u = comb[0]
                unit_p = fuku_map.get(u, 0) or fuku_map.get(str(u), 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["複勝"] = payout_breakdown.get("複勝", 0) + pay
                    all_hits.append({
                        "type": "複勝", "comb": f"{u}", "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

            # 3. 馬単
            elif t_type == "馬単" and len(comb) >= 2:
                comb_key1 = f"{comb[0]}-{comb[1]}"
                comb_key2 = f"{comb[0]}→{comb[1]}"
                unit_p = utan_map.get(comb_key1, 0) or utan_map.get(comb_key2, 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["馬単"] = payout_breakdown.get("馬単", 0) + pay
                    all_hits.append({
                        "type": "馬単", "comb": comb_key2, "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

            # 4. 馬連
            elif t_type == "馬連" and len(comb) >= 2:
                comb_key = f"{min(comb[0], comb[1])}-{max(comb[0], comb[1])}"
                unit_p = uren_map.get(comb_key, 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["馬連"] = payout_breakdown.get("馬連", 0) + pay
                    all_hits.append({
                        "type": "馬連", "comb": comb_key, "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

            # 5. 3連単
            elif t_type in ("3連単", "３連単", "三連単") and len(comb) >= 3:
                comb_key = f"{comb[0]}-{comb[1]}-{comb[2]}"
                unit_p = tierce_map.get(comb_key, 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["3連単"] = payout_breakdown.get("3連単", 0) + pay
                    all_hits.append({
                        "type": "3連単", "comb": f"{comb[0]}→{comb[1]}→{comb[2]}", "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

            # 6. 3連複
            elif t_type in ("3連複", "３連複", "三連複") and len(comb) >= 3:
                s_comb = sorted(comb)
                comb_key = f"{s_comb[0]}-{s_comb[1]}-{s_comb[2]}"
                unit_p = trio_map.get(comb_key, 0)
                if unit_p > 0:
                    pay = int((amt / 100.0) * unit_p)
                    total_payout += pay
                    payout_breakdown["3連複"] = payout_breakdown.get("3連複", 0) + pay
                    all_hits.append({
                        "type": "3連複", "comb": comb_key, "amount": amt, "unit_payout": unit_p, "payout": pay
                    })

        return total_payout, all_hits, bet_breakdown, payout_breakdown

