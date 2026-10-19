# -*- coding: utf-8 -*-
"""
app/strategies/strategy_virtual.py

🎮【仮想】戦略: 観戦エンタメ・クラス別最適券種
- 出現頻度: 1日・各場3R以上保証 (上限なし)
- 券種 (クラス別最適化):
    - クラスA (新馬・未勝利): ワイド流し2点 (◎ - ◯, ▲) 各500円 = 1,000円
    - クラスB (1・2勝クラス): 単勝 1,000円
    - クラスC (3勝・オープン): 馬連流し2点 (◎ - ◯, ▲) 各500円 = 1,000円
- 投票: 実投票なし (is_real_bet_target = False, is_virtual_bet = True)
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyVirtual(BaseBettingStrategy):
    """🎮【仮想】観戦エンタメ枠戦略 (投票なし・仮想台帳集計あり)"""

    @property
    def strategy_type(self) -> str:
        return "virtual_bet"

    @property
    def priority(self) -> int:
        return 3  # 実勝負の次

    def evaluate_eligibility(
        self,
        race_info: Dict[str, Any],
        honmei_horse: Dict[str, Any],
        horses: List[Dict[str, Any]],
        confidence_score: float,
        win_edge: float,
        prob_gachi: float,
        surface_type: int,
        is_summer: bool,
    ) -> bool:
        # 会場スロット保証による明示的選定の場合
        if race_info.get("is_virtual_guarantee", False):
            return True

        if len(horses) < 6 or surface_type == 2:
            return False

        # 自然適格条件: 期待値 >= 1.00 & 確信度 >= 0.04
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        conf_margin = float(s_win[0].get("prob_win", 0.0) - (s_win[1].get("prob_win", 0.0) if len(s_win) > 1 else 0.0))
        return (win_edge >= 1.00) and (conf_margin >= 0.04)

    def generate_tickets(
        self,
        race_info: Dict[str, Any],
        honmei_horse: Dict[str, Any],
        horses: List[Dict[str, Any]],
        confidence_score: float,
        win_edge: float,
        bias_boost_mult: float,
        bias_tag: str,
        suffix: str,
    ) -> StrategyDecision:
        w0_u = int(honmei_horse.get("umaban", 0))
        w0_name = str(honmei_horse.get("horse_name", f"馬{w0_u}"))
        w0_odds = float(honmei_horse.get("odds") or 50.0)

        race_class_code = race_info.get("race_class_code", honmei_horse.get("race_class_code", 2))
        is_class_a = (race_class_code in (0, 1)) or (race_info.get("class_group") == "A_Maiden")
        is_class_c = (race_class_code in (4, 5)) or (race_info.get("class_group") == "C_3_Open")

        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        candidate_opps = [h for h in s_win if int(h.get("umaban", 0)) != w0_u]
        opponents = candidate_opps[:2]

        tickets: List[BetTicket] = []

        if is_class_a:
            # クラスA (新馬・未勝利): ワイド流し2点 各500円
            ticket_type_name = "ワイド2点"
            for opp_h in opponents:
                opp_u = int(opp_h["umaban"])
                opp_name = str(opp_h.get("horse_name", f"馬{opp_u}"))
                comb = sorted([w0_u, opp_u])
                tickets.append(BetTicket(
                    ticket_type="ワイド",
                    combination=comb,
                    combination_str=f"{comb[0]}-{comb[1]}",
                    amount=500,
                    odds=round(max(2.0, w0_odds * 0.4), 1),
                    expected_payout=int(round(500 * max(2.0, w0_odds * 0.4))),
                    is_real_odds=False,
                    memo=f"ワイド {comb[0]}-{comb[1]} ({opp_name}) 500円"
                ))
        elif is_class_c:
            # クラスC (3勝・オープン): 馬連流し2点 各500円
            ticket_type_name = "馬連2点"
            for opp_h in opponents:
                opp_u = int(opp_h["umaban"])
                opp_name = str(opp_h.get("horse_name", f"馬{opp_u}"))
                comb = sorted([w0_u, opp_u])
                tickets.append(BetTicket(
                    ticket_type="馬連",
                    combination=comb,
                    combination_str=f"{comb[0]}-{comb[1]}",
                    amount=500,
                    odds=round(max(3.0, w0_odds * 0.8), 1),
                    expected_payout=int(round(500 * max(3.0, w0_odds * 0.8))),
                    is_real_odds=False,
                    memo=f"馬連 {comb[0]}-{comb[1]} ({opp_name}) 500円"
                ))
        else:
            # クラスB (1・2勝クラス): 単勝 1,000円
            ticket_type_name = "単勝"
            tickets.append(BetTicket(
                ticket_type="単勝",
                combination=[w0_u],
                combination_str=f"{w0_u}",
                amount=1000,
                odds=w0_odds,
                expected_payout=int(round(1000 * w0_odds)),
                is_real_odds=True,
                memo=f"単勝 {w0_u}番 ({w0_name}) 1,000円"
            ))

        total_amount = sum(t.amount for t in tickets)
        strategy_name = f"🎮【仮想】クラス別観戦エンタメ ({ticket_type_name})"
        real_bet_badge = f"🎮【仮想】{ticket_type_name}{bias_tag}{suffix}"
        summary = f"{strategy_name} [軸{w0_u} {w0_name} / 想定{total_amount:,}円 (非投票)]"
        reason_text = "【仮想】会場最低3R保証枠" if race_info.get("is_virtual_guarantee") else f"【仮想】自然適格枠 (EV {win_edge:.2f} >= 1.00)"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=False,
            is_virtual_bet=True,
            bet_category="virtual",
            assigned_tier="VIRTUAL",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            badge_text="🎮仮想",
            real_bet_level="virtual",
            tickets=tickets,
            total_amount=total_amount,
            summary=summary,
            reason=reason_text
        )
