# -*- coding: utf-8 -*-
"""
app/strategies/strategy_a.py

戦略A: 芝厳選エッジ投資 (単複ケリー傾斜配分)
"""

from typing import Dict, Any, List
import numpy as np
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyA(BaseBettingStrategy):
    """
    【戦略A】芝厳選エッジ投資
    - 芝レース限定 (surface_type == 0)
    - 確信度 >= 0.72
    - 分数ケリー基準による単勝(約75%)＋複勝(約25%)の傾斜配分 (計1,000円ベース)
    """

    @property
    def strategy_type(self) -> str:
        return "strategy_a"

    @property
    def priority(self) -> int:
        return 2

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
        return surface_type == 0 and confidence_score >= 0.72

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

        strategy_name = "⭐ 戦略A: 芝厳選エッジ投資 (単複ケリー傾斜配分)"
        real_bet_badge = f"⭐【芝厳選】単複ケリー傾斜{bias_tag}{suffix}"

        mult = float(np.clip(win_edge, 0.7, 1.8)) * bias_boost_mult
        bt = int(round(1000.0 * mult * 0.75 / 100.0) * 100)
        bf = int(round(1000.0 * mult * 0.25 / 100.0) * 100)

        tickets: List[BetTicket] = []

        if bt > 0:
            tickets.append(BetTicket(
                ticket_type="単勝",
                combination=[w0_u],
                combination_str=f"{w0_u}",
                amount=bt,
                odds=w0_odds,
                expected_payout=int(round(bt * w0_odds)),
                is_real_odds=True,
                memo=f"単勝(ケリー傾斜) {w0_u}番 ({w0_name}){bias_tag} {bt:,}円"
            ))

        if bf > 0:
            est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
            tickets.append(BetTicket(
                ticket_type="複勝",
                combination=[w0_u],
                combination_str=f"{w0_u}",
                amount=bf,
                odds=est_fuku_odds,
                expected_payout=int(round(bf * est_fuku_odds)),
                is_real_odds=False,
                memo=f"複勝(ケリー傾斜) {w0_u}番 ({w0_name}){bias_tag} {bf:,}円"
            ))

        total_amount = sum(t.amount for t in tickets)
        summary = f"{strategy_name} [単勝{bt:,}円 ＋ 複勝{bf:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="A",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            real_bet_level="standard",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason="戦略A採択 (芝厳選ケリー傾斜)"
        )

