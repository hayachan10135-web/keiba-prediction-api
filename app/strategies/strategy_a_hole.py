# -*- coding: utf-8 -*-
"""
app/strategies/strategy_a_hole.py

戦略A-2: 芝波乱妙味拾い上げ (単勝500円＋複勝500円)
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyAHole(BaseBettingStrategy):
    """
    【戦略A-2】芝波乱妙味拾い上げ
    - 芝レース限定 (surface_type == 0)
    - 確信度 >= 0.70
    - 単勝オッズ >= 6.0倍 かつ ガチ確率 (prob_gachi) <= 0.10 (荒れる波乱レース)
    - 単勝500円 ＋ 複勝500円 (計1,000円ベース、追い風ブースト時 1.5倍)
    """

    @property
    def strategy_type(self) -> str:
        return "strategy_a_hole"

    @property
    def priority(self) -> int:
        return 4

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
        w0_odds = float(honmei_horse.get("odds") or 50.0)
        return (surface_type == 0) and (confidence_score >= 0.70) and (w0_odds >= 6.0) and (prob_gachi <= 0.10)

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

        h_tan_amt = int(round(500 * bias_boost_mult / 100.0) * 100)
        h_fuku_amt = int(round(500 * bias_boost_mult / 100.0) * 100)

        strategy_name = f"⚡ 戦略A-2: 芝波乱妙味 (単{h_tan_amt}円＋複{h_fuku_amt}円)"
        real_bet_badge = f"⚡【芝波乱妙味】単複好配当狙い{bias_tag}{suffix}"

        tickets: List[BetTicket] = []

        # 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=h_tan_amt,
            odds=w0_odds,
            expected_payout=int(round(h_tan_amt * w0_odds)),
            is_real_odds=True,
            memo=f"単勝(妙味狙い) {w0_u}番 ({w0_name}){bias_tag} {h_tan_amt:,}円"
        ))

        # 複勝
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        tickets.append(BetTicket(
            ticket_type="複勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=h_fuku_amt,
            odds=est_fuku_odds,
            expected_payout=int(round(h_fuku_amt * est_fuku_odds)),
            is_real_odds=False,
            memo=f"複勝(妙味狙い) {w0_u}番 ({w0_name}){bias_tag} {h_fuku_amt:,}円"
        ))

        total_amount = sum(t.amount for t in tickets)
        summary = f"{strategy_name} [単勝{h_tan_amt:,}円 ＋ 複勝{h_fuku_amt:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="A_hole",
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
            reason="戦略A-2採択 (芝波乱妙味)"
        )

