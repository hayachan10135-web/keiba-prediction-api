# -*- coding: utf-8 -*-
"""
app/strategies/strategy_c.py

戦略C: ダート手堅い本命防衛 (単勝＋複勝)
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyC(BaseBettingStrategy):
    """
    【戦略C】ダート手堅い本命防衛
    - ダートレース限定 (surface_type == 1)
    - 夏競馬除外 (is_summer == False)
    - 確信度 >= 0.72 かつ 本命オッズ 2.2〜6.0倍
    - 単勝700円 ＋ 複勝300円 (計1,000円ベース、追い風ブースト時 1.5倍)
    """

    @property
    def strategy_type(self) -> str:
        return "strategy_c"

    @property
    def priority(self) -> int:
        return 3

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
        return (surface_type == 1) and (not is_summer) and (confidence_score >= 0.72) and (2.2 <= w0_odds <= 6.0)

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

        c_tan_amt = int(round(700 * bias_boost_mult / 100.0) * 100)
        c_fuku_amt = int(round(300 * bias_boost_mult / 100.0) * 100)

        strategy_name = f"🛡️ 戦略C: ダート手堅い本命 (単{c_tan_amt}円＋複{c_fuku_amt}円)"
        real_bet_badge = f"🛡️【ダート本命】手堅い単複防衛{bias_tag}{suffix}"

        tickets: List[BetTicket] = []

        # 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=c_tan_amt,
            odds=w0_odds,
            expected_payout=int(round(c_tan_amt * w0_odds)),
            is_real_odds=True,
            memo=f"単勝(本命厚め) {w0_u}番 ({w0_name}){bias_tag} {c_tan_amt:,}円"
        ))

        # 複勝
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        tickets.append(BetTicket(
            ticket_type="複勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=c_fuku_amt,
            odds=est_fuku_odds,
            expected_payout=int(round(c_fuku_amt * est_fuku_odds)),
            is_real_odds=False,
            memo=f"複勝(防衛) {w0_u}番 ({w0_name}){bias_tag} {c_fuku_amt:,}円"
        ))

        total_amount = sum(t.amount for t in tickets)
        summary = f"{strategy_name} [単勝{c_tan_amt:,}円 ＋ 複勝{c_fuku_amt:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="C",
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
            reason="戦略C採択 (ダート手堅い本命単複)"
        )

