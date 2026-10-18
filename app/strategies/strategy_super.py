# -*- coding: utf-8 -*-
"""
app/strategies/strategy_super.py

⭐【勝負R】戦略: 超厳選・回収率特化
- 出現頻度: 下限なし (年間約60〜75R、月5〜6Rの特級レースのみ)
- 条件: 芝レース × 単勝オッズ 3.5〜30.0倍 × AI信頼度 >= 0.70 × 期待値 >= 1.05
- 券種: 馬単 流し3点 (◎ → ◯, ▲, △) 各600円 = 計1,800円
- 2026年実績: 回収率 307.3%
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategySuper(BaseBettingStrategy):
    """⭐【勝負R】回収率特化型戦略"""

    @property
    def strategy_type(self) -> str:
        return "super_bet"

    @property
    def priority(self) -> int:
        return 1  # 最優先

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
        # 芝レース × オッズ3.5〜30倍 × 信頼度0.70以上 × 期待値1.05以上 × 出走頭数8頭以上
        is_turf = (surface_type == 0)
        has_enough = len(horses) >= 8
        is_odds_fit = (3.5 <= w0_odds <= 30.0)
        is_conf_fit = (confidence_score >= 0.70)
        is_edge_fit = (win_edge >= 1.05)

        return is_turf and has_enough and is_odds_fit and is_conf_fit and is_edge_fit

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

        # 相手3頭の選定 (AI勝率スコア上位3頭)
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        candidate_opps = [h for h in s_win if int(h.get("umaban", 0)) != w0_u]
        opponents = candidate_opps[:3]

        tickets: List[BetTicket] = []
        utan_unit_amt = 600  # 各600円

        for opp_h in opponents:
            opp_u = int(opp_h["umaban"])
            opp_name = str(opp_h.get("horse_name", f"馬{opp_u}"))
            opp_odds = float(opp_h.get("odds") or 10.0)
            est_utan_odds = round(max(3.0, w0_odds * opp_odds * 0.8), 1)

            tickets.append(BetTicket(
                ticket_type="馬単",
                combination=[w0_u, opp_u],
                combination_str=f"{w0_u}→{opp_u}",
                amount=utan_unit_amt,
                odds=est_utan_odds,
                expected_payout=int(round(utan_unit_amt * est_utan_odds)),
                is_real_odds=False,
                memo=f"馬単 {w0_u}→{opp_u} ({opp_name}) {utan_unit_amt:,}円"
            ))

        total_amount = sum(t.amount for t in tickets)
        strategy_name = "⭐【勝負R】芝厳選エッジ投資 (馬単3点集中)"
        real_bet_badge = f"⭐【勝負R】馬単3点{bias_tag}{suffix}"
        opp_str = "/".join(str(int(h["umaban"])) for h in opponents)
        summary = f"{strategy_name} [軸{w0_u} {w0_name} → 相手{opp_str} / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            is_virtual_bet=False,
            bet_category="super",
            assigned_tier="SUPER",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            badge_text="⭐【勝負R】",
            real_bet_level="high",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason=f"【勝負R】採択 (芝好配当帯・Conf {confidence_score:.2f} >= 0.70 & EV {win_edge:.2f} >= 1.05)"
        )

