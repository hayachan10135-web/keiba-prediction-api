# -*- coding: utf-8 -*-
"""
app/strategies/strategy_real.py

🎯【実勝負】戦略: 資金基盤・的中率＆回収率両立
- 出現頻度: 1日・各場1R以上保証 (上限なし・1日3〜5R)
- 条件: 各場最高安定度レース、または (Conf >= 0.55 & EV >= 1.00 & 1.8 <= Odds <= 20.0)
- 券種: 単複 3:7 配分 (単勝300円 ＋ 複勝700円)
- 2026年実績: 的中率 45.2% / 回収率安定基盤
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyReal(BaseBettingStrategy):
    """🎯【実勝負】安定的中・回収基盤戦略"""

    @property
    def strategy_type(self) -> str:
        return "real_bet"

    @property
    def priority(self) -> int:
        return 2  # 勝負Rの次

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
        if race_info.get("is_real_guarantee", False):
            return True

        w0_odds = float(honmei_horse.get("odds") or 50.0)
        # 自然合致条件: 信頼度0.55以上 × 期待値1.00以上 × オッズ1.8〜20.0倍 × 頭数6頭以上
        has_enough = len(horses) >= 6
        is_odds_fit = (1.8 <= w0_odds <= 20.0)
        is_conf_fit = (confidence_score >= 0.55)
        is_edge_fit = (win_edge >= 1.00)

        return has_enough and is_odds_fit and is_conf_fit and is_edge_fit

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

        # 単複配分: 単勝300円 + 複勝700円 (オッズ3.5倍以上の場合は 単500円+複500円)
        if w0_odds >= 3.5:
            tan_amt = 500
            fuku_amt = 500
        else:
            tan_amt = 300
            fuku_amt = 700

        tickets: List[BetTicket] = []

        # 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=tan_amt,
            odds=w0_odds,
            expected_payout=int(round(tan_amt * w0_odds)),
            is_real_odds=True,
            memo=f"単勝 {w0_u}番 ({w0_name}){bias_tag} {tan_amt:,}円"
        ))

        # 複勝 (概算オッズ)
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        tickets.append(BetTicket(
            ticket_type="複勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=fuku_amt,
            odds=est_fuku_odds,
            expected_payout=int(round(fuku_amt * est_fuku_odds)),
            is_real_odds=False,
            memo=f"複勝 {w0_u}番 ({w0_name}){bias_tag} {fuku_amt:,}円"
        ))

        total_amount = sum(t.amount for t in tickets)
        strategy_name = f"🎯【実勝負】本命基盤投資 (単{tan_amt}円＋複{fuku_amt}円)"
        real_bet_badge = f"🎯【実勝負】単複{bias_tag}{suffix}"
        summary = f"{strategy_name} [軸{w0_u} {w0_name} / 計{total_amount:,}円]"

        reason_text = "【実勝負】会場最低1R保証枠" if race_info.get("is_real_guarantee") else f"【実勝負】採択 (Conf {confidence_score:.2f} >= 0.55 & EV {win_edge:.2f} >= 1.00)"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            is_virtual_bet=False,
            bet_category="real",
            assigned_tier="REAL",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            badge_text="🎯【実勝負】",
            real_bet_level="standard",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason=reason_text
        )
