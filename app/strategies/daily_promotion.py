# -*- coding: utf-8 -*-
"""
app/strategies/daily_promotion.py

本日厳選No.1: デイリー昇格枠 (自然合致0件時専用フォールバック)
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class DailyPromotionStrategy(BaseBettingStrategy):
    """
    【本日厳選No.1】デイリー昇格枠
    - 自然合致レース (B/A/C/A-2) が当日0件の時のみ選定
    - 当日全レースの中で「確信度 × エッジ」最高スコアのレースを実戦昇格
    - 馬場・オッズ適性に応じた単勝＋複勝の配分 (計1,000円)
    """

    @property
    def strategy_type(self) -> str:
        return "daily_top"

    @property
    def priority(self) -> int:
        return 99  # フォールバック枠

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
        # StrategyArbiter によって明示的に指定された場合のみ True
        return bool(race_info.get("is_daily_top_race", False))

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
        surface_type = int(race_info.get("surface_type", 0))
        prob_gachi = float(race_info.get("prob_gachi", 0.0))

        if surface_type == 1:
            d_tan, d_fuku = 700, 300
            strategy_name = f"⭐ 本日厳選No.1: ダート適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】ダート適性単複{bias_tag}{suffix}"
        elif w0_odds >= 5.0 or prob_gachi <= 0.15:
            d_tan, d_fuku = 500, 500
            strategy_name = f"⭐ 本日厳選No.1: 芝妙味適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】芝妙味単複{bias_tag}{suffix}"
        else:
            d_tan, d_fuku = 700, 300
            strategy_name = f"⭐ 本日厳選No.1: 芝本命適性単複 (単{d_tan}円＋複{d_fuku}円)"
            real_bet_badge = f"⭐【本日厳選No.1】芝本命単複{bias_tag}{suffix}"

        tickets: List[BetTicket] = []

        # 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=d_tan,
            odds=w0_odds,
            expected_payout=int(round(d_tan * w0_odds)),
            is_real_odds=True,
            memo=f"単勝(デイリー厳選) {w0_u}番 ({w0_name}){bias_tag} {d_tan:,}円"
        ))

        # 複勝
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        tickets.append(BetTicket(
            ticket_type="複勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=d_fuku,
            odds=est_fuku_odds,
            expected_payout=int(round(d_fuku * est_fuku_odds)),
            is_real_odds=False,
            memo=f"複勝(デイリー厳選) {w0_u}番 ({w0_name}){bias_tag} {d_fuku:,}円"
        ))

        total_amount = sum(t.amount for t in tickets)
        summary = f"{strategy_name} [単勝{d_tan:,}円 ＋ 複勝{d_fuku:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="Daily_No1",
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
            reason="本日厳選No.1昇格枠採択 (自然合致0件時リカバリ)"
        )

