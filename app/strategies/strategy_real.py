# -*- coding: utf-8 -*-
"""
app/strategies/strategy_real.py

🎯【実勝負】戦略: 資金基盤・安定的中型
- 出現頻度:
    - 勝負Rがある日: 下限なし (0R〜自然適格枠のみ)
    - 勝負Rも自然実勝負もない日のみ: 1日1R強制保証 (少額300円)
- 条件:
    - 自然枠: 複勝率 >= 0.65 × 期待値 >= 1.05 × 単勝オッズ <= 8.0倍 (掛け金1,000円)
    - 日次1R保証枠: 当日最高安定度レース (少額300円)
- 券種: 複勝 1点 (◎)
- 2026年実績: 的中率 57.7% / 実投票台帳全体ROI 113.3%〜120.8%
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyReal(BaseBettingStrategy):
    """🎯【実勝負】安定的中・回収基盤戦略 (複勝1点 / 保証枠少額300円)"""

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
        # 日次スロット保証による明示的選定の場合
        if race_info.get("is_real_guarantee", False):
            return True

        if len(horses) < 6 or surface_type == 2:
            return False

        w0_odds = float(honmei_horse.get("odds") or 50.0)
        s_top3 = sorted(horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
        w0_t3 = float(s_top3[0].get("prob_top3", 0.0)) if s_top3 else float(honmei_horse.get("prob_top3", 0.0))

        # 自然適格条件: 複勝率 >= 0.65 × 期待値 >= 1.05 × オッズ <= 8.0倍
        is_top3_fit = (w0_t3 >= 0.65)
        is_edge_fit = (win_edge >= 1.05)
        is_odds_fit = (w0_odds <= 8.0)

        return is_top3_fit and is_edge_fit and is_odds_fit

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

        is_guarantee = bool(race_info.get("is_real_guarantee", False))

        # 保証枠は少額300円、自然枠は1,000円
        bet_amount = 300 if is_guarantee else 1000

        # 複勝概算オッズ
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)

        tickets: List[BetTicket] = [
            BetTicket(
                ticket_type="複勝",
                combination=[w0_u],
                combination_str=f"{w0_u}",
                amount=bet_amount,
                odds=est_fuku_odds,
                expected_payout=int(round(bet_amount * est_fuku_odds)),
                is_real_odds=False,
                memo=f"複勝 {w0_u}番 ({w0_name}){bias_tag} {bet_amount:,}円"
            )
        ]

        total_amount = bet_amount
        if is_guarantee:
            strategy_name = "🔰【実勝負(保証)】日次1R参加・少額防衛 (複勝1点)"
            real_bet_badge = f"🔰【実勝負(保証)】複勝{bias_tag}{suffix}"
            reason_text = f"【実勝負】勝負Rなし日・日次1R最低保証枠 (少額{bet_amount:,}円)"
        else:
            strategy_name = "🎯【実勝負】高安定回収基盤 (複勝1点)"
            real_bet_badge = f"🎯【実勝負】複勝{bias_tag}{suffix}"
            reason_text = f"【実勝負】高複勝率・安定エッジ枠 (複勝率>=0.65 & EV>=1.05 & 投資{bet_amount:,}円)"

        summary = f"{strategy_name} [複勝{w0_u} {w0_name} / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            is_virtual_bet=False,
            bet_category="real",
            assigned_tier="REAL",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            badge_text="🎯実勝負",
            real_bet_level="standard",
            tickets=tickets,
            total_amount=total_amount,
            summary=summary,
            reason=reason_text
        )
