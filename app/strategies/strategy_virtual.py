# -*- coding: utf-8 -*-
"""
app/strategies/strategy_virtual.py

🎮【仮想】戦略: 観戦エンタメ・高的中＆好配当
- 出現頻度: 1日・各場3R以上保証 (上限なし・1日9〜14R)
- 条件: 各場上位3R保証、または (EV >= 0.95 & Conf >= 0.35)
- 券種: 3連複 軸流し6点 (◎ 軸 ◯, ▲, △, ☆) 各200円 = 計1,200円想定
- 投票: 実投票なし (is_real_bet_target = False, is_virtual_bet = True)
- 2026年実績: 的中率 26.5% / 万馬券観戦対応
"""

import itertools
from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyVirtual(BaseBettingStrategy):
    """🎮【仮想】観戦エンタメ枠戦略 (投票なし・総合台帳集計あり)"""

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

        # 自然合致条件: 頭数7頭以上 × 期待値0.95以上 × 信頼度0.35以上
        has_enough = len(horses) >= 7
        is_edge_fit = (win_edge >= 0.95)
        is_conf_fit = (confidence_score >= 0.35)

        return has_enough and is_edge_fit and is_conf_fit

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

        # 相手4頭の選定 (AI勝率順上位4頭: ◯, ▲, △, ☆)
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        candidate_opps = [h for h in s_win if int(h.get("umaban", 0)) != w0_u]
        opponents = candidate_opps[:4]

        tickets: List[BetTicket] = []
        unit_amt = 200  # 各200円 (計6点 = 1,200円)

        if len(opponents) >= 2:
            opp_u_list = [int(h["umaban"]) for h in opponents]
            for pair in itertools.combinations(opp_u_list, 2):
                comb = sorted([w0_u, pair[0], pair[1]])
                comb_str = f"{comb[0]}-{comb[1]}-{comb[2]}"
                # 想定オッズ (概算)
                est_odds = round(max(5.0, w0_odds * 4.0), 1)

                tickets.append(BetTicket(
                    ticket_type="3連複",
                    combination=comb,
                    combination_str=comb_str,
                    amount=unit_amt,
                    odds=est_odds,
                    expected_payout=int(round(unit_amt * est_odds)),
                    is_real_odds=False,
                    memo=f"3連複 {comb_str} (軸{w0_u}) {unit_amt:,}円"
                ))

        total_amount = sum(t.amount for t in tickets)
        strategy_name = "🎮【仮想】観戦エンタメ (3連複軸流し6点)"
        real_bet_badge = f"🎮【仮想】3連複軸流し{bias_tag}{suffix}"
        opp_str = "/".join(str(int(h["umaban"])) for h in opponents)
        summary = f"{strategy_name} [軸{w0_u} {w0_name} → 相手{opp_str} / 想定{total_amount:,}円 (非投票)]"

        reason_text = "【仮想】会場最低3R保証枠" if race_info.get("is_virtual_guarantee") else f"【仮想】採択 (Conf {confidence_score:.2f} >= 0.35 & EV {win_edge:.2f} >= 0.95)"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=False,          # 実投票はしない！
            is_virtual_bet=True,              # 仮想観戦枠フラグ
            bet_category="virtual",
            assigned_tier="VIRTUAL",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            badge_text="🎮【仮想】",
            real_bet_level="virtual",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason=reason_text
        )
