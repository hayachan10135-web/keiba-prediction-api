# -*- coding: utf-8 -*-
"""
app/strategies/strategy_b.py

戦略B: 超高確信度馬単レバレッジ (単勝 ＋ 馬単3点)
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategyB(BaseBettingStrategy):
    """
    【戦略B】超高確信度馬単レバレッジ
    - 芝・ダート不問
    - 確信度 >= 0.78 かつ 本命オッズ 2.5〜10.0倍
    - 単勝500円 ＋ 馬単◎→相手3点 各200円 (追い風ブースト時 1.5倍)
    - ヒモ穴候補 (妙味穴/連続急落/絶好穴) の優先昇格
    """

    @property
    def strategy_type(self) -> str:
        return "strategy_b"

    @property
    def priority(self) -> int:
        return 1

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
        return confidence_score >= 0.78 and (2.5 <= w0_odds <= 10.0)

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

        b_tan_amt = int(round(500 * bias_boost_mult / 100.0) * 100)
        b_utan_amt = int(round(200 * bias_boost_mult / 100.0) * 100)

        strategy_name = f"🔥 戦略B: 超高確信度馬単レバレッジ (単{b_tan_amt}円＋馬単各{b_utan_amt}円)"
        real_bet_badge = f"🔥【超勝負R】単勝＋馬単レバレッジ{bias_tag}{suffix}"

        tickets: List[BetTicket] = []

        # 1. 単勝
        tickets.append(BetTicket(
            ticket_type="単勝",
            combination=[w0_u],
            combination_str=f"{w0_u}",
            amount=b_tan_amt,
            odds=w0_odds,
            expected_payout=int(round(b_tan_amt * w0_odds)),
            is_real_odds=True,
            memo=f"単勝 {w0_u}番 ({w0_name}){bias_tag} {b_tan_amt}円"
        ))

        # 2. 馬単 相手3頭の選定
        # (1) 過剰人気馬(overbet)の排除
        # (2) 逆風馬(adverse)のヒモ除外
        # (3) 妙味穴(value_gap)、連続急落穴、絶好バイアス穴(4.0〜50倍)の優先昇格
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        candidate_opps = [h for h in s_win if int(h.get("umaban", 0)) != w0_u]
        non_overbet_opps = [h for h in candidate_opps if h.get("odds_trend") != "overbet"]
        base_pool = non_overbet_opps if len(non_overbet_opps) >= 3 else candidate_opps

        non_adverse_opps = [h for h in base_pool if "逆風" not in str(h.get("bias_match") or "")]
        candidate_pool = non_adverse_opps if len(non_adverse_opps) >= 3 else base_pool

        boosted_opps = []
        normal_opps = []
        for h in candidate_pool:
            o_val = float(h.get("odds") or 0.0)
            is_val_gap = (h.get("odds_trend") == "value_gap")
            alert_txt = str(h.get("odds_alert") or "")
            b_match = str(h.get("bias_match") or "")
            is_cont_plunge = ("連続急落" in alert_txt) and (3.5 <= o_val <= 35.0)
            is_bias_sweet = ("絶好バイアス" in b_match) and (4.0 <= o_val <= 50.0)

            if is_val_gap or is_cont_plunge or is_bias_sweet:
                if is_val_gap:
                    tag_reason = "💎妙味穴"
                elif is_bias_sweet:
                    tag_reason = "🎯絶好穴"
                else:
                    tag_reason = "🔥連続急落"
                boosted_opps.append((h, tag_reason))
            else:
                normal_opps.append(h)

        selected_opps = []
        selected_u_set = set()
        for h, reason in boosted_opps[:2]:
            u_n = int(h["umaban"])
            selected_opps.append((h, reason))
            selected_u_set.add(u_n)

        for h in normal_opps:
            u_n = int(h["umaban"])
            if u_n not in selected_u_set:
                selected_opps.append((h, "AI本線"))
                selected_u_set.add(u_n)
            if len(selected_opps) >= 3:
                break

        for h in candidate_pool:
            u_n = int(h["umaban"])
            if u_n not in selected_u_set:
                selected_opps.append((h, "AI本線"))
                selected_u_set.add(u_n)
            if len(selected_opps) >= 3:
                break

        for h_opp, tag in selected_opps[:3]:
            opp = int(h_opp["umaban"])
            opp_name = str(h_opp.get("horse_name", f"馬{opp}"))
            opp_odds = float(h_opp.get("odds") or 10.0)
            est_utan_odds = round(max(3.0, w0_odds * opp_odds * 0.8), 1)
            tag_str = f" [{tag}]" if tag != "AI本線" else ""
            tickets.append(BetTicket(
                ticket_type="馬単",
                combination=[w0_u, opp],
                combination_str=f"{w0_u}→{opp}",
                amount=b_utan_amt,
                odds=est_utan_odds,
                expected_payout=int(round(b_utan_amt * est_utan_odds)),
                is_real_odds=False,
                memo=f"馬単 軸{w0_u}→{opp}番 ({opp_name}){tag_str} {b_utan_amt}円"
            ))

        total_amount = sum(t.amount for t in tickets)
        utan_count = len(tickets) - 1
        summary = f"{strategy_name} [単勝{w0_u} {b_tan_amt:,}円 ＋ 馬単{utan_count}点 各{b_utan_amt:,}円 / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            assigned_tier="B",
            strategy_type=self.strategy_type,
            strategy_name=strategy_name,
            real_bet_badge=real_bet_badge,
            real_bet_level="high",
            tickets=tickets,
            total_amount=total_amount,
            confidence_score=confidence_score,
            edge_ratio=round(win_edge, 2),
            synth_odds=round(w0_odds, 1),
            summary=summary,
            reason="戦略B採択 (超高確信度馬単レバレッジ)"
        )

