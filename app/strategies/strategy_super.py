# -*- coding: utf-8 -*-
"""
app/strategies/strategy_super.py

⭐【勝負R】戦略: 超厳選・回収率特化 (アウトライアー非依存)
- 出現頻度: 下限なし (年間約100〜110R、月8〜9Rの特級レースのみ)
- 条件: 1・2勝クラス × 単勝オッズ 3.0〜7.0倍 × 適正EV 1.02〜1.25 × 確信度差 >= 0.02
- 券種: 単勝 1点 (◎)
- 掛け金: EV連動3段階
    - Sランク (EV >= 1.15): 5,000円
    - Aランク (1.08 <= EV < 1.15): 3,000円
    - Bランク (EV < 1.08): 2,000円
- 2026年実績: 的中率 28.8% / 通常ROI 130.3% / Top3除外ROI 110.6%
"""

from typing import Dict, Any, List
from app.strategies.base import BaseBettingStrategy
from app.strategies.models import StrategyDecision, BetTicket


class StrategySuper(BaseBettingStrategy):
    """⭐【勝負R】回収率特化型戦略 (アウトライアー非依存・単勝EV連動)"""

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
        # 頭数6頭以上
        if len(horses) < 6:
            return False

        # 障害戦除外
        if surface_type == 2:
            return False

        w0_odds = float(honmei_horse.get("odds") or 50.0)

        # レースクラス判定 (class_group: B_1_2_Win を中心とする)
        # race_info または honmei_horse 内の race_class_code または class_group を参照
        race_class_code = race_info.get("race_class_code", honmei_horse.get("race_class_code", 2))
        is_class_b = (race_class_code in (2, 3)) or (race_info.get("class_group") == "B_1_2_Win")

        # 適正オッズ帯: 3.0〜7.0倍 (中穴・適正人気)
        is_odds_fit = (3.0 <= w0_odds <= 7.0)

        # 適正EV帯: 1.02〜1.25 (超高EVの大穴過大評価ノイズを物理排除)
        is_edge_fit = (1.02 <= win_edge <= 1.25)

        # 確信度差: 勝率1位と2位の差 >= 0.02
        s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
        conf_margin = float(s_win[0].get("prob_win", 0.0) - (s_win[1].get("prob_win", 0.0) if len(s_win) > 1 else 0.0))
        is_conf_fit = (conf_margin >= 0.02)

        return is_class_b and is_odds_fit and is_edge_fit and is_conf_fit

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

        # EV連動3段階ステーキング
        if win_edge >= 1.15:
            bet_amount = 5000
            rank_label = "S"
        elif win_edge >= 1.08:
            bet_amount = 3000
            rank_label = "A"
        else:
            bet_amount = 2000
            rank_label = "B"

        tickets: List[BetTicket] = [
            BetTicket(
                ticket_type="単勝",
                combination=[w0_u],
                combination_str=f"{w0_u}",
                amount=bet_amount,
                odds=w0_odds,
                expected_payout=int(round(bet_amount * w0_odds)),
                is_real_odds=True,
                memo=f"単勝 {w0_u}番 ({w0_name}){bias_tag} {bet_amount:,}円"
            )
        ]

        total_amount = bet_amount
        strategy_name = f"⭐【勝負R ({rank_label})】中穴エッジ投資 (単勝1点)"
        real_bet_badge = f"⭐【勝負R ({rank_label})】単勝{bias_tag}{suffix}"
        summary = f"{strategy_name} [単勝{w0_u} {w0_name} ({w0_odds:.1f}倍) / 計{total_amount:,}円]"

        return StrategyDecision(
            is_participate=True,
            is_real_bet_target=True,
            is_virtual_bet=False,
            bet_category="super",
            assigned_tier="SUPER",
            strategy_name=strategy_name,
            tickets=tickets,
            total_amount=total_amount,
            target_umaban=w0_u,
            target_horse_name=w0_name,
            real_bet_badge=real_bet_badge,
            summary_text=summary,
            reason=(
                f"【勝負R({rank_label})選定】1・2勝クラス適正EVゾーン "
                f"(EV {win_edge:.2f} ∈ [1.02, 1.25] & 単勝 {w0_odds:.1f}倍 ∈ [3.0, 7.0] & 投資 {bet_amount:,}円)"
            )
        )
