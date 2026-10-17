# -*- coding: utf-8 -*-
"""
app/strategies/models.py

買い目チケットおよび戦略決定データクラス
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class BetTicket(BaseModel):
    """個別買い目チケット"""
    ticket_type: str                  # 単勝, 複勝, 馬単, 馬連, 3連複, 3連単
    combination: List[int]            # 馬番リスト [7] または [7, 3]
    combination_str: str              # "7" または "7→3"
    amount: int                       # 投資金額 (円)
    odds: float                       # オッズ (単勝/馬単オッズ または 想定オッズ)
    expected_payout: int              # 想定払戻金額 (円)
    is_real_odds: bool = False        # 確定リアルオッズに基づくか
    memo: str = ""                    # 表示用メモ

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_type": self.ticket_type,
            "combination": self.combination,
            "combination_str": self.combination_str,
            "amount": self.amount,
            "odds": self.odds,
            "expected_payout": self.expected_payout,
            "is_real_odds": self.is_real_odds,
            "memo": self.memo,
        }


class StrategyDecision(BaseModel):
    """戦略適用判定結果"""
    is_participate: bool = False
    is_real_bet_target: bool = False
    assigned_tier: Optional[str] = None       # "B", "A", "C", "A_hole", "Daily_No1", None
    strategy_type: str = "skip"               # "strategy_b", "strategy_a", "strategy_c", "strategy_a_hole", "daily_top", "skip"
    strategy_name: str = "見送り推奨 (資金保全)"
    real_bet_badge: Optional[str] = None
    real_bet_level: str = "none"              # "high", "standard", "none"
    skip_reason: Optional[str] = None
    tickets: List[BetTicket] = Field(default_factory=list)
    total_amount: int = 0
    confidence_score: float = 0.0
    edge_ratio: float = 0.0
    synth_odds: float = 0.0
    is_final_odds: bool = False
    summary: str = ""
    reason: str = ""

