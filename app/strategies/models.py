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
    """戦略適用判定結果 (3段階レース区分: 勝負R / 実勝負 / 仮想 + 見送り)"""
    is_participate: bool = False
    is_real_bet_target: bool = False          # 実投票対象フラグ (勝負R または 実勝負)
    is_virtual_bet: bool = False              # 仮想観戦枠フラグ (投票なし・総合台帳集計あり)
    bet_category: str = "skip"                # "super" (勝負R), "real" (実勝負), "virtual" (仮想), "skip" (見送り)
    assigned_tier: Optional[str] = None       # "SUPER", "REAL", "VIRTUAL", None
    strategy_type: str = "skip"               # "super_bet", "real_bet", "virtual_bet", "skip"
    strategy_name: str = "見送り推奨 (資金保全)"
    real_bet_badge: Optional[str] = None      # 画面表示用バッジ名
    badge_text: str = "見送り推奨"            # 表示テキスト
    real_bet_level: str = "none"              # "high", "standard", "virtual", "none"
    skip_reason: Optional[str] = None
    tickets: List[BetTicket] = Field(default_factory=list)
    total_amount: int = 0
    confidence_score: float = 0.0
    edge_ratio: float = 0.0
    synth_odds: float = 0.0
    is_final_odds: bool = False
    summary: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_participate": self.is_participate,
            "is_real_bet_target": self.is_real_bet_target,
            "is_virtual_bet": self.is_virtual_bet,
            "bet_category": self.bet_category,
            "assigned_tier": self.assigned_tier,
            "strategy_type": self.strategy_type,
            "strategy_name": self.strategy_name,
            "real_bet_badge": self.real_bet_badge,
            "badge_text": self.badge_text,
            "real_bet_level": self.real_bet_level,
            "skip_reason": self.skip_reason,
            "tickets": [t.to_dict() for t in self.tickets],
            "total_amount": self.total_amount,
            "confidence_score": self.confidence_score,
            "edge_ratio": self.edge_ratio,
            "synth_odds": self.synth_odds,
            "is_final_odds": self.is_final_odds,
            "summary": self.summary,
            "reason": self.reason,
        }

