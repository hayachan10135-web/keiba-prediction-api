# -*- coding: utf-8 -*-
"""
app/strategies/base.py

戦略抽象基底クラス (BaseBettingStrategy)
すべての買い目戦略 (Strategy B, A, C, A-2, DailyPromotion 等) はこのインターフェースを実装する
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.strategies.models import StrategyDecision, BetTicket


class BaseBettingStrategy(ABC):
    """買い目戦略基底クラス"""

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """戦略の一意識別子 (例: 'strategy_b', 'strategy_a')"""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """カスケード評価優先度 (小さいほど優先)"""
        pass

    @abstractmethod
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
        """
        当該レースがこの戦略の出撃条件に適合するか判定
        """
        pass

    @abstractmethod
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
        """
        買い目を生成して判定結果を返却
        """
        pass

