# -*- coding: utf-8 -*-
"""
app.strategies package

買い目戦略、調停オーケストレーターの疎結合レイヤ (Strategy Pattern)
"""

from app.strategies.models import StrategyDecision, BetTicket
from app.strategies.base import BaseBettingStrategy
from app.strategies.strategy_super import StrategySuper
from app.strategies.strategy_real import StrategyReal
from app.strategies.strategy_virtual import StrategyVirtual
from app.strategies.venue_guarantee import (
    resolve_venue_guarantees_for_date,
    get_race_guarantee_status,
    invalidate_venue_guarantee_cache,
)
from app.strategies.strategy_b import StrategyB
from app.strategies.strategy_a import StrategyA
from app.strategies.strategy_c import StrategyC
from app.strategies.strategy_a_hole import StrategyAHole
from app.strategies.daily_promotion import (
    DailyPromotionStrategy,
    NATURAL_REAL_STRATEGIES,
    get_or_select_daily_top_race_id,
    ensure_daily_single_real_bet,
    set_daily_promotion_dependencies,
    invalidate_daily_top_cache,
    _daily_top_race_id_cache
)
from app.strategies.arbiter import StrategyArbiter

__all__ = [
    "StrategyDecision",
    "BetTicket",
    "BaseBettingStrategy",
    "StrategySuper",
    "StrategyReal",
    "StrategyVirtual",
    "resolve_venue_guarantees_for_date",
    "get_race_guarantee_status",
    "invalidate_venue_guarantee_cache",
    "StrategyB",
    "StrategyA",
    "StrategyC",
    "StrategyAHole",
    "DailyPromotionStrategy",
    "NATURAL_REAL_STRATEGIES",
    "get_or_select_daily_top_race_id",
    "ensure_daily_single_real_bet",
    "set_daily_promotion_dependencies",
    "invalidate_daily_top_cache",
    "_daily_top_race_id_cache",
    "StrategyArbiter",
]
