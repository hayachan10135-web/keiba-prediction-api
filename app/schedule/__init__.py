# -*- coding: utf-8 -*-
"""
app/schedule/__init__.py

レース開催日程・スケジュール・会場管理パッケージ
"""

from app.schedule.period import (
    get_current_kaisai_period,
    clear_kaisai_period_caches,
    discover_period_kaisai_dates,
    _period_venues_cache,
    _period_checked_empty_dates,
    _latest_post_times,
    _last_weight_check_ts,
    _period_kaisai_dates_cache,
)

from app.schedule.races import (
    set_races_dependencies,
    register_race_schedule,
    ensure_schedules_registered_for_venues,
    attach_prediction_badges_to_venues,
    get_today_races,
    _today_races_cache,
)

__all__ = [
    "get_current_kaisai_period",
    "clear_kaisai_period_caches",
    "discover_period_kaisai_dates",
    "_period_venues_cache",
    "_period_checked_empty_dates",
    "_latest_post_times",
    "_last_weight_check_ts",
    "_period_kaisai_dates_cache",
    "set_races_dependencies",
    "register_race_schedule",
    "ensure_schedules_registered_for_venues",
    "attach_prediction_badges_to_venues",
    "get_today_races",
    "_today_races_cache",
]

