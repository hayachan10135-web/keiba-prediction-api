# -*- coding: utf-8 -*-
"""
app.scrapers package

Netkeiba クローラー、オッズ取得、出馬表・馬体重取得、確定払戻取得、発走時刻変更検知
耐障害リトライ (指数バックオフ) を標準装備
"""

from app.scrapers.netkeiba_client import (
    safe_scrape_get,
    safe_decode_html,
    fetch_real_odds,
    fetch_shutuba_table,
    fetch_netkeiba_race_result,
    normalize_race_res,
    register_post_time_callback,
    register_schedule_callback,
    USER_AGENTS,
    HEADERS_PC,
    is_waf_in_cooldown,
    get_waf_cooldown_remaining,
    trigger_waf_cooldown,
    reset_waf_cooldown
)
from app.scrapers.schedule_crawler import (
    parse_netkeiba_race_list_html,
    fetch_venues_for_date_raw,
    fetch_venues_from_database,
    invalidate_venues_cache,
    register_crawler_post_time_handler,
    register_venues_loaded_handler,
    VENUE_CODE_MAP
)

__all__ = [
    "safe_scrape_get",
    "safe_decode_html",
    "fetch_real_odds",
    "fetch_shutuba_table",
    "fetch_netkeiba_race_result",
    "normalize_race_res",
    "register_post_time_callback",
    "register_schedule_callback",
    "USER_AGENTS",
    "HEADERS_PC",
    "parse_netkeiba_race_list_html",
    "fetch_venues_for_date_raw",
    "fetch_venues_from_database",
    "invalidate_venues_cache",
    "register_crawler_post_time_handler",
    "register_venues_loaded_handler",
    "VENUE_CODE_MAP",
    "is_waf_in_cooldown",
    "get_waf_cooldown_remaining",
    "trigger_waf_cooldown",
    "reset_waf_cooldown"
]
