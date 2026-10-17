# -*- coding: utf-8 -*-
"""
app.maintenance package

過去開催データ一括再計算、自己修復、スキーマ移行、日付正規化エンジン
"""

from app.maintenance.recalculator import (
    recalculate_period_records,
    set_recalculator_dependencies,
)
from app.maintenance.healer import (
    CURRENT_DB_VERSION,
    fix_mislabeled_20261011_races,
    clear_all_database,
    check_and_apply_db_migration,
    normalize_mislabeled_dates,
    sync_and_process_target_dates,
    fix_duplicated_horses_cache_2026,
    heal_today_weights_and_results,
    heal_settlement_marks,
    set_healer_dependencies,
)

__all__ = [
    "recalculate_period_records",
    "set_recalculator_dependencies",
    "CURRENT_DB_VERSION",
    "fix_mislabeled_20261011_races",
    "clear_all_database",
    "check_and_apply_db_migration",
    "normalize_mislabeled_dates",
    "sync_and_process_target_dates",
    "fix_duplicated_horses_cache_2026",
    "heal_today_weights_and_results",
    "heal_settlement_marks",
    "set_healer_dependencies",
]
