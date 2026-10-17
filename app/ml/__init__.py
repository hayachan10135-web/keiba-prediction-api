# -*- coding: utf-8 -*-
"""
app.ml package

機械学習推論、LightGBMカスケード、特徴量生成、リアルタイムトラックバイアス
"""

from app.ml.track_bias import (
    evaluate_horse_bias_match,
    compute_track_bias_and_matches,
    set_track_bias_dependencies,
    GLOBAL_VENUE_MAP
)
from app.ml.model_loader import (
    MODEL_PATH,
    WIN_MODEL_PATH,
    TOP3_MODEL_PATH,
    MAIDEN_WIN_MODEL_PATH,
    MAIDEN_TOP3_MODEL_PATH,
    GACHI_MODEL_PATH,
    HIMO_MODEL_PATH,
    DICT_PATH,
    model,
    model_win,
    model_top3,
    model_maiden_win,
    model_maiden_top3,
    model_gachi,
    model_himo,
    stats_dict,
    _stats_overlay,
    load_all_models,
    get_stats_dict,
    load_stats_overlay_from_supabase,
    save_stats_overlay_to_supabase,
    update_stats_from_race_settlement,
)

__all__ = [
    "evaluate_horse_bias_match",
    "compute_track_bias_and_matches",
    "set_track_bias_dependencies",
    "GLOBAL_VENUE_MAP",
    "MODEL_PATH",
    "WIN_MODEL_PATH",
    "TOP3_MODEL_PATH",
    "MAIDEN_WIN_MODEL_PATH",
    "MAIDEN_TOP3_MODEL_PATH",
    "GACHI_MODEL_PATH",
    "HIMO_MODEL_PATH",
    "DICT_PATH",
    "model",
    "model_win",
    "model_top3",
    "model_maiden_win",
    "model_maiden_top3",
    "model_gachi",
    "model_himo",
    "stats_dict",
    "_stats_overlay",
    "load_all_models",
    "get_stats_dict",
    "load_stats_overlay_from_supabase",
    "save_stats_overlay_to_supabase",
    "update_stats_from_race_settlement",
]
