# -*- coding: utf-8 -*-
"""
app/ml/model_loader.py

LightGBM 学習済みモデルおよび統計辞書 (stats_dict) のロード・管理・オンライン動的更新
"""

import os
import re
import json
import datetime
from typing import Dict, Any, Optional, List
import lightgbm as lgb

from app.core.database import get_supabase_client
from app.core.timezone import today_jst_str, now_jst

MODEL_PATH = "lgb_model.txt"
WIN_MODEL_PATH = "lgb_win_model.txt"
TOP3_MODEL_PATH = "lgb_top3_model.txt"
MAIDEN_WIN_MODEL_PATH = "lgb_maiden_win_model.txt"
MAIDEN_TOP3_MODEL_PATH = "lgb_maiden_top3_model.txt"
GACHI_MODEL_PATH = "lgb_gachi_model.txt"
HIMO_MODEL_PATH = "lgb_himo_model.txt"
CB_WIN_MODEL_PATH = "cb_win_model.cbm"
CB_TOP3_MODEL_PATH = "cb_top3_model.cbm"
XGB_WIN_MODEL_PATH = "xgb_win_model.json"
XGB_TOP3_MODEL_PATH = "xgb_top3_model.json"
ENSEMBLE_CONFIG_PATH = "ensemble_config.json"
DICT_PATH = "stats_dict.json"

# --- グローバルモデルインスタンス ---
model: Optional[lgb.Booster] = None
model_win: Optional[lgb.Booster] = None
model_top3: Optional[lgb.Booster] = None
model_maiden_win: Optional[lgb.Booster] = None
model_maiden_top3: Optional[lgb.Booster] = None
model_gachi: Optional[lgb.Booster] = None
model_himo: Optional[lgb.Booster] = None

# 3大GBDTアンサンブルモデル
model_cb_win: Optional[Any] = None
model_cb_top3: Optional[Any] = None
model_xgb_win: Optional[Any] = None
model_xgb_top3: Optional[Any] = None
ensemble_config: Optional[dict] = None

stats_dict: Dict[str, Any] = {
    "jockey_stats": {},
    "jockey_course_stats": {},
    "trainer_stats": {},
    "trainer_course_stats": {},
    "horse_stats": {}
}

_stats_overlay: Dict[str, Any] = {
    "updated_at": None,
    "horse_stats": {},
    "jockey_stats": {},
    "trainer_stats": {}
}


def load_all_models():
    """全LightGBMモデルおよび統計辞書をロード"""
    global model, model_win, model_top3, model_maiden_win, model_maiden_top3, model_gachi, model_himo, stats_dict

    if os.path.exists(MODEL_PATH):
        try:
            model = lgb.Booster(model_file=MODEL_PATH)
            print("現行LightGBMモデルを正常にロードしました。")
        except Exception as e:
            print(f"現行モデルロード失敗: {e}")

    if os.path.exists(WIN_MODEL_PATH):
        try:
            model_win = lgb.Booster(model_file=WIN_MODEL_PATH)
            print("1着特化モデル (m_win) を正常にロードしました。")
        except Exception as e:
            print(f"1着特化モデルロード失敗: {e}")

    if os.path.exists(TOP3_MODEL_PATH):
        try:
            model_top3 = lgb.Booster(model_file=TOP3_MODEL_PATH)
            print("3着内特化モデル (m_top3) を正常にロードしました。")
        except Exception as e:
            print(f"3着内特化モデルロード失敗: {e}")

    if os.path.exists(MAIDEN_WIN_MODEL_PATH):
        try:
            model_maiden_win = lgb.Booster(model_file=MAIDEN_WIN_MODEL_PATH)
            print("未勝利戦Winモデル (m_maiden_win) を正常にロードしました。")
        except Exception as e:
            print(f"未勝利戦Winモデルロード失敗: {e}")

    if os.path.exists(MAIDEN_TOP3_MODEL_PATH):
        try:
            model_maiden_top3 = lgb.Booster(model_file=MAIDEN_TOP3_MODEL_PATH)
            print("未勝利戦Top3モデル (m_maiden_top3) を正常にロードしました。")
        except Exception as e:
            print(f"未勝利戦Top3モデルロード失敗: {e}")

    if os.path.exists(GACHI_MODEL_PATH):
        try:
            model_gachi = lgb.Booster(model_file=GACHI_MODEL_PATH)
            print("ガチガチ判定メタモデル (m_gachi) を正常にロードしました。")
        except Exception as e:
            print(f"ガチガチ判定メタモデルロード失敗: {e}")

    if os.path.exists(HIMO_MODEL_PATH):
        try:
            model_himo = lgb.Booster(model_file=HIMO_MODEL_PATH)
            print("中波乱判定メタモデル (m_himo) を正常にロードしました。")
        except Exception as e:
            print(f"中波乱判定メタモデルロード失敗: {e}")

    # CatBoost モデルのロード
    global model_cb_win, model_cb_top3
    if os.path.exists(CB_WIN_MODEL_PATH) and os.path.exists(CB_TOP3_MODEL_PATH):
        try:
            from catboost import CatBoostClassifier
            model_cb_win = CatBoostClassifier()
            model_cb_win.load_model(CB_WIN_MODEL_PATH)
            model_cb_top3 = CatBoostClassifier()
            model_cb_top3.load_model(CB_TOP3_MODEL_PATH)
            print("CatBoostアンサンブルモデル (cb_win / cb_top3) を正常にロードしました。")
        except Exception as e:
            print(f"CatBoostモデルロード失敗: {e}")

    # XGBoost モデルのロード
    global model_xgb_win, model_xgb_top3
    if os.path.exists(XGB_WIN_MODEL_PATH) and os.path.exists(XGB_TOP3_MODEL_PATH):
        try:
            import xgboost as xgb
            model_xgb_win = xgb.Booster()
            model_xgb_win.load_model(XGB_WIN_MODEL_PATH)
            model_xgb_top3 = xgb.Booster()
            model_xgb_top3.load_model(XGB_TOP3_MODEL_PATH)
            print("XGBoostアンサンブルモデル (xgb_win / xgb_top3) を正常にロードしました。")
        except Exception as e:
            print(f"XGBoostモデルロード失敗: {e}")

    # アンサンブル設定 (ensemble_config.json) のロード
    global ensemble_config
    if os.path.exists(ENSEMBLE_CONFIG_PATH):
        try:
            with open(ENSEMBLE_CONFIG_PATH, "r", encoding="utf-8") as f:
                ensemble_config = json.load(f)
            print("アンサンブル設定 (ensemble_config.json) を正常にロードしました。")
        except Exception as e:
            print(f"アンサンブル設定ロード失敗: {e}")

    if os.path.exists(DICT_PATH):
        try:
            with open(DICT_PATH, "r", encoding="utf-8") as f:
                stats_dict = json.load(f)
            print("統計辞書を正常にロードしました。")
        except Exception as e:
            print(f"統計辞書ロード失敗: {e}")


# モジュールインポート時に自動ロード
load_all_models()


def get_stats_dict() -> Dict[str, Any]:
    """統計辞書を取得"""
    return stats_dict


def load_stats_overlay_from_supabase() -> bool:
    """Supabase keiba_app_meta から永続化された統計辞書オーバーレイを取得し、メモリ上の stats_dict にマージ"""
    global _stats_overlay, stats_dict
    supabase = get_supabase_client()
    if not supabase:
        return False
    try:
        res = supabase.table("keiba_app_meta").select("value").eq("key", "stats_overlay").execute()
        if res.data and len(res.data) > 0 and res.data[0].get("value"):
            overlay_data = res.data[0]["value"]
            if isinstance(overlay_data, dict):
                _stats_overlay = overlay_data
                merged_horses = 0
                for cat in ["horse_stats", "jockey_stats", "trainer_stats", "jockey_course_stats", "trainer_course_stats", "jockey_trainer_stats", "horse_surface_stats", "jockey_surface_stats"]:
                    if cat in overlay_data and isinstance(overlay_data[cat], dict):
                        if cat not in stats_dict:
                            stats_dict[cat] = {}
                        stats_dict[cat].update(overlay_data[cat])
                        if cat == "horse_stats":
                            merged_horses = len(overlay_data[cat])
                print(f"[StatsOverlay] Supabaseから統計オーバーレイをマージしました (更新馬: {merged_horses}頭, 最終更新: {overlay_data.get('updated_at')})")
                return True
    except Exception as e:
        print(f"[StatsOverlay] ロードエラー: {e}")
    return False


def save_stats_overlay_to_supabase() -> bool:
    """メモリ上の _stats_overlay を Supabase keiba_app_meta に永続化"""
    global _stats_overlay
    supabase = get_supabase_client()
    if not supabase:
        return False
    try:
        _stats_overlay["updated_at"] = f"{today_jst_str()} {now_jst().strftime('%H:%M:%S')}"
        supabase.table("keiba_app_meta").upsert({
            "key": "stats_overlay",
            "value": _stats_overlay
        }).execute()
        return True
    except Exception as e:
        print(f"[StatsOverlay] 保存エラー: {e}")
        return False


def update_stats_from_race_settlement(race_id: str, race_date: str, pred_data: dict, race_res: dict) -> int:
    """
    レース確定時に出走全頭の最新成績（着順、人気、オッズ、距離、馬場等）を取り込み、
    stats_dict および _stats_overlay を即時インクリメンタル更新して Supabase に永続化する。
    """
    global stats_dict, _stats_overlay
    orders = race_res.get("orders", {})
    if not orders:
        return 0

    horses = pred_data.get("horses", []) if pred_data else []
    race_details = pred_data.get("race_details", {}) if pred_data else {}
    dist_val = 1600
    surf_val = 0
    try:
        dist_match = re.search(r"(\d{3,4})", str(race_details.get("distance", "")))
        if dist_match:
            dist_val = int(dist_match.group(1))
    except Exception:
        pass
    surf_str = str(race_details.get("surface", ""))
    if "ダ" in surf_str:
        surf_val = 1
    elif "芝" in surf_str:
        surf_val = 0

    updated_count = 0
    if "horse_stats" not in stats_dict:
        stats_dict["horse_stats"] = {}
    if "horse_stats" not in _stats_overlay:
        _stats_overlay["horse_stats"] = {}

    for h in horses:
        try:
            u_raw = h.get("umaban")
            if u_raw is None:
                continue
            u_int = int(u_raw)
            order_raw = orders.get(u_int) or orders.get(str(u_int))
            if order_raw is None or not str(order_raw).isdigit():
                continue
            order_val = int(order_raw)
            if order_val <= 0 or order_val > 30:
                continue

            h_name = str(h.get("horse_name", "")).strip().replace(" ", "").replace("　", "")
            if not h_name:
                continue

            cur_h = stats_dict["horse_stats"].get(h_name, {})
            last_date_existing = str(cur_h.get("last_date", ""))
            if last_date_existing and last_date_existing >= race_date:
                continue

            prev1_existing_order = cur_h.get("prev1_order", 99.0)
            c_races = int(cur_h.get("career_races", 0)) + 1
            is_win = 1 if order_val == 1 else 0
            is_top3 = 1 if order_val <= 3 else 0
            c_wins = int(cur_h.get("career_wins", 0)) + is_win
            prior_races = cur_h.get("career_races", 0)
            prior_top3 = round(float(cur_h.get("career_top3_rate", 0.0)) * prior_races) if prior_races > 0 else 0
            new_top3_rate = round(float(prior_top3 + is_top3) / float(c_races), 4)

            pop_val = float(h.get("popularity") or cur_h.get("prev1_pop", 10.0))
            odds_val = float(h.get("odds") or cur_h.get("prev1_odds", 50.0))
            weight_val = float(h.get("horse_weight") or cur_h.get("last_weight", 480.0))
            kinryo_val = float(h.get("kinryo") or cur_h.get("last_kinryo", 55.0))
            blinker_val = 1 if h.get("is_blinker") else 0

            updated_horse_meta = {
                "career_races": c_races,
                "career_wins": c_wins,
                "career_top3_rate": new_top3_rate,
                "prev1_order": float(order_val),
                "prev1_pop": pop_val,
                "prev1_odds": odds_val,
                "prev2_order": float(prev1_existing_order),
                "last_date": race_date,
                "last_dist": dist_val,
                "last_surface": surf_val,
                "last_turn": cur_h.get("last_turn", 1),
                "last_weight": weight_val,
                "last_kinryo": kinryo_val,
                "last_is_blinker": blinker_val,
                "was_last_rest": cur_h.get("was_last_rest", 0),
                "last_3f": cur_h.get("last_3f", 36.0),
                "last_corner_final": cur_h.get("last_corner_final", 8.0),
                "prev1_time_diff": cur_h.get("prev1_time_diff", 0.5),
                "prev_time_diff_min": cur_h.get("prev_time_diff_min", 0.5),
                "prev1_last_3f_rank": cur_h.get("prev1_last_3f_rank", 10.0),
                "prev1_class_diff": 0.0
            }

            stats_dict["horse_stats"][h_name] = updated_horse_meta
            _stats_overlay["horse_stats"][h_name] = updated_horse_meta
            updated_count += 1
        except Exception as e_h:
            print(f"[StatsOverlay] 馬 {h.get('horse_name')} 更新スキップ: {e_h}")

    if updated_count > 0:
        save_stats_overlay_to_supabase()
        print(f"[StatsOverlay] レース {race_id} ({race_date}) より {updated_count}頭の統計を動的更新＆永続化しました。")

    return updated_count

