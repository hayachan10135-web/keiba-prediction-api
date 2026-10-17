from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict
import os
import math
import time
import datetime
import zoneinfo
import re
import json
import asyncio
import threading
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import lightgbm as lgb
import traceback
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from ipat_fast_client import IPATFastClient, VENUE_CODE_MAP
from umaca_fast_client import UMACAFastClient

from app.core.timezone import JST, now_jst, now_jst_naive, today_jst_str, parse_post_datetime
from app.strategies import (
    StrategyArbiter,
    NATURAL_REAL_STRATEGIES,
    get_or_select_daily_top_race_id,
    ensure_daily_single_real_bet,
    set_daily_promotion_dependencies,
    invalidate_daily_top_cache,
    _daily_top_race_id_cache,
)
from app.ml import (
    compute_track_bias_and_matches,
    set_track_bias_dependencies,
    evaluate_horse_bias_match,
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
    load_stats_overlay_from_supabase,
    save_stats_overlay_to_supabase,
    update_stats_from_race_settlement,
    predict_race_core,
    build_prediction_and_recs,
    set_predictor_dependencies,
    calc_trio_odds,
    apply_odds_anomaly_indicators,
    compute_hybrid_rec_from_inputs,
    rank_min_desc,
    rank_min_asc,
    calculate_dynamic_bet,
    ODDS_STAGE_ORDER,
    ODDS_STAGE_RANK,
    infer_odds_stage_from_minutes,
)
from app.betting.factory import BettingClientFactory
from app.betting.manual_service import execute_manual_bet
from app.settlement import (
    PayoutVerifier,
    DailySummary,
    AggregateSummaryResponse,
    TicketTypeSummary,
    PeriodStatItem,
    invalidate_summary_caches,
    verify_race_result,
    process_settlement_calculation,
    get_cached_settlement,
    get_cached_result,
    save_cached_result,
    save_settlement,
    resolve_race_date_for_id,
    extract_top3_orders,
    extract_mark_orders,
    sync_orders_marks,
    format_mark_orders_text,
    calculate_hybrid_pass_benefit_info,
    set_settlement_engine_dependencies
)
from app.betting.settings import (
    load_betting_settings_from_supabase,
    save_betting_settings_to_supabase,
    calculate_remaining_real_bet_amount,
    _betting_settings
)
from app.routers.betting import router as betting_router
from app.routers.admin import router as admin_router
from app.routers.summary import router as summary_router
from app.routers.batch import (
    router as batch_router,
    set_batch_workers,
    batch_status,
    BatchStartResponse,
    BatchStatusResponse
)
from app.batch import (
    run_batch_worker,
    set_batch_worker_dependencies,
    background_maintenance_loop,
    sync_daily_schedules_if_needed,
    set_scheduler_dependencies,
)
from app.maintenance import (
    recalculate_period_records,
    set_recalculator_dependencies,
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
from app.schedule import (
    get_current_kaisai_period,
    clear_kaisai_period_caches,
    discover_period_kaisai_dates,
    register_race_schedule,
    ensure_schedules_registered_for_venues,
    attach_prediction_badges_to_venues,
    get_today_races,
    set_races_dependencies,
    _period_venues_cache,
    _period_checked_empty_dates,
    _latest_post_times,
    _last_weight_check_ts,
    _today_races_cache,
)
from app.routers.settlement import (
    router as settlement_router,
    set_settlement_handler,
    VerificationResult
)
from app.routers.prediction import (
    router as prediction_router,
    set_prediction_handlers,
    PredictResponse,
    HorsePrediction,
    TicketBet,
    HybridRecommendation,
    TanFukuRecommendation,
    RaceDetails,
    TrackBiasInfo
)
from app.scrapers import (
    safe_scrape_get,
    safe_decode_html,
    fetch_real_odds,
    fetch_shutuba_table,
    fetch_netkeiba_race_result,
    normalize_race_res,
    register_post_time_callback,
    register_schedule_callback,
    parse_netkeiba_race_list_html,
    fetch_venues_for_date_raw,
    register_crawler_post_time_handler,
    register_venues_loaded_handler,
    USER_AGENTS,
    HEADERS_PC
)


_strategy_arbiter = StrategyArbiter()

app = FastAPI(title="Keiba Prediction & Verification API")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.include_router(betting_router)
app.include_router(admin_router)
app.include_router(summary_router)
app.include_router(batch_router)
app.include_router(settlement_router)
app.include_router(prediction_router)

@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    err_list = exc.errors()
    tb_str = traceback.format_exc()
    err_msg = f"【ResponseValidationError】レスポンス型不整合エラー ({len(err_list)}件):\n{json.dumps(err_list, ensure_ascii=False, default=str)}\n{tb_str}"
    print(f"[ERROR ResponseValidationError] {err_msg}")
    return JSONResponse(
        status_code=500,
        content={"detail": err_msg}
    )

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    err_list = exc.errors()
    err_msg = f"【RequestValidationError】リクエスト入力バリデーションエラー:\n{json.dumps(err_list, ensure_ascii=False, default=str)}"
    print(f"[ERROR RequestValidationError] {err_msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": err_msg}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    tb_str = traceback.format_exc()
    err_msg = f"【サーバー内部未捕捉例外 [{exc.__class__.__name__}]】: {str(exc)}\n{tb_str}"
    print(f"[ERROR GlobalException] {err_msg}")
    return JSONResponse(
        status_code=500,
        content={"detail": err_msg}
    )

# --- 日本時間 (JST) ユーティリティ ---
JST = zoneinfo.ZoneInfo("Asia/Tokyo")

def now_jst() -> datetime.datetime:
    """日本時間の現在時刻を取得"""
    return datetime.datetime.now(JST)

def now_jst_naive() -> datetime.datetime:
    """比較用のタイムゾーンなしJST現在日時を取得"""
    return now_jst().replace(tzinfo=None)

def today_jst_str() -> str:
    """日本時間の当日日付 (YYYY-MM-DD)"""
    return now_jst().strftime("%Y-%m-%d")

def parse_post_datetime(p_dt_str: Optional[str], fallback_date_str: Optional[str] = None) -> Optional[datetime.datetime]:
    """Supabase等から返る発走日時文字列 ('YYYY-MM-DD HH:MM:SS', ISO形式 'T'/'Z'/'+09:00', 'HH:MM') を安全にnaive JST datetimeへ変換"""
    if not p_dt_str or not isinstance(p_dt_str, str):
        return None
    s = p_dt_str.strip()
    m_full = re.search(r"(\d{4})-(\d{2})-(\d{2})[T\s]+(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m_full:
        try:
            return datetime.datetime(
                int(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3)),
                int(m_full.group(4)), int(m_full.group(5)), int(m_full.group(6) or 0)
            )
        except Exception:
            pass
    if fallback_date_str:
        m_hm = re.search(r"(\d{1,2}):(\d{2})", s)
        m_d = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", fallback_date_str.strip())
        if m_hm and m_d:
            try:
                return datetime.datetime(
                    int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3)),
                    int(m_hm.group(1)), int(m_hm.group(2)), 0
                )
            except Exception:
                pass
    return None





GLOBAL_VENUE_MAP = {
    1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京",
    6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"
}

CANCELLED_RACE_NAMES = {
    "202605010308": "東京8R (積雪中止)",
    "202605010309": "東京9R (積雪中止)",
    "202605010310": "東京10R (積雪中止)",
    "202605010311": "東京11R 早春S (積雪中止)",
    "202605010312": "東京12R (積雪中止)",
    "202610010604": "小倉4R (中止)",
}
# モデル・辞書パス定数は app/ml/model_loader.py へ移管完了

# --- Supabase クライアント初期化 ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabaseクライアントを正常に初期化しました。")
    except Exception as e:
        print(f"Supabase初期化エラー: {e}")
else:
    print("警告: SUPABASE_URL または SUPABASE_KEY が環境変数に設定されていません。")

# --- Firebase Admin SDK 初期化 & プッシュ通知ヘルパー ---
firebase_app = None
FIREBASE_KEY_PATH = os.environ.get("FIREBASE_KEY_PATH", "firebase_service_account.json")
FIREBASE_KEY_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY", "")

try:
    import firebase_admin
    from firebase_admin import credentials, messaging

    if FIREBASE_KEY_JSON:
        try:
            cred_dict = json.loads(FIREBASE_KEY_JSON)
            cred = credentials.Certificate(cred_dict)
            firebase_app = firebase_admin.initialize_app(cred)
            print("[Firebase] 環境変数 FIREBASE_SERVICE_ACCOUNT_KEY から正常に初期化しました。")
        except Exception as fe:
            print(f"[Firebase] JSONパース失敗: {fe}")
    elif os.path.exists(FIREBASE_KEY_PATH):
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_app = firebase_admin.initialize_app(cred)
        print(f"[Firebase] ファイル {FIREBASE_KEY_PATH} から正常に初期化しました。")
    else:
        print("[Firebase] サービスアカウント情報未設定（設定完了後にプッシュ通知が有効化されます）。")
except Exception as f_err:
    print(f"[Firebase] 初期化待機: {f_err}")


def send_fcm_notification(title: str, body: str, data: Optional[Dict[str, Any]] = None, topic: str = "keiba_real_bet") -> bool:
    """Firebase Cloud Messaging で指定トピックへプッシュ通知を配信"""
    if not firebase_app:
        print(f"[FCM未設定・通知待機] {title} | {body} (Topic: {topic})")
        return False
    try:
        from firebase_admin import messaging
        str_data = {str(k): str(v) for k, v in (data or {}).items()}
        msg = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body
            ),
            data=str_data,
            topic=topic
        )
        resp = messaging.send(msg)
        print(f"[FCM送信成功] {title} (ID: {resp})")
        return True
    except Exception as e:
        print(f"[FCM送信失敗] {title}: {e}")
        return False

# LightGBMモデル・統計辞書・オンライン更新機構は app/ml/model_loader.py へ移管完了
_auto_betted_races = set()       # 二重投票防止用メモリセット (race_id)


# calc_trio_odds は app/ml/predictor.py へ移管完了







@app.get("/")
@app.get("/health")
@app.head("/")
@app.head("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Keiba Prediction & Verification API is running",
        "firebase_configured": bool(firebase_app is not None),
        "server_time_jst": now_jst().strftime("%Y-%m-%d %H:%M:%S")
    }



_last_balance_warning_date: Optional[str] = None
_last_warned_shortfall: int = 0

def check_and_notify_balance_shortfall():
    """
    自動投票モード (即PAT / UMACAスマート) が有効な場合、残り投票予定額と現在残高を比較し、
    不足が発生しそうな場合に FCM プッシュ通知を送信
    """
    global _last_balance_warning_date, _last_warned_shortfall
    b_settings = load_betting_settings_from_supabase()
    cur_mode = b_settings.get("betting_mode", "off")
    if cur_mode not in ("ipat", "umaca"):
        return

    platform_label = "UMACAスマート" if cur_mode == "umaca" else "即PAT"
    cur_balance = b_settings.get("last_balance")
    if cur_balance is None:
        try:
            client = BettingClientFactory.get_client(cur_mode)
            l_res = client.login()
            if l_res.get("success"):
                cur_balance = l_res.get("balance", 0)
                b_settings["last_balance"] = cur_balance
                save_betting_settings_to_supabase(b_settings)
        except Exception:
            return

    if cur_balance is None:
        return

    rem_amt, rem_count, target_races = calculate_remaining_real_bet_amount()
    if rem_count == 0 or rem_amt == 0:
        return

    shortfall = rem_amt - cur_balance
    if shortfall > 0:
        today_s = today_jst_str()
        if _last_balance_warning_date == today_s and _last_warned_shortfall >= shortfall:
            return

        _last_balance_warning_date = today_s
        _last_warned_shortfall = shortfall

        race_names_str = "、".join([r["race_name"] for r in target_races[:3]])
        if len(target_races) > 3:
            race_names_str += f" 他{len(target_races)-3}レース"

        title = f"⚠️【{platform_label}残高不足警告】{shortfall:,}円 不足の恐れ"
        body = f"本日残り実戦投票予定: {rem_count}レース 計{rem_amt:,}円\n{platform_label}残高: {cur_balance:,}円 (不足額: -{shortfall:,}円)\n対象: {race_names_str}\n※{platform_label}へ追加入金（チャージ）を行ってください。"

        print(f"[BalanceWarning] {title} を送信します")
        send_fcm_notification(
            title=title,
            body=body,
            data={
                "type": "BALANCE_SHORTFALL_WARNING",
                "shortfall": str(shortfall),
                "remaining_amount": str(rem_amt),
                "current_balance": str(cur_balance)
            }
        )



@app.get("/notify/test")
@app.post("/notify/test")
def trigger_test_notification():
    """テスト用のプッシュ通知をFirebaseトピック (keiba_real_bet) へ送信"""
    if not firebase_app:
        return {
            "status": "error",
            "message": "Firebaseが未初期化です。Renderの環境変数 FIREBASE_SERVICE_ACCOUNT_KEY に秘密鍵JSONを設定してください。"
        }
    success = send_fcm_notification(
        title="🔔【テスト通知】実戦推奨レース速報",
        body="これはFirebaseプッシュ通知の疎通テストです。正常に受信できています！",
        data={
            "type": "TEST",
            "message": "FCM test notification successful"
        }
    )
    if success:
        return {"status": "ok", "message": "テスト通知をトピック (keiba_real_bet) へ送信しました。スマホをご確認ください！"}
    else:
        return {"status": "error", "message": "FCM送信に失敗しました。サーバーログをご確認ください。"}




# --- 開催期間・スケジュール・会場管理ロジックは app/schedule/ へ移管完了 ---







# rank_min_desc, rank_min_asc, calculate_dynamic_bet は app/ml/predictor.py へ移管完了


# --- 【実馬券選定 ＆ 0レース時の当日No.1昇格保証】ヘルパー機構 (app/strategies/daily_promotion.py 連携) ---
_auto_betted_dates: Set[str] = set()


def _invalidate_main_prediction_caches(target_date: str):
    if "_all_predictions_cache" in globals():
        _all_predictions_cache.pop(target_date, None)
    if "_today_races_cache" in globals():
        _today_races_cache.pop(target_date, None)


set_daily_promotion_dependencies(cache_invalidation_handler=_invalidate_main_prediction_caches)

def has_already_betted_race(date_str: str, race_id: str) -> bool:
    """指定レース(race_id)がすでに自動投票完了済みか確認 (同一レース二重発注防止)"""
    if race_id in _auto_betted_races:
        return True
    try:
        settings = load_betting_settings_from_supabase()
        betted = settings.get("betted_races_by_date", {})
        for item in betted.get(date_str, []):
            if isinstance(item, dict) and item.get("race_id") == race_id:
                _auto_betted_races.add(race_id)
                return True
    except Exception:
        pass
    return False

# get_or_select_daily_top_race_id および ensure_daily_single_real_bet は
# app/strategies/daily_promotion.py へ移管完了


# build_prediction_and_recs および compute_hybrid_rec_from_inputs は app/ml/predictor.py へ移管完了


# =========================================================================
# ★ Supabase CRUD ヘルパー
# =========================================================================

# ODDS_STAGE_ORDER, ODDS_STAGE_RANK, infer_odds_stage_from_minutes は app/ml/predictor.py へ移管完了


# apply_odds_anomaly_indicators は app/ml/predictor.py へ移管完了

# compute_track_bias_and_matches は app/ml/track_bias.py へ移管完了


def get_cached_prediction(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_predictions_cache").select("data_json, race_date").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            d_json = res.data[0]["data_json"]
            r_date = res.data[0].get("race_date") or today_jst_str()
            if isinstance(d_json, dict) and "horses" in d_json:
                orig_horses = d_json.get("horses", [])
                seen_u = set()
                dedup_horses = []
                for h in orig_horses:
                    u = h.get("umaban")
                    if u is not None and u not in seen_u:
                        seen_u.add(u)
                        dedup_horses.append(h)
                if len(dedup_horses) != len(orig_horses):
                    d_json["horses"] = dedup_horses
                    if supabase:
                        try:
                            supabase.table("keiba_predictions_cache").update({"data_json": d_json}).eq("race_id", race_id).execute()
                            print(f"[Self-Heal] {race_id}: キャッシュ内の重複馬番を修復しました ({len(orig_horses)}頭 -> {len(dedup_horses)}頭)")
                        except Exception as ue:
                            pass
                apply_odds_anomaly_indicators(d_json.get("horses", []))
                c_details = (d_json.get("race_details") or {}).get("course_details", "")
                surf_str = "芝" if "芝" in c_details else ("ダート" if "ダ" in c_details else "障害")
                tb_info = compute_track_bias_and_matches(race_id, r_date, surf_str, d_json.get("horses", []))
                if tb_info:
                    d_json["track_bias"] = tb_info
            return d_json
    except Exception as e:
        print(f"Supabase予想取得エラー ({race_id}): {e}")
    return None

def save_cached_prediction(race_id: str, race_date: str, data_dict: dict):
    if not supabase:
        return
    horses = (data_dict or {}).get("horses", [])
    if not horses or len(horses) == 0:
        print(f"[SaveCache Warning] レース {race_id} の出走馬が0頭のためキャッシュ保存を拒絶しました。")
        return
    for h in horses:
        h_name = h.get("horse_name", "")
        if not h_name or re.match(r"^馬\d+$", h_name):
            print(f"[SaveCache Warning] レース {race_id} にダミー馬名 '{h_name}' が含まれるためキャッシュ保存を拒絶しました。")
            return
    try:
        supabase.table("keiba_predictions_cache").upsert({
            "race_id": race_id,
            "race_date": race_date,
            "data_json": data_dict
        }).execute()
        _all_predictions_cache.pop(race_date, None)
        _today_races_cache.pop(race_date, None)
    except Exception as e:
        print(f"Supabase予想保存エラー ({race_id}): {e}")


def sync_virtual_badge_to_prediction(race_id: str, pred_data: dict, race_date: Optional[str] = None) -> dict:
    """レース一覧側で選定された確信度上位3枠（仮想観戦枠）情報をhybrid_recommendationにも反映（対象外は見送りに整合）"""
    if not pred_data or not isinstance(pred_data, dict):
        return pred_data
    h_rec = pred_data.get("hybrid_recommendation")
    if not h_rec or not isinstance(h_rec, dict):
        return pred_data
    # 既にリアル勝負枠の場合は何もしない
    if h_rec.get("is_real_bet_target"):
        return pred_data

    target_d = race_date or today_jst_str()

    def _apply_venue_race_sync(r_item: dict):
        if r_item.get("is_virtual_bet_target"):
            h_rec["is_virtual_bet_target"] = True
            h_rec["virtual_bet_slot"] = r_item.get("virtual_bet_slot")
            h_rec["real_bet_level"] = r_item.get("real_bet_level", "virtual")
            h_rec["real_bet_badge"] = r_item.get("real_bet_badge")
            h_rec["virtual_badge"] = r_item.get("real_bet_badge")
        else:
            h_rec["is_virtual_bet_target"] = False
            h_rec["virtual_bet_slot"] = None
            h_rec["real_bet_level"] = "none"
            h_rec["real_bet_badge"] = None
            h_rec["virtual_badge"] = None

    # 1. _today_races_cache から検索
    cached_venues_entry = _today_races_cache.get(target_d)
    if cached_venues_entry:
        v_data = cached_venues_entry[1]
        venues_list = v_data.get("venues", []) if isinstance(v_data, dict) else (v_data if isinstance(v_data, list) else [])
        for venue in venues_list:
            for r in venue.get("races", []):
                if r.get("race_id") == race_id:
                    _apply_venue_race_sync(r)
                    return pred_data

    # 2. _today_races_cache に無ければ会場一覧から付与して解決
    try:
        raw_venues = fetch_venues_for_date_raw(target_d)
        if raw_venues:
            attached_venues = attach_prediction_badges_to_venues(target_d, raw_venues)
            for venue in attached_venues:
                for r in venue.get("races", []):
                    if r.get("race_id") == race_id:
                        _apply_venue_race_sync(r)
                        return pred_data
    except Exception as e:
        print(f"仮想枠バッジ同期エラー ({race_id}): {e}")

    return pred_data


# predict_race_core は app/ml/predictor.py へ移管完了
set_predictor_dependencies(
    get_cached_pred=get_cached_prediction,
    save_cached_pred=save_cached_prediction,
    sync_virtual_badge=sync_virtual_badge_to_prediction,
    resolve_race_date=resolve_race_date_for_id,
    latest_post_times=_latest_post_times,
)


_all_predictions_cache: Dict[str, Tuple[float, dict]] = {}

def get_all_predictions_for_date(date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)")):
    """指定日付の全レースのキャッシュ済み予想を一括返却（アプリの先読み・ゼロレイテンシ用）"""
    target_date = date if date else today_jst_str()
    if not supabase:
        return {"date": target_date, "count": 0, "predictions": {}}

    now_ts = time.time()
    today_str = today_jst_str()
    all_cache_ttl = 30.0 if target_date == today_str else 3600.0
    cached_all = _all_predictions_cache.get(target_date)
    if cached_all and (now_ts - cached_all[0] < all_cache_ttl):
        return cached_all[1]

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return {"date": target_date, "count": 0, "predictions": {}}

        pred_map = {row["race_id"]: (row.get("data_json") or {}) for row in rows if row.get("race_id")}
        all_day_rids = list(pred_map.keys())

        # 当日確定結果を一括バッチフェッチ (全レース分を1回のクエリに集約)
        parsed_settled = []
        try:
            r_res = supabase.table("keiba_race_results_cache").select("race_id, result_json").eq("race_date", target_date).execute()
            raw_results = r_res.data or []
            if not raw_results and all_day_rids:
                r_res2 = supabase.table("keiba_race_results_cache").select("race_id, result_json").in_("race_id", all_day_rids).execute()
                raw_results = r_res2.data or []

            for r in raw_results:
                rid = r.get("race_id")
                rj = r.get("result_json") or {}
                orders = rj.get("orders") or {}
                if orders:
                    parsed_settled.append({"race_id": rid, "orders": orders})
        except Exception as e_res:
            print(f"[get_all_predictions] results fetch warning: {e_res}")

        preds = {}
        for row in rows:
            r_id = row.get("race_id")
            d_json = row.get("data_json")
            if r_id and d_json:
                if isinstance(d_json, dict) and "horses" in d_json:
                    apply_odds_anomaly_indicators(d_json.get("horses", []))
                    c_det = (d_json.get("race_details") or {}).get("course_details", "")
                    surf_str = "芝" if "芝" in c_det else ("ダート" if "ダ" in c_det else "障害")
                    tb_info = compute_track_bias_and_matches(
                        r_id,
                        target_date,
                        surf_str,
                        d_json.get("horses", []),
                        prefetched_settled_races=parsed_settled,
                        prefetched_pred_map=pred_map
                    )
                    if tb_info:
                        d_json["track_bias"] = tb_info
                preds[r_id] = d_json

        # 実馬券厳選保証: 自然条件(戦略B/A/C/A-2)を満たすレースは全て実馬券とし、0レースの時のみ当日No.1の1レースを実馬券に昇格
        real_count = sum(1 for p in preds.values() if (p.get("hybrid_recommendation") or {}).get("is_real_bet_target"))
        has_downgraded_ref = any("参考推奨" in str((p.get("hybrid_recommendation") or {}).get("real_bet_badge") or "") for p in preds.values())
        if (real_count == 0 or has_downgraded_ref) and target_date >= today_jst_str() and preds:
            ensure_daily_single_real_bet(target_date)
            fresh_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
            for row in (fresh_res.data or []):
                if row.get("race_id") and row.get("data_json"):
                    preds[row["race_id"]] = row["data_json"]
        # 各会場確信度上位3枠（仮想観戦枠）バッジを予測データにも同期反映（対象外は見送りに完全整合）
        cached_venues_entry = _today_races_cache.get(target_date)
        venues_list = []
        if cached_venues_entry:
            v_data = cached_venues_entry[1]
            venues_list = v_data.get("venues", []) if isinstance(v_data, dict) else (v_data if isinstance(v_data, list) else [])
        else:
            try:
                raw_v = fetch_venues_for_date_raw(target_date)
                if raw_v:
                    venues_list = attach_prediction_badges_to_venues(target_date, raw_v)
            except Exception:
                venues_list = []

        if venues_list:
            for venue in venues_list:
                for r in venue.get("races", []):
                    rid = r.get("race_id")
                    if rid in preds:
                        hr = preds[rid].get("hybrid_recommendation")
                        if hr and not hr.get("is_real_bet_target"):
                            if r.get("is_virtual_bet_target"):
                                hr["is_virtual_bet_target"] = True
                                hr["virtual_bet_slot"] = r.get("virtual_bet_slot")
                                hr["real_bet_level"] = r.get("real_bet_level", "virtual")
                                hr["real_bet_badge"] = r.get("real_bet_badge")
                                hr["virtual_badge"] = r.get("real_bet_badge")
                            else:
                                hr["is_virtual_bet_target"] = False
                                hr["virtual_bet_slot"] = None
                                hr["real_bet_level"] = "none"
                                hr["real_bet_badge"] = None
                                hr["virtual_badge"] = None
        else:
            for rid, p_obj in preds.items():
                hr = p_obj.get("hybrid_recommendation")
                if hr and not hr.get("is_real_bet_target") and hr.get("real_bet_level") != "virtual":
                    hr["is_virtual_bet_target"] = False
                    hr["virtual_bet_slot"] = None
                    hr["real_bet_level"] = "none"
                    hr["real_bet_badge"] = None
                    hr["virtual_badge"] = None

        result_payload = {
            "date": target_date,
            "count": len(preds),
            "predictions": preds
        }
        _all_predictions_cache[target_date] = (now_ts, result_payload)
        return result_payload
    except Exception as e:
        print(f"全予想一括取得エラー ({target_date}): {e}")
        return {"date": target_date, "count": 0, "predictions": {}}


def predict_race(race_id: str, force_refresh: bool = Query(False, description="手動で予想を再実行する場合True")):
    return predict_race_core(race_id, force_refresh=force_refresh, is_auto_final=False)

set_races_dependencies(ensure_daily_single_real_bet=ensure_daily_single_real_bet)
set_prediction_handlers(get_today_races, get_all_predictions_for_date, predict_race)

set_settlement_engine_dependencies(
    update_stats=update_stats_from_race_settlement,
    send_fcm=send_fcm_notification,
    get_prediction=get_cached_prediction,
    predict_race_core=predict_race_core,
    sync_virtual_badge=sync_virtual_badge_to_prediction,
    period_venues_cache=_period_venues_cache
)
set_settlement_handler(verify_race_result)






# =========================================================================
# ★ 一括処理（1開催期間内の未予想・未検証レースの高速差分実行）
# =========================================================================

# run_batch_worker は app/batch/worker.py へ移管完了
set_batch_worker_dependencies(
    get_current_kaisai_period=get_current_kaisai_period,
    discover_period_kaisai_dates=discover_period_kaisai_dates,
    fetch_venues_for_date_raw=fetch_venues_for_date_raw,
    predict_race_core=predict_race_core,
    verify_race_result=verify_race_result,
    now_jst_naive=now_jst_naive,
    today_jst_str=today_jst_str,
)


# recalculate_period_records は app/maintenance/recalculator.py へ移管完了
set_recalculator_dependencies(
    today_jst_str=today_jst_str,
    save_cached_prediction=save_cached_prediction,
    period_venues_cache=_period_venues_cache,
)

set_batch_workers(run_batch_worker, recalculate_period_records)


# =========================================================================
# ★ バックグラウンドメンテナンスループ (JST完全同期・最適化タイムテーブル版)
#   ① 開催日 朝07:30 : 当日の出走表・発走スケジュール一括登録
#   ② 開催日 朝08:30 : 当日の未予想レース暫定一括予想 ＆ 朝ダイジェスト通知
#   ③ 各レース発走20分前 : 直前オッズで再予想・本確定 ＆ 実戦推奨通知
#   ④ 各レース発走15分後 : レース結果自動検証（未確定時は5分毎リトライ）＆ 結果通知
#   ⑤ 開催日 17:00 / 19:00 : 当日最終の未確定レース一括照合
#   ⑥ 金曜日 夜20:00 : 1開催期間分の一時キャッシュ破棄（成績台帳は永続保持）
# =========================================================================

# sync_daily_schedules_if_needed は app/batch/scheduler.py へ移管完了


# background_maintenance_loop は app/batch/scheduler.py へ移管完了
set_scheduler_dependencies(
    now_jst=now_jst,
    now_jst_naive=now_jst_naive,
    today_jst_str=today_jst_str,
    parse_post_datetime=parse_post_datetime,
    get_current_kaisai_period=get_current_kaisai_period,
    clear_kaisai_period_caches=clear_kaisai_period_caches,
    discover_period_kaisai_dates=discover_period_kaisai_dates,
    fetch_venues_for_date_raw=fetch_venues_for_date_raw,
    ensure_daily_single_real_bet=ensure_daily_single_real_bet,
    send_fcm_notification=send_fcm_notification,
    check_and_notify_balance_shortfall=check_and_notify_balance_shortfall,
    predict_race_core=predict_race_core,
    verify_race_result=verify_race_result,
    has_already_betted_race=has_already_betted_race,
    run_batch_worker=run_batch_worker,
    period_checked_empty_dates=_period_checked_empty_dates,
    auto_betted_races=_auto_betted_races,
    auto_betted_dates=_auto_betted_dates,
    venue_code_map=VENUE_CODE_MAP,
)



# fix_mislabeled_20261011_races, clear_all_database, check_and_apply_db_migration は app/maintenance/healer.py へ移管完了
set_healer_dependencies(
    today_jst_str=today_jst_str,
    now_jst_naive=now_jst_naive,
    parse_post_datetime=parse_post_datetime,
    get_today_races=get_today_races,
    predict_race_core=predict_race_core,
    verify_race_result=verify_race_result,
    ensure_daily_single_real_bet=ensure_daily_single_real_bet,
    infer_odds_stage_from_minutes=infer_odds_stage_from_minutes,
    get_cached_prediction=get_cached_prediction,
    get_cached_result=get_cached_result,
    all_predictions_cache=_all_predictions_cache,
    today_races_cache=_today_races_cache,
    last_weight_check_ts=_last_weight_check_ts,
)


@app.post("/admin/clear-db")
def endpoint_clear_db(sync_today: bool = False):
    res = clear_all_database(sync_today=sync_today)
    if supabase and res.get("status") == "ok":
        try:
            supabase.table("keiba_app_meta").upsert({
                "key": "schema_version",
                "value": {"version": CURRENT_DB_VERSION, "updated_at": today_jst_str()}
            }).execute()
        except Exception:
            pass
        try:
            supabase.table("keiba_predictions_cache").upsert({
                "race_id": "SCHEMA_VERSION",
                "race_date": "9999-12-31",
                "data_json": {"version": CURRENT_DB_VERSION, "updated_at": today_jst_str()}
            }).execute()
        except Exception:
            pass
    return res


@app.post("/admin/fix-20261011")
def endpoint_fix_20261011():
    return fix_mislabeled_20261011_races()





# normalize_mislabeled_dates, sync_and_process_target_dates は app/maintenance/healer.py へ移管完了


def sync_and_process_1003_1004():
    return sync_and_process_target_dates(["2026-10-03", "2026-10-04"])


def sync_and_process_0926_0927():
    return sync_and_process_target_dates(["2026-09-26", "2026-09-27"])


@app.post("/admin/normalize-dates")
def endpoint_normalize_dates():
    return normalize_mislabeled_dates()


@app.post("/admin/process-1003-1004")
def endpoint_process_1003_1004():
    return sync_and_process_1003_1004()


@app.post("/admin/process-0926-0927")
def endpoint_process_0926_0927():
    return sync_and_process_0926_0927()


@app.post("/admin/process-date")
def endpoint_process_date(date: str = Query(..., description="対象日付 (YYYY-MM-DD)")):
    return sync_and_process_target_dates([date])


class BatchUpsertPayload(BaseModel):
    predictions: Optional[List[dict]] = None
    settlements: Optional[List[dict]] = None
    scheduled: Optional[List[dict]] = None


@app.post("/admin/upsert-verified-data")
def endpoint_upsert_verified_data(payload: BatchUpsertPayload):
    """検証済みの予想・収支結果・スケジュールデータをSupabaseに一括アップサート"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase未接続")

    # バリデーション: 不完全データ、ダミー馬名、出走馬0頭のアップサートを厳格に遮断
    if payload.predictions:
        for p in payload.predictions:
            r_id = p.get("race_id", "")
            d_json = p.get("data_json", {})
            horses = d_json.get("horses", [])
            if not horses or len(horses) == 0:
                raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} の出走馬データが0頭です。")
            for h in horses:
                h_name = h.get("horse_name", "")
                if not h_name or re.match(r"^馬\d+$", h_name):
                    raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} にダミー馬名 '{h_name}' (馬番 {h.get('umaban')}) が含まれています。")

    if payload.settlements:
        for s in payload.settlements:
            r_id = s.get("race_id", "")
            h_name = s.get("horse_name", "")
            if not s.get("is_cancelled") and s.get("strategy_type") != "cancelled":
                if not h_name or re.match(r"^馬\d+$", h_name):
                    raise HTTPException(status_code=400, detail=f"不正データ拒絶: レース {r_id} の本命馬名がダミー '{h_name}' です。")

    pred_count = 0
    settle_count = 0
    sched_count = 0
    try:
        if payload.predictions:
            supabase.table("keiba_predictions_cache").upsert(payload.predictions, on_conflict="race_id").execute()
            pred_count = len(payload.predictions)
        if payload.settlements:
            supabase.table("keiba_race_settlements").upsert(payload.settlements, on_conflict="race_id").execute()
            settle_count = len(payload.settlements)
        if payload.scheduled:
            supabase.table("keiba_scheduled_races").upsert(payload.scheduled, on_conflict="race_id").execute()
            sched_count = len(payload.scheduled)
        invalidate_summary_caches()
        return {
            "status": "ok",
            "predictions_count": pred_count,
            "settlements_count": settle_count,
            "scheduled_count": sched_count
        }
    except Exception as e:
        print(f"[UpsertVerifiedData Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/sync-active-post-times")
def endpoint_sync_active_post_times(dates: Optional[str] = Query(None, description="カンマ区切りの対象日 (YYYY-MM-DD)")):
    """現在メイン画面に表示される開催期間の全レースについて、実際の出走時刻をスクレイピングしてDBに反映（変更対応）"""
    if dates:
        target_dates = [d.strip() for d in dates.split(",") if d.strip()]
    else:
        now_dt = now_jst_naive()
        target_dates = discover_period_kaisai_dates(now_dt)
        if not target_dates:
            target_dates = ["2026-10-12", "2026-10-11", "2026-10-10"]

    results = {}
    for d_str in target_dates:
        venues = fetch_venues_for_date_raw(d_str, force_refresh=True)
        count = 0
        for v in venues:
            for r in v.get("races", []):
                r_id = r.get("race_id")
                r_info = r.get("race_info", "")
                m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                if r_id and m:
                    hour_int, min_int = int(m.group(1)), int(m.group(2))
                    pt_str = f"{hour_int:02d}:{min_int:02d}発走"
                    _latest_post_times[r_id] = pt_str
                    # DB更新
                    if supabase:
                        try:
                            # 1. scheduled_races
                            d_parts = [int(p) for p in d_str.split("-")]
                            post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour_int, min_int, 0)
                            supabase.table("keiba_scheduled_races").update({
                                "post_datetime": post_dt.strftime("%Y-%m-%d %H:%M:%S")
                            }).eq("race_id", r_id).execute()
                            # 2. predictions_cache
                            p_res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", r_id).execute()
                            if p_res.data and p_res.data[0].get("data_json"):
                                dj = p_res.data[0]["data_json"]
                                dj_details = dj.get("race_details") or {}
                                if dj_details.get("post_time") != pt_str:
                                    dj_details["post_time"] = pt_str
                                    dj["race_details"] = dj_details
                                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", r_id).execute()
                        except Exception:
                            pass
                    count += 1
        results[d_str] = count

    return {"status": "ok", "updated_races_by_date": results}





@app.post("/admin/refresh-stats-dict")
def endpoint_refresh_stats_dict(rebuild_from_cached_results: bool = False):
    """
    統計辞書オーバーレイをSupabaseから再読み込み・マージ、
    または過去の確定キャッシュ（keiba_race_results_cache）から動的集計して最新化する。
    """
    global stats_dict, _stats_overlay
    # 1. Supabaseから永続化オーバーレイをリロード
    loaded = load_stats_overlay_from_supabase()
    rebuilt_races = 0
    updated_horses = 0

    if rebuild_from_cached_results and supabase:
        try:
            # 確定レース結果キャッシュを取得して順次適用
            res = supabase.table("keiba_race_results_cache").select("race_id, race_date, result_json").order("race_date", desc=False).execute()
            if res.data:
                for row in res.data:
                    rid = row.get("race_id")
                    rdate = row.get("race_date")
                    r_res = row.get("result_json")
                    if rid and rdate and r_res:
                        pred_row = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", rid).execute()
                        if pred_row.data and pred_row.data[0].get("data_json"):
                            p_data = pred_row.data[0]["data_json"]
                            cnt = update_stats_from_race_settlement(rid, rdate, p_data, r_res)
                            if cnt > 0:
                                rebuilt_races += 1
                                updated_horses += cnt
        except Exception as e:
            print(f"[RefreshStatsDict] キャッシュからの再構築エラー: {e}")

    return {
        "status": "ok",
        "loaded_from_supabase": loaded,
        "rebuilt_races": rebuilt_races,
        "updated_horses": updated_horses,
        "total_horses": len(stats_dict.get("horse_stats", {})),
        "overlay_horses_count": len(_stats_overlay.get("horse_stats", {})),
        "overlay_last_updated": _stats_overlay.get("updated_at")
    }


@app.post("/admin/rebuild-predictions")
def endpoint_rebuild_predictions(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)。省略時は直近10開催期間内のキャッシュ全レース"),
    sync_settlements: bool = Query(True, description="成績台帳(keiba_race_settlements)にもトラックバイアスとオッズ情報を反映する場合True")
):
    """
    最新のオッズ異常検知ロジックおよびトラックバイアス分析エンジンを
    既存のキャッシュおよび成績台帳へ再適用 (delete/upsert) して完全同期
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not connected")

    try:
        query = supabase.table("keiba_predictions_cache").select("race_id, race_date").neq("race_id", "SCHEMA_VERSION")
        if date:
            query = query.eq("race_date", date)
        else:
            cutoff = (datetime.datetime.now(JST).date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
            query = query.gte("race_date", cutoff)
        
        res = query.execute()
        rows = res.data or []
        updated_predictions = 0
        updated_settlements = 0

        for row in rows:
            r_id = row.get("race_id")
            r_date = row.get("race_date") or today_jst_str()
            if not r_id:
                continue

            try:
                # 1レースごとに個別に data_json を取得 (巨大ペイロードによる通信バッファ枯渇を完全防止)
                p_item = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", r_id).execute()
                if not p_item.data:
                    continue
                d_json = p_item.data[0].get("data_json") or {}
                horses = d_json.get("horses", [])
                if not horses:
                    continue

                # 1. オッズ変動・異常検知を再適用
                apply_odds_anomaly_indicators(horses)

                # 2. トラックバイアスを再計算
                c_details = (d_json.get("race_details") or {}).get("course_details", "")
                surf_str = "芝" if "芝" in c_details else ("ダート" if "ダ" in c_details else "障害")
                tb_info = compute_track_bias_and_matches(r_id, r_date, surf_str, horses)
                if tb_info:
                    d_json["track_bias"] = tb_info

                # 3. keiba_predictions_cache を最新JSONで更新 (delete/upsert)
                supabase.table("keiba_predictions_cache").upsert({
                    "race_id": r_id,
                    "race_date": r_date,
                    "data_json": d_json
                }).execute()
                updated_predictions += 1

                # 4. 成績台帳 (keiba_race_settlements) にも反映
                if sync_settlements:
                    try:
                        s_res = supabase.table("keiba_race_settlements").select("race_id, bet_details").eq("race_id", r_id).execute()
                        if s_res.data and len(s_res.data) > 0:
                            o_alerts = [
                                {
                                    "umaban": h.get("umaban"),
                                    "horse_name": h.get("horse_name"),
                                    "odds": h.get("odds"),
                                    "odds_trend": h.get("odds_trend"),
                                    "odds_alert": h.get("odds_alert"),
                                    "bias_match": h.get("bias_match"),
                                    "odds_history": h.get("odds_history")
                                }
                                for h in horses
                                if h.get("odds_alert") or h.get("odds_trend") or h.get("bias_match") or h.get("odds_history")
                            ]
                            patch_dict = {}
                            if tb_info:
                                patch_dict["track_bias"] = tb_info
                            if o_alerts:
                                patch_dict["odds_alerts"] = o_alerts
                            
                            b_det = s_res.data[0].get("bet_details") or {}
                            t3 = b_det.get("real_bet_meta", {}).get("top3_orders")
                            if t3:
                                patch_dict["top3_orders"] = t3
                            
                            try:
                                supabase.table("keiba_race_settlements").update(patch_dict).eq("race_id", r_id).execute()
                                updated_settlements += 1
                            except Exception:
                                # カラム未作成時のフォールバック (bet_details内)
                                if tb_info:
                                    b_det["track_bias"] = tb_info
                                if o_alerts:
                                    b_det["odds_alerts"] = o_alerts
                                supabase.table("keiba_race_settlements").update({"bet_details": b_det}).eq("race_id", r_id).execute()
                                updated_settlements += 1
                    except Exception as se:
                        pass

                time.sleep(0.08)
            except Exception as row_err:
                print(f"[Rebuild race warning] {r_id}: {row_err}")
                continue

        invalidate_summary_caches()
        return {
            "status": "ok",
            "updated_predictions": updated_predictions,
            "updated_settlements": updated_settlements,
            "message": f"{updated_predictions}件の予想キャッシュおよび{updated_settlements}件の成績台帳を最新ロジックで完全再同期しました。"
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Rebuild Fatal Error] {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Rebuild error: {e}")





@app.get("/debug/track-bias/{race_id}")
def endpoint_debug_track_bias(race_id: str):
    """トラックバイアス計算の内部動作を詳細に診断して返す"""
    if not supabase:
        return {"error": "Supabase not connected"}
    
    day_prefix = race_id[:10] if len(race_id) >= 10 else race_id[:6]
    start_rid = f"{day_prefix}01"
    
    # 1. results_cache
    res_cache = supabase.table("keiba_race_results_cache")\
        .select("race_id, result_json")\
        .gte("race_id", start_rid)\
        .lt("race_id", race_id)\
        .execute()
    results_rows = res_cache.data or []

    # 2. predictions_cache
    past_rids = [r["race_id"] for r in results_rows if r.get("race_id")]
    pred_map = {}
    if past_rids:
        try:
            p_res = supabase.table("keiba_predictions_cache")\
                .select("race_id, data_json")\
                .in_("race_id", past_rids)\
                .execute()
            pred_map = {row["race_id"]: (row.get("data_json") or {}) for row in (p_res.data or [])}
        except Exception as pe:
            print(f"[Debug] pred_map fetch err: {pe}")

    # 当該レースの情報
    this_p = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", race_id).execute()
    this_dj = (this_p.data[0].get("data_json") or {}) if this_p.data else {}
    this_course = (this_dj.get("race_details") or {}).get("course_details", "")
    
    eval_list = []
    for r in results_rows:
        rid = str(r.get("race_id") or "")
        rj = r.get("result_json") or {}
        orders = rj.get("orders") or {}
        dj = pred_map.get(rid, {})
        c_details = (dj.get("race_details") or {}).get("course_details", "")
        
        eval_list.append({
            "race_id": rid,
            "has_orders": bool(orders),
            "orders_count": len(orders),
            "course_details": c_details,
            "dj_found": bool(dj)
        })

    return {
        "target_race_id": race_id,
        "day_prefix": day_prefix,
        "start_rid": start_rid,
        "this_course_details": this_course,
        "results_count": len(results_rows),
        "eval_list": eval_list
    }


# fix_duplicated_horses_cache_2026, heal_today_weights_and_results は app/maintenance/healer.py へ移管完了


@app.post("/admin/heal-settlement-marks")
@app.get("/admin/heal-settlement-marks")
def endpoint_heal_settlement_marks(date: Optional[str] = Query(None, description="修復対象日付 (YYYY-MM-DD, 省略時は全期間)")):
    return heal_settlement_marks(date=date)


@app.post("/admin/heal-today")
@app.get("/admin/heal-today")
def endpoint_heal_today(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)"),
    force_resettle: bool = Query(False, description="全レースの結果照合・確定を強制再計算するか")
):
    return heal_today_weights_and_results(date, force_resettle=force_resettle)


@app.post("/admin/reselect-daily-real-bet")
def endpoint_reselect_daily_real_bet(target_date: Optional[str] = Query(None)):
    """当日の実戦厳選レースを再選定し、未発走レースへ移行する"""
    t_date = target_date or today_jst_str()
    _daily_top_race_id_cache.pop(t_date, None)
    _all_predictions_cache.pop(t_date, None)
    top_rid = ensure_daily_single_real_bet(t_date)
    pred = get_cached_prediction(top_rid) if top_rid else None
    return {
        "status": "ok",
        "date": t_date,
        "selected_race_id": top_rid,
        "prediction": pred
    }


def _startup_background_init():
    try:
        normalize_mislabeled_dates()
        fix_duplicated_horses_cache_2026()
        ensure_daily_single_real_bet(today_jst_str())
        endpoint_sync_active_post_times(today_jst_str())
        heal_today_weights_and_results(today_jst_str())
    except Exception as e:
        print(f"[Startup Bg Init Warning] {e}")


@app.on_event("startup")
async def startup_event():
    # DB構成（バージョン）に変更があった時のみ初期化し、通常の再起動時はデータを完全保持
    check_and_apply_db_migration()
    load_stats_overlay_from_supabase()
    load_betting_settings_from_supabase()
    asyncio.create_task(background_maintenance_loop())
    # 起動時の整合性チェック・馬体重/結果補完は非同期スレッドで実行し、API応答をブロックしない
    asyncio.create_task(asyncio.to_thread(_startup_background_init))

