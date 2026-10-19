from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict
import os
from app.core.config import settings
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
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request, Depends
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from ipat_fast_client import IPATFastClient, VENUE_CODE_MAP
from umaca_fast_client import UMACAFastClient

from app.core.security import verify_api_key
from app.core.timezone import JST, now_jst, now_jst_naive, today_jst_str, parse_post_datetime, is_before_morning_prediction
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
from app.routers.admin import (
    router as admin_router,
    set_admin_dependencies,
    endpoint_sync_active_post_times,
)
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

app = FastAPI(
    title="Keiba Prediction & Verification API",
    description="競馬AI予想 ＆ 自律型自動投票システム API",
    version="2.0.0"
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 全ルーターに X-API-Key 認証ガードを適用
app.include_router(betting_router, dependencies=[Depends(verify_api_key)])
app.include_router(admin_router, dependencies=[Depends(verify_api_key)])
app.include_router(summary_router, dependencies=[Depends(verify_api_key)])
app.include_router(batch_router, dependencies=[Depends(verify_api_key)])
app.include_router(settlement_router, dependencies=[Depends(verify_api_key)])
app.include_router(prediction_router, dependencies=[Depends(verify_api_key)])

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









# --- データベース クライアント初期化 (最速ローカル SQLite 統一) ---
from app.core.database import get_db

supabase: Optional[Any] = get_db()
if supabase is None:
    print("[Database Warning] データベースクライアントの初期化に失敗しました。")


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



@app.get("/notify/test", dependencies=[Depends(verify_api_key)])
@app.post("/notify/test", dependencies=[Depends(verify_api_key)])
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
        # インメモリ全予想キャッシュをインクリメンタル更新（全体破棄によるDB再取得スパイクを防止しゼロレイテンシー維持）
        if race_date in _all_predictions_cache:
            ts, payload = _all_predictions_cache[race_date]
            if isinstance(payload, dict) and "predictions" in payload:
                payload["predictions"][race_id] = data_dict
                _all_predictions_cache[race_date] = (time.time(), payload)
        else:
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
    target_d = race_date or today_jst_str()
    # 当日未確定レースで既にリアル勝負枠の場合は同期不要 (ただし見送り判定なのにフラグが残っている不整合は解除へ進める)
    if h_rec.get("is_real_bet_target") and target_d >= today_jst_str() and not pred_data.get("is_settled"):
        if h_rec.get("strategy_type") != "skip" or (h_rec.get("total_amount") or 0) > 0:
            return pred_data

    def _apply_venue_race_sync(r_item: dict):
        if r_item.get("is_real_bet_target"):
            h_rec["is_real_bet_target"] = True
            h_rec["real_bet_badge"] = r_item.get("real_bet_badge")
            h_rec["real_bet_level"] = r_item.get("real_bet_level", "standard")
            h_rec["is_virtual_bet_target"] = False
            h_rec["virtual_bet_slot"] = None
            h_rec["virtual_badge"] = None
        elif r_item.get("is_virtual_bet_target"):
            h_rec["is_real_bet_target"] = False
            h_rec["is_virtual_bet_target"] = True
            h_rec["virtual_bet_slot"] = r_item.get("virtual_bet_slot")
            h_rec["real_bet_level"] = r_item.get("real_bet_level", "virtual")
            h_rec["real_bet_badge"] = r_item.get("real_bet_badge")
            h_rec["virtual_badge"] = r_item.get("real_bet_badge")
        else:
            h_rec["is_real_bet_target"] = False
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
    all_cache_ttl = 300.0 if target_date == today_str else 3600.0
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

        if is_before_morning_prediction(target_date):
            # 朝08:30の一括事前予想前は、当日および未来日のバッジ（実戦・勝負・仮想・見送り等）は一切表示しない（無印状態）
            for rid, p_obj in preds.items():
                hr = p_obj.get("hybrid_recommendation")
                if hr and isinstance(hr, dict):
                    hr["is_real_bet_target"] = False
                    hr["real_bet_badge"] = None
                    hr["real_bet_level"] = "none"
                    hr["is_virtual_bet_target"] = False
                    hr["virtual_bet_slot"] = None
                    hr["virtual_badge"] = None
                    hr["is_promoted_today"] = False
        else:
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
                            if hr and isinstance(hr, dict):
                                if r.get("is_real_bet_target"):
                                    hr["is_real_bet_target"] = True
                                    hr["real_bet_badge"] = r.get("real_bet_badge")
                                    hr["real_bet_level"] = r.get("real_bet_level", "standard")
                                    hr["is_virtual_bet_target"] = False
                                    hr["virtual_bet_slot"] = None
                                    hr["virtual_badge"] = None
                                elif r.get("is_virtual_bet_target"):
                                    hr["is_real_bet_target"] = False
                                    hr["is_virtual_bet_target"] = True
                                    hr["virtual_bet_slot"] = r.get("virtual_bet_slot")
                                    hr["real_bet_level"] = r.get("real_bet_level", "virtual")
                                    hr["real_bet_badge"] = r.get("real_bet_badge")
                                    hr["virtual_badge"] = r.get("real_bet_badge")
                                else:
                                    hr["is_real_bet_target"] = False
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

        # 【整合性絶対保証】見送りレース (strategy_type == 'skip' または total_amount == 0 または recommendation.is_pass) は絶対に実馬券対象外にする
        for p_obj in preds.values():
            hr = p_obj.get("hybrid_recommendation")
            if hr and isinstance(hr, dict):
                if hr.get("strategy_type") == "skip" or (hr.get("total_amount") or 0) == 0 or p_obj.get("recommendation", {}).get("is_pass"):
                    hr["is_real_bet_target"] = False
                    hr["real_bet_badge"] = None
                    hr["real_bet_level"] = "none"

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


# --- 管理者向けエンドポイント群は app/routers/admin.py へ集約完了 ---
set_admin_dependencies(
    get_cached_prediction=get_cached_prediction,
    all_predictions_cache=_all_predictions_cache,
)


def _clear_morning_unannounced_badges():
    """朝08:30前に誤って付与された当日レースのバッジ（実戦・勝負・仮想・見送り等）をSupabaseキャッシュおよびメモリから解除"""
    today_s = today_jst_str()
    if is_before_morning_prediction(today_s):
        _daily_top_race_id_cache.pop(today_s, None)
        _all_predictions_cache.pop(today_s, None)
        _today_races_cache.pop(today_s, None)
        if supabase:
            try:
                res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", today_s).execute()
                for row in (res.data or []):
                    rid = row.get("race_id")
                    dj = row.get("data_json") or {}
                    hrec = dj.get("hybrid_recommendation")
                    if hrec and (hrec.get("is_real_bet_target") or hrec.get("real_bet_badge") or hrec.get("is_virtual_bet_target") or hrec.get("virtual_badge")):
                        hrec["is_real_bet_target"] = False
                        hrec["real_bet_badge"] = None
                        hrec["real_bet_level"] = "none"
                        hrec["is_virtual_bet_target"] = False
                        hrec["virtual_bet_slot"] = None
                        hrec["virtual_badge"] = None
                        hrec["is_promoted_today"] = False
                        supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", rid).execute()
                        print(f"[ClearMorningBadges] {rid} の朝発表前バッジを解除しました。")
            except Exception as e:
                print(f"[ClearMorningBadges Warning] {e}")


def _startup_background_init():
    try:
        normalize_mislabeled_dates()
        fix_duplicated_horses_cache_2026()
        _clear_morning_unannounced_badges()
        # 過去日（10/17等）の確定バッジ整合を確実に復元
        try:
            raw_v_1017 = fetch_venues_for_date_raw("2026-10-17")
            if raw_v_1017:
                attach_prediction_badges_to_venues("2026-10-17", raw_v_1017)
        except Exception as e_1017:
            print(f"[Startup 10/17 Heal Warning] {e_1017}")
        if not is_before_morning_prediction(today_jst_str()):
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

