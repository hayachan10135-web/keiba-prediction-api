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




# --- 開催期間（土曜日 〜 次週金曜日 20:00）管理ヘルパー ---
_period_venues_cache: Dict[str, Any] = {}
_period_checked_empty_dates: set = set()
_latest_post_times: Dict[str, str] = {}  # {race_id: "09:50発走"} (スクレイピングした実際の最新出走時刻)
_last_weight_check_ts: Dict[str, float] = {}  # {race_id: timestamp} (馬体重発表チェックのクールダウン管理)


def get_current_kaisai_period(now_dt: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """
    1開催期間（土曜日 00:00 〜 次週金曜日 24:00）の境界を算出。
    ※金曜夜〜土曜朝の使用不可待機期間は完全撤廃し、常時24時間利用可能 (is_waiting_period=False)。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    today_d = now_dt.date()
    weekday = today_d.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    # 待機期間・使用不可期間は完全撤廃（常時利用可能）
    is_waiting = False

    days_since_sat = (weekday - 5) % 7
    period_start_d = today_d - datetime.timedelta(days=days_since_sat)
    period_end_d = period_start_d + datetime.timedelta(days=6)

    return {
        "period_start": period_start_d.strftime("%Y-%m-%d"),
        "period_end": period_end_d.strftime("%Y-%m-%d"),
        "period_start_date": period_start_d,
        "period_end_date": period_end_d,
        "today_str": today_d.strftime("%Y-%m-%d"),
        "today_date": today_d,
        "is_waiting_period": is_waiting
    }


def clear_kaisai_period_caches():
    """金曜20:00の開催期間キャッシュ整理（直近10開催期間＝約75日間は完全保持し、それ以前の過去分のみ安全破棄）"""
    global _period_venues_cache, _period_checked_empty_dates, _latest_post_times, _last_weight_check_ts
    _period_venues_cache.clear()
    _period_checked_empty_dates.clear()
    _latest_post_times.clear()
    _last_weight_check_ts.clear()
    if not supabase:
        return
    try:
        # 直近10開催期間（約75日間）より古いキャッシュのみを削除（10開催分は完全保持）
        cutoff_date = (datetime.datetime.now(JST).date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
        supabase.table("keiba_predictions_cache").delete().neq("race_id", "SCHEMA_VERSION").lt("race_date", cutoff_date).execute()
        supabase.table("keiba_race_results_cache").delete().neq("race_id", "0").lt("race_date", cutoff_date).execute()
        supabase.table("keiba_scheduled_races").delete().neq("race_id", "0").lt("race_date", cutoff_date).execute()
        print(f"[Cleanup] 金曜20:00 キャッシュ整理完了: {cutoff_date} 以前の古いキャッシュのみ破棄しました（直近10開催期間分は完全保持）。")
    except Exception as e:
        print(f"[Cleanup] キャッシュ破棄エラー: {e}")
    invalidate_summary_caches()


# --- サマリー・総合収支のインメモリキャッシュ (高速化用) ---
# --- スケジュール登録ヘルパー ---
def register_race_schedule(race_id: str, race_date: str, race_name: str, post_time_str: str):
    if not supabase:
        return
    try:
        m = re.search(r"(\d{1,2}):(\d{2})", post_time_str)
        if not m:
            return
        hour, minute = int(m.group(1)), int(m.group(2))

        d_parts = [int(p) for p in race_date.split("-")]
        post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
        post_dt_str = post_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 既存レコードがある場合は is_settled / is_predicted を上書きリセットしない
        existing = supabase.table("keiba_scheduled_races").select("race_id, is_settled, is_predicted").eq("race_id", race_id).execute()
        if existing.data and len(existing.data) > 0:
            supabase.table("keiba_scheduled_races").update({
                "race_date": race_date,
                "race_name": race_name,
                "post_datetime": post_dt_str
            }).eq("race_id", race_id).execute()
        else:
            supabase.table("keiba_scheduled_races").insert({
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "post_datetime": post_dt_str,
                "is_settled": 0,
                "is_predicted": 0
            }).execute()
    except Exception as e:
        print(f"Supabaseスケジュール登録エラー ({race_id}): {e}")


def ensure_schedules_registered_for_venues(formatted_date: str, venues: List[dict]):
    """レース一覧取得時に実際の最新発走時刻を keiba_scheduled_races および _latest_post_times に同期（発走時刻変更対応）"""
    if not venues:
        return
    try:
        d_parts = [int(p) for p in formatted_date.split("-")]
        updates_for_schedule = []
        for v in venues:
            for r in v.get("races", []):
                r_id = r.get("race_id")
                if not r_id:
                    continue
                r_info = r.get("race_info") or r.get("race_name") or ""
                m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                if m:
                    hour, minute = int(m.group(1)), int(m.group(2))
                    post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
                    post_dt_str = post_dt.strftime("%Y-%m-%d %H:%M:%S")
                    post_time_str = f"{hour:02d}:{minute:02d}発走"
                    _latest_post_times[r_id] = post_time_str
                    updates_for_schedule.append({
                        "race_id": r_id,
                        "race_date": formatted_date,
                        "race_name": r.get("race_name", r.get("race_no", "")),
                        "post_datetime": post_dt_str,
                        "post_time_str": post_time_str
                    })

        if supabase and updates_for_schedule:
            rids_list = [row["race_id"] for row in updates_for_schedule]
            existing_rids = set()
            try:
                ex_res = supabase.table("keiba_scheduled_races").select("race_id").in_("race_id", rids_list).execute()
                existing_rids = {er["race_id"] for er in (ex_res.data or []) if "race_id" in er}
            except Exception:
                pass
            for row in updates_for_schedule:
                try:
                    if row["race_id"] in existing_rids:
                        supabase.table("keiba_scheduled_races").update({
                            "race_date": row["race_date"],
                            "post_datetime": row["post_datetime"],
                            "race_name": row["race_name"]
                        }).eq("race_id", row["race_id"]).execute()
                    else:
                        supabase.table("keiba_scheduled_races").insert({
                            "race_id": row["race_id"],
                            "race_date": row["race_date"],
                            "race_name": row["race_name"],
                            "post_datetime": row["post_datetime"],
                            "is_predicted": 0,
                            "is_settled": 0,
                            "retry_count": 0
                        }).execute()
                except Exception:
                    pass
    except Exception as e:
        print(f"スケジュール一括同期エラー ({formatted_date}): {e}")


# =========================================================================
# ★ スケジューラー＆クローラー連携コールバック登録
# =========================================================================
def _on_crawler_post_time(r_id: str, pt_str: str):
    _latest_post_times[r_id] = pt_str

def _on_crawler_schedule_registered(r_id: str, target_date: str, r_name: str, pt_str: str):
    register_race_schedule(r_id, target_date, r_name, pt_str)

def _on_crawler_venues_loaded(formatted_date: str, venues: List[dict]):
    ensure_schedules_registered_for_venues(formatted_date, venues)

register_crawler_post_time_handler(_on_crawler_post_time)
register_venues_loaded_handler(_on_crawler_venues_loaded)
register_post_time_callback(_on_crawler_post_time)
register_schedule_callback(_on_crawler_schedule_registered)



_period_kaisai_dates_cache: Dict[str, Tuple[float, List[str]]] = {}
_today_races_cache: Dict[str, Tuple[float, dict]] = {}


def discover_period_kaisai_dates(now_dt: Optional[datetime.datetime] = None) -> List[str]:
    """
    現在の1開催期間（土曜日〜次週金曜日20:00）における、本日までのJRA開催日リスト(昇順)を返す。
    金曜20:00〜土曜朝7:30の待機期間中は空リストを返す。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)

    now_ts = time.time()
    today_k = p_info["today_str"]
    c_dates = _period_kaisai_dates_cache.get(today_k)
    if c_dates and (now_ts - c_dates[0] < 300.0):
        return list(c_dates[1])

    # 過去10開催期間（約75日間）のDBに保存されている開催日を取得
    cutoff_d = (now_dt.date() - datetime.timedelta(days=75)).strftime("%Y-%m-%d")
    past_db_dates = set()
    if supabase:
        try:
            # 1. scheduled_races
            s_res = supabase.table("keiba_scheduled_races").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).execute()
            for row in (s_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
            # 2. predictions_cache
            p_res = supabase.table("keiba_predictions_cache").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).neq("race_id", "SCHEMA_VERSION").execute()
            for row in (p_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
            # 3. race_settlements (成績台帳)
            st_res = supabase.table("keiba_race_settlements").select("race_date").gte("race_date", cutoff_d).lte("race_date", p_info["today_str"]).execute()
            for row in (st_res.data or []):
                if row.get("race_date"):
                    past_db_dates.add(row["race_date"])
        except Exception:
            pass

    start_d = p_info["period_start_date"]
    today_d = p_info["today_date"]

    available_dates = list(past_db_dates)
    cur_d = start_d
    while cur_d <= today_d:
        d_str = cur_d.strftime("%Y-%m-%d")

        if d_str in _period_venues_cache:
            available_dates.append(d_str)
        elif d_str in _period_checked_empty_dates:
            pass
        elif cur_d.weekday() in (5, 6, 0) or d_str in past_db_dates:
            # 土・日・月（祝日3日間開催等）またはDB登録済みの日を走査
            v = fetch_venues_for_date_raw(d_str)
            if v:
                available_dates.append(d_str)
        else:
            # 火・水・木・金の通常平日は無駄な通信を避けるため空として記録
            _period_checked_empty_dates.add(d_str)

        cur_d += datetime.timedelta(days=1)

    # 金曜夕方（17:00以降）であれば翌日（土曜）の出走表確定状況も先行走査
    if today_d.weekday() == 4 and now_dt.hour >= 17:
        tomorrow_d = today_d + datetime.timedelta(days=1)
        tom_str = tomorrow_d.strftime("%Y-%m-%d")
        if tom_str in _period_venues_cache:
            available_dates.append(tom_str)
        elif tom_str not in _period_checked_empty_dates:
            v = fetch_venues_for_date_raw(tom_str)
            if v:
                available_dates.append(tom_str)

    res_dates = sorted(list(set(available_dates)))[-6:]
    _period_kaisai_dates_cache[today_k] = (now_ts, res_dates)
    return res_dates


def attach_prediction_badges_to_venues(formatted_date: str, venues: List[dict]) -> List[dict]:
    """レース一覧の各レースにSupabase上の予想・実戦推奨バッジ情報および確定・的中結果を付与"""
    if not venues:
        return []
    cached_preds_map = {}
    raw_preds_cache = {}
    settled_map = {}
    if supabase:
        try:
            p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
            for row in (p_res.data or []):
                d_json = row.get("data_json") or {}
                raw_preds_cache[row["race_id"]] = d_json
                h_rec = d_json.get("hybrid_recommendation")
                r_name = d_json.get("race_name", "")
                if h_rec:
                    cached_preds_map[row["race_id"]] = {
                        "is_real_bet_target": h_rec.get("is_real_bet_target", False),
                        "real_bet_badge": h_rec.get("real_bet_badge"),
                        "real_bet_level": h_rec.get("real_bet_level", "none"),
                        "edge_ratio": h_rec.get("edge_ratio"),
                        "is_final_odds": h_rec.get("is_final_odds", False),
                        "confidence_score": float(h_rec.get("confidence_score") or 0.0),
                        "is_virtual_bet_target": h_rec.get("is_virtual_bet_target", False),
                        "virtual_bet_slot": h_rec.get("virtual_bet_slot"),
                        "virtual_badge": h_rec.get("virtual_badge"),
                        "race_name": r_name
                    }
        except Exception as e:
            print(f"レース一覧キャッシュ付与エラー: {e}")

        try:
            s_res = supabase.table("keiba_race_settlements").select("race_id, is_hit, is_pass, total_payout").eq("race_date", formatted_date).execute()
            for s_row in (s_res.data or []):
                is_hit_val = 1 if (s_row.get("is_hit") == 1 and s_row.get("is_pass") == 0) else 0
                settled_map[s_row["race_id"]] = {
                    "is_settled": 1,
                    "is_hit": is_hit_val,
                    "is_pass": s_row.get("is_pass", 0),
                    "total_payout": s_row.get("total_payout", 0)
                }
        except Exception as se:
            print(f"レース一覧確定情報付与エラー: {se}")

        # 実馬券厳選保証: 自然条件(戦略B/A/C/A-2)を満たすレースは全て実馬券とし、0レースの時のみ当日No.1の1レースを実馬券に昇格
        natural_real_count = sum(1 for v in cached_preds_map.values() if v.get("is_real_bet_target"))
        has_downgraded_ref = any("参考推奨" in str(v.get("real_bet_badge") or "") for v in cached_preds_map.values())
        if (natural_real_count == 0 or has_downgraded_ref) and formatted_date >= today_jst_str() and cached_preds_map:
            ensure_daily_single_real_bet(formatted_date)
            try:
                p_res2 = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
                for row in (p_res2.data or []):
                    d_json = row.get("data_json") or {}
                    h_rec = d_json.get("hybrid_recommendation")
                    if h_rec and row["race_id"] in cached_preds_map:
                        cached_preds_map[row["race_id"]]["is_real_bet_target"] = h_rec.get("is_real_bet_target", False)
                        cached_preds_map[row["race_id"]]["real_bet_badge"] = h_rec.get("real_bet_badge")
                        cached_preds_map[row["race_id"]]["real_bet_level"] = h_rec.get("real_bet_level", "none")
            except Exception:
                pass

    for venue in venues:
        races_in_venue = venue.get("races", [])
        for race in races_in_venue:
            info = cached_preds_map.get(race["race_id"])
            if info:
                race["is_real_bet_target"] = info["is_real_bet_target"]
                badge_str = info["real_bet_badge"]
                if badge_str and isinstance(badge_str, str):
                    badge_str = badge_str.replace("🎯", "⭐")
                race["real_bet_badge"] = badge_str
                race["real_bet_level"] = info.get("real_bet_level", "none")
                race["edge_ratio"] = info["edge_ratio"]
                race["is_final_odds"] = info["is_final_odds"]
                race["confidence_score"] = info.get("confidence_score", 0.0)
                race["is_virtual_bet_target"] = info.get("is_virtual_bet_target", False)
                race["virtual_bet_slot"] = info.get("virtual_bet_slot")
            else:
                race["is_real_bet_target"] = False
                race["real_bet_badge"] = None
                race["real_bet_level"] = "none"
                race["edge_ratio"] = None
                race["is_final_odds"] = False
                race["confidence_score"] = 0.0
                race["is_virtual_bet_target"] = False
                race["virtual_bet_slot"] = None

            s_info = settled_map.get(race["race_id"])
            if s_info:
                race["is_settled"] = s_info.get("is_settled", 0)
                race["is_hit"] = s_info.get("is_hit", 0)
                race["is_pass"] = s_info.get("is_pass", 0)
            else:
                race["is_settled"] = 0
                race["is_hit"] = 0
                race["is_pass"] = 0

        # 各会場ごとに確信度上位3レースを「仮想馬券枠 (各場3R)」として保証
        # （予想済み かつ 平地・新馬戦・障害戦除くレースの中で確信度が高い順）
        eligible_for_virtual = []
        for r in races_in_venue:
            if float(r.get("confidence_score") or 0.0) <= 0.0:
                continue
            r_nm = r.get("race_name", "")
            r_info = r.get("race_info", "")
            is_shinba = ("新馬" in r_nm) or ("メイクデビュー" in r_nm)
            is_obstacle = ("障害" in r_nm) or ("障" in r_info)
            if not is_shinba and not is_obstacle:
                eligible_for_virtual.append(r)

        # 確信度順にソートして上位3レースを選出
        eligible_for_virtual.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        top3_virtual = eligible_for_virtual[:3]
        top3_virtual_ids = set()
        for slot_idx, v_race in enumerate(top3_virtual, start=1):
            v_race["is_virtual_bet_target"] = True
            v_race["virtual_bet_slot"] = slot_idx
            vr_id = v_race.get("race_id")
            if vr_id:
                top3_virtual_ids.add(vr_id)
            # リアル勝負枠でない場合は「仮想観戦」バッジとレベルを付与
            if not v_race.get("is_real_bet_target"):
                v_race["real_bet_level"] = "virtual"
                v_race["real_bet_badge"] = f"🎮【仮想観戦】第{slot_idx}推奨"
                # キャッシュDB (keiba_predictions_cache) 側も完全同期
                if vr_id and vr_id in raw_preds_cache:
                    c_d = raw_preds_cache[vr_id]
                    c_hrec = c_d.get("hybrid_recommendation")
                    if c_hrec and (
                        c_hrec.get("real_bet_level") != "virtual"
                        or c_hrec.get("real_bet_badge") != v_race["real_bet_badge"]
                        or c_hrec.get("virtual_badge") != v_race["real_bet_badge"]
                    ):
                        c_hrec["is_virtual_bet_target"] = True
                        c_hrec["virtual_bet_slot"] = slot_idx
                        c_hrec["real_bet_level"] = "virtual"
                        c_hrec["real_bet_badge"] = v_race["real_bet_badge"]
                        c_hrec["virtual_badge"] = v_race["real_bet_badge"]
                        if supabase:
                            try:
                                supabase.table("keiba_predictions_cache").update({"data_json": c_d}).eq("race_id", vr_id).execute()
                            except Exception as ue:
                                print(f"仮想枠キャッシュ同期エラー ({vr_id}): {ue}")

        # 会場内でトップ3から漏れた全レース（新馬・障害含む）の仮想枠ステータスを「見送り(none)」に完全整合
        for other_r in races_in_venue:
            or_id = other_r.get("race_id")
            if or_id and or_id not in top3_virtual_ids and not other_r.get("is_real_bet_target"):
                other_r["is_virtual_bet_target"] = False
                other_r["virtual_bet_slot"] = None
                other_r["real_bet_level"] = "none"
                other_r["real_bet_badge"] = None
                if or_id in raw_preds_cache:
                    c_d = raw_preds_cache[or_id]
                    c_hrec = c_d.get("hybrid_recommendation")
                    if c_hrec and (
                        c_hrec.get("real_bet_level") == "virtual"
                        or c_hrec.get("is_virtual_bet_target")
                        or c_hrec.get("virtual_badge") is not None
                        or c_hrec.get("real_bet_badge") is not None
                    ):
                        c_hrec["is_virtual_bet_target"] = False
                        c_hrec["virtual_bet_slot"] = None
                        c_hrec["real_bet_level"] = "none"
                        c_hrec["real_bet_badge"] = None
                        c_hrec["virtual_badge"] = None
                        if supabase:
                            try:
                                supabase.table("keiba_predictions_cache").update({"data_json": c_d}).eq("race_id", or_id).execute()
                            except Exception:
                                pass

    return venues


def get_today_races(
    date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)"),
    force_refresh: bool = Query(False, description="手動でレース一覧・発走時刻を再取得する場合True")
):
    now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)

    if date:
        clean_d = date.replace("-", "")
        target_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else date
    else:
        target_date = p_info["today_str"]

    now_ts = time.time()
    cache_ttl = 30.0 if target_date == p_info["today_str"] else 3600.0
    if not force_refresh:
        c_val = _today_races_cache.get(target_date)
        if c_val and (now_ts - c_val[0] < cache_ttl):
            return c_val[1]

    available_dates = discover_period_kaisai_dates(now_dt)

    if not date:
        if not available_dates:
            return {
                "date": "",
                "available_dates": [],
                "is_waiting_period": False,
                "message": "現在表示可能な開催データがありません。",
                "venues": []
            }
        # 本日が開催日なら本日、平日の場合は直近（最新）の開催日を初期選択
        if p_info["today_str"] in available_dates:
            target_date = p_info["today_str"]
        else:
            target_date = available_dates[-1]

    venues = fetch_venues_for_date_raw(target_date, force_refresh=force_refresh)
    if venues:
        if target_date not in available_dates:
            available_dates = sorted(list(set(available_dates + [target_date])))
        venues = attach_prediction_badges_to_venues(target_date, venues)
        resp = {
            "date": target_date,
            "available_dates": available_dates[-6:],
            "is_waiting_period": False,
            "venues": venues
        }
        _today_races_cache[target_date] = (now_ts, resp)
        return resp

    resp_empty = {
        "date": target_date,
        "available_dates": available_dates[-6:],
        "is_waiting_period": False,
        "message": "指定された開催日のレース情報が見つかりませんでした。",
        "venues": []
    }
    _today_races_cache[target_date] = (now_ts, resp_empty)
    return resp_empty



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

def run_batch_worker(
    target_date: Optional[str] = None,
    mode: str = "all",
    force_predict: bool = False,
    is_19h_settle: bool = False
):
    global batch_status
    batch_status["is_running"] = True
    batch_status["current"] = 0
    batch_status["predicted"] = 0
    batch_status["settled"] = 0
    batch_status["message"] = "開催期間内の対象レースを走査中..."

    try:
        now_dt = now_jst_naive()
        today_str = today_jst_str()
        p_info = get_current_kaisai_period(now_dt)

        period_dates = discover_period_kaisai_dates(now_dt)

        # target_date が明示されている場合は指定日のみを対象とする
        if target_date:
            clean_d = target_date.replace("-", "")
            fmt_d = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else target_date
            target_dates = [fmt_d]
        elif mode == "all" and not is_19h_settle:
            target_dates = period_dates if period_dates else []
        else:
            target_dates = period_dates

        if not target_dates:
            batch_status["total"] = 0
            batch_status["message"] = "今開催期間の対象レースが見つかりませんでした。"
            return

        # 対象開催日の全レースを収集
        all_period_races = []
        for d_str in target_dates:
            venues = fetch_venues_for_date_raw(d_str)
            for v in venues:
                for r in v.get("races", []):
                    all_period_races.append((d_str, r))

        if not all_period_races:
            batch_status["total"] = 0
            batch_status["message"] = "開催期間内に処理可能なレースが見つかりませんでした。"
            return

        # Supabaseからすでに予想済み・検証済みのrace_idを一括取得して差分判定を高速化
        predicted_ids = set()
        settled_ids = set()
        if supabase:
            try:
                p_res = supabase.table("keiba_predictions_cache").select("race_id").in_("race_date", target_dates).execute()
                predicted_ids = {row["race_id"] for row in (p_res.data or [])}
            except Exception as e:
                print(f"[Batch] 予想済みID一括取得エラー: {e}")

            try:
                s_res = supabase.table("keiba_race_settlements").select("race_id, actual_order, is_pass").in_("race_date", target_dates).execute()
                for row in (s_res.data or []):
                    if row.get("actual_order") is not None or row.get("is_pass") == 1 or row.get("is_pass") is True:
                        settled_ids.add(row["race_id"])
            except Exception as e:
                print(f"[Batch] 検証済みID一括取得エラー: {e}")

        # 実際に未予想または未検証のレースだけを抽出
        work_items = []
        for d_str, race_item in all_period_races:
            r_id = race_item["race_id"]
            r_info = race_item.get("race_info", "")

            do_predict = False
            if mode in ("all", "predict"):
                if force_predict or (r_id not in predicted_ids):
                    do_predict = True

            do_settle = False
            if mode in ("all", "settle") and (r_id not in settled_ids):
                # 過去開催日のレースはすべて発走済み、当日開催分は発走時刻後のみ検証対象
                is_past_post = True
                if d_str > today_str:
                    is_past_post = False
                elif d_str == today_str and not is_19h_settle:
                    m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                    if m:
                        try:
                            p_hour, p_minute = int(m.group(1)), int(m.group(2))
                            p_dt = now_dt.replace(hour=p_hour, minute=p_minute, second=0, microsecond=0)
                            if now_dt < p_dt:
                                is_past_post = False
                        except Exception:
                            pass
                if is_past_post:
                    do_settle = True

            if do_predict or do_settle:
                work_items.append((d_str, race_item, do_predict, do_settle))

        batch_status["total"] = len(work_items)

        if not work_items:
            batch_status["message"] = f"完了: 今開催期間(全{len(all_period_races)}R)はすべて予想・検証済みです"
            return

        for idx, (d_str, race_item, do_predict, do_settle) in enumerate(work_items, 1):
            r_id = race_item["race_id"]
            batch_status["current"] = idx
            batch_status["message"] = f"処理中 ({idx}/{len(work_items)}件: 予想{batch_status['predicted']}件 / 検証{batch_status['settled']}件)"

            if do_predict:
                try:
                    predict_race_core(r_id, force_refresh=force_predict, is_auto_final=False)
                    batch_status["predicted"] += 1
                    predicted_ids.add(r_id)
                except Exception as pe:
                    print(f"[Batch] 予想スキップ/エラー ({r_id}): {pe}")

            if do_settle:
                try:
                    verify_race_result(r_id, is_19h_settle=is_19h_settle)
                    batch_status["settled"] += 1
                    settled_ids.add(r_id)
                except Exception:
                    pass

        batch_status["message"] = f"完了: 対象{len(work_items)}R処理 (新規予想{batch_status['predicted']}件 / 新規検証{batch_status['settled']}件)"

    except Exception as e:
        batch_status["message"] = f"エラー終了: {e}"
        print(f"[Batch] ワーカー全体エラー: {e}")
    finally:
        batch_status["is_running"] = False
        invalidate_summary_caches()


def recalculate_period_records():
    """今開催期間のキャッシュ済み予想・確定成績に最新の買い目・実戦推奨ロジックを再適用して一括更新"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase未接続")

    preds_res = supabase.table("keiba_predictions_cache").select("race_id, race_date, data_json").execute()
    results_res = supabase.table("keiba_race_results_cache").select("race_id, race_date, result_json").execute()

    preds_list = preds_res.data or []
    res_map = {r["race_id"]: r.get("result_json") for r in (results_res.data or [])}
    valid_rids = set(res_map.keys())

    updated_preds = 0
    updated_settlements = 0
    participating_count = 0
    total_bet_sum = 0
    total_pay_sum = 0
    total_pass_benefit = 0
    rb_count = 0
    rb_bet_sum = 0
    rb_pay_sum = 0

    for row in preds_list:
        rid = row["race_id"]
        rdate = row.get("race_date") or today_jst_str()
        pred_data = row.get("data_json") or {}
        if not pred_data:
            continue

        rname = pred_data.get("race_name", "")
        rdetails = pred_data.get("race_details") or {}
        horses = pred_data.get("horses") or []
        old_h = pred_data.get("hybrid_recommendation") or {}

        c_details = rdetails.get("course_details", "") if isinstance(rdetails, dict) else ""
        surf = 0
        if "障" in c_details or "障害" in rname:
            surf = 2
        elif "ダ" in c_details:
            surf = 1

        honmei = next((h for h in horses if h.get("mark") == "◎"), None)
        if not honmei and horses:
            honmei = sorted(horses, key=lambda x: x.get("score", 0), reverse=True)[0]
        u0 = int(honmei["umaban"]) if honmei else 0

        # 既存の馬連オッズ情報（実オッズ含む）を復元
        old_bets = old_h.get("hypothetical_umaren_bets") or old_h.get("umaren_bets") or []
        odds_by_opp = {}
        for ob in old_bets:
            comb = ob.get("combination", [])
            if len(comb) == 2:
                opp_u = comb[1] if comb[0] == u0 else comb[0]
                odds_by_opp[int(opp_u)] = (float(ob.get("odds") or 0.0), bool(ob.get("is_real_odds", True)))

        opponents = []
        for mark_sym in ["◯", "▲", "△"]:
            h_opp = next((h for h in horses if h.get("mark") == mark_sym or (mark_sym == "◯" and h.get("mark") in ("〇", "○"))), None)
            if h_opp:
                ou = int(h_opp["umaban"])
                if ou in odds_by_opp:
                    o_val, o_real = odds_by_opp[ou]
                else:
                    oa = float(honmei.get("odds") or 0.0) if honmei else 0.0
                    ob_val = float(h_opp.get("odds") or 0.0)
                    o_val = round(max(2.5, 0.65 * oa * ob_val), 1) if (oa > 0 and ob_val > 0) else 0.0
                    o_real = False
                opponents.append((ou, o_val, o_real, mark_sym))

        # 既存の三連複実オッズを復元
        old_t_bets = old_h.get("trio_bets") or ([old_h.get("trio_bet")] if old_h.get("trio_bet") else [])
        real_trio_map = {}
        for tb in old_t_bets:
            if isinstance(tb, dict):
                c = tb.get("combination") or []
                if len(c) == 3 and tb.get("is_real_odds"):
                    real_trio_map[tuple(sorted(c))] = float(tb.get("odds") or 0.0)

        new_h = compute_hybrid_rec_from_inputs(
            race_name=rname,
            race_details=rdetails,
            horses=horses,
            surface_type=surf,
            prob_gachi=float(old_h.get("prob_gachi") or 0.0),
            prob_himo=float(old_h.get("prob_himo") or 0.0),
            opponents=opponents,
            u0=u0,
            honmei_horse=honmei,
            race_date_str=rdate,
            force_final_odds=bool(old_h.get("is_final_odds", True)),
            real_trio_odds=real_trio_map
        )

        # 三連複対象レースで実オッズが未取得ならnetkeibaから取得して反映
        if new_h.get("trio_bets") and not real_trio_map:
            fetched_trio = fetch_real_odds(rid).get("trio", {})
            if fetched_trio:
                new_h = compute_hybrid_rec_from_inputs(
                    race_name=rname,
                    race_details=rdetails,
                    horses=horses,
                    surface_type=surf,
                    prob_gachi=float(old_h.get("prob_gachi") or 0.0),
                    prob_himo=float(old_h.get("prob_himo") or 0.0),
                    opponents=opponents,
                    u0=u0,
                    honmei_horse=honmei,
                    race_date_str=rdate,
                    force_final_odds=bool(old_h.get("is_final_odds", True)),
                    real_trio_odds=fetched_trio
                )

        pred_data["hybrid_recommendation"] = new_h

        if ("新馬" in rname) or ("メイクデビュー" in rname) or surf == 2:
            rec_obj = pred_data.get("recommendation") or {}
            rec_obj["grade"] = "D"
            rec_obj["confidence_label"] = "見送り推奨"
            rec_obj["is_pass"] = True
            rec_obj["tansho_amount"] = 0
            rec_obj["fukusho_amount"] = 0
            rec_obj["total_amount"] = 0
            pred_data["recommendation"] = rec_obj

        save_cached_prediction(rid, rdate, pred_data)
        updated_preds += 1

        res_json = res_map.get(rid)
        if res_json:
            st_res = process_settlement_calculation(
                race_id=rid,
                race_date=rdate,
                race_name=rname,
                pred_data=pred_data,
                race_res=res_json,
                from_cache=True
            )
            updated_settlements += 1
            if not st_res.get("is_pass"):
                participating_count += 1
                total_bet_sum += st_res.get("total_bet", 0)
                total_pay_sum += st_res.get("total_payout", 0)
                if st_res.get("is_real_bet_target"):
                    rb_count += 1
                    rb_bet_sum += st_res.get("total_bet", 0)
                    rb_pay_sum += st_res.get("total_payout", 0)
            else:
                total_pass_benefit += (st_res.get("pass_benefit") or 0)

    _period_venues_cache.clear()
    roi = round((total_pay_sum / total_bet_sum) * 100.0, 2) if total_bet_sum > 0 else 0.0
    rb_roi = round((rb_pay_sum / rb_bet_sum) * 100.0, 2) if rb_bet_sum > 0 else 0.0
    return {
        "status": "ok",
        "updated_predictions": updated_preds,
        "updated_settlements": updated_settlements,
        "participating_races": participating_count,
        "total_bet": total_bet_sum,
        "total_payout": total_pay_sum,
        "profit": total_pay_sum - total_bet_sum,
        "recovery_rate": roi,
        "real_bet_races": rb_count,
        "real_bet_bet": rb_bet_sum,
        "real_bet_payout": rb_pay_sum,
        "real_bet_profit": rb_pay_sum - rb_bet_sum,
        "real_bet_recovery_rate": rb_roi,
        "pass_benefit_total": total_pass_benefit
    }

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

async def sync_daily_schedules_if_needed(today_str: str):
    """当日のJRA開催有無を確認し、開催日であればレーススケジュールをkeiba_scheduled_racesに事前登録"""
    if not supabase:
        return
    if today_str in _period_checked_empty_dates:
        return
    try:
        res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", today_str).execute()
        valid_ids = [r["race_id"] for r in (res.data or []) if len(r["race_id"]) == 12]
        if len(valid_ids) >= 20:
            return

        venues = fetch_venues_for_date_raw(today_str)
        if venues:
            print(f"[Scheduler] {today_str} (JST) のJRA開催スケジュール ({sum(len(v.get('races', [])) for v in venues)}R) を事前登録しました。")
    except Exception as e:
        print(f"[Scheduler] スケジュール事前登録エラー: {e}")


async def background_maintenance_loop():
    print("[Scheduler] リアルタイム・メンテナンススケジューラー (JST同期・1開催期間サイクル版) を開始しました。")
    last_friday_cleaned_period = None
    last_17h_settle_date = None
    last_19h_settle_date = None
    last_830_predict_date = None

    while True:
        try:
            now_dt = now_jst_naive()
            today_str = today_jst_str()
            p_info = get_current_kaisai_period(now_dt)

            # --- ⑥ 金曜 20:00 以降の古いキャッシュ自動整理 (75日以前の過去分のみ安全破棄) ---
            if now_dt.weekday() == 4 and now_dt.hour >= 20:
                if last_friday_cleaned_period != p_info["period_start"]:
                    clear_kaisai_period_caches()
                    last_friday_cleaned_period = p_info["period_start"]
                    last_830_predict_date = None
                    last_17h_settle_date = None
                    last_19h_settle_date = None

            # --- ① JST朝 7:30 以降、当日のレーススケジュールを事前登録 ---
            if now_dt.hour > 7 or (now_dt.hour == 7 and now_dt.minute >= 30):
                await sync_daily_schedules_if_needed(today_str)

            # --- ② JST朝 8:30 以降、当日の未予想レース事前予想（朝オッズによる暫定判定）を一括実行 ---
            if (now_dt.hour > 8 or (now_dt.hour == 8 and now_dt.minute >= 30)) and last_830_predict_date != today_str:
                period_dates = discover_period_kaisai_dates(now_dt)
                if today_str in period_dates:
                    already_predicted_count = 0
                    if supabase:
                        try:
                            chk_p = supabase.table("keiba_predictions_cache").select("race_id").eq("race_date", today_str).execute()
                            already_predicted_count = len(chk_p.data or [])
                        except Exception:
                            pass
                    if already_predicted_count >= 24:
                        print(f"[Scheduler] 当日({today_str})の事前予想は既に完了済み({already_predicted_count}R)のため朝8:30一括予想をスキップします。")
                        await asyncio.to_thread(ensure_daily_single_real_bet, today_str)
                        last_830_predict_date = today_str
                    elif not batch_status["is_running"]:
                        print(f"[Scheduler] 朝8:30 当日({today_str})全レースの一括事前予想（暫定判定）を実行します...")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "predict", False, False))
                        await asyncio.to_thread(ensure_daily_single_real_bet, today_str)
                        last_830_predict_date = today_str

                        # 朝の実戦推奨候補を走査してダイジェスト通知
                        try:
                            if supabase:
                                p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", today_str).execute()
                                real_races = []
                                for row in (p_res.data or []):
                                    h_rec = row.get("data_json", {}).get("hybrid_recommendation") or {}
                                    if h_rec.get("is_real_bet_target"):
                                        r_name = row.get("data_json", {}).get("race_name", row["race_id"])
                                        b_txt = h_rec.get("real_bet_badge", "実戦推奨")
                                        e_ratio = h_rec.get("edge_ratio", 0.0)
                                        real_races.append(f"・{r_name} (エッジ{e_ratio}倍 {b_txt})")

                                if real_races:
                                    summary_body = "\n".join(real_races[:5])
                                    if len(real_races) > 5:
                                        summary_body += f"\n他{len(real_races)-5}レース"
                                    summary_body += "\n※発走60分前(馬体重)・30分前・15分前・5分前(最終確定)にオッズ推移を追跡し判定します。"
                                    send_fcm_notification(
                                        title=f"🌅【本日実戦推奨候補】計{len(real_races)}レース該当！",
                                        body=summary_body,
                                        data={"type": "MORNING_SUMMARY", "race_date": today_str}
                                    )
                                    check_and_notify_balance_shortfall()
                        except Exception as fe:
                            print(f"[Scheduler] 朝通知エラー: {fe}")
                else:
                    # 本日が非開催日（平日等）の場合はチェック済みとしてマーク
                    last_830_predict_date = today_str

            if supabase:
                # --- ③ 各レース【発走 60分前 / 30分前 / 15分前 / 5分前】のマルチステージ自動推論 ＆ 最終確定通知 ---
                try:
                    res_pred = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .lt("is_predicted", 5)\
                        .execute()
                    unpredicted_races = res_pred.data or []

                    for r in unpredicted_races:
                        r_id = r["race_id"]
                        p_dt_str = r.get("post_datetime")
                        cur_stage = int(r.get("is_predicted") or 0)
                        post_dt = parse_post_datetime(p_dt_str, today_str)
                        if not post_dt:
                            continue

                        mins_to_post = (post_dt - now_dt).total_seconds() / 60.0
                        if mins_to_post <= 0 or mins_to_post > 65.0:
                            continue

                        # Stage 5: 発走 5分前 (0 < mins <= 6.5) -> 最終オッズ推論 ＆ 確定通知
                        if mins_to_post <= 6.5 and cur_stage < 5:
                            print(f"[Auto-Predict Stage5: 5分前最終確定] {r_id} ({r.get('race_name')}) の最終推論＆買い目確定を実行します...")
                            try:
                                pred_res = await asyncio.to_thread(predict_race_core, r_id, True, True, "05m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 5})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Predict Stage5] 確定完了: {r_id}")

                                h_rec = pred_res.get("hybrid_recommendation") or {}
                                if h_rec.get("is_real_bet_target"):
                                    badge = h_rec.get("real_bet_badge", "🔥 実戦推奨(確定)")
                                    edge = h_rec.get("edge_ratio", 0.0)
                                    target_u = pred_res.get("recommendation", {}).get("target_umaban")
                                    h_name = pred_res.get("recommendation", {}).get("horse_name", "")
                                    r_name = pred_res.get("race_name", r.get("race_name", f"Race {r_id}"))

                                    all_tickets = (
                                        (h_rec.get("tansho_bets") or [])
                                        + (h_rec.get("fukusho_bets") or [])
                                        + (h_rec.get("umatan_bets") or [])
                                        + (h_rec.get("umaren_bets") or [])
                                        + (h_rec.get("trio_bets") or [])
                                        + (h_rec.get("sanrentan_bets") or [])
                                    )
                                    bets_summary = ", ".join([f"{b.get('ticket_type')} {b.get('combination_str')}({b.get('amount')}円)" for b in all_tickets[:3]])
                                    tot_amt = h_rec.get("total_amount", 0)

                                    body_text = f"◎{target_u}番 {h_name} (期待エッジ:{edge}倍)\n【推奨買い目】{bets_summary} 計{tot_amt:,}円"
                                    send_fcm_notification(
                                        title=f"{badge} {r_name}！",
                                        body=body_text,
                                        data={
                                            "type": "REAL_BET_FINAL",
                                            "race_id": r_id,
                                            "edge_ratio": str(edge),
                                            "badge": badge
                                        }
                                    )

                                    # --- 【自動投票実行判定 (即PAT / UMACAスマート)】 ---
                                    b_settings = load_betting_settings_from_supabase()
                                    cur_bet_mode = b_settings.get("betting_mode", "off")

                                    if cur_bet_mode in ("ipat", "umaca"):
                                        mode_label = "UMACA" if cur_bet_mode == "umaca" else "即PAT"
                                        if has_already_betted_race(today_str, r_id):
                                            print(f"[AutoBet] {r_id} は既に投票済みのためスキップします (二重発注防止)")
                                        else:
                                            _auto_betted_races.add(r_id)
                                            print(f"[AutoBet] 🎫 {r_id} ({r_name}) の{mode_label}自動投票を開始します...")
                                            try:
                                                v_code = r_id[4:6] if len(r_id) == 12 else "05"
                                                v_name = VENUE_CODE_MAP.get(v_code, "東京")
                                                r_no = int(r_id[10:12]) if len(r_id) == 12 else 11

                                                bet_items = []
                                                for b in all_tickets:
                                                    c_list = b.get("combination")
                                                    if not c_list:
                                                        t_u = b.get("target_umaban")
                                                        c_list = [t_u] if t_u is not None else []
                                                    bet_items.append({
                                                        "ticket_type": b.get("ticket_type", "単勝"),
                                                        "combination": c_list,
                                                        "amount": b.get("amount", 100)
                                                    })

                                                client = BettingClientFactory.get_client(cur_bet_mode)
                                                bet_res = await asyncio.to_thread(client.execute_bets, v_name, r_no, bet_items, False)

                                                if bet_res.get("status") == "success":
                                                    rcpt = bet_res.get("receipt_no", "")
                                                    _auto_betted_dates.add(today_str)
                                                    _auto_betted_races.add(r_id)
                                                    try:
                                                        cur_s = load_betting_settings_from_supabase()
                                                        b_dict = cur_s.get("betted_races_by_date", {})
                                                        b_list = b_dict.get(today_str, [])
                                                        b_list.append({
                                                            "race_id": r_id,
                                                            "receipt_no": str(rcpt),
                                                            "amount": tot_amt,
                                                            "mode": cur_bet_mode,
                                                            "time": now_jst().strftime("%H:%M:%S")
                                                        })
                                                        b_dict[today_str] = b_list
                                                        cur_s["betted_races_by_date"] = b_dict
                                                        if bet_res.get("balance") is not None:
                                                            cur_s["last_balance"] = bet_res.get("balance")
                                                            cur_s["last_balance_updated_at"] = now_jst().strftime("%Y-%m-%d %H:%M:%S")
                                                        save_betting_settings_to_supabase(cur_s)
                                                    except Exception as me:
                                                        print(f"[AutoBet Meta Save Error] {me}")
                                                    print(f"[AutoBet] 🎉 {mode_label}投票成功！ 受付番号: {rcpt}")
                                                    send_fcm_notification(
                                                        title=f"🎫【{mode_label}投票完了】{v_name}{r_no}R",
                                                        body=f"受付番号:{rcpt} / {len(bet_items)}点 計{tot_amt:,}円 投票完了！\n残高より正常に引き落とされました。",
                                                        data={
                                                            "type": "AUTO_BET_SUCCESS",
                                                            "race_id": r_id,
                                                            "receipt_no": str(rcpt),
                                                            "amount": str(tot_amt),
                                                            "mode": cur_bet_mode
                                                        }
                                                    )
                                                else:
                                                    err_msg = bet_res.get("reason") or bet_res.get("error") or "投票処理エラー"
                                                    print(f"[AutoBet] ⚠️ {mode_label}投票失敗: {err_msg}")
                                                    send_fcm_notification(
                                                        title=f"⚠️【{mode_label}投票エラー】{v_name}{r_no}R",
                                                        body=f"自動投票に失敗しました: {err_msg}",
                                                        data={
                                                            "type": "AUTO_BET_FAILED",
                                                            "race_id": r_id,
                                                            "error": str(err_msg),
                                                            "mode": cur_bet_mode
                                                        }
                                                    )
                                            except Exception as ab_err:
                                                print(f"[AutoBet] 例外エラー: {ab_err}")
                                                send_fcm_notification(
                                                    title=f"⚠️【{mode_label}投票例外】{r_name}",
                                                    body=f"投票通信中に例外が発生しました: {ab_err}",
                                                    data={
                                                        "type": "AUTO_BET_FAILED",
                                                        "race_id": r_id,
                                                        "error": str(ab_err),
                                                        "mode": cur_bet_mode
                                                    }
                                                )
                            except Exception as pe:
                                print(f"[Auto-Predict Stage5] 最終推論失敗 ({r_id}): {pe}")

                        # Stage 4: 発走 15分前 (6.5 < mins <= 17.0) -> 本馬場入場後オッズ更新
                        elif 6.5 < mins_to_post <= 17.0 and cur_stage < 4:
                            print(f"[Auto-Predict Stage4: 15分前直前更新] {r_id} ({r.get('race_name')}) の15分前オッズ更新を実行します...")
                            try:
                                await asyncio.to_thread(predict_race_core, r_id, True, False, "15m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 4})\
                                    .eq("race_id", r_id)\
                                    .execute()
                            except Exception as pe:
                                print(f"[Auto-Predict Stage4] 更新失敗 ({r_id}): {pe}")

                        # Stage 3: 発走 30分前 (17.0 < mins <= 33.0) -> パドック後オッズ更新
                        elif 17.0 < mins_to_post <= 33.0 and cur_stage < 3:
                            print(f"[Auto-Predict Stage3: 30分前パドック更新] {r_id} ({r.get('race_name')}) の30分前オッズ更新を実行します...")
                            try:
                                await asyncio.to_thread(predict_race_core, r_id, True, False, "30m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 3})\
                                    .eq("race_id", r_id)\
                                    .execute()
                            except Exception as pe:
                                print(f"[Auto-Predict Stage3] 更新失敗 ({r_id}): {pe}")

                        # Stage 2: 発走 58分前〜33分前 -> 馬体重反映・第1回再計算 (馬体重取得成功時のみstage=2へ進める)
                        elif 33.0 < mins_to_post <= 58.0 and cur_stage < 2:
                            try:
                                pred_s2 = await asyncio.to_thread(predict_race_core, r_id, True, False, "60m")
                                has_w = any(h.get("horse_weight") is not None for h in (pred_s2 or {}).get("horses", []))
                                if has_w:
                                    supabase.table("keiba_scheduled_races")\
                                        .update({"is_predicted": 2})\
                                        .eq("race_id", r_id)\
                                        .execute()
                                    print(f"[Auto-Predict Stage2] {r_id}: 馬体重反映完了")
                            except Exception as we:
                                print(f"[Auto-Predict Stage2] 馬体重更新チェック失敗 ({r_id}): {we}")
                except Exception as e:
                    print(f"[Scheduler] 予想ループエラー: {e}")

                # --- ③-B 発走後レースで馬体重が未取得のまま残っている場合の自動バックグラウンド補完 ---
                try:
                    res_all_sched = supabase.table("keiba_scheduled_races").select("race_id, post_datetime").eq("race_date", today_str).execute()
                    past_rids_today = []
                    for sr in (res_all_sched.data or []):
                        p_dt = parse_post_datetime(sr.get("post_datetime"), today_str)
                        if p_dt and now_dt >= p_dt:
                            past_rids_today.append(sr["race_id"])
                    if past_rids_today:
                        p_chk = supabase.table("keiba_predictions_cache").select("race_id, data_json").in_("race_id", past_rids_today).execute()
                        healed_cnt = 0
                        for prow in (p_chk.data or []):
                            if healed_cnt >= 4:
                                break
                            prid = prow.get("race_id")
                            pdj = prow.get("data_json") or {}
                            phorses = pdj.get("horses", [])
                            if phorses and not any(h.get("horse_weight") is not None for h in phorses):
                                await asyncio.to_thread(predict_race_core, prid, False, False, None)
                                healed_cnt += 1
                except Exception as he_loop:
                    print(f"[Scheduler] 発走後馬体重補完チェック警告: {he_loop}")

                # 予想更新時に即PAT残高不足の恐れがあればアラート通知
                check_and_notify_balance_shortfall()

                # --- ④ 各レース【発走 10分後】の個別自動照合（未確定時は3分おきにリトライ） ---
                try:
                    res_settle = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .eq("is_settled", 0)\
                        .execute()
                    unsettled_races = res_settle.data or []

                    for r in unsettled_races:
                        r_id = r["race_id"]
                        p_dt_str = r.get("post_datetime")
                        retries = r.get("retry_count", 0) or 0
                        post_dt = parse_post_datetime(p_dt_str, today_str)
                        if not post_dt:
                            continue

                        # 初回は発走10分後、以降は3分間隔でリトライ
                        trigger_settle_time = post_dt + datetime.timedelta(minutes=10 + 3 * retries)
                        if now_dt >= trigger_settle_time:
                            try:
                                await asyncio.to_thread(verify_race_result, r_id, True, False)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_settled": 1})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Settle] 結果照合完了: {r_id}")
                            except Exception as se_err:
                                print(f"[Auto-Settle Retry] {r_id} (retry={retries}): {se_err}")
                                if retries >= 15:
                                    supabase.table("keiba_scheduled_races")\
                                        .update({"is_settled": 1})\
                                        .eq("race_id", r_id)\
                                        .execute()
                                else:
                                    supabase.table("keiba_scheduled_races")\
                                        .update({"retry_count": retries + 1})\
                                        .eq("race_id", r_id)\
                                        .execute()
                except Exception as e:
                    print(f"[Scheduler] 照合ループエラー: {e}")

            # --- ⑤ JST 17:00 および 19:00 の当日最終未確定一括照合 ---
            if now_dt.hour == 17 and last_17h_settle_date != today_str:
                if today_str in discover_period_kaisai_dates(now_dt):
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 17:00 当日レースの最終結果照合を実行します...")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "settle", False, True))
                        last_17h_settle_date = today_str
                else:
                    last_17h_settle_date = today_str

            if now_dt.hour == 19 and last_19h_settle_date != today_str:
                if today_str in discover_period_kaisai_dates(now_dt):
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 19:00 当日レースの予備最終結果照合を実行します...")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "settle", False, True))
                        last_19h_settle_date = today_str
                else:
                    last_19h_settle_date = today_str

        except Exception as e:
            print(f"[Scheduler] 全体ループエラー: {e}")

        await asyncio.sleep(60)



def fix_mislabeled_20261011_races():
    """10/11に誤登録された10/12(0405: 4回5日目)レースを10/12へ移行し、10/11当日の0404レースを再登録"""
    if not supabase:
        print("[Fix] Supabase未接続のため修正処理をスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}

    try:
        print("[Fix] 10/11誤登録データの修正・移行処理を開始します...")

        # 1. keiba_scheduled_races: 10/11に登録された0405レースを10/12に移行
        res_sched = supabase.table("keiba_scheduled_races").select("*").eq("race_date", "2026-10-11").execute()
        sched_rows = res_sched.data or []
        migrated_sched = 0
        for r in sched_rows:
            r_id = r["race_id"]
            if len(r_id) == 12 and r_id[6:10] == "0405":
                new_post_dt = r["post_datetime"].replace("2026-10-11", "2026-10-12")
                supabase.table("keiba_scheduled_races").update({
                    "race_date": "2026-10-12",
                    "post_datetime": new_post_dt,
                    "is_settled": 0,
                    "retry_count": 0,
                    "is_predicted": 0
                }).eq("race_id", r_id).execute()
                migrated_sched += 1

        # 2. keiba_predictions_cache: 0405レースのrace_dateを2026-10-12に移行
        res_cache = supabase.table("keiba_predictions_cache").select("race_id").eq("race_date", "2026-10-11").execute()
        cache_rows = res_cache.data or []
        migrated_cache = 0
        for c in cache_rows:
            r_id = c["race_id"]
            if len(r_id) == 12 and r_id[6:10] == "0405":
                supabase.table("keiba_predictions_cache").update({
                    "race_date": "2026-10-12"
                }).eq("race_id", r_id).execute()
                migrated_cache += 1

        # 3. keiba_race_settlements: 0405レースが誤って保存されていれば削除
        try:
            supabase.table("keiba_race_settlements").delete().filter("race_date", "eq", "2026-10-11").like("race_id", "%0405%").execute()
        except Exception:
            pass

        # 4. keiba_race_results_cache: 0405レースが誤って保存されていれば削除
        try:
            supabase.table("keiba_race_results_cache").delete().filter("race_date", "eq", "2026-10-11").like("race_id", "%0405%").execute()
        except Exception:
            pass

        # 5. 正しい本日 (2026-10-11 / 0404) のレーススケジュールを即時取得・登録
        today_info = get_today_races("20261011")
        venues = today_info.get("venues", [])
        synced_count = 0
        for v in venues:
            for r in v.get("races", []):
                r_id = r["race_id"]
                try:
                    fetch_shutuba_table(r_id, race_date="2026-10-11")
                    synced_count += 1
                except Exception as fe:
                    print(f"[Fix] 出馬表登録エラー ({r_id}): {fe}")

        msg = f"修正完了: 10/12移行(スケジュール={migrated_sched}件, 予想キャッシュ={migrated_cache}件), 本日(10/11)登録={synced_count}件"
        print(f"[Fix] {msg}")
        return {"status": "ok", "message": msg}
    except Exception as e:
        err_msg = f"[Fix] 修正処理エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


# --- DB構成・スキーマバージョン管理 ---
CURRENT_DB_VERSION = "v5_umaren_3point_strategy"

def clear_all_database(sync_today: bool = False):
    """Supabaseの全テーブル（メタ、キャッシュ、台帳、スケジュール）を完全クリア・初期化"""
    if not supabase:
        print("[ClearDB] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}
    try:
        print("[ClearDB] Supabase全テーブルの初期化クリーンアップを開始します...")
        for tbl in ["keiba_predictions_cache", "keiba_race_results_cache", "keiba_race_settlements", "keiba_scheduled_races"]:
            try:
                supabase.table(tbl).delete().neq("race_id", "0").execute()
            except Exception as e:
                print(f"[ClearDB] {tbl} clear warning: {e}")
        try:
            supabase.table("keiba_app_meta").delete().neq("key", "__keep__").neq("key", "stats_overlay").execute()
        except Exception as e:
            print(f"[ClearDB] app_meta clear warning: {e}")

        invalidate_summary_caches()

        synced = 0
        if sync_today:
            today_str = today_jst_str()
            today_clean = today_str.replace("-", "")
            today_info = get_today_races(today_clean)
            venues = today_info.get("venues", [])
            for v in venues:
                for r in v.get("races", []):
                    try:
                        fetch_shutuba_table(r["race_id"], race_date=today_str)
                        synced += 1
                    except Exception:
                        pass

        msg = f"全DBテーブル初期化完了 (本日同期: {synced}レース)"
        print(f"[ClearDB] {msg}")
        return {"status": "ok", "message": msg, "synced_races": synced}
    except Exception as e:
        err_msg = f"[ClearDB] 初期化エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


def check_and_apply_db_migration():
    """DB構成・スキーマバージョンが変更された場合のみ、旧データをまっさらにしてバージョンを更新"""
    if not supabase:
        return
    try:
        saved_version = None
        try:
            m_res = supabase.table("keiba_app_meta").select("value").eq("key", "schema_version").execute()
            if m_res.data and len(m_res.data) > 0:
                saved_version = m_res.data[0]["value"].get("version")
        except Exception:
            pass

        if not saved_version:
            try:
                res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", "SCHEMA_VERSION").execute()
                if res.data and len(res.data) > 0:
                    saved_version = res.data[0]["data_json"].get("version")
            except Exception:
                pass

        if saved_version != CURRENT_DB_VERSION:
            print(f"[DB Migration] DB構成変更検知 ({saved_version} -> {CURRENT_DB_VERSION}): DBを初期化します。")
            clear_all_database()
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
            print(f"[DB Migration] 新バージョン {CURRENT_DB_VERSION} を正常に保存しました。")
        else:
            print(f"[DB Migration] DBバージョン ({saved_version}) は最新です。データ初期化はスキップします。")
    except Exception as e:
        print(f"[DB Migration] バージョン確認エラー: {e}")


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





def normalize_mislabeled_dates():
    """2026-10-13等のフォールバック日および新潟0401の誤認日(2026-10-03)を正規の開催日へ修正"""
    if not supabase:
        print("[NormalizeDates] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}

    migrated_settlements = 0
    migrated_results = 0
    migrated_predictions = 0

    try:
        # 1. 4回新潟1日目 (2026040401xx) が 2026-10-03 と誤保存されているものを 2026-10-17 へ修正
        for tbl_name in ("keiba_race_settlements", "keiba_race_results_cache", "keiba_predictions_cache", "keiba_scheduled_races"):
            try:
                res_niigata = supabase.table(tbl_name).select("race_id, race_date").like("race_id", "2026040401%").neq("race_date", "2026-10-17").execute()
                for nrow in (res_niigata.data or []):
                    nrid = nrow.get("race_id")
                    if nrid:
                        supabase.table(tbl_name).update({"race_date": "2026-10-17"}).eq("race_id", nrid).execute()
                        if tbl_name == "keiba_race_settlements":
                            migrated_settlements += 1
            except Exception as ne:
                print(f"[NormalizeDates] Niigata 0401 fix warning ({tbl_name}): {ne}")

        res_s = supabase.table("keiba_race_settlements").select("race_id, race_date").eq("race_date", "2026-10-13").execute()
        rows_s = res_s.data or []
        for row in rows_s:
            r_id = row.get("race_id", "")
            correct_date = resolve_race_date_for_id(r_id)

            if correct_date and correct_date != "2026-10-13":
                supabase.table("keiba_race_settlements").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                migrated_settlements += 1

                try:
                    supabase.table("keiba_race_results_cache").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                    migrated_results += 1
                except Exception:
                    pass
                try:
                    supabase.table("keiba_predictions_cache").update({"race_date": correct_date}).eq("race_id", r_id).execute()
                    migrated_predictions += 1
                except Exception:
                    pass

        # keiba_scheduled_races の 2026-10-13 レコードも更新・整理
        try:
            res_sc = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", "2026-10-13").execute()
            for row_sc in (res_sc.data or []):
                sc_id = row_sc.get("race_id", "")
                c_date = resolve_race_date_for_id(sc_id)
                if c_date and c_date != "2026-10-13":
                    supabase.table("keiba_scheduled_races").update({"race_date": c_date}).eq("race_id", sc_id).execute()
                else:
                    supabase.table("keiba_scheduled_races").delete().eq("race_id", sc_id).execute()
        except Exception as e:
            print(f"[NormalizeDates] scheduled_races cleanup error: {e}")

        if migrated_settlements > 0 or migrated_results > 0 or migrated_predictions > 0:
            invalidate_summary_caches()

        msg = f"日付正規化完了: settlements={migrated_settlements}件, results={migrated_results}件, predictions={migrated_predictions}件"
        print(f"[NormalizeDates] {msg}")
        return {"status": "ok", "message": msg}
    except Exception as e:
        err_msg = f"[NormalizeDates] 修正処理エラー: {e}"
        print(err_msg)
        return {"status": "error", "message": str(e)}


def sync_and_process_target_dates(target_dates: List[str]):
    """指定開催日（例: ['2026-09-26', '2026-09-27']）の全レースを keiba_scheduled_races に登録し、全レース予想＆結果検証を実行"""
    if not supabase:
        print("[Process Dates] Supabase未接続")
        return {"status": "error", "message": "Supabase not connected"}

    total_scheduled = 0
    total_predicted = 0
    total_settled = 0
    all_races = []

    for d_str in target_dates:
        venues = fetch_venues_for_date_raw(d_str)
        race_items = [r for v in venues for r in v.get("races", [])]
        rows_to_upsert = []
        for r in race_items:
            r_id = r["race_id"]
            r_info = r.get("race_info") or r.get("race_name") or ""
            m = re.search(r"(\d{1,2}):(\d{2})", r_info)
            hour, minute = (int(m.group(1)), int(m.group(2))) if m else (15, 0)
            d_parts = [int(p) for p in d_str.split("-")]
            post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
            rows_to_upsert.append({
                "race_id": r_id,
                "race_date": d_str,
                "race_name": r.get("race_name", r.get("race_no", "")),
                "post_datetime": post_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "is_predicted": 0,
                "is_settled": 0,
                "retry_count": 0
            })
            all_races.append((d_str, r_id))
        if rows_to_upsert:
            try:
                supabase.table("keiba_scheduled_races").upsert(rows_to_upsert, on_conflict="race_id").execute()
                total_scheduled += len(rows_to_upsert)
            except Exception as e:
                print(f"[Schedule] upsert error for {d_str}: {e}")

    # 全レースの予想 ＆ 結果検証を一括実行
    for d_str, r_id in all_races:
        try:
            predict_race_core(r_id, force_refresh=True)
            total_predicted += 1
            if supabase:
                supabase.table("keiba_scheduled_races").update({"is_predicted": 2}).eq("race_id", r_id).execute()
        except Exception as pe:
            print(f"[Predict Error] {r_id}: {pe}")

        try:
            verify_race_result(r_id, bypass_cache=True)
            total_settled += 1
            if supabase:
                supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", r_id).execute()
        except Exception as se:
            print(f"[Settle Error] {r_id}: {se}")

    msg = f"{', '.join(target_dates)} 全レース処理完了: スケジュール登録={total_scheduled}R, 予想={total_predicted}R, 検証={total_settled}R"
    print(f"[Batch Process] {msg}")
    return {
        "status": "ok",
        "message": msg,
        "scheduled_count": total_scheduled,
        "predicted_count": total_predicted,
        "settled_count": total_settled
    }


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


def fix_duplicated_horses_cache_2026():
    """出馬表パース時の重複取得（1番二重登録等）を含む予測キャッシュを自動検知してクリア（当日分のみ高速チェック）"""
    if not supabase:
        return
    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", today_jst_str()).execute()
        rids_to_clear = []
        for row in (res.data or []):
            rid = row.get("race_id")
            dj = row.get("data_json") or {}
            horses = dj.get("horses", [])
            umaban_list = [h.get("umaban") for h in horses if "umaban" in h and h.get("umaban") is not None]
            if len(umaban_list) != len(set(umaban_list)):
                rids_to_clear.append(rid)
        if rids_to_clear:
            print(f"[FixDuplicates] 馬番重複を検知した {len(rids_to_clear)} レースのキャッシュをリセットします: {rids_to_clear}")
            supabase.table("keiba_predictions_cache").delete().in_("race_id", rids_to_clear).execute()
    except Exception as e:
        print(f"[FixDuplicates] クリーンアップ例外: {e}")


def heal_today_weights_and_results(target_date: Optional[str] = None, force_resettle: bool = False) -> dict:
    """指定日(デフォルト本日)の馬体重未反映レースおよび結果未検証レースを一括補完・検証"""
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    t_date = target_date or today_jst_str()
    normalize_mislabeled_dates()
    ensure_daily_single_real_bet(t_date)

    now_dt = now_jst_naive()
    sched_res = supabase.table("keiba_scheduled_races").select("*").eq("race_date", t_date).order("race_id").execute()
    sched_rows = sched_res.data or []

    pred_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", t_date).execute()
    pred_map = {r["race_id"]: (r.get("data_json") or {}) for r in (pred_res.data or []) if r.get("race_id")}

    settle_res = supabase.table("keiba_race_settlements").select("race_id, actual_order, is_pass, target_umaban").eq("race_date", t_date).execute()
    settle_rows = settle_res.data or []
    settled_rids = {
        r["race_id"] for r in settle_rows
        if r.get("actual_order") is not None or r.get("is_pass") in (1, True)
    }
    settle_pass_map = {r["race_id"]: r.get("is_pass") for r in settle_rows if r.get("race_id")}
    settle_target_map = {r["race_id"]: r.get("target_umaban") for r in settle_rows if r.get("race_id")}

    healed_weights = []
    settled_now = []
    errors = []

    for sr in sched_rows:
        rid = sr["race_id"]
        p_dt_str = sr.get("post_datetime", "")
        post_dt = parse_post_datetime(p_dt_str, t_date) or now_dt

        mins_to_post = (post_dt - now_dt).total_seconds() / 60.0 if t_date == today_jst_str() else -999.0
        dj = pred_map.get(rid) or {}
        horses = dj.get("horses", [])
        has_weights = any(h.get("horse_weight") is not None for h in horses)

        # 1. 馬体重の補完・最新化 (発走58分前以降の全レース)
        if not has_weights and mins_to_post <= 58.0:
            try:
                _last_weight_check_ts.pop(rid, None)
                if mins_to_post > 0:
                    stg = infer_odds_stage_from_minutes(mins_to_post)
                    updated_p = predict_race_core(rid, force_refresh=True, is_auto_final=(stg == "05m"), odds_stage=stg)
                else:
                    # 発走後は買い目を変えずに馬体重・性齢のみ補完
                    updated_p = predict_race_core(rid, force_refresh=False, is_auto_final=False, odds_stage=None)
                if any(h.get("horse_weight") is not None for h in (updated_p or {}).get("horses", [])):
                    healed_weights.append(rid)
            except Exception as we:
                errors.append(f"weight({rid}): {we}")

        # 2. 結果検証の補完 (発走8分後以降の未検証レース または 実馬券昇格後にis_pass=1のままのレース または 仮想馬券でis_pass=1のままのレース または 本命馬番不整合レース)
        is_real_target = bool((dj.get("hybrid_recommendation") or {}).get("is_real_bet_target"))
        is_virt_target = bool(
            sr.get("is_virtual_bet_target")
            or (dj.get("hybrid_recommendation") or {}).get("is_virtual_bet_target")
            or sr.get("real_bet_level") == "virtual"
            or (sr.get("real_bet_badge") and "仮想" in str(sr.get("real_bet_badge")))
        ) and not is_real_target
        pred_target_u = (dj.get("recommendation") or {}).get("target_umaban")
        needs_target_resettle = pred_target_u is not None and (settle_target_map.get(rid) is not None) and (settle_target_map.get(rid) != pred_target_u)
        needs_real_resettle = is_real_target and (settle_pass_map.get(rid) in (1, True))
        needs_virt_resettle = is_virt_target and (settle_pass_map.get(rid) in (1, True))
        if mins_to_post <= -8.0 and (force_resettle or rid not in settled_rids or int(sr.get("is_settled") or 0) == 0 or needs_real_resettle or needs_virt_resettle or needs_target_resettle):
            try:
                verify_race_result(rid, bypass_cache=True, is_19h_settle=False)
                supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", rid).execute()
                settled_now.append(rid)
            except Exception as se:
                errors.append(f"settle({rid}): {se}")

    _all_predictions_cache.pop(t_date, None)
    _today_races_cache.pop(t_date, None)
    invalidate_summary_caches()

    return {
        "status": "ok",
        "date": t_date,
        "healed_weights_count": len(healed_weights),
        "healed_weights_races": healed_weights,
        "settled_now_count": len(settled_now),
        "settled_now_races": settled_now,
        "errors": errors
    }


@app.post("/admin/heal-settlement-marks")
@app.get("/admin/heal-settlement-marks")
def heal_settlement_marks(date: Optional[str] = Query(None, description="修復対象日付 (YYYY-MM-DD, 省略時は全期間)")):
    """keiba_race_settlements の印・本命馬・確定着順を最新の predictions_cache と results_cache から走査して一括自己修復"""
    if not supabase:
        return {"status": "error", "message": "Supabase not connected"}

    query = supabase.table("keiba_race_settlements").select("*")
    if date:
        query = query.eq("race_date", date)
    res = query.order("race_id", desc=True).limit(200).execute()
    rows = res.data or []

    repaired = []
    for r in rows:
        rid = r["race_id"]
        r_date = r.get("race_date")
        pred = get_cached_prediction(rid)
        result = get_cached_result(rid)
        if pred and result:
            pred_target = (pred.get("recommendation") or {}).get("target_umaban")
            if pred_target is None:
                for h in pred.get("horses", []):
                    if h.get("mark") == "◎":
                        pred_target = int(h.get("umaban"))
                        break
            orders = result.get("orders", {})
            fresh_top3 = extract_top3_orders(pred, orders)
            fresh_marks = extract_mark_orders(pred, orders)

            needs_update = False
            update_payload = {}

            if pred_target is not None and r.get("target_umaban") != pred_target:
                # 本命馬番が不一致 -> process_settlement_calculation で完全再計算
                try:
                    process_settlement_calculation(
                        race_id=rid,
                        race_date=r_date,
                        race_name=r.get("race_name", f"Race {rid}"),
                        pred_data=pred,
                        race_res=result,
                        from_cache=True
                    )
                    repaired.append({"race_id": rid, "type": "target_umaban_recalculated", "old_target": r.get("target_umaban"), "new_target": pred_target})
                    continue
                except Exception as e_recalc:
                    print(f"[HealMarks Recalc Error] {rid}: {e_recalc}")

            if fresh_top3 and fresh_top3 != r.get("top3_orders"):
                update_payload["top3_orders"] = fresh_top3
                needs_update = True

            if fresh_marks and fresh_marks != (r.get("bet_details") or {}).get("real_bet_meta", {}).get("mark_orders"):
                b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
                rb_meta = b_det.get("real_bet_meta") if isinstance(b_det.get("real_bet_meta"), dict) else {}
                rb_meta["mark_orders"] = fresh_marks
                rb_meta["top3_orders"] = fresh_top3
                b_det["real_bet_meta"] = rb_meta
                update_payload["bet_details"] = b_det
                needs_update = True

            if needs_update:
                try:
                    supabase.table("keiba_race_settlements").update(update_payload).eq("race_id", rid).execute()
                    repaired.append({"race_id": rid, "type": "marks_synchronized"})
                except Exception as e_up:
                    print(f"[HealMarks Update Error] {rid}: {e_up}")

    invalidate_summary_caches()
    return {
        "status": "ok",
        "scanned_count": len(rows),
        "repaired_count": len(repaired),
        "repaired_details": repaired
    }


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

