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
from app.strategies.arbiter import StrategyArbiter
from app.betting.factory import BettingClientFactory
from app.betting.manual_service import execute_manual_bet
from app.settlement.payout_verifier import PayoutVerifier
from app.betting.settings import (
    load_betting_settings_from_supabase,
    save_betting_settings_to_supabase,
    calculate_remaining_real_bet_amount,
    _betting_settings
)
from app.routers.betting import router as betting_router
from app.routers.admin import router as admin_router
from app.settlement.summary_calculator import (
    DailySummary,
    AggregateSummaryResponse,
    TicketTypeSummary,
    PeriodStatItem,
    invalidate_summary_caches
)
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

MODEL_PATH = "lgb_model.txt"
WIN_MODEL_PATH = "lgb_win_model.txt"
TOP3_MODEL_PATH = "lgb_top3_model.txt"
MAIDEN_WIN_MODEL_PATH = "lgb_maiden_win_model.txt"
MAIDEN_TOP3_MODEL_PATH = "lgb_maiden_top3_model.txt"
GACHI_MODEL_PATH = "lgb_gachi_model.txt"
HIMO_MODEL_PATH = "lgb_himo_model.txt"
DICT_PATH = "stats_dict.json"

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

# --- LightGBM & 統計辞書のロード ---
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
        print("現行LightGBMモデルを正常にロードしました。")
    except Exception as e:
        print(f"現行モデルロード失敗: {e}")

model_win = None
if os.path.exists(WIN_MODEL_PATH):
    try:
        model_win = lgb.Booster(model_file=WIN_MODEL_PATH)
        print("1着特化モデル (m_win) を正常にロードしました。")
    except Exception as e:
        print(f"1着特化モデルロード失敗: {e}")

model_top3 = None
if os.path.exists(TOP3_MODEL_PATH):
    try:
        model_top3 = lgb.Booster(model_file=TOP3_MODEL_PATH)
        print("3着内特化モデル (m_top3) を正常にロードしました。")
    except Exception as e:
        print(f"3着内特化モデルロード失敗: {e}")

model_maiden_win = None
if os.path.exists(MAIDEN_WIN_MODEL_PATH):
    try:
        model_maiden_win = lgb.Booster(model_file=MAIDEN_WIN_MODEL_PATH)
        print("未勝利戦Winモデル (m_maiden_win) を正常にロードしました。")
    except Exception as e:
        print(f"未勝利戦Winモデルロード失敗: {e}")

model_maiden_top3 = None
if os.path.exists(MAIDEN_TOP3_MODEL_PATH):
    try:
        model_maiden_top3 = lgb.Booster(model_file=MAIDEN_TOP3_MODEL_PATH)
        print("未勝利戦Top3モデル (m_maiden_top3) を正常にロードしました。")
    except Exception as e:
        print(f"未勝利戦Top3モデルロード失敗: {e}")

model_gachi = None
if os.path.exists(GACHI_MODEL_PATH):
    try:
        model_gachi = lgb.Booster(model_file=GACHI_MODEL_PATH)
        print("ガチガチ判定メタモデル (m_gachi) を正常にロードしました。")
    except Exception as e:
        print(f"ガチガチ判定メタモデルロード失敗: {e}")

model_himo = None
if os.path.exists(HIMO_MODEL_PATH):
    try:
        model_himo = lgb.Booster(model_file=HIMO_MODEL_PATH)
        print("中波乱判定メタモデル (m_himo) を正常にロードしました。")
    except Exception as e:
        print(f"中波乱判定メタモデルロード失敗: {e}")

stats_dict = {"jockey_stats": {}, "jockey_course_stats": {}, "trainer_stats": {}, "trainer_course_stats": {}, "horse_stats": {}}
if os.path.exists(DICT_PATH):
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        print("統計辞書を正常にロードしました。")
    except Exception as e:
        print(f"統計辞書ロード失敗: {e}")

_auto_betted_races = set()       # 二重投票防止用メモリセット (race_id)

# --- 統計辞書（stats_dict）オンライン最新化＆Supabase永続化機構 ---
_stats_overlay: Dict[str, Any] = {
    "updated_at": None,
    "horse_stats": {},
    "jockey_stats": {},
    "trainer_stats": {}
}

def load_stats_overlay_from_supabase() -> bool:
    """Supabase keiba_app_meta から永続化された統計辞書オーバーレイを取得し、メモリ上の stats_dict にマージ"""
    global _stats_overlay, stats_dict
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
            # 既にこのレース（同一日付以降）で更新済みなら重複更新を回避
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


# --- Pydantic スキーマ ---
def calc_trio_odds(o1: float, o2: float, o3: float) -> float:
    """Harville確率モデルに基づく三連複推定オッズ計算"""
    p1 = 0.80 / max(1.1, float(o1))
    p2 = 0.80 / max(1.1, float(o2))
    p3 = 0.80 / max(1.1, float(o3))
    p_sum = 0.0
    for a, b, c in [(p1, p2, p3), (p1, p3, p2), (p2, p1, p3), (p2, p3, p1), (p3, p1, p2), (p3, p2, p1)]:
        denom1 = max(0.01, 1.0 - a)
        denom2 = max(0.01, 1.0 - a - b)
        p_sum += a * (b / denom1) * (c / denom2)
    if p_sum <= 0:
        return 10.0
    return round(max(2.0, min(1000.0, 0.75 / p_sum)), 1)







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



def rank_min_desc(values: List[float]) -> List[int]:
    """降順ランク (値が大きいほど1位、同率は同率最小順位: pandas rank(ascending=False, method='min') と同等)"""
    sorted_unique = sorted(list(set(values)), reverse=True)
    val_to_rank = {}
    current_rank = 1
    for v in sorted_unique:
        val_to_rank[v] = current_rank
        current_rank += values.count(v)
    return [val_to_rank[v] for v in values]


def rank_min_asc(values: List[float]) -> List[int]:
    """昇順ランク (値が小さいほど1位、同率は同率最小順位: pandas rank(ascending=True, method='min') と同等)"""
    sorted_unique = sorted(list(set(values)), reverse=False)
    val_to_rank = {}
    current_rank = 1
    for v in sorted_unique:
        val_to_rank[v] = current_rank
        current_rank += values.count(v)
    return [val_to_rank[v] for v in values]


def calculate_dynamic_bet(score: float, odds: Optional[float] = None):
    if odds is not None and odds < 2.5:
        return "D", "見送り推奨(低オッズ)", True, 0, 0

    if score >= 0.650:
        return "S", "鉄板・大勝負", False, 300, 700
    elif score >= 0.550:
        return "A", "勝負レース", False, 200, 400
    elif score >= 0.400:
        return "B", "標準推奨", False, 100, 200
    elif score >= 0.330:
        return "C", "少額推奨", False, 100, 100
    else:
        return "D", "見送り推奨", True, 0, 0


# --- 【実馬券選定 ＆ 0レース時の当日No.1昇格保証】ヘルパー機構 ---
_daily_top_race_id_cache: Dict[str, str] = {}
_auto_betted_dates: Set[str] = set()
NATURAL_REAL_STRATEGIES = ("strategy_b", "strategy_a", "strategy_c", "strategy_a_hole")

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

def get_or_select_daily_top_race_id(target_date: str) -> Optional[str]:
    """
    当日(target_date)全レースの中で、自然条件（戦略B/A/C/A-2）を満たすレースが0レースのときのみ、
    『確信度 × エッジ適性スコア』が最も高い1レースの race_id を特定して返す。
    自然条件を満たすレースが1つ以上存在する場合は None を返す（全合致レースをそのまま実馬券とする）。
    """
    if not target_date or not supabase:
        return None

    now_t = now_jst_naive()
    is_today = (target_date == today_jst_str())

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return None

        # 自然条件（戦略B / A / C / A-2）を満たすレースが1つでもあれば、昇格枠は不要
        existing_daily_top_rid = None
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            h_rec = dj.get("hybrid_recommendation") or {}
            if h_rec.get("strategy_type") in NATURAL_REAL_STRATEGIES:
                _daily_top_race_id_cache.pop(target_date, None)
                return None
            if h_rec.get("strategy_type") == "daily_top" and h_rec.get("is_real_bet_target"):
                existing_daily_top_rid = rid

        # 既に daily_top に昇格済みのレースがあれば、再推論・発走後もそのレースを維持する
        if existing_daily_top_rid:
            _daily_top_race_id_cache[target_date] = existing_daily_top_rid
            return existing_daily_top_rid

        candidates = []
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")
            if "障" in r_name or "障害" in c_det or "新馬" in r_name or "メイクデビュー" in r_name:
                continue
            horses = dj.get("horses") or []
            if len(horses) < 8:
                continue

            # 当日の場合は未発走（発走時刻前）のレースを優先対象とする
            is_unrun = True
            if is_today:
                p_time_str = (dj.get("race_details") or {}).get("post_time", "")
                post_dt = parse_post_datetime(p_time_str, target_date)
                if post_dt and now_t >= post_dt:
                    is_unrun = False

            h_rec = dj.get("hybrid_recommendation") or {}
            conf = float(h_rec.get("confidence_score") or 0.0)
            edge = float(h_rec.get("edge_ratio") or 1.0)
            aptitude_score = conf * min(max(edge, 0.8), 2.5)

            candidates.append({
                "race_id": rid,
                "is_unrun": is_unrun,
                "confidence_score": conf,
                "edge_ratio": edge,
                "aptitude_score": aptitude_score
            })

        if not candidates:
            return None

        unrun_candidates = [c for c in candidates if c["is_unrun"]]
        pool = unrun_candidates if unrun_candidates else candidates
        pool.sort(key=lambda x: (x["aptitude_score"], x["confidence_score"]), reverse=True)
        chosen_rid = pool[0]["race_id"]

        _daily_top_race_id_cache[target_date] = chosen_rid
        return chosen_rid
    except Exception as e:
        print(f"[get_or_select_daily_top_race_id] エラー: {e}")
        return None

def ensure_daily_single_real_bet(target_date: str) -> Optional[str]:
    """
    【実馬券厳選ルール】
    1. 戦略B（勝負馬券）および戦略A・C・A-2（実馬券1〜3）を満たすレースはすべて実馬券（is_real_bet_target = True）とする。
    2. 上記を満たすレースが0レースのときのみ、当日全レースの中で『確信度 × エッジ適性スコア』が最も高い1レースを実馬券に昇格する。
    """
    if not target_date or not supabase:
        return None

    now_t = now_jst_naive()
    is_today = (target_date == today_jst_str())

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        rows = res.data or []
        if not rows:
            return None

        natural_real_rids = []
        existing_daily_top_rid = None
        # Step 1: 過去に「参考推奨」へ格下げされていた自然合致レース（戦略B/A/C/A-2）があれば全て実馬券に復元
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            h_rec = dj.get("hybrid_recommendation") or {}
            st_type = h_rec.get("strategy_type")
            if st_type in NATURAL_REAL_STRATEGIES:
                natural_real_rids.append(rid)
                if not h_rec.get("is_real_bet_target") or "参考推奨" in str(h_rec.get("real_bet_badge") or ""):
                    h_rec["is_participate"] = True
                    h_rec["is_real_bet_target"] = True
                    h_rec["real_bet_level"] = "high" if st_type == "strategy_b" else "standard"
                    badge_map = {
                        "strategy_b": "🔥【超勝負R】単勝＋馬単レバレッジ",
                        "strategy_a": "⭐【芝厳選】単複ケリー傾斜",
                        "strategy_c": "🛡️【ダート本命】手堅い単複防衛",
                        "strategy_a_hole": "⚡【芝波乱妙味】単複好配当狙い"
                    }
                    h_rec["real_bet_badge"] = badge_map.get(st_type, "⭐【実馬券推奨】")
                    dj["hybrid_recommendation"] = h_rec
                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", rid).execute()
            elif st_type == "daily_top" and h_rec.get("is_real_bet_target"):
                existing_daily_top_rid = rid

        # Step 2: 自然合致レースが1レース以上存在する場合は、不要な daily_top 昇格枠を解除して終了
        if natural_real_rids:
            for r in rows:
                rid = str(r.get("race_id") or "")
                dj = r.get("data_json") or {}
                h_rec = dj.get("hybrid_recommendation") or {}
                if h_rec.get("strategy_type") == "daily_top":
                    h_rec["is_participate"] = False
                    h_rec["is_real_bet_target"] = False
                    h_rec["real_bet_level"] = "none"
                    h_rec["real_bet_badge"] = None
                    h_rec["strategy_type"] = "skip"
                    h_rec["strategy_name"] = "見送り推奨 (資金保全)"
                    h_rec["tansho_bets"] = []
                    h_rec["fukusho_bets"] = []
                    h_rec["tansho_amount"] = 0
                    h_rec["fukusho_amount"] = 0
                    h_rec["total_amount"] = 0
                    dj["hybrid_recommendation"] = h_rec
                    rec_obj = dj.get("recommendation") or {}
                    rec_obj["is_pass"] = True
                    rec_obj["grade"] = "D"
                    rec_obj["confidence_label"] = "見送り推奨"
                    rec_obj["tansho_amount"] = 0
                    rec_obj["fukusho_amount"] = 0
                    rec_obj["total_amount"] = 0
                    rec_obj["summary"] = "見送り推奨 (資金保全)"
                    dj["recommendation"] = rec_obj
                    supabase.table("keiba_predictions_cache").update({"data_json": dj}).eq("race_id", rid).execute()
            _daily_top_race_id_cache.pop(target_date, None)
            _all_predictions_cache.pop(target_date, None)
            _today_races_cache.pop(target_date, None)
            return natural_real_rids[0]

        # Step 3: 自然合致が0レースのときのみ、確信度 × エッジ適性スコア 最高位の1レースを実馬券に昇格
        candidates = []
        for r in rows:
            rid = str(r.get("race_id") or "")
            dj = r.get("data_json") or {}
            r_name = str(dj.get("race_name") or "")
            c_det = str((dj.get("race_details") or {}).get("course_details") or "")
            if "障" in r_name or "障害" in c_det or "新馬" in r_name or "メイクデビュー" in r_name:
                continue
            horses = dj.get("horses") or []
            if len(horses) < 8:
                continue

            is_unrun = True
            if is_today:
                p_time_str = (dj.get("race_details") or {}).get("post_time", "")
                post_dt = parse_post_datetime(p_time_str, target_date)
                if post_dt and now_t >= post_dt:
                    is_unrun = False

            h_rec = dj.get("hybrid_recommendation") or {}
            conf = float(h_rec.get("confidence_score") or 0.0)
            edge = float(h_rec.get("edge_ratio") or 1.0)
            score = conf * min(max(edge, 0.8), 2.5)
            candidates.append({
                "race_id": rid,
                "data_json": dj,
                "is_unrun": is_unrun,
                "confidence_score": conf,
                "edge_ratio": edge,
                "score": score
            })

        if not candidates:
            return None

        # 既に daily_top に設定済みのレースがあれば、勝手に他レースへ移らないよう維持
        existing_cand = next((c for c in candidates if c["race_id"] == existing_daily_top_rid), None) if existing_daily_top_rid else None
        if existing_cand:
            top_race = existing_cand
        else:
            unrun_candidates = [c for c in candidates if c["is_unrun"]]
            pool = unrun_candidates if unrun_candidates else candidates
            pool.sort(key=lambda x: (x["score"], x["confidence_score"]), reverse=True)
            top_race = pool[0]

        top_rid = top_race["race_id"]
        top_dj = top_race["data_json"]

        # 本命馬（◎ または prob_win 最大馬）を特定
        horses = top_dj.get("horses") or []
        honmei_horse = next((h for h in horses if h.get("mark") == "◎"), None)
        if not honmei_horse:
            honmei_horse = max(horses, key=lambda x: x.get("prob_win", 0.0)) if horses else {}
        w0_u = int(honmei_horse.get("umaban") or 1)
        w0_name = str(honmei_horse.get("horse_name") or "")
        w0_odds = float(honmei_horse.get("odds") or 3.0)
        c_det = str((top_dj.get("race_details") or {}).get("course_details") or "")
        is_dirt = ("ダ" in c_det) or ("ダート" in c_det)

        h_rec = top_dj.get("hybrid_recommendation") or {}
        h_rec["is_participate"] = True
        h_rec["is_real_bet_target"] = True
        h_rec["real_bet_level"] = "standard"
        h_rec["strategy_type"] = "daily_top"
        h_rec["reason"] = "戦略Daily_No1採択"
        h_rec["confidence_score"] = top_race["confidence_score"]
        h_rec["edge_ratio"] = top_race["edge_ratio"]

        tansho_bets = []
        fukusho_bets = []
        if is_dirt:
            d_tan = 700
            d_fuku = 300
            h_rec["strategy_name"] = f"⭐ 本日厳選No.1: ダート適性単複 (単{d_tan}円＋複{d_fuku}円)"
            h_rec["real_bet_badge"] = "⭐【本日厳選No.1】ダート適性単複 (0R時昇格)"
        elif w0_odds >= 5.0:
            d_tan = 500
            d_fuku = 500
            h_rec["strategy_name"] = f"⭐ 本日厳選No.1: 芝妙味適性単複 (単{d_tan}円＋複{d_fuku}円)"
            h_rec["real_bet_badge"] = "⭐【本日厳選No.1】芝妙味単複 (0R時昇格)"
        else:
            d_tan = 700
            d_fuku = 300
            h_rec["strategy_name"] = f"⭐ 本日厳選No.1: 芝本命適性単複 (単{d_tan}円＋複{d_fuku}円)"
            h_rec["real_bet_badge"] = "⭐【本日厳選No.1】芝本命単複 (0R時昇格)"

        tansho_bets.append({
            "ticket_type": "単勝",
            "combination": [w0_u],
            "combination_str": f"{w0_u}",
            "amount": d_tan,
            "odds": w0_odds,
            "expected_payout": int(round(d_tan * w0_odds)),
            "is_real_odds": True,
            "memo": f"単勝(デイリー厳選) {w0_u}番 ({w0_name}) {d_tan}円"
        })
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        fukusho_bets.append({
            "ticket_type": "複勝",
            "combination": [w0_u],
            "combination_str": f"{w0_u}",
            "amount": d_fuku,
            "odds": est_fuku_odds,
            "expected_payout": int(round(d_fuku * est_fuku_odds)),
            "is_real_odds": False,
            "memo": f"複勝(デイリー厳選) {w0_u}番 ({w0_name}) {d_fuku}円"
        })

        h_rec["tansho_bets"] = tansho_bets
        h_rec["fukusho_bets"] = fukusho_bets
        h_rec["tansho_amount"] = d_tan
        h_rec["fukusho_amount"] = d_fuku
        h_rec["umatan_amount"] = 0
        h_rec["total_amount"] = d_tan + d_fuku
        h_rec["summary"] = f"{h_rec['strategy_name']}：単勝{d_tan}円＋複勝{d_fuku}円 計{d_tan+d_fuku}円"
        h_rec["is_virtual_bet_target"] = True
        h_rec["virtual_bets"] = tansho_bets + fukusho_bets
        h_rec["virtual_total_amount"] = d_tan + d_fuku
        h_rec["virtual_strategy_name"] = h_rec["strategy_name"]
        h_rec["virtual_badge"] = h_rec["real_bet_badge"]
        top_dj["hybrid_recommendation"] = h_rec

        top_rec = top_dj.get("recommendation") or {}
        top_rec["target_umaban"] = w0_u
        top_rec["horse_name"] = w0_name
        top_rec["is_pass"] = False
        top_rec["grade"] = "A"
        top_rec["confidence_label"] = "厳選勝負"
        top_rec["tansho_amount"] = d_tan
        top_rec["fukusho_amount"] = d_fuku
        top_rec["total_amount"] = d_tan + d_fuku
        top_rec["summary"] = h_rec["summary"]
        top_dj["recommendation"] = top_rec

        supabase.table("keiba_predictions_cache").update({"data_json": top_dj}).eq("race_id", top_rid).execute()
        try:
            supabase.table("keiba_scheduled_races").update({"is_predicted": 1}).eq("race_id", top_rid).eq("is_predicted", 0).execute()
        except Exception:
            pass
        print(f"[DailySingleRealBet] {target_date}: 自然合致0Rのため厳選No.1レースとして {top_rid} (◎{w0_u}番 {w0_name}) を昇格しました。")

        # 他の daily_top 昇格レースがあれば解除
        for r in rows:
            r_rid = str(r.get("race_id") or "")
            if r_rid != top_rid:
                r_dj = r.get("data_json") or {}
                r_h = r_dj.get("hybrid_recommendation") or {}
                if r_h.get("strategy_type") == "daily_top" and r_h.get("is_real_bet_target"):
                    r_h["is_participate"] = False
                    r_h["is_real_bet_target"] = False
                    r_h["real_bet_level"] = "none"
                    r_h["real_bet_badge"] = None
                    r_h["strategy_type"] = "skip"
                    r_h["tansho_bets"] = []
                    r_h["fukusho_bets"] = []
                    r_h["tansho_amount"] = 0
                    r_h["fukusho_amount"] = 0
                    r_h["total_amount"] = 0
                    r_dj["hybrid_recommendation"] = r_h
                    supabase.table("keiba_predictions_cache").update({"data_json": r_dj}).eq("race_id", r_rid).execute()

        _daily_top_race_id_cache[target_date] = top_rid
        _all_predictions_cache.pop(target_date, None)
        _today_races_cache.pop(target_date, None)
        return top_rid
    except Exception as e:
        print(f"[ensure_daily_single_real_bet] エラー: {e}")
        return None


def build_prediction_and_recs(race_name: str, race_details: dict, race_id: str, horses: List[dict], course_meta: dict) -> dict:
    empty_recommendation = {
        "target_umaban": None,
        "horse_name": None,
        "grade": "D",
        "confidence_label": "見送り推奨",
        "is_pass": True,
        "tansho_amount": 0,
        "fukusho_amount": 0,
        "total_amount": 0,
        "summary": "推奨馬なし (見送り)"
    }

    if not horses or len(horses) == 0:
        raise ValueError(f"レース {race_id} の出走馬データが空のため予想を生成できません。ダミー値による補完は禁止されています。")

    expected_num_features = 48
    if model_win:
        try:
            expected_num_features = model_win.num_feature()
        except Exception:
            expected_num_features = 48
    elif model:
        try:
            expected_num_features = model.num_feature()
        except Exception:
            expected_num_features = 48

    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    distance = course_meta.get("distance", 1600.0)
    surface_type = course_meta.get("surface_type", 0)
    condition_code = course_meta.get("condition_code", 0)
    turn_type = int(course_meta.get("turn_type", 2 if (venue_code == 4 and surface_type == 0 and int(distance) == 1000) else (1 if venue_code in (4, 5, 7) else 0)))
    race_num_int = int(course_meta.get("race_num_int", int(race_id[10:12]) if len(race_id) == 12 and race_id[10:12].isdigit() else 6))
    is_daily_top_race = bool(course_meta.get("is_daily_top_race", False))

    course_key = f"{venue_code}_{surface_type}"

    j_stats = stats_dict.get("jockey_stats", {})
    jc_stats = stats_dict.get("jockey_course_stats", {})
    t_stats = stats_dict.get("trainer_stats", {})
    tc_stats = stats_dict.get("trainer_course_stats", {})
    h_stats = stats_dict.get("horse_stats", {})

    head_count = len(horses)
    kinryo_list = [float(h.get("kinryo", 55.0)) for h in horses]
    mean_kinryo = sum(kinryo_list) / max(head_count, 1)
    if head_count > 1:
        kinryo_var = sum((x - mean_kinryo) ** 2 for x in kinryo_list) / (head_count - 1)
        kinryo_std_in_race = math.sqrt(max(0.0, kinryo_var))
    else:
        kinryo_std_in_race = 0.0

    # 牝馬限定戦フラグ (出走表テキスト または 全馬が牝馬)
    all_female = head_count > 0 and all(int(h.get("sex_code", 0)) == 1 for h in horses)
    is_female_only = 1 if (course_meta.get("is_female_only_text") or all_female) else 0

    today_date = now_jst().date()
    r_date_meta = course_meta.get("race_date")
    if r_date_meta and isinstance(r_date_meta, str):
        try:
            today_date = datetime.date.fromisoformat(r_date_meta[:10])
        except Exception:
            pass

    horse_feature_data = []
    jc_rates = []
    tc_rates = []
    p1_orders = []
    career_list = []
    wins_list = []

    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = float(h.get("kinryo", 55.0))

        # 馬名の正規化
        raw_name = h.get("horse_name", "")
        h_clean = re.sub(r"[\s\u3000]", "", raw_name)

        stat = h_stats.get(h_clean)
        if stat:
            career_races = float(stat.get("career_races", 0.0))
            career_wins = float(stat.get("career_wins", round(career_races * float(stat.get("career_top3_rate", 0.25)) * 0.33)))
            career_top3_rate = float(stat.get("career_top3_rate", 0.25))
            prev1_order = float(stat.get("prev1_order", 99.0))
            prev1_pop = float(stat.get("prev1_pop", 10.0))
            prev1_odds = float(stat.get("prev1_odds", 50.0))
            prev2_order = float(stat.get("prev2_order", 99.0))

            last_date_str = stat.get("last_date", "")
            if last_date_str:
                try:
                    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    days_since_prev = float(max(0, (today_date - last_date).days))
                except Exception:
                    days_since_prev = 35.0
            else:
                days_since_prev = 35.0

            last_dist = float(stat.get("last_dist", distance))
            dist_diff = float(distance - last_dist)
        else:
            # 過去3年出走データなし（新馬など）
            career_races = 0.0
            career_wins = 0.0
            career_top3_rate = 0.25
            prev1_order = 99.0
            prev1_pop = 10.0
            prev1_odds = 50.0
            prev2_order = 99.0
            days_since_prev = 999.0
            dist_diff = 0.0

        career_list.append(career_races)
        wins_list.append(career_wins)

        j_clean = re.sub(r"[▲△◇☆\s\u3000]", "", h.get("jockey", ""))
        t_clean = re.sub(r"\[.*?\]|[\s\u3000]", "", h.get("trainer", "") or "")

        j_rate = j_stats.get(j_clean, 0.25)
        jc_rate = jc_stats.get(f"{j_clean}_{course_key}", j_rate)
        t_rate = t_stats.get(t_clean, 0.25)
        tc_rate = tc_stats.get(f"{t_clean}_{course_key}", t_rate)

        rel_umaban = u / max(head_count, 1)
        kinryo_diff = k - mean_kinryo

        jc_rates.append(jc_rate)
        tc_rates.append(tc_rate)
        p1_orders.append(prev1_order)

        horse_feature_data.append({
            "w": w, "u": u, "k": k, "dist_diff": dist_diff,
            "career_races": career_races, "career_wins": career_wins, "career_top3_rate": career_top3_rate,
            "prev1_order": prev1_order, "prev1_pop": prev1_pop, "prev1_odds": prev1_odds, "prev2_order": prev2_order,
            "days_since_prev": days_since_prev, "j_rate": j_rate, "jc_rate": jc_rate, "t_rate": t_rate, "tc_rate": tc_rate,
            "rel_umaban": rel_umaban, "kinryo_diff": kinryo_diff,
            "base_feats": [
                w, u, k,
                venue_code, surface_type, distance, dist_diff, condition_code,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev,
                j_rate, jc_rate, t_rate, tc_rate,
                float(head_count), float(rel_umaban), float(kinryo_diff)
            ]
        })

    # レースクラスコードの決定（出走表テキスト優先、不明時はメンバーのキャリア・勝利数から推定）
    race_class_code = course_meta.get("race_class_code")
    if race_class_code is None:
        max_c = max(career_list) if career_list else 0.0
        max_w = max(wins_list) if wins_list else 0.0
        sorted_w = sorted(wins_list)
        q75_w = sorted_w[min(len(sorted_w) - 1, int(len(sorted_w) * 0.75))] if sorted_w else 0.0
        if max_c == 0:
            race_class_code = 0
        elif max_w == 0 and race_num_int <= 6:
            race_class_code = 1
        elif q75_w <= 1.0:
            race_class_code = 2
        elif q75_w <= 2.0:
            race_class_code = 3
        elif q75_w <= 3.0 and not (race_num_int == 11 and max_w >= 4):
            race_class_code = 4
        else:
            race_class_code = 5

    j_ranks = rank_min_desc(jc_rates)
    t_ranks = rank_min_desc(tc_rates)
    p1_ranks = rank_min_asc(p1_orders)

    feature_rows = []
    maiden_feature_rows = []
    for idx, fd in enumerate(horse_feature_data):
        h = horses[idx]
        h_clean = re.sub(r"[\s\u3000]", "", h.get("horse_name", ""))
        stat = h_stats.get(h_clean)

        # 新規拡張特徴量 (7種: ブリンカー、条件替わり、叩き2走目)
        is_blinker_val = float(1 if h.get("is_blinker") else 0)
        is_first_blinker_val = float(1 if h.get("is_first_blinker") else 0)

        last_surface = int(stat.get("last_surface", surface_type)) if stat else surface_type
        is_turf_to_dirt_val = float(1 if (last_surface == 0 and surface_type == 1) else 0)
        is_dirt_to_turf_val = float(1 if (last_surface == 1 and surface_type == 0) else 0)

        last_dist = float(stat.get("last_dist", distance)) if stat else distance
        h_dist_diff = float(distance - last_dist)
        is_dist_shorten_val = float(1 if h_dist_diff <= -200.0 else 0)
        is_dist_lengthen_val = float(1 if h_dist_diff >= 200.0 else 0)

        h_days_since = 35.0
        if stat and stat.get("last_date"):
            try:
                ld = datetime.datetime.strptime(stat["last_date"], "%Y-%m-%d").date()
                h_days_since = float(max(0, (today_date - ld).days))
            except Exception:
                pass
        h_career = float(stat.get("career_races", 0.0)) if stat else 0.0
        was_last_rest = bool(stat.get("was_last_rest", 0)) if stat else False
        is_tataki_2nd_val = float(1 if (was_last_rest and 7.0 <= h_days_since <= 35.0 and h_career >= 2) else 0)

        new_feats = [
            is_blinker_val, is_first_blinker_val,
            is_turf_to_dirt_val, is_dirt_to_turf_val,
            is_dist_shorten_val, is_dist_lengthen_val,
            is_tataki_2nd_val
        ]

        # 末脚・上がり3F ＆ 展開・脚質特徴量 (4種)
        prev1_last_3f_val = float(stat.get("last_3f", 36.0)) if stat and stat.get("last_3f") is not None else 36.0
        prev1_corner_final_val = float(stat.get("last_corner_final", 8.0)) if stat and stat.get("last_corner_final") is not None else 8.0
        prev1_is_front_val = 1.0 if prev1_corner_final_val <= 3.0 else 0.0
        prev1_is_closer_val = 1.0 if prev1_corner_final_val >= 8.0 else 0.0

        agari_feats = [
            prev1_last_3f_val,
            prev1_corner_final_val,
            prev1_is_front_val,
            prev1_is_closer_val
        ]

        # ★ 馬体重増減・コース回り・レースクラス条件・性齢 (12種)
        raw_hw = h.get("horse_weight")
        if raw_hw is not None and float(raw_hw) > 0:
            hw_val = float(raw_hw)
        elif stat and stat.get("last_weight") is not None and float(stat.get("last_weight") or 0) > 0:
            hw_val = float(stat["last_weight"])
        else:
            hw_val = 470.0

        raw_wd = h.get("weight_diff")
        wd_val = float(raw_wd) if raw_wd is not None else 0.0
        abs_wd_val = abs(wd_val)
        k_val = float(h.get("kinryo", 55.0))
        weight_ratio_val = k_val / max(hw_val, 350.0)

        last_turn = int(stat.get("last_turn", turn_type)) if stat and stat.get("last_turn") is not None else turn_type
        is_turn_change_val = float(1 if (h_career > 0 and turn_type != last_turn) else 0)

        sex_code_val = float(h.get("sex_code", 0))
        age_val = float(h.get("age", 4.0))

        extra12_feats = [
            hw_val, wd_val, abs_wd_val, weight_ratio_val,
            float(turn_type), is_turn_change_val,
            float(race_num_int), float(race_class_code), float(is_female_only), float(kinryo_std_in_race),
            sex_code_val, age_val
        ]

        row25 = fd["base_feats"] + [float(j_ranks[idx]), float(t_ranks[idx]), float(p1_ranks[idx])]
        row32 = row25 + new_feats
        row36 = row32 + agari_feats
        row48 = row36 + extra12_feats

        # 50特徴量 (DynamicPercentile_Top5Leverage) の構築
        c_races = float(fd.get("career_races", 0.0))
        c_wins = float(fd.get("career_wins", 0.0))
        c_win_rate = float(c_wins / max(c_races, 1.0)) if c_races > 0 else 0.08
        j_win_rate = round(float(fd.get("j_rate", 0.25)) * 0.33, 4)
        jc_win_rate = round(float(fd.get("jc_rate", 0.25)) * 0.33, 4)
        t_win_rate = round(float(fd.get("t_rate", 0.25)) * 0.33, 4)
        tc_win_rate = round(float(fd.get("tc_rate", 0.25)) * 0.33, 4)

        j_clean_str = re.sub(r"[▲△◇☆\s\u3000]", "", h.get("jockey", ""))
        t_clean_str = re.sub(r"\[.*?\]|[\s\u3000]", "", h.get("trainer", "") or "")
        jt_key = f"{j_clean_str}_{t_clean_str}"
        jt_rate_val = float(stats_dict.get("jockey_trainer_stats", {}).get(jt_key, 0.25))

        k_val = float(fd.get("k", 55.0))
        last_k = float(stat.get("last_kinryo", k_val)) if stat else k_val
        kinryo_diff_prev_val = float(k_val - last_k)

        p1_order_val = float(fd.get("prev1_order", 99.0))
        p1_pop_val = float(fd.get("prev1_pop", 10.0))
        p1_time_diff_val = float(stat.get("prev1_time_diff", 3.0)) if stat else 3.0
        p1_pop_order_diff_val = float(p1_pop_val - p1_order_val)
        p1_l3f_rank_val = float(stat.get("prev1_last_3f_rank", 10.0)) if stat else 10.0

        hs_key = f"{h_clean}_{surface_type}"
        horse_surf_val = float(stats_dict.get("horse_surface_stats", {}).get(hs_key, 0.25))

        prev_time_diff_min_val = float(stat.get("prev_time_diff_min", 3.0)) if stat else 3.0

        js_key = f"{j_clean_str}_{surface_type}"
        jockey_surf_val = float(stats_dict.get("jockey_surface_stats", {}).get(js_key, 0.25))

        p1_class_diff_val = float(stat.get("prev1_class_diff", 0.0)) if stat else 0.0

        row50 = [
            float(fd["w"]), float(fd["u"]), float(k_val), float(venue_code), float(surface_type), float(distance), float(fd["dist_diff"]), float(condition_code),
            float(c_races), float(c_win_rate), float(fd["career_top3_rate"]),
            float(p1_order_val), float(p1_pop_val), float(fd["prev1_odds"]), float(fd["prev2_order"]), float(fd["days_since_prev"]),
            float(j_win_rate), float(fd["j_rate"]), float(jc_win_rate), float(fd["jc_rate"]),
            float(t_win_rate), float(fd["t_rate"]), float(tc_win_rate), float(fd["tc_rate"]),
            float(head_count), float(fd["rel_umaban"]), float(fd["kinryo_diff"]),
            float(j_ranks[idx]), float(t_ranks[idx]), float(p1_ranks[idx]),
            float(is_blinker_val), float(is_first_blinker_val), float(is_turf_to_dirt_val), float(is_dirt_to_turf_val),
            float(is_dist_shorten_val), float(is_dist_lengthen_val), float(is_tataki_2nd_val),
            float(prev1_last_3f_val), float(prev1_corner_final_val), float(prev1_is_front_val), float(prev1_is_closer_val),
            float(jt_rate_val), float(kinryo_diff_prev_val), float(p1_time_diff_val), float(p1_pop_order_diff_val), float(p1_l3f_rank_val),
            float(horse_surf_val), float(prev_time_diff_min_val), float(jockey_surf_val), float(p1_class_diff_val)
        ]

        # 未勝利戦専用特徴量 (54特徴量)
        is_prev_near_miss_val = float(1 if (p1_time_diff_val <= 0.3 or p1_order_val <= 3.0) else 0)
        is_career_prime_val = float(1 if c_races in [1.0, 2.0] else 0)
        is_career_stale_val = float(1 if c_races >= 8.0 else 0)
        is_prev_fastest_3f_val = float(1 if p1_l3f_rank_val <= 3.0 else 0)
        row_maiden = row50[:49] + [float(jockey_surf_val), is_prev_near_miss_val, is_career_prime_val, is_career_stale_val, is_prev_fastest_3f_val]
        maiden_feature_rows.append(row_maiden)

        if expected_num_features == 50:
            feature_rows.append(row50)
        elif expected_num_features == 48:
            feature_rows.append(row48)
        elif expected_num_features == 36:
            feature_rows.append(row36)
        elif expected_num_features == 32:
            feature_rows.append(row32)
        elif expected_num_features == 25:
            feature_rows.append(row25)
        elif expected_num_features == 19:
            feature_rows.append(row25[:19])
        else:
            feature_rows.append(row50[:expected_num_features] if len(row50) >= expected_num_features else row48[:expected_num_features])

    # ★ Model B (Value Hunter: 1番人気抑制＋中穴過小評価馬強化) による全レース統一推論
    if model_win and model_top3:
            try:
                n_win = model_win.num_feature()
                n_top3 = model_top3.num_feature()
                w_scores = model_win.predict([r[:n_win] for r in feature_rows])
                t_scores = model_top3.predict([r[:n_top3] for r in feature_rows])
                for idx, h in enumerate(horses):
                    h["prob_win"] = round(float(w_scores[idx]), 4)
                    h["prob_top3"] = round(float(t_scores[idx]), 4)
                    h["score"] = h["prob_win"]
            except Exception as e:
                raise RuntimeError(f"AIモデル(Win/Top3)推論エラー ({race_id}): {e}")
    elif model:
        try:
            n_mod = model.num_feature()
            scores = model.predict([r[:n_mod] for r in feature_rows])
            for idx, h in enumerate(horses):
                h["prob_win"] = round(float(scores[idx]), 4)
                h["prob_top3"] = round(float(scores[idx]), 4)
                h["score"] = round(float(scores[idx]), 4)
        except Exception as e:
            raise RuntimeError(f"AIモデル推論エラー ({race_id}): {e}")
    else:
        raise RuntimeError(f"AIモデルがロードされていないため、レース {race_id} の推論を実行できません。")

    # ※ 馬体重および増減（horse_weight, weight_diff, abs_weight_diff, weight_ratio）は
    #    48特徴量モデル（row48）の正式な特徴量として既に組み込まれているため、
    #    二重補正を避け、モデル本来の48特徴量推論スコアをそのまま使用する。

    # 1着特化モデルのスコア順で◎, ◯を決定
    sorted_by_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    for h in horses:
        h["mark"] = "-"

    if len(sorted_by_win) >= 1:
        sorted_by_win[0]["mark"] = "◎"
    if len(sorted_by_win) >= 2:
        sorted_by_win[1]["mark"] = "◯"

    # 残りの馬の中から、3着内特化モデルのスコア順で ▲, △, ☆ を決定
    remaining_horses = [h for h in sorted_by_win if h["mark"] == "-"]
    sorted_by_top3 = sorted(remaining_horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
    if len(sorted_by_top3) >= 1:
        sorted_by_top3[0]["mark"] = "▲"
    if len(sorted_by_top3) >= 2:
        sorted_by_top3[1]["mark"] = "△"
    if len(sorted_by_top3) >= 3:
        sorted_by_top3[2]["mark"] = "☆"

    sorted_horses = sorted_by_win
    honmei_horse = next((h for h in sorted_horses if h["mark"] == "◎"), None)

    # ★ 案1 & 案2 対応: 買い目生成前に各馬のオッズ異常・乖離インジケーターを算出して付与
    prev_h_list = (course_meta or {}).get("prev_horses")
    stg = (course_meta or {}).get("odds_stage")
    apply_odds_anomaly_indicators(horses, prev_horses=prev_h_list, odds_stage=stg)

    if honmei_horse:
        u_num = honmei_horse["umaban"]
        h_name = honmei_horse["horse_name"]
        h_score = honmei_horse["score"]
        h_odds = honmei_horse.get("odds")

        grade, conf_label, is_pass, b_tan, b_fuku = calculate_dynamic_bet(h_score, h_odds)
        total_amt = b_tan + b_fuku

        if is_pass:
            if h_odds is not None and h_odds < 2.5:
                summary_text = f"{u_num}番 {h_name} [{conf_label}] (単勝{h_odds}倍のためトリガミ回避・見送り推奨)"
            else:
                summary_text = f"{u_num}番 {h_name} [{conf_label}] (期待値不足のため見送り推奨)"
        else:
            summary_text = f"{u_num}番 {h_name} [{conf_label}] (単勝{b_tan}円 + 複勝{b_fuku}円 / 計{total_amt}円)"

        recommendation = {
            "target_umaban": u_num,
            "horse_name": h_name,
            "grade": grade,
            "confidence_label": conf_label,
            "is_pass": is_pass,
            "tansho_amount": b_tan,
            "fukusho_amount": b_fuku,
            "total_amount": total_amt,
            "summary": summary_text
        }
    else:
        raise ValueError(f"レース {race_id} の本命馬(◎)を特定できませんでした。ダミー買い目の生成は禁止されています。")

    # -------------------------------------------------------------
    # ★ 波乱度メタモデル × 動的資金配分・券種切り替え戦略
    # -------------------------------------------------------------
    hybrid_rec: Optional[dict] = None
    track_bias_data: Optional[dict] = None
    if len(horses) < 3:
        raise ValueError(f"レース {race_id} の出走頭数が少なすぎます ({len(horses)}頭)。ダミー予想の生成は禁止されています。")

    # 馬番・印の取得
    h_honmei = next((h for h in horses if h["mark"] == "◎"), sorted_horses[0])
    h_taiko = next((h for h in horses if h["mark"] == "◯"), (sorted_horses[1] if len(sorted_horses) > 1 else sorted_horses[0]))
    h_kuro = next((h for h in horses if h["mark"] == "▲"), (sorted_horses[2] if len(sorted_horses) > 2 else sorted_horses[0]))
    h_osae = next((h for h in horses if h["mark"] == "△"), (sorted_horses[3] if len(sorted_horses) > 3 else sorted_horses[0]))
    h_ana = next((h for h in horses if h["mark"] == "☆"), (sorted_horses[4] if len(sorted_horses) > 4 else sorted_horses[0]))

    u0 = int(h_honmei["umaban"])
    u1 = int(h_taiko["umaban"])
    u2 = int(h_kuro["umaban"])
    u3 = int(h_osae["umaban"])
    u4 = int(h_ana["umaban"])

    # オッズの集計
    s_odds = sorted(horses, key=lambda x: (x.get("odds") if x.get("odds") is not None and x.get("odds") > 0 else 999.0))
    o1 = float(s_odds[0].get("odds") or 50.0) if len(s_odds) >= 1 else 50.0
    o2 = float(s_odds[1].get("odds") or 50.0) if len(s_odds) >= 2 else 50.0
    o3 = float(s_odds[2].get("odds") or 50.0) if len(s_odds) >= 3 else 50.0

    inv_sum = sum(1.0 / max(float(h.get("odds") or 50.0), 1.01) for h in horses)
    shares = [(1.0 / max(float(h.get("odds") or 50.0), 1.01)) / inv_sum for h in horses]
    top3_share = sum(sorted(shares, reverse=True)[:3])
    entropy = -sum(p * math.log(max(p, 1e-6)) for p in shares)

    s_ai = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    ai_w1 = float(s_ai[0].get("prob_win", 0.0)) if len(s_ai) >= 1 else 0.0
    ai_w2 = float(s_ai[1].get("prob_win", 0.0)) if len(s_ai) >= 2 else 0.0
    ai_diff = ai_w1 - ai_w2

    front_runners = 0
    for h in horses:
        h_clean = re.sub(r"[\s\u3000]", "", h.get("horse_name", ""))
        st = h_stats.get(h_clean, {})
        if float(st.get("last_corner_final", 8.0)) <= 3.0:
            front_runners += 1
    front_ratio = front_runners / max(len(horses), 1)

    race_feat_row = [
        float(head_count), float(distance), float(surface_type), float(condition_code), float(venue_code),
        float(o1), float(o2), float(o3), float(o2 / max(o1, 0.5)),
        float(top3_share), float(entropy),
        float(ai_w1), float(ai_w2), float(ai_diff),
        float(front_runners), float(front_ratio),
        float(turn_type), float(race_class_code), float(is_female_only), float(kinryo_std_in_race)
    ]

    if not model_gachi:
        raise RuntimeError(f"model_gachi がロードされていないため、レース {race_id} のガチ度を推論できません。")
    try:
        n_g = model_gachi.num_feature()
        prob_gachi = round(float(model_gachi.predict([race_feat_row[:n_g]])[0]), 4)
    except Exception as e:
        raise RuntimeError(f"model_gachi 推論エラー ({race_id}): {e}")

    if not model_himo:
        raise RuntimeError(f"model_himo がロードされていないため、レース {race_id} の混戦度を推論できません。")
    try:
        n_h = model_himo.num_feature()
        prob_himo = round(float(model_himo.predict([race_feat_row[:n_h]])[0]), 4)
    except Exception as e:
        raise RuntimeError(f"model_himo 推論エラー ({race_id}): {e}")

    # Webリアルタイムオッズ取得
    real_odds_data = fetch_real_odds(race_id)
    umaren_real_map = real_odds_data.get("umaren", {})
    trio_real_map = real_odds_data.get("trio", {})

    def get_umaren_odds_info(ua, ub):
        comb = (min(ua, ub), max(ua, ub))
        if comb in umaren_real_map:
            return umaren_real_map[comb], True
        else:
            h_a = next((h for h in horses if int(h.get("umaban", 0)) == ua), {})
            h_b = next((h for h in horses if int(h.get("umaban", 0)) == ub), {})
            oa = float(h_a.get("odds") or 0.0)
            ob = float(h_b.get("odds") or 0.0)
            est_o = round(max(2.5, 0.65 * oa * ob), 1) if (oa > 0 and ob > 0) else 0.0
            return est_o, False

    u01_odds, u01_is_real = get_umaren_odds_info(u0, u1) if (u0 and u1) else (0.0, False)
    u02_odds, u02_is_real = get_umaren_odds_info(u0, u2) if (u0 and u2) else (0.0, False)
    u03_odds, u03_is_real = get_umaren_odds_info(u0, u3) if (u0 and u3) else (0.0, False)

    any_real_odds = u01_is_real or u02_is_real or u03_is_real

    opponents = []
    if u1: opponents.append((u1, u01_odds, u01_is_real, "◯"))
    if u2: opponents.append((u2, u02_odds, u02_is_real, "▲"))
    if u3: opponents.append((u3, u03_odds, u03_is_real, "△"))

    r_date_str = (course_meta or {}).get("race_date") or today_jst_str()
    force_final_flag = bool((course_meta or {}).get("force_final_odds", False))

    c_det = (race_details or {}).get("course_details", "") if isinstance(race_details, dict) else ""
    surf_str = "芝" if "芝" in c_det else ("ダート" if "ダ" in c_det else "障害")
    track_bias_data = compute_track_bias_and_matches(
        race_id=race_id,
        race_date=r_date_str,
        surface_name=surf_str,
        horses=sorted_horses
    )

    hybrid_rec = compute_hybrid_rec_from_inputs(
        race_name=race_name,
        race_details=race_details,
        horses=sorted_horses,
        surface_type=surface_type,
        prob_gachi=prob_gachi,
        prob_himo=prob_himo,
        opponents=opponents,
        u0=u0,
        honmei_horse=honmei_horse,
        race_date_str=r_date_str,
        force_final_odds=force_final_flag,
        real_trio_odds=trio_real_map,
        track_bias_data=track_bias_data,
        is_daily_top_race=is_daily_top_race
    )

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    # recommendation (TanFuku) を hybrid_rec (B->A->Cカスケード動的配分) と完全に同期
    if hybrid_rec:
        is_p = hybrid_rec.get("is_participate", False)
        t_amt = hybrid_rec.get("tansho_amount", 0)
        f_amt = hybrid_rec.get("fukusho_amount", 0)
        tot_amt = hybrid_rec.get("total_amount", 0)
        st_type = hybrid_rec.get("strategy_type", "skip")
        grd = "S" if st_type == "strategy_b" else ("A" if st_type == "strategy_a" else ("B" if st_type == "strategy_c" else "D"))
        clabel = "超勝負R" if st_type == "strategy_b" else ("芝厳選" if st_type == "strategy_a" else ("ダート本命" if st_type == "strategy_c" else "見送り推奨"))
        recommendation = {
            "target_umaban": u0 if is_p else (honmei_horse["umaban"] if honmei_horse else None),
            "horse_name": honmei_horse["horse_name"] if honmei_horse else None,
            "grade": grd,
            "confidence_label": clabel,
            "is_pass": not is_p,
            "tansho_amount": t_amt,
            "fukusho_amount": f_amt,
            "total_amount": tot_amt,
            "summary": hybrid_rec.get("summary", "")
        }

    return {
        "race_name": race_name,
        "race_details": race_details,
        "horses": horses_sorted_by_num,
        "recommendation": recommendation,
        "hybrid_recommendation": hybrid_rec,
        "track_bias": track_bias_data
    }


def compute_hybrid_rec_from_inputs(
    race_name: str,
    race_details: Optional[dict],
    horses: List[dict],
    surface_type: int,
    prob_gachi: float,
    prob_himo: float,
    opponents: List[tuple],
    u0: int,
    honmei_horse: Optional[dict],
    race_date_str: str,
    force_final_odds: bool = False,
    real_trio_odds: Optional[dict] = None,
    track_bias_data: Optional[dict] = None,
    is_daily_top_race: bool = False
) -> dict:
    """
    【モジュール分離アーキテクチャ】
    買い目戦略・調停・トリガミ防止・スリッページ停止判定を app/strategies/arbiter.py へ委任
    """
    return _strategy_arbiter.evaluate_race(
        race_name=race_name,
        race_details=race_details,
        horses=horses,
        surface_type=surface_type,
        prob_gachi=prob_gachi,
        prob_himo=prob_himo,
        opponents=opponents,
        u0=u0,
        honmei_horse=honmei_horse,
        race_date_str=race_date_str,
        force_final_odds=force_final_odds,
        real_trio_odds=real_trio_odds,
        track_bias_data=track_bias_data,
        is_daily_top_race=is_daily_top_race
    )


# =========================================================================
# ★ Supabase CRUD ヘルパー
# =========================================================================

ODDS_STAGE_ORDER = ["0830", "60m", "30m", "15m", "05m"]
ODDS_STAGE_RANK = {"0830": 1, "60m": 2, "30m": 3, "15m": 4, "05m": 5}


def infer_odds_stage_from_minutes(mins_to_post: Optional[float]) -> str:
    """発走までの残り分数から現在のオッズ取得ステージを判定"""
    if mins_to_post is None or mins_to_post > 65.0:
        return "0830"
    elif mins_to_post > 33.0:
        return "60m"
    elif mins_to_post > 17.0:
        return "30m"
    elif mins_to_post > 6.5:
        return "15m"
    else:
        return "05m"


def apply_odds_anomaly_indicators(
    horses: List[dict],
    prev_horses: Optional[List[dict]] = None,
    odds_stage: Optional[str] = None
):
    """
    【Step 4】異常オッズ・直前急変の検知（マルチステージ時系列支持率変動 ＆ AI期待値乖離インジケーター）
    ・5段階の時系列スナップショット ["0830", "60m", "30m", "15m", "05m"] を `odds_history` に保持。
    1. 時系列オッズ変動:
       - 🔥連続急落 / 🔥直前急落 / 🔥大口急落 (plunge):
         3ステージ以上の連続下落、または馬体重発表後(60m/30m/15m)からの直前20%以上急落、または基準比25%以上急落（かつ1.0倍以上低下）
       - 📉連続支持増 / 📉支持増 (drop):
         基準比15%〜25%の低下（かつ0.5倍以上低下）、または連続下落で10%以上低下
       - 📈人気軟化 (drift):
         基準比30%以上の上昇（かつ1.5倍以上上昇）
    2. AI評価×市場オッズの異常乖離シグナル:
       - AI印あり(◎◯▲△☆) かつ 5番人気以下(または単勝10倍以上) かつ 期待値(prob_win * odds) >= 1.15: 💎妙味穴 (value_gap)
       - AI無印(-) かつ 1〜2番人気(または単勝3.5倍以下) かつ 期待値(prob_win * odds) < 0.45: ⚠️過剰人気 (overbet)
    ※ LightGBMモデルの推論確率(prob_win / prob_top3)自体は一切変更しない（ゼロリーク・ベースライン厳守）。
    """
    if not horses:
        return
    prev_map = {}
    if prev_horses:
        for ph in prev_horses:
            if "umaban" in ph and str(ph["umaban"]).isdigit():
                prev_map[int(ph["umaban"])] = ph

    for h in horses:
        try:
            u = int(h.get("umaban", 0))
        except Exception:
            continue
        curr_raw = h.get("odds")
        curr_odds = round(float(curr_raw), 1) if (curr_raw is not None and float(curr_raw) > 0) else None
        prev_h = prev_map.get(u, {})

        # 既存の odds_history を引き継ぎ・統合
        hist: Dict[str, float] = {}
        for src in (prev_h.get("odds_history"), h.get("odds_history")):
            if isinstance(src, dict):
                for k, v in src.items():
                    if k in ODDS_STAGE_RANK and v is not None:
                        try:
                            fv = round(float(v), 1)
                            if fv > 0:
                                hist[k] = fv
                        except Exception:
                            pass

        base_raw = hist.get("0830") or h.get("base_odds") or prev_h.get("base_odds") or prev_h.get("odds") or curr_odds
        base_odds = round(float(base_raw), 1) if (base_raw is not None and float(base_raw) > 0) else None
        if base_odds and "0830" not in hist:
            hist["0830"] = base_odds

        if curr_odds:
            if odds_stage in ODDS_STAGE_RANK:
                hist[odds_stage] = curr_odds
            elif not hist:
                hist["0830"] = curr_odds

        ordered_hist = {k: hist[k] for k in ODDS_STAGE_ORDER if k in hist}
        h["odds_history"] = ordered_hist
        h["base_odds"] = base_odds

        # 馬体重発表後(60m)の基準オッズおよび直前比較用オッズ
        weight_base = ordered_hist.get("60m") or base_odds
        stage_vals = [ordered_hist[k] for k in ODDS_STAGE_ORDER if k in ordered_hist]
        if curr_odds and (not stage_vals or stage_vals[-1] != curr_odds):
            stage_vals.append(curr_odds)

        # 連続同一値を圧縮した推移リスト (表示・連続下落判定用)
        compact_vals: List[float] = []
        for v in stage_vals:
            if not compact_vals or compact_vals[-1] != v:
                compact_vals.append(v)

        is_continuous_drop = len(compact_vals) >= 3 and all(
            compact_vals[i] > compact_vals[i + 1] for i in range(len(compact_vals) - 1)
        )

        if curr_odds and base_odds and base_odds > 0:
            pct = round(((curr_odds - base_odds) / base_odds) * 100.0, 1)
        else:
            pct = 0.0
        h["odds_change_pct"] = pct

        late_pct = 0.0
        is_late_plunge = False
        if curr_odds and weight_base and weight_base > 0 and "60m" in ordered_hist:
            late_pct = round(((curr_odds - weight_base) / weight_base) * 100.0, 1)
            if curr_odds <= weight_base * 0.80 and (weight_base - curr_odds) >= 1.0:
                is_late_plunge = True

        # 推移テキスト (3点以上ある場合は直近3ステップの推移を表示)
        if len(compact_vals) >= 3:
            traj_str = "→".join(f"{v:.1f}" for v in compact_vals[-3:]) + "倍"
        elif base_odds and curr_odds:
            traj_str = f"{base_odds:.1f}→{curr_odds:.1f}倍"
        else:
            traj_str = f"{curr_odds:.1f}倍" if curr_odds else ""

        mark = str(h.get("mark") or "-")
        pop_raw = h.get("popularity")
        pop = int(pop_raw) if (pop_raw is not None and str(pop_raw).isdigit() and int(pop_raw) > 0) else None
        pw_raw = float(h.get("prob_win") or h.get("score") or 0.0)
        prob_win = pw_raw / 100.0 if pw_raw > 1.0 else pw_raw
        ev = round(prob_win * curr_odds, 2) if curr_odds else 0.0

        trend = None
        alert = None
        if curr_odds and base_odds and (
            (curr_odds <= base_odds * 0.75 and (base_odds - curr_odds) >= 1.0)
            or is_late_plunge
            or (is_continuous_drop and curr_odds <= base_odds * 0.82 and (base_odds - curr_odds) >= 1.0)
        ):
            trend = "plunge"
            if is_continuous_drop:
                alert = f"🔥連続急落 {traj_str} ({pct:+.0f}%)"
            elif is_late_plunge and pct > -25.0:
                alert = f"🔥直前急落 {weight_base:.1f}→{curr_odds:.1f}倍 ({late_pct:+.0f}%)"
            else:
                alert = f"🔥大口急落 {traj_str} ({pct:+.0f}%)"
        elif curr_odds and base_odds and (
            (curr_odds <= base_odds * 0.85 and (base_odds - curr_odds) >= 0.5)
            or (is_continuous_drop and curr_odds <= base_odds * 0.90 and (base_odds - curr_odds) >= 0.5)
        ):
            trend = "drop"
            if is_continuous_drop:
                alert = f"📉連続支持増 {traj_str} ({pct:+.0f}%)"
            else:
                alert = f"📉支持増 {traj_str} ({pct:+.0f}%)"
        elif mark in ("◎", "◯", "〇", "○", "▲", "△", "☆") and curr_odds and ((pop is not None and pop >= 5) or curr_odds >= 10.0) and ev >= 1.15:
            trend = "value_gap"
            pop_str = f"{pop}人気" if pop else f"{curr_odds:.1f}倍"
            alert = f"💎妙味穴 ({pop_str} / 期待値{ev:.2f})"
        elif mark == "-" and curr_odds and ((pop is not None and pop <= 2) or curr_odds <= 3.5) and ev < 0.45:
            trend = "overbet"
            pop_str = f"{pop}人気" if pop else f"{curr_odds:.1f}倍"
            alert = f"⚠️過剰人気 ({pop_str} / AI無印)"
        elif curr_odds and base_odds and curr_odds >= base_odds * 1.30 and (curr_odds - base_odds) >= 1.5:
            trend = "drift"
            alert = f"📈人気軟化 {traj_str} ({pct:+.0f}%)"

        h["odds_trend"] = trend
        h["odds_alert"] = alert


_track_bias_cache: Dict[str, Tuple[float, dict]] = {}
_day_results_cache: Dict[str, Tuple[float, List[dict]]] = {}
_past_pred_cache: Dict[str, dict] = {}

def compute_track_bias_and_matches(
    race_id: str,
    race_date: str,
    surface_name: str,
    horses: List[dict],
    prefetched_settled_races: Optional[List[dict]] = None,
    prefetched_pred_map: Optional[Dict[str, dict]] = None
) -> Optional[dict]:
    """
    【Step 5】トラックバイアス（当日の馬場傾向：内枠/外枠・先行/差し有利）のリアルタイム分析
    1. 当日(race_date)・同競馬場(race_id[4:6])・同トラック種別(芝/ダート)の確定済みレース(1〜3着)を集計。
    2. 内枠(1〜3枠)・外枠(6〜8枠)の複勝シェア率、および先行逃げ(4角4番手以内)の勝率から馬場バイアス指数を算出。
    3. 出走各馬の枠番・脚質とのマッチ度(bias_match)を判定。
    ※ モデル推論確率(prob_win/prob_top3)自体は変更しない（ゼロリーク・ベースライン厳守）。
    """
    if not surface_name or surface_name in ("障害", ""):
        return None

    venue_code_str = race_id[4:6] if len(race_id) >= 6 else "00"
    try:
        venue_code_int = int(venue_code_str)
    except Exception:
        venue_code_int = 0
    venue_name = GLOBAL_VENUE_MAP.get(venue_code_int, "競馬場")

    cache_key = f"{race_id}_{surface_name}"
    now_ts = time.time()
    cached_entry = _track_bias_cache.get(cache_key)

    bias_info = None
    if cached_entry and (now_ts - cached_entry[0] < 120.0):
        bias_info = cached_entry[1]
    elif supabase:
        try:
            day_prefix = race_id[:10] if len(race_id) >= 10 else race_id[:6]
            settled_races = []

            if prefetched_settled_races is not None:
                for r in prefetched_settled_races:
                    rid = str(r.get("race_id") or "")
                    if rid[:10] == day_prefix and rid < race_id:
                        settled_races.append(r)
            else:
                c_day_entry = _day_results_cache.get(day_prefix)
                if c_day_entry and (now_ts - c_day_entry[0] < 60.0):
                    for r in c_day_entry[1]:
                        if str(r.get("race_id") or "") < race_id:
                            settled_races.append(r)
                else:
                    start_rid = f"{day_prefix}01"
                    day_settled = []
                    try:
                        res_cache = supabase.table("keiba_race_results_cache")\
                            .select("race_id, result_json")\
                            .gte("race_id", start_rid)\
                            .lte("race_id", f"{day_prefix}12")\
                            .execute()
                        for r in (res_cache.data or []):
                            rid = r.get("race_id")
                            rj = r.get("result_json") or {}
                            orders = rj.get("orders") or {}
                            if orders:
                                day_settled.append({"race_id": rid, "orders": orders})
                    except Exception as e:
                        print(f"[TrackBias] results_cache query warning: {e}")

                    if not day_settled:
                        try:
                            s_res = supabase.table("keiba_race_settlements")\
                                .select("race_id, bet_details")\
                                .gte("race_id", start_rid)\
                                .lte("race_id", f"{day_prefix}12")\
                                .execute()
                            for r in (s_res.data or []):
                                rid = r.get("race_id")
                                b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
                                top3 = b_det.get("real_bet_meta", {}).get("top3_orders") or []
                                if top3:
                                    day_settled.append({"race_id": rid, "top3_orders": top3})
                        except Exception as e:
                            print(f"[TrackBias] settlements query warning: {e}")

                    _day_results_cache[day_prefix] = (now_ts, day_settled)
                    for r in day_settled:
                        if str(r.get("race_id") or "") < race_id:
                            settled_races.append(r)

            sample_races = 0
            inside_count = 0
            middle_count = 0
            outside_count = 0
            front_wins = 0

            if settled_races:
                # 確定レースのコース条件(芝/ダート)および枠番マップを取得
                past_rids = [str(r["race_id"]) for r in settled_races if r.get("race_id")]
                pred_map = {}
                if prefetched_pred_map is not None:
                    pred_map = prefetched_pred_map
                elif past_rids:
                    missing_rids = [r for r in past_rids if r not in _past_pred_cache]
                    if missing_rids:
                        try:
                            p_res = supabase.table("keiba_predictions_cache")\
                                .select("race_id, data_json")\
                                .in_("race_id", missing_rids)\
                                .execute()
                            for row in (p_res.data or []):
                                if row.get("race_id"):
                                    _past_pred_cache[row["race_id"]] = (row.get("data_json") or {})
                        except Exception as pe:
                            print(f"[TrackBias] predictions_cache fetch warning: {pe}")
                    pred_map = {r: _past_pred_cache.get(r, {}) for r in past_rids}

                is_target_turf = "芝" in surface_name
                is_target_dirt = ("ダ" in surface_name) or ("ダート" in surface_name)

                for s_item in settled_races:
                    r_id = str(s_item.get("race_id") or "")
                    dj = pred_map.get(r_id, {})
                    c_details = (dj.get("race_details") or {}).get("course_details", "")
                    # トラック種別の一致判定 (芝: '芝', ダート: 'ダ' or 'ダート')
                    past_is_turf = "芝" in c_details
                    past_is_dirt = ("ダ" in c_details) or ("ダート" in c_details)
                    if is_target_turf and not past_is_turf:
                        continue
                    if is_target_dirt and not past_is_dirt:
                        continue
                    if not is_target_turf and not is_target_dirt:
                        continue

                    # 1〜3着馬番リストの構築
                    top3_list = []
                    if "orders" in s_item:
                        for u_k, o_v in (s_item.get("orders") or {}).items():
                            try:
                                u_int = int(u_k)
                                o_int = int(o_v)
                                if o_int in (1, 2, 3):
                                    top3_list.append({"umaban": u_int, "order": o_int})
                            except Exception:
                                pass
                        top3_list.sort(key=lambda x: x["order"])
                    elif "top3_orders" in s_item:
                        for it in (s_item.get("top3_orders") or []):
                            if isinstance(it, dict):
                                top3_list.append({
                                    "umaban": int(it.get("umaban") or 0),
                                    "order": int(it.get("order") or 1),
                                    "horse_name": it.get("horse_name", "")
                                })
                            else:
                                try:
                                    top3_list.append({"umaban": int(it), "order": 1, "horse_name": ""})
                                except Exception:
                                    pass

                    if len(top3_list) < 3:
                        continue

                    sample_races += 1
                    r_horses = dj.get("horses") or []
                    w_map = {int(h["umaban"]): int(h.get("wakuban") or 0) for h in r_horses if "umaban" in h and str(h["umaban"]).isdigit()}
                    name_map = {int(h["umaban"]): h.get("horse_name", "") for h in r_horses if "umaban" in h and str(h["umaban"]).isdigit()}

                    for item in top3_list[:3]:
                        u_num = item["umaban"]
                        order_num = item["order"]
                        h_name = item.get("horse_name") or name_map.get(u_num, "")
                        h_clean = re.sub(r"[\s\u3000]", "", str(h_name))

                        waku_num = w_map.get(u_num, 0)
                        # 枠番が未設定の場合のフォールバック (馬番から近似)
                        if waku_num == 0 and u_num > 0:
                            waku_num = min(8, max(1, (u_num - 1) // 2 + 1))

                        if waku_num in (1, 2, 3):
                            inside_count += 1
                        elif waku_num in (4, 5):
                            middle_count += 1
                        elif waku_num in (6, 7, 8):
                            outside_count += 1

                        if order_num == 1 and h_clean:
                            st = stats_dict.get("horse_stats", {}).get(h_clean, {})
                            if float(st.get("last_corner_final", 8.0)) <= 4.0:
                                front_wins += 1

            if sample_races < 2:
                bias_info = {
                    "venue_name": venue_name,
                    "surface_name": surface_name,
                    "sample_races": sample_races,
                    "bias_summary": f"⚖️ 当日{surface_name}分析中 (確定{sample_races}R)",
                    "waku_bias": "flat",
                    "running_style_bias": "flat",
                    "inside_top3_ratio": 0.0,
                    "outside_top3_ratio": 0.0,
                    "detail_text": f"本日{surface_name}: 確定{sample_races}レースのためフラット判定"
                }
            else:
                total_top3 = sample_races * 3
                in_pct = round((inside_count / total_top3) * 100.0, 1)
                out_pct = round((outside_count / total_top3) * 100.0, 1)
                front_pct = round((front_wins / sample_races) * 100.0, 1)

                waku_bias = "inside" if in_pct >= 50.0 else ("outside" if out_pct >= 50.0 else "flat")
                style_bias = "front" if front_pct >= 60.0 else ("closer" if front_pct <= 25.0 else "flat")

                if waku_bias == "inside" and style_bias == "front":
                    sum_txt = "🔥内枠先行超有利 (前残りバイアス)"
                elif waku_bias == "inside" and style_bias == "closer":
                    sum_txt = "🟢内枠有利 (ロスなし差し)"
                elif waku_bias == "inside":
                    sum_txt = "🟢内枠有利 (経済コース優位)"
                elif waku_bias == "outside" and style_bias == "closer":
                    sum_txt = "⚡外枠外差し有利 (上がり最速優位)"
                elif waku_bias == "outside" and style_bias == "front":
                    sum_txt = "🔵外枠有利 (スムーズ先行)"
                elif waku_bias == "outside":
                    sum_txt = "🔵外枠有利 (馬場外伸び)"
                elif style_bias == "front":
                    sum_txt = "🏃先行逃げ有利 (前残り馬場)"
                elif style_bias == "closer":
                    sum_txt = "🌪️差し追込有利 (直線外伸び)"
                else:
                    sum_txt = "⚖️内外・脚質フラット"

                detail = f"本日{venue_name}{surface_name}{sample_races}R確定: 内枠(1-3枠) 複勝率{in_pct:.0f}% / 外枠 {out_pct:.0f}% / 先行逃げ {front_wins}勝"
                bias_info = {
                    "venue_name": venue_name,
                    "surface_name": surface_name,
                    "sample_races": sample_races,
                    "bias_summary": sum_txt,
                    "waku_bias": waku_bias,
                    "running_style_bias": style_bias,
                    "inside_top3_ratio": in_pct,
                    "outside_top3_ratio": out_pct,
                    "detail_text": detail
                }

            _track_bias_cache[cache_key] = (now_ts, bias_info)
        except Exception as be:
            print(f"[TrackBias Warning] 集計エラー: {be}")

    # 各出走馬へのバイアスマッチング (bias_match)
    if bias_info and horses:
        waku_b = bias_info.get("waku_bias", "flat")
        style_b = bias_info.get("running_style_bias", "flat")
        h_stats = stats_dict.get("horse_stats", {})

        for h in horses:
            w = int(h.get("wakuban") or 0)
            h_clean = re.sub(r"[\s\u3000]", "", str(h.get("horse_name") or ""))
            st = h_stats.get(h_clean, {})
            is_front = float(st.get("last_corner_final", 8.0)) <= 4.0

            match_tag = None
            if waku_b == "inside" and w in (1, 2, 3):
                if is_front and style_b == "front":
                    match_tag = "🎯絶好バイアス (内先行)"
                else:
                    match_tag = "🟢内枠有利"
            elif waku_b == "inside" and w in (7, 8):
                match_tag = "⚠️外枠逆風"
            elif waku_b == "outside" and w in (6, 7, 8):
                if not is_front and style_b == "closer":
                    match_tag = "🎯絶好バイアス (外差し)"
                else:
                    match_tag = "🔵外枠有利"
            elif waku_b == "outside" and w in (1, 2):
                match_tag = "⚠️内枠逆風"
            elif style_b == "front" and is_front:
                match_tag = "🏃前残り合致"
            elif style_b == "closer" and not is_front:
                match_tag = "🌪️差し合致"

            h["bias_match"] = match_tag

    return bias_info


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


def predict_race_core(
    race_id: str,
    force_refresh: bool = False,
    is_auto_final: bool = False,
    odds_stage: Optional[str] = None
) -> dict:
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    cached_pred = get_cached_prediction(race_id)

    # ★ 発走後判定（レース終了後にオッズ変動で再推論が走り本命馬が事後的にすり替わるのを完全防止） ★
    is_post_race = False
    if cached_pred:
        r_d_str = resolve_race_date_for_id(race_id) or today_jst_str()
        if r_d_str < today_jst_str():
            is_post_race = True
        elif r_d_str == today_jst_str():
            p_str = (cached_pred.get("race_details") or {}).get("post_time", "")
            m_t = re.search(r"(\d{1,2}):(\d{2})", p_str)
            if m_t:
                d_parts = [int(p) for p in r_d_str.split("-")]
                est_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], int(m_t.group(1)), int(m_t.group(2)))
                if now_jst_naive() >= est_dt:
                    is_post_race = True

    if cached_pred and (not force_refresh or is_post_race):
        # scoreキーの欠損を補完（FastAPIのレスポンスバリデーション安全化）
        for h in cached_pred.get("horses", []):
            if "score" not in h or h["score"] is None:
                h["score"] = round(float(h.get("prob_win") or 0.0) * 100.0, 1)

        # ★ 実際の最新出走時刻の反映・更新（固定マップからの是正および発走時刻変更対応） ★
        latest_pt = _latest_post_times.get(race_id)
        sched_post_dt_raw = None
        sched_race_date = None
        if supabase:
            try:
                s_res = supabase.table("keiba_scheduled_races").select("race_date, post_datetime").eq("race_id", race_id).execute()
                if s_res.data and len(s_res.data) > 0:
                    sched_race_date = s_res.data[0].get("race_date")
                    sched_post_dt_raw = s_res.data[0].get("post_datetime")
                    if not latest_pt and sched_post_dt_raw:
                        m_pt = re.search(r"(\d{1,2}):(\d{2})", sched_post_dt_raw)
                        if m_pt:
                            latest_pt = f"{int(m_pt.group(1)):02d}:{int(m_pt.group(2)):02d}発走"
                            _latest_post_times[race_id] = latest_pt
            except Exception:
                pass

        if latest_pt:
            r_details = cached_pred.get("race_details") or {}
            cur_pt = r_details.get("post_time", "")
            if cur_pt != latest_pt:
                r_details["post_time"] = latest_pt
                cached_pred["race_details"] = r_details
                if supabase:
                    try:
                        supabase.table("keiba_predictions_cache").update({"data_json": cached_pred}).eq("race_id", race_id).execute()
                    except Exception as ue:
                        print(f"予想キャッシュ出走時刻更新エラー ({race_id}): {ue}")

        # ★ 【Step 3 & Step 4】マルチステージ自動リフレッシュ（60m馬体重 -> 30m -> 15m -> 05m最終確定）＆過去レース自己修復 ★
        has_cached_weights = any(h.get("horse_weight") is not None for h in cached_pred.get("horses", []))
        has_cached_sex_age = any(bool(h.get("sex_age")) for h in cached_pred.get("horses", []))
        cached_stage = str(cached_pred.get("last_odds_stage") or "0830")
        cached_stage_rank = ODDS_STAGE_RANK.get(cached_stage, 1)

        should_auto_refresh = False
        now_ts = time.time()

        if sched_race_date == today_jst_str() and sched_post_dt_raw:
            try:
                post_dt_obj = parse_post_datetime(sched_post_dt_raw, sched_race_date)
                if post_dt_obj:
                    now_dt_obj = now_jst_naive()
                    mins_to_post = (post_dt_obj - now_dt_obj).total_seconds() / 60.0
                    if 0.0 < mins_to_post <= 65.0:
                        target_stage = odds_stage or infer_odds_stage_from_minutes(mins_to_post)
                        target_rank = ODDS_STAGE_RANK.get(target_stage, 1)
                        cooldown_sec = 60.0 if target_stage == "05m" else 180.0
                        # 新しいステージ（60m/30m/15m/05m）に到達したか、馬体重がまだ未取得の場合に自動再推論
                        if target_rank > cached_stage_rank or (not has_cached_weights and now_ts - _last_weight_check_ts.get(race_id, 0.0) >= cooldown_sec):
                            _last_weight_check_ts[race_id] = now_ts
                            should_auto_refresh = True
                            odds_stage = target_stage
                            if target_stage == "05m":
                                is_auto_final = True
                            print(f"[Auto-StageRefresh] {race_id}: ステージ {cached_stage} -> {target_stage} (残り{mins_to_post:.1f}分) の最新オッズ・馬体重推論を実行します。")
            except Exception:
                pass

        # 発走済み・過去レースで馬体重や性齢が欠損している場合は、確定済みの買い目を変えずにメタデータのみ自己修復
        if not should_auto_refresh and (not has_cached_weights or not has_cached_sex_age):
            if now_ts - _last_weight_check_ts.get(race_id, 0.0) >= 30.0:
                is_past_or_post = False
                if sched_race_date and sched_race_date < today_jst_str():
                    is_past_or_post = True
                elif sched_race_date == today_jst_str() and sched_post_dt_raw:
                    post_dt_obj = parse_post_datetime(sched_post_dt_raw, sched_race_date)
                    if post_dt_obj and now_jst_naive() >= post_dt_obj:
                        is_past_or_post = True
                elif not sched_race_date:
                    is_past_or_post = True

                if is_past_or_post:
                    _last_weight_check_ts[race_id] = now_ts
                    try:
                        _, _, fresh_horses, _ = fetch_shutuba_table(race_id, race_date=sched_race_date)
                        fh_map = {int(fh["umaban"]): fh for fh in fresh_horses if "umaban" in fh}
                        patched = False
                        for ch in cached_pred.get("horses", []):
                            u_num = int(ch.get("umaban", 0))
                            fh = fh_map.get(u_num)
                            if fh:
                                if ch.get("horse_weight") is None and fh.get("horse_weight") is not None:
                                    ch["horse_weight"] = fh["horse_weight"]
                                    ch["weight_diff"] = fh.get("weight_diff", 0)
                                    patched = True
                                if not ch.get("sex_age") and fh.get("sex_age"):
                                    ch["sex_age"] = fh["sex_age"]
                                    patched = True
                        if patched and supabase:
                            cached_pred["last_odds_stage"] = cached_pred.get("last_odds_stage") if cached_pred.get("last_odds_stage") != "0830" else "60m"
                            supabase.table("keiba_predictions_cache").update({"data_json": cached_pred}).eq("race_id", race_id).execute()
                            if sched_race_date:
                                _all_predictions_cache.pop(sched_race_date, None)
                                _today_races_cache.pop(sched_race_date, None)
                            print(f"[Auto-Weight Heal] {race_id}: 終了済レースの馬体重・性齢キャッシュを自動補完しました。")
                    except Exception as he:
                        print(f"[Auto-Weight Heal Warning] {race_id}: {he}")

        if not should_auto_refresh:
            return sync_virtual_badge_to_prediction(race_id, cached_pred, sched_race_date)

    try:
        target_race_date = None
        scheduled_post_dt = None
        if supabase:
            try:
                s_res = supabase.table("keiba_scheduled_races").select("race_date, post_datetime").eq("race_id", race_id).execute()
                if s_res.data and len(s_res.data) > 0:
                    target_race_date = s_res.data[0].get("race_date")
                    p_dt_str = s_res.data[0].get("post_datetime")
                    if p_dt_str:
                        scheduled_post_dt = parse_post_datetime(p_dt_str, target_race_date)
            except Exception:
                pass

        # 手動での予想し直し（force_refresh=True かつ 既存予想がある場合）は出走時刻前のみ許可
        if force_refresh and not is_auto_final and cached_pred is not None:
            now_dt = now_jst_naive()
            if scheduled_post_dt and now_dt >= scheduled_post_dt:
                raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")
            if not scheduled_post_dt and target_race_date:
                p_str = (cached_pred.get("race_details") or {}).get("post_time", "")
                est_dt = parse_post_datetime(p_str, target_race_date)
                if est_dt and now_dt >= est_dt:
                    raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")

        if not target_race_date:
            # キャッシュ中の開催期間から日付を逆引き
            for d_cached, c_entry in _period_venues_cache.items():
                v_list = c_entry.get("venues", []) if isinstance(c_entry, dict) else (c_entry if isinstance(c_entry, list) else [])
                for v in v_list:
                    if isinstance(v, dict) and any(r.get("race_id") == race_id for r in v.get("races", [])):
                        target_race_date = d_cached
                        break
                if target_race_date:
                    break
        known_schedule_date = target_race_date
        if not target_race_date:
            target_race_date = resolve_race_date_for_id(race_id) or today_jst_str()

        race_name, race_details, horses, course_meta = fetch_shutuba_table(race_id, race_date=known_schedule_date)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした。")

        # 発走時刻から現在のオッズ取得ステージを自動推定（未指定時）
        if not odds_stage:
            if is_auto_final:
                odds_stage = "05m"
            elif target_race_date == today_jst_str():
                est_post_dt = scheduled_post_dt
                if not est_post_dt:
                    p_str = (race_details or {}).get("post_time", "")
                    est_post_dt = parse_post_datetime(p_str, target_race_date)
                if est_post_dt:
                    mins_left = (est_post_dt - now_jst_naive()).total_seconds() / 60.0
                    odds_stage = infer_odds_stage_from_minutes(mins_left)
                else:
                    odds_stage = "0830"
            else:
                odds_stage = "0830"

        # 手動再予想時に出走表側の発走時刻でも発走後と判明した場合はブロック
        if force_refresh and not is_auto_final and cached_pred is not None:
            p_str = (race_details or {}).get("post_time", "")
            est_dt = parse_post_datetime(p_str, target_race_date)
            if est_dt and now_jst_naive() >= est_dt:
                raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")

        if course_meta is None:
            course_meta = {}
        course_meta["race_date"] = target_race_date
        course_meta["odds_stage"] = odds_stage
        prev_horses_list = cached_pred.get("horses", []) if isinstance(cached_pred, dict) else None
        course_meta["prev_horses"] = prev_horses_list
        if is_auto_final or odds_stage == "05m":
            course_meta["force_final_odds"] = True

        top_rid = get_or_select_daily_top_race_id(target_race_date)
        prev_was_daily_top = bool(
            isinstance(cached_pred, dict)
            and (cached_pred.get("hybrid_recommendation") or {}).get("strategy_type") == "daily_top"
            and (cached_pred.get("hybrid_recommendation") or {}).get("is_real_bet_target")
        )
        course_meta["is_daily_top_race"] = bool((top_rid and race_id == top_rid) or (prev_was_daily_top and top_rid is not None))

        result = build_prediction_and_recs(race_name, race_details, race_id, horses, course_meta)
        apply_odds_anomaly_indicators(result["horses"], prev_horses=prev_horses_list, odds_stage=odds_stage)

        resp_data = {
            "race_id": race_id,
            "race_name": result["race_name"],
            "race_details": result["race_details"],
            "horses": result["horses"],
            "recommendation": result["recommendation"],
            "hybrid_recommendation": result.get("hybrid_recommendation"),
            "track_bias": result.get("track_bias"),
            "last_odds_stage": odds_stage
        }

        save_cached_prediction(race_id, target_race_date, resp_data)

        # 自然合致レースの増減に応じた当日最低1R実馬券保証（daily_top昇格・解除）を即時同期
        try:
            promoted_rid = ensure_daily_single_real_bet(target_race_date)
            if promoted_rid == race_id and not (resp_data.get("hybrid_recommendation") or {}).get("is_real_bet_target"):
                reloaded = get_cached_prediction(race_id)
                if reloaded:
                    resp_data = reloaded
        except Exception as de:
            print(f"[predict_race_core] daily_top同期警告 ({race_id}): {de}")

        if supabase:
            try:
                has_weights_now = any(h.get("horse_weight") is not None for h in result["horses"])
                if is_auto_final or odds_stage == "05m":
                    new_pred_flag = 5
                elif odds_stage == "15m":
                    new_pred_flag = 4
                elif odds_stage == "30m":
                    new_pred_flag = 3
                elif odds_stage == "60m":
                    new_pred_flag = 2 if has_weights_now else 1
                else:
                    new_pred_flag = 1
                supabase.table("keiba_scheduled_races").update({"is_predicted": new_pred_flag}).eq("race_id", race_id).lt("is_predicted", new_pred_flag).execute()
            except Exception:
                pass

        return sync_virtual_badge_to_prediction(race_id, resp_data, target_race_date)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


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



def extract_top3_orders(pred_data: Optional[dict], orders: Optional[dict]) -> List[dict]:
    """【確定】 1着・2着・3着の確定馬番・馬名・印情報を抽出 (キーがstr/int両対応)"""
    if not orders:
        return []
    orders_int = {}
    for k, v in orders.items():
        try:
            orders_int[int(k)] = int(v)
        except (ValueError, TypeError):
            pass

    horses = pred_data.get("horses", []) if pred_data else []
    horse_by_num = {int(h["umaban"]): h for h in horses if "umaban" in h and str(h["umaban"]).isdigit()}

    top3 = []
    for rank in [1, 2, 3]:
        for u, o in orders_int.items():
            if o == rank:
                h = horse_by_num.get(u, {})
                mark = h.get("mark", "-")
                if mark in ("〇", "○"):
                    mark = "◯"
                top3.append({
                    "order": rank,
                    "order_text": f"{rank}着",
                    "umaban": u,
                    "horse_name": h.get("horse_name", f"馬{u}"),
                    "mark": mark
                })
    top3.sort(key=lambda x: x["order"])
    return top3[:3]


def extract_mark_orders(pred_data: Optional[dict], orders: Optional[dict]) -> List[dict]:
    """◎, ◯, ▲ の各馬の確定着順情報を抽出 (キーがstr/int両対応)"""
    if not pred_data:
        return []
    horses = pred_data.get("horses", [])
    orders_int = {}
    for k, v in (orders or {}).items():
        try:
            orders_int[int(k)] = int(v)
        except (ValueError, TypeError):
            pass

    target_marks = ["◎", "◯", "▲"]
    found = []

    for mark in target_marks:
        h = next((x for x in horses if x.get("mark") == mark or (mark == "◯" and x.get("mark") in ("〇", "○"))), None)
        if h:
            u = int(h["umaban"])
            name = h.get("horse_name", f"馬{u}")
            ord_num = orders_int.get(u)
            if ord_num is not None and isinstance(ord_num, int) and 1 <= ord_num < 90:
                ord_text = f"{ord_num}着"
            else:
                ord_text = "着外" if orders_int else "未確定"
            found.append({
                "mark": mark,
                "umaban": u,
                "horse_name": name,
                "order": ord_num if (ord_num and ord_num < 90) else None,
                "order_text": ord_text,
                "popularity": h.get("popularity")
            })
    return found


def format_mark_orders_text(mark_orders: List[dict]) -> str:
    """印の着順をテキストに整形"""
    if not mark_orders:
        return ""
    parts = []
    for mo in mark_orders:
        parts.append(f"{mo['mark']}{mo['umaban']}番 {mo['horse_name']}({mo['order_text']})")
    return "【確定着順】 " + " / ".join(parts)


def sync_orders_marks(top3_orders: Optional[List[dict]], mark_orders: Optional[List[dict]], pred_data: Optional[dict]) -> Tuple[List[dict], List[dict]]:
    """確定着順(top3_orders)および印馬着順(mark_orders)の印・馬名を最新の予想データ(pred_data)の印と完全同期"""
    if not pred_data:
        return top3_orders or [], mark_orders or []
    horses = pred_data.get("horses", [])
    if not horses:
        return top3_orders or [], mark_orders or []

    h_map = {}
    for h in horses:
        if "umaban" in h:
            try:
                h_map[int(h["umaban"])] = h
            except (ValueError, TypeError):
                pass

    synced_top3 = []
    for t in (top3_orders or []):
        t_copy = dict(t)
        try:
            u = int(t_copy.get("umaban", 0))
            if u in h_map:
                m = h_map[u].get("mark", "-")
                if m in ("〇", "○"):
                    m = "◯"
                t_copy["mark"] = m
                if h_map[u].get("horse_name"):
                    t_copy["horse_name"] = h_map[u]["horse_name"]
        except Exception:
            pass
        synced_top3.append(t_copy)

    order_info_by_umaban = {}
    for mo in (mark_orders or []):
        try:
            u = int(mo.get("umaban", 0))
            if u:
                order_info_by_umaban[u] = mo
        except Exception:
            pass

    synced_marks = []
    target_marks = ["◎", "◯", "▲"]
    for mark in target_marks:
        h = next((x for x in horses if x.get("mark") == mark or (mark == "◯" and x.get("mark") in ("〇", "○"))), None)
        if h:
            try:
                u = int(h["umaban"])
                name = h.get("horse_name", f"馬{u}")
                prev_info = order_info_by_umaban.get(u, {})
                synced_marks.append({
                    "mark": mark,
                    "umaban": u,
                    "horse_name": name,
                    "order": prev_info.get("order"),
                    "order_text": prev_info.get("order_text", "未確定"),
                    "popularity": h.get("popularity", prev_info.get("popularity"))
                })
            except Exception:
                pass

    if not synced_marks and mark_orders:
        synced_marks = mark_orders

    return synced_top3, synced_marks



def calculate_hybrid_pass_benefit_info(
    h_rec: Optional[dict],
    race_res: Optional[dict],
    target_u: Optional[int],
    actual_order: Optional[int],
    pred_data: Optional[dict] = None
) -> dict:
    """見送りレースにおける仮想損益 (+200円 - 的中時払戻) の算出"""
    hypo_bet = 200
    hypo_pay = 0
    payouts_data = race_res.get("payouts", {}) if race_res else {}
    tan_map = payouts_data.get("tansho", {})
    fuku_map = payouts_data.get("fukusho", {})

    # 本命馬の単複(単100+複100)を仮想買い目として検証
    # pred_dataが渡されている場合は、予想カードに表示される本命馬(◎)の馬番と完全同期
    if pred_data:
        p_rec = pred_data.get("recommendation") or {}
        p_target = p_rec.get("target_umaban")
        if p_target is not None:
            target_u = int(p_target)
        else:
            # recommendation にない場合は horses の mark='◎' をフォールバック探索
            for h in pred_data.get("horses", []):
                if h.get("mark") == "◎":
                    target_u = int(h.get("umaban"))
                    break

    hit_details = []
    if target_u is not None:
        unit_tan = tan_map.get(target_u, 0) or tan_map.get(str(target_u), 0)
        unit_fuku = fuku_map.get(target_u, 0) or fuku_map.get(str(target_u), 0)
        pay_tan = int(1 * unit_tan)
        pay_fuku = int(1 * unit_fuku)
        hypo_pay = pay_tan + pay_fuku
        if pay_tan > 0:
            hit_details.append(f"単勝{target_u}({pay_tan:,}円)")
        if pay_fuku > 0:
            hit_details.append(f"複勝{target_u}({pay_fuku:,}円)")

    pass_benefit = hypo_bet - hypo_pay

    orders = race_res.get("orders", {}) if race_res else {}
    mark_orders = extract_mark_orders(pred_data, orders)
    top3_orders = extract_top3_orders(pred_data, orders)
    marks_summary = format_mark_orders_text(mark_orders)

    raw_reason = h_rec.get("reason", "期待値基準外") if h_rec else "見送り推奨"
    if "条件不一致" in raw_reason:
        reason_str = "条件不一致 (資金保全のため見送り)"
    else:
        reason_str = raw_reason
    hit_summary = f" ({' ＆ '.join(hit_details)} 的中)" if hit_details else " (不的中・払戻0円)"
    benefit_note = "資金保全成功！" if pass_benefit > 0 else (f"機会損失 (-{abs(pass_benefit):,}円)" if pass_benefit < 0 else "損益なし")
    marks_line = f"\n{marks_summary}" if marks_summary else ""

    msg = f"見送り推奨レースです。(判定: {reason_str})\n【仮想損益】仮想投資: {hypo_bet:,}円 / 仮想払戻: {hypo_pay:,}円{hit_summary} → 見送り損益: {pass_benefit:+,}円 ({benefit_note}){marks_line}"

    return {
        "hypothetical_bet": hypo_bet,
        "hypothetical_payout": hypo_pay,
        "pass_benefit": pass_benefit,
        "pay_tan": pay_tan if target_u is not None else 0,
        "pay_fuku": pay_fuku if target_u is not None else 0,
        "actual_order": actual_order,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "message": msg
    }


def get_cached_settlement(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_settlements").select("*").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            d = res.data[0]
            profit = d["total_payout"] - d["total_bet"]
            recovery = round((d["total_payout"] / d["total_bet"]) * 100.0, 2) if d["total_bet"] > 0 else 0.0
            order = d.get("actual_order")

            hypo_bet = None
            hypo_pay = None
            pass_benefit = None
            mark_orders = None

            hit_type = d.get("hit_type")
            u_bet = d.get("umaren_bet") or 0
            t_bet = d.get("trio_bet") or 0
            u_pay = d.get("umaren_payout") or 0
            t_pay = d.get("trio_payout") or 0

            res_cache = get_cached_result(race_id) or {}
            pred_cache = get_cached_prediction(race_id)
            orders = res_cache.get("orders", {})
            b_det = d.get("bet_details") if isinstance(d.get("bet_details"), dict) else {}
            rb_meta = b_det.get("real_bet_meta") if isinstance(b_det.get("real_bet_meta"), dict) else {}

            # ★ 1. 本命馬番の不整合チェック＆完全自己修復 ★
            # 予想データ(pred_cache)の本命馬番とDBのtarget_umabanが不一致の場合、
            # オッズ変動等で後から本命が昇格・変更されたため、確定結果を最新予想データで強制再計算＆DB自動更新
            if pred_cache and res_cache:
                p_rec = pred_cache.get("recommendation") or {}
                expected_target_u = p_rec.get("target_umaban")
                if expected_target_u is None:
                    for h in pred_cache.get("horses", []):
                        if h.get("mark") == "◎":
                            expected_target_u = int(h.get("umaban"))
                            break

                if expected_target_u is not None and d.get("target_umaban") != expected_target_u:
                    print(f"[Settlement Target Mismatch Heal] {race_id}: DB target={d.get('target_umaban')} vs Pred target={expected_target_u}. Recalculating settlement...")
                    r_date_val = d.get("race_date") or resolve_race_date_for_id(race_id)
                    r_name_val = d.get("race_name") or f"Race {race_id}"
                    return process_settlement_calculation(
                        race_id=race_id,
                        race_date=r_date_val,
                        race_name=r_name_val,
                        pred_data=pred_cache,
                        race_res=res_cache,
                        from_cache=True
                    )

            # ★ 2. 確定着順(top3_orders)および印馬着順(mark_orders)の最新化＆印同期 ★
            if pred_cache and orders:
                top3_orders = extract_top3_orders(pred_cache, orders)
                mark_orders = extract_mark_orders(pred_cache, orders)
            else:
                top3_orders = d.get("top3_orders") or rb_meta.get("top3_orders")
                mark_orders = d.get("mark_orders") or rb_meta.get("mark_orders")
                if pred_cache:
                    top3_orders, mark_orders = sync_orders_marks(top3_orders, mark_orders, pred_cache)

            # 1〜3着(top3_orders)が未キャッシュの場合は自動取得＆永続化
            if not top3_orders and d.get("hit_type") != "中止":
                try:
                    fresh_res = fetch_netkeiba_race_result(race_id)
                    if fresh_res and fresh_res.get("orders"):
                        res_cache = fresh_res
                        orders = fresh_res.get("orders", {})
                        r_date_val = d.get("race_date") or resolve_race_date_for_id(race_id)
                        r_name_val = d.get("race_name") or fresh_res.get("race_name", f"Race {race_id}")
                        save_cached_result(race_id, r_name_val, r_date_val, fresh_res)
                        top3_orders = extract_top3_orders(pred_cache, orders)
                        healed_marks = extract_mark_orders(pred_cache, orders)
                        if top3_orders:
                            rb_meta["top3_orders"] = top3_orders
                            if healed_marks:
                                rb_meta["mark_orders"] = healed_marks
                            b_det["real_bet_meta"] = rb_meta
                            d["bet_details"] = b_det
                            supabase.table("keiba_race_settlements").update({"bet_details": b_det, "top3_orders": top3_orders}).eq("race_id", race_id).execute()
                except Exception as te:
                    print(f"[Settlement Top3 Heal Warning] {race_id}: {te}")

            # DBのtop3_ordersが未設定または最新と異なる場合はDBに永続化同期
            if top3_orders and top3_orders != d.get("top3_orders"):
                try:
                    if supabase:
                        supabase.table("keiba_race_settlements").update({"top3_orders": top3_orders}).eq("race_id", race_id).execute()
                        d["top3_orders"] = top3_orders
                except Exception as t3e:
                    print(f"[Settlement Top3 Sync Warning] {race_id}: {t3e}")

            is_virtual_or_pass = bool(
                d.get("is_pass")
                or d.get("strategy_type") == "virtual"
                or (d.get("hit_type") and "仮想" in str(d.get("hit_type")))
            )

            if is_virtual_or_pass:
                if pred_cache and res_cache:
                    h_rec = pred_cache.get("hybrid_recommendation") if pred_cache else None
                    expected_target_u = (pred_cache.get("recommendation") or {}).get("target_umaban")
                    expected_h_name = (pred_cache.get("recommendation") or {}).get("horse_name")
                    cur_target_u = expected_target_u if expected_target_u is not None else d.get("target_umaban")

                    orders_map = res_cache.get("orders", {})
                    cur_order = orders_map.get(cur_target_u) if cur_target_u is not None else order
                    if cur_order is None and cur_target_u is not None:
                        cur_order = orders_map.get(str(cur_target_u))

                    pass_info = calculate_hybrid_pass_benefit_info(h_rec, res_cache, cur_target_u, cur_order, pred_data=pred_cache)
                    hypo_bet = pass_info["hypothetical_bet"]
                    hypo_pay = pass_info["hypothetical_payout"]
                    pass_benefit = pass_info["pass_benefit"]
                    msg = pass_info["message"]
                    mark_orders = pass_info.get("mark_orders") or mark_orders
                    top3_orders = pass_info.get("top3_orders") or top3_orders

                    is_real = bool(rb_meta.get("is_real_bet_target") or rb_meta.get("real_bet_level") in ["high", "standard"])
                    is_virt = bool(
                        (h_rec or {}).get("is_virtual_bet_target")
                        or rb_meta.get("is_virtual_bet_target")
                        or rb_meta.get("real_bet_level") == "virtual"
                        or (rb_meta.get("real_bet_badge") and "仮想" in str(rb_meta.get("real_bet_badge")))
                        or d.get("strategy_type") == "virtual"
                        or (d.get("hit_type") and "仮想" in str(d.get("hit_type")))
                    ) and not is_real

                    if is_virt:
                        pay_tan = pass_info.get("pay_tan", 0)
                        pay_fuku = pass_info.get("pay_fuku", 0)
                        v_profit = hypo_pay - hypo_bet
                        v_rec = round((hypo_pay / hypo_bet) * 100.0, 1) if hypo_bet > 0 else 0.0
                        is_virt_hit = (hypo_pay > 0)
                        hit_type = "仮想的中" if is_virt_hit else "仮想不的中"
                        marks_summary = format_mark_orders_text(mark_orders)
                        marks_line = f"\n{marks_summary}" if marks_summary else ""
                        if is_virt_hit:
                            v_hits = []
                            if pay_tan > 0: v_hits.append(f"単勝{cur_target_u}({pay_tan:,}円)")
                            if pay_fuku > 0: v_hits.append(f"複勝{cur_target_u}({pay_fuku:,}円)")
                            msg = f"【仮想的中】 {' ＆ '.join(v_hits)}！ 仮想払戻: {hypo_pay:,}円 (収支: {v_profit:+,}円 / 回収率: {v_rec}%){marks_line}"
                        else:
                            ord_info = f" (◎{cur_target_u}番: {cur_order}着)" if (cur_order and cur_order < 90) else ""
                            msg = f"【仮想不的中】{ord_info} (仮想投資: {hypo_bet:,}円 / 払戻: 0円 / 収支: {v_profit:,}円 / 回収率: 0.0%){marks_line}"

                        d["is_pass"] = False
                        d["target_umaban"] = cur_target_u
                        d["horse_name"] = expected_h_name or d.get("horse_name")
                        d["actual_order"] = cur_order if (cur_order and cur_order < 90) else None
                        d["total_bet"] = hypo_bet
                        d["total_payout"] = hypo_pay
                        d["profit"] = v_profit
                        d["recovery_rate"] = v_rec
                        profit = v_profit
                        recovery = v_rec
                        d["is_hit"] = is_virt_hit
                        d["hit_type"] = hit_type
                        d["strategy_type"] = "virtual"
                        d["tansho_bet"] = 100
                        d["fukusho_bet"] = 100
                        d["tansho_payout"] = pay_tan
                        d["fukusho_payout"] = pay_fuku
                        d["bet_breakdown"] = {"単勝": 100, "複勝": 100}
                        p_bd = {}
                        if pay_tan > 0: p_bd["単勝"] = pay_tan
                        if pay_fuku > 0: p_bd["複勝"] = pay_fuku
                        d["payout_breakdown"] = p_bd

                        try:
                            if supabase:
                                supabase.table("keiba_race_settlements").update({
                                    "is_pass": False,
                                    "target_umaban": cur_target_u,
                                    "horse_name": expected_h_name or d.get("horse_name"),
                                    "actual_order": cur_order if (cur_order and cur_order < 90) else None,
                                    "total_bet": hypo_bet,
                                    "total_payout": hypo_pay,
                                    "is_hit": is_virt_hit,
                                    "hit_type": hit_type,
                                    "strategy_type": "virtual",
                                    "tansho_bet": 100,
                                    "fukusho_bet": 100,
                                    "tansho_payout": pay_tan,
                                    "fukusho_payout": pay_fuku,
                                    "bet_breakdown": {"単勝": 100, "複勝": 100},
                                    "payout_breakdown": p_bd,
                                    "top3_orders": top3_orders,
                                    "message": msg
                                }).eq("race_id", race_id).execute()
                        except Exception as e_vh:
                            print(f"[VirtHeal Error] {race_id}: {e_vh}")
                    elif cur_target_u is not None and (d.get("target_umaban") != cur_target_u or d.get("hypothetical_payout") != hypo_pay):
                        # ★ DB側の本命馬番や払戻金が予想データと不整合を起こしている場合、自動自己修復 ★
                        try:
                            heal_payload = {
                                "target_umaban": cur_target_u,
                                "horse_name": expected_h_name,
                                "actual_order": cur_order if (cur_order and cur_order < 90) else None,
                                "hypothetical_payout": hypo_pay,
                                "pass_benefit": pass_benefit,
                                "top3_orders": top3_orders,
                                "message": msg
                            }
                            if supabase:
                                supabase.table("keiba_race_settlements").update(heal_payload).eq("race_id", race_id).execute()
                            d.update(heal_payload)
                            order = cur_order
                            print(f"[Settlement Heal] {race_id}: 本命馬不整合を自己修復更新しました ({d.get('target_umaban')} -> {cur_target_u}番 {expected_h_name}, 払戻:{hypo_pay}円)")
                        except Exception as he_settle:
                            print(f"[Settlement Heal Warning] {race_id}: {he_settle}")
                else:
                    hypo_bet = d.get("hypothetical_bet") or 0
                    hypo_pay = d.get("hypothetical_payout") or 0
                    pass_benefit = d.get("pass_benefit") or 0
                    mark_orders = d.get("mark_orders") or rb_meta.get("mark_orders")
                    marks_summary = format_mark_orders_text(mark_orders)
                    marks_line = f"\n{marks_summary}" if marks_summary else ""
                    benefit_note = "資金保全成功！" if pass_benefit > 0 else (f"機会損失 (-{abs(pass_benefit):,}円)" if pass_benefit < 0 else "損益なし")
                    msg = f"見送り推奨レースです。\n【仮想損益】仮想投資: {hypo_bet:,}円 / 仮想払戻: {hypo_pay:,}円 → 見送り損益: {pass_benefit:+,}円 ({benefit_note}){marks_line}"
            elif d["is_hit"]:
                mark_orders = d.get("mark_orders") or extract_mark_orders(pred_cache, orders) or rb_meta.get("mark_orders")
                marks_summary = format_mark_orders_text(mark_orders)
                marks_line = f"\n{marks_summary}" if marks_summary else ""

                hit_label = hit_type if hit_type else "的中"
                p_details = d.get("payout_details")
                if isinstance(p_details, str):
                    try:
                        p_details = json.loads(p_details)
                    except Exception:
                        p_details = []
                if isinstance(p_details, list) and len(p_details) > 0:
                    hit_texts = [f"{h.get('type','')}{h.get('comb','')}({h.get('payout',0):,}円)" for h in p_details]
                    msg = f"【{hit_label}】 {' ＆ '.join(hit_texts)}！ 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery}%){marks_line}"
                else:
                    msg = f"【{hit_label}】 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery}%){marks_line}"
            else:
                mark_orders = d.get("mark_orders") or extract_mark_orders(pred_cache, orders) or rb_meta.get("mark_orders")
                marks_summary = format_mark_orders_text(mark_orders)
                marks_line = f"\n{marks_summary}" if marks_summary else ""
                order_info = f" (◎{d.get('target_umaban')}番: {order}着)" if (order and order < 90) else ""
                msg = f"不的中{order_info} (投資: {d['total_bet']:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%){marks_line}"


            b_breakdown = d.get("bet_breakdown")
            tan_bet = d.get("tansho_bet") or 0
            fuku_bet = d.get("fukusho_bet") or 0
            utan_bet = d.get("umatan_bet") or (b_breakdown.get("馬単", 0) if isinstance(b_breakdown, dict) else 0) or 0
            if not b_breakdown or not isinstance(b_breakdown, dict):
                b_breakdown = {}
                if tan_bet > 0:
                    b_breakdown["単勝"] = tan_bet
                if fuku_bet > 0:
                    b_breakdown["複勝"] = fuku_bet
                if utan_bet > 0:
                    b_breakdown["馬単"] = utan_bet
                if u_bet > 0:
                    b_breakdown["馬連"] = u_bet
                if t_bet > 0:
                    b_breakdown["三連複"] = t_bet

            p_breakdown = d.get("payout_breakdown")
            tan_pay = d.get("tansho_payout") or 0
            fuku_pay = d.get("fukusho_payout") or 0
            utan_pay = d.get("umatan_payout") or (p_breakdown.get("馬単", 0) if isinstance(p_breakdown, dict) else 0) or 0
            if not p_breakdown or not isinstance(p_breakdown, dict):
                p_breakdown = {}
                if tan_pay > 0:
                    p_breakdown["単勝"] = tan_pay
                if fuku_pay > 0:
                    p_breakdown["複勝"] = fuku_pay
                if utan_pay > 0:
                    p_breakdown["馬単"] = utan_pay
                if u_pay > 0:
                    p_breakdown["馬連"] = u_pay
                if t_pay > 0:
                    p_breakdown["三連複"] = t_pay

            return {
                "race_id": d["race_id"],
                "race_name": d["race_name"],
                "is_pass": bool(d["is_pass"]),
                "grade": d["grade"],
                "target_umaban": d["target_umaban"],
                "horse_name": d["horse_name"],
                "actual_order": order if (order and order < 90) else (3 if d["is_hit"] else None),
                "is_hit": bool(d["is_hit"]),
                "total_bet": d["total_bet"],
                "total_payout": d["total_payout"],
                "profit": profit,
                "recovery_rate": recovery,
                "from_cache": True,
                "message": msg,
                "hypothetical_bet": hypo_bet if hypo_bet is not None else d.get("hypothetical_bet"),
                "hypothetical_payout": hypo_pay if hypo_pay is not None else d.get("hypothetical_payout"),
                "pass_benefit": pass_benefit if pass_benefit is not None else d.get("pass_benefit"),
                "strategy_type": d.get("strategy_type", "hybrid"),
                "hit_type": hit_type,
                "bet_breakdown": b_breakdown,
                "payout_breakdown": p_breakdown,
                "bet_details": d.get("bet_details"),
                "payout_details": d.get("payout_details"),
                "umaren_bet": u_bet,
                "trio_bet": t_bet,
                "umaren_payout": u_pay,
                "trio_payout": t_pay,
                "tansho_bet": tan_bet,
                "fukusho_bet": fuku_bet,
                "umatan_bet": utan_bet,
                "tansho_payout": tan_pay,
                "fukusho_payout": fuku_pay,
                "umatan_payout": utan_pay,
                "target_popularity": rb_meta.get("target_popularity"),
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }
    except Exception as e:
        print(f"Supabase収支取得エラー ({race_id}): {e}")
    return None




def get_cached_result(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_results_cache").select("result_json").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            return normalize_race_res(res.data[0]["result_json"])
    except Exception as e:
        print(f"Supabase結果キャッシュ取得エラー ({race_id}): {e}")
    return None


def save_cached_result(race_id: str, race_name: str, race_date: str, result_dict: dict):
    if not supabase:
        return
    try:
        supabase.table("keiba_race_results_cache").upsert({
            "race_id": race_id,
            "race_name": race_name,
            "race_date": race_date,
            "result_json": result_dict
        }).execute()
    except Exception as e:
        print(f"Supabase結果キャッシュ保存エラー ({race_id}): {e}")


def save_settlement(data: dict):
    if not supabase:
        return
    try:
        row_dict = {
            "race_id": data["race_id"],
            "race_date": data["race_date"],
            "race_name": data["race_name"],
            "strategy_type": data.get("strategy_type", "hybrid"),
            "target_umaban": data.get("target_umaban"),
            "horse_name": data.get("horse_name"),
            "grade": data.get("grade"),
            "is_pass": 1 if data.get("is_pass") else 0,
            "umaren_bet": data.get("umaren_bet", 0),
            "trio_bet": data.get("trio_bet", 0),
            "total_bet": data.get("total_bet", 0),
            "actual_order": data.get("actual_order"),
            "umaren_payout": data.get("umaren_payout", 0),
            "trio_payout": data.get("trio_payout", 0),
            "total_payout": data.get("total_payout", 0),
            "is_hit": 1 if data.get("is_hit") else 0,
            "hit_type": data.get("hit_type"),
            "hypothetical_bet": data.get("hypothetical_bet", 0),
            "hypothetical_payout": data.get("hypothetical_payout", 0),
            "pass_benefit": data.get("pass_benefit", 0),
            "bet_breakdown": data.get("bet_breakdown", {}),
            "payout_breakdown": data.get("payout_breakdown", {}),
            "bet_details": data.get("bet_details"),
            "payout_details": data.get("payout_details"),
            "tansho_bet": data.get("tansho_bet", 0),
            "fukusho_bet": data.get("fukusho_bet", 0),
            "tansho_payout": data.get("tansho_payout", 0),
            "fukusho_payout": data.get("fukusho_payout", 0)
        }
        if "track_bias" in data and data["track_bias"] is not None:
            row_dict["track_bias"] = data["track_bias"]
        if "odds_alerts" in data and data["odds_alerts"] is not None:
            row_dict["odds_alerts"] = data["odds_alerts"]
        if "top3_orders" in data and data["top3_orders"] is not None:
            row_dict["top3_orders"] = data["top3_orders"]

        try:
            supabase.table("keiba_race_settlements").upsert(row_dict).execute()
        except Exception as col_err:
            # カラム（track_bias, odds_alerts, top3_orders）が未定義の場合の安全フォールバック（bet_details内に退避）
            if "track_bias" in row_dict or "odds_alerts" in row_dict or "top3_orders" in row_dict:
                b_det = row_dict.get("bet_details") or {}
                if "track_bias" in row_dict:
                    b_det["track_bias"] = row_dict.pop("track_bias")
                if "odds_alerts" in row_dict:
                    b_det["odds_alerts"] = row_dict.pop("odds_alerts")
                if "top3_orders" in row_dict:
                    b_det["top3_orders"] = row_dict.pop("top3_orders")
                row_dict["bet_details"] = b_det
                supabase.table("keiba_race_settlements").upsert(row_dict).execute()
            else:
                raise col_err
        invalidate_summary_caches()
    except Exception as e:
        print(f"Supabase収支台帳保存エラー: {e}")




def process_settlement_calculation(
    race_id: str,
    race_date: str,
    race_name: str,
    pred_data: dict,
    race_res: dict,
    from_cache: bool = False
) -> dict:
    t_bias = pred_data.get("track_bias") if isinstance(pred_data, dict) else None
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
        for h in (pred_data.get("horses", []) if isinstance(pred_data, dict) else [])
        if h.get("odds_alert") or h.get("odds_trend") or h.get("bias_match") or h.get("odds_history")
    ]

    # 開催中止・取りやめレースの全額返還処理 (投資0円・払戻0円・見送り扱い)
    if race_res.get("is_cancelled"):
        settlement = {
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "strategy_type": "cancelled",
            "target_umaban": None,
            "horse_name": None,
            "grade": "C",
            "is_pass": True,
            "umaren_bet": 0,
            "trio_bet": 0,
            "total_bet": 0,
            "actual_order": None,
            "umaren_payout": 0,
            "trio_payout": 0,
            "total_payout": 0,
            "is_hit": False,
            "hit_type": "開催中止",
            "hypothetical_bet": 0,
            "hypothetical_payout": 0,
            "pass_benefit": 0,
            "bet_breakdown": {},
            "payout_breakdown": {},
            "bet_details": {
                "real_bet_meta": {
                    "is_real_bet_target": False,
                    "real_bet_badge": "⛔【開催中止】全額返還",
                    "real_bet_level": "none",
                    "mark_orders": [],
                    "top3_orders": []
                }
            },
            "payout_details": None,
            "tansho_bet": 0,
            "fukusho_bet": 0,
            "tansho_payout": 0,
            "fukusho_payout": 0,
            "mark_orders": [],
            "top3_orders": [],
            "track_bias": t_bias,
            "odds_alerts": o_alerts
        }
        save_settlement(settlement)
        if supabase:
            try:
                supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
            except Exception:
                pass

        return {
            "race_id": race_id,
            "race_name": race_name,
            "is_pass": True,
            "grade": "C",
            "target_umaban": None,
            "horse_name": None,
            "actual_order": None,
            "is_hit": False,
            "total_bet": 0,
            "total_payout": 0,
            "profit": 0,
            "recovery_rate": 0.0,
            "from_cache": from_cache,
            "message": "【開催中止】積雪・荒天等のためレース取りやめ（馬券全額返還 / 投資0円・払戻0円）。",
            "hypothetical_bet": 0,
            "hypothetical_payout": 0,
            "pass_benefit": 0,
            "strategy_type": "cancelled",
            "hit_type": "開催中止",
            "bet_breakdown": {},
            "payout_breakdown": {},
            "bet_details": None,
            "payout_details": None,
            "umaren_bet": 0,
            "trio_bet": 0,
            "umaren_payout": 0,
            "trio_payout": 0,
            "mark_orders": [],
            "top3_orders": []
        }

    # レース確定結果から各出走馬の直近走データ・統計（stats_dict / stats_overlay）を最新化
    try:
        update_stats_from_race_settlement(race_id, race_date, pred_data, race_res)
    except Exception as e_stats:
        print(f"[StatsOverlay] レース確定統計更新エラー ({race_id}): {e_stats}")

    try:
        pred_data = sync_virtual_badge_to_prediction(race_id, pred_data, race_date)
    except Exception as e_v_sync:
        print(f"[VirtualBadgeSync Error] {race_id}: {e_v_sync}")

    rec = pred_data.get("recommendation", {})
    hybrid_rec = pred_data.get("hybrid_recommendation")

    target_u = rec.get("target_umaban")
    h_name = rec.get("horse_name")
    grade = rec.get("grade", "B")
    orders = race_res.get("orders", {})
    actual_order = orders.get(target_u) if target_u is not None else None
    if actual_order is None and target_u is not None:
        actual_order = orders.get(str(target_u))

    real_bet_meta = {
        "is_real_bet_target": bool((hybrid_rec or {}).get("is_real_bet_target", False)),
        "real_bet_badge": (hybrid_rec or {}).get("real_bet_badge"),
        "real_bet_level": (hybrid_rec or {}).get("real_bet_level", "none"),
        "is_virtual_bet_target": bool((hybrid_rec or {}).get("is_virtual_bet_target", False)),
        "virtual_bet_slot": (hybrid_rec or {}).get("virtual_bet_slot"),
        "edge_ratio": (hybrid_rec or {}).get("edge_ratio"),
        "is_final_odds": bool((hybrid_rec or {}).get("is_final_odds", False))
    }

    # ハイブリッド推奨を最優先で照合（見送りレース または 仮想馬券レース）
    if not hybrid_rec or not hybrid_rec.get("is_participate", False):
        is_real = bool(real_bet_meta.get("is_real_bet_target") or real_bet_meta.get("real_bet_level") in ["high", "standard"])
        is_virt = bool(
            (hybrid_rec or {}).get("is_virtual_bet_target")
            or real_bet_meta.get("is_virtual_bet_target")
            or real_bet_meta.get("real_bet_level") == "virtual"
            or (real_bet_meta.get("real_bet_badge") and "仮想" in str(real_bet_meta.get("real_bet_badge")))
        ) and not is_real

        pass_info = calculate_hybrid_pass_benefit_info(hybrid_rec, race_res, target_u, actual_order, pred_data=pred_data)
        hypo_bet = pass_info["hypothetical_bet"]
        hypo_pay = pass_info["hypothetical_payout"]
        pass_benefit = pass_info["pass_benefit"]
        pay_tan = pass_info.get("pay_tan", 0)
        pay_fuku = pass_info.get("pay_fuku", 0)
        mark_orders = pass_info.get("mark_orders", [])
        top3_orders = pass_info.get("top3_orders", [])
        marks_summary = format_mark_orders_text(mark_orders)
        marks_line = f"\n{marks_summary}" if marks_summary else ""

        real_bet_meta["mark_orders"] = mark_orders
        real_bet_meta["top3_orders"] = top3_orders

        if is_virt:
            # ★ 仮想馬券: 全参戦（仮想含む）モードで収支を集計 ★
            v_profit = hypo_pay - hypo_bet
            v_rec = round((hypo_pay / hypo_bet) * 100.0, 1) if hypo_bet > 0 else 0.0
            is_virt_hit = (hypo_pay > 0)
            hit_type_virt = "仮想的中" if is_virt_hit else "仮想不的中"

            p_breakdown = {}
            if pay_tan > 0: p_breakdown["単勝"] = pay_tan
            if pay_fuku > 0: p_breakdown["複勝"] = pay_fuku

            p_details = []
            if pay_tan > 0:
                p_details.append({"type": "単勝", "comb": f"{target_u}", "amount": 100, "payout": pay_tan})
            if pay_fuku > 0:
                p_details.append({"type": "複勝", "comb": f"{target_u}", "amount": 100, "payout": pay_fuku})

            if is_virt_hit:
                hit_texts = [f"{p['type']}{p['comb']}({p['payout']:,}円)" for p in p_details]
                msg_virt = f"【仮想的中】 {' ＆ '.join(hit_texts)}！ 仮想払戻: {hypo_pay:,}円 (収支: {v_profit:+,}円 / 回収率: {v_rec}%){marks_line}"
            else:
                ord_info = f" (◎{target_u}番: {actual_order}着)" if (actual_order and actual_order < 90) else ""
                msg_virt = f"【仮想不的中】{ord_info} (仮想投資: {hypo_bet:,}円 / 払戻: 0円 / 収支: {v_profit:,}円 / 回収率: 0.0%){marks_line}"

            settlement = {
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "strategy_type": "virtual",
                "target_umaban": target_u,
                "horse_name": h_name,
                "grade": grade,
                "is_pass": False,
                "tansho_bet": 100,
                "fukusho_bet": 100,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": hypo_bet,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "tansho_payout": pay_tan,
                "fukusho_payout": pay_fuku,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": hypo_pay,
                "is_hit": is_virt_hit,
                "hit_type": hit_type_virt,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "bet_breakdown": {"単勝": 100, "複勝": 100},
                "payout_breakdown": p_breakdown,
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": p_details,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders,
                "track_bias": t_bias,
                "odds_alerts": o_alerts
            }
            save_settlement(settlement)
            if supabase:
                try:
                    supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
                except Exception:
                    pass

            return {
                "race_id": race_id,
                "race_name": race_name,
                "is_pass": False,
                "grade": grade,
                "target_umaban": target_u,
                "horse_name": h_name,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "is_hit": is_virt_hit,
                "total_bet": hypo_bet,
                "total_payout": hypo_pay,
                "profit": v_profit,
                "recovery_rate": v_rec,
                "from_cache": from_cache,
                "message": msg_virt,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "strategy_type": "virtual",
                "hit_type": hit_type_virt,
                "bet_breakdown": {"単勝": 100, "複勝": 100},
                "payout_breakdown": p_breakdown,
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": p_details,
                "tansho_bet": 100,
                "fukusho_bet": 100,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "tansho_payout": pay_tan,
                "fukusho_payout": pay_fuku,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }
        else:
            msg = pass_info["message"]
            settlement = {
                "race_id": race_id,
                "race_date": race_date,
                "race_name": race_name,
                "strategy_type": hybrid_rec.get("strategy_type", "skip") if hybrid_rec else "skip",
                "target_umaban": target_u,
                "horse_name": h_name,
                "grade": grade,
                "is_pass": True,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": 0,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "tansho_payout": 0,
                "fukusho_payout": 0,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": 0,
                "is_hit": False,
                "hit_type": "見送り",
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "bet_breakdown": {},
                "payout_breakdown": {},
                "bet_details": {"real_bet_meta": real_bet_meta},
                "payout_details": None,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders,
                "track_bias": t_bias,
                "odds_alerts": o_alerts
            }
            save_settlement(settlement)
            if supabase:
                try:
                    supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
                except Exception:
                    pass

            return {
                "race_id": race_id,
                "race_name": race_name,
                "is_pass": True,
                "grade": grade,
                "target_umaban": target_u,
                "horse_name": h_name,
                "actual_order": actual_order if (actual_order and actual_order < 90) else None,
                "is_hit": False,
                "total_bet": 0,
                "total_payout": 0,
                "profit": 0,
                "recovery_rate": 0.0,
                "from_cache": from_cache,
                "message": msg,
                "hypothetical_bet": hypo_bet,
                "hypothetical_payout": hypo_pay,
                "pass_benefit": pass_benefit,
                "strategy_type": hybrid_rec.get("strategy_type", "skip") if hybrid_rec else "skip",
                "hit_type": "見送り",
                "bet_breakdown": {},
                "payout_breakdown": {},
                "bet_details": None,
                "payout_details": None,
                "tansho_bet": 0,
                "fukusho_bet": 0,
                "umatan_bet": 0,
                "umaren_bet": 0,
                "trio_bet": 0,
                "tansho_payout": 0,
                "fukusho_payout": 0,
                "umatan_payout": 0,
                "umaren_payout": 0,
                "trio_payout": 0,
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }

    # 参戦レースの場合: PayoutVerifier で全券種（単勝・複勝・馬単・馬連・三連複・3連単）を一括照合
    all_tickets = []
    for b in (hybrid_rec.get("tansho_bets") or []):
        all_tickets.append({"ticket_type": "単勝", "combination": b.get("combination", [b.get("target_umaban")]), "amount": b.get("amount", 100)})
    for b in (hybrid_rec.get("fukusho_bets") or []):
        all_tickets.append({"ticket_type": "複勝", "combination": b.get("combination", [b.get("target_umaban")]), "amount": b.get("amount", 100)})
    for b in (hybrid_rec.get("umatan_bets") or []):
        all_tickets.append({"ticket_type": "馬単", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    for b in (hybrid_rec.get("umaren_bets") or []):
        all_tickets.append({"ticket_type": "馬連", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    for b in (hybrid_rec.get("sanrentan_bets") or []):
        all_tickets.append({"ticket_type": "3連単", "combination": b.get("combination", []), "amount": b.get("amount", 100)})
    trio_bets = hybrid_rec.get("trio_bets") or []
    trio_bet = hybrid_rec.get("trio_bet")
    if not trio_bets and trio_bet and isinstance(trio_bet, dict) and trio_bet.get("combination"):
        trio_bets = [trio_bet]
    for b in trio_bets:
        all_tickets.append({"ticket_type": "三連複", "combination": b.get("combination", []), "amount": b.get("amount", 100)})

    payouts_data = race_res.get("payouts", {})
    total_payout, all_hits, bet_breakdown, payout_breakdown = PayoutVerifier.verify_all_tickets(all_tickets, payouts_data)

    tan_bet_tot = bet_breakdown.get("単勝", 0)
    fuku_bet_tot = bet_breakdown.get("複勝", 0)
    utan_bet_tot = bet_breakdown.get("馬単", 0)
    u_bet_tot = bet_breakdown.get("馬連", 0)
    s_bet_tot = bet_breakdown.get("3連単", 0)
    t_bet_tot = bet_breakdown.get("三連複", 0)

    tansho_payout = payout_breakdown.get("単勝", 0)
    fukusho_payout = payout_breakdown.get("複勝", 0)
    umatan_payout = payout_breakdown.get("馬単", 0)
    umaren_payout = payout_breakdown.get("馬連", 0)
    sanrentan_payout = payout_breakdown.get("3連単", 0)
    trio_payout = payout_breakdown.get("三連複", 0)

    total_bet = sum(bet_breakdown.values())
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    hit_types_found = [h["type"] for h in all_hits]
    if len(all_hits) >= 2:
        hit_type = "W的中🔥"
    elif "馬単" in hit_types_found:
        hit_type = "馬単的中🎯"
    elif "単勝" in hit_types_found:
        hit_type = "単勝的中🎯"
    elif "複勝" in hit_types_found:
        hit_type = "複勝的中🎯"
    elif "3連単" in hit_types_found:
        hit_type = "3連単的中🎯"
    elif "三連複" in hit_types_found:
        hit_type = "三連複的中🎯"
    elif "馬連" in hit_types_found:
        hit_type = "馬連的中🎯"
    else:
        hit_type = "不的中"

    orders = race_res.get("orders", {})
    mark_orders = extract_mark_orders(pred_data, orders)
    top3_orders = extract_top3_orders(pred_data, orders)
    marks_summary = format_mark_orders_text(mark_orders)
    marks_line = f"\n{marks_summary}" if marks_summary else ""

    if is_hit:
        hit_texts = [f"{h['type']}{h['comb']}({h['payout']:,}円)" for h in all_hits]
        msg = f"【{hit_type}】{' ＆ '.join(hit_texts)}！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery_rate}%){marks_line}"
    else:
        order_info = f" (◎{target_u}番: {actual_order}着)" if (actual_order and actual_order < 90) else ""
        msg = f"不的中{order_info} (投資: {total_bet:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%){marks_line}"

    bet_breakdown = {}
    payout_breakdown = {}
    if tan_bet_tot > 0:
        bet_breakdown["単勝"] = tan_bet_tot
        payout_breakdown["単勝"] = tansho_payout
    if fuku_bet_tot > 0:
        bet_breakdown["複勝"] = fuku_bet_tot
        payout_breakdown["複勝"] = fukusho_payout
    if utan_bet_tot > 0:
        bet_breakdown["馬単"] = utan_bet_tot
        payout_breakdown["馬単"] = umatan_payout
    if u_bet_tot > 0:
        bet_breakdown["馬連"] = u_bet_tot
        payout_breakdown["馬連"] = umaren_payout
    if s_bet_tot > 0:
        bet_breakdown["3連単"] = s_bet_tot
        payout_breakdown["3連単"] = sanrentan_payout
    if t_bet_tot > 0:
        bet_breakdown["三連複"] = t_bet_tot
        payout_breakdown["三連複"] = trio_payout

    real_bet_meta["mark_orders"] = mark_orders
    real_bet_meta["top3_orders"] = top3_orders

    settlement = {
        "race_id": race_id,
        "race_date": race_date,
        "race_name": race_name,
        "strategy_type": hybrid_rec.get("strategy_type", "cascade"),
        "target_umaban": target_u,
        "horse_name": h_name,
        "grade": grade,
        "is_pass": False,
        "tansho_bet": tan_bet_tot,
        "fukusho_bet": fuku_bet_tot,
        "umatan_bet": utan_bet_tot,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "total_bet": total_bet,
        "actual_order": actual_order if (actual_order and actual_order < 90) else None,
        "tansho_payout": tansho_payout,
        "fukusho_payout": fukusho_payout,
        "umatan_payout": umatan_payout,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "total_payout": total_payout,
        "is_hit": is_hit,
        "hit_type": hit_type,
        "hypothetical_bet": 0,
        "hypothetical_payout": 0,
        "pass_benefit": 0,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "tansho_bets": tansho_bets,
            "fukusho_bets": fukusho_bets,
            "umatan_bets": umatan_bets,
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet,
            "real_bet_meta": real_bet_meta
        },
        "payout_details": all_hits,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "track_bias": t_bias,
        "odds_alerts": o_alerts
    }
    save_settlement(settlement)
    if supabase:
        try:
            supabase.table("keiba_scheduled_races").update({"is_settled": 1}).eq("race_id", race_id).execute()
        except Exception:
            pass

    # 実戦推奨レースの結果確定プッシュ通知 (当日開催の新規確定時のみ配信し、過去日の一括再検証時は通知を抑止)
    is_today_race = (race_date == today_jst_str())
    if not from_cache and is_today_race and hybrid_rec.get("is_real_bet_target"):
        try:
            if is_hit:
                send_fcm_notification(
                    title=f"🎯【実戦的中！】{race_name} 払戻: {total_payout:,}円",
                    body=f"【{hit_type}】収支: {'+' if profit >= 0 else ''}{profit:,}円 (回収率: {recovery_rate}%)",
                    data={"type": "SETTLEMENT_HIT", "race_id": race_id}
                )
            else:
                o_str = f" (◎{target_u}番 {actual_order}着)" if actual_order else ""
                send_fcm_notification(
                    title=f"🏁【結果確定】{race_name} 不的中",
                    body=f"投資: {total_bet:,}円 / 払戻: 0円{o_str} (収支: {profit:,}円)",
                    data={"type": "SETTLEMENT_MISS", "race_id": race_id}
                )
        except Exception as ne:
            print(f"[FCM通知エラー]: {ne}")

    return {
        "race_id": race_id,
        "race_name": race_name,
        "is_pass": False,
        "grade": grade,
        "target_umaban": target_u,
        "horse_name": h_name,
        "actual_order": actual_order if (actual_order and actual_order < 90) else None,
        "is_hit": is_hit,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "recovery_rate": recovery_rate,
        "from_cache": from_cache,
        "message": msg,
        "strategy_type": hybrid_rec.get("strategy_type", "cascade"),
        "hit_type": hit_type,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "tansho_bets": tansho_bets,
            "fukusho_bets": fukusho_bets,
            "umatan_bets": umatan_bets,
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "tansho_bet": tan_bet_tot,
        "fukusho_bet": fuku_bet_tot,
        "umatan_bet": utan_bet_tot,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "tansho_payout": tansho_payout,
        "fukusho_payout": fukusho_payout,
        "umatan_payout": umatan_payout,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "is_real_bet_target": bool(hybrid_rec.get("is_real_bet_target", False))
    }


def resolve_race_date_for_id(race_id: str) -> str:
    """レースIDに対応する正確な開催日 (YYYY-MM-DD) をキャッシュ・DB・netkeibaから逆引き"""
    for d_cached, c_entry in _period_venues_cache.items():
        v_list = c_entry.get("venues", []) if isinstance(c_entry, dict) else (c_entry if isinstance(c_entry, list) else [])
        for v in v_list:
            if isinstance(v, dict) and any(r.get("race_id") == race_id for r in v.get("races", [])):
                return d_cached

    # 2026年秋開催の会場別確定日程マッピング
    if len(race_id) == 12 and race_id.startswith("2026"):
        venue_code = race_id[4:6]
        day_code = race_id[6:10]
        if venue_code in ("05", "08"):  # 4回東京(05)・4回京都(08)
            if day_code == "0401":
                return "2026-10-03"
            elif day_code == "0402":
                return "2026-10-04"
            elif day_code == "0403":
                return "2026-10-10"
            elif day_code == "0404":
                return "2026-10-11"
            elif day_code == "0405":
                return "2026-10-12"
            elif day_code == "0406":
                return "2026-10-17"
            elif day_code == "0407":
                return "2026-10-18"
        elif venue_code == "04":  # 4回新潟(04)
            if day_code == "0401":
                return "2026-10-17"
            elif day_code == "0402":
                return "2026-10-18"
        elif venue_code in ("06", "07"):  # 4回中山(06)・4回中京(07)
            if day_code == "0408":
                return "2026-09-26"
            elif day_code == "0409":
                return "2026-09-27"

    if supabase:
        for tbl in ("keiba_scheduled_races", "keiba_predictions_cache", "keiba_race_settlements"):
            try:
                res = supabase.table(tbl).select("race_date").eq("race_id", race_id).execute()
                if res.data and len(res.data) > 0 and res.data[0].get("race_date"):
                    r_date = res.data[0]["race_date"]
                    if r_date and r_date != "2026-10-13":
                        return r_date
            except Exception:
                pass

    # netkeibaのHTMLから正確な開催年月日を抽出
    try:
        url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        resp = safe_scrape_get(url, timeout=5.0)
        html = safe_decode_html(resp)
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", html)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    except Exception:
        pass

    return today_jst_str()


def verify_race_result(
    race_id: str,
    bypass_cache: bool = Query(False, description="キャッシュをバイパスして最新結果を取得"),
    is_19h_settle: bool = Query(False, description="19時の一括確定モード")
):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    if not bypass_cache:
        cached_settlement = get_cached_settlement(race_id)
        if cached_settlement and (
            cached_settlement.get("actual_order") is not None
            or cached_settlement.get("is_pass")
            or cached_settlement.get("hit_type") == "中止"
            or cached_settlement.get("top3_orders")
        ):
            return cached_settlement

    race_date = resolve_race_date_for_id(race_id)

    cached_data = get_cached_result(race_id) if not bypass_cache else None
    from_cache = False

    if cached_data:
        c_orders = cached_data.get("orders", {}) if isinstance(cached_data, dict) else {}
        c_payouts = cached_data.get("payouts", {}) if isinstance(cached_data, dict) else {}
        c_order_vals = [int(v) for v in c_orders.values() if str(v).isdigit()]
        c_has_top3 = (1 in c_order_vals) and (2 in c_order_vals) and (3 in c_order_vals)
        c_has_pay = bool(c_payouts.get("tansho") or c_payouts.get("umaren") or c_payouts.get("fukusho"))
        if (c_has_top3 and c_has_pay) or (isinstance(cached_data, dict) and cached_data.get("is_cancelled")):
            from_cache = True
            race_res = cached_data
        else:
            race_res = fetch_netkeiba_race_result(race_id)
    else:
        race_res = fetch_netkeiba_race_result(race_id)

    if not race_res:
        raise HTTPException(status_code=400, detail="レース結果がまだ確定していないか、取得できませんでした。発走後しばらくしてから再試行してください。")

    # 予想情報を取得
    pred_data = predict_race_core(race_id)
    rec = pred_data.get("recommendation", {})
    race_name = pred_data.get("race_name", race_res.get("race_name", f"Race {race_id}"))

    target_u = rec.get("target_umaban")
    orders = race_res.get("orders", {})
    payouts = race_res.get("payouts", {})
    actual_order = orders.get(target_u) if target_u is not None else None
    if actual_order is None and target_u is not None:
        actual_order = orders.get(str(target_u))

    # ★ 着順および払戻金が実数値で確認できるかの厳格チェック
    has_top3 = (1 in orders.values()) and (2 in orders.values()) and (3 in orders.values())
    has_payouts = (len(payouts.get("tansho", {})) > 0) or (len(payouts.get("umaren", {})) > 0) or (len(payouts.get("fukusho", {})) > 0)
    is_cancelled = bool(race_res.get("is_cancelled"))

    # 1〜3着確定および払戻金確定（または開催中止確定）があれば、JRA公式結果としてレース全体が確定している
    # （◎馬が競走中止・除外・取消・大敗等の場合でも、レース照合は正常完了させる）
    is_fully_confirmed = (has_top3 and has_payouts) or is_cancelled
    if not is_19h_settle and not is_fully_confirmed:
        raise HTTPException(
            status_code=400,
            detail=f"着順または払戻金が実数値で確定していません（入線直後・審議中等のため未確定）。確定後に再度照合してください。(確認状況: 1〜3着確定={has_top3}, 払戻金確定={has_payouts})"
        )

    save_cached_result(race_id, race_name, race_date, race_res)

    return process_settlement_calculation(
        race_id=race_id,
        race_date=race_date,
        race_name=race_name,
        pred_data=pred_data,
        race_res=race_res,
        from_cache=from_cache
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

