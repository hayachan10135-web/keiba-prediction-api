from typing import List, Optional, Dict, Any
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
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="Keiba Prediction & Verification API")

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


# --- 堅牢なスクレイピング用 HTTP クライアント基盤 (Step 1 冗長化) ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

HEADERS_PC = {
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://race.netkeiba.com/"
}


def create_robust_session() -> requests.Session:
    """自動リトライ、コネクションプール、タイムアウト制御を備えた堅牢なセッション"""
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=30)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


GLOBAL_HTTP_SESSION = create_robust_session()


def safe_scrape_get(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
    referer: str = "https://race.netkeiba.com/",
    max_retries: int = 3
) -> requests.Response:
    """
    User-Agentローテーション、リトライ、ジッター、接続保持を統合した安全なHTTP GET
    """
    req_headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": referer,
        "Connection": "keep-alive"
    }
    if headers:
        req_headers.update(headers)

    last_exc = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                sleep_time = 0.3 * (2 ** attempt) + random.uniform(0.05, 0.2)
                time.sleep(sleep_time)
            resp = GLOBAL_HTTP_SESSION.get(url, headers=req_headers, timeout=timeout)
            if resp.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    continue
            return resp
        except (requests.exceptions.RequestException, Exception) as e:
            last_exc = e
            if attempt < max_retries - 1:
                continue

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def safe_decode_html(resp: requests.Response) -> str:
    """EUC-JP / UTF-8 をスマートに自動判定して文字列デコード"""
    content = resp.content
    raw_head = content[:400].lower()
    if b'charset="utf-8"' in raw_head or b"charset=utf-8" in raw_head or b"utf-8" in raw_head:
        return content.decode("utf-8", errors="replace")
    try:
        return content.decode("euc-jp")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


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
class HorsePrediction(BaseModel):
    umaban: int
    wakuban: int
    horse_name: str
    jockey: str
    trainer: Optional[str] = None
    kinryo: float
    odds: Optional[float] = None
    popularity: Optional[int] = None
    score: Optional[float] = 0.0
    prob_win: Optional[float] = None
    prob_top3: Optional[float] = None
    mark: str
    is_blinker: Optional[bool] = False
    is_first_blinker: Optional[bool] = False
    horse_weight: Optional[int] = None
    weight_diff: Optional[int] = None
    sex_age: Optional[str] = None
    base_odds: Optional[float] = None
    odds_history: Optional[Dict[str, float]] = None
    odds_change_pct: Optional[float] = None
    odds_trend: Optional[str] = None
    odds_alert: Optional[str] = None

class TicketBet(BaseModel):
    ticket_type: str
    combination: List[int]
    combination_str: str
    amount: int
    odds: Optional[float] = None
    expected_payout: Optional[int] = None
    is_real_odds: Optional[bool] = False
    memo: Optional[str] = None

class HybridRecommendation(BaseModel):
    is_participate: bool
    strategy_name: str = "波乱度メタモデル最適化"
    strategy_type: Optional[str] = "daiharan"
    total_amount: int
    summary: str
    tansho_bets: Optional[List[TicketBet]] = []
    fukusho_bets: Optional[List[TicketBet]] = []
    umatan_bets: Optional[List[TicketBet]] = []
    tansho_amount: Optional[int] = 0
    fukusho_amount: Optional[int] = 0
    umatan_amount: Optional[int] = 0
    umaren_bets: List[TicketBet] = []
    sanrentan_bets: Optional[List[TicketBet]] = []
    hypothetical_umaren_bets: Optional[List[TicketBet]] = []
    trio_bets: Optional[List[TicketBet]] = []
    trio_bet: Optional[TicketBet] = None
    reason: Optional[str] = None
    is_real_odds_applied: Optional[bool] = False
    total_expected_payout: Optional[int] = None
    prob_gachi: Optional[float] = None
    prob_himo: Optional[float] = None
    edge_ratio: Optional[float] = None
    confidence_score: Optional[float] = None
    is_real_bet_target: Optional[bool] = False
    real_bet_badge: Optional[str] = None
    real_bet_level: Optional[str] = None
    is_final_odds: Optional[bool] = False
    # 仮想馬券フィールド
    is_virtual_bet_target: Optional[bool] = False
    virtual_bet_slot: Optional[int] = None
    virtual_strategy_name: Optional[str] = None
    virtual_badge: Optional[str] = None
    virtual_bets: Optional[List[TicketBet]] = []
    virtual_total_amount: Optional[int] = 0

class TanFukuRecommendation(BaseModel):
    target_umaban: Optional[int]
    horse_name: Optional[str]
    grade: str
    confidence_label: str
    is_pass: bool
    tansho_amount: int
    fukusho_amount: int
    total_amount: int
    summary: str

class RaceDetails(BaseModel):
    race_num: str
    post_time: str
    course_details: str

class PredictResponse(BaseModel):
    race_id: str
    race_name: str
    race_details: RaceDetails
    horses: List[HorsePrediction]
    recommendation: TanFukuRecommendation
    hybrid_recommendation: Optional[HybridRecommendation] = None


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

class TicketTypeSummary(BaseModel):
    ticket_type: str
    bet_amount: int
    payout_amount: int
    hit_count: int
    bet_count: int
    recovery_rate: float

class VerificationResult(BaseModel):
    race_id: str
    race_name: str
    is_pass: bool
    grade: str
    target_umaban: Optional[int]
    horse_name: Optional[str]
    actual_order: Optional[int]
    is_hit: bool
    total_bet: int
    total_payout: int
    profit: int
    recovery_rate: float
    from_cache: bool
    message: str
    hypothetical_bet: Optional[int] = None
    hypothetical_payout: Optional[int] = None
    pass_benefit: Optional[int] = None
    strategy_type: Optional[str] = "hybrid"
    hit_type: Optional[str] = None
    bet_breakdown: Optional[Dict[str, int]] = {}
    payout_breakdown: Optional[Dict[str, int]] = {}
    bet_details: Optional[Any] = None
    payout_details: Optional[Any] = None
    umaren_bet: Optional[int] = 0
    trio_bet: Optional[int] = 0
    tansho_bet: Optional[int] = 0
    fukusho_bet: Optional[int] = 0
    umatan_bet: Optional[int] = 0
    umaren_payout: Optional[int] = 0
    trio_payout: Optional[int] = 0
    tansho_payout: Optional[int] = 0
    fukusho_payout: Optional[int] = 0
    umatan_payout: Optional[int] = 0
    mark_orders: Optional[List[dict]] = None
    top3_orders: Optional[List[dict]] = None
    target_popularity: Optional[int] = None

class DailySummary(BaseModel):
    target_date: str
    total_races: int
    bet_races: int
    pass_races: int
    hit_races: int
    hit_rate: float
    total_bet_amount: int
    total_payout_amount: int
    total_profit: int
    recovery_rate: float
    settled_races: List[dict]
    pass_benefit_total: Optional[int] = 0
    pass_avoided_losses: Optional[int] = 0
    pass_missed_profits: Optional[int] = 0
    ticket_breakdowns: Optional[List[TicketTypeSummary]] = []
    umaren_hit_races: Optional[int] = 0
    trio_hit_races: Optional[int] = 0
    umaren_bet_total: Optional[int] = 0
    trio_bet_total: Optional[int] = 0
    umaren_payout_total: Optional[int] = 0
    trio_payout_total: Optional[int] = 0
    umaren_recovery_rate: Optional[float] = 0.0
    trio_recovery_rate: Optional[float] = 0.0
    real_bet_races: Optional[int] = 0
    real_bet_hits: Optional[int] = 0
    real_bet_hit_rate: Optional[float] = 0.0
    real_bet_amount: Optional[int] = 0
    real_bet_payout: Optional[int] = 0
    real_bet_profit: Optional[int] = 0
    real_bet_recovery_rate: Optional[float] = 0.0
    cumulative_days: Optional[int] = 0
    cumulative_total_races: Optional[int] = 0
    cumulative_bet_races: Optional[int] = 0
    cumulative_pass_races: Optional[int] = 0
    cumulative_hit_races: Optional[int] = 0
    cumulative_hit_rate: Optional[float] = 0.0
    cumulative_bet_amount: Optional[int] = 0
    cumulative_payout_amount: Optional[int] = 0
    cumulative_profit: Optional[int] = 0
    cumulative_recovery_rate: Optional[float] = 0.0
    cumulative_pass_benefit: Optional[int] = 0
    cumulative_real_bet_races: Optional[int] = 0
    cumulative_real_bet_hits: Optional[int] = 0
    cumulative_real_bet_hit_rate: Optional[float] = 0.0
    cumulative_real_bet_amount: Optional[int] = 0
    cumulative_real_bet_payout: Optional[int] = 0
    cumulative_real_bet_profit: Optional[int] = 0
    cumulative_real_bet_recovery_rate: Optional[float] = 0.0

class PeriodStatItem(BaseModel):
    title: str
    period_key: str
    days: int = 0
    total_races: int = 0
    bet_races: int = 0
    pass_races: int = 0
    hit_races: int = 0
    hit_rate: float = 0.0
    total_bet: int = 0
    total_payout: int = 0
    total_profit: int = 0
    recovery_rate: float = 0.0
    real_bet_races: int = 0
    real_bet_hits: int = 0
    real_bet_hit_rate: float = 0.0
    real_bet_amount: int = 0
    real_bet_payout: int = 0
    real_bet_profit: int = 0
    real_bet_recovery_rate: float = 0.0
    pass_benefit: int = 0

class AggregateSummaryResponse(BaseModel):
    lifetime: PeriodStatItem
    yearly: List[PeriodStatItem] = []
    monthly: List[PeriodStatItem] = []

class BatchStartResponse(BaseModel):
    status: str
    message: str

class BatchStatusResponse(BaseModel):
    is_running: bool
    total: int
    current: int
    predicted: int
    settled: int
    message: str

batch_status = {
    "is_running": False,
    "total": 0,
    "current": 0,
    "predicted": 0,
    "settled": 0,
    "message": "待機中"
}

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


def fetch_real_odds(race_id: str) -> dict:
    """netkeibaのリアルタイムオッズAPIから馬連(type=4)および三連複(type=7)の実オッズを取得"""
    odds_data = {"umaren": {}, "trio": {}}
    if not race_id or len(race_id) != 12:
        return odds_data

    # 1. 馬連 (type=4)
    try:
        u_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=4&action=init&output=json"
        oresp = safe_scrape_get(u_url, timeout=6.0)
        if oresp.status_code == 200:
            d = oresp.json()
            if isinstance(d, str):
                d = json.loads(d)
            db = d.get("data", {})
            if isinstance(db, str):
                db = json.loads(db)
            u_dict = db.get("odds", {}).get("4", {})
            if isinstance(u_dict, dict):
                for comb_str, val in u_dict.items():
                    # comb_str: "0102"
                    if len(comb_str) == 4 and comb_str.isdigit() and isinstance(val, list) and len(val) >= 1:
                        o_str = str(val[0]).replace(",", "").strip()
                        if o_str and o_str != "---":
                            try:
                                c1, c2 = int(comb_str[:2]), int(comb_str[2:])
                                odds_data["umaren"][(min(c1, c2), max(c1, c2))] = float(o_str)
                            except ValueError:
                                pass
    except Exception as e:
        print(f"馬連実オッズ取得エラー ({race_id}): {e}")

    # 2. 三連複 (type=7)
    try:
        t_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=7&action=init&output=json"
        oresp = safe_scrape_get(t_url, timeout=6.0)
        if oresp.status_code == 200:
            d = oresp.json()
            if isinstance(d, str):
                d = json.loads(d)
            db = d.get("data", {})
            if isinstance(db, str):
                db = json.loads(db)
            t_dict = db.get("odds", {}).get("7", {})
            if isinstance(t_dict, dict):
                for comb_str, val in t_dict.items():
                    # comb_str: "010203"
                    if len(comb_str) == 6 and comb_str.isdigit() and isinstance(val, list) and len(val) >= 1:
                        o_str = str(val[0]).replace(",", "").strip()
                        if o_str and o_str != "---":
                            try:
                                c1, c2, c3 = int(comb_str[:2]), int(comb_str[2:4]), int(comb_str[4:])
                                s_comb = tuple(sorted([c1, c2, c3]))
                                odds_data["trio"][s_comb] = float(o_str)
                            except ValueError:
                                pass
    except Exception as e:
        print(f"三連複実オッズ取得エラー ({race_id}): {e}")

    return odds_data


# --- 開催期間（土曜日 〜 次週金曜日 20:00）管理ヘルパー ---
_period_venues_cache: Dict[str, Any] = {}
_period_checked_empty_dates: set = set()
_latest_post_times: Dict[str, str] = {}  # {race_id: "09:50発走"} (スクレイピングした実際の最新出走時刻)
_last_weight_check_ts: Dict[str, float] = {}  # {race_id: timestamp} (馬体重発表チェックのクールダウン管理)


def get_current_kaisai_period(now_dt: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    """
    1開催期間（土曜日 00:00 〜 次週金曜日 20:00）の境界と待機状態を算出。
    - 金曜 20:00 以降 〜 土曜 07:30 未満は「次週データ取得待機期間 (is_waiting_period=True)」とする。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    today_d = now_dt.date()
    weekday = today_d.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6

    is_waiting = False
    if weekday == 4 and now_dt.hour >= 20:
        is_waiting = True
    elif weekday == 5 and (now_dt.hour < 7 or (now_dt.hour == 7 and now_dt.minute < 30)):
        is_waiting = True

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
    """金曜20:00の開催期間キャッシュ破棄（成績 keiba_race_settlements は永続保持）"""
    global _period_venues_cache, _period_checked_empty_dates, _latest_post_times, _last_weight_check_ts
    _period_venues_cache.clear()
    _period_checked_empty_dates.clear()
    _latest_post_times.clear()
    _last_weight_check_ts.clear()
    if not supabase:
        return
    try:
        supabase.table("keiba_predictions_cache").delete().neq("race_id", "SCHEMA_VERSION").execute()
        supabase.table("keiba_race_results_cache").delete().neq("race_id", "0").execute()
        supabase.table("keiba_scheduled_races").delete().neq("race_id", "0").execute()
        print("[Cleanup] 金曜20:00 開催期間の一時キャッシュ（予想・結果HTML・スケジュール）を破棄しました（成績台帳は保持）。")
    except Exception as e:
        print(f"[Cleanup] キャッシュ破棄エラー: {e}")
    invalidate_summary_caches()


# --- サマリー・総合収支のインメモリキャッシュ (高速化用) ---
_summary_dates_cache: Dict[str, Any] = {
    "dates": [],
    "updated_at": 0.0
}
_aggregate_summary_cache: Dict[str, Any] = {
    "data": None,
    "updated_at": 0.0
}
CACHE_TTL_SUMMARY_DATES = 600.0  # 10分
CACHE_TTL_AGGREGATE = 600.0      # 10分

def invalidate_summary_caches():
    """新レース確定や一括同期時にサマリーキャッシュを破棄"""
    global _summary_dates_cache, _aggregate_summary_cache
    _summary_dates_cache["updated_at"] = 0.0
    _aggregate_summary_cache["updated_at"] = 0.0


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
            # 1. keiba_scheduled_races の post_datetime を最新時刻で更新
            for row in updates_for_schedule:
                try:
                    supabase.table("keiba_scheduled_races").update({
                        "post_datetime": row["post_datetime"],
                        "race_name": row["race_name"]
                    }).eq("race_id", row["race_id"]).execute()
                except Exception:
                    pass
    except Exception as e:
        print(f"スケジュール一括同期エラー ({formatted_date}): {e}")


# =========================================================================
# ★ レース一覧パーサー（前日誤取得防止・当日レース確実抽出ロジック）
# =========================================================================
def parse_netkeiba_race_list_html(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    venues_data = []

    VENUE_CODE_MAP = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }

    # kaisai_id を探して各会場の正確な kai_day を抽出 (例: payback_list.html?kaisai_id=2026050404)
    venue_target_kai_day = {}
    for a in soup.find_all("a", href=re.compile(r"kaisai_id=(\d{10})")):
        m = re.search(r"kaisai_id=(\d{10})", a["href"])
        if m:
            kid = m.group(1)
            v_code = kid[4:6]
            kai_day = kid[6:10]
            v_name = VENUE_CODE_MAP.get(v_code)
            if v_name:
                venue_target_kai_day[v_name] = kai_day

    raw_venue_races = {}
    seen_ids = set()

    for a in soup.find_all("a", href=re.compile(r"race_id=(\d{12})")):
        href = a["href"]
        m = re.search(r"race_id=(\d{12})", href)
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen_ids:
            continue
        seen_ids.add(race_id)

        v_code = race_id[4:6]
        venue_name = VENUE_CODE_MAP.get(v_code, "中央開催")
        kai_day = race_id[6:10]
        race_num_int = int(race_id[10:12])
        race_no = f"{race_num_int}R"

        raw_text = a.get_text(separator=" ", strip=True)
        clean_text = " ".join(raw_text.split())

        all_post_m = re.findall(r"(\d{1,2}:\d{2})", clean_text)
        post_time_est = f"{all_post_m[-1]}発走" if all_post_m else ""
        if post_time_est:
            _latest_post_times[race_id] = post_time_est

        if venue_name not in raw_venue_races:
            raw_venue_races[venue_name] = []

        raw_venue_races[venue_name].append({
            "race_no": race_no,
            "race_id": race_id,
            "kai_day": kai_day,
            "race_num_int": race_num_int,
            "race_name": clean_text if clean_text else f"{race_no}",
            "race_info": post_time_est
        })

    for v_name, r_list in raw_venue_races.items():
        # kaisai_id から対象の kai_day が判明していれば最優先
        target_kd = venue_target_kai_day.get(v_name)
        if target_kd:
            day_races = [r for r in r_list if r["kai_day"] == target_kd]
        else:
            day_races = []

        if not day_races:
            counts = {}
            for r in r_list:
                kd = r["kai_day"]
                counts[kd] = counts.get(kd, 0) + 1

            # レース数が最も多いものを採用、同数の場合はkai_dayが若い方（当日優先）を採用
            best_kai_day = sorted(counts.keys(), key=lambda kd: (counts[kd], -int(kd) if kd.isdigit() else 0), reverse=True)[0]
            day_races = [r for r in r_list if r["kai_day"] == best_kai_day]

        day_races.sort(key=lambda x: x["race_num_int"])

        venues_data.append({
            "venue_name": v_name,
            "races": [
                {
                    "race_no": r["race_no"],
                    "race_id": r["race_id"],
                    "race_name": r["race_name"],
                    "race_info": r["race_info"]
                }
                for r in day_races
            ]
        })

    return venues_data


def fetch_venues_for_date_raw(date_str: str, force_refresh: bool = False) -> List[dict]:
    """指定日付 (YYYY-MM-DD または YYYYMMDD) のJRA開催会場・レース一覧を取得しキャッシュ (出走時刻変更対応)"""
    clean_d = date_str.replace("-", "")
    if len(clean_d) != 8 or not clean_d.isdigit():
        return []
    formatted_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}"

    now = time.time()
    today_s = today_jst_str()
    # 当日開催日（または指定日）の出走時刻変更を迅速に検知するためTTLを60秒とする
    cache_ttl = 60.0 if formatted_date == today_s else 3600.0

    if not force_refresh and formatted_date in _period_venues_cache:
        c_entry = _period_venues_cache[formatted_date]
        if isinstance(c_entry, dict) and "venues" in c_entry:
            if now - c_entry.get("updated_at", 0) < cache_ttl:
                return json.loads(json.dumps(c_entry["venues"]))
        elif isinstance(c_entry, list):
            return json.loads(json.dumps(c_entry))

    urls = [
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={clean_d}",
        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={clean_d}",
        f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={clean_d}"
    ]
    for url in urls:
        try:
            resp = safe_scrape_get(url, timeout=10.0)
            if resp.status_code != 200:
                continue
            text = safe_decode_html(resp)

            venues = parse_netkeiba_race_list_html(text)
            # 1会場あたり8R以上ある場合のみ正式な出走表確定済み開催日とみなす（次週の未確定枠を除外）
            valid_venues = [v for v in venues if len(v.get("races", [])) >= 8]
            if valid_venues:
                _period_venues_cache[formatted_date] = {
                    "venues": valid_venues,
                    "updated_at": now
                }
                ensure_schedules_registered_for_venues(formatted_date, valid_venues)
                return json.loads(json.dumps(valid_venues))
        except Exception:
            continue

    _period_checked_empty_dates.add(formatted_date)
    return []


def discover_period_kaisai_dates(now_dt: Optional[datetime.datetime] = None) -> List[str]:
    """
    現在の1開催期間（土曜日〜次週金曜日20:00）における、本日までのJRA開催日リスト(昇順)を返す。
    金曜20:00〜土曜朝7:30の待機期間中は空リストを返す。
    """
    if now_dt is None:
        now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)
    if p_info["is_waiting_period"]:
        return []

    start_d = p_info["period_start_date"]
    today_d = p_info["today_date"]

    # DB上ですでに今開催期間のスケジュールまたは予想キャッシュがある日付を取得
    db_dates = set()
    if supabase:
        try:
            s_res = supabase.table("keiba_scheduled_races").select("race_date").gte("race_date", p_info["period_start"]).lte("race_date", p_info["today_str"]).execute()
            for row in (s_res.data or []):
                if row.get("race_date"):
                    db_dates.add(row["race_date"])
        except Exception:
            pass

    available_dates = []
    cur_d = start_d
    while cur_d <= today_d:
        d_str = cur_d.strftime("%Y-%m-%d")
        # 当日の場合は朝7:30以降のみ新規取得対象とする
        if cur_d == today_d and (now_dt.hour < 7 or (now_dt.hour == 7 and now_dt.minute < 30)) and d_str not in db_dates and d_str not in _period_venues_cache:
            cur_d += datetime.timedelta(days=1)
            continue

        if d_str in _period_venues_cache:
            available_dates.append(d_str)
        elif d_str in _period_checked_empty_dates:
            pass
        elif cur_d.weekday() in (5, 6, 0) or d_str in db_dates:
            # 土・日・月（祝日3日間開催等）またはDB登録済みの日を走査
            v = fetch_venues_for_date_raw(d_str)
            if v:
                available_dates.append(d_str)
        else:
            # 火・水・木・金の通常平日は無駄な通信を避けるため空として記録
            _period_checked_empty_dates.add(d_str)

        cur_d += datetime.timedelta(days=1)

    return sorted(list(set(available_dates)))


def attach_prediction_badges_to_venues(formatted_date: str, venues: List[dict]) -> List[dict]:
    """レース一覧の各レースにSupabase上の予想・実戦推奨バッジ情報および確定・的中結果を付与"""
    if not venues:
        return []
    cached_preds_map = {}
    settled_map = {}
    if supabase:
        try:
            p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
            for row in (p_res.data or []):
                d_json = row.get("data_json") or {}
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
        # （平地・新馬戦・障害戦除くレースの中で確信度が高い順）
        eligible_for_virtual = []
        for r in races_in_venue:
            r_nm = r.get("race_name", "")
            r_info = r.get("race_info", "")
            is_shinba = ("新馬" in r_nm) or ("メイクデビュー" in r_nm)
            is_obstacle = ("障害" in r_nm) or ("障" in r_info)
            if not is_shinba and not is_obstacle:
                eligible_for_virtual.append(r)

        # 確信度順にソートして上位3レースを選出
        eligible_for_virtual.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        top3_virtual = eligible_for_virtual[:3]
        for slot_idx, v_race in enumerate(top3_virtual, start=1):
            v_race["is_virtual_bet_target"] = True
            v_race["virtual_bet_slot"] = slot_idx
            # リアル勝負枠でない場合は「仮想観戦」バッジとレベルを付与
            if not v_race.get("is_real_bet_target"):
                v_race["real_bet_level"] = "virtual"
                v_race["real_bet_badge"] = f"🎮【仮想観戦】第{slot_idx}推奨"

    return venues


@app.get("/races/today")
def get_today_races(
    date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)"),
    force_refresh: bool = Query(False, description="手動でレース一覧・発走時刻を再取得する場合True")
):
    now_dt = now_jst_naive()
    p_info = get_current_kaisai_period(now_dt)

    # 金曜20:00〜土曜朝7:30の待機期間中は無理に出走表を取得しに行かず空を返す
    if p_info["is_waiting_period"] and not date:
        return {
            "date": "",
            "available_dates": [],
            "is_waiting_period": True,
            "message": "次週の開催データ取得待機中です（土曜朝7:30頃に自動取得されます）。",
            "venues": []
        }

    available_dates = discover_period_kaisai_dates(now_dt)

    if date:
        clean_d = date.replace("-", "")
        target_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}" if len(clean_d) == 8 else date
    else:
        if not available_dates:
            return {
                "date": "",
                "available_dates": [],
                "is_waiting_period": True,
                "message": "現在表示可能な開催データはありません（土曜朝7:30頃に取得されます）。",
                "venues": []
            }
        # 本日が開催日なら本日、平日なら今開催期間の直近（最新）の開催日を初期選択
        if p_info["today_str"] in available_dates:
            target_date = p_info["today_str"]
        else:
            target_date = available_dates[-1]

    venues = fetch_venues_for_date_raw(target_date, force_refresh=force_refresh)
    if venues:
        if target_date not in available_dates:
            available_dates = sorted(list(set(available_dates + [target_date])))
        venues = attach_prediction_badges_to_venues(target_date, venues)
        return {
            "date": target_date,
            "available_dates": available_dates,
            "is_waiting_period": False,
            "venues": venues
        }

    return {
        "date": target_date,
        "available_dates": available_dates,
        "is_waiting_period": len(available_dates) == 0,
        "message": "指定された開催日のレース情報が見つかりませんでした。",
        "venues": []
    }



def fetch_shutuba_table(race_id: str, race_date: Optional[str] = None):
    # PC版出走表を第一候補、SP版をフォールバックとして二重冗長化
    urls_to_try = [
        f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
        f"https://race.sp.netkeiba.com/race/shutuba.html?race_id={race_id}"
    ]
    html = ""
    last_err = ""
    for target_url in urls_to_try:
        try:
            resp = safe_scrape_get(target_url, timeout=10.0)
            if resp.status_code == 200 and len(resp.content) > 500:
                html = safe_decode_html(resp)
                if "HorseList" in html or "shutuba" in html or "出馬表" in html:
                    break
        except Exception as e:
            last_err = str(e)
            continue

    if not html:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba出馬表の取得に失敗しました (race_id={race_id}, error={last_err})。ダミー値生成は行いません。"
        )
    soup = BeautifulSoup(html, "html.parser")

    race_num_int = int(race_id[10:12]) if len(race_id) == 12 and race_id[10:12].isdigit() else 1
    race_num = f"{race_num_int}R"
    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    race_name = ""
    name_candidates = soup.select(".RaceName, .RaceName_Text, .Race_Title, h1.RaceName")
    for cand in name_candidates:
        txt = cand.get_text(strip=True)
        if txt and not txt.isdigit():
            race_name = txt
            break

    if not race_name:
        title_tag = soup.find("title")
        if title_tag:
            m = re.search(r"\d+R\s+([^|・\-_]+)", title_tag.get_text(strip=True))
            if m:
                race_name = m.group(1).strip()

    if not race_name:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            m = re.search(r"\d+R\s+([^|・\-_]+)", og_title["content"])
            if m:
                race_name = m.group(1).strip()

    if not race_name:
        race_name = f"{race_num} 一般競走"

    race_data_tag = soup.select_one(".RaceData01, .RaceData")
    race_data_text = race_data_tag.get_text(separator=" ", strip=True) if race_data_tag else ""
    race_data2_tag = soup.select_one(".RaceData02")
    race_data2_text = race_data2_tag.get_text(separator=" ", strip=True) if race_data2_tag else ""

    all_post_m = re.findall(r"(\d{1,2}:\d{2})\s*発走", race_data_text)
    if all_post_m:
        post_time = f"{all_post_m[-1]}発走"
    else:
        post_time_m = re.search(r"(\d{1,2}:\d{2})", race_data_text)
        post_time = f"{post_time_m.group(1)}発走" if post_time_m else ""

    if post_time:
        _latest_post_times[race_id] = post_time

    weather_m = re.search(r"天候\s*:\s*([^\s/]+)", race_data_text)
    weather = weather_m.group(1) if weather_m else ""

    baba_m = re.search(r"(?:芝|ダート|ダ)?\s*:\s*(良|稍重|重|不良)", race_data_text)
    baba = baba_m.group(1) if baba_m else "良"

    dist_m = re.search(r"(\d{3,4})m", race_data_text)
    distance = float(dist_m.group(1)) if dist_m else 1600.0

    surface_name = "芝"
    surface_type = 0
    if "障" in race_data_text or "障害" in race_name:
        surface_name = "障害"
        surface_type = 2
    elif "ダ" in race_data_text:
        surface_name = "ダート"
        surface_type = 1

    # ★ コース回り (0:右回り, 1:左回り, 2:直線)
    if "直線" in race_data_text or "直" in race_data_text or (venue_code == 4 and surface_type == 0 and int(distance) == 1000):
        turn_type = 2
        turn_label = "直線"
    elif "左" in race_data_text or venue_code in (4, 5, 7):
        turn_type = 1
        turn_label = "左"
    else:
        turn_type = 0
        turn_label = "右"

    # ★ レースクラス・条件判定 (0:新馬, 1:未勝利, 2:1勝, 3:2勝, 4:3勝, 5:オープン/重賞)
    combined_race_text = f"{race_name} {race_data2_text}"
    race_class_code: Optional[int] = None
    class_label = ""
    if "新馬" in combined_race_text or "メイクデビュー" in combined_race_text:
        race_class_code = 0
        class_label = "新馬"
    elif "未勝利" in combined_race_text:
        race_class_code = 1
        class_label = "未勝利"
    elif "1勝" in combined_race_text or "１勝" in combined_race_text or "500万" in combined_race_text:
        race_class_code = 2
        class_label = "1勝クラス"
    elif "2勝" in combined_race_text or "２勝" in combined_race_text or "1000万" in combined_race_text:
        race_class_code = 3
        class_label = "2勝クラス"
    elif "3勝" in combined_race_text or "３勝" in combined_race_text or "1600万" in combined_race_text:
        race_class_code = 4
        class_label = "3勝クラス"
    elif any(k in combined_race_text for k in ["オープン", "ＯＰ", "OP", "G1", "G2", "G3", "GI", "GII", "GIII", "重賞", "(L)", "リステッド"]):
        race_class_code = 5
        class_label = "オープン"

    is_female_only_text = bool(re.search(r"牝(?:馬|\)|\]|限定|\s)", combined_race_text))

    condition_code = 0
    if "稍" in baba:
        condition_code = 1
    elif "不良" in baba:
        condition_code = 3
    elif "重" in baba:
        condition_code = 2

    weather_str = f" 天候:{weather}" if weather else ""
    class_str = f" [{class_label}]" if class_label else ""
    course_details = f"{surface_name}{int(distance)}m・{turn_label} ({baba}){class_str}{weather_str}"

    race_details = {
        "race_num": race_num,
        "post_time": post_time,
        "course_details": course_details
    }

    # スケジュール登録（指定日付またはJST日付を使用）
    if post_time:
        target_date = race_date if race_date else today_jst_str()
        register_race_schedule(race_id, target_date, race_name, post_time)

    odds_map = {}
    try:
        odds_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&output=json"
        oresp = safe_scrape_get(odds_url, timeout=6.0)
        raw_res = oresp.json()
        if isinstance(raw_res, str):
            raw_res = json.loads(raw_res)

        if isinstance(raw_res, dict):
            data_block = raw_res.get("data", {})
            if isinstance(data_block, str):
                data_block = json.loads(data_block)
            tansho_dict = data_block.get("odds", {}).get("1", {})

            for u_str, val in tansho_dict.items():
                if isinstance(val, list) and len(val) >= 1:
                    o_text = str(val[0]).strip()
                    if o_text and o_text != "---":
                        try:
                            o_val = float(o_text)
                            p_val = None
                            if len(val) >= 3 and str(val[2]).isdigit() and int(val[2]) > 0:
                                p_val = int(val[2])
                            elif len(val) >= 2 and str(val[1]).isdigit() and int(val[1]) > 0:
                                p_val = int(val[1])
                            odds_map[int(u_str)] = {"odds": o_val, "popularity": p_val}
                        except ValueError:
                            pass
    except Exception:
        pass

    horses = []
    rows = soup.select("tr.HorseList")

    for row in rows:
        umaban_tag = row.select_one("td.Umaban, td[class*='Umaban']")
        waku_tag = row.select_one("td.Waku, td[class*='Waku']")
        if umaban_tag and umaban_tag.get_text(strip=True).isdigit():
            umaban = int(umaban_tag.get_text(strip=True))
            wakuban = 1
            if waku_tag:
                wm = re.search(r"\d+", waku_tag.get_text(strip=True))
                if wm:
                    wakuban = int(wm.group())
        elif waku_tag and waku_tag.get_text(strip=True).isdigit():
            # SP版では td.WakuX のテキストが馬番、クラス名の数値が枠番
            umaban = int(waku_tag.get_text(strip=True))
            wakuban = 1
            w_cls = " ".join(waku_tag.get("class") or [])
            wm_cls = re.search(r"Waku(\d)", w_cls)
            if wm_cls:
                wakuban = int(wm_cls.group(1))
        else:
            continue

        name_tag = row.select_one(".HorseName a, .Horse_Info dt.Horse a, .Horse_Info a")
        if not name_tag or not name_tag.get_text(strip=True):
            raise HTTPException(status_code=502, detail=f"出馬表から馬番 {umaban} の正式な馬名を取得できませんでした (ダミー値補完禁止)")
        horse_name = name_tag.get_text(strip=True)

        jockey_tag = row.select_one(".Jockey em, .Jockey a")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"
        jockey = re.sub(r"\s*\d{2}(?:\.\d)?$", "", jockey).strip() or "未定"

        trainer_tag = row.select_one("td.Trainer, .Trainer a")
        trainer = trainer_tag.get_text(strip=True) if trainer_tag else "未定"

        # ★ 性齢 (PC版 td.Barei / SP版 dd.Age) と 斤量 (td.Barei の次の td または td.Kinryo / SP版 dd.Jockey)
        sex_age = None
        sex_code = 0
        age = 4.0
        barei_tag = row.select_one("td.Barei, dd.Age")
        if barei_tag:
            b_raw = barei_tag.get_text(" ", strip=True)
            m_sa = re.search(r"([牡牝セ騸]\d+)", b_raw)
            b_str = m_sa.group(1) if m_sa else b_raw.split()[0] if b_raw else ""
            if b_str:
                sex_age = b_str
                if "牝" in b_str:
                    sex_code = 1
                elif "セ" in b_str or "騸" in b_str:
                    sex_code = 2
                m_age = re.search(r"(\d+)", b_str)
                if m_age:
                    age = float(m_age.group(1))

        kinryo = 55.0
        kinryo_tag = (barei_tag.find_next_sibling("td") if (barei_tag and barei_tag.name == "td") else None) or row.select_one("td.Kinryo, dd.Jockey")
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        # ★ 馬体重・増減 (PC版 td.Weight / SP版 .Weight 等: 例 "464(-2)" / "480(+6)" / "472(0)")
        horse_weight: Optional[int] = None
        weight_diff: Optional[int] = None
        weight_tag = row.select_one("td.Weight, .Weight, td[class*='Weight'], span.Weight")
        w_search_text = weight_tag.get_text(strip=True) if weight_tag else row.get_text(separator=" ", strip=True)
        wm = re.search(r"\b([3-6]\d{2})(?:kg)?\s*\(([+-]?\d+|0|前計不|--)\)", w_search_text)
        if not wm and weight_tag:
            wm = re.search(r"\b([3-6]\d{2})\b", w_search_text)
        if wm:
            horse_weight = int(wm.group(1))
            diff_raw = wm.group(2) if wm.lastindex and wm.lastindex >= 2 else None
            if diff_raw and re.match(r"^[+-]?\d+$", diff_raw):
                weight_diff = int(diff_raw)
            else:
                weight_diff = 0

        odds_item = odds_map.get(umaban, {})
        odds = odds_item.get("odds")
        popularity = odds_item.get("popularity")

        # --- ブリンカー判定 (初B / B) ---
        is_blinker = False
        is_first_blinker = False
        blinker_tag = row.select_one(".Blinker, .Blink, [class*='Blink'], [class*='blinker']")
        row_raw_text = row.get_text(separator=" ", strip=True)
        if blinker_tag:
            b_txt = blinker_tag.get_text(strip=True)
            is_blinker = True
            if "初" in b_txt:
                is_first_blinker = True
        elif re.search(r"初[BＢ]|初ブリンカー", row_raw_text):
            is_blinker = True
            is_first_blinker = True
        elif re.search(r"[BＢ]|\bB\b", row_raw_text) and any(t.get_text(strip=True) in ["B", "Ｂ"] for t in row.select("span, em, font")):
            is_blinker = True

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "trainer": trainer,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity,
            "is_blinker": is_blinker,
            "is_first_blinker": is_first_blinker,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "sex_age": sex_age,
            "sex_code": sex_code,
            "age": age
        })

    course_meta = {
        "distance": distance,
        "surface_type": surface_type,
        "condition_code": condition_code,
        "turn_type": turn_type,
        "race_num_int": race_num_int,
        "race_class_code": race_class_code,
        "is_female_only_text": is_female_only_text
    }

    if not horses or len(horses) == 0:
        raise HTTPException(status_code=502, detail=f"レース {race_id} の出走馬データが0頭です。不完全なデータは処理できません。")

    return race_name, race_details, sorted(horses, key=lambda x: x["umaban"]), course_meta


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
        hybrid_rec = compute_hybrid_rec_from_inputs(
            race_name=race_name,
            race_details=race_details,
            horses=horses,
            surface_type=surface_type,
            prob_gachi=prob_gachi,
            prob_himo=prob_himo,
            opponents=opponents,
            u0=u0,
            honmei_horse=honmei_horse,
            race_date_str=r_date_str,
            force_final_odds=force_final_flag,
            real_trio_odds=trio_real_map
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
        "hybrid_recommendation": hybrid_rec
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
    real_trio_odds: Optional[dict] = None
) -> dict:
    """
    【B -> A -> C カスケード動的配分モデル】
    ・優先度 1 (戦略B: 超高確信度好配当レバレッジ):
      条件: confidence_score >= 0.78 かつ 2.5 <= w0_odds <= 10.0
      買い目: 単勝 500円 ＋ 馬単 相手3頭(AI勝率2位〜4位) 各200円 (計1,100円)
    ・優先度 2 (戦略A: 芝厳選エッジ比例ケリー傾斜):
      条件: surface_type == 0 (芝) かつ confidence_score >= 0.72
      買い目: 単複ケリー傾斜配分 (単75% + 複25%)
    ・優先度 3 (戦略C: ダート手堅い本命単複):
      条件: surface_type == 1 (ダート) かつ 春秋開催(6〜9月以外) かつ confidence_score >= 0.72 かつ 2.2 <= w0_odds <= 6.0
      買い目: 単勝 700円 ＋ 複勝 300円 (計1,000円)
    ・見送り: 上記いずれにも非該当、または頭数6頭未満、障害戦、新馬戦 (投資0円)
    """
    course_details_str = (race_details or {}).get("course_details", "") if isinstance(race_details, dict) else ""
    if "障害" in race_name or "障" in course_details_str:
        surface_type = 2
    is_shinba = ("新馬" in race_name) or ("メイクデビュー" in race_name)

    has_enough_horses = len(horses) >= 6
    is_valid_race_type = (surface_type != 2) and (not is_shinba) and has_enough_horses

    # AI勝率順ソート (Model B)
    s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    w0 = s_win[0] if s_win else {}
    w1 = s_win[1] if len(s_win) > 1 else w0

    # AI複勝率順ソート (Model B)
    s_top3 = sorted(horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
    t0 = s_top3[0] if s_top3 else {}
    t1 = s_top3[1] if len(s_top3) > 1 else t0
    t2 = s_top3[2] if len(s_top3) > 2 else t1

    w0_prob = float(w0.get("prob_win", 0.0))
    w1_prob = float(w1.get("prob_win", 0.0))
    w0_odds = float(w0.get("odds") or 50.0)
    w0_u = int(w0.get("umaban", 0))
    w0_name = str(w0.get("horse_name", f"馬{w0_u}"))

    t0_prob = float(t0.get("prob_top3", 0.0))
    t1_prob = float(t1.get("prob_top3", 0.0))
    t2_prob = float(t2.get("prob_top3", 0.0))

    win_margin = w0_prob - w1_prob
    win_edge = w0_prob * w0_odds
    top3_margin_0_1 = t0_prob - t1_prob
    top3_margin_1_2 = t1_prob - t2_prob

    confidence_score = round(
        win_margin * 2.0 +
        top3_margin_0_1 +
        top3_margin_1_2 +
        float(np.clip(win_edge - 1.0, -0.5, 0.5)),
        4
    )
    edge_ratio = round(win_edge, 2)

    # 確定オッズ・発走前判定 (Stage 5: 発走7分前〜5分前以降は確定オッズ扱い)
    is_final_odds = force_final_odds
    if not is_final_odds:
        post_time_str = race_details.get("post_time", "") if race_details else ""
        m_time = re.search(r"(\d{1,2}):(\d{2})", post_time_str)
        if m_time:
            try:
                hh, mm = int(m_time.group(1)), int(m_time.group(2))
                now_dt = now_jst_naive()
                r_date_str = race_date_str or today_jst_str()
                d_parts = [int(p) for p in r_date_str.split("-")]
                post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hh, mm)
                if now_dt >= post_dt - datetime.timedelta(minutes=7):
                    is_final_odds = True
            except Exception:
                pass

    suffix = " (確定)" if is_final_odds else " (暫定)"

    # 季節判定 (夏競馬: 6〜9月)
    try:
        month_num = int(race_date_str[5:7]) if (race_date_str and len(race_date_str) >= 7 and race_date_str[5:7].isdigit()) else now_jst().month
    except Exception:
        month_num = now_jst().month
    is_summer = 1 if 6 <= month_num <= 9 else 0

    assigned_tier = None
    tansho_bets = []
    fukusho_bets = []
    umatan_bets = []
    skip_reason = None

    if is_valid_race_type:
        # -------------------------------------------------------------
        # 優先度 1: 戦略B (超高確信度馬単レバレッジ)
        # -------------------------------------------------------------
        if confidence_score >= 0.78 and (2.5 <= w0_odds <= 10.0):
            assigned_tier = "B"
            strategy_type = "strategy_b"
            strategy_name = "🔥 戦略B: 超高確信度馬単レバレッジ (単500円＋馬単600円)"
            real_bet_badge = f"🔥【超勝負R】単勝＋馬単レバレッジ{suffix}"
            real_bet_level = "high"
            is_participate = True
            is_real_bet_target = True

            # 単勝 500円
            tansho_bets.append({
                "ticket_type": "単勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 500,
                "odds": w0_odds,
                "expected_payout": int(round(500 * w0_odds)),
                "is_real_odds": True,
                "memo": f"単勝 {w0_u}番 ({w0_name}) 500円"
            })

            # 馬単 相手3頭 各200円 (AI勝率順 2位〜4位)
            umatan_opps = [int(h["umaban"]) for h in s_win[1:4]]
            for opp in umatan_opps:
                h_opp = next((h for h in horses if int(h.get("umaban", 0)) == opp), {})
                opp_name = str(h_opp.get("horse_name", f"馬{opp}"))
                opp_odds = float(h_opp.get("odds") or 10.0)
                est_utan_odds = round(max(3.0, w0_odds * opp_odds * 0.8), 1)
                umatan_bets.append({
                    "ticket_type": "馬単",
                    "combination": [w0_u, opp],
                    "combination_str": f"{w0_u}→{opp}",
                    "amount": 200,
                    "odds": est_utan_odds,
                    "expected_payout": int(round(200 * est_utan_odds)),
                    "is_real_odds": False,
                    "memo": f"馬単 軸{w0_u}→{opp}番 ({opp_name}) 200円"
                })

        # -------------------------------------------------------------
        # 優先度 2: 戦略A (芝厳選エッジ比例ケリー傾斜)
        # -------------------------------------------------------------
        elif surface_type == 0 and confidence_score >= 0.72:
            assigned_tier = "A"
            strategy_type = "strategy_a"
            strategy_name = "⭐ 戦略A: 芝厳選エッジ投資 (単複ケリー傾斜配分)"
            real_bet_badge = f"⭐【芝厳選】単複ケリー傾斜{suffix}"
            real_bet_level = "standard"
            is_participate = True
            is_real_bet_target = True

            mult = float(np.clip(win_edge, 0.7, 1.8))
            bt = int(round(1000.0 * mult * 0.75 / 100.0) * 100)
            bf = int(round(1000.0 * mult * 0.25 / 100.0) * 100)

            if bt > 0:
                tansho_bets.append({
                    "ticket_type": "単勝",
                    "combination": [w0_u],
                    "combination_str": f"{w0_u}",
                    "amount": bt,
                    "odds": w0_odds,
                    "expected_payout": int(round(bt * w0_odds)),
                    "is_real_odds": True,
                    "memo": f"単勝(ケリー傾斜) {w0_u}番 ({w0_name}) {bt}円"
                })
            if bf > 0:
                est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
                fukusho_bets.append({
                    "ticket_type": "複勝",
                    "combination": [w0_u],
                    "combination_str": f"{w0_u}",
                    "amount": bf,
                    "odds": est_fuku_odds,
                    "expected_payout": int(round(bf * est_fuku_odds)),
                    "is_real_odds": False,
                    "memo": f"複勝(ケリー傾斜) {w0_u}番 ({w0_name}) {bf}円"
                })

        # -------------------------------------------------------------
        # 優先度 3: 戦略C (ダート手堅い本命単複)
        # -------------------------------------------------------------
        elif surface_type == 1 and is_summer == 0 and confidence_score >= 0.72 and (2.2 <= w0_odds <= 6.0):
            assigned_tier = "C"
            strategy_type = "strategy_c"
            strategy_name = "🛡️ 戦略C: ダート手堅い本命 (単700円＋複300円)"
            real_bet_badge = f"🛡️【ダート本命】手堅い単複防衛{suffix}"
            real_bet_level = "standard"
            is_participate = True
            is_real_bet_target = True

            tansho_bets.append({
                "ticket_type": "単勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 700,
                "odds": w0_odds,
                "expected_payout": int(round(700 * w0_odds)),
                "is_real_odds": True,
                "memo": f"単勝(本命厚め) {w0_u}番 ({w0_name}) 700円"
            })
            est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
            fukusho_bets.append({
                "ticket_type": "複勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 300,
                "odds": est_fuku_odds,
                "expected_payout": int(round(300 * est_fuku_odds)),
                "is_real_odds": False,
                "memo": f"複勝(防衛) {w0_u}番 ({w0_name}) 300円"
            })

        # -------------------------------------------------------------
        # 優先度 4: 戦略A-2 (芝波乱妙味拾い上げ: 単500円＋複500円)
        # 条件: 芝 かつ confidence_score >= 0.70 かつ w0_odds >= 6.0 かつ prob_gachi <= 0.10
        # -------------------------------------------------------------
        elif surface_type == 0 and confidence_score >= 0.70 and w0_odds >= 6.0 and prob_gachi <= 0.10:
            assigned_tier = "A_hole"
            strategy_type = "strategy_a_hole"
            strategy_name = "⚡ 戦略A-2: 芝波乱妙味 (単500円＋複500円)"
            real_bet_badge = f"⚡【芝波乱妙味】単複好配当狙い{suffix}"
            real_bet_level = "standard"
            is_participate = True
            is_real_bet_target = True

            tansho_bets.append({
                "ticket_type": "単勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 500,
                "odds": w0_odds,
                "expected_payout": int(round(500 * w0_odds)),
                "is_real_odds": True,
                "memo": f"単勝(妙味狙い) {w0_u}番 ({w0_name}) 500円"
            })
            est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
            fukusho_bets.append({
                "ticket_type": "複勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 500,
                "odds": est_fuku_odds,
                "expected_payout": int(round(500 * est_fuku_odds)),
                "is_real_odds": False,
                "memo": f"複勝(妙味狙い) {w0_u}番 ({w0_name}) 500円"
            })

        else:
            # カスケード全条件不一致 -> 見送り
            is_participate = False
            is_real_bet_target = False
            real_bet_level = "none"
            real_bet_badge = None
            strategy_name = "見送り推奨 (資金保全)"
            strategy_type = "skip"
            skip_reason = "カスケード条件不一致 (資金保全のため見送り)"
    else:
        # 頭数不足 / 障害戦 / 新馬戦 -> 見送り
        is_participate = False
        is_real_bet_target = False
        real_bet_level = "none"
        real_bet_badge = None
        strategy_name = "見送り推奨 (資金保全)"
        strategy_type = "skip"
        if not has_enough_horses:
            skip_reason = f"出走頭数不足 ({len(horses)}頭 < 6頭のため見送り)"
        elif surface_type == 2:
            skip_reason = "障害戦除外 (不確実性高のため見送り)"
        elif is_shinba:
            skip_reason = "新馬戦除外 (過去走データなしのため見送り)"
        else:
            skip_reason = "見送り推奨"

    tan_amt = sum(b["amount"] for b in tansho_bets)
    fuku_amt = sum(b["amount"] for b in fukusho_bets)
    utan_amt = sum(b["amount"] for b in umatan_bets)
    total_amount = tan_amt + fuku_amt + utan_amt

    if is_participate:
        if assigned_tier == "B":
            summary_str = f"{strategy_name} [単勝{w0_u} 500円 ＋ 馬単{len(umatan_bets)}点 各200円 / 計{total_amount:,}円]"
        elif assigned_tier in ("A", "A_hole"):
            summary_str = f"{strategy_name} [単勝{tan_amt:,}円 ＋ 複勝{fuku_amt:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"
        elif assigned_tier == "C":
            summary_str = f"{strategy_name} [単勝{tan_amt:,}円 ＋ 複勝{fuku_amt:,}円 (軸{w0_u} {w0_name}) / 計{total_amount:,}円]"
        reason_str = f"戦略{assigned_tier}採択"
    else:
        summary_str = f"見送り推奨 ({skip_reason})"
        reason_str = skip_reason

    # 仮想馬券 (Tier 2 観戦枠 / 単100円 + 複100円)
    virtual_bets = []
    virtual_total_amount = 0
    virtual_strategy_name = None
    virtual_badge = None

    if is_real_bet_target:
        virtual_bets = tansho_bets + fukusho_bets + umatan_bets
        virtual_total_amount = total_amount
        virtual_strategy_name = strategy_name
        virtual_badge = real_bet_badge
    elif w0_u > 0:
        est_fuku_odds = round(max(1.1, w0_odds * 0.35), 1)
        virtual_strategy_name = "🎮 仮想観戦枠: 安定単複 (単100円＋複100円)"
        virtual_badge = "🎮【仮想観戦】単複ミニマム"
        virtual_bets = [
            {
                "ticket_type": "単勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 100,
                "odds": w0_odds,
                "expected_payout": int(round(100 * w0_odds)),
                "is_real_odds": True,
                "memo": f"単勝 {w0_u} ({w0_name}) 100円"
            },
            {
                "ticket_type": "複勝",
                "combination": [w0_u],
                "combination_str": f"{w0_u}",
                "amount": 100,
                "odds": est_fuku_odds,
                "expected_payout": int(round(100 * est_fuku_odds)),
                "is_real_odds": False,
                "memo": f"複勝 {w0_u} ({w0_name}) 100円"
            }
        ]
        virtual_total_amount = 200

    hybrid_rec = {
        "is_participate": is_participate,
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "total_amount": total_amount,
        "summary": summary_str,
        "tansho_bets": tansho_bets,
        "fukusho_bets": fukusho_bets,
        "umatan_bets": umatan_bets,
        "tansho_amount": tan_amt,
        "fukusho_amount": fuku_amt,
        "umatan_amount": utan_amt,
        "umaren_bets": [],
        "sanrentan_bets": [],
        "hypothetical_umaren_bets": [],
        "trio_bets": [],
        "trio_bet": None,
        "reason": reason_str,
        "is_real_odds_applied": any(b.get("is_real_odds", False) for b in (tansho_bets + fukusho_bets + umatan_bets)),
        "total_expected_payout": sum(b.get("expected_payout", 0) for b in (tansho_bets + fukusho_bets + umatan_bets)),
        "prob_gachi": prob_gachi,
        "prob_himo": prob_himo,
        "edge_ratio": edge_ratio,
        "synth_odds": round(w0_odds, 1),
        "confidence_score": confidence_score,
        "is_real_bet_target": is_real_bet_target,
        "real_bet_badge": real_bet_badge,
        "real_bet_level": real_bet_level,
        "is_final_odds": is_final_odds,
        # 仮想馬券フィールド
        "is_virtual_bet_target": is_real_bet_target, # リアル対象は仮想枠にも自動認定
        "virtual_bet_slot": None,
        "virtual_strategy_name": virtual_strategy_name,
        "virtual_badge": virtual_badge,
        "virtual_bets": virtual_bets,
        "virtual_total_amount": virtual_total_amount
    }

    return hybrid_rec


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


def get_cached_prediction(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            d_json = res.data[0]["data_json"]
            if isinstance(d_json, dict) and "horses" in d_json:
                apply_odds_anomaly_indicators(d_json.get("horses", []))
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
    except Exception as e:
        print(f"Supabase予想保存エラー ({race_id}): {e}")


def predict_race_core(
    race_id: str,
    force_refresh: bool = False,
    is_auto_final: bool = False,
    odds_stage: Optional[str] = None
) -> dict:
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    cached_pred = get_cached_prediction(race_id)
    if not force_refresh and cached_pred:
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
                post_dt_obj = datetime.datetime.strptime(sched_post_dt_raw, "%Y-%m-%d %H:%M:%S")
                now_dt_obj = now_jst_naive()
                mins_to_post = (post_dt_obj - now_dt_obj).total_seconds() / 60.0
                if 0.0 < mins_to_post <= 65.0:
                    target_stage = odds_stage or infer_odds_stage_from_minutes(mins_to_post)
                    target_rank = ODDS_STAGE_RANK.get(target_stage, 1)
                    cooldown_sec = 60.0 if target_stage == "05m" else 180.0
                    # 新しいステージ（60m/30m/15m/05m）に到達したか、馬体重がまだ未取得の場合に自動再推論
                    if target_rank > cached_stage_rank or (not has_cached_weights and now_ts - _last_weight_check_ts.get(race_id, 0.0) >= cooldown_sec):
                        if target_rank > cached_stage_rank or (now_ts - _last_weight_check_ts.get(race_id, 0.0) >= cooldown_sec):
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
            if now_ts - _last_weight_check_ts.get(race_id, 0.0) >= 180.0:
                is_past_or_post = False
                if sched_race_date and sched_race_date < today_jst_str():
                    is_past_or_post = True
                elif sched_race_date == today_jst_str() and sched_post_dt_raw:
                    try:
                        post_dt_obj = datetime.datetime.strptime(sched_post_dt_raw, "%Y-%m-%d %H:%M:%S")
                        if now_jst_naive() >= post_dt_obj:
                            is_past_or_post = True
                    except Exception:
                        pass
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
                            supabase.table("keiba_predictions_cache").update({"data_json": cached_pred}).eq("race_id", race_id).execute()
                            print(f"[Auto-Weight Heal] {race_id}: 終了済レースの馬体重・性齢キャッシュを自動補完しました。")
                    except Exception as he:
                        print(f"[Auto-Weight Heal Warning] {race_id}: {he}")

        if not should_auto_refresh:
            return cached_pred

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
                        try:
                            scheduled_post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            pass
            except Exception:
                pass

        # 手動での予想し直し（force_refresh=True かつ 既存予想がある場合）は出走時刻前のみ許可
        if force_refresh and not is_auto_final and cached_pred is not None:
            now_dt = now_jst_naive()
            if scheduled_post_dt and now_dt >= scheduled_post_dt:
                raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")
            if not scheduled_post_dt and target_race_date:
                p_str = (cached_pred.get("race_details") or {}).get("post_time", "")
                m_t = re.search(r"(\d{1,2}):(\d{2})", p_str)
                if m_t:
                    d_parts = [int(p) for p in target_race_date.split("-")]
                    est_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], int(m_t.group(1)), int(m_t.group(2)))
                    if now_dt >= est_dt:
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
            target_race_date = today_jst_str()

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
                    m_t = re.search(r"(\d{1,2}):(\d{2})", p_str)
                    if m_t:
                        d_parts = [int(p) for p in target_race_date.split("-")]
                        est_post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], int(m_t.group(1)), int(m_t.group(2)))
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
            m_t = re.search(r"(\d{1,2}):(\d{2})", p_str)
            if m_t:
                d_parts = [int(p) for p in target_race_date.split("-")]
                est_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], int(m_t.group(1)), int(m_t.group(2)))
                if now_jst_naive() >= est_dt:
                    raise HTTPException(status_code=400, detail="出走時刻を過ぎているため、手動での予想のし直しはできません。")

        if course_meta is None:
            course_meta = {}
        course_meta["race_date"] = target_race_date
        if is_auto_final or odds_stage == "05m":
            course_meta["force_final_odds"] = True

        result = build_prediction_and_recs(race_name, race_details, race_id, horses, course_meta)
        prev_horses_list = cached_pred.get("horses", []) if isinstance(cached_pred, dict) else None
        apply_odds_anomaly_indicators(result["horses"], prev_horses=prev_horses_list, odds_stage=odds_stage)

        resp_data = {
            "race_id": race_id,
            "race_name": result["race_name"],
            "race_details": result["race_details"],
            "horses": result["horses"],
            "recommendation": result["recommendation"],
            "hybrid_recommendation": result.get("hybrid_recommendation"),
            "last_odds_stage": odds_stage
        }

        save_cached_prediction(race_id, target_race_date, resp_data)

        if supabase:
            try:
                is_final = bool((resp_data.get("hybrid_recommendation") or {}).get("is_final_odds", False))
                has_weights_now = any(h.get("horse_weight") is not None for h in result["horses"])
                if is_final or is_auto_final or odds_stage == "05m":
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

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


@app.get("/predict/all")
def get_all_predictions_for_date(date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)")):
    """指定日付の全レースのキャッシュ済み予想を一括返却（アプリの先読み・ゼロレイテンシ用）"""
    target_date = date if date else today_jst_str()
    if not supabase:
        return {"date": target_date, "count": 0, "predictions": {}}

    try:
        res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
        preds = {}
        for row in res.data or []:
            r_id = row.get("race_id")
            d_json = row.get("data_json")
            if r_id and d_json:
                if isinstance(d_json, dict) and "horses" in d_json:
                    apply_odds_anomaly_indicators(d_json.get("horses", []))
                preds[r_id] = d_json
        return {
            "date": target_date,
            "count": len(preds),
            "predictions": preds
        }
    except Exception as e:
        print(f"全予想一括取得エラー ({target_date}): {e}")
        return {"date": target_date, "count": 0, "predictions": {}}


@app.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race(race_id: str, force_refresh: bool = Query(False, description="手動で予想を再実行する場合True")):
    return predict_race_core(race_id, force_refresh=force_refresh, is_auto_final=False)



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
            top3_orders = d.get("top3_orders") or extract_top3_orders(pred_cache, orders) or rb_meta.get("top3_orders")

            # ★ 見送りレース・参戦レース問わず、1〜3着(top3_orders)が未キャッシュの場合は自動取得＆永続化 ★
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
                            supabase.table("keiba_race_settlements").update({"bet_details": b_det}).eq("race_id", race_id).execute()
                except Exception as te:
                    print(f"[Settlement Top3 Heal Warning] {race_id}: {te}")

            if d["is_pass"]:
                if pred_cache and res_cache:
                    h_rec = pred_cache.get("hybrid_recommendation") if pred_cache else None
                    pass_info = calculate_hybrid_pass_benefit_info(h_rec, res_cache, d.get("target_umaban"), order, pred_data=pred_cache)
                    hypo_bet = pass_info["hypothetical_bet"]
                    hypo_pay = pass_info["hypothetical_payout"]
                    pass_benefit = pass_info["pass_benefit"]
                    msg = pass_info["message"]
                    mark_orders = pass_info.get("mark_orders") or rb_meta.get("mark_orders")
                    top3_orders = pass_info.get("top3_orders") or top3_orders
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
            if not b_breakdown or not isinstance(b_breakdown, dict):
                b_breakdown = {}
                if u_bet > 0:
                    b_breakdown["馬連"] = u_bet
                if t_bet > 0:
                    b_breakdown["三連複"] = t_bet

            p_breakdown = d.get("payout_breakdown")
            if not p_breakdown or not isinstance(p_breakdown, dict):
                p_breakdown = {}
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
                "mark_orders": mark_orders,
                "top3_orders": top3_orders
            }
    except Exception as e:
        print(f"Supabase収支取得エラー ({race_id}): {e}")
    return None


def normalize_race_res(race_res: Optional[dict]) -> Optional[dict]:
    if not race_res or not isinstance(race_res, dict):
        return race_res
    # orders を int キーに正規化 (JSONシリアライズ時の文字列キーを整数に復元)
    raw_orders = race_res.get("orders") or {}
    norm_orders = {}
    for k, v in raw_orders.items():
        try:
            norm_orders[int(k)] = int(v)
        except (ValueError, TypeError):
            try:
                norm_orders[k] = int(v)
            except Exception:
                pass
    race_res["orders"] = norm_orders

    # payouts の tansho / fukusho のキーも int に正規化
    payouts = race_res.get("payouts") or {}
    if isinstance(payouts, dict):
        for p_type in ["tansho", "fukusho"]:
            if p_type in payouts and isinstance(payouts[p_type], dict):
                norm_p = {}
                for k, v in payouts[p_type].items():
                    try:
                        norm_p[int(k)] = int(v)
                    except (ValueError, TypeError):
                        norm_p[k] = v
                payouts[p_type] = norm_p
        race_res["payouts"] = payouts
    return race_res


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
        supabase.table("keiba_race_settlements").upsert({
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
        }).execute()
        invalidate_summary_caches()
    except Exception as e:
        print(f"Supabase収支台帳保存エラー: {e}")


def fetch_netkeiba_race_result(race_id: str) -> Optional[dict]:
    orders = {}
    payouts = {
        "tansho": {},
        "fukusho": {},
        "wakuren": {},
        "umaren": {},
        "wide": {},
        "umatan": {},
        "trio": {},
        "tierce": {}
    }
    race_name = f"Race {race_id}"
    sp_html_text = ""
    pc_html_text = ""
    sp_title = ""
    pc_title = ""

    # SP版スクレイピング試行
    sp_url = f"https://race.sp.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = safe_scrape_get(sp_url, timeout=10.0)
        html = safe_decode_html(resp)
        sp_html_text = html
        soup = BeautifulSoup(html, "html.parser")
        sp_title = soup.title.get_text(strip=True) if soup.title else ""

        if not race_name or race_name.startswith("Race "):
            r_title = soup.select_one(".RaceName, h1, .Race_Title")
            if r_title and r_title.get_text(strip=True):
                race_name = r_title.get_text(strip=True)
            elif sp_title:
                m = re.match(r"^(.+?)\s*(?:結果・払戻|\s*\|)", sp_title)
                if m:
                    race_name = m.group(1).strip()

        for tr in soup.select("table[class*='Payout'] tr, table[class*='Pay'] tr, .Payout_Detail tr"):
            th = tr.select_one("th")
            tds = tr.find_all("td")
            if th and len(tds) >= 2:
                kind = th.get_text(strip=True)
                nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", tds[0].get_text())]
                pays = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", tds[1].get_text()) if x.replace(",", "").isdigit()]
                if "単勝" in kind:
                    for n, p in zip(nums, pays):
                        payouts["tansho"][n] = p
                elif "複勝" in kind:
                    for n, p in zip(nums, pays):
                        payouts["fukusho"][n] = p
                elif "枠連" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{min(nums[0], nums[1])}-{max(nums[0], nums[1])}"
                        payouts["wakuren"][comb_key] = pays[0]
                elif "馬連" in kind and "馬単" not in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{min(nums[0], nums[1])}-{max(nums[0], nums[1])}"
                        payouts["umaren"][comb_key] = pays[0]
                elif "ワイド" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        pair_count = min(len(nums) // 2, len(pays))
                        for pi in range(pair_count):
                            w_nums = nums[pi * 2 : pi * 2 + 2]
                            w_key = f"{min(w_nums[0], w_nums[1])}-{max(w_nums[0], w_nums[1])}"
                            payouts["wide"][w_key] = pays[pi]
                elif "馬単" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{nums[0]}-{nums[1]}"
                        payouts["umatan"][comb_key] = pays[0]
                elif "3連複" in kind or "三連複" in kind:
                    if len(nums) >= 3 and len(pays) >= 1:
                        s_nums = sorted(nums[:3])
                        comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                        payouts["trio"][comb_key] = pays[0]
                elif "3連単" in kind or "三連単" in kind:
                    if len(nums) >= 3 and len(pays) >= 1:
                        comb_key = f"{nums[0]}-{nums[1]}-{nums[2]}"
                        payouts["tierce"][comb_key] = pays[0]

        for row in soup.select("tr.HorseList, .RaceResultList tr, table tr, tr"):
            o_elem = row.select_one(".Rank, td.Rank, td.Result_Num, .Result_Num")
            u_elem = row.select_one(".Umaban, td.Umaban, .umaban")
            if o_elem and u_elem:
                o_txt = o_elem.get_text(strip=True)
                u_txt = u_elem.get_text(strip=True)
                if o_txt.isdigit() and u_txt.isdigit():
                    orders[int(u_txt)] = int(o_txt)
    except Exception:
        pass

    # PC版スクレイピング (詳細・全券種確実取得)
    if not payouts["umaren"] or len(orders) < 3:
        pc_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        try:
            resp = safe_scrape_get(pc_url, timeout=10.0)
            html = safe_decode_html(resp)
            pc_html_text = html
            soup = BeautifulSoup(html, "html.parser")
            pc_title = soup.title.get_text(strip=True) if soup.title else ""

            if not race_name or race_name.startswith("Race "):
                r_title = soup.select_one(".RaceName, .Race_Title, h1")
                if r_title and r_title.get_text(strip=True):
                    race_name = r_title.get_text(strip=True)
                elif pc_title:
                    m = re.match(r"^(.+?)\s*(?:結果・払戻|\s*\|)", pc_title)
                    if m:
                        race_name = m.group(1).strip()

            for tr in soup.select("table.Payout_Detail_Table tr, table[class*='Payout'] tr"):
                th = tr.select_one("th")
                td_res = tr.select_one("td.Result")
                td_pay = tr.select_one("td.Payout")
                if th and td_res and td_pay:
                    kind = th.get_text(strip=True)
                    pays = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", td_pay.get_text()) if x.replace(",", "").isdigit()]

                    if "単勝" in kind:
                        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        for n, p in zip(nums, pays):
                            payouts["tansho"][n] = p
                    elif "複勝" in kind:
                        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        for n, p in zip(nums, pays):
                            payouts["fukusho"][n] = p
                    elif "枠連" in kind:
                        c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        if len(c_nums) >= 2 and len(pays) >= 1:
                            comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                            payouts["wakuren"][comb_key] = pays[0]
                    elif "馬連" in kind and "馬単" not in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                    payouts["umaren"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                payouts["umaren"][comb_key] = pays[0]
                    elif "ワイド" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                    payouts["wide"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                payouts["wide"][comb_key] = pays[0]
                    elif "馬単" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{c_nums[0]}-{c_nums[1]}"
                                    payouts["umatan"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{c_nums[0]}-{c_nums[1]}"
                                payouts["umatan"][comb_key] = pays[0]
                    elif "3連複" in kind or "三連複" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 3 and idx < len(pays):
                                    s_nums = sorted(c_nums[:3])
                                    comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                                    payouts["trio"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 3 and len(pays) >= 1:
                                s_nums = sorted(c_nums[:3])
                                comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                                payouts["trio"][comb_key] = pays[0]
                    elif "3連単" in kind or "三連単" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 3 and idx < len(pays):
                                    comb_key = f"{c_nums[0]}-{c_nums[1]}-{c_nums[2]}"
                                    payouts["tierce"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 3 and len(pays) >= 1:
                                comb_key = f"{c_nums[0]}-{c_nums[1]}-{c_nums[2]}"
                                payouts["tierce"][comb_key] = pays[0]

            for tr in soup.select("table.RaceTable01 tr, table.ResultTable tr, tr.HorseList"):
                o_elem = tr.select_one("td.Rank, td.Result_Num, div.Rank")
                u_elem = tr.select_one("td.Umaban, td.Txt_C, div.Umaban, td[class*='Txt_C']")
                if o_elem and u_elem:
                    o_txt = o_elem.get_text(strip=True)
                    u_txt = u_elem.get_text(strip=True)
                    if o_txt.isdigit() and u_txt.isdigit():
                        orders[int(u_txt)] = int(o_txt)
        except Exception:
            pass

    is_confirmed = (len(payouts["tansho"]) > 0) or (len(payouts["umaren"]) > 0) or (1 in orders.values())

    if len(payouts["tansho"]) > 0 and 1 not in orders.values():
        for winner_u in payouts["tansho"].keys():
            orders[winner_u] = 1

    # レース中止・取りやめ判定 (荒天・積雪・不成立等)
    all_html = sp_html_text + " " + pc_html_text
    is_cancelled = False
    if not is_confirmed:
        cancel_kws = ["開催中止", "競走取りやめ", "レース取りやめ", "開催取りやめ", "不成立", "天候調査"]
        has_cancel_kw = any(kw in all_html for kw in cancel_kws)
        empty_title = (sp_title.startswith("|") or pc_title.startswith("|") or (not sp_title and not pc_title))
        if has_cancel_kw or (empty_title and len(orders) == 0):
            is_cancelled = True

    if is_cancelled:
        return {
            "race_name": race_name,
            "orders": {},
            "payouts": payouts,
            "is_cancelled": True
        }

    if not is_confirmed:
        return None

    return normalize_race_res({
        "race_name": race_name,
        "orders": orders,
        "payouts": payouts,
        "is_cancelled": False
    })


def process_settlement_calculation(
    race_id: str,
    race_date: str,
    race_name: str,
    pred_data: dict,
    race_res: dict,
    from_cache: bool = False
) -> dict:
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
            "top3_orders": []
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
        "edge_ratio": (hybrid_rec or {}).get("edge_ratio"),
        "is_final_odds": bool((hybrid_rec or {}).get("is_final_odds", False))
    }

    # ハイブリッド推奨を最優先で照合（見送りレース）
    if not hybrid_rec or not hybrid_rec.get("is_participate", False):
        pass_info = calculate_hybrid_pass_benefit_info(hybrid_rec, race_res, target_u, actual_order, pred_data=pred_data)
        hypo_bet = pass_info["hypothetical_bet"]
        hypo_pay = pass_info["hypothetical_payout"]
        pass_benefit = pass_info["pass_benefit"]
        mark_orders = pass_info.get("mark_orders", [])
        top3_orders = pass_info.get("top3_orders", [])
        msg = pass_info["message"]

        real_bet_meta["mark_orders"] = mark_orders
        real_bet_meta["top3_orders"] = top3_orders

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
            "top3_orders": top3_orders
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

    # 参戦レースの場合
    tansho_bets = hybrid_rec.get("tansho_bets") or []
    fukusho_bets = hybrid_rec.get("fukusho_bets") or []
    umatan_bets = hybrid_rec.get("umatan_bets") or []
    umaren_bets = hybrid_rec.get("umaren_bets") or []
    sanrentan_bets = hybrid_rec.get("sanrentan_bets") or []
    trio_bets = hybrid_rec.get("trio_bets") or []
    trio_bet = hybrid_rec.get("trio_bet")
    if not trio_bets and trio_bet and isinstance(trio_bet, dict) and trio_bet.get("combination"):
        trio_bets = [trio_bet]

    payouts_data = race_res.get("payouts", {})
    all_hits = []

    # 1. 単勝照合
    tansho_payout = 0
    tansho_hits = []
    tan_map = payouts_data.get("tansho", {})
    for tb in tansho_bets:
        comb = tb.get("combination", [])
        u = comb[0] if comb else None
        if u is not None:
            unit_p = tan_map.get(u, 0) or tan_map.get(str(u), 0)
            if unit_p > 0:
                pay = int((tb.get("amount", 100) / 100.0) * unit_p)
                tansho_payout += pay
                tansho_hits.append({
                    "type": "単勝",
                    "comb": f"{u}",
                    "amount": tb.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    # 2. 複勝照合
    fukusho_payout = 0
    fukusho_hits = []
    fuku_map = payouts_data.get("fukusho", {})
    for fb in fukusho_bets:
        comb = fb.get("combination", [])
        u = comb[0] if comb else None
        if u is not None:
            unit_p = fuku_map.get(u, 0) or fuku_map.get(str(u), 0)
            if unit_p > 0:
                pay = int((fb.get("amount", 100) / 100.0) * unit_p)
                fukusho_payout += pay
                fukusho_hits.append({
                    "type": "複勝",
                    "comb": f"{u}",
                    "amount": fb.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    # 3. 馬単照合
    umatan_payout = 0
    umatan_hits = []
    utan_map = payouts_data.get("umatan", {})
    for ub in umatan_bets:
        comb = ub.get("combination", [])
        if len(comb) == 2:
            comb_key1 = f"{comb[0]}-{comb[1]}"
            comb_key2 = f"{comb[0]}→{comb[1]}"
            unit_p = utan_map.get(comb_key1, 0) or utan_map.get(comb_key2, 0)
            if unit_p > 0:
                pay = int((ub.get("amount", 100) / 100.0) * unit_p)
                umatan_payout += pay
                umatan_hits.append({
                    "type": "馬単",
                    "comb": f"{comb[0]}→{comb[1]}",
                    "amount": ub.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    # 4. 馬連照合
    umaren_payout = 0
    umaren_hits = []
    for ub in umaren_bets:
        comb = ub.get("combination", [])
        if len(comb) == 2:
            comb_key = f"{min(comb[0], comb[1])}-{max(comb[0], comb[1])}"
            unit_p = payouts_data.get("umaren", {}).get(comb_key, 0)
            if unit_p > 0:
                pay = int((ub.get("amount", 100) / 100.0) * unit_p)
                umaren_payout += pay
                umaren_hits.append({
                    "type": "馬連",
                    "comb": comb_key,
                    "amount": ub.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    # 5. 3連単照合
    sanrentan_payout = 0
    sanrentan_hits = []
    for sb in sanrentan_bets:
        comb = sb.get("combination", [])
        if len(comb) == 3:
            comb_key = f"{comb[0]}-{comb[1]}-{comb[2]}"
            unit_p = payouts_data.get("tierce", {}).get(comb_key, 0)
            if unit_p > 0:
                pay = int((sb.get("amount", 100) / 100.0) * unit_p)
                sanrentan_payout += pay
                sanrentan_hits.append({
                    "type": "3連単",
                    "comb": f"{comb[0]}→{comb[1]}→{comb[2]}",
                    "amount": sb.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    # 6. 三連複照合
    trio_payout = 0
    trio_hits = []
    for tb in trio_bets:
        if tb and tb.get("combination"):
            comb = sorted(tb.get("combination", []))
            if len(comb) == 3:
                comb_key = f"{comb[0]}-{comb[1]}-{comb[2]}"
                unit_p = payouts_data.get("trio", {}).get(comb_key, 0)
                if unit_p > 0:
                    pay = int((tb.get("amount", 100) / 100.0) * unit_p)
                    trio_payout += pay
                    trio_hits.append({
                        "type": "三連複",
                        "comb": comb_key,
                        "amount": tb.get("amount"),
                        "unit_payout": unit_p,
                        "payout": pay
                    })

    tan_bet_tot = sum(b.get("amount", 0) for b in tansho_bets)
    fuku_bet_tot = sum(b.get("amount", 0) for b in fukusho_bets)
    utan_bet_tot = sum(b.get("amount", 0) for b in umatan_bets)
    u_bet_tot = sum(b.get("amount", 0) for b in umaren_bets)
    s_bet_tot = sum(b.get("amount", 0) for b in sanrentan_bets)
    t_bet_tot = sum(b.get("amount", 0) for b in trio_bets)

    total_bet = tan_bet_tot + fuku_bet_tot + utan_bet_tot + u_bet_tot + s_bet_tot + t_bet_tot
    total_payout = tansho_payout + fukusho_payout + umatan_payout + umaren_payout + sanrentan_payout + trio_payout
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    all_hits = tansho_hits + fukusho_hits + umatan_hits + umaren_hits + sanrentan_hits + trio_hits
    if len(all_hits) >= 2:
        hit_type = "W的中🔥"
    elif len(umatan_hits) > 0:
        hit_type = "馬単的中🎯"
    elif len(tansho_hits) > 0:
        hit_type = "単勝的中🎯"
    elif len(fukusho_hits) > 0:
        hit_type = "複勝的中🎯"
    elif len(sanrentan_hits) > 0:
        hit_type = "3連単的中🎯"
    elif len(trio_hits) > 0:
        hit_type = "三連複的中🎯"
    elif len(umaren_hits) > 0:
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
        "top3_orders": top3_orders
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

    # 2026年4回東京(05)・京都(08)開催の確定日程マッピング
    if len(race_id) == 12 and race_id.startswith("2026"):
        day_code = race_id[6:10]
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
        elif day_code == "0408":
            return "2026-09-26"
        elif day_code == "0409":
            return "2026-09-27"

    if supabase:
        for tbl in ("keiba_scheduled_races", "keiba_predictions_cache", "keiba_race_settlements"):
            try:
                res = supabase.table(tbl).select("race_date").eq("race_id", race_id).execute()
                if res.data and len(res.data) > 0 and res.data[0].get("race_date"):
                    r_date = res.data[0]["race_date"]
                    if r_date and r_date != today_jst_str():
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


@app.get("/result/{race_id}", response_model=VerificationResult)
def verify_race_result(
    race_id: str,
    bypass_cache: bool = Query(False, description="キャッシュをバイパスして最新結果を取得"),
    is_19h_settle: bool = Query(False, description="19時の一括確定モード")
):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    if not bypass_cache:
        cached_settlement = get_cached_settlement(race_id)
        if cached_settlement and (cached_settlement.get("actual_order") is not None or cached_settlement.get("is_pass")):
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
        if c_has_top3 and c_has_pay:
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
    has_target_order = (actual_order is not None and isinstance(actual_order, int) and 1 <= actual_order <= 30)

    # 1〜3着確定および払戻金確定があれば、JRA公式結果としてレース全体が確定している
    # （◎馬が競走中止・除外・取消・大敗等の場合でも、レース照合は正常完了させる）
    is_fully_confirmed = has_top3 and has_payouts
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


@app.get("/summary/dates", response_model=List[str])
@app.get("/daily-summary/dates", response_model=List[str])
def get_summary_dates(force_refresh: bool = False):
    """過去の予想・集計可能日付一覧を取得 (降順・キャッシュ付き)"""
    global _summary_dates_cache
    now = time.time()
    if not force_refresh and _summary_dates_cache["dates"] and (now - _summary_dates_cache["updated_at"] < CACHE_TTL_SUMMARY_DATES):
        return _summary_dates_cache["dates"]

    today_str = today_jst_str()
    try:
        dates_set = set(discover_period_kaisai_dates())
        if supabase:
            # 1. 確定・集計済みレコードがある日付（永続保存されている全過去日）
            try:
                page = 0
                while True:
                    res_settle = supabase.table("keiba_race_settlements").select("race_date").range(page * 1000, (page + 1) * 1000 - 1).execute()
                    batch = res_settle.data or []
                    for row in batch:
                        d = row.get("race_date")
                        if d and len(d) == 10 and d.startswith("20"):
                            dates_set.add(d)
                    if len(batch) < 1000:
                        break
                    page += 1
            except Exception as e:
                print(f"settlements dates error: {e}")

            # 2. レース開催日として登録された日付
            try:
                page = 0
                while True:
                    res_sched = supabase.table("keiba_scheduled_races").select("race_date").range(page * 1000, (page + 1) * 1000 - 1).execute()
                    batch = res_sched.data or []
                    for row in batch:
                        d = row.get("race_date")
                        if d and len(d) == 10 and d.startswith("20"):
                            dates_set.add(d)
                    if len(batch) < 1000:
                        break
                    page += 1
            except Exception as e:
                print(f"scheduled dates error: {e}")

        sorted_dates = sorted(list(dates_set), reverse=True)
        res = sorted_dates if sorted_dates else [today_str]
        _summary_dates_cache["dates"] = res
        _summary_dates_cache["updated_at"] = now
        return res
    except Exception as e:
        print(f"日付一覧取得エラー: {e}")
        return [today_str]


@app.get("/summary/today", response_model=DailySummary)
def get_daily_summary(date: Optional[str] = Query(None, description="集計対象日付 (YYYY-MM-DD)")):
    target_date = date if date else today_jst_str()

    rows = []
    if supabase:
        try:
            res = supabase.table("keiba_race_settlements").select("*").eq("race_date", target_date).order("race_id", desc=False).execute()
            rows = res.data or []
        except Exception as e:
            print(f"Supabaseサマリー取得エラー: {e}")

    # 結果キャッシュと予想キャッシュを一括取得（金曜20:00破棄後は空でも keiba_race_settlements から完全復元）
    res_cache_map = {}
    pred_cache_map = {}
    if supabase and rows:
        try:
            res_cache = supabase.table("keiba_race_results_cache").select("race_id, result_json").eq("race_date", target_date).execute()
            res_cache_map = {item["race_id"]: item.get("result_json", {}) for item in (res_cache.data or [])}
            pred_cache = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", target_date).execute()
            pred_cache_map = {item["race_id"]: item.get("data_json", {}) for item in (pred_cache.data or [])}
        except Exception as e:
            print(f"Supabaseサマリー用キャッシュ取得エラー: {e}")

    pass_benefit_total = 0
    pass_avoided_losses = 0
    pass_missed_profits = 0

    for r in rows:
        p_data = pred_cache_map.get(r["race_id"], {})
        h_rec = p_data.get("hybrid_recommendation") or {}
        b_det = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
        rb_meta = b_det.get("real_bet_meta") or {}

        # 実戦推奨情報を付与（キャッシュ破棄後も bet_details.real_bet_meta から復元）
        r["is_real_bet_target"] = h_rec.get("is_real_bet_target", rb_meta.get("is_real_bet_target", False))
        badge_val = h_rec.get("real_bet_badge") or rb_meta.get("real_bet_badge")
        if badge_val and isinstance(badge_val, str):
            badge_val = badge_val.replace("🎯", "⭐")
        r["real_bet_badge"] = badge_val
        if r.get("strategy_name") and isinstance(r["strategy_name"], str):
            r["strategy_name"] = r["strategy_name"].replace("🎯", "⭐")
        r["real_bet_level"] = h_rec.get("real_bet_level") or rb_meta.get("real_bet_level", "none")
        r["edge_ratio"] = h_rec.get("edge_ratio") if h_rec.get("edge_ratio") is not None else rb_meta.get("edge_ratio")
        r["is_final_odds"] = h_rec.get("is_final_odds", rb_meta.get("is_final_odds", False))

        if r.get("strategy_type") == "cancelled" or r.get("hit_type") == "開催中止":
            r["is_real_bet_target"] = False
            r["real_bet_badge"] = "⛔【開催中止】全額返還"
            r["real_bet_level"] = "none"
            r["hypothetical_bet"] = 0
            r["hypothetical_payout"] = 0
            r["pass_benefit"] = 0
            r["mark_orders"] = []
            r["top3_orders"] = []
        elif r.get("is_pass") == 1:
            c_res = res_cache_map.get(r["race_id"], {})
            if p_data and c_res:
                pass_info = calculate_hybrid_pass_benefit_info(h_rec, c_res, r.get("target_umaban"), r.get("actual_order"), pred_data=p_data)
                pb = pass_info["pass_benefit"]
                r["hypothetical_bet"] = pass_info["hypothetical_bet"]
                r["hypothetical_payout"] = pass_info["hypothetical_payout"]
                r["pass_benefit"] = pb
                r["mark_orders"] = pass_info.get("mark_orders") or rb_meta.get("mark_orders")
                r["top3_orders"] = pass_info.get("top3_orders") or rb_meta.get("top3_orders")
            else:
                pb = r.get("pass_benefit") or 0
                r["hypothetical_bet"] = r.get("hypothetical_bet") or 0
                r["hypothetical_payout"] = r.get("hypothetical_payout") or 0
                r["pass_benefit"] = pb
                r["mark_orders"] = r.get("mark_orders") or rb_meta.get("mark_orders")
                r["top3_orders"] = r.get("top3_orders") or rb_meta.get("top3_orders")
            pass_benefit_total += pb
            if pb > 0:
                pass_avoided_losses += pb
            elif pb < 0:
                pass_missed_profits += abs(pb)
        else:
            c_res = res_cache_map.get(r["race_id"], {})
            if not r.get("mark_orders"):
                r["mark_orders"] = extract_mark_orders(p_data, c_res.get("orders", {})) if (p_data and c_res) else rb_meta.get("mark_orders")

        # mark_orders を Android (Gson) の期待する List[dict] に正規化
        if isinstance(r.get("mark_orders"), dict):
            converted = []
            h_mark_map = {h.get("mark"): h for h in (p_data.get("horses", []) if p_data else [])}
            for m_key, o_val in r["mark_orders"].items():
                matched_h = h_mark_map.get(m_key, {})
                ord_int = int(o_val) if (o_val is not None and str(o_val).isdigit()) else None
                converted.append({
                    "mark": str(m_key),
                    "umaban": matched_h.get("umaban"),
                    "horse_name": matched_h.get("horse_name", f"馬{matched_h.get('umaban', '')}"),
                    "order": ord_int,
                    "order_text": f"{ord_int}着" if ord_int is not None else "着外",
                    "popularity": matched_h.get("popularity")
                })
            r["mark_orders"] = converted
        elif not isinstance(r.get("mark_orders"), list):
            r["mark_orders"] = []

        # top3_orders も List[Top3OrderInfo] (List[dict]) に正規化
        if isinstance(r.get("top3_orders"), list) and r["top3_orders"] and not isinstance(r["top3_orders"][0], dict):
            r["top3_orders"] = []
        elif not isinstance(r.get("top3_orders"), list):
            r["top3_orders"] = []

        # 本命馬の人気 (popularity) および mark_orders の人気情報を補完
        h_horses = p_data.get("horses", []) if p_data else []
        h_pop_map = {int(h["umaban"]): h.get("popularity") for h in h_horses if "umaban" in h and h.get("popularity") is not None}
        target_u = r.get("target_umaban")
        if target_u and int(target_u) in h_pop_map:
            r["target_popularity"] = h_pop_map[int(target_u)]
        else:
            h0 = next((h for h in h_horses if h.get("mark") == "◎"), None)
            if h0 and h0.get("popularity") is not None:
                r["target_popularity"] = h0.get("popularity")

        if r.get("mark_orders") and h_pop_map and isinstance(r["mark_orders"], list):
            for mo in r["mark_orders"]:
                if isinstance(mo, dict) and mo.get("popularity") is None and mo.get("umaban") in h_pop_map:
                    mo["popularity"] = h_pop_map[mo["umaban"]]
        # Android (Gson) は bet_breakdown / payout_breakdown に Map<String, Int> を要求するため、
        # 文字列等の不正形式の場合は必ず dict に正規化してクラッシュを防ぐ
        if not isinstance(r.get("bet_breakdown"), dict):
            b_dict = {}
            if (r.get("umaren_bet") or 0) > 0: b_dict["馬連"] = r["umaren_bet"]
            if (r.get("trio_bet") or 0) > 0: b_dict["三連複"] = r["trio_bet"]
            r["bet_breakdown"] = b_dict

        if not isinstance(r.get("payout_breakdown"), dict):
            p_dict = {}
            if (r.get("umaren_payout") or 0) > 0: p_dict["馬連"] = r["umaren_payout"]
            if (r.get("trio_payout") or 0) > 0: p_dict["三連複"] = r["trio_payout"]
            r["payout_breakdown"] = p_dict

        r_id_str = str(r.get("race_id", ""))
        if len(r_id_str) >= 12 and not r.get("venue_name"):
            try:
                v_code = int(r_id_str[4:6])
                r["venue_name"] = GLOBAL_VENUE_MAP.get(v_code, "JRA")
            except Exception:
                pass
        if len(r_id_str) >= 12 and not r.get("race_num"):
            try:
                r_num_int = int(r_id_str[10:12])
                r["race_num"] = f"{r_num_int}R"
            except Exception:
                pass

        # 中止レース正式名称の補完
        if r_id_str in CANCELLED_RACE_NAMES:
            r["race_name"] = CANCELLED_RACE_NAMES[r_id_str]

        # 収支・回収率の算術整合性を厳密保証
        r_bet = r.get("total_bet") or 0
        r_payout = r.get("total_payout") or 0
        if r.get("strategy_type") == "cancelled" or r.get("hit_type") == "開催中止":
            r["total_bet"] = 0
            r["total_payout"] = 0
            r["profit"] = 0
            r["recovery_rate"] = 0.0
            r["is_pass"] = 1
            r["is_hit"] = 0
        else:
            r["profit"] = r_payout - r_bet
            r["recovery_rate"] = round((r_payout / r_bet) * 100.0, 1) if r_bet > 0 else 0.0

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r.get("is_pass") == 0)
    pass_races = sum(1 for r in rows if r.get("is_pass") == 1)
    hit_races = sum(1 for r in rows if r.get("is_hit") == 1 and r.get("is_pass") == 0)

    total_bet = sum((r.get("total_bet") or 0) for r in rows)
    total_payout = sum((r.get("total_payout") or 0) for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

    # 実戦推奨レースのみの収支集計
    real_bet_rows = [r for r in rows if r.get("is_real_bet_target") and r.get("is_pass") == 0]
    real_bet_races = len(real_bet_rows)
    real_bet_hits = sum(1 for r in real_bet_rows if r.get("is_hit") == 1)
    real_bet_hit_rate = round((real_bet_hits / real_bet_races) * 100.0, 2) if real_bet_races > 0 else 0.0
    real_bet_amount = sum((r.get("total_bet") or 0) for r in real_bet_rows)
    real_bet_payout = sum((r.get("total_payout") or 0) for r in real_bet_rows)
    real_bet_profit = real_bet_payout - real_bet_amount
    real_bet_recovery_rate = round((real_bet_payout / real_bet_amount) * 100.0, 2) if real_bet_amount > 0 else 0.0

    # 汎用券種別内訳集計 (動的)
    ticket_stats = {}
    for r in rows:
        if r.get("is_pass") == 0:
            b_map = r.get("bet_breakdown")
            if not b_map or not isinstance(b_map, dict):
                b_map = {}
                if (r.get("umaren_bet") or 0) > 0:
                    b_map["馬連"] = r["umaren_bet"]
                if (r.get("trio_bet") or 0) > 0:
                    b_map["三連複"] = r["trio_bet"]

            p_map = r.get("payout_breakdown")
            if not p_map or not isinstance(p_map, dict):
                p_map = {}
                if (r.get("umaren_payout") or 0) > 0:
                    p_map["馬連"] = r["umaren_payout"]
                if (r.get("trio_payout") or 0) > 0:
                    p_map["三連複"] = r["trio_payout"]

            for t_type, b_amt in b_map.items():
                if b_amt > 0:
                    st = ticket_stats.setdefault(t_type, {
                        "ticket_type": t_type,
                        "bet_amount": 0,
                        "payout_amount": 0,
                        "hit_count": 0,
                        "bet_count": 0,
                        "recovery_rate": 0.0
                    })
                    st["bet_amount"] += b_amt
                    st["bet_count"] += 1
                    p_amt = p_map.get(t_type, 0)
                    st["payout_amount"] += p_amt
                    if p_amt > 0:
                        st["hit_count"] += 1

    ticket_breakdowns = []
    for t_type, st in ticket_stats.items():
        st["recovery_rate"] = round((st["payout_amount"] / st["bet_amount"]) * 100.0, 2) if st["bet_amount"] > 0 else 0.0
        ticket_breakdowns.append(st)

    # ハイブリッド内訳集計 (後方互換用)
    umaren_hit_races = sum(1 for r in rows if (r.get("umaren_payout") or 0) > 0 and r.get("is_pass") == 0)
    trio_hit_races = sum(1 for r in rows if (r.get("trio_payout") or 0) > 0 and r.get("is_pass") == 0)
    umaren_bet_total = sum((r.get("umaren_bet") or 0) for r in rows if r.get("is_pass") == 0)
    trio_bet_total = sum((r.get("trio_bet") or 0) for r in rows if r.get("is_pass") == 0)
    umaren_payout_total = sum((r.get("umaren_payout") or 0) for r in rows if r.get("is_pass") == 0)
    trio_payout_total = sum((r.get("trio_payout") or 0) for r in rows if r.get("is_pass") == 0)
    umaren_rec = round((umaren_payout_total / umaren_bet_total) * 100.0, 2) if umaren_bet_total > 0 else 0.0
    trio_rec = round((trio_payout_total / trio_bet_total) * 100.0, 2) if trio_bet_total > 0 else 0.0

    # 累計（生涯・年間・月間）データは別エンドポイント /summary/aggregates でキャッシュ集計するため、
    # 日別サマリーでは当日データのみを最速(約0.2秒)で返却し、後方互換フィールドは0とする
    cum_days = 0
    cum_total_races = 0
    cum_bet_races = 0
    cum_pass_races = 0
    cum_hit_races = 0
    cum_hit_rate = 0.0
    cum_bet_amount = 0
    cum_payout_amount = 0
    cum_profit = 0
    cum_recovery_rate = 0.0
    cum_pass_benefit = 0
    cum_rb_races = 0
    cum_rb_hits = 0
    cum_rb_hit_rate = 0.0
    cum_rb_amount = 0
    cum_rb_payout = 0
    cum_rb_profit = 0
    cum_rb_recovery_rate = 0.0

    return {
        "target_date": target_date,
        "total_races": total_races,
        "bet_races": bet_races,
        "pass_races": pass_races,
        "hit_races": hit_races,
        "hit_rate": hit_rate,
        "total_bet_amount": total_bet,
        "total_payout_amount": total_payout,
        "total_profit": total_profit,
        "recovery_rate": recovery_rate,
        "settled_races": rows,
        "pass_benefit_total": pass_benefit_total,
        "pass_avoided_losses": pass_avoided_losses,
        "pass_missed_profits": pass_missed_profits,
        "ticket_breakdowns": ticket_breakdowns,
        "umaren_hit_races": umaren_hit_races,
        "trio_hit_races": trio_hit_races,
        "umaren_bet_total": umaren_bet_total,
        "trio_bet_total": trio_bet_total,
        "umaren_payout_total": umaren_payout_total,
        "trio_payout_total": trio_payout_total,
        "umaren_recovery_rate": umaren_rec,
        "trio_recovery_rate": trio_rec,
        "real_bet_races": real_bet_races,
        "real_bet_hits": real_bet_hits,
        "real_bet_hit_rate": real_bet_hit_rate,
        "real_bet_amount": real_bet_amount,
        "real_bet_payout": real_bet_payout,
        "real_bet_profit": real_bet_profit,
        "real_bet_recovery_rate": real_bet_recovery_rate,
        "cumulative_days": cum_days,
        "cumulative_total_races": cum_total_races,
        "cumulative_bet_races": cum_bet_races,
        "cumulative_pass_races": cum_pass_races,
        "cumulative_hit_races": cum_hit_races,
        "cumulative_hit_rate": cum_hit_rate,
        "cumulative_bet_amount": cum_bet_amount,
        "cumulative_payout_amount": cum_payout_amount,
        "cumulative_profit": cum_profit,
        "cumulative_recovery_rate": cum_recovery_rate,
        "cumulative_pass_benefit": cum_pass_benefit,
        "cumulative_real_bet_races": cum_rb_races,
        "cumulative_real_bet_hits": cum_rb_hits,
        "cumulative_real_bet_hit_rate": cum_rb_hit_rate,
        "cumulative_real_bet_amount": cum_rb_amount,
        "cumulative_real_bet_payout": cum_rb_payout,
        "cumulative_real_bet_profit": cum_rb_profit,
        "cumulative_real_bet_recovery_rate": cum_rb_recovery_rate
    }


def calc_period_stat(title: str, key: str, rows: list) -> dict:
    """期間ごとの収支・回収率・実戦推奨成績を集計"""
    days = len(set(r.get("race_date") for r in rows if r.get("race_date")))
    total_races = len(rows)
    bet_races = sum(1 for r in rows if r.get("is_pass") == 0)
    pass_races = sum(1 for r in rows if r.get("is_pass") == 1)
    hit_races = sum(1 for r in rows if r.get("is_hit") == 1 and r.get("is_pass") == 0)
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0
    total_bet = sum((r.get("total_bet") or 0) for r in rows)
    total_payout = sum((r.get("total_payout") or 0) for r in rows)
    total_profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    pass_benefit = sum((r.get("pass_benefit") or 0) for r in rows if r.get("is_pass") == 1)

    rb_races = 0
    rb_hits = 0
    rb_amount = 0
    rb_payout = 0
    for r in rows:
        if r.get("is_pass") == 0:
            bd = r.get("bet_details") if isinstance(r.get("bet_details"), dict) else {}
            rbm = bd.get("real_bet_meta") or {}
            if rbm.get("is_real_bet_target"):
                rb_races += 1
                if r.get("is_hit") == 1:
                    rb_hits += 1
                rb_amount += (r.get("total_bet") or 0)
                rb_payout += (r.get("total_payout") or 0)
    rb_profit = rb_payout - rb_amount
    rb_recovery = round((rb_payout / rb_amount) * 100.0, 2) if rb_amount > 0 else 0.0
    rb_hit_rate = round((rb_hits / rb_races) * 100.0, 2) if rb_races > 0 else 0.0

    return {
        "title": title,
        "period_key": key,
        "days": days,
        "total_races": total_races,
        "bet_races": bet_races,
        "pass_races": pass_races,
        "hit_races": hit_races,
        "hit_rate": hit_rate,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "total_profit": total_profit,
        "recovery_rate": recovery_rate,
        "real_bet_races": rb_races,
        "real_bet_hits": rb_hits,
        "real_bet_hit_rate": rb_hit_rate,
        "real_bet_amount": rb_amount,
        "real_bet_payout": rb_payout,
        "real_bet_profit": rb_profit,
        "real_bet_recovery_rate": rb_recovery,
        "pass_benefit": pass_benefit
    }


@app.get("/summary/aggregates", response_model=AggregateSummaryResponse)
def get_aggregate_summary(force_refresh: bool = False):
    """生涯・年間・月間の総合収支サマリーを取得（キャッシュ付き）"""
    global _aggregate_summary_cache
    now = time.time()
    if not force_refresh and _aggregate_summary_cache["data"] and (now - _aggregate_summary_cache["updated_at"] < CACHE_TTL_AGGREGATE):
        return _aggregate_summary_cache["data"]

    cum_rows = []
    if supabase:
        try:
            page = 0
            page_size = 1000
            while True:
                cum_res = supabase.table("keiba_race_settlements").select(
                    "race_date, is_pass, is_hit, total_bet, total_payout, pass_benefit, bet_details"
                ).range(page * page_size, (page + 1) * page_size - 1).execute()
                batch = cum_res.data or []
                cum_rows.extend(batch)
                if len(batch) < page_size:
                    break
                page += 1
        except Exception as e:
            print(f"Supabase総合集計データ取得エラー: {e}")

    # 1. 生涯（全期間）
    lifetime = calc_period_stat("生涯（全期間）", "lifetime", cum_rows)

    # 2. 年別 & 月別グループ分け
    year_map = defaultdict(list)
    month_map = defaultdict(list)
    for r in cum_rows:
        rd = r.get("race_date") or ""
        if len(rd) >= 4:
            year_map[rd[:4]].append(r)
        if len(rd) >= 7:
            month_map[rd[:7]].append(r)

    yearly = []
    for y in sorted(year_map.keys(), reverse=True):
        yearly.append(calc_period_stat(f"{y}年", y, year_map[y]))

    monthly = []
    for ym in sorted(month_map.keys(), reverse=True):
        y_part = ym[:4]
        m_part = int(ym[5:7]) if ym[5:7].isdigit() else ym[5:7]
        monthly.append(calc_period_stat(f"{y_part}年{m_part}月", ym, month_map[ym]))

    result = {
        "lifetime": lifetime,
        "yearly": yearly,
        "monthly": monthly
    }
    _aggregate_summary_cache["data"] = result
    _aggregate_summary_cache["updated_at"] = now
    return result


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

        if p_info["is_waiting_period"] and not target_date:
            batch_status["total"] = 0
            batch_status["message"] = "現在は次週の開催データ取得待機期間（金曜20:00〜土曜朝7:30）です。"
            return

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


def start_batch_thread(target_date: Optional[str] = None, mode: str = "all", force_predict: bool = False) -> BatchStartResponse:
    global batch_status
    if batch_status["is_running"]:
        return {
            "status": "already_running",
            "message": f"現在処理中です ({batch_status['current']}/{batch_status['total']}レース進行中)"
        }

    thread = threading.Thread(
        target=run_batch_worker,
        args=(target_date, mode, force_predict),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "message": f"バックグラウンドで今開催期間の未処理レース一括処理を開始しました (モード: {mode})。"
    }


@app.post("/batch/today", response_model=BatchStartResponse)
def run_today_batch(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD、未指定時は開催期間全体)"),
    mode: str = Query("all", description="処理モード: all (予想＆照合), predict (予想のみ), settle (照合のみ)"),
    force: bool = Query(False, description="強制再予想フラグ")
):
    return start_batch_thread(date, mode=mode, force_predict=force)


@app.post("/batch/predict", response_model=BatchStartResponse)
def run_predict_batch(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)"),
    force: bool = Query(False, description="強制再予想フラグ")
):
    return start_batch_thread(date, mode="predict", force_predict=force)


@app.get("/batch/status", response_model=BatchStatusResponse)
def get_batch_status():
    return batch_status


@app.post("/batch/recalculate-period")
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
        if valid_rids and rid not in valid_rids:
            try:
                supabase.table("keiba_predictions_cache").delete().eq("race_id", rid).execute()
                supabase.table("keiba_scheduled_races").delete().eq("race_id", rid).execute()
            except Exception:
                pass
            continue
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

            # --- ⑥ 金曜 20:00 以降（待機期間突入時）の1開催期間キャッシュ破棄 ---
            if p_info["is_waiting_period"]:
                if last_friday_cleaned_period != p_info["period_start"]:
                    clear_kaisai_period_caches()
                    last_friday_cleaned_period = p_info["period_start"]
                    last_830_predict_date = None
                    last_17h_settle_date = None
                    last_19h_settle_date = None
                # 待機期間中（金曜20:00〜土曜朝7:30未満）は無理に出走表を取得しに行かない
                await asyncio.sleep(60)
                continue

            # --- ① JST朝 7:30 以降、当日のレーススケジュールを事前登録 ---
            if now_dt.hour > 7 or (now_dt.hour == 7 and now_dt.minute >= 30):
                await sync_daily_schedules_if_needed(today_str)

            # --- ② JST朝 8:30 以降、当日の未予想レース事前予想（朝オッズによる暫定判定）を一括実行 ---
            if (now_dt.hour > 8 or (now_dt.hour == 8 and now_dt.minute >= 30)) and last_830_predict_date != today_str:
                period_dates = discover_period_kaisai_dates(now_dt)
                if today_str in period_dates:
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 朝8:30 当日({today_str})全レースの一括事前予想（暫定判定）を実行します...")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "predict", False, False))
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
                        p_dt_str = r["post_datetime"]
                        cur_stage = int(r.get("is_predicted") or 0)
                        try:
                            post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue

                        mins_to_post = (post_dt - now_dt).total_seconds() / 60.0
                        if mins_to_post <= 0 or mins_to_post > 65.0:
                            continue

                        # Stage 5: 発走 5分前 (0 < mins <= 6.5) -> 最終オッズ推論 ＆ 確定通知
                        if mins_to_post <= 6.5 and cur_stage < 5:
                            print(f"[Auto-Predict Stage5: 5分前最終確定] {r_id} ({r.get('race_name')}) の最終推論＆買い目確定を実行します...")
                            try:
                                pred_res = predict_race_core(r_id, force_refresh=True, is_auto_final=True, odds_stage="05m")
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

                                    all_tickets = (h_rec.get("tansho_bets") or []) + (h_rec.get("fukusho_bets") or []) + (h_rec.get("umatan_bets") or [])
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
                            except Exception as pe:
                                print(f"[Auto-Predict Stage5] 最終推論失敗 ({r_id}): {pe}")

                        # Stage 4: 発走 15分前 (6.5 < mins <= 17.0) -> 本馬場入場後オッズ更新
                        elif 6.5 < mins_to_post <= 17.0 and cur_stage < 4:
                            print(f"[Auto-Predict Stage4: 15分前直前更新] {r_id} ({r.get('race_name')}) の15分前オッズ更新を実行します...")
                            try:
                                predict_race_core(r_id, force_refresh=True, is_auto_final=False, odds_stage="15m")
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
                                predict_race_core(r_id, force_refresh=True, is_auto_final=False, odds_stage="30m")
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 3})\
                                    .eq("race_id", r_id)\
                                    .execute()
                            except Exception as pe:
                                print(f"[Auto-Predict Stage3] 更新失敗 ({r_id}): {pe}")

                        # Stage 2: 発走 60分前 (33.0 < mins <= 65.0) -> 馬体重反映・第1回再計算
                        elif 33.0 < mins_to_post <= 65.0 and cur_stage < 2:
                            try:
                                predict_race_core(r_id, force_refresh=False, is_auto_final=False, odds_stage="60m")
                            except Exception as we:
                                print(f"[Auto-Predict Stage2] 馬体重更新チェック失敗 ({r_id}): {we}")
                except Exception as e:
                    print(f"[Scheduler] 予想ループエラー: {e}")

                # --- ④ 各レース【発走 15分後】の個別自動照合（未確定時は5分おきにリトライ） ---
                try:
                    res_settle = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("is_settled", 0)\
                        .execute()
                    unsettled_races = res_settle.data or []

                    for r in unsettled_races:
                        r_id = r["race_id"]
                        p_dt_str = r["post_datetime"]
                        retries = r.get("retry_count", 0) or 0
                        try:
                            post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue

                        # 初回は発走15分後、以降は5分間隔でリトライ
                        trigger_settle_time = post_dt + datetime.timedelta(minutes=15 + 5 * retries)
                        if now_dt >= trigger_settle_time:
                            try:
                                verify_race_result(r_id)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_settled": 1})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Settle] 結果照合完了: {r_id}")
                            except Exception:
                                if retries >= 12:
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

            if now_dt.hour == 19 and last_19h_settle_date != today_str:
                if today_str in discover_period_kaisai_dates(now_dt):
                    if not batch_status["is_running"]:
                        print(f"[Scheduler] 19:00 当日レースの予備最終結果照合を実行します...")
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "settle", False, True))
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


def fix_cancelled_races_2026():
    """2026年の中止レース（2/7 東京8-12R、2/8 小倉4R等）を正規の開催中止（返還・投資0/払戻0/不参戦）に是正"""
    if not supabase:
        print("[FixCancelled] Supabase未接続")
        return {"status": "skipped", "message": "Supabase not connected"}

    target_race_ids = [
        "202605010308",  # 2026-02-07 東京8R (積雪中止)
        "202605010309",  # 2026-02-07 東京9R (積雪中止)
        "202605010310",  # 2026-02-07 東京10R (積雪中止)
        "202605010311",  # 2026-02-07 東京11R (積雪中止)
        "202605010312",  # 2026-02-07 東京12R (積雪中止)
        "202610010604",  # 2026-02-08 小倉4R (中止)
    ]

    updated_count = 0
    try:
        for r_id in target_race_ids:
            row_data = {
                "strategy_type": "cancelled",
                "target_umaban": None,
                "horse_name": None,
                "grade": "C",
                "is_pass": 1,
                "umaren_bet": 0,
                "trio_bet": 0,
                "total_bet": 0,
                "actual_order": None,
                "umaren_payout": 0,
                "trio_payout": 0,
                "total_payout": 0,
                "is_hit": 0,
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
                "fukusho_payout": 0
            }
            supabase.table("keiba_race_settlements").update(row_data).eq("race_id", r_id).execute()
            updated_count += 1
            print(f"[FixCancelled] {r_id} を開催中止（全額返還）に更新しました。")

        # サマリー集計キャッシュを無効化・再計算
        invalidate_summary_caches()
        msg = f"中止レース是正完了: {updated_count}件を全額返還（投資0/払戻0）に修正し、集計キャッシュを無効化しました。"
        print(f"[FixCancelled] {msg}")
        return {"status": "ok", "message": msg, "updated_count": updated_count}
    except Exception as e:
        err_msg = f"[FixCancelled Error] {e}"
        print(err_msg)
        return {"status": "error", "message": err_msg}


@app.get("/admin/fix-cancelled-races")
@app.post("/admin/fix-cancelled-races")
def endpoint_fix_cancelled_races():
    return fix_cancelled_races_2026()


def normalize_mislabeled_dates():
    """2026-10-13等のフォールバック日で誤保存された過去レースを正規の開催日 (2026-10-03, 2026-10-04) へ修正"""
    if not supabase:
        print("[NormalizeDates] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}

    migrated_settlements = 0
    migrated_results = 0
    migrated_predictions = 0

    try:
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


@app.get("/admin/stats-dict-info")
def endpoint_stats_dict_info():
    """統計辞書（stats_dict）および永続化オーバーレイ（stats_overlay）のステータス情報"""
    global stats_dict, _stats_overlay
    h_dict = stats_dict.get("horse_stats", {})
    j_dict = stats_dict.get("jockey_stats", {})
    t_dict = stats_dict.get("trainer_stats", {})
    overlay_horses = _stats_overlay.get("horse_stats", {})

    file_exists = os.path.exists(DICT_PATH)
    file_size_mb = round(os.path.getsize(DICT_PATH) / (1024 * 1024), 2) if file_exists else 0.0

    return {
        "status": "ok",
        "file_exists": file_exists,
        "file_size_mb": file_size_mb,
        "total_horses": len(h_dict),
        "total_jockeys": len(j_dict),
        "total_trainers": len(t_dict),
        "overlay_horses_count": len(overlay_horses),
        "overlay_last_updated": _stats_overlay.get("updated_at"),
        "supabase_connected": supabase is not None
    }


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


@app.get("/admin/test-umaca-connection")
def endpoint_test_umaca_connection():
    """Renderコンテナ上からJRA（UMACAスマートおよび照会サービス）へのネットワーク疎通テスト"""
    urls = {
        "voting_site": "https://www.ipat.jra.go.jp/sp/umaca/",
        "inquiry_site": "https://www.nvinq.jra.go.jp/jrasp/SMP000400",
        "jra_top": "https://www.jra.go.jp/"
    }
    results = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    }

    for name, url in urls.items():
        try:
            start_t = time.time()
            resp = requests.get(url, headers=headers, timeout=10)
            elapsed = round(time.time() - start_t, 2)
            html_snippet = resp.text[:300].replace("\n", " ").strip()
            is_blocked = resp.status_code in (403, 401, 503) or "アクセスが制限されています" in resp.text
            results[name] = {
                "status_code": resp.status_code,
                "elapsed_sec": elapsed,
                "is_blocked": is_blocked,
                "content_bytes": len(resp.content),
                "snippet": html_snippet[:120]
            }
        except Exception as e:
            results[name] = {
                "status_code": None,
                "error": str(e)
            }

    can_reach_all = all(r.get("status_code") == 200 for r in results.values())
    return {
        "status": "ok",
        "can_reach_all": can_reach_all,
        "render_ip_check": results,
        "summary": "【接続成功】RenderからJRAへの通信はブロックされていません。クラウド化が可能です。" if can_reach_all else "【接続制限あり】一部の通信がブロックされています。"
    }


@app.on_event("startup")
async def startup_event():
    # DB構成（バージョン）に変更があった時のみ初期化し、通常の再起動時はデータを完全保持
    check_and_apply_db_migration()
    normalize_mislabeled_dates()
    fix_cancelled_races_2026()
    load_stats_overlay_from_supabase()
    asyncio.create_task(background_maintenance_loop())
    # 起動時にメイン画面対象日の出走時刻をバックグラウンドで最新化
    asyncio.create_task(asyncio.to_thread(endpoint_sync_active_post_times, None))

