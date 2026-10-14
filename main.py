from typing import List, Optional, Dict, Any
import os
import math
import datetime
import zoneinfo
import re
import json
import asyncio
import threading
import requests
import lightgbm as lgb
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI(title="Keiba Prediction & Verification API")

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


HEADERS_PC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/"
}

MODEL_PATH = "lgb_model.txt"
WIN_MODEL_PATH = "lgb_win_model.txt"
TOP3_MODEL_PATH = "lgb_top3_model.txt"
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

# --- Pydantic スキーマ ---
class HorsePrediction(BaseModel):
    umaban: int
    wakuban: int
    horse_name: str
    jockey: str
    trainer: Optional[str]
    kinryo: float
    odds: Optional[float]
    popularity: Optional[int]
    score: float
    prob_win: Optional[float] = None
    prob_top3: Optional[float] = None
    mark: str
    is_blinker: Optional[bool] = False
    is_first_blinker: Optional[bool] = False
    horse_weight: Optional[int] = None
    weight_diff: Optional[int] = None
    sex_age: Optional[str] = None

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
    is_real_bet_target: Optional[bool] = False
    real_bet_badge: Optional[str] = None
    real_bet_level: Optional[str] = None
    is_final_odds: Optional[bool] = False

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
    umaren_payout: Optional[int] = 0
    trio_payout: Optional[int] = 0
    mark_orders: Optional[List[dict]] = None
    top3_orders: Optional[List[dict]] = None

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
        oresp = requests.get(u_url, headers=HEADERS_PC, timeout=6)
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
        oresp = requests.get(t_url, headers=HEADERS_PC, timeout=6)
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
_period_venues_cache: Dict[str, List[dict]] = {}
_period_checked_empty_dates: set = set()


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
    global _period_venues_cache, _period_checked_empty_dates
    _period_venues_cache.clear()
    _period_checked_empty_dates.clear()
    if not supabase:
        return
    try:
        supabase.table("keiba_predictions_cache").delete().neq("race_id", "SCHEMA_VERSION").execute()
        supabase.table("keiba_race_results_cache").delete().neq("race_id", "0").execute()
        supabase.table("keiba_scheduled_races").delete().neq("race_id", "0").execute()
        print("[Cleanup] 金曜20:00 開催期間の一時キャッシュ（予想・結果HTML・スケジュール）を破棄しました（成績台帳は保持）。")
    except Exception as e:
        print(f"[Cleanup] キャッシュ破棄エラー: {e}")


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
    """レース一覧取得時に未登録のレーススケジュールを一括で keiba_scheduled_races に同期"""
    if not supabase or not venues:
        return
    try:
        existing_res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", formatted_date).execute()
        existing_ids = {r["race_id"] for r in (existing_res.data or [])}

        all_race_items = []
        for v in venues:
            for r in v.get("races", []):
                if r.get("race_id") and r["race_id"] not in existing_ids:
                    all_race_items.append(r)

        if not all_race_items:
            return

        # 既に予想や確定成績が存在する場合はフラグを引き継ぐ
        pred_ids = set()
        settled_ids = set()
        try:
            p_res = supabase.table("keiba_predictions_cache").select("race_id").eq("race_date", formatted_date).execute()
            pred_ids = {r["race_id"] for r in (p_res.data or [])}
        except Exception:
            pass
        try:
            s_res = supabase.table("keiba_race_settlements").select("race_id").eq("race_date", formatted_date).execute()
            settled_ids = {r["race_id"] for r in (s_res.data or [])}
        except Exception:
            pass

        d_parts = [int(p) for p in formatted_date.split("-")]
        rows_to_upsert = []
        for r in all_race_items:
            r_id = r["race_id"]
            r_info = r.get("race_info") or r.get("race_name") or ""
            m = re.search(r"(\d{1,2}):(\d{2})", r_info)
            hour, minute = (int(m.group(1)), int(m.group(2))) if m else (15, 0)
            post_dt = datetime.datetime(d_parts[0], d_parts[1], d_parts[2], hour, minute, 0)
            rows_to_upsert.append({
                "race_id": r_id,
                "race_date": formatted_date,
                "race_name": r.get("race_name", r.get("race_no", "")),
                "post_datetime": post_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "is_predicted": 2 if r_id in pred_ids else 0,
                "is_settled": 1 if r_id in settled_ids else 0
            })

        if rows_to_upsert:
            supabase.table("keiba_scheduled_races").upsert(rows_to_upsert, on_conflict="race_id").execute()
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

        post_m = re.search(r"(\d{1,2}:\d{2})", clean_text)
        post_time_est = f"{post_m.group(1)}発走" if post_m else ""

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


def fetch_venues_for_date_raw(date_str: str) -> List[dict]:
    """指定日付 (YYYY-MM-DD または YYYYMMDD) のJRA開催会場・レース一覧を取得しキャッシュ"""
    clean_d = date_str.replace("-", "")
    if len(clean_d) != 8 or not clean_d.isdigit():
        return []
    formatted_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}"

    if formatted_date in _period_venues_cache:
        return json.loads(json.dumps(_period_venues_cache[formatted_date]))

    urls = [
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={clean_d}",
        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={clean_d}",
        f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={clean_d}"
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS_PC, timeout=10)
            raw_head = resp.content[:300].lower()
            if b'charset="utf-8"' in raw_head or b"charset=utf-8" in raw_head:
                text = resp.content.decode("utf-8", errors="replace")
            else:
                try:
                    text = resp.content.decode("euc-jp")
                except UnicodeDecodeError:
                    text = resp.content.decode("utf-8", errors="replace")

            venues = parse_netkeiba_race_list_html(text)
            # 1会場あたり8R以上ある場合のみ正式な出走表確定済み開催日とみなす（次週の未確定枠を除外）
            valid_venues = [v for v in venues if len(v.get("races", [])) >= 8]
            if valid_venues:
                _period_venues_cache[formatted_date] = valid_venues
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
    """レース一覧の各レースにSupabase上の予想・実戦推奨バッジ情報を付与"""
    if not venues:
        return []
    cached_preds_map = {}
    if supabase:
        try:
            p_res = supabase.table("keiba_predictions_cache").select("race_id, data_json").eq("race_date", formatted_date).execute()
            for row in (p_res.data or []):
                h_rec = (row.get("data_json") or {}).get("hybrid_recommendation")
                if h_rec:
                    cached_preds_map[row["race_id"]] = {
                        "is_real_bet_target": h_rec.get("is_real_bet_target", False),
                        "real_bet_badge": h_rec.get("real_bet_badge"),
                        "real_bet_level": h_rec.get("real_bet_level", "none"),
                        "edge_ratio": h_rec.get("edge_ratio"),
                        "is_final_odds": h_rec.get("is_final_odds", False)
                    }
        except Exception as e:
            print(f"レース一覧キャッシュ付与エラー: {e}")

    for venue in venues:
        for race in venue.get("races", []):
            info = cached_preds_map.get(race["race_id"])
            if info:
                race["is_real_bet_target"] = info["is_real_bet_target"]
                race["real_bet_badge"] = info["real_bet_badge"]
                race["real_bet_level"] = info.get("real_bet_level", "none")
                race["edge_ratio"] = info["edge_ratio"]
                race["is_final_odds"] = info["is_final_odds"]
            else:
                race["is_real_bet_target"] = False
                race["real_bet_badge"] = None
                race["real_bet_level"] = "none"
                race["edge_ratio"] = None
                race["is_final_odds"] = False
    return venues


@app.get("/races/today")
def get_today_races(date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)")):
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

    venues = fetch_venues_for_date_raw(target_date)
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
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    raw_head = resp.content[:300].lower()
    if b'charset="utf-8"' in raw_head or b"charset=utf-8" in raw_head or b"utf-8" in raw_head:
        html = resp.content.decode("utf-8", errors="replace")
    else:
        try:
            html = resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            html = resp.content.decode("utf-8", errors="replace")
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

    post_time_m = re.search(r"(\d{1,2}:\d{2})\s*発走", race_data_text)
    post_time = f"{post_time_m.group(1)}発走" if post_time_m else ""

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
        oresp = requests.get(odds_url, headers=HEADERS_PC, timeout=5)
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
        if not umaban_tag or not umaban_tag.get_text(strip=True).isdigit():
            continue
        umaban = int(umaban_tag.get_text(strip=True))

        wakuban = 1
        waku_tag = row.select_one("td.Waku, td[class*='Waku']")
        if waku_tag:
            wm = re.search(r"\d+", waku_tag.get_text(strip=True))
            if wm:
                wakuban = int(wm.group())

        name_tag = row.select_one(".HorseName a, .Horse_Info a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        jockey_tag = row.select_one(".Jockey a")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        trainer_tag = row.select_one(".Trainer a, td.Trainer")
        trainer = trainer_tag.get_text(strip=True) if trainer_tag else "未定"

        # ★ 性齢 (td.Barei) と 斤量 (td.Barei の次の td または td.Kinryo)
        sex_age = None
        sex_code = 0
        age = 4.0
        barei_tag = row.select_one("td.Barei")
        if barei_tag:
            b_str = barei_tag.get_text(strip=True)
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
        kinryo_tag = barei_tag.find_next_sibling("td") if barei_tag else row.select_one("td.Kinryo")
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        # ★ 馬体重・増減 (td.Weight: 例 "464(-2)" / "480(+6)")
        horse_weight: Optional[int] = None
        weight_diff: Optional[int] = None
        weight_tag = row.select_one("td.Weight")
        if weight_tag:
            w_str = weight_tag.get_text(strip=True)
            wm = re.search(r"(\d{3})\s*(?:\(([+-]?\d+)\))?", w_str)
            if wm:
                horse_weight = int(wm.group(1))
                weight_diff = int(wm.group(2)) if wm.group(2) is not None else 0

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

    if not horses:
        return {
            "race_name": race_name,
            "race_details": race_details,
            "horses": [],
            "recommendation": empty_recommendation
        }

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
        except Exception:
            for idx, h in enumerate(horses):
                odds = h["odds"] if h["odds"] else 50.0
                h["prob_win"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
                h["prob_top3"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
                h["score"] = h["prob_win"]
    elif model:
        try:
            n_mod = model.num_feature()
            scores = model.predict([r[:n_mod] for r in feature_rows])
            for idx, h in enumerate(horses):
                h["prob_win"] = round(float(scores[idx]), 4)
                h["prob_top3"] = round(float(scores[idx]), 4)
                h["score"] = round(float(scores[idx]), 4)
        except Exception:
            for idx, h in enumerate(horses):
                odds = h["odds"] if h["odds"] else 50.0
                h["prob_win"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
                h["prob_top3"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
                h["score"] = h["prob_win"]
    else:
        for idx, h in enumerate(horses):
            odds = h["odds"] if h["odds"] else 50.0
            base_score = 1.0 / (1.0 + (odds ** 0.5))
            h["prob_win"] = round(float(base_score), 4)
            h["prob_top3"] = round(float(base_score), 4)
            h["score"] = round(float(base_score), 4)

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
        recommendation = empty_recommendation

    # -------------------------------------------------------------
    # ★ 波乱度メタモデル × 動的資金配分・券種切り替え戦略
    # -------------------------------------------------------------
    hybrid_rec: Optional[dict] = None
    if len(horses) >= 3:
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

        prob_gachi = 0.0
        prob_himo = 0.0
        if model_gachi:
            try:
                n_g = model_gachi.num_feature()
                prob_gachi = round(float(model_gachi.predict([race_feat_row[:n_g]])[0]), 4)
            except Exception:
                prob_gachi = 0.0
        if model_himo:
            try:
                n_h = model_himo.num_feature()
                prob_himo = round(float(model_himo.predict([race_feat_row[:n_h]])[0]), 4)
            except Exception:
                prob_himo = 0.0

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
            force_final_odds=False,
            real_trio_odds=trio_real_map
        )

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

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
    course_details_str = (race_details or {}).get("course_details", "") if isinstance(race_details, dict) else ""
    if "障害" in race_name or "障" in course_details_str:
        surface_type = 2
    is_shinba = ("新馬" in race_name) or ("メイクデビュー" in race_name)

    s_odds = sorted(horses, key=lambda x: (x.get("odds") if x.get("odds") is not None and float(x.get("odds") or 0) > 0 else 999.0))
    o1 = float(s_odds[0].get("odds") or 50.0) if len(s_odds) >= 1 else 50.0

    honmei_score = float(honmei_horse.get("prob_win", honmei_horse.get("score", 0.0))) if honmei_horse else 0.0
    honmei_odds = float(honmei_horse.get("odds") or 0.0) if honmei_horse else 0.0
    raw_edge = round(honmei_score * honmei_odds, 2) if honmei_odds > 0 else 0.0
    edge_ratio = 0.0 if (is_shinba or surface_type == 2) else raw_edge

    inv_u = sum(1.0 / opp_odds for _, opp_odds, _, _ in opponents if opp_odds and opp_odds > 0)
    synth_odds = round(1.0 / inv_u, 2) if inv_u > 0 else 0.0
    any_real_odds = any(opp_real for _, _, opp_real, _ in opponents)

    # 仮想買い目 (見送り時の検証用: 均等200円×3点 = 600円)
    hypothetical_umaren_bets = []
    for opp_u, opp_odds, opp_real, opp_mark in opponents:
        hypo_amt = 200
        exp_p = int(round(hypo_amt * opp_odds)) if opp_odds > 0 else 0
        hypothetical_umaren_bets.append({
            "ticket_type": "馬連",
            "combination": [u0, opp_u],
            "combination_str": f"{u0}-{opp_u}",
            "amount": hypo_amt,
            "odds": opp_odds,
            "expected_payout": exp_p,
            "is_real_odds": opp_real,
            "memo": f"仮想馬連(◎-{opp_mark} {hypo_amt}円 / {'実オッズ' if opp_real else '想定'}{opp_odds}倍)"
        })

    # ------------------------------------------------------------------
    # ★ 3段階階層化 × 動的パーセンタイル戦略 (DynamicPercentile_Top5Leverage)
    #    Tier 1: 勝負馬券 (上位2%, Score >= 0.999) - W勝負 (三連複1,000円, 馬連1,200円 = 2,200円)
    #    Tier 2: 実馬券   (上位5%, 0.821 <= Score < 0.999) - W勝負 (三連複500円, 馬連600円 = 1,100円)
    #    Tier 3: 仮想馬券 (上位12%, 0.665 <= Score < 0.821) - アプリ内集計 (三連複500円, 馬連600円 = 1,100円)
    #    見送り: Score < 0.665 または 頭数不足/新馬/障害 (0円)
    # ------------------------------------------------------------------
    has_enough_horses = len(horses) >= 6
    is_valid_race_type = (not is_shinba) and (surface_type != 2)

    # AI勝率順
    s_win = sorted(horses, key=lambda x: x.get("prob_win", 0.0), reverse=True)
    w0 = s_win[0] if s_win else {}
    w1 = s_win[1] if len(s_win) > 1 else w0

    # AI複勝率順
    s_top3 = sorted(horses, key=lambda x: x.get("prob_top3", 0.0), reverse=True)
    t0 = s_top3[0] if s_top3 else {}
    t1 = s_top3[1] if len(s_top3) > 1 else t0
    t2 = s_top3[2] if len(s_top3) > 2 else t1

    w0_prob = float(w0.get("prob_win", 0.0))
    w1_prob = float(w1.get("prob_win", 0.0))
    w0_odds = float(w0.get("odds") or 50.0)

    t0_prob = float(t0.get("prob_top3", 0.0))
    t1_prob = float(t1.get("prob_top3", 0.0))
    t2_prob = float(t2.get("prob_top3", 0.0))
    t0_odds = float(t0.get("odds") or 50.0)
    t1_odds = float(t1.get("odds") or 50.0)

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

    # 確定オッズ・発走前判定
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
                if now_dt >= post_dt - datetime.timedelta(minutes=25):
                    is_final_odds = True
            except Exception:
                pass

    suffix = " (確定)" if is_final_odds else " (暫定)"

    # 3段階判定
    is_tier1 = has_enough_horses and is_valid_race_type and (confidence_score >= 0.999)
    is_tier2 = has_enough_horses and is_valid_race_type and (0.821 <= confidence_score < 0.999)
    is_tier3 = has_enough_horses and is_valid_race_type and (0.665 <= confidence_score < 0.821)

    is_participate = is_tier1 or is_tier2 or is_tier3
    is_real_bet_target = is_tier1 or is_tier2

    # 軸馬と相手馬の選定
    w0_u = int(w0.get("umaban", 0))
    w1_u = int(w1.get("umaban", 0))
    t0_u = int(t0.get("umaban", 0))
    t1_u = int(t1.get("umaban", 0))

    s_odds = sorted(horses, key=lambda x: (float(x.get("odds") or 999.0)))
    fav1_u = int(s_odds[0]["umaban"]) if s_odds else t0_u

    # 三連複相手5頭 (軸: t0, t1)
    trio_opponents = []
    if fav1_u not in (t0_u, t1_u):
        trio_opponents.append(fav1_u)
    for h in s_top3:
        hu = int(h.get("umaban", 0))
        if hu not in (t0_u, t1_u) and hu not in trio_opponents:
            trio_opponents.append(hu)
        if len(trio_opponents) >= 5:
            break

    # 馬連相手3頭 (軸: w0)
    umaren_opponents = [w1_u]
    for h in s_top3:
        hu = int(h.get("umaban", 0))
        if hu != w0_u and hu not in umaren_opponents:
            umaren_opponents.append(hu)
        if len(umaren_opponents) >= 3:
            break

    umaren_bets = []
    trio_bets = []

    if is_participate:
        # 配分設定
        if is_tier1:  # 勝負馬券 (2x配分: 三連複200円×5点=1,000円 / 馬連400円×3点=1,200円)
            trio_unit = 200
            umaren_unit = 400
            real_bet_level = "high"
            real_bet_badge = f"🔥【超勝負R】厚め推奨{suffix}"
            strategy_name = "🔥 勝負馬券 (上位2%超厳選・資金傾斜配分)"
            strategy_type = "w_bet"
        elif is_tier2:  # 実馬券 (1x配分: 三連複100円×5点=500円 / 馬連200円×3点=600円)
            trio_unit = 100
            umaren_unit = 200
            real_bet_level = "standard"
            real_bet_badge = f"🎯【実戦推奨】標準推奨{suffix}"
            strategy_name = "🎯 実馬券 (上位5%厳選・標準購入推奨)"
            strategy_type = "w_bet"
        else:  # 仮想馬券 (1x配分: 三連複100円×5点=500円 / 馬連200円×3点=600円 / アプリ内集計のみ)
            trio_unit = 100
            umaren_unit = 200
            real_bet_level = "virtual"
            real_bet_badge = f"📊【仮想集計R】アプリ内モニタリング{suffix}"
            strategy_name = "📊 仮想馬券 (上位12%・アプリ集計モニタリング)"
            strategy_type = "virtual"

        # 馬連買い目
        for opp in umaren_opponents:
            comb = sorted([w0_u, opp])
            h_opp = next((h for h in horses if int(h.get("umaban", 0)) == opp), {})
            opp_odds = float(h_opp.get("odds") or 0.0)
            exp_p = int(round(umaren_unit * opp_odds)) if opp_odds > 0 else 0
            umaren_bets.append({
                "ticket_type": "馬連",
                "combination": comb,
                "combination_str": f"{comb[0]}-{comb[1]}",
                "amount": umaren_unit,
                "odds": opp_odds,
                "expected_payout": exp_p,
                "is_real_odds": True,
                "memo": f"馬連(軸{w0_u}→{opp}番 {umaren_unit}円 / {opp_odds}倍)"
            })

        # 三連複買い目
        for opp in trio_opponents:
            comb = sorted([t0_u, t1_u, opp])
            comb_tup = tuple(comb)
            h_opp = next((h for h in horses if int(h.get("umaban", 0)) == opp), {})
            opp_odds = float(h_opp.get("odds") or 50.0)
            if real_trio_odds and comb_tup in real_trio_odds:
                trio_odds_val = round(float(real_trio_odds[comb_tup]), 1)
            else:
                trio_odds_val = round(calc_trio_odds(t0_odds, t1_odds, opp_odds), 1)
            exp_tp = int(round(trio_unit * trio_odds_val))
            trio_bets.append({
                "ticket_type": "三連複",
                "combination": comb,
                "combination_str": f"{comb[0]}-{comb[1]}-{comb[2]}",
                "amount": trio_unit,
                "odds": trio_odds_val,
                "expected_payout": exp_tp,
                "is_real_odds": bool(real_trio_odds and comb_tup in real_trio_odds),
                "memo": f"2頭軸[{t0_u}={t1_u}]→{opp}番 {trio_unit}円 / {trio_odds_val}倍"
            })

        total_amount = sum(b["amount"] for b in umaren_bets) + sum(b["amount"] for b in trio_bets)
        t_opps_str = ",".join(str(o) for o in trio_opponents)
        u_opps_str = ",".join(str(o) for o in umaren_opponents)
        summary_str = f"{strategy_name} [馬連{len(umaren_bets)}点(軸{w0_u}→{u_opps_str}) ＋ 三連複{len(trio_bets)}点(軸[{t0_u}={t1_u}]→{t_opps_str}) / 計{total_amount:,}円]"
        reason_str = f"動的確信度スコア: {confidence_score:.3f} (閾値判定合致)"

        hybrid_rec = {
            "is_participate": True,
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "total_amount": total_amount,
            "summary": summary_str,
            "umaren_bets": umaren_bets,
            "sanrentan_bets": [],
            "hypothetical_umaren_bets": umaren_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bets[0] if trio_bets else None,
            "reason": reason_str,
            "is_real_odds_applied": any(b.get("is_real_odds", False) for b in (umaren_bets + trio_bets)),
            "total_expected_payout": sum(b.get("expected_payout", 0) for b in (umaren_bets + trio_bets)),
            "prob_gachi": prob_gachi,
            "prob_himo": prob_himo,
            "edge_ratio": edge_ratio,
            "synth_odds": synth_odds,
            "confidence_score": confidence_score,
            "is_real_bet_target": is_real_bet_target,
            "real_bet_badge": real_bet_badge,
            "real_bet_level": real_bet_level,
            "is_final_odds": is_final_odds
        }
    else:
        # 見送り
        real_bet_level = "none"
        real_bet_badge = None
        if not has_enough_horses:
            skip_reason = "出走頭数不足 (6頭未満のため見送り)"
        elif is_shinba:
            skip_reason = "新馬戦除外 (過去走実績なしのため見送り)"
        elif surface_type == 2:
            skip_reason = "障害戦除外 (不確実性高のため見送り)"
        else:
            skip_reason = f"動的確信度スコア不足 ({confidence_score:.3f} < 0.665: 資金保全のため見送り)"

        hybrid_rec = {
            "is_participate": False,
            "strategy_name": "見送り推奨 (資金保全)",
            "strategy_type": "skip",
            "total_amount": 0,
            "summary": f"見送り推奨 ({skip_reason})",
            "umaren_bets": [],
            "sanrentan_bets": [],
            "hypothetical_umaren_bets": hypothetical_umaren_bets,
            "trio_bets": [],
            "trio_bet": None,
            "reason": skip_reason,
            "is_real_odds_applied": False,
            "total_expected_payout": 0,
            "prob_gachi": prob_gachi,
            "prob_himo": prob_himo,
            "edge_ratio": edge_ratio,
            "synth_odds": synth_odds,
            "confidence_score": confidence_score,
            "is_real_bet_target": False,
            "real_bet_badge": None,
            "real_bet_level": "none",
            "is_final_odds": is_final_odds
        }

    return hybrid_rec


# =========================================================================
# ★ Supabase CRUD ヘルパー
# =========================================================================

def get_cached_prediction(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_predictions_cache").select("data_json").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["data_json"]
    except Exception as e:
        print(f"Supabase予想取得エラー ({race_id}): {e}")
    return None

def save_cached_prediction(race_id: str, race_date: str, data_dict: dict):
    if not supabase:
        return
    try:
        supabase.table("keiba_predictions_cache").upsert({
            "race_id": race_id,
            "race_date": race_date,
            "data_json": data_dict
        }).execute()
    except Exception as e:
        print(f"Supabase予想保存エラー ({race_id}): {e}")


def predict_race_core(race_id: str, force_refresh: bool = False, is_auto_final: bool = False) -> dict:
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    cached_pred = get_cached_prediction(race_id)
    if not force_refresh and cached_pred:
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
            for d_cached, v_list in _period_venues_cache.items():
                for v in v_list:
                    if any(r.get("race_id") == race_id for r in v.get("races", [])):
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

        result = build_prediction_and_recs(race_name, race_details, race_id, horses, course_meta)
        resp_data = {
            "race_id": race_id,
            "race_name": result["race_name"],
            "race_details": result["race_details"],
            "horses": result["horses"],
            "recommendation": result["recommendation"],
            "hybrid_recommendation": result.get("hybrid_recommendation")
        }

        save_cached_prediction(race_id, target_race_date, resp_data)

        if supabase:
            try:
                is_final = bool((resp_data.get("hybrid_recommendation") or {}).get("is_final_odds", False))
                pred_flag = 2 if (is_final or is_auto_final) else 1
                if pred_flag == 2:
                    supabase.table("keiba_scheduled_races").update({"is_predicted": 2}).eq("race_id", race_id).execute()
                else:
                    supabase.table("keiba_scheduled_races").update({"is_predicted": 1}).eq("race_id", race_id).neq("is_predicted", 2).execute()
            except Exception:
                pass

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


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
                "order_text": ord_text
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
    """見送りレースにおける仮想損益 (+400 - 的中時払戻) の算出"""
    hypo_bet = 400
    hypo_pay = 0
    payouts_data = race_res.get("payouts", {}) if race_res else {}
    umaren_payouts = payouts_data.get("umaren", {})

    bets_to_check = []
    if h_rec and h_rec.get("hypothetical_umaren_bets"):
        bets_to_check = h_rec["hypothetical_umaren_bets"]
    elif h_rec and h_rec.get("umaren_bets"):
        bets_to_check = h_rec["umaren_bets"]
    else:
        horses = pred_data.get("horses", []) if pred_data else []
        h0 = next((h for h in horses if h.get("mark") == "◎"), None)
        h1 = next((h for h in horses if h.get("mark") in ("◯", "〇", "○")), None)
        h2 = next((h for h in horses if h.get("mark") == "▲"), None)
        if not (h0 and h1 and h2) and len(horses) >= 3:
            s_h = sorted(horses, key=lambda x: x.get("score", 0), reverse=True)
            h0 = h0 or s_h[0]
            h1 = h1 or s_h[1]
            h2 = h2 or s_h[2]

        if h0 and h1 and h2:
            u0, u1, u2 = int(h0["umaban"]), int(h1["umaban"]), int(h2["umaban"])
            h3 = next((h for h in horses if h.get("mark") == "△"), None)
            u3 = int(h3["umaban"]) if h3 else None
            bets_to_check = [
                {"combination": [u0, u1], "amount": 200},
                {"combination": [u0, u2], "amount": 200}
            ]
            if u3:
                bets_to_check.append({"combination": [u0, u3], "amount": 200})
    if bets_to_check:
        total_b = sum(ub.get("amount", 0) for ub in bets_to_check)
        if total_b > 0:
            hypo_bet = total_b

    hit_details = []
    for ub in bets_to_check:
        c = ub.get("combination", [])
        if len(c) == 2:
            ckey = f"{min(c[0], c[1])}-{max(c[0], c[1])}"
            unit_p = umaren_payouts.get(ckey, 0)
            if unit_p > 0:
                pay = int((ub.get("amount", 100) / 100.0) * unit_p)
                hypo_pay += pay
                hit_details.append(f"馬連{ckey}(払戻{pay:,}円)")

    # 見送り損益 = +400 - 的中払戻金
    pass_benefit = hypo_bet - hypo_pay

    orders = race_res.get("orders", {}) if race_res else {}
    mark_orders = extract_mark_orders(pred_data, orders)
    top3_orders = extract_top3_orders(pred_data, orders)
    marks_summary = format_mark_orders_text(mark_orders)

    reason_str = h_rec.get("reason", "期待値基準外") if h_rec else "見送り推奨"
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
            rb_meta = b_det.get("real_bet_meta") or {}
            top3_orders = d.get("top3_orders") or extract_top3_orders(pred_cache, orders) or rb_meta.get("top3_orders")

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

    # SP版スクレイピング試行
    sp_url = f"https://race.sp.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = requests.get(sp_url, headers=HEADERS_PC, timeout=10)
        try:
            html = resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            html = resp.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        r_title = soup.select_one(".RaceName, h1, .Race_Title")
        if r_title:
            race_name = r_title.get_text(strip=True)

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
            resp = requests.get(pc_url, headers=HEADERS_PC, timeout=10)
            try:
                html = resp.content.decode("euc-jp")
            except UnicodeDecodeError:
                html = resp.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            r_title = soup.select_one(".RaceName, h1")
            if r_title and not race_name:
                race_name = r_title.get_text(strip=True)

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

    if not is_confirmed:
        return None

    return normalize_race_res({
        "race_name": race_name,
        "orders": orders,
        "payouts": payouts
    })


def process_settlement_calculation(
    race_id: str,
    race_date: str,
    race_name: str,
    pred_data: dict,
    race_res: dict,
    from_cache: bool = False
) -> dict:
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
            "strategy_type": "hybrid",
            "target_umaban": target_u,
            "horse_name": h_name,
            "grade": grade,
            "is_pass": True,
            "umaren_bet": 0,
            "trio_bet": 0,
            "total_bet": 0,
            "actual_order": actual_order if (actual_order and actual_order < 90) else None,
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
            "tansho_bet": 0,
            "fukusho_bet": 0,
            "tansho_payout": 0,
            "fukusho_payout": 0,
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
            "strategy_type": "hybrid",
            "hit_type": "見送り",
            "bet_breakdown": {},
            "payout_breakdown": {},
            "bet_details": None,
            "payout_details": None,
            "umaren_bet": 0,
            "trio_bet": 0,
            "umaren_payout": 0,
            "trio_payout": 0,
            "mark_orders": mark_orders,
            "top3_orders": top3_orders
        }

    # 参戦レースの場合
    umaren_bets = hybrid_rec.get("umaren_bets", [])
    sanrentan_bets = hybrid_rec.get("sanrentan_bets", [])
    trio_bets = hybrid_rec.get("trio_bets") or []
    trio_bet = hybrid_rec.get("trio_bet")
    if not trio_bets and trio_bet and isinstance(trio_bet, dict) and trio_bet.get("combination"):
        trio_bets = [trio_bet]

    u_bet_tot = sum(b.get("amount", 0) for b in umaren_bets)
    s_bet_tot = sum(b.get("amount", 0) for b in sanrentan_bets)
    t_bet_tot = sum(b.get("amount", 0) for b in trio_bets)
    total_bet = u_bet_tot + s_bet_tot + t_bet_tot

    payouts_data = race_res.get("payouts", {})

    # 1. 馬連照合
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

    # 2. 3連単照合
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

    # 3. 三連複照合 (複数点 trio_bets 対応)
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

    total_payout = umaren_payout + sanrentan_payout + trio_payout
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    if (len(umaren_hits) > 0 and len(trio_hits) > 0) or (len(sanrentan_hits) > 0 and len(umaren_hits) > 0):
        hit_type = "W的中🔥"
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

    all_hits = sanrentan_hits + umaren_hits + trio_hits
    if is_hit:
        hit_texts = [f"{h['type']}{h['comb']}({h['payout']:,}円)" for h in all_hits]
        msg = f"【{hit_type}】{' ＆ '.join(hit_texts)}！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery_rate}%){marks_line}"
    else:
        order_info = f" (◎{target_u}番: {actual_order}着)" if (actual_order and actual_order < 90) else ""
        msg = f"不的中{order_info} (投資: {total_bet:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%){marks_line}"

    bet_breakdown = {}
    payout_breakdown = {}
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
        "strategy_type": hybrid_rec.get("strategy_type", "hybrid"),
        "target_umaban": target_u,
        "horse_name": h_name,
        "grade": grade,
        "is_pass": False,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "total_bet": total_bet,
        "actual_order": actual_order if (actual_order and actual_order < 90) else None,
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
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet,
            "real_bet_meta": real_bet_meta
        },
        "payout_details": all_hits,
        "tansho_bet": 0,
        "fukusho_bet": 0,
        "tansho_payout": 0,
        "fukusho_payout": 0,
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
        "strategy_type": hybrid_rec.get("strategy_type", "hybrid"),
        "hit_type": hit_type,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "umaren_bets": umaren_bets,
            "sanrentan_bets": sanrentan_bets,
            "trio_bets": trio_bets,
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders,
        "is_real_bet_target": bool(hybrid_rec.get("is_real_bet_target", False))
    }


def resolve_race_date_for_id(race_id: str) -> str:
    """レースIDに対応する正確な開催日 (YYYY-MM-DD) をキャッシュ・DB・netkeibaから逆引き"""
    for d_cached, v_list in _period_venues_cache.items():
        for v in v_list:
            if any(r.get("race_id") == race_id for r in v.get("races", [])):
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
        resp = requests.get(url, headers=HEADERS_PC, timeout=5)
        try:
            html = resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            html = resp.text
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
        from_cache = True
        race_res = cached_data
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
def get_summary_dates():
    """過去の予想・集計可能日付一覧を取得 (降順)"""
    today_str = today_jst_str()
    try:
        dates_set = set(discover_period_kaisai_dates())
        if supabase:
            # 1. 確定・集計済みレコードがある日付（永続保存されている全過去日）
            try:
                res_settle = supabase.table("keiba_race_settlements").select("race_date").execute()
                for row in (res_settle.data or []):
                    d = row.get("race_date")
                    if d and len(d) == 10 and d.startswith("20"):
                        dates_set.add(d)
            except Exception as e:
                print(f"settlements dates error: {e}")

            # 2. レース開催日として登録された日付
            try:
                res_sched = supabase.table("keiba_scheduled_races").select("race_date").execute()
                for row in (res_sched.data or []):
                    d = row.get("race_date")
                    if d and len(d) == 10 and d.startswith("20"):
                        dates_set.add(d)
            except Exception as e:
                print(f"scheduled dates error: {e}")

        sorted_dates = sorted(list(dates_set), reverse=True)
        return sorted_dates if sorted_dates else [today_str]
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
        r["real_bet_badge"] = h_rec.get("real_bet_badge") or rb_meta.get("real_bet_badge")
        r["real_bet_level"] = h_rec.get("real_bet_level") or rb_meta.get("real_bet_level", "none")
        r["edge_ratio"] = h_rec.get("edge_ratio") if h_rec.get("edge_ratio") is not None else rb_meta.get("edge_ratio")
        r["is_final_odds"] = h_rec.get("is_final_odds", rb_meta.get("is_final_odds", False))

        if r.get("is_pass") == 1:
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
            if not r.get("top3_orders"):
                r["top3_orders"] = extract_top3_orders(p_data, c_res.get("orders", {})) if (p_data and c_res) else rb_meta.get("top3_orders")

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r["is_pass"] == 0)
    pass_races = sum(1 for r in rows if r["is_pass"] == 1)
    hit_races = sum(1 for r in rows if r["is_hit"] == 1 and r["is_pass"] == 0)

    total_bet = sum(r["total_bet"] for r in rows)
    total_payout = sum(r["total_payout"] for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

    # 実戦推奨レースのみの収支集計
    real_bet_rows = [r for r in rows if r.get("is_real_bet_target") and r.get("is_pass") == 0]
    real_bet_races = len(real_bet_rows)
    real_bet_hits = sum(1 for r in real_bet_rows if r.get("is_hit") == 1)
    real_bet_hit_rate = round((real_bet_hits / real_bet_races) * 100.0, 2) if real_bet_races > 0 else 0.0
    real_bet_amount = sum(r.get("total_bet", 0) for r in real_bet_rows)
    real_bet_payout = sum(r.get("total_payout", 0) for r in real_bet_rows)
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
    umaren_bet_total = sum(r.get("umaren_bet", 0) for r in rows if r.get("is_pass") == 0)
    trio_bet_total = sum(r.get("trio_bet", 0) for r in rows if r.get("is_pass") == 0)
    umaren_payout_total = sum(r.get("umaren_payout", 0) for r in rows if r.get("is_pass") == 0)
    trio_payout_total = sum(r.get("trio_payout", 0) for r in rows if r.get("is_pass") == 0)
    umaren_rec = round((umaren_payout_total / umaren_bet_total) * 100.0, 2) if umaren_bet_total > 0 else 0.0
    trio_rec = round((trio_payout_total / trio_bet_total) * 100.0, 2) if trio_bet_total > 0 else 0.0

    # ★ 累計トータル収支・成績の集計（永続テーブル keiba_race_settlements の全期間集計）
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

    if supabase:
        try:
            cum_res = supabase.table("keiba_race_settlements").select(
                "race_date, is_pass, is_hit, total_bet, total_payout, pass_benefit, bet_details"
            ).execute()
            cum_rows = cum_res.data or []
            cum_days = len(set(cr.get("race_date") for cr in cum_rows if cr.get("race_date")))
            cum_total_races = len(cum_rows)
            cum_bet_races = sum(1 for cr in cum_rows if cr.get("is_pass") == 0)
            cum_pass_races = sum(1 for cr in cum_rows if cr.get("is_pass") == 1)
            cum_hit_races = sum(1 for cr in cum_rows if cr.get("is_hit") == 1 and cr.get("is_pass") == 0)
            cum_bet_amount = sum((cr.get("total_bet") or 0) for cr in cum_rows)
            cum_payout_amount = sum((cr.get("total_payout") or 0) for cr in cum_rows)
            cum_profit = cum_payout_amount - cum_bet_amount
            cum_recovery_rate = round((cum_payout_amount / cum_bet_amount) * 100.0, 2) if cum_bet_amount > 0 else 0.0
            cum_hit_rate = round((cum_hit_races / cum_bet_races) * 100.0, 2) if cum_bet_races > 0 else 0.0
            cum_pass_benefit = sum((cr.get("pass_benefit") or 0) for cr in cum_rows if cr.get("is_pass") == 1)

            for cr in cum_rows:
                if cr.get("is_pass") == 0:
                    bd = cr.get("bet_details") if isinstance(cr.get("bet_details"), dict) else {}
                    rbm = bd.get("real_bet_meta") or {}
                    if rbm.get("is_real_bet_target"):
                        cum_rb_races += 1
                        if cr.get("is_hit") == 1:
                            cum_rb_hits += 1
                        cum_rb_amount += (cr.get("total_bet") or 0)
                        cum_rb_payout += (cr.get("total_payout") or 0)
            cum_rb_profit = cum_rb_payout - cum_rb_amount
            cum_rb_recovery_rate = round((cum_rb_payout / cum_rb_amount) * 100.0, 2) if cum_rb_amount > 0 else 0.0
            cum_rb_hit_rate = round((cum_rb_hits / cum_rb_races) * 100.0, 2) if cum_rb_races > 0 else 0.0
        except Exception as e:
            print(f"Supabase累計サマリー取得エラー: {e}")

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
                                    summary_body += "\n※各レース発走20分前に最新オッズで確定通知します。"
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
                # --- ③ 各レース【発走 20分前】の直前再予想実行（直前オッズで本確定へ昇格） ---
                try:
                    res_pred = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .lt("is_predicted", 2)\
                        .execute()
                    unpredicted_races = res_pred.data or []

                    for r in unpredicted_races:
                        r_id = r["race_id"]
                        p_dt_str = r["post_datetime"]
                        try:
                            post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue

                        trigger_predict_time = post_dt - datetime.timedelta(minutes=20)
                        if trigger_predict_time <= now_dt < post_dt:
                            print(f"[Auto-Predict] 発走20分前検知: {r_id} ({r.get('race_name')}) の確定予想を実行します...")
                            try:
                                pred_res = predict_race_core(r_id, force_refresh=True, is_auto_final=True)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 2})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Predict] 確定予想完了: {r_id}")

                                # 本確定の実戦推奨レースの場合、プッシュ通知を発火！
                                h_rec = pred_res.get("hybrid_recommendation") or {}
                                if h_rec.get("is_real_bet_target"):
                                    badge = h_rec.get("real_bet_badge", "🔥 実戦推奨(確定)")
                                    edge = h_rec.get("edge_ratio", 0.0)
                                    target_u = pred_res.get("recommendation", {}).get("target_umaban")
                                    h_name = pred_res.get("recommendation", {}).get("horse_name", "")
                                    r_name = pred_res.get("race_name", r.get("race_name", f"Race {r_id}"))

                                    u_bets = h_rec.get("umaren_bets", [])
                                    bets_summary = ", ".join([f"{b.get('combination_str')}({b.get('amount')}円)" for b in u_bets[:3]])
                                    tot_amt = h_rec.get("total_amount", 0)

                                    body_text = f"◎{target_u}番 {h_name} (期待エッジ:{edge}倍)\n【推奨買い目】馬連: {bets_summary} 計{tot_amt:,}円"
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
                                print(f"[Auto-Predict] 予想実行失敗 ({r_id}): {pe}")
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

def clear_all_database():
    """Supabaseの全テーブル（メタ、キャッシュ、台帳、スケジュール）を完全クリア・初期化"""
    if not supabase:
        print("[ClearDB] Supabase未接続のためスキップします。")
        return {"status": "skipped", "message": "Supabase not connected"}
    try:
        print("[ClearDB] Supabase全テーブルの初期化クリーンアップを開始します...")
        try:
            supabase.table("keiba_predictions_cache").delete().neq("race_id", "0").execute()
        except Exception as e:
            print(f"[ClearDB] predictions_cache clear warning: {e}")
        try:
            supabase.table("keiba_race_results_cache").delete().neq("race_id", "0").execute()
        except Exception as e:
            print(f"[ClearDB] results_cache clear warning: {e}")
        try:
            supabase.table("keiba_race_settlements").delete().neq("race_id", "0").execute()
        except Exception as e:
            print(f"[ClearDB] race_settlements clear warning: {e}")
        try:
            supabase.table("keiba_scheduled_races").delete().neq("race_id", "0").execute()
        except Exception as e:
            print(f"[ClearDB] scheduled_races clear warning: {e}")
        try:
            supabase.table("keiba_app_meta").delete().neq("key", "__keep__").execute()
        except Exception as e:
            print(f"[ClearDB] app_meta clear warning: {e}")

        # 初期化後、直ちに本日（JST）のレーススケジュールを再取得・再登録
        today_str = today_jst_str()
        today_clean = today_str.replace("-", "")
        today_info = get_today_races(today_clean)
        venues = today_info.get("venues", [])
        synced = 0
        for v in venues:
            for r in v.get("races", []):
                try:
                    fetch_shutuba_table(r["race_id"], race_date=today_str)
                    synced += 1
                except Exception:
                    pass

        msg = f"全DBテーブル初期化完了 ＆ 本日レーススケジュール再登録完了 ({synced}レース)"
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
def endpoint_clear_db():
    res = clear_all_database()
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
        return {
            "status": "ok",
            "predictions_count": pred_count,
            "settlements_count": settle_count,
            "scheduled_count": sched_count
        }
    except Exception as e:
        print(f"[UpsertVerifiedData Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    # DB構成（バージョン）に変更があった時のみ初期化し、通常の再起動時はデータを完全保持
    check_and_apply_db_migration()
    normalize_mislabeled_dates()
    asyncio.create_task(background_maintenance_loop())
