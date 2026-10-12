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
    mark: str
    is_blinker: Optional[bool] = False
    is_first_blinker: Optional[bool] = False

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
    trio_bet: Optional[TicketBet] = None
    reason: Optional[str] = None
    is_real_odds_applied: Optional[bool] = False
    total_expected_payout: Optional[int] = None
    prob_gachi: Optional[float] = None
    prob_himo: Optional[float] = None

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
    return max(2.0, min(1000.0, 0.75 / p_sum))

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
def health_check():
    return {"status": "ok", "message": "Keiba Prediction & Verification API is running", "server_time_jst": now_jst().strftime("%Y-%m-%d %H:%M:%S")}


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

        supabase.table("keiba_scheduled_races").upsert({
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "post_datetime": post_dt_str,
            "is_settled": 0
        }, on_conflict="race_id").execute()
    except Exception as e:
        print(f"Supabaseスケジュール登録エラー ({race_id}): {e}")


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


@app.get("/races/today")
def get_today_races(date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)")):
    today_dt = now_jst().date()
    if date:
        clean_date = date.replace("-", "")
        target_dates = [clean_date]
    else:
        target_dates = [
            (today_dt + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range(7)
        ]

    for d_str in target_dates:
        # PC版サブHTML(レース一覧本体)を最優先取得し、フォールバックでPC版・SP版を使用
        urls = [
            f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={d_str}",
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={d_str}",
            f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={d_str}"
        ]
        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS_PC, timeout=10)
                # UTF-8が明記されているか判定
                raw_head = resp.content[:300].lower()
                if b'charset="utf-8"' in raw_head or b"charset=utf-8" in raw_head:
                    text = resp.content.decode("utf-8", errors="replace")
                else:
                    try:
                        text = resp.content.decode("euc-jp")
                    except UnicodeDecodeError:
                        text = resp.content.decode("utf-8", errors="replace")

                venues = parse_netkeiba_race_list_html(text)
                if venues:
                    formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
                    return {"date": formatted_date, "venues": venues}
            except Exception:
                continue

    return {
        "date": today_dt.strftime("%Y-%m-%d"),
        "message": "直近7日間の開催情報が見つかりませんでした。",
        "venues": []
    }


def fetch_shutuba_table(race_id: str, race_date: Optional[str] = None):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_num_int = int(race_id[10:12]) if len(race_id) == 12 and race_id[10:12].isdigit() else 1
    race_num = f"{race_num_int}R"

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
    if "ダ" in race_data_text:
        surface_name = "ダート"
        surface_type = 1
    elif "障" in race_data_text:
        surface_name = "障害"
        surface_type = 2

    condition_code = 0
    if "稍" in baba:
        condition_code = 1
    elif "不良" in baba:
        condition_code = 3
    elif "重" in baba:
        condition_code = 2

    weather_str = f" 天候:{weather}" if weather else ""
    course_details = f"{surface_name}{int(distance)}m ({baba}){weather_str}"

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

        kinryo = 55.0
        kinryo_tag = row.select_one("td.Barei, td.Weight, td.Kinryo")
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

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
            "is_first_blinker": is_first_blinker
        })

    course_meta = {
        "distance": distance,
        "surface_type": surface_type,
        "condition_code": condition_code
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

    expected_num_features = 25
    if model:
        try:
            expected_num_features = model.num_feature()
        except Exception:
            expected_num_features = 25

    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    distance = course_meta.get("distance", 1600.0)
    surface_type = course_meta.get("surface_type", 0)
    condition_code = course_meta.get("condition_code", 0)

    course_key = f"{venue_code}_{surface_type}"

    j_stats = stats_dict.get("jockey_stats", {})
    jc_stats = stats_dict.get("jockey_course_stats", {})
    t_stats = stats_dict.get("trainer_stats", {})
    tc_stats = stats_dict.get("trainer_course_stats", {})
    h_stats = stats_dict.get("horse_stats", {})

    head_count = len(horses)
    kinryo_list = [float(h.get("kinryo", 55.0)) for h in horses]
    mean_kinryo = sum(kinryo_list) / max(head_count, 1)

    today_date = now_jst().date()

    horse_feature_data = []
    jc_rates = []
    tc_rates = []
    p1_orders = []

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
            career_top3_rate = 0.25
            prev1_order = 99.0
            prev1_pop = 10.0
            prev1_odds = 50.0
            prev2_order = 99.0
            days_since_prev = 999.0
            dist_diff = 0.0

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
            "base_feats": [
                w, u, k,
                venue_code, surface_type, distance, dist_diff, condition_code,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev,
                j_rate, jc_rate, t_rate, tc_rate,
                float(head_count), float(rel_umaban), float(kinryo_diff)
            ]
        })

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

        row25 = fd["base_feats"] + [float(j_ranks[idx]), float(t_ranks[idx]), float(p1_ranks[idx])]
        row32 = row25 + new_feats
        row36 = row32 + agari_feats

        if expected_num_features == 36:
            feature_rows.append(row36)
        elif expected_num_features == 32:
            feature_rows.append(row32)
        elif expected_num_features == 25:
            feature_rows.append(row25)
        elif expected_num_features == 19:
            feature_rows.append(row25[:19])
        else:
            feature_rows.append(row36[:expected_num_features])

    if model_win and model_top3:
        try:
            w_scores = model_win.predict(feature_rows)
            t_scores = model_top3.predict(feature_rows)
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
            scores = model.predict(feature_rows)
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
    # ★ 馬連・超高配当三連複ハイブリッド推奨ロジック (黒字最適化)
    # -------------------------------------------------------------
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
            float(front_runners), float(front_ratio)
        ]

        prob_gachi = 0.0
        prob_himo = 0.0
        if model_gachi:
            try:
                prob_gachi = round(float(model_gachi.predict([race_feat_row])[0]), 4)
            except Exception:
                prob_gachi = 0.0
        if model_himo:
            try:
                prob_himo = round(float(model_himo.predict([race_feat_row])[0]), 4)
            except Exception:
                prob_himo = 0.0

        # Webリアルタイムオッズ取得
        real_odds_data = fetch_real_odds(race_id)
        umaren_real_map = real_odds_data.get("umaren", {})

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

        # 相手馬リスト（最大3頭: ◯, ▲, △）
        opponents = []
        if u1: opponents.append((u1, u01_odds, u01_is_real, "◯"))
        if u2: opponents.append((u2, u02_odds, u02_is_real, "▲"))
        if u3: opponents.append((u3, u03_odds, u03_is_real, "△"))

        # ルール判定:
        # 1. 堅実本命戦: ガチ度が高い、または1番人気が2.2倍以下の確固たる本命レース
        is_target_gachi = (prob_gachi >= 0.22 or o1 <= 2.2) and (surface_type != 2) and (len(horses) >= 5)

        # 2. 混戦妙味戦: 堅実判定ではなく、かつ極端な大荒れ・危険レースではないレース (1番人気が5.0倍未満)
        is_target_himo = not is_target_gachi and (surface_type != 2) and (len(horses) >= 5) and (o1 < 5.0)

        # 3. 大波乱・難解戦: 障害戦、少頭数(<5)、または1番人気が5.0倍以上の大混戦
        is_target_skip = not (is_target_gachi or is_target_himo)

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

        if is_target_gachi and len(opponents) >= 1:
            # 【ルール1】堅実本命戦 → 馬連傾斜配分 (◎-◯ 500円, ◎-▲ 200円, ◎-△ 100円 = 計800円)
            umaren_bets = []
            if len(opponents) >= 3:
                amounts = [500, 200, 100]
            elif len(opponents) == 2:
                amounts = [600, 200]
            else:
                amounts = [800]

            for (opp_u, opp_odds, opp_real, opp_mark), amt in zip(opponents, amounts):
                exp_p = int(round(amt * opp_odds)) if opp_odds > 0 else 0
                umaren_bets.append({
                    "ticket_type": "馬連",
                    "combination": [u0, opp_u],
                    "combination_str": f"{u0}-{opp_u}",
                    "amount": amt,
                    "odds": opp_odds,
                    "expected_payout": exp_p,
                    "is_real_odds": opp_real,
                    "memo": f"本命傾斜(◎-{opp_mark} {amt}円 / {'実オッズ' if opp_real else '想定'}{opp_odds}倍 → 想定払戻{exp_p:,}円)"
                })
            tot_amt = sum(b["amount"] for b in umaren_bets)
            tot_exp = sum(b.get("expected_payout", 0) for b in umaren_bets if b.get("combination") == [u0, u1])
            hybrid_rec = {
                "is_participate": True,
                "strategy_name": "堅実本命戦 (馬連傾斜配分)",
                "strategy_type": "gachi",
                "total_amount": tot_amt,
                "summary": f"参戦推奨: 馬連傾斜配分({tot_amt}円) [本命線厚張り500円+抑え] (堅実度{prob_gachi*100:.1f}%)",
                "umaren_bets": umaren_bets,
                "sanrentan_bets": [],
                "hypothetical_umaren_bets": umaren_bets,
                "trio_bet": None,
                "reason": f"堅実判定クリア（1番人気{o1}倍・ガチ度基準合致 / 馬連傾斜配分）",
                "is_real_odds_applied": any_real_odds,
                "total_expected_payout": tot_exp,
                "prob_gachi": prob_gachi,
                "prob_himo": prob_himo
            }
        elif is_target_himo and len(opponents) >= 1:
            # 【ルール2】混戦妙味戦 → 馬連均等配分 (◎-◯ 200円, ◎-▲ 200円, ◎-△ 200円 = 計600円)
            umaren_bets = []
            if len(opponents) >= 3:
                amounts = [200, 200, 200]
            elif len(opponents) == 2:
                amounts = [300, 300]
            else:
                amounts = [600]

            for (opp_u, opp_odds, opp_real, opp_mark), amt in zip(opponents, amounts):
                exp_p = int(round(amt * opp_odds)) if opp_odds > 0 else 0
                umaren_bets.append({
                    "ticket_type": "馬連",
                    "combination": [u0, opp_u],
                    "combination_str": f"{u0}-{opp_u}",
                    "amount": amt,
                    "odds": opp_odds,
                    "expected_payout": exp_p,
                    "is_real_odds": opp_real,
                    "memo": f"混戦均等(◎-{opp_mark} {amt}円 / {'実オッズ' if opp_real else '想定'}{opp_odds}倍 → 想定払戻{exp_p:,}円)"
                })
            tot_amt = sum(b["amount"] for b in umaren_bets)
            hybrid_rec = {
                "is_participate": True,
                "strategy_name": "混戦妙味戦 (馬連均等配分)",
                "strategy_type": "himo",
                "total_amount": tot_amt,
                "summary": f"参戦推奨: 馬連均等配分({tot_amt}円) [◎から◯▲△ 3点流し各200円] (混戦度{prob_himo*100:.1f}%)",
                "umaren_bets": umaren_bets,
                "sanrentan_bets": [],
                "hypothetical_umaren_bets": umaren_bets,
                "trio_bet": None,
                "reason": f"混戦判定クリア（中穴跳ね上がり狙い / 馬連3点均等配分）",
                "is_real_odds_applied": any_real_odds,
                "total_expected_payout": 0,
                "prob_gachi": prob_gachi,
                "prob_himo": prob_himo
            }
        else:
            # 【ルール3】大波乱・難解判定 → 完全見送り（0円）で資金を守る！
            reason_detail = []
            if surface_type == 2:
                reason_detail.append("障害レース除外")
            elif len(horses) < 5:
                reason_detail.append("少頭数レース除外")
            elif o1 >= 5.0:
                reason_detail.append(f"超混戦大荒れ懸念（1番人気:{o1}倍）")
            else:
                reason_detail.append(f"リスク回避（1番人気:{o1}倍）")
            reason_str = " / ".join(reason_detail)
            hybrid_rec = {
                "is_participate": False,
                "strategy_name": "大波乱・難解 (完全見送り)",
                "strategy_type": "skip",
                "total_amount": 0,
                "summary": f"見送り推奨 (大波乱・難解判定: 資金保全のため0円見送り)",
                "umaren_bets": [],
                "sanrentan_bets": [],
                "hypothetical_umaren_bets": hypothetical_umaren_bets,
                "trio_bet": None,
                "reason": reason_str,
                "is_real_odds_applied": any_real_odds,
                "total_expected_payout": 0,
                "prob_gachi": prob_gachi,
                "prob_himo": prob_himo
            }

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "race_details": race_details,
        "horses": horses_sorted_by_num,
        "recommendation": recommendation,
        "hybrid_recommendation": hybrid_rec
    }


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


@app.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race(race_id: str, force_refresh: bool = Query(False, description="手動で予想を再実行する場合True")):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    if not force_refresh:
        cached_pred = get_cached_prediction(race_id)
        if cached_pred:
            return cached_pred

    try:
        target_race_date = None
        if supabase:
            try:
                s_res = supabase.table("keiba_scheduled_races").select("race_date").eq("race_id", race_id).execute()
                if s_res.data and len(s_res.data) > 0:
                    target_race_date = s_res.data[0]["race_date"]
            except Exception:
                pass
        if not target_race_date:
            target_race_date = today_jst_str()

        race_name, race_details, horses, course_meta = fetch_shutuba_table(race_id, race_date=target_race_date)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした。")

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

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


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
            top3_orders = d.get("top3_orders") or extract_top3_orders(pred_cache, orders)

            if d["is_pass"]:
                h_rec = pred_cache.get("hybrid_recommendation") if pred_cache else None
                pass_info = calculate_hybrid_pass_benefit_info(h_rec, res_cache, d.get("target_umaban"), order, pred_data=pred_cache)
                hypo_bet = pass_info["hypothetical_bet"]
                hypo_pay = pass_info["hypothetical_payout"]
                pass_benefit = pass_info["pass_benefit"]
                msg = pass_info["message"]
                mark_orders = pass_info.get("mark_orders")
                top3_orders = pass_info.get("top3_orders") or top3_orders
            elif d["is_hit"]:
                mark_orders = d.get("mark_orders") or extract_mark_orders(pred_cache, orders)
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
                mark_orders = d.get("mark_orders") or extract_mark_orders(pred_cache, orders)
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


def get_cached_result(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_results_cache").select("result_json").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["result_json"]
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

    return {
        "race_name": race_name,
        "orders": orders,
        "payouts": payouts
    }


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
    actual_order = orders.get(target_u) if target_u else None

    # ハイブリッド推奨を最優先で照合（見送りレース）
    if not hybrid_rec or not hybrid_rec.get("is_participate", False):
        pass_info = calculate_hybrid_pass_benefit_info(hybrid_rec, race_res, target_u, actual_order, pred_data=pred_data)
        hypo_bet = pass_info["hypothetical_bet"]
        hypo_pay = pass_info["hypothetical_payout"]
        pass_benefit = pass_info["pass_benefit"]
        mark_orders = pass_info.get("mark_orders", [])
        top3_orders = pass_info.get("top3_orders", [])
        msg = pass_info["message"]

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
            "bet_details": None,
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
    trio_bet = hybrid_rec.get("trio_bet")
    u_bet_tot = sum(b.get("amount", 0) for b in umaren_bets)
    s_bet_tot = sum(b.get("amount", 0) for b in sanrentan_bets)
    t_bet_tot = trio_bet.get("amount", 0) if trio_bet else 0
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

    # 3. 三連複照合
    trio_payout = 0
    trio_hits = []
    if trio_bet and trio_bet.get("combination"):
        comb = sorted(trio_bet.get("combination", []))
        if len(comb) == 3:
            comb_key = f"{comb[0]}-{comb[1]}-{comb[2]}"
            unit_p = payouts_data.get("trio", {}).get(comb_key, 0)
            if unit_p > 0:
                pay = int((trio_bet.get("amount", 100) / 100.0) * unit_p)
                trio_payout += pay
                trio_hits.append({
                    "type": "三連複",
                    "comb": comb_key,
                    "amount": trio_bet.get("amount"),
                    "unit_payout": unit_p,
                    "payout": pay
                })

    total_payout = umaren_payout + sanrentan_payout + trio_payout
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    if len(sanrentan_hits) > 0 and len(umaren_hits) > 0:
        hit_type = "W的中🔥"
    elif len(sanrentan_hits) > 0:
        hit_type = "3連単的中🎯"
    elif len(umaren_hits) > 0:
        hit_type = "馬連的中🎯"
    elif len(trio_hits) > 0:
        hit_type = "三連複的中🎯"
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
            "trio_bet": trio_bet
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
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout,
        "mark_orders": mark_orders,
        "top3_orders": top3_orders
    }


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
        # 照合済みキャッシュが存在し、確定着順が判明しているか、または見送り確定レースの場合は即座に返却
        if cached_settlement and (cached_settlement.get("actual_order") is not None or cached_settlement.get("is_pass")):
            return cached_settlement

    race_date = None
    if supabase:
        try:
            s_res = supabase.table("keiba_scheduled_races").select("race_date").eq("race_id", race_id).execute()
            if s_res.data and len(s_res.data) > 0:
                race_date = s_res.data[0]["race_date"]
        except Exception:
            pass
    if not race_date:
        today_str = today_jst_str()
        race_date = f"{race_id[:4]}-{today_str[5:7]}-{today_str[8:]}"

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
    pred_data = predict_race(race_id)
    rec = pred_data.get("recommendation", {})
    race_name = pred_data.get("race_name", race_res.get("race_name", f"Race {race_id}"))

    target_u = rec.get("target_umaban")
    orders = race_res.get("orders", {})
    payouts = race_res.get("payouts", {})
    actual_order = orders.get(target_u) if target_u else None

    # ★ 着順および払戻金が実数値で確認できるかの厳格チェック
    # 1. 1〜3着が実数値で確定しているか
    has_top3 = (1 in orders.values()) and (2 in orders.values()) and (3 in orders.values())
    # 2. 払戻金が公開されているか (単勝または馬連などの払戻金が存在するか)
    has_payouts = (len(payouts.get("tansho", {})) > 0) or (len(payouts.get("umaren", {})) > 0) or (len(payouts.get("fukusho", {})) > 0)
    # 3. 予想対象馬(◎)の確定着順が実数値として判明しているか (1〜30着)
    has_target_order = (actual_order is not None and isinstance(actual_order, int) and 1 <= actual_order <= 30)

    # 19時の全体確認以外は、着順・払戻が実数値で揃っていない場合は照合済みにしない！
    is_fully_confirmed = has_top3 and has_payouts and (has_target_order or target_u is None)
    if not is_19h_settle and not is_fully_confirmed:
        raise HTTPException(
            status_code=400,
            detail=f"着順または払戻金が実数値で確定していません（入線直後・審議中等のため未確定）。確定後に再度照合してください。(確認状況: 1〜3着確定={has_top3}, 払戻金確定={has_payouts}, ◎着順確定={has_target_order})"
        )

    # 実数値で確認できた場合のみ、キャッシュ保存
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
    if not supabase:
        return [today_str]
    try:
        dates_set = set()
        # 1. 確定・集計済みレコードがある日付
        try:
            res_settle = supabase.table("keiba_race_settlements").select("race_date").execute()
            for row in (res_settle.data or []):
                d = row.get("race_date")
                if d and len(d) == 10 and d.startswith("20"):
                    dates_set.add(d)
        except Exception as e:
            print(f"settlements dates error: {e}")

        # 2. 予想キャッシュが存在する日付（未確定・進行中含む）
        try:
            res_pred = supabase.table("keiba_predictions_cache").select("race_date").execute()
            for row in (res_pred.data or []):
                d = row.get("race_date")
                if d and len(d) == 10 and d.startswith("20"):
                    dates_set.add(d)
        except Exception as e:
            print(f"predictions dates error: {e}")

        # 3. レース開催日として登録された日付
        try:
            res_sched = supabase.table("keiba_scheduled_races").select("race_date").execute()
            for row in (res_sched.data or []):
                d = row.get("race_date")
                if d and len(d) == 10 and d.startswith("20"):
                    dates_set.add(d)
        except Exception as e:
            print(f"scheduled dates error: {e}")

        dates_set.add(today_str)
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

    # 結果キャッシュと予想キャッシュを一括取得
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
        if r.get("is_pass") == 1:
            p_data = pred_cache_map.get(r["race_id"], {})
            h_rec = p_data.get("hybrid_recommendation")
            c_res = res_cache_map.get(r["race_id"], {})
            pass_info = calculate_hybrid_pass_benefit_info(h_rec, c_res, r.get("target_umaban"), r.get("actual_order"), pred_data=p_data)
            pb = pass_info["pass_benefit"]
            r["hypothetical_bet"] = pass_info["hypothetical_bet"]
            r["hypothetical_payout"] = pass_info["hypothetical_payout"]
            r["pass_benefit"] = pb
            r["mark_orders"] = pass_info.get("mark_orders")
            r["top3_orders"] = pass_info.get("top3_orders")
            pass_benefit_total += pb
            if pb > 0:
                pass_avoided_losses += pb
            elif pb < 0:
                pass_missed_profits += abs(pb)
        else:
            p_data = pred_cache_map.get(r["race_id"], {})
            c_res = res_cache_map.get(r["race_id"], {})
            if not r.get("mark_orders"):
                r["mark_orders"] = extract_mark_orders(p_data, c_res.get("orders", {}))
            if not r.get("top3_orders"):
                r["top3_orders"] = extract_top3_orders(p_data, c_res.get("orders", {}))

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r["is_pass"] == 0)
    pass_races = sum(1 for r in rows if r["is_pass"] == 1)
    hit_races = sum(1 for r in rows if r["is_hit"] == 1 and r["is_pass"] == 0)

    total_bet = sum(r["total_bet"] for r in rows)
    total_payout = sum(r["total_payout"] for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

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
        "trio_recovery_rate": trio_rec
    }


# =========================================================================
# ★ 手動一括処理（アプリボタン用・確定検証と現時点の予想登録）
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
    batch_status["message"] = "開催レース走査中..."

    try:
        today_str = target_date if target_date else today_jst_str()
        today_clean = today_str.replace("-", "")
        now_dt = now_jst()

        today_info = get_today_races(today_clean)
        venues = today_info.get("venues", [])

        all_races = []
        for v in venues:
            for r in v.get("races", []):
                all_races.append(r)

        batch_status["total"] = len(all_races)

        if not all_races:
            batch_status["message"] = f"{today_str}: 開催レースが見つかりませんでした。"
            return

        for idx, race_item in enumerate(all_races, 1):
            r_id = race_item["race_id"]
            r_info = race_item.get("race_info", "")
            batch_status["current"] = idx
            batch_status["message"] = f"処理中 ({idx}/{len(all_races)}R: 予想{batch_status['predicted']}件 / 確定{batch_status['settled']}件)"

            # 1. AI予想の実行 (未キャッシュまたはforce_predictの場合)
            if mode in ("all", "predict"):
                try:
                    if force_predict:
                        predict_race(r_id, force_refresh=True)
                        batch_status["predicted"] += 1
                    else:
                        cached_p = get_cached_prediction(r_id)
                        if not cached_p:
                            predict_race(r_id)
                            batch_status["predicted"] += 1
                except Exception as pe:
                    print(f"[Batch] 予想スキップ/エラー ({r_id}): {pe}")

            # 2. 確定済みなら結果検証して収支台帳へ記録
            if mode in ("all", "settle"):
                # 発走前の未来レースは結果照合をスキップ（無駄なnetkeibaスクレイピングとタイムアウトを完全防止）
                is_past_post = True
                m = re.search(r"(\d{1,2}):(\d{2})", r_info)
                if m and not is_19h_settle:
                    try:
                        p_hour, p_minute = int(m.group(1)), int(m.group(2))
                        p_dt = now_dt.replace(hour=p_hour, minute=p_minute, second=0, microsecond=0)
                        if now_dt < p_dt:
                            is_past_post = False
                    except Exception:
                        pass

                if is_past_post:
                    try:
                        settled = get_cached_settlement(r_id)
                        if not settled or (settled.get("actual_order") is None and not settled.get("is_pass")):
                            verify_race_result(r_id, is_19h_settle=is_19h_settle)
                            batch_status["settled"] += 1
                    except Exception:
                        pass

        batch_status["message"] = f"完了: 全{len(all_races)}R (予想{batch_status['predicted']}件 / 確定検証{batch_status['settled']}件)"

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

    # デーモンスレッドで完全に非同期切り離し実行（クライアントには即時200応答を返すためタイムアウトしない）
    thread = threading.Thread(
        target=run_batch_worker,
        args=(target_date, mode, force_predict),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "message": f"バックグラウンドで全レース一括処理を開始しました (モード: {mode})。"
    }


@app.post("/batch/today", response_model=BatchStartResponse)
def run_today_batch(
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)"),
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


# =========================================================================
# ★ バックグラウンドメンテナンスループ (JST完全同期版)
# =========================================================================

async def sync_daily_schedules_if_needed(today_str: str):
    """当日のレース発走時刻を走査し、keiba_scheduled_racesに事前登録"""
    if not supabase:
        return
    try:
        res = supabase.table("keiba_scheduled_races").select("race_id").eq("race_date", today_str).execute()
        # 0405など別日のレースが混入している場合はカウントから除外
        valid_ids = [r["race_id"] for r in (res.data or []) if len(r["race_id"]) == 12 and r["race_id"][6:10] != "0405"]
        if len(valid_ids) >= 20:
            return  # すでに本日のスケジュールが20R以上登録済み

        print(f"[Scheduler] {today_str} (JST) のレーススケジュールを事前走査・登録します...")
        today_clean = today_str.replace("-", "")
        today_info = get_today_races(today_clean)
        venues = today_info.get("venues", [])

        for v in venues:
            for r in v.get("races", []):
                r_id = r["race_id"]
                try:
                    fetch_shutuba_table(r_id, race_date=today_str)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Scheduler] スケジュール事前登録エラー: {e}")


async def background_maintenance_loop():
    print("[Scheduler] リアルタイム・メンテナンススケジューラー (JST同期) を開始しました。")
    last_cleaned_date = None
    last_19h_settle_date = None

    while True:
        try:
            # JST（日本時間）ベースで日時を取得
            now_dt = now_jst_naive()
            today_str = today_jst_str()

            # --- 1. JST朝 8:00 以降、当日のレーススケジュールを事前登録 ---
            if now_dt.hour >= 8:
                await sync_daily_schedules_if_needed(today_str)

            if supabase:
                # --- 2. 各レース【発走30分前】の自動予想実行 ---
                try:
                    res_pred = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("race_date", today_str)\
                        .eq("is_predicted", 0)\
                        .execute()
                    unpredicted_races = res_pred.data or []

                    for r in unpredicted_races:
                        r_id = r["race_id"]
                        p_dt_str = r["post_datetime"]
                        try:
                            post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue

                        trigger_predict_time = post_dt - datetime.timedelta(minutes=30)
                        if now_dt >= trigger_predict_time:
                            print(f"[Auto-Predict] 発走30分前検知: {r_id} ({r.get('race_name')}) の予想を実行します...")
                            try:
                                predict_race(r_id, force_refresh=True)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_predicted": 1})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Predict] 予想完了: {r_id}")
                            except Exception as pe:
                                print(f"[Auto-Predict] 予想実行失敗 ({r_id}): {pe}")
                except Exception as e:
                    print(f"[Scheduler] 予想ループエラー: {e}")

                # --- 3. 各レース【発走60分後】の個別自動照合 ---
                try:
                    res_settle = supabase.table("keiba_scheduled_races")\
                        .select("*")\
                        .eq("is_settled", 0)\
                        .execute()
                    unsettled_races = res_settle.data or []

                    for r in unsettled_races:
                        r_id = r["race_id"]
                        p_dt_str = r["post_datetime"]
                        retries = r.get("retry_count", 0)
                        try:
                            post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            continue

                        trigger_settle_time = post_dt + datetime.timedelta(minutes=60)
                        if now_dt >= trigger_settle_time:
                            try:
                                verify_race_result(r_id)
                                supabase.table("keiba_scheduled_races")\
                                    .update({"is_settled": 1})\
                                    .eq("race_id", r_id)\
                                    .execute()
                                print(f"[Auto-Settle] 結果照合完了: {r_id}")
                            except Exception:
                                if retries >= 10:
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

            # --- 4. JST 19:00 の未確定確認 ---
            if now_dt.hour == 19 and last_19h_settle_date != today_str:
                if not batch_status["is_running"]:
                    print(f"[Scheduler] 19:00 当日レースの最終結果照合を実行します...")
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: run_batch_worker(today_str, "all", False, True))
                    last_19h_settle_date = today_str

            # --- 5. JST 0:00 (深夜0時) の一時キャッシュ削除 ---
            if now_dt.hour == 0 and last_cleaned_date != today_str:
                if supabase:
                    try:
                        supabase.table("keiba_predictions_cache").delete().lt("race_date", today_str).execute()
                        supabase.table("keiba_race_results_cache").delete().lt("race_date", today_str).execute()
                        supabase.table("keiba_scheduled_races").delete().lt("race_date", today_str).execute()
                        print(f"[Cleanup] JST 0:00 前日以前の一時キャッシュを削除しました。")
                    except Exception as ce:
                        print(f"[Cleanup] 失敗: {ce}")
                last_cleaned_date = today_str

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
CURRENT_DB_VERSION = "v4_flexible_multi_tickets"

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


@app.on_event("startup")
async def startup_event():
    # DB構成（バージョン）に変更があった時のみ初期化し、通常の再起動時はデータを完全保持
    check_and_apply_db_migration()
    asyncio.create_task(background_maintenance_loop())
