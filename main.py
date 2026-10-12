from typing import List, Optional, Dict, Any
import os
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
        print("LightGBMモデルを正常にロードしました。")
    except Exception as e:
        print(f"モデルロード失敗: {e}")

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
    memo: Optional[str] = None

class HybridRecommendation(BaseModel):
    is_participate: bool
    strategy_name: str = "馬連・超高配当三連複ハイブリッド"
    total_amount: int
    summary: str
    umaren_bets: List[TicketBet] = []
    trio_bet: Optional[TicketBet] = None
    reason: Optional[str] = None

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

    if model:
        try:
            scores = model.predict(feature_rows)
            for idx, h in enumerate(horses):
                h["score"] = round(float(scores[idx]), 4)
        except Exception:
            for idx, h in enumerate(horses):
                odds = h["odds"] if h["odds"] else 50.0
                h["score"] = round(float(1.0 / (1.0 + (odds ** 0.5))), 4)
    else:
        for idx, h in enumerate(horses):
            odds = h["odds"] if h["odds"] else 50.0
            base_score = 1.0 / (1.0 + (odds ** 0.5))
            h["score"] = round(float(base_score), 4)

    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)

    for h in sorted_horses:
        h["mark"] = "-"

    base_marks = ["◎", "◯", "▲", "△", "×"]
    for idx in range(min(5, len(sorted_horses))):
        sorted_horses[idx]["mark"] = base_marks[idx]

    if len(sorted_horses) >= 6:
        sorted_horses[5]["mark"] = "☆"
        base_hoshi_score = sorted_horses[5]["score"]
        DIFF_THRESHOLD = 0.015

        if len(sorted_horses) >= 7:
            if (base_hoshi_score - sorted_horses[6]["score"]) <= DIFF_THRESHOLD:
                sorted_horses[6]["mark"] = "☆"
                if len(sorted_horses) >= 8:
                    if (base_hoshi_score - sorted_horses[7]["score"]) <= DIFF_THRESHOLD:
                        sorted_horses[7]["mark"] = "☆"

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
    hybrid_rec: Optional[dict] = None
    if len(sorted_horses) >= 3:
        h0, h1, h2 = sorted_horses[0], sorted_horses[1], sorted_horses[2]
        s0 = float(h0.get("score", 0.0))
        o0 = float(h0.get("odds") or 0.0)
        o1 = float(h1.get("odds") or 0.0)
        o2 = float(h2.get("odds") or 0.0)
        u0, u1, u2 = int(h0["umaban"]), int(h1["umaban"]), int(h2["umaban"])

        est_u01 = round(max(2.5, 0.65 * o0 * o1), 1) if (o0 > 0 and o1 > 0) else 0.0
        est_u02 = round(max(2.5, 0.65 * o0 * o2), 1) if (o0 > 0 and o2 > 0) else 0.0

        can_bet = True
        reasons = []
        if surface_type == 2:
            can_bet = False
            reasons.append("障害レース除外")
        if not (1.5 <= o0 <= 2.8):
            can_bet = False
            reasons.append(f"◎単勝オッズ範囲外({o0}倍: 推奨1.5〜2.8倍)")
        if s0 < 0.52:
            can_bet = False
            reasons.append(f"◎AIスコア不足({s0:.3f}: 推奨0.520以上)")
        if est_u01 < 3.2:
            can_bet = False
            reasons.append(f"本線想定馬連配当不足({est_u01}倍: 推奨3.2倍以上)")

        if not can_bet:
            hybrid_rec = {
                "is_participate": False,
                "strategy_name": "馬連・超高配当三連複ハイブリッド",
                "total_amount": 0,
                "summary": "見送り推奨 (期待値基準外)",
                "umaren_bets": [],
                "trio_bet": None,
                "reason": " / ".join(reasons)
            }
        else:
            umaren_bets = [
                {
                    "ticket_type": "馬連",
                    "combination": [u0, u1],
                    "combination_str": f"{u0}-{u1}",
                    "amount": 300,
                    "odds": est_u01,
                    "memo": "本線 (300円)"
                },
                {
                    "ticket_type": "馬連",
                    "combination": [u0, u2],
                    "combination_str": f"{u0}-{u2}",
                    "amount": 100,
                    "odds": est_u02,
                    "memo": "抑え (100円)"
                }
            ]

            # 三連複相手（★）の選定: 3〜6位の中で想定配当 >= 35倍
            # ★初ブリンカー馬がいれば最優先でヒモ穴に指名！
            trio_target = None
            candidates = []
            for idx in range(2, min(7, len(sorted_horses))):
                hx = sorted_horses[idx]
                ox = float(hx.get("odds") or 0.0)
                if ox <= 0:
                    continue
                est_trio = round(calc_trio_odds(o0, o1, ox), 1)
                if est_trio >= 35.0:
                    candidates.append({
                        "horse": hx,
                        "odds": est_trio,
                        "is_first_blinker": bool(hx.get("is_first_blinker", False))
                    })

            if candidates:
                candidates.sort(key=lambda c: (1 if c["is_first_blinker"] else 0), reverse=True)
                best_c = candidates[0]
                target_h = best_c["horse"]
                target_ux = int(target_h["umaban"])
                target_odds = best_c["odds"]
                blinker_note = "【初ブリンカー激走期待！】" if best_c["is_first_blinker"] else ""

                trio_target = {
                    "ticket_type": "三連複",
                    "combination": [u0, u1, target_ux],
                    "combination_str": f"{u0}-{u1}-{target_ux}",
                    "amount": 100,
                    "odds": target_odds,
                    "memo": f"超高配当ボーナス{blinker_note} (想定{target_odds}倍 / 相手{target_ux}番:{target_h['horse_name']})"
                }

            total_amt = 400 + (100 if trio_target else 0)
            trio_summary = f" ＋ 三連複1点({trio_target['combination_str']}: 100円)" if trio_target else " (三連複は見送り)"

            hybrid_rec = {
                "is_participate": True,
                "strategy_name": "馬連・超高配当三連複ハイブリッド",
                "total_amount": total_amt,
                "summary": f"参戦推奨: 馬連2点(400円){trio_summary} / 計{total_amt}円",
                "umaren_bets": umaren_bets,
                "trio_bet": trio_target,
                "reason": "基準クリア（◎勝負気配 ＆ 本線妙味あり）"
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


def calculate_hybrid_pass_benefit_info(
    h_rec: Optional[dict],
    race_res: Optional[dict],
    target_u: Optional[int],
    actual_order: Optional[int]
) -> dict:
    """見送りレースにおける『馬連＋三連複』仮想損益・見送り効果の算出"""
    hypo_bet = 400 + (100 if (h_rec and h_rec.get("trio_bet")) else 0)
    hypo_pay = 0
    payouts_data = race_res.get("payouts", {}) if race_res else {}

    hit_details = []
    if h_rec and h_rec.get("umaren_bets"):
        for ub in h_rec.get("umaren_bets", []):
            c = ub.get("combination", [])
            if len(c) == 2:
                ckey = f"{min(c[0], c[1])}-{max(c[0], c[1])}"
                unit_p = payouts_data.get("umaren", {}).get(ckey, 0)
                if unit_p > 0:
                    pay = int((ub.get("amount", 100) / 100.0) * unit_p)
                    hypo_pay += pay
                    hit_details.append(f"馬連{ckey}({pay:,}円)")
        tb = h_rec.get("trio_bet")
        if tb and tb.get("combination"):
            tc = sorted(tb.get("combination", []))
            if len(tc) == 3:
                tkey = f"{tc[0]}-{tc[1]}-{tc[2]}"
                unit_tp = payouts_data.get("trio", {}).get(tkey, 0)
                if unit_tp > 0:
                    pay = int((tb.get("amount", 100) / 100.0) * unit_tp)
                    hypo_pay += pay
                    hit_details.append(f"三連複{tkey}({pay:,}円)")

    pass_benefit = hypo_bet - hypo_pay
    is_benefit_pos = pass_benefit >= 0
    order_str = f"{actual_order}着でした。" if (actual_order and actual_order < 90) else "着外でした。"
    reason_str = h_rec.get("reason", "期待値基準外") if h_rec else "見送り推奨"
    u_str = f"◎{target_u}番は{order_str} / " if target_u else ""
    hit_summary = f" ({' ＆ '.join(hit_details)})" if hit_details else " (払戻0円)"

    msg = f"見送り推奨レースです。(判定: {reason_str} / {u_str}資金保全)\n【仮想ハイブリッド比較】仮想投資: {hypo_bet:,}円 / 仮想払戻: {hypo_pay:,}円{hit_summary} → 見送り損益: {pass_benefit:+,}円 ({'資金保全成功！' if is_benefit_pos else '機会損失'})"

    return {
        "hypothetical_bet": hypo_bet,
        "hypothetical_payout": hypo_pay,
        "pass_benefit": pass_benefit,
        "actual_order": actual_order,
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

            hit_type = d.get("hit_type")
            u_bet = d.get("umaren_bet") or 0
            t_bet = d.get("trio_bet") or 0
            u_pay = d.get("umaren_payout") or 0
            t_pay = d.get("trio_payout") or 0

            if d["is_pass"]:
                res_cache = get_cached_result(race_id)
                pred_cache = get_cached_prediction(race_id)
                h_rec = pred_cache.get("hybrid_recommendation") if pred_cache else None
                pass_info = calculate_hybrid_pass_benefit_info(h_rec, res_cache, d.get("target_umaban"), order)
                hypo_bet = pass_info["hypothetical_bet"]
                hypo_pay = pass_info["hypothetical_payout"]
                pass_benefit = pass_info["pass_benefit"]
                msg = pass_info["message"]
            elif d["is_hit"]:
                hit_label = hit_type if hit_type else "的中"
                p_details = d.get("payout_details")
                if isinstance(p_details, str):
                    try:
                        p_details = json.loads(p_details)
                    except Exception:
                        p_details = []
                if isinstance(p_details, list) and len(p_details) > 0:
                    hit_texts = [f"{h.get('type','')}{h.get('comb','')}({h.get('payout',0):,}円)" for h in p_details]
                    msg = f"【{hit_label}】 {' ＆ '.join(hit_texts)}！ 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery}%)"
                else:
                    msg = f"【{hit_label}】 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery}%)"
            else:
                msg = f"不的中 (投資: {d['total_bet']:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%)"

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
                "trio_payout": t_pay
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


@app.get("/result/{race_id}", response_model=VerificationResult)
def verify_race_result(race_id: str):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    cached_settlement = get_cached_settlement(race_id)
    if cached_settlement:
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

    cached_data = get_cached_result(race_id)
    from_cache = False

    if cached_data:
        from_cache = True
        race_res = cached_data
    else:
        race_res = fetch_netkeiba_race_result(race_id)
        if not race_res:
            raise HTTPException(status_code=400, detail="レース結果がまだ確定していないか、取得できませんでした。発走後しばらくしてから再試行してください。")
        save_cached_result(race_id, race_res["race_name"], race_date, race_res)

    pred_data = predict_race(race_id)
    rec = pred_data.get("recommendation", {})
    hybrid_rec = pred_data.get("hybrid_recommendation")
    race_name = pred_data.get("race_name", race_res["race_name"])

    target_u = rec.get("target_umaban")
    h_name = rec.get("horse_name")
    grade = rec.get("grade", "B")
    actual_order = race_res["orders"].get(target_u) if target_u else None

    # ハイブリッド推奨を最優先で照合
    if not hybrid_rec or not hybrid_rec.get("is_participate", False):
        pass_info = calculate_hybrid_pass_benefit_info(hybrid_rec, race_res, target_u, actual_order)
        hypo_bet = pass_info["hypothetical_bet"]
        hypo_pay = pass_info["hypothetical_payout"]
        pass_benefit = pass_info["pass_benefit"]
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
            "fukusho_payout": 0
        }
        save_settlement(settlement)

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
            "trio_payout": 0
        }

    # 参戦レースの場合
    umaren_bets = hybrid_rec.get("umaren_bets", [])
    trio_bet = hybrid_rec.get("trio_bet")
    u_bet_tot = sum(b.get("amount", 0) for b in umaren_bets)
    t_bet_tot = trio_bet.get("amount", 0) if trio_bet else 0
    total_bet = u_bet_tot + t_bet_tot

    payouts_data = race_res.get("payouts", {})
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

    total_payout = umaren_payout + trio_payout
    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = total_payout > 0

    if len(umaren_hits) > 0 and len(trio_hits) > 0:
        hit_type = "W的中🔥"
    elif len(umaren_hits) > 0:
        hit_type = "馬連的中🎯"
    elif len(trio_hits) > 0:
        hit_type = "三連複的中🎯"
    else:
        hit_type = "不的中"

    all_hits = umaren_hits + trio_hits
    if is_hit:
        hit_texts = [f"{h['type']}{h['comb']}({h['payout']:,}円)" for h in all_hits]
        msg = f"【{hit_type}】{' ＆ '.join(hit_texts)}！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円 / 回収率: {recovery_rate}%)"
    else:
        msg = f"不的中 (投資: {total_bet:,}円 / 払戻: 0円 / 収支: {profit:,}円 / 回収率: 0.0%)"

    bet_breakdown = {"馬連": u_bet_tot}
    payout_breakdown = {"馬連": umaren_payout}
    if t_bet_tot > 0:
        bet_breakdown["三連複"] = t_bet_tot
        payout_breakdown["三連複"] = trio_payout

    settlement = {
        "race_id": race_id,
        "race_date": race_date,
        "race_name": race_name,
        "strategy_type": "hybrid",
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
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "tansho_bet": 0,
        "fukusho_bet": 0,
        "tansho_payout": 0,
        "fukusho_payout": 0
    }
    save_settlement(settlement)

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
        "strategy_type": "hybrid",
        "hit_type": hit_type,
        "bet_breakdown": bet_breakdown,
        "payout_breakdown": payout_breakdown,
        "bet_details": {
            "umaren_bets": umaren_bets,
            "trio_bet": trio_bet
        },
        "payout_details": all_hits,
        "umaren_bet": u_bet_tot,
        "trio_bet": t_bet_tot,
        "umaren_payout": umaren_payout,
        "trio_payout": trio_payout
    }


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
            pass_info = calculate_hybrid_pass_benefit_info(h_rec, c_res, r.get("target_umaban"), r.get("actual_order"))
            pb = pass_info["pass_benefit"]
            r["hypothetical_bet"] = pass_info["hypothetical_bet"]
            r["hypothetical_payout"] = pass_info["hypothetical_payout"]
            r["pass_benefit"] = pb
            pass_benefit_total += pb
            if pb > 0:
                pass_avoided_losses += pb
            elif pb < 0:
                pass_missed_profits += abs(pb)

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

def run_batch_worker(target_date: Optional[str] = None, mode: str = "all", force_predict: bool = False):
    global batch_status
    batch_status["is_running"] = True
    batch_status["current"] = 0
    batch_status["predicted"] = 0
    batch_status["settled"] = 0
    batch_status["message"] = "開催レース走査中..."

    try:
        today_str = target_date if target_date else today_jst_str()
        today_clean = today_str.replace("-", "")

        today_info = get_today_races(today_clean)
        venues = today_info.get("venues", [])

        all_race_ids = []
        for v in venues:
            for r in v.get("races", []):
                all_race_ids.append(r["race_id"])

        batch_status["total"] = len(all_race_ids)

        if not all_race_ids:
            batch_status["message"] = f"{today_str}: 開催レースが見つかりませんでした。"
            return

        for idx, r_id in enumerate(all_race_ids, 1):
            batch_status["current"] = idx
            batch_status["message"] = f"処理中 ({idx}/{len(all_race_ids)}R: 予想{batch_status['predicted']}件 / 確定{batch_status['settled']}件)"

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
                try:
                    settled = get_cached_settlement(r_id)
                    if not settled:
                        verify_race_result(r_id)
                        batch_status["settled"] += 1
                except Exception:
                    pass

        batch_status["message"] = f"完了: 全{len(all_race_ids)}R (予想{batch_status['predicted']}件 / 確定検証{batch_status['settled']}件)"

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
                    await loop.run_in_executor(None, run_batch_worker, today_str)
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
