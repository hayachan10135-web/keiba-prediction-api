from typing import List, Optional, Dict, Any
import os
import datetime
import zoneinfo
import re
import json
import asyncio
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

stats_dict = {"jockey_stats": {}, "jockey_course_stats": {}, "trainer_stats": {}, "trainer_course_stats": {}}
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

    # 会場ごとに「レース数が最も多い kai_day」を採用（同数なら最新の max(kai_day) を採用）
    for v_name, r_list in raw_venue_races.items():
        counts = {}
        for r in r_list:
            kd = r["kai_day"]
            counts[kd] = counts.get(kd, 0) + 1

        best_kai_day = sorted(counts.keys(), key=lambda kd: (counts[kd], kd), reverse=True)[0]
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
        # PC版を優先取得し、フォールバックでSP版を使用
        urls = [
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={d_str}",
            f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={d_str}"
        ]
        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS_PC, timeout=10)
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


def fetch_shutuba_table(race_id: str):
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

    # スケジュール登録（JST日付を使用）
    if post_time:
        today_date = today_jst_str()
        register_race_schedule(race_id, today_date, race_name, post_time)

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

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "trainer": trainer,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    course_meta = {
        "distance": distance,
        "surface_type": surface_type,
        "condition_code": condition_code
    }

    return race_name, race_details, sorted(horses, key=lambda x: x["umaban"]), course_meta


def calculate_dynamic_bet(score: float, odds: Optional[float] = None):
    if odds is not None and odds < 2.0:
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

    expected_num_features = 19
    if model:
        try:
            expected_num_features = model.num_feature()
        except Exception:
            expected_num_features = 19

    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    distance = course_meta.get("distance", 1600.0)
    surface_type = course_meta.get("surface_type", 0)
    condition_code = course_meta.get("condition_code", 0)
    distance_diff = 0.0

    course_key = f"{venue_code:02d}_{surface_type}"

    j_stats = stats_dict.get("jockey_stats", {})
    jc_stats = stats_dict.get("jockey_course_stats", {})
    t_stats = stats_dict.get("trainer_stats", {})
    tc_stats = stats_dict.get("trainer_course_stats", {})

    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0

        career_races = h.get("career_races", 5.0)
        career_top3_rate = h.get("career_top3_rate", 0.25)
        prev1_order = h.get("prev1_order", 5.0)
        prev1_pop = h.get("prev1_pop", 5.0)
        prev1_odds = h.get("prev1_odds", 15.0)
        prev2_order = h.get("prev2_order", 5.0)
        days_since_prev = h.get("days_since_prev", 35.0)

        j_clean = re.sub(r"[▲△◇☆\s]", "", h["jockey"])
        t_clean = re.sub(r"\[.*?\]|\s", "", h.get("trainer", "") or "")

        j_rate = j_stats.get(j_clean, 0.25)
        jc_rate = jc_stats.get(f"{j_clean}_{course_key}", j_rate)
        t_rate = t_stats.get(t_clean, 0.25)
        tc_rate = tc_stats.get(f"{t_clean}_{course_key}", t_rate)

        if expected_num_features == 19:
            feature_rows.append([
                w, u, k,
                venue_code, surface_type, distance, distance_diff, condition_code,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev,
                j_rate, jc_rate, t_rate, tc_rate
            ])
        else:
            feature_rows.append([
                w, u, k,
                venue_code, surface_type, distance, distance_diff, condition_code,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev
            ])

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
            if h_odds is not None and h_odds < 2.0:
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

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "race_details": race_details,
        "horses": horses_sorted_by_num,
        "recommendation": recommendation
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
        race_name, race_details, horses, course_meta = fetch_shutuba_table(race_id)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした。")

        result = build_prediction_and_recs(race_name, race_details, race_id, horses, course_meta)
        resp_data = {
            "race_id": race_id,
            "race_name": result["race_name"],
            "race_details": result["race_details"],
            "horses": result["horses"],
            "recommendation": result["recommendation"]
        }

        today_str = today_jst_str()
        save_cached_prediction(race_id, today_str, resp_data)

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


def get_cached_settlement(race_id: str) -> Optional[dict]:
    if not supabase:
        return None
    try:
        res = supabase.table("keiba_race_settlements").select("*").eq("race_id", race_id).execute()
        if res.data and len(res.data) > 0:
            d = res.data[0]
            profit = d["total_payout"] - d["total_bet"]
            recovery = round((d["total_payout"] / d["total_bet"]) * 100.0, 2) if d["total_bet"] > 0 else 0.0
            order = d["actual_order"]

            if order and 1 <= order <= 30:
                order_str = f"{order}着でした。"
            elif d["is_hit"]:
                order_str = "馬券圏内（3着以内）に入線しました。"
            else:
                order_str = "着外でした。"

            if d["is_pass"]:
                msg = f"見送り推奨レースです。(本命{d['target_umaban']}番は{order_str} / 資金保全成功)"
            elif d["is_hit"]:
                msg = f"{d['horse_name']} は {order_str} 的中！ 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円)"
            else:
                msg = f"{d['horse_name']} は {order_str} 不的中 (収支: {profit:,}円)"

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
                "message": msg
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
            "target_umaban": data["target_umaban"],
            "horse_name": data["horse_name"],
            "grade": data["grade"],
            "is_pass": 1 if data["is_pass"] else 0,
            "tansho_bet": data["tansho_bet"],
            "fukusho_bet": data["fukusho_bet"],
            "total_bet": data["total_bet"],
            "actual_order": data["actual_order"],
            "tansho_payout": data["tansho_payout"],
            "fukusho_payout": data["fukusho_payout"],
            "total_payout": data["total_payout"],
            "is_hit": 1 if data["is_hit"] else 0
        }).execute()
    except Exception as e:
        print(f"Supabase収支台帳保存エラー: {e}")


def fetch_netkeiba_race_result(race_id: str) -> Optional[dict]:
    orders = {}
    payouts = {"tansho": {}, "fukusho": {}}
    race_name = f"Race {race_id}"

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

    if not payouts["tansho"] or len(orders) < 3:
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
                    nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                    pays = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", td_pay.get_text()) if x.replace(",", "").isdigit()]
                    if "単勝" in kind:
                        for n, p in zip(nums, pays):
                            payouts["tansho"][n] = p
                    elif "複勝" in kind:
                        for n, p in zip(nums, pays):
                            payouts["fukusho"][n] = p

            for tr in soup.select("table.RaceTable01 tr, table.ResultTable tr, tr.HorseList"):
                o_elem = tr.select_one("td.Rank, td.Result_Num, div.Rank")
                u_elem = tr.select_one("td.Umaban, div.Umaban")
                if o_elem and u_elem:
                    o_txt = o_elem.get_text(strip=True)
                    u_txt = u_elem.get_text(strip=True)
                    if o_txt.isdigit() and u_txt.isdigit():
                        orders[int(u_txt)] = int(o_txt)
        except Exception:
            pass

    is_confirmed = (len(payouts["tansho"]) > 0) or (1 in orders.values())

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

    r_year = race_id[:4]
    today_str = today_jst_str()
    race_date = f"{r_year}-{today_str[5:7]}-{today_str[8:]}"

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
    rec = pred_data["recommendation"]
    race_name = pred_data["race_name"]

    target_u = rec["target_umaban"]
    h_name = rec["horse_name"]
    grade = rec["grade"]
    is_pass = rec["is_pass"]
    b_tan = rec["tansho_amount"]
    b_fuku = rec["fukusho_amount"]
    total_bet = rec["total_amount"]

    if is_pass or target_u is None:
        actual_order = race_res["orders"].get(target_u) if target_u else None
        settlement = {
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "target_umaban": target_u,
            "horse_name": h_name,
            "grade": grade,
            "is_pass": True,
            "tansho_bet": 0,
            "fukusho_bet": 0,
            "total_bet": 0,
            "actual_order": actual_order,
            "tansho_payout": 0,
            "fukusho_payout": 0,
            "total_payout": 0,
            "is_hit": False
        }
        save_settlement(settlement)
        order_str = f"{actual_order}着でした。" if actual_order else "着外でした。"
        return {
            "race_id": race_id,
            "race_name": race_name,
            "is_pass": True,
            "grade": grade,
            "target_umaban": target_u,
            "horse_name": h_name,
            "actual_order": actual_order,
            "is_hit": False,
            "total_bet": 0,
            "total_payout": 0,
            "profit": 0,
            "recovery_rate": 0.0,
            "from_cache": from_cache,
            "message": f"見送り推奨レースです。(本命{target_u}番は{order_str} / 資金保全成功)" if target_u else "見送り推奨レースのため、投資・払戻はありません。"
        }

    actual_order = race_res["orders"].get(target_u, 99)
    tansho_unit_payout = race_res["payouts"]["tansho"].get(target_u, 0)
    fukusho_unit_payout = race_res["payouts"]["fukusho"].get(target_u, 0)

    if tansho_unit_payout > 0:
        actual_order = 1
    elif fukusho_unit_payout > 0 and actual_order > 3:
        actual_order = 3

    payout_t = int((b_tan / 100.0) * tansho_unit_payout) if actual_order == 1 else 0
    payout_f = int((b_fuku / 100.0) * fukusho_unit_payout) if (actual_order <= 3 or fukusho_unit_payout > 0) else 0
    total_payout = payout_t + payout_f

    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = (actual_order <= 3) or (total_payout > 0)

    if 1 <= actual_order <= 30:
        order_str = f"{actual_order}着でした。"
    elif is_hit:
        order_str = "馬券圏内（3着以内）に入線しました。"
    else:
        order_str = "着外でした。"

    settlement = {
        "race_id": race_id,
        "race_date": race_date,
        "race_name": race_name,
        "target_umaban": target_u,
        "horse_name": h_name,
        "grade": grade,
        "is_pass": False,
        "tansho_bet": b_tan,
        "fukusho_bet": b_fuku,
        "total_bet": total_bet,
        "actual_order": actual_order if actual_order < 90 else None,
        "tansho_payout": payout_t,
        "fukusho_payout": payout_f,
        "total_payout": total_payout,
        "is_hit": is_hit
    }
    save_settlement(settlement)

    if is_hit:
        msg = f"{h_name} は {order_str} 的中！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円)"
    else:
        msg = f"{h_name} は {order_str} 不的中 (収支: {profit:,}円)"

    return {
        "race_id": race_id,
        "race_name": race_name,
        "is_pass": False,
        "grade": grade,
        "target_umaban": target_u,
        "horse_name": h_name,
        "actual_order": actual_order if actual_order < 90 else None,
        "is_hit": is_hit,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "recovery_rate": recovery_rate,
        "from_cache": from_cache,
        "message": msg
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

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r["is_pass"] == 0)
    pass_races = sum(1 for r in rows if r["is_pass"] == 1)
    hit_races = sum(1 for r in rows if r["is_hit"] == 1 and r["is_pass"] == 0)

    total_bet = sum(r["total_bet"] for r in rows)
    total_payout = sum(r["total_payout"] for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

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
        "settled_races": rows
    }


# =========================================================================
# ★ 手動一括処理（アプリボタン用・確定検証と現時点の予想登録）
# =========================================================================

def run_batch_worker(target_date: Optional[str] = None):
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
            batch_status["message"] = f"処理中 ({idx}/{len(all_race_ids)}レース)"

            # 確定済みなら結果検証して収支台帳へ記録
            try:
                settled = get_cached_settlement(r_id)
                if not settled:
                    verify_race_result(r_id)
                    batch_status["settled"] += 1
            except Exception:
                pass

        batch_status["message"] = f"完了: 全{len(all_race_ids)}R中、確定検証{batch_status['settled']}件"

    except Exception as e:
        batch_status["message"] = f"エラー終了: {e}"
    finally:
        batch_status["is_running"] = False


@app.post("/batch/today", response_model=BatchStartResponse)
def run_today_batch(
    background_tasks: BackgroundTasks,
    date: Optional[str] = Query(None, description="対象日付 (YYYY-MM-DD)")
):
    global batch_status
    if batch_status["is_running"]:
        return {
            "status": "already_running",
            "message": f"現在処理中です ({batch_status['current']}/{batch_status['total']}レース進行中)"
        }

    background_tasks.add_task(run_batch_worker, date)

    return {
        "status": "started",
        "message": "バックグラウンドで全レース一括照合を開始しました。"
    }


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
        if res.data and len(res.data) >= 12:
            return  # すでに本日のスケジュールが12R以上登録済み

        print(f"[Scheduler] {today_str} (JST) のレーススケジュールを事前走査・登録します...")
        today_clean = today_str.replace("-", "")
        today_info = get_today_races(today_clean)
        venues = today_info.get("venues", [])

        for v in venues:
            for r in v.get("races", []):
                r_id = r["race_id"]
                try:
                    fetch_shutuba_table(r_id)
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


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_maintenance_loop())
