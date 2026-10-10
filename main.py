from typing import List, Optional, Dict, Any
import os
import datetime
import re
import json
import sqlite3
import asyncio
import requests
import lightgbm as lgb
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Keiba Prediction & Verification API")

HEADERS_PC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/"
}

# --- モデル & 統計辞書のロード ---
MODEL_PATH = "lgb_model.txt"
DICT_PATH = "stats_dict.json"
DB_PATH = "results_cache.db"

# --- SQLite データベース初期化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. 出馬表・推論結果キャッシュテーブル（当日アクセス高速化用・0時にクリア）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_cache (
            race_id TEXT PRIMARY KEY,
            race_date TEXT,
            data_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 2. 確定レース結果一時キャッシュテーブル（0時にクリア）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_results_cache (
            race_id TEXT PRIMARY KEY,
            race_name TEXT,
            race_date TEXT,
            result_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. 確定収支台帳テーブル（永続保存：0時になっても削除しない）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_settlements (
            race_id TEXT PRIMARY KEY,
            race_date TEXT,
            race_name TEXT,
            target_umaban INTEGER,
            horse_name TEXT,
            grade TEXT,
            is_pass INTEGER,
            tansho_bet INTEGER,
            fukusho_bet INTEGER,
            total_bet INTEGER,
            actual_order INTEGER,
            tansho_payout INTEGER,
            fukusho_payout INTEGER,
            total_payout INTEGER,
            is_hit INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 4. 発走後自動照合用スケジュールテーブル（0時にクリア）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_races (
            race_id TEXT PRIMARY KEY,
            race_date TEXT,
            race_name TEXT,
            post_datetime TEXT,
            is_settled INTEGER DEFAULT 0,
            retry_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
        print("LightGBMモデルを正常にロードしました。")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
else:
    print("モデルファイルが見つかりません。")

stats_dict = {"jockey_stats": {}, "jockey_course_stats": {}, "trainer_stats": {}, "trainer_course_stats": {}}
if os.path.exists(DICT_PATH):
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        print("騎手・調教師統計辞書を正常にロードしました。")
    except Exception as e:
        print(f"統計辞書のロードに失敗しました: {e}")
else:
    print("stats_dict.json が見つかりません。デフォルト値(0.25)を使用します。")

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
    grade: str                   # "S", "A", "B", "C", "D"
    confidence_label: str        # "鉄板・大勝負", "勝負レース", "標準推奨", "少額推奨", "見送り推奨", "見送り推奨(低オッズ)"
    is_pass: bool                # 見送り判定フラグ
    tansho_amount: int           # 単勝購入額 (円)
    fukusho_amount: int          # 複勝購入額 (円)
    total_amount: int            # 合計購入額 (円)
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


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Keiba Prediction & Verification API is running"}


# --- 自動スケジュール登録ヘルパー ---
def register_race_schedule(race_id: str, race_name: str, post_time_str: str):
    try:
        m = re.search(r"(\d{1,2}):(\d{2})", post_time_str)
        if not m:
            return
        hour, minute = int(m.group(1)), int(m.group(2))
        today_date = datetime.date.today().strftime("%Y-%m-%d")
        post_dt = datetime.datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        post_dt_str = post_dt.strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT OR IGNORE INTO scheduled_races (race_id, race_date, race_name, post_datetime, is_settled)
            VALUES (?, ?, ?, ?, 0)
        """, (race_id, today_date, race_name, post_dt_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"スケジュール登録エラー ({race_id}): {e}")


# --- 開催日レース一覧 (GET /races/today) ---
def parse_netkeiba_sp_page(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    venues_data = []

    race_links = soup.find_all("a", href=re.compile(r"race_id=(\d{12})"))
    if not race_links:
        return []

    VENUE_CODE_MAP = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }

    raw_venue_races = {}
    seen_ids = set()

    for a in race_links:
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

        if venue_name not in raw_venue_races:
            raw_venue_races[venue_name] = []

        raw_venue_races[venue_name].append({
            "race_no": race_no,
            "race_id": race_id,
            "kai_day": kai_day,
            "race_num_int": race_num_int,
            "race_name": clean_text if clean_text else f"{race_no}",
            "race_info": ""
        })

    for v_name, r_list in raw_venue_races.items():
        earliest_kai_day = min(r["kai_day"] for r in r_list)
        day_races = [r for r in r_list if r["kai_day"] == earliest_kai_day]
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
    today = datetime.date.today()
    if date:
        clean_date = date.replace("-", "")
        target_dates = [clean_date]
    else:
        target_dates = [
            (today + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range(7)
        ]

    for d_str in target_dates:
        url = f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={d_str}"
        try:
            resp = requests.get(url, headers=HEADERS_PC, timeout=10)
            try:
                text = resp.content.decode("euc-jp")
            except UnicodeDecodeError:
                text = resp.content.decode("utf-8", errors="replace")

            venues = parse_netkeiba_sp_page(text)
            if venues:
                formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
                return {"date": formatted_date, "venues": venues}
        except Exception:
            continue

    return {
        "date": today.strftime("%Y-%m-%d"),
        "message": "直近7日間の開催情報が見つかりませんでした。",
        "venues": []
    }


# --- 出馬表取得 ---
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

    if post_time:
        register_race_schedule(race_id, race_name, post_time)

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
    except Exception as e:
        print(f"オッズAPI取得スキップ ({race_id}): {e}")

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


# --- 動的資金配分ロジック (単勝2.0倍未満ケン判定付き) ---
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


# --- 19特徴量推論 ＆ 動的資金配分推奨生成 ---
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
        except Exception as e:
            print(f"推論エラー (フォールバック計算に移行): {e}")
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
# ★ 予想キャッシュ（当日初回アクセス時は保存、以降は高速参照）
# =========================================================================

def get_cached_prediction(race_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT data_json FROM predictions_cache WHERE race_id = ?", (race_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_cached_prediction(race_id: str, race_date: str, data_dict: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # 修正: プレースホルダーを3つに修正
    cur.execute("""
        INSERT OR REPLACE INTO predictions_cache (race_id, race_date, data_json)
        VALUES (?, ?, ?)
    """, (race_id, race_date, json.dumps(data_dict, ensure_ascii=False)))
    conn.commit()
    conn.close()


# --- 推論エンドポイント (キャッシュ優先 ＆ force_refresh対応) ---
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

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        save_cached_prediction(race_id, today_str, resp_data)

        return resp_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")


# =========================================================================
# ★ 結果検証 ＆ 確定後キャッシュ永続化 ＆ 当日回収率集計
# =========================================================================

def get_cached_settlement(race_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM race_settlements WHERE race_id = ?", (race_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        d = dict(row)
        profit = d["total_payout"] - d["total_bet"]
        recovery = round((d["total_payout"] / d["total_bet"]) * 100.0, 2) if d["total_bet"] > 0 else 0.0
        
        if d["is_pass"]:
            msg = f"見送り推奨レースです。(本命{d['target_umaban']}番は{d['actual_order']}着 / 資金保全成功)" if d["actual_order"] else "見送り推奨レースのため、投資・払戻はありません。"
        elif d["is_hit"]:
            msg = f"{d['horse_name']} は {d['actual_order']}着でした。 的中！ 払戻: {d['total_payout']:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円)"
        else:
            order_str = f"{d['actual_order']}着" if (d['actual_order'] and d['actual_order'] < 90) else "着外"
            msg = f"{d['horse_name']} は {order_str}でした。 不的中 (収支: {profit:,}円)"

        return {
            "race_id": d["race_id"],
            "race_name": d["race_name"],
            "is_pass": bool(d["is_pass"]),
            "grade": d["grade"],
            "target_umaban": d["target_umaban"],
            "horse_name": d["horse_name"],
            "actual_order": d["actual_order"],
            "is_hit": bool(d["is_hit"]),
            "total_bet": d["total_bet"],
            "total_payout": d["total_payout"],
            "profit": profit,
            "recovery_rate": recovery,
            "from_cache": True,
            "message": msg
        }
    return None


def get_cached_result(race_id: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT result_json FROM race_results_cache WHERE race_id = ?", (race_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def save_cached_result(race_id: str, race_name: str, race_date: str, result_dict: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO race_results_cache (race_id, race_name, race_date, result_json)
        VALUES (?, ?, ?, ?)
    """, (race_id, race_name, race_date, json.dumps(result_dict, ensure_ascii=False)))
    conn.commit()
    conn.close()


def save_settlement(data: dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO race_settlements (
            race_id, race_date, race_name, target_umaban, horse_name, grade,
            is_pass, tansho_bet, fukusho_bet, total_bet, actual_order,
            tansho_payout, fukusho_payout, total_payout, is_hit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["race_id"], data["race_date"], data["race_name"], data["target_umaban"],
        data["horse_name"], data["grade"], 1 if data["is_pass"] else 0,
        data["tansho_bet"], data["fukusho_bet"], data["total_bet"], data["actual_order"],
        data["tansho_payout"], data["fukusho_payout"], data["total_payout"], 1 if data["is_hit"] else 0
    ))
    conn.commit()
    conn.close()


def fetch_netkeiba_race_result(race_id: str) -> Optional[dict]:
    orders = {}
    payouts = {"tansho": {}, "fukusho": {}}
    race_name = f"Race {race_id}"

    # 1. SP版速報ページ
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
    except Exception as e:
        print(f"SP版取得エラー: {e}")

    # 2. PC版速報ページ (フォールバック)
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
        except Exception as e:
            print(f"PC版取得エラー: {e}")

    # 3. db.netkeiba.com (アーカイブ用フォールバック)
    if not payouts["tansho"] and not any(v == 1 for v in orders.values()):
        db_url = f"https://db.netkeiba.com/race/{race_id}/"
        try:
            resp = requests.get(db_url, headers=HEADERS_PC, timeout=10)
            try:
                html = resp.content.decode("euc-jp")
            except UnicodeDecodeError:
                html = resp.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")

            for tr in soup.select("table.race_table_01 tr"):
                tds = tr.find_all("td")
                if len(tds) >= 4:
                    o_txt = tds[0].get_text(strip=True)
                    u_txt = tds[2].get_text(strip=True)
                    if o_txt.isdigit() and u_txt.isdigit():
                        orders[int(u_txt)] = int(o_txt)

            for tr in soup.select("table.pay_table_01 tr"):
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
        except Exception as e:
            print(f"DB版取得エラー: {e}")

    is_confirmed = (len(payouts["tansho"]) > 0) or (1 in orders.values())

    if len(payouts["tansho"]) > 0 and 1 not in orders.values():
        for winner_u in payouts["tansho"].keys():
            orders[winner_u] = 1

    print(f"[{race_id}] パース完了: 確定判定={is_confirmed} / 単勝払戻={payouts['tansho']} / 着順={orders}")

    if not is_confirmed:
        return None

    return {
        "race_name": race_name,
        "orders": orders,
        "payouts": payouts
    }


# --- 個別レース結果検証エンドポイント (確定後はミリ秒即時リターン) ---
@app.get("/result/{race_id}", response_model=VerificationResult)
def verify_race_result(race_id: str):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    # 1. すでに確定・検証済みの場合は、推論もスクレイピングも一切スキップして即時返却 (5ms)
    cached_settlement = get_cached_settlement(race_id)
    if cached_settlement:
        return cached_settlement

    r_year = race_id[:4]
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    race_date = f"{r_year}-{today_str[5:7]}-{today_str[8:]}"

    # 2. 結果取得
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

    # 3. 推論結果の取得（キャッシュから読み出されるため高速）
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

    # 見送り判定の場合
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
            "message": f"見送り推奨レースです。(本命{target_u}番は{actual_order}着 / 資金保全成功)" if actual_order else "見送り推奨レースのため、投資・払戻はありません。"
        }

    # 4. 確定着順と払戻の照合
    actual_order = race_res["orders"].get(target_u, 99)
    tansho_unit_payout = race_res["payouts"]["tansho"].get(target_u, 0)
    fukusho_unit_payout = race_res["payouts"]["fukusho"].get(target_u, 0)

    payout_t = int((b_tan / 100.0) * tansho_unit_payout) if actual_order == 1 else 0
    payout_f = int((b_fuku / 100.0) * fukusho_unit_payout) if (actual_order <= 3 or fukusho_unit_payout > 0) else 0
    total_payout = payout_t + payout_f

    profit = total_payout - total_bet
    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    is_hit = (actual_order <= 3) or (total_payout > 0)

    # 収支台帳へ記録（永続保存）
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
        "actual_order": actual_order,
        "tansho_payout": payout_t,
        "fukusho_payout": payout_f,
        "total_payout": total_payout,
        "is_hit": is_hit
    }
    save_settlement(settlement)

    order_str = f"{actual_order}着" if actual_order < 90 else "着外"
    msg = f"{h_name} は {order_str}でした。 不的中 (収支: {profit:,}円)"
    if is_hit:
        msg += f" 的中！ 払戻: {total_payout:,}円 (収支: {'+' if profit >= 0 else ''}{profit:,}円)"
    else:
        msg += f" 不的中 (収支: {profit:,}円)"

    return {
        "race_id": race_id,
        "race_name": race_name,
        "is_pass": False,
        "grade": grade,
        "target_umaban": target_u,
        "horse_name": h_name,
        "actual_order": actual_order if actual_order != 99 else None,
        "is_hit": is_hit,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "recovery_rate": recovery_rate,
        "from_cache": from_cache,
        "message": msg
    }


# --- 当日・指定日の総合成績・回収率集計エンドポイント ---
@app.get("/summary/today", response_model=DailySummary)
def get_daily_summary(date: Optional[str] = Query(None, description="集計対象日付 (YYYY-MM-DD)")):
    target_date = date if date else datetime.date.today().strftime("%Y-%m-%d")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM race_settlements
        WHERE race_date = ?
        ORDER BY race_id ASC
    """, (target_date,))
    rows = cur.fetchall()
    conn.close()

    total_races = len(rows)
    bet_races = sum(1 for r in rows if r["is_pass"] == 0)
    pass_races = sum(1 for r in rows if r["is_pass"] == 1)
    hit_races = sum(1 for r in rows if r["is_hit"] == 1 and r["is_pass"] == 0)

    total_bet = sum(r["total_bet"] for r in rows)
    total_payout = sum(r["total_payout"] for r in rows)
    total_profit = total_payout - total_bet

    recovery_rate = round((total_payout / total_bet) * 100.0, 2) if total_bet > 0 else 0.0
    hit_rate = round((hit_races / bet_races) * 100.0, 2) if bet_races > 0 else 0.0

    settled_list = [dict(r) for r in rows]

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
        "settled_races": settled_list
    }


# =========================================================================
# ★ 出走1時間後自動照合 ＆ 毎日0時の一時キャッシュ自動クリーンアップ
# =========================================================================

async def background_maintenance_loop():
    print("[Scheduler] メンテナンススケジューラーを開始しました。")
    last_cleaned_date = None

    while True:
        try:
            now = datetime.datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # 毎日 0:00 の一時キャッシュ削除 (収支台帳は残す)
            if now.hour == 0 and last_cleaned_date != today_str:
                print(f"[Cleanup] 0:00 定期クリーンアップを開始します (収支台帳は保持)")
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("DELETE FROM predictions_cache WHERE race_date < ?", (today_str,))
                cur.execute("DELETE FROM race_results_cache WHERE race_date < ?", (today_str,))
                cur.execute("DELETE FROM scheduled_races WHERE race_date < ?", (today_str,))
                cur.execute("VACUUM")
                conn.commit()
                conn.close()
                last_cleaned_date = today_str
                print(f"[Cleanup] 前日以前の一時キャッシュを削除しました。")

            # 発走1時間後の自動照合巡回
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT race_id, race_name, post_datetime, retry_count
                FROM scheduled_races
                WHERE is_settled = 0
            """)
            scheduled_races = cur.fetchall()
            conn.close()

            for r in scheduled_races:
                r_id = r["race_id"]
                p_dt_str = r["post_datetime"]
                retries = r["retry_count"]

                try:
                    post_dt = datetime.datetime.strptime(p_dt_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

                trigger_time = post_dt + datetime.timedelta(minutes=60)
                if now >= trigger_time:
                    try:
                        verify_race_result(r_id)
                        conn_up = sqlite3.connect(DB_PATH)
                        cur_up = conn_up.cursor()
                        cur_up.execute("UPDATE scheduled_races SET is_settled = 1 WHERE race_id = ?", (r_id,))
                        conn_up.commit()
                        conn_up.close()
                        print(f"[Scheduler] 自動照合完了: {r_id}")
                    except Exception as err:
                        conn_up = sqlite3.connect(DB_PATH)
                        cur_up = conn_up.cursor()
                        if retries >= 10:
                            cur_up.execute("UPDATE scheduled_races SET is_settled = 1 WHERE race_id = ?", (r_id,))
                        else:
                            cur_up.execute("UPDATE scheduled_races SET retry_count = retry_count + 1 WHERE race_id = ?", (r_id,))
                        conn_up.commit()
                        conn_up.close()

        except Exception as e:
            print(f"[Scheduler] ループエラー: {e}")

        await asyncio.sleep(90)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_maintenance_loop())
