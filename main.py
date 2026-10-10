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

MODEL_PATH = "lgb_model.txt"
DICT_PATH = "stats_dict.json"
DB_PATH = "results_cache.db"

# --- SQLite データベース初期化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions_cache (
            race_id TEXT PRIMARY KEY,
            race_date TEXT,
            data_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS race_results_cache (
            race_id TEXT PRIMARY KEY,
            race_name TEXT,
            race_date TEXT,
            result_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
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

stats_dict = {"jockey_stats": {}, "jockey_course_stats": {}, "trainer_stats": {}, "trainer_course_stats": {}}
if os.path.exists(DICT_PATH):
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            stats_dict = json.load(f)
        print("統計辞書を正常にロードしました。")
    except Exception as e:
        print(f"統計辞書のロードに失敗しました: {e}")

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

class BatchProcessResult(BaseModel):
    date: str
    total_found_races: int
    predicted_races: int
    settled_races: int
    message: str

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
