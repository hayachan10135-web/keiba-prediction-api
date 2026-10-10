from typing import List, Optional, Set, Tuple
import os
import datetime
import re
import json
import requests
import lightgbm as lgb
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# FastAPI アプリケーション定義
app = FastAPI(title="Keiba Prediction API")

HEADERS_PC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://race.netkeiba.com/"
}

# --- LightGBM モデルのロード ---
MODEL_PATH = "lgb_model.txt"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
        print("LightGBMモデルを正常にロードしました。")
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
else:
    print("モデルファイルが見つかりません。フォールバックスコアリングを使用します。")

# --- Pydantic レスポンススキーマ ---
class HorsePrediction(BaseModel):
    umaban: int
    wakuban: int
    horse_name: str
    jockey: str
    kinryo: float
    odds: Optional[float]
    popularity: Optional[int]
    score: float
    mark: str

class Recommendations(BaseModel):
    tansho_fukusho: List[int]     # 単複(1頭): [◎の馬番]
    wide: List[str]               # ワイド(2点): ["◎-◯", "◎-▲"]
    sanrenpuku: List[str]         # 三連複フォーメーションの全買い目リスト ["8-9-7", "8-9-1", ...]
    sanrenpuku_summary: str       # フォーメーション概要表示用文字列

class PredictResponse(BaseModel):
    race_id: str
    race_name: str
    horses: List[HorsePrediction]
    recommendations: Recommendations


# --- 1. ヘルスチェック ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- 2. 開催日レース一覧 (GET /races/today) ---
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
        cleaned_races = [
            {
                "race_no": r["race_no"],
                "race_id": r["race_id"],
                "race_name": r["race_name"],
                "race_info": r["race_info"]
            }
            for r in day_races
        ]
        venues_data.append({
            "venue_name": v_name,
            "races": cleaned_races
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


# --- 3. 出馬表テーブル ＆ オッズAPIからデータ統合 ---
def fetch_shutuba_table(race_id: str):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_name_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_name_tag.get_text(strip=True) if race_name_tag else f"Race {race_id}"

    # netkeiba オッズ取得API
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

        name_tag = row.select_one(".HorseName a, .Horse_Info a
