from typing import List, Optional
import os
import datetime
import re
import requests
import lightgbm as lgb
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# FastAPI アプリケーション定義
app = FastAPI(title="Keiba Prediction API")

HEADERS_SP = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
}

# --- LightGBM モデルのロード ---
MODEL_PATH = "lgb_model.txt"
model = None
if os.path.exists(MODEL_PATH):
    model = lgb.Booster(model_file=MODEL_PATH)
    print("LightGBMモデルを正常にロードしました。")
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
    tansho: List[int]
    fukusho: List[int]
    umaren: List[str]
    wide: List[str]
    sanrenpuku: List[str]

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
            resp = requests.get(url, headers=HEADERS_SP, timeout=10)
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


# --- 3. スマホ版出馬表から馬情報・オッズ・人気を一括スクレイピング ---
def fetch_shutuba_table_sp(race_id: str):
    url = f"https://race.sp.netkeiba.com/?pid=shutuba&race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_SP, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_name_tag = soup.select_one(".RaceName, .Race_Name, h1")
    race_name = race_name_tag.get_text(strip=True) if race_name_tag else f"Race {race_id}"

    horses = []
    # 各出走馬の行またはカード要素
    horse_blocks = soup.select(".HorseList, .Horse_List, tr.HorseList, tr[id^='tr_']")

    for idx, block in enumerate(horse_blocks, 1):
        # 馬番
        umaban = None
        umaban_tag = block.select_one(".Umaban, .Umaban_Box, span[class*='Umaban']")
        if umaban_tag:
            u_match = re.search(r"\d+", umaban_tag.get_text(strip=True))
            if u_match:
                umaban = int(u_match.group())
        if umaban is None:
            umaban = idx

        # 枠番
        wakuban = 1
        waku_tag = block.select_one(".Waku, span[class*='Waku'], td[class*='Waku']")
        if waku_tag:
            w_match = re.search(r"\d+", waku_tag.get_text(strip=True))
            if w_match:
                wakuban = int(w_match.group())

        # 馬名
        name_tag = block.select_one(".HorseName, .Horse_Name, a[href*='horse/']")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        # 騎手
        jockey_tag = block.select_one(".Jockey, a[href*='jockey/']")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        # 斤量
        kinryo = 55.0
        kinryo_tag = block.select_one(".Weight, .Kinryo, .Barei")
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        # 単勝オッズ・人気
        odds = None
        popularity = None
        block_text = block.get_text(separator=" ", strip=True)

        # オッズ（例: 3.2倍、14.5など）
        odds_match = re.search(r"(?:単勝)?\s*(\d+\.\d+)\s*(?:倍)?", block_text)
        if odds_match:
            try:
                odds = float(odds_match.group(1))
            except ValueError:
                pass

        # 人気（例: 1人気、1人など）
        pop_match = re.search(r"(\d+)\s*人(?:気)?", block_text)
        if pop_match:
            try:
                popularity = int(pop_match.group(1))
            except ValueError:
                pass

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    # 重複排除と馬番順ソート
    unique_horses = {h["umaban"]: h for h in horses}.values()
    return race_name, sorted(unique_horses, key=lambda x: x["umaban"])


def build_prediction_and_recs(race_name: str, horses: List[dict]) -> dict:
    if not horses:
        return {"race_name": race_name, "horses": [], "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}}

    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0
        feature_rows.append([w, u, k, o, p])

    if model:
        scores = model.predict(feature_rows)
        for idx, h in enumerate(horses):
            h["score"] = round(float(scores[idx]), 4)
    else:
        for idx, h in enumerate(horses):
            odds = h["odds"] if h["odds"] else 50.0
            base_score = 1.0 / (1.0 + (odds ** 0.5))
            h["score"] = round(float(base_score), 4)

    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)
    marks = ["◎", "◯", "▲", "△", "☆"]
    for idx, h in enumerate(sorted_horses):
        h["mark"] = marks[idx] if idx < len(marks) else "-"

    honmei = sorted_horses[0]["umaban"] if len(sorted_horses) > 0 else None
    taiko = sorted_horses[1]["umaban"] if len(sorted_horses) > 1 else None
    kuro = sorted_horses[2]["umaban"] if len(sorted_horses) > 2 else None
    osae = [h["umaban"] for h in sorted_horses[3:5]]

    recs = {
        "tansho": [honmei] if honmei else [],
        "fukusho": [honmei, taiko] if taiko else ([honmei] if honmei else []),
        "umaren": [],
        "wide": [],
        "sanrenpuku": []
    }

    if honmei and taiko:
        opponents = [taiko] + ([kuro] if kuro else []) + osae
        recs["umaren"] = [f"{honmei}-{opp}" for opp in opponents]
        recs["wide"] = [f"{honmei}-{opp}" for opp in opponents[:3]]
        second_tier = [str(x) for x in ([taiko] + ([kuro] if kuro else []))]
        third_tier = [str(x) for x in opponents]
        recs["sanrenpuku"] = [f"{honmei} - {','.join(second_tier)} - {','.join(third_tier)}"]

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "horses": horses_sorted_by_num,
        "recommendations": recs
    }


@app.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race(race_id: str):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    try:
        race_name, horses = fetch_shutuba_table_sp(race_id)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした。")

        result = build_prediction_and_recs(race_name, horses)
        return {
            "race_id": race_id,
            "race_name": result["race_name"],
            "horses": result["horses"],
            "recommendations": result["recommendations"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論処理中にエラーが発生しました: {e}")
