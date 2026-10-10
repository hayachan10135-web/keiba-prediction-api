from typing import List, Optional
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

        name_tag = row.select_one(".HorseName a, .Horse_Info a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        jockey_tag = row.select_one(".Jockey a")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

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
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    return race_name, sorted(horses, key=lambda x: x["umaban"])


# --- 4. 推論 ＆ マーク・推奨買い目の生成 ---
def build_prediction_and_recs(race_name: str, race_id: str, horses: List[dict]) -> dict:
    if not horses:
        return {
            "race_name": race_name,
            "horses": [],
            "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}
        }

    # モデルの期待する特徴量数を判定 (5次元 / 12次元 / 16次元)
    expected_num_features = 16
    if model:
        try:
            expected_num_features = model.num_feature()
        except Exception:
            expected_num_features = 16

    # 競馬場コード (race_id の 4〜6 桁目)
    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0

        if expected_num_features == 16:
            surface_type = 0
            distance = 1600.0
            distance_diff = 0.0
            career_races = h.get("career_races", 5.0)
            career_top3_rate = h.get("career_top3_rate", 0.25)
            prev1_order = h.get("prev1_order", 5.0)
            prev1_pop = h.get("prev1_pop", 5.0)
            prev1_odds = h.get("prev1_odds", 15.0)
            prev2_order = h.get("prev2_order", 5.0)
            days_since_prev = h.get("days_since_prev", 35.0)

            feature_rows.append([
                w, u, k, o, p,
                venue_code, surface_type, distance, distance_diff,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev
            ])
        elif expected_num_features == 12:
            career_races = h.get("career_races", 5.0)
            career_top3_rate = h.get("career_top3_rate", 0.25)
            prev1_order = h.get("prev1_order", 5.0)
            prev1_pop = h.get("prev1_pop", 5.0)
            prev1_odds = h.get("prev1_odds", 15.0)
            prev2_order = h.get("prev2_order", 5.0)
            days_since_prev = h.get("days_since_prev", 35.0)
            feature_rows.append([
                w, u, k, o, p,
                career_races, career_top3_rate,
                prev1_order, prev1_pop, prev1_odds, prev2_order, days_since_prev
            ])
        else:
            feature_rows.append([w, u, k, o, p])

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

    # スコア降順にソート
    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)

    # 全頭初期化
    for h in sorted_horses:
        h["mark"] = "-"

    # 基本の5頭 (1〜5位)
    base_marks = ["◎", "◯", "▲", "△", "×"]
    for idx in range(min(5, len(sorted_horses))):
        sorted_horses[idx]["mark"] = base_marks[idx]

    # 6位以降の「☆」割り当て判定（最大3頭まで拡張）
    if len(sorted_horses) >= 6:
        sorted_horses[5]["mark"] = "☆"
        base_hoshi_score = sorted_horses[5]["score"]
        DIFF_THRESHOLD = 0.015

        # 7位馬の判定
        if len(sorted_horses) >= 7:
            if (base_hoshi_score - sorted_horses[6]["score"]) <= DIFF_THRESHOLD:
                sorted_horses[6]["mark"] = "☆"

                # 8位馬の判定
                if len(sorted_horses) >= 8:
                    if (base_hoshi_score - sorted_horses[7]["score"]) <= DIFF_THRESHOLD:
                        sorted_horses[7]["mark"] = "☆"

    # 買い目推奨の生成
    marked_horses = [h for h in sorted_horses if h["mark"] != "-"]
    honmei = sorted_horses[0]["umaban"] if len(sorted_horses) > 0 else None
    taiko = sorted_horses[1]["umaban"] if len(sorted_horses) > 1 else None
    kuro = sorted_horses[2]["umaban"] if len(sorted_horses) > 2 else None

    opponents = [h["umaban"] for h in marked_horses[1:]]

    recs = {
        "tansho": [honmei] if honmei else [],
        "fukusho": [honmei, taiko] if taiko else ([honmei] if honmei else []),
        "umaren": [],
        "wide": [],
        "sanrenpuku": []
    }

    if honmei and opponents:
        recs["umaren"] = [f"{honmei}-{opp}" for opp in opponents]
        recs["wide"] = [f"{honmei}-{opp}" for opp in opponents[:4]]
        
        second_tier_list = [str(x) for x in ([taiko] + ([kuro] if kuro else []))]
        third_tier_list = [str(x) for x in opponents]
        second_tier_str = ",".join(second_tier_list)
        third_tier_str = ",".join(third_tier_list)
        recs["sanrenpuku"] = [f"{honmei} - {second_tier_str} - {third_tier_str}"]

    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "horses": horses_sorted_by_num,
        "recommendations": recs
    }


# --- 5. 推論エンドポイント ---
@app.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race(race_id: str):
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    try:
        race_name, horses = fetch_shutuba_table(race_id)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした。")

        result = build_prediction_and_recs(race_name, race_id, horses)
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
