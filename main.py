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

app = FastAPI(title="Keiba Prediction API")

HEADERS_PC = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
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


# --- 1. 死活監視・ヘルスチェック ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- 2. 当日・直近週末開催レース一覧 (GET /races/today) ---
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
        urls = [
            f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={d_str}",
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={d_str}"
        ]

        for url in urls:
            try:
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"}, timeout=10)
                try:
                    text = resp.content.decode("euc-jp")
                except UnicodeDecodeError:
                    text = resp.content.decode("utf-8", errors="replace")

                venues = parse_netkeiba_sp_page(text)
                if venues:
                    formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
                    return {
                        "date": formatted_date,
                        "venues": venues
                    }
            except Exception:
                continue

    return {
        "date": today.strftime("%Y-%m-%d"),
        "message": "直近7日間の開催情報（出馬表）が見つかりませんでした。",
        "venues": []
    }


# --- 3. 出走表取得・LightGBM推論・買い目レコメンド (GET /predict/{race_id}) ---
def fetch_shutuba_table(race_id: str):
    # 1. 出馬表基本情報の取得
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_title_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_title_tag.get_text(strip=True) if race_title_tag else f"Race {race_id}"

    # 2. オッズの取得（netkeibaの単勝オッズ専用ページから確実に抽出）
    odds_map = {}
    try:
        # 単勝・複勝オッズ専用ページを取得
        odds_url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
        o_resp = requests.get(odds_url, headers=HEADERS_PC, timeout=10)
        try:
            o_html = o_resp.content.decode("euc-jp")
        except UnicodeDecodeError:
            o_html = o_resp.content.decode("utf-8", errors="replace")
        o_soup = BeautifulSoup(o_html, "html.parser")

        # オッズテーブルの走査 (tr要素から馬番・単勝オッズ・人気を抽出)
        for row in o_soup.select("tr"):
            u_tag = row.select_one("td.Umaban, td[class*='Umaban']")
            o_tag = row.select_one("td.Odds, td[class*='Odds'], span.Odds")
            p_tag = row.select_one("td.Ninki, td[class*='Ninki'], span.Ninki")
            
            if u_tag and o_tag:
                u_text = u_tag.get_text(strip=True)
                if u_text.isdigit():
                    u_num = int(u_text)
                    o_match = re.search(r"(\d+\.\d+)", o_tag.get_text(strip=True))
                    o_val = float(o_match.group(1)) if o_match else None
                    
                    p_val = None
                    if p_tag:
                        p_match = re.search(r"\d+", p_tag.get_text(strip=True))
                        if p_match:
                            p_val = int(p_match.group())
                    
                    if o_val is not None:
                        odds_map[u_num] = {"odds": o_val, "popularity": p_val}

        # 万が一専用ページから取れなかった場合のAPIフォールバック
        if not odds_map:
            api_url = f"https://race.netkeiba.com/api/api_get_jra_odds.php?race_id={race_id}&type=1&action=init"
            api_resp = requests.get(api_url, headers=HEADERS_PC, timeout=5)
            api_data = api_resp.json()
            tansho = api_data.get("data", {}).get("odds", {}).get("1", {})
            for u_str, val in tansho.items():
                if isinstance(val, list) and len(val) >= 1:
                    om = re.search(r"(\d+\.\d+)", str(val[0]))
                    if om:
                        pm = int(val[1]) if len(val) > 1 and str(val[1]).isdigit() else None
                        odds_map[int(u_str)] = {"odds": float(om.group(1)), "popularity": pm}
    except Exception as e:
        print(f"オッズ取得警告 ({race_id}): {e}")

    horses = []
    rows = soup.select("tr.HorseList, table.Shutuba_Table tbody tr")

    for row in rows:
        umaban_tag = row.select_one(".Umaban, td[class*='Umaban']")
        if not umaban_tag:
            continue
        umaban_text = umaban_tag.get_text(strip=True)
        if not umaban_text.isdigit():
            continue
        umaban = int(umaban_text)

        waku_tag = row.select_one(".Waku, td[class*='Waku']")
        waku_match = re.search(r"\d+", waku_tag.get_text(strip=True)) if waku_tag else None
        wakuban = int(waku_match.group()) if waku_match else 1

        name_tag = row.select_one(".HorseName a, .Horse_Info a, .Horse02 a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        jockey_tag = row.select_one(".Jockey a, .Jockey")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        kinryo_tag = row.select_one(".Barei, .Weight, .Kinryo")
        kinryo = 55.0
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        odds_info = odds_map.get(umaban, {})
        odds = odds_info.get("odds")
        popularity = odds_info.get("popularity")

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    return race_name, horses


def build_prediction_and_recs(race_name: str, horses: List[dict]) -> dict:
    if not horses:
        return {"race_name": race_name, "horses": [], "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}}

    # 特徴量行列の作成 (wakuban, umaban, kinryo, odds, popularity)
    feature_rows = []
    for h in horses:
        w = h["wakuban"]
        u = h["umaban"]
        k = h["kinryo"]
        o = h["odds"] if h["odds"] is not None else 50.0
        p = h["popularity"] if h["popularity"] is not None else 10.0
        feature_rows.append([w, u, k, o, p])

    if model:
        # 学習済みLightGBMモデルによる予測
        scores = model.predict(feature_rows)
        for idx, h in enumerate(horses):
            h["score"] = round(float(scores[idx]), 4)
    else:
        # フォールバック
        for idx, h in enumerate(horses):
            odds = h["odds"] if h["odds"] else 50.0
            base_score = 1.0 / (1.0 + (odds ** 0.5))
            h["score"] = round(float(base_score), 4)

    # スコア順にソートして印を割り当て
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
        race_name, horses = fetch_shutuba_table(race_id)
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
