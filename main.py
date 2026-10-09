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


# --- 3. 出馬表テーブル ＆ オッズ取得 ---
def fetch_shutuba_table(race_id: str):
    print(f"\n==========================================")
    print(f"[DEBUG] レースデータ取得開始: race_id={race_id}")
    
    # 1. 出馬表から馬基本情報
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS_PC, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    race_name_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_name_tag.get_text(strip=True) if race_name_tag else f"Race {race_id}"
    print(f"[DEBUG] レース名: {race_name}")

    # 2. netkeiba オッズAPI呼び出し
    odds_map = {}
    odds_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&output=json"
    print(f"[DEBUG] オッズAPIリクエスト: {odds_url}")
    
    try:
        oresp = requests.get(odds_url, headers=HEADERS_PC, timeout=5)
        print(f"[DEBUG] オッズAPIステータスコード: {oresp.status_code}")
        print(f"[DEBUG] オッズAPIレスポンス先頭200文字: {oresp.text[:200]}")
        
        try:
            raw_res = oresp.json()
        except Exception:
            raw_res = json.loads(oresp.text)

        if isinstance(raw_res, str):
            raw_res = json.loads(raw_res)

        print(f"[DEBUG] パース後JSONの型: {type(raw_res)}")
        if isinstance(raw_res, dict):
            print(f"[DEBUG] JSONキー一覧: {list(raw_res.keys())}")
            # data -> odds -> 1
            data_block = raw_res.get("data", {})
            if isinstance(data_block, str):
                data_block = json.loads(data_block)
            
            odds_block = data_block.get("odds", {})
            tansho_dict = odds_block.get("1", {})
            print(f"[DEBUG] 単勝データ件数: {len(tansho_dict)}")

            for u_str, val in tansho_dict.items():
                if isinstance(val, list) and len(val) >= 1:
                    o_text = str(val[0]).strip()
                    if o_text and o_text != "---":
                        try:
                            o_val = float(o_text)
                            # val[2] が人気順位（val[1]はフラグ等のためval[2]を採用）
                            p_val = None
                            if len(val) >= 3 and str(val[2]).isdigit() and int(val[2]) > 0:
                                p_val = int(val[2])
                            elif len(val) >= 2 and str(val[1]).isdigit() and int(val[1]) > 0:
                                p_val = int(val[1])
                                
                            odds_map[int(u_str)] = {"odds": o_val, "popularity": p_val}
                        except ValueError:
                            pass
        print(f"[DEBUG] オッズAPIから取得成功した馬番数: {len(odds_map)} (サンプル: {dict(list(odds_map.items())[:3])})")
    except Exception as e:
        print(f"[DEBUG EXCEPTION] オッズAPI取得失敗: {type(e).__name__} - {e}")

    # 3. HTMLフォールバック (APIが0件だった場合)
    if not odds_map:
        print(f"[DEBUG] APIから取得できなかったため、HTMLフォールバックを実行します...")
        try:
            b1_url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b1"
            b1_resp = requests.get(b1_url, headers=HEADERS_PC, timeout=5)
            try:
                b1_html = b1_resp.content.decode("euc-jp")
            except UnicodeDecodeError:
                b1_html = b1_resp.content.decode("utf-8", errors="replace")
            b1_soup = BeautifulSoup(b1_html, "html.parser")

            for tr in b1_soup.select("tr"):
                u_td = tr.select_one("td.Umaban, td[class*='Umaban']")
                o_td = tr.select_one("td.Odds, td[class*='Odds']")
                p_td = tr.select_one("td.Ninki, td[class*='Ninki']")
                if u_td and o_td:
                    u_txt = u_td.get_text(strip=True)
                    o_txt = o_td.get_text(strip=True)
                    if u_txt.isdigit():
                        u_num = int(u_txt)
                        om = re.search(r"(\d+\.\d+)", o_txt)
                        if om:
                            o_val = float(om.group(1))
                            p_val = None
                            if p_td:
                                pm = re.search(r"(\d+)", p_td.get_text(strip=True))
                                if pm:
                                    p_val = int(pm.group(1))
                            odds_map[u_num] = {"odds": o_val, "popularity": p_val}
            print(f"[DEBUG] HTMLフォールバックで取得できた馬番数: {len(odds_map)}")
        except Exception as e:
            print(f"[DEBUG EXCEPTION] HTMLフォールバック失敗: {e}")

    horses = []
    rows = soup.select("tr.HorseList")
    print(f"[DEBUG] 出馬表テーブル行数: {len(rows)}")

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

    print(f"[DEBUG] 最終抽出完了頭数: {len(horses)}")
    if horses:
        print(f"[DEBUG] 先頭馬データ: {horses[0]}")
    print(f"==========================================\n")

    return race_name, sorted(horses, key=lambda x: x["umaban"])


def build_prediction_and_recs(race_name: str, horses: List[dict]) -> dict:
    if not horses:
        return {
            "race_name": race_name,
            "horses": [],
            "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}
        }

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
