from typing import List, Optional
import datetime
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Keiba Prediction API")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
}

# --- Pydantic レスポンススキーマ ---
class HorsePrediction(BaseModel):
    umaban: int                # 馬番
    wakuban: int               # 枠番
    horse_name: str            # 馬名
    jockey: str                # 騎手名
    kinryo: float              # 斤量
    odds: Optional[float]      # 単勝オッズ
    popularity: Optional[int]  # 人気順
    score: float               # AI予測スコア (0.0〜1.0)
    mark: str                  # 予想印 (◎, ◯, ▲, △, ☆, -)

class Recommendations(BaseModel):
    tansho: List[int]          # 単勝推奨
    fukusho: List[int]         # 複勝推奨
    umaren: List[str]          # 馬連流し
    wide: List[str]            # ワイド流し
    sanrenpuku: List[str]      # 3連複フォーメーション

class PredictResponse(BaseModel):
    race_id: str
    race_name: str
    horses: List[HorsePrediction]
    recommendations: Recommendations


# --- 1. UptimeRobot用のヘルスチェック (GET /) ---
@app.get("/")
def health_check():
    """死活監視・常時起動用のエンドポイント"""
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- netkeiba SP版開催パース用ヘルパー関数 ---
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


# --- 2. Step 3-2: 当日・直近週末開催レース一覧取得 (GET /races/today) ---
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
                resp = requests.get(url, headers=HEADERS, timeout=10)
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


# --- 3. Step 3-3: 推論＆買い目レコメンド (GET /predict/{race_id}) ---
def fetch_shutuba_table(race_id: str):
    """出馬表ページから出走馬データをスクレイピング"""
    url = f"https://race.sp.netkeiba.com/race/shutuba.html?race_id={race_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    try:
        html = resp.content.decode("euc-jp")
    except UnicodeDecodeError:
        html = resp.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # レース名
    race_title_tag = soup.select_one(".RaceName, .RaceName_Text, h1")
    race_name = race_title_tag.get_text(strip=True) if race_title_tag else f"Race {race_id}"

    horses = []
    horse_rows = soup.select(".Shutuba_Table tr.HorseList, .RaceTable01 tr")

    for row in horse_rows:
        tds = row.select("td")
        if len(tds) < 5:
            continue

        # 枠番
        waku_text = row.select_one(".Waku span, td:nth-child(1)")
        waku = int(re.sub(r"\D", "", waku_text.get_text())) if waku_text and re.search(r"\d+", waku_text.get_text()) else 1

        # 馬番
        umaban_text = row.select_one(".Umaban, td:nth-child(2)")
        umaban_match = re.search(r"\d+", umaban_text.get_text()) if umaban_text else None
        if not umaban_match:
            continue
        umaban = int(umaban_match.group())

        # 馬名
        name_tag = row.select_one(".HorseName a, .Horse_Info a, .Horse02 a")
        horse_name = name_tag.get_text(strip=True) if name_tag else f"馬{umaban}"

        # 斤量・騎手
        jockey_tag = row.select_one(".Jockey a, .Jockey")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"

        kinryo_tag = row.select_one(".Weight, .Kinryo")
        kinryo_match = re.search(r"\d+(\.\d+)?", kinryo_tag.get_text()) if kinryo_tag else None
        kinryo = float(kinryo_match.group()) if kinryo_match else 55.0

        # 単勝オッズ・人気
        odds_tag = row.select_one(".Popular span, .Odds")
        odds_match = re.search(r"\d+(\.\d+)?", odds_tag.get_text()) if odds_tag else None
        odds = float(odds_match.group()) if odds_match else 10.0

        pop_tag = row.select_one(".Popular_Num")
        pop_match = re.search(r"\d+", pop_tag.get_text()) if pop_tag else None
        popularity = int(pop_match.group()) if pop_match else None

        horses.append({
            "umaban": umaban,
            "wakuban": waku,
            "horse_name": horse_name,
            "jockey": jockey,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity
        })

    return race_name, horses


def build_prediction_and_recs(race_name: str, horses: List[dict]) -> dict:
    """オッズ・人気をベースにスコアリングし、印と買い目を生成"""
    if not horses:
        return {"race_name": race_name, "horses": [], "recommendations": {"tansho": [], "fukusho": [], "umaren": [], "wide": [], "sanrenpuku": []}}

    # スコアリング（オッズが低い＝支持が高いほどベーススコアが高くなる計算）
    for h in horses:
        odds = h["odds"] if h["odds"] else 50.0
        # 0.10 〜 0.95 の範囲に正規化スコアを算出
        base_score = 1.0 / (1.0 + (odds ** 0.5))
        h["score"] = round(float(base_score), 4)

    # スコア降順にソートして印を割り当て
    sorted_horses = sorted(horses, key=lambda x: x["score"], reverse=True)
    marks = ["◎", "◯", "▲", "△", "☆"]
    for idx, h in enumerate(sorted_horses):
        h["mark"] = marks[idx] if idx < len(marks) else "-"

    # 各印の馬番を抽出
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
        # 馬連流し: ◎から相手へ
        recs["umaren"] = [f"{honmei}-{opp}" for opp in opponents]
        # ワイド流し: ◎から上位相手へ
        recs["wide"] = [f"{honmei}-{opp}" for opp in opponents[:3]]
        # 3連複フォーメーション: ◎ - ◯,▲ - ◯,▲,△,☆
        second_tier = [str(x) for x in ([taiko] + ([kuro] if kuro else []))]
        third_tier = [str(x) for x in opponents]
        recs["sanrenpuku"] = [f"{honmei} - {','.join(second_tier)} - {','.join(third_tier)}"]

    # 馬番順（出馬表の並び）に復元
    horses_sorted_by_num = sorted(sorted_horses, key=lambda x: x["umaban"])

    return {
        "race_name": race_name,
        "horses": horses_sorted_by_num,
        "recommendations": recs
    }


@app.get("/predict/{race_id}", response_model=PredictResponse)
def predict_race(race_id: str):
    """
    指定したレースIDの出走馬詳細・AI予測スコア・印・買い目レコメンドを取得するエンドポイント
    """
    if len(race_id) != 12 or not race_id.isdigit():
        raise HTTPException(status_code=400, detail="レースIDは12桁の数字で指定してください。")

    try:
        race_name, horses = fetch_shutuba_table(race_id)
        if not horses:
            raise HTTPException(status_code=404, detail="出走馬情報を取得できませんでした（開催前または存在しないIDの可能性があります）。")
        
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
