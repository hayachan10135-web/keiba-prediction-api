from typing import Optional
import datetime
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Keiba Prediction API")

# モバイル用User-Agentで静的HTMLを確実に取得
HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
}

# --- 1. UptimeRobot用のヘルスチェック (GET /) ---
@app.get("/")
def health_check():
    """死活監視・常時起動用のエンドポイント"""
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- netkeiba SP版開催パース用ヘルパー関数 ---
def parse_netkeiba_sp_page(html_text: str):
    soup = BeautifulSoup(html_text, "html.parser")
    venues_data = []

    # 会場ごとの開催ブロックを取得
    kaisai_blocks = soup.select(".RaceList_DataBox, .KaisaiBlock, .RaceList")
    
    # リンクから race_id を含むものすべてを走査
    race_links = soup.find_all("a", href=re.compile(r"race_id=(\d{12})"))
    if not race_links:
        return []

    # race_id ごとにグループ化
    # race_id の 5〜6文字目が競馬場コード (01:札幌 ... 10:小倉)
    VENUE_CODE_MAP = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
        "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
    }

    venue_races_map = {}
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
        race_num_int = int(race_id[10:12])
        race_no = f"{race_num_int}R"

        # テキストからレース名等の取得を試みる
        raw_text = a.get_text(separator=" ", strip=True)
        # 不要な改行や重複スペースを整理
        clean_text = " ".join(raw_text.split())

        if venue_name not in venue_races_map:
            venue_races_map[venue_name] = []

        venue_races_map[venue_name].append({
            "race_no": race_no,
            "race_id": race_id,
            "race_name": clean_text if clean_text else f"{race_no}",
            "race_info": ""
        })

    for v_name, r_list in venue_races_map.items():
        # レース番号順にソート
        r_list.sort(key=lambda x: int(x["race_id"][10:12]))
        venues_data.append({
            "venue_name": v_name,
            "races": r_list
        })

    return venues_data


# --- 2. Step 3-2: 当日・直近週末開催レース一覧取得 (GET /races/today) ---
@app.get("/races/today")
def get_today_races(date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)")):
    """
    当日または直近週末の開催競馬場および全レース一覧を取得するエンドポイント
    """
    today = datetime.date.today()

    if date:
        clean_date = date.replace("-", "")
        target_dates = [clean_date]
    else:
        # 今日から直近7日間を対象に探索
        target_dates = [
            (today + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range(7)
        ]

    for d_str in target_dates:
        # SP版のレース一覧URL
        urls = [
            f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={d_str}",
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={d_str}"
        ]

        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                # SP版はutf-8 / euc-jpの自動判別
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
