import datetime
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Keiba Prediction API")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# --- 1. UptimeRobot用のヘルスチェック (GET /) ---
@app.get("/")
def health_check():
    """死活監視・常時起動用のエンドポイント"""
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- 2. Step 3-2: 当日開催・レース一覧取得 (GET /races/today) ---
@app.get("/races/today")
def get_today_races():
    """
    当日（または直近開催日）の開催競馬場および全レース一覧を取得するエンドポイント
    """
    url = "https://race.netkeiba.com/top/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        # netkeibaのHTMLエンコーディングに対応
        resp.encoding = "euc-jp"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"netkeibaへのアクセスに失敗しました: {e}")

    venues_data = []

    # netkeibaトップの各開催場ブロック（東京、京都など）を取得
    kaisai_blocks = soup.select(".RaceList_DataBox")

    # 平日などで開催ブロックがない場合
    if not kaisai_blocks:
        return {
            "date": datetime.date.today().strftime("%Y-%m-%d"),
            "message": "現在、中央競馬の開催情報はありません。",
            "venues": []
        }

    for block in kaisai_blocks:
        # 開催タイトル（例: "4回 東京 1日目"）
        title_tag = block.select_one(".RaceList_DataTitle")
        venue_title = title_tag.get_text(strip=True) if title_tag else "中央開催"

        races = []
        for race_item in block.select("li.RaceList_DataItem"):
            # レース番号（例: "1R"）
            num_tag = race_item.select_one(".Race_Num span")
            race_num = num_tag.get_text(strip=True) if num_tag else ""

            # レース名（例: "2歳未勝利"）
            name_tag = race_item.select_one(".ItemTitle")
            race_name = name_tag.get_text(strip=True) if name_tag else ""

            # 発走時刻・距離など（例: "10:05発走 / 芝1600m"）
            info_tag = race_item.select_one(".RaceData")
            race_info = info_tag.get_text(separator=" ", strip=True) if info_tag else ""

            # リンクから12桁の race_id を抽出
            link_tag = race_item.select_one("a")
            race_id = ""
            if link_tag and "href" in link_tag.attrs:
                m = re.search(r"race_id=(\d{12})", link_tag["href"])
                if m:
                    race_id = m.group(1)

            if race_id:
                races.append({
                    "race_no": race_num,
                    "race_id": race_id,
                    "race_name": race_name,
                    "race_info": race_info
                })

        if races:
            venues_data.append({
                "venue_name": venue_title,
                "races": races
            })

    return {
        "date": datetime.date.today().strftime("%Y-%m-%d"),
        "venues": venues_data
    }

# 既存の /predict などのエンドポイントはそのまま維持してください
