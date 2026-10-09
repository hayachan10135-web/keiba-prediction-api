from typing import Optional
import datetime
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Keiba Prediction API")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# --- 1. UptimeRobot用のヘルスチェック (GET /) ---
@app.get("/")
def health_check():
    """死活監視・常時起動用のエンドポイント"""
    return {"status": "ok", "message": "Keiba Prediction API is running"}


# --- netkeiba開催HTMLパース用ヘルパー関数 ---
def parse_netkeiba_page(soup: BeautifulSoup):
    venues_data = []

    # パターン1: race_list.html 用のセレクタ (.RaceList_Box)
    kaisai_boxes = soup.select(".RaceList_Box")
    
    # パターン2: top ページ用のセレクタ (.RaceList_DataBox)
    if not kaisai_boxes:
        kaisai_boxes = soup.select(".RaceList_DataBox")

    for box in kaisai_boxes:
        # 開催会場タイトル（例: "4回 東京 1日目"）
        title_tag = box.select_one(".RaceList_DataTitle, .RaceList_DataTitle_Box")
        venue_title = title_tag.get_text(strip=True) if title_tag else "中央開催"

        races = []
        # レース項目抽出
        for race_item in box.select("li.RaceList_DataItem, .RaceList_DataList li"):
            num_tag = race_item.select_one(".Race_Num span, .RaceNum")
            race_num = num_tag.get_text(strip=True) if num_tag else ""

            name_tag = race_item.select_one(".ItemTitle, .RaceName")
            race_name = name_tag.get_text(strip=True) if name_tag else ""

            info_tag = race_item.select_one(".RaceData, .RaceData01")
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

    return venues_data


# --- 2. Step 3-2: 当日・直近週末開催レース一覧取得 (GET /races/today) ---
@app.get("/races/today")
def get_today_races(date: Optional[str] = Query(None, description="対象日付 (YYYYMMDD または YYYY-MM-DD)。未指定時は直近の開催日を自動検索")):
    """
    当日または直近週末の開催競馬場および全レース一覧を取得するエンドポイント
    """
    today = datetime.date.today()

    if date:
        clean_date = date.replace("-", "")
        target_dates = [clean_date]
    else:
        # 今日から直近7日間を対象に探索（土日開催を確実にキャッチ）
        target_dates = [
            (today + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range(7)
        ]

    for d_str in target_dates:
        # 枠順確定一覧ページ（race_list.html）を優先参照
        urls = [
            f"https://race.netkeiba.com/top/race_list.html?kaisai_date={d_str}",
            f"https://race.netkeiba.com/top/?kaisai_date={d_str}"
        ]

        for url in urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                resp.encoding = "euc-jp"
                soup = BeautifulSoup(resp.text, "html.parser")
                venues = parse_netkeiba_page(soup)

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
