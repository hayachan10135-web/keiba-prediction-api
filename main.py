from typing import Optional
import datetime
import re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Keiba Prediction API")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
}

@app.get("/")
def health_check():
    """死活監視・常時起動用のエンドポイント"""
    return {"status": "ok", "message": "Keiba Prediction API is running"}


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

    # 競馬場ごとに races を一時保持
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
        kai_day = race_id[6:10]  # 例: "0403" (4回3日目)
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

    # 各競馬場ごとに「直近の開催日（一番若い kai_day）」のみに絞り込み、1R〜12Rを整列
    for v_name, r_list in raw_venue_races.items():
        # 最も若い開催日コード（＝当日/土曜日）を特定
        earliest_kai_day = min(r["kai_day"] for r in r_list)
        day_races = [r for r in r_list if r["kai_day"] == earliest_kai_day]

        # 1R〜12Rにソートし、余分な内部キーを除去
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
