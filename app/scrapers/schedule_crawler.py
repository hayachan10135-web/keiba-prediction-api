# -*- coding: utf-8 -*-
"""
app/scrapers/schedule_crawler.py

JRA 開催日程・レース一覧・発走時刻クローラー
- parse_netkeiba_race_list_html: HTMLから当日レース一覧を正確に抽出
- fetch_venues_for_date_raw: 指定日付の全会場・レース情報を取得（キャッシュ機能・TTL制御付き）
"""

import re
import time
import json
import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable
from bs4 import BeautifulSoup

from app.scrapers.netkeiba_client import safe_scrape_get, safe_decode_html, get_today_jst_str
from app.core.race_utils import clean_pure_race_name, VENUE_CODE_MAP
from app.core.database import get_supabase_client

# キャッシュ
_period_venues_cache: Dict[str, Any] = {}
_period_checked_empty_dates: set = set()

# コールバック
_post_time_handlers: List[Callable[[str, str], None]] = []
_venues_loaded_handlers: List[Callable[[str, List[dict]], None]] = []


def register_crawler_post_time_handler(cb: Callable[[str, str], None]):
    """発走時刻検知ハンドラー (race_id, post_time_str)"""
    if cb not in _post_time_handlers:
        _post_time_handlers.append(cb)


def register_venues_loaded_handler(cb: Callable[[str, List[dict]], None]):
    """会場・レース一覧確定時のスケジュール同期ハンドラー (formatted_date, venues)"""
    if cb not in _venues_loaded_handlers:
        _venues_loaded_handlers.append(cb)


def _notify_post_time(race_id: str, post_time_str: str):
    for cb in _post_time_handlers:
        try:
            cb(race_id, post_time_str)
        except Exception:
            pass


def _notify_venues_loaded(formatted_date: str, venues: List[dict]):
    for cb in _venues_loaded_handlers:
        try:
            cb(formatted_date, venues)
        except Exception:
            pass


def parse_netkeiba_race_list_html(html_text: str) -> List[dict]:
    """
    レース一覧パーサー（前日誤取得防止・当日レース確実抽出ロジック）
    """
    soup = BeautifulSoup(html_text, "html.parser")
    venues_data = []

    # kaisai_id を探して各会場の正確な kai_day を抽出 (例: payback_list.html?kaisai_id=2026050404)
    venue_target_kai_day = {}
    for a in soup.find_all("a", href=re.compile(r"kaisai_id=(\d{10})")):
        m = re.search(r"kaisai_id=(\d{10})", a["href"])
        if m:
            kid = m.group(1)
            v_code = kid[4:6]
            kai_day = kid[6:10]
            v_name = VENUE_CODE_MAP.get(v_code)
            if v_name:
                venue_target_kai_day[v_name] = kai_day

    raw_venue_races = {}
    seen_ids = set()

    for a in soup.find_all("a", href=re.compile(r"race_id=(\d{12})")):
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

        all_post_m = re.findall(r"(\d{1,2}:\d{2})", clean_text)
        post_time_est = f"{all_post_m[-1]}発走" if all_post_m else ""
        if post_time_est:
            _notify_post_time(race_id, post_time_est)

        # 純粋なレース名（会場名・R番号・時刻・頭数等を除去）
        pure_name = clean_pure_race_name(clean_text, race_id)
        if not pure_name:
            pure_name = f"{race_no} 一般競走"

        # 付帯コース情報（例: ダ1200m 15頭）
        c_match = re.search(r"((?:芝|ダ|ダート|障|障害)\d{3,4}m(?:\s*\d{1,2}頭)?)", clean_text)
        course_desc = c_match.group(1) if c_match else ""
        info_parts = [p for p in [post_time_est, course_desc] if p]
        full_info = " ".join(info_parts)

        if venue_name not in raw_venue_races:
            raw_venue_races[venue_name] = []

        raw_venue_races[venue_name].append({
            "race_no": race_no,
            "race_id": race_id,
            "kai_day": kai_day,
            "race_num_int": race_num_int,
            "race_name": pure_name,
            "race_info": full_info
        })

    for v_name, r_list in raw_venue_races.items():
        # kaisai_id から対象の kai_day が判明していれば最優先
        target_kd = venue_target_kai_day.get(v_name)
        if target_kd:
            day_races = [r for r in r_list if r["kai_day"] == target_kd]
        else:
            day_races = []

        if not day_races:
            counts = {}
            for r in r_list:
                kd = r["kai_day"]
                counts[kd] = counts.get(kd, 0) + 1

            if counts:
                # レース数が最も多いものを採用、同数の場合はkai_dayが若い方（当日優先）を採用
                best_kai_day = sorted(counts.keys(), key=lambda kd: (counts[kd], -int(kd) if kd.isdigit() else 0), reverse=True)[0]
                day_races = [r for r in r_list if r["kai_day"] == best_kai_day]

        day_races.sort(key=lambda x: x["race_num_int"])

        venues_data.append({
            "venue_name": v_name,
            "races": [
                {
                    "race_no": r["race_no"],
                    "race_id": r["race_id"],
                    "race_name": r["race_name"],
                    "race_info": r["race_info"]
                }
                for r in day_races
            ]
        })

    return venues_data


def fetch_venues_for_date_raw(date_str: str, force_refresh: bool = False) -> List[dict]:
    """指定日付 (YYYY-MM-DD または YYYYMMDD) のJRA開催会場・レース一覧を取得しキャッシュ (出走時刻変更対応)"""
    clean_d = date_str.replace("-", "")
    if len(clean_d) != 8 or not clean_d.isdigit():
        return []
    formatted_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}"

    now = time.time()
    today_s = get_today_jst_str()
    # 当日開催日（または指定日）の出走時刻変更を迅速に検知するためTTLを60秒とする
    cache_ttl = 60.0 if formatted_date == today_s else 3600.0

    if not force_refresh and formatted_date in _period_venues_cache:
        c_entry = _period_venues_cache[formatted_date]
        if isinstance(c_entry, dict) and "venues" in c_entry:
            if now - c_entry.get("updated_at", 0) < cache_ttl:
                return json.loads(json.dumps(c_entry["venues"]))
        elif isinstance(c_entry, list):
            return json.loads(json.dumps(c_entry))

    urls = [
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={clean_d}",
        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={clean_d}",
        f"https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={clean_d}"
    ]
    for url in urls:
        try:
            resp = safe_scrape_get(url, timeout=10.0)
            if resp.status_code != 200:
                continue
            text = safe_decode_html(resp)

            venues = parse_netkeiba_race_list_html(text)
            # 1会場あたり8R以上ある場合のみ正式な出走表確定済み開催日とみなす（次週の未確定枠を除外）
            valid_venues = [v for v in venues if len(v.get("races", [])) >= 8]
            if valid_venues:
                _period_venues_cache[formatted_date] = {
                    "venues": valid_venues,
                    "updated_at": now
                }
                _notify_venues_loaded(formatted_date, valid_venues)
                return json.loads(json.dumps(valid_venues))
        except Exception:
            continue

    # 外部スクレイピング取得が失敗した場合（netkeibaアクセス制限400・ネットワーク不調等）、DBからフォールバック復元
    db_venues = fetch_venues_from_database(formatted_date)
    if db_venues:
        _period_venues_cache[formatted_date] = {
            "venues": db_venues,
            "updated_at": now
        }
        _notify_venues_loaded(formatted_date, db_venues)
        return json.loads(json.dumps(db_venues))

    _period_checked_empty_dates.add(formatted_date)
    return []


def fetch_venues_from_database(date_str: str) -> List[dict]:
    """
    データベース (keiba_scheduled_races / keiba_predictions_cache) から
    指定日付の会場・レース一覧を完全復元（外部スクレイピング障害時の完全フェイルセーフ）
    """
    clean_d = date_str.replace("-", "")
    if len(clean_d) != 8 or not clean_d.isdigit():
        return []
    formatted_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}"

    try:
        supabase = get_supabase_client()
        if not supabase:
            return []

        sched_res = supabase.table("keiba_scheduled_races")\
            .select("race_id, race_date, race_name, post_datetime")\
            .eq("race_date", formatted_date)\
            .execute()

        sched_data = sched_res.data or []
        if not sched_data:
            return []

        # コース情報補完用に predictions_cache から race_details を取得
        race_ids = [r["race_id"] for r in sched_data if r.get("race_id")]
        course_info_map = {}
        try:
            p_res = supabase.table("keiba_predictions_cache")\
                .select("race_id, data_json")\
                .in_("race_id", race_ids)\
                .execute()
            for p in (p_res.data or []):
                dj = p.get("data_json") or {}
                rd = dj.get("race_details") or {}
                cd = rd.get("course_details", "").strip()
                if cd:
                    course_info_map[p["race_id"]] = cd
        except Exception:
            pass

        # 会場ごとにグルーピング
        raw_venue_races = {}
        for r in sched_data:
            race_id = r.get("race_id", "")
            if len(race_id) != 12 or not race_id.isdigit():
                continue

            v_code = race_id[4:6]
            venue_name = VENUE_CODE_MAP.get(v_code, "中央開催")
            race_num_int = int(race_id[10:12])
            race_no = f"{race_num_int}R"
            race_name = r.get("race_name") or f"{race_no} 一般競走"

            post_dt = r.get("post_datetime") or ""
            post_time_est = ""
            if len(post_dt) >= 16 and " " in post_dt:
                post_time_est = f"{post_dt[11:16]}発走"
            elif ":" in post_dt:
                post_time_est = f"{post_dt}発走"

            if post_time_est:
                _notify_post_time(race_id, post_time_est)

            course_desc = course_info_map.get(race_id, "")
            info_parts = [p for p in [post_time_est, course_desc] if p]
            full_info = " ".join(info_parts)

            if venue_name not in raw_venue_races:
                raw_venue_races[venue_name] = []

            raw_venue_races[venue_name].append({
                "race_no": race_no,
                "race_id": race_id,
                "race_num_int": race_num_int,
                "race_name": race_name,
                "race_info": full_info
            })

        venues_data = []
        # 会場コード順にソート (東京:05, 京都:08, 新潟:04など)
        sorted_venue_names = sorted(raw_venue_races.keys(), key=lambda vn: raw_venue_races[vn][0]["race_id"][4:6])

        for v_name in sorted_venue_names:
            r_list = raw_venue_races[v_name]
            r_list.sort(key=lambda x: x["race_num_int"])
            venues_data.append({
                "venue_name": v_name,
                "races": [
                    {
                        "race_no": r["race_no"],
                        "race_id": r["race_id"],
                        "race_name": r["race_name"],
                        "race_info": r["race_info"]
                    }
                    for r in r_list
                ]
            })

        return venues_data
    except Exception as e:
        print(f"[fetch_venues_from_database] DBからのレース一覧復元エラー ({formatted_date}): {e}")
        return []


def invalidate_venues_cache(date_str: Optional[str] = None):
    """キャッシュ破棄"""
    if date_str:
        clean_d = date_str.replace("-", "")
        formatted_date = f"{clean_d[:4]}-{clean_d[4:6]}-{clean_d[6:]}"
        _period_venues_cache.pop(formatted_date, None)
        _period_checked_empty_dates.discard(formatted_date)
    else:
        _period_venues_cache.clear()
        _period_checked_empty_dates.clear()

