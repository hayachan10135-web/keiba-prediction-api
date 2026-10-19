# -*- coding: utf-8 -*-
"""
app/scrapers/netkeiba_client.py

Netkeiba HTTP クライアント ＆ スクレイピング基盤
- 要件 FR-SCR-05: 指数バックオフ付き最大3回リトライ (1s -> 2s -> 4s)
- User-Agentローテーション、接続プール (HTTPAdapter)、タイムアウト制御
- 出走表 (fetch_shutuba_table)
- リアルタイムオッズ (fetch_real_odds)
- 確定着順・払戻結果 (fetch_netkeiba_race_result)
- 結果データ正規化ヘルパー (normalize_race_res)
"""

import os
import re
import sys
import time
import json
import random
import datetime
import threading
from typing import Optional, List, Dict, Tuple, Any, Callable
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from fastapi import HTTPException

from app.core.config import settings
from app.core.race_utils import clean_pure_race_name


# --- 堅牢なスクレイピング用 HTTP クライアント基盤 ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 Edg/128.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0"
]

HEADERS_PC = {
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://race.netkeiba.com/"
}

# --- グローバル・レートリミッター（最小アクセス間隔の強制） ---
_rate_limiter_lock = threading.Lock()
_last_request_time: float = 0.0
MIN_REQUEST_INTERVAL_SECONDS: float = 1.3  # netkeiba WAF 安全閾値

# --- WAF サーキットブレーカー（HTTP 400 検知時の自動冷却機能） ---
_circuit_breaker_lock = threading.Lock()
_waf_blocked_until: float = 0.0
WAF_COOLDOWN_SECONDS: float = 1800.0  # 30分間冷却


def is_waf_in_cooldown() -> bool:
    """WAF冷却モード作動中かどうかを判定"""
    return time.time() < _waf_blocked_until


def get_waf_cooldown_remaining() -> int:
    """WAF冷却モードの残り秒数を取得"""
    rem = int(_waf_blocked_until - time.time())
    return max(0, rem)


def trigger_waf_cooldown(reason: str = "HTTP 400 WAF Block"):
    """WAF冷却モードを発動"""
    global _waf_blocked_until
    with _circuit_breaker_lock:
        _waf_blocked_until = time.time() + WAF_COOLDOWN_SECONDS
    print(f"[Netkeiba CircuitBreaker] ⚠️ {reason} を検知しました。{WAF_COOLDOWN_SECONDS/60:.0f}分間、外部スクレイピングを完全停止してDBフォールバックに移行します。")


def reset_waf_cooldown():
    """WAF冷却モードを手動解除"""
    global _waf_blocked_until
    with _circuit_breaker_lock:
        _waf_blocked_until = 0.0
    print("[Netkeiba CircuitBreaker] WAF冷却モードを手動リセットしました。")


def create_robust_session() -> requests.Session:
    """自動リトライ、コネクションプール、タイムアウト制御、外部プロキシ対応を備えた堅牢なセッション"""
    s = requests.Session()
    retries = Retry(
        total=getattr(settings, "SCRAPER_MAX_RETRIES", 3),
        backoff_factor=getattr(settings, "SCRAPER_BACKOFF_FACTOR", 0.5),
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=15, pool_maxsize=30)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # 環境変数 NETKEIBA_PROXY_URL によるプロキシ設定対応
    proxy_url = os.getenv("NETKEIBA_PROXY_URL")
    if proxy_url:
        s.proxies = {"http": proxy_url, "https": proxy_url}
        print(f"[Scraper] netkeiba通信に外部プロキシを適用しました: {proxy_url[:15]}...")

    return s


GLOBAL_HTTP_SESSION = create_robust_session()


def safe_scrape_get(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
    referer: str = "https://race.netkeiba.com/",
    max_retries: int = 3
) -> requests.Response:
    """
    User-Agentローテーション、レートリミット、WAFサーキットブレーカー、リトライ、ジッター、接続保持を統合した安全なHTTP GET
    """
    global _last_request_time

    # 1. WAFサーキットブレーカー判定（冷却中なら外部リクエストを行わず400モックを即座に返し、DBフォールバックを促す）
    if is_waf_in_cooldown():
        mock_resp = requests.Response()
        mock_resp.status_code = 400
        mock_resp._content = b""
        return mock_resp

    # 2. グローバル・レートリミッター（連続リクエスト時の安全ウェイト強制 + ジッター）
    with _rate_limiter_lock:
        now = time.time()
        elapsed = now - _last_request_time
        target_interval = MIN_REQUEST_INTERVAL_SECONDS + random.uniform(0.1, 0.4)
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
        _last_request_time = time.time()

    req_headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": referer,
        "Connection": "keep-alive"
    }
    if headers:
        req_headers.update(headers)

    last_exc = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                sleep_time = 0.5 * (2 ** attempt) + random.uniform(0.1, 0.3)
                time.sleep(sleep_time)
            resp = GLOBAL_HTTP_SESSION.get(url, headers=req_headers, timeout=timeout)

            # WAFの400アクセス制限を検知した場合、サーキットブレーカーを作動させてリトライ中断
            if resp.status_code == 400:
                trigger_waf_cooldown(f"HTTP 400 for {url}")
                return resp

            if resp.status_code in [429, 500, 502, 503, 504]:
                if attempt < max_retries - 1:
                    continue
            return resp
        except (requests.exceptions.RequestException, Exception) as e:
            last_exc = e
            if attempt < max_retries - 1:
                continue

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def safe_decode_html(resp: requests.Response) -> str:
    """EUC-JP / UTF-8 をスマートに自動判定して文字列デコード"""
    content = resp.content
    raw_head = content[:400].lower()
    if b'charset="utf-8"' in raw_head or b"charset=utf-8" in raw_head or b"utf-8" in raw_head:
        return content.decode("utf-8", errors="replace")
    try:
        return content.decode("euc-jp")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


# コールバック（疎結合化用）
_post_time_callbacks: List[Callable[[str, str], None]] = []
_schedule_callbacks: List[Callable[[str, str, str, str], None]] = []


def register_post_time_callback(cb: Callable[[str, str], None]):
    """発走時刻検知時のコールバック (race_id, post_time)"""
    if cb not in _post_time_callbacks:
        _post_time_callbacks.append(cb)


def register_schedule_callback(cb: Callable[[str, str, str, str], None]):
    """出馬表検知時のスケジュール登録コールバック (race_id, target_date, race_name, post_time)"""
    if cb not in _schedule_callbacks:
        _schedule_callbacks.append(cb)


def _notify_post_time(race_id: str, post_time: str):
    for cb in _post_time_callbacks:
        try:
            cb(race_id, post_time)
        except Exception:
            pass


def _notify_schedule(race_id: str, target_date: str, race_name: str, post_time: str):
    for cb in _schedule_callbacks:
        try:
            cb(race_id, target_date, race_name, post_time)
        except Exception:
            pass


def get_today_jst_str() -> str:
    tz_jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(tz_jst).strftime("%Y-%m-%d")


def fetch_real_odds(race_id: str) -> dict:
    """netkeibaのリアルタイムオッズAPIから馬連(type=4)および三連複(type=7)の実オッズを取得"""
    odds_data = {"umaren": {}, "trio": {}}
    if not race_id or len(race_id) != 12:
        return odds_data

    # 1. 馬連 (type=4)
    try:
        u_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=4&action=init&output=json"
        oresp = safe_scrape_get(u_url, timeout=6.0)
        if oresp.status_code == 200:
            d = oresp.json()
            if isinstance(d, str):
                d = json.loads(d)
            db = d.get("data", {})
            if isinstance(db, str):
                db = json.loads(db)
            u_dict = db.get("odds", {}).get("4", {})
            if isinstance(u_dict, dict):
                for comb_str, val in u_dict.items():
                    if len(comb_str) == 4 and comb_str.isdigit() and isinstance(val, list) and len(val) >= 1:
                        o_str = str(val[0]).replace(",", "").strip()
                        if o_str and o_str != "---":
                            try:
                                c1, c2 = int(comb_str[:2]), int(comb_str[2:])
                                odds_data["umaren"][(min(c1, c2), max(c1, c2))] = float(o_str)
                            except ValueError:
                                pass
    except Exception as e:
        print(f"馬連実オッズ取得エラー ({race_id}): {e}")

    # 2. 三連複 (type=7)
    try:
        t_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=7&action=init&output=json"
        oresp = safe_scrape_get(t_url, timeout=6.0)
        if oresp.status_code == 200:
            d = oresp.json()
            if isinstance(d, str):
                d = json.loads(d)
            db = d.get("data", {})
            if isinstance(db, str):
                db = json.loads(db)
            t_dict = db.get("odds", {}).get("7", {})
            if isinstance(t_dict, dict):
                for comb_str, val in t_dict.items():
                    if len(comb_str) == 6 and comb_str.isdigit() and isinstance(val, list) and len(val) >= 1:
                        o_str = str(val[0]).replace(",", "").strip()
                        if o_str and o_str != "---":
                            try:
                                c1, c2, c3 = int(comb_str[:2]), int(comb_str[2:4]), int(comb_str[4:])
                                s_comb = tuple(sorted([c1, c2, c3]))
                                odds_data["trio"][s_comb] = float(o_str)
                            except ValueError:
                                pass
    except Exception as e:
        print(f"三連複実オッズ取得エラー ({race_id}): {e}")

    return odds_data


def fetch_shutuba_table(race_id: str, race_date: Optional[str] = None):
    """
    出走表スクレイピング
    PC版出走表を第一候補、SP版をフォールバックとして二重冗長化
    """
    urls_to_try = [
        f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}",
        f"https://race.sp.netkeiba.com/race/shutuba.html?race_id={race_id}"
    ]
    html = ""
    last_err = ""
    for target_url in urls_to_try:
        try:
            resp = safe_scrape_get(target_url, timeout=10.0)
            if resp.status_code == 200 and len(resp.content) > 500:
                html = safe_decode_html(resp)
                if "HorseList" in html or "shutuba" in html or "出馬表" in html:
                    break
        except Exception as e:
            last_err = str(e)
            continue

    if not html:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba出馬表の取得に失敗しました (race_id={race_id}, error={last_err})。ダミー値生成は行いません。"
        )
    soup = BeautifulSoup(html, "html.parser")

    race_num_int = int(race_id[10:12]) if len(race_id) == 12 and race_id[10:12].isdigit() else 1
    race_num = f"{race_num_int}R"
    try:
        venue_code = int(race_id[4:6])
    except Exception:
        venue_code = 5

    race_name = ""
    name_candidates = soup.select(".RaceName, .RaceName_Text, .Race_Title, h1.RaceName")
    for cand in name_candidates:
        txt = cand.get_text(strip=True)
        if txt and not txt.isdigit():
            race_name = txt
            break

    if not race_name:
        title_tag = soup.find("title")
        if title_tag:
            m = re.search(r"\d+R\s+([^|・\-_]+)", title_tag.get_text(strip=True))
            if m:
                race_name = m.group(1).strip()

    if not race_name:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            m = re.search(r"\d+R\s+([^|・\-_]+)", og_title["content"])
            if m:
                race_name = m.group(1).strip()

    if not race_name:
        raise ValueError(f"レース名が取得できませんでした (race_id: {race_id})。デフォルト値混入防止のため処理を中断します。")

    race_name = clean_pure_race_name(race_name, race_id)

    race_data_tag = soup.select_one(".RaceData01, .RaceData")
    race_data_text = race_data_tag.get_text(separator=" ", strip=True) if race_data_tag else ""
    race_data2_tag = soup.select_one(".RaceData02")
    race_data2_text = race_data2_tag.get_text(separator=" ", strip=True) if race_data2_tag else ""

    all_post_m = re.findall(r"(\d{1,2}:\d{2})\s*発走", race_data_text)
    if all_post_m:
        post_time = f"{all_post_m[-1]}発走"
    else:
        post_time_m = re.search(r"(\d{1,2}:\d{2})", race_data_text)
        post_time = f"{post_time_m.group(1)}発走" if post_time_m else ""

    if post_time:
        _notify_post_time(race_id, post_time)

    weather_m = re.search(r"天候\s*:\s*([^\s/]+)", race_data_text)
    weather = weather_m.group(1) if weather_m else ""

    baba_m = re.search(r"(?:芝|ダート|ダ)?\s*:\s*(良|稍重|重|不良)", race_data_text)
    baba = baba_m.group(1) if baba_m else "良"

    dist_m = re.search(r"(\d{3,4})m", race_data_text)
    distance = float(dist_m.group(1)) if dist_m else 1600.0

    surface_name = "芝"
    surface_type = 0
    if "障" in race_data_text or "障害" in race_name:
        surface_name = "障害"
        surface_type = 2
    elif "ダ" in race_data_text:
        surface_name = "ダート"
        surface_type = 1

    # コース回り (0:右回り, 1:左回り, 2:直線)
    if "直線" in race_data_text or "直" in race_data_text or (venue_code == 4 and surface_type == 0 and int(distance) == 1000):
        turn_type = 2
        turn_label = "直線"
    elif "左" in race_data_text or venue_code in (4, 5, 7):
        turn_type = 1
        turn_label = "左"
    else:
        turn_type = 0
        turn_label = "右"

    # レースクラス・条件判定 (0:新馬, 1:未勝利, 2:1勝, 3:2勝, 4:3勝, 5:オープン/重賞)
    combined_race_text = f"{race_name} {race_data2_text}"
    race_class_code: Optional[int] = None
    class_label = ""
    if "新馬" in combined_race_text or "メイクデビュー" in combined_race_text:
        race_class_code = 0
        class_label = "新馬"
    elif "未勝利" in combined_race_text:
        race_class_code = 1
        class_label = "未勝利"
    elif "1勝" in combined_race_text or "１勝" in combined_race_text or "500万" in combined_race_text:
        race_class_code = 2
        class_label = "1勝クラス"
    elif "2勝" in combined_race_text or "２勝" in combined_race_text or "1000万" in combined_race_text:
        race_class_code = 3
        class_label = "2勝クラス"
    elif "3勝" in combined_race_text or "３勝" in combined_race_text or "1600万" in combined_race_text:
        race_class_code = 4
        class_label = "3勝クラス"
    elif any(k in combined_race_text for k in ["オープン", "ＯＰ", "OP", "G1", "G2", "G3", "GI", "GII", "GIII", "重賞", "(L)", "リステッド"]):
        race_class_code = 5
        class_label = "オープン"

    is_female_only_text = bool(re.search(r"牝(?:馬|\)|\]|限定|\s)", combined_race_text))

    condition_code = 0
    if "稍" in baba:
        condition_code = 1
    elif "不良" in baba:
        condition_code = 3
    elif "重" in baba:
        condition_code = 2

    weather_str = f" 天候:{weather}" if weather else ""
    class_str = f" [{class_label}]" if class_label else ""
    course_details = f"{surface_name}{int(distance)}m・{turn_label} ({baba}){class_str}{weather_str}"

    race_details = {
        "race_num": race_num,
        "post_time": post_time,
        "course_details": course_details
    }

    # スケジュール登録コールバック
    if post_time:
        target_date = race_date if race_date else get_today_jst_str()
        _notify_schedule(race_id, target_date, race_name, post_time)

    odds_map = {}
    try:
        odds_url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=init&output=json"
        oresp = safe_scrape_get(odds_url, timeout=6.0)
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
    except Exception:
        pass

    horses = []
    # PC版出馬表テーブルまたはID付き行に限定
    rows = soup.select("table.Shutuba_Table tr.HorseList, #shutuba_table tr.HorseList, tr.HorseList[id^='tr_']")
    if not rows:
        rows = soup.select("tr.HorseList")

    seen_umaban = set()

    for row in rows:
        umaban_tag = row.select_one("td.Umaban, td[class*='Umaban']")
        waku_tag = row.select_one("td.Waku, td[class*='Waku']")
        if umaban_tag and umaban_tag.get_text(strip=True).isdigit():
            umaban = int(umaban_tag.get_text(strip=True))
            wakuban = 1
            if waku_tag:
                wm = re.search(r"\d+", waku_tag.get_text(strip=True))
                if wm:
                    wakuban = int(wm.group())
        elif waku_tag and waku_tag.get_text(strip=True).isdigit():
            umaban = int(waku_tag.get_text(strip=True))
            wakuban = 1
            w_cls = " ".join(waku_tag.get("class") or [])
            wm_cls = re.search(r"Waku(\d)", w_cls)
            if wm_cls:
                wakuban = int(wm_cls.group(1))
        else:
            continue

        if umaban in seen_umaban:
            continue
        seen_umaban.add(umaban)

        name_tag = row.select_one(".HorseName a, .Horse_Info dt.Horse a, .Horse_Info a")
        if not name_tag or not name_tag.get_text(strip=True):
            raise HTTPException(status_code=502, detail=f"出馬表から馬番 {umaban} の正式な馬名を取得できませんでした (ダミー値補完禁止)")
        horse_name = name_tag.get_text(strip=True)

        jockey_tag = row.select_one(".Jockey em, .Jockey a")
        jockey = jockey_tag.get_text(strip=True) if jockey_tag else "未定"
        jockey = re.sub(r"\s*\d{2}(?:\.\d)?$", "", jockey).strip() or "未定"

        trainer_tag = row.select_one("td.Trainer, .Trainer a")
        trainer = trainer_tag.get_text(strip=True) if trainer_tag else "未定"

        sex_age = None
        sex_code = 0
        age = 4.0
        barei_tag = row.select_one("td.Barei, dd.Age")
        if barei_tag:
            b_raw = barei_tag.get_text(" ", strip=True)
            m_sa = re.search(r"([牡牝セ騸]\d+)", b_raw)
            b_str = m_sa.group(1) if m_sa else b_raw.split()[0] if b_raw else ""
            if b_str:
                sex_age = b_str
                if "牝" in b_str:
                    sex_code = 1
                elif "セ" in b_str or "騸" in b_str:
                    sex_code = 2
                m_age = re.search(r"(\d+)", b_str)
                if m_age:
                    age = float(m_age.group(1))

        kinryo = 55.0
        kinryo_tag = (barei_tag.find_next_sibling("td") if (barei_tag and barei_tag.name == "td") else None) or row.select_one("td.Kinryo, dd.Jockey")
        if kinryo_tag:
            km = re.search(r"(\d{2}(?:\.\d)?)", kinryo_tag.get_text(strip=True))
            if km:
                kinryo = float(km.group(1))

        horse_weight: Optional[int] = None
        weight_diff: Optional[int] = None
        weight_tag = row.select_one("td.Weight, .Weight, td[class*='Weight'], span.Weight")
        w_search_text = weight_tag.get_text(strip=True) if weight_tag else row.get_text(separator=" ", strip=True)
        wm = re.search(r"\b([3-6]\d{2})(?:kg)?\s*\(([+-]?\d+|0|前計不|--)\)", w_search_text)
        if not wm and weight_tag:
            wm = re.search(r"\b([3-6]\d{2})\b", w_search_text)
        if wm:
            horse_weight = int(wm.group(1))
            diff_raw = wm.group(2) if wm.lastindex and wm.lastindex >= 2 else None
            if diff_raw and re.match(r"^[+-]?\d+$", diff_raw):
                weight_diff = int(diff_raw)
            else:
                weight_diff = 0

        odds_item = odds_map.get(umaban, {})
        odds = odds_item.get("odds")
        popularity = odds_item.get("popularity")

        is_blinker = False
        is_first_blinker = False
        blinker_tag = row.select_one(".Blinker, .Blink, [class*='Blink'], [class*='blinker']")
        row_raw_text = row.get_text(separator=" ", strip=True)
        if blinker_tag:
            b_txt = blinker_tag.get_text(strip=True)
            is_blinker = True
            if "初" in b_txt:
                is_first_blinker = True
        elif re.search(r"初[BＢ]|初ブリンカー", row_raw_text):
            is_blinker = True
            is_first_blinker = True
        elif re.search(r"[BＢ]|\bB\b", row_raw_text) and any(t.get_text(strip=True) in ["B", "Ｂ"] for t in row.select("span, em, font")):
            is_blinker = True

        horses.append({
            "umaban": umaban,
            "wakuban": wakuban,
            "horse_name": horse_name,
            "jockey": jockey,
            "trainer": trainer,
            "kinryo": kinryo,
            "odds": odds,
            "popularity": popularity,
            "is_blinker": is_blinker,
            "is_first_blinker": is_first_blinker,
            "horse_weight": horse_weight,
            "weight_diff": weight_diff,
            "sex_age": sex_age,
            "sex_code": sex_code,
            "age": age
        })

    course_meta = {
        "distance": distance,
        "surface_type": surface_type,
        "condition_code": condition_code,
        "turn_type": turn_type,
        "race_num_int": race_num_int,
        "race_class_code": race_class_code,
        "is_female_only_text": is_female_only_text
    }

    if not horses or len(horses) == 0:
        raise HTTPException(status_code=502, detail=f"レース {race_id} の出走馬データが0頭です。不完全なデータは処理できません。")

    return race_name, race_details, sorted(horses, key=lambda x: x["umaban"]), course_meta


def normalize_race_res(race_res: Optional[dict]) -> Optional[dict]:
    """orders / payouts のJSONキー（文字列化された馬番）を整数に復元"""
    if not race_res or not isinstance(race_res, dict):
        return race_res

    raw_orders = race_res.get("orders") or {}
    norm_orders = {}
    for k, v in raw_orders.items():
        try:
            norm_orders[int(k)] = int(v)
        except (ValueError, TypeError):
            try:
                norm_orders[k] = int(v)
            except Exception:
                pass
    race_res["orders"] = norm_orders

    payouts = race_res.get("payouts") or {}
    if isinstance(payouts, dict):
        for p_type in ["tansho", "fukusho"]:
            if p_type in payouts and isinstance(payouts[p_type], dict):
                norm_p = {}
                for k, v in payouts[p_type].items():
                    try:
                        norm_p[int(k)] = int(v)
                    except (ValueError, TypeError):
                        norm_p[k] = v
                payouts[p_type] = norm_p
        race_res["payouts"] = payouts
    return race_res


def fetch_netkeiba_race_result(race_id: str) -> Optional[dict]:
    """
    netkeiba から確定着順・払戻結果をスクレイピング
    SP版とPC版の二重冗長スクレイピング
    """
    orders = {}
    payouts = {
        "tansho": {},
        "fukusho": {},
        "wakuren": {},
        "umaren": {},
        "wide": {},
        "umatan": {},
        "trio": {},
        "tierce": {}
    }
    race_name = ""
    sp_html_text = ""
    pc_html_text = ""
    sp_title = ""
    pc_title = ""

    # SP版スクレイピング試行
    sp_url = f"https://race.sp.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        resp = safe_scrape_get(sp_url, timeout=10.0)
        html = safe_decode_html(resp)
        sp_html_text = html
        soup = BeautifulSoup(html, "html.parser")
        sp_title = soup.title.get_text(strip=True) if soup.title else ""

        if not race_name or race_name.startswith("Race "):
            r_title = soup.select_one(".RaceName, h1, .Race_Title")
            if r_title and r_title.get_text(strip=True):
                race_name = r_title.get_text(strip=True)
            elif sp_title:
                m = re.match(r"^(.+?)\s*(?:結果・払戻|\s*\|)", sp_title)
                if m:
                    race_name = m.group(1).strip()

        for tr in soup.select("table[class*='Payout'] tr, table[class*='Pay'] tr, .Payout_Detail tr"):
            th = tr.select_one("th")
            tds = tr.find_all("td")
            if th and len(tds) >= 2:
                kind = th.get_text(strip=True)
                nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", tds[0].get_text())]
                pays = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", tds[1].get_text()) if x.replace(",", "").isdigit()]
                if "単勝" in kind:
                    for n, p in zip(nums, pays):
                        payouts["tansho"][n] = p
                elif "複勝" in kind:
                    for n, p in zip(nums, pays):
                        payouts["fukusho"][n] = p
                elif "枠連" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{min(nums[0], nums[1])}-{max(nums[0], nums[1])}"
                        payouts["wakuren"][comb_key] = pays[0]
                elif "馬連" in kind and "馬単" not in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{min(nums[0], nums[1])}-{max(nums[0], nums[1])}"
                        payouts["umaren"][comb_key] = pays[0]
                elif "ワイド" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        pair_count = min(len(nums) // 2, len(pays))
                        for pi in range(pair_count):
                            w_nums = nums[pi * 2 : pi * 2 + 2]
                            w_key = f"{min(w_nums[0], w_nums[1])}-{max(w_nums[0], w_nums[1])}"
                            payouts["wide"][w_key] = pays[pi]
                elif "馬単" in kind:
                    if len(nums) >= 2 and len(pays) >= 1:
                        comb_key = f"{nums[0]}-{nums[1]}"
                        payouts["umatan"][comb_key] = pays[0]
                elif "3連複" in kind or "三連複" in kind:
                    if len(nums) >= 3 and len(pays) >= 1:
                        s_nums = sorted(nums[:3])
                        comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                        payouts["trio"][comb_key] = pays[0]
                elif "3連単" in kind or "三連単" in kind:
                    if len(nums) >= 3 and len(pays) >= 1:
                        comb_key = f"{nums[0]}-{nums[1]}-{nums[2]}"
                        payouts["tierce"][comb_key] = pays[0]

        for row in soup.select("tr.HorseList, .RaceResultList tr, table tr, tr"):
            o_elem = row.select_one(".Rank, td.Rank, td.Result_Num, .Result_Num")
            u_elem = row.select_one(".Umaban, td.Umaban, .umaban")
            if o_elem and u_elem:
                o_txt = o_elem.get_text(strip=True)
                u_txt = u_elem.get_text(strip=True)
                if o_txt.isdigit() and u_txt.isdigit():
                    orders[int(u_txt)] = int(o_txt)
    except Exception:
        pass

    # PC版スクレイピング (詳細・全券種確実取得)
    if not payouts["umaren"] or len(orders) < 3:
        pc_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        try:
            resp = safe_scrape_get(pc_url, timeout=10.0)
            html = safe_decode_html(resp)
            pc_html_text = html
            soup = BeautifulSoup(html, "html.parser")
            pc_title = soup.title.get_text(strip=True) if soup.title else ""

            if not race_name or race_name.startswith("Race "):
                r_title = soup.select_one(".RaceName, .Race_Title, h1")
                if r_title and r_title.get_text(strip=True):
                    race_name = r_title.get_text(strip=True)
                elif pc_title:
                    m = re.match(r"^(.+?)\s*(?:結果・払戻|\s*\|)", pc_title)
                    if m:
                        race_name = m.group(1).strip()

            for tr in soup.select("table.Payout_Detail_Table tr, table[class*='Payout'] tr"):
                th = tr.select_one("th")
                td_res = tr.select_one("td.Result")
                td_pay = tr.select_one("td.Payout")
                if th and td_res and td_pay:
                    kind = th.get_text(strip=True)
                    pays = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", td_pay.get_text()) if x.replace(",", "").isdigit()]

                    if "単勝" in kind:
                        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        for n, p in zip(nums, pays):
                            payouts["tansho"][n] = p
                    elif "複勝" in kind:
                        nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        for n, p in zip(nums, pays):
                            payouts["fukusho"][n] = p
                    elif "枠連" in kind:
                        c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                        if len(c_nums) >= 2 and len(pays) >= 1:
                            comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                            payouts["wakuren"][comb_key] = pays[0]
                    elif "馬連" in kind and "馬単" not in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                    payouts["umaren"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                payouts["umaren"][comb_key] = pays[0]
                    elif "ワイド" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                    payouts["wide"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{min(c_nums[0], c_nums[1])}-{max(c_nums[0], c_nums[1])}"
                                payouts["wide"][comb_key] = pays[0]
                    elif "馬単" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 2 and idx < len(pays):
                                    comb_key = f"{c_nums[0]}-{c_nums[1]}"
                                    payouts["umatan"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 2 and len(pays) >= 1:
                                comb_key = f"{c_nums[0]}-{c_nums[1]}"
                                payouts["umatan"][comb_key] = pays[0]
                    elif "3連複" in kind or "三連複" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 3 and idx < len(pays):
                                    s_nums = sorted(c_nums[:3])
                                    comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                                    payouts["trio"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 3 and len(pays) >= 1:
                                s_nums = sorted(c_nums[:3])
                                comb_key = f"{s_nums[0]}-{s_nums[1]}-{s_nums[2]}"
                                payouts["trio"][comb_key] = pays[0]
                    elif "3連単" in kind or "三連単" in kind:
                        uls = td_res.find_all("ul")
                        if uls:
                            for idx, ul in enumerate(uls):
                                c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", ul.get_text())]
                                if len(c_nums) >= 3 and idx < len(pays):
                                    comb_key = f"{c_nums[0]}-{c_nums[1]}-{c_nums[2]}"
                                    payouts["tierce"][comb_key] = pays[idx]
                        else:
                            c_nums = [int(x) for x in re.findall(r"\b\d{1,2}\b", td_res.get_text())]
                            if len(c_nums) >= 3 and len(pays) >= 1:
                                comb_key = f"{c_nums[0]}-{c_nums[1]}-{c_nums[2]}"
                                payouts["tierce"][comb_key] = pays[0]

            for tr in soup.select("table.RaceTable01 tr, table.ResultTable tr, tr.HorseList"):
                o_elem = tr.select_one("td.Rank, td.Result_Num, div.Rank")
                u_elem = tr.select_one("td.Umaban, td.Txt_C, div.Umaban, td[class*='Txt_C']")
                if o_elem and u_elem:
                    o_txt = o_elem.get_text(strip=True)
                    u_txt = u_elem.get_text(strip=True)
                    if o_txt.isdigit() and u_txt.isdigit():
                        orders[int(u_txt)] = int(o_txt)
        except Exception:
            pass

    is_confirmed = (len(payouts["tansho"]) > 0) or (len(payouts["umaren"]) > 0) or (1 in orders.values())

    if len(payouts["tansho"]) > 0 and 1 not in orders.values():
        for winner_u in payouts["tansho"].keys():
            orders[winner_u] = 1

    # レース中止・取りやめ判定 (荒天・積雪・不成立等)
    all_html = sp_html_text + " " + pc_html_text
    is_cancelled = False
    if not is_confirmed:
        cancel_kws = ["開催中止", "競走取りやめ", "レース取りやめ", "開催取りやめ", "不成立", "天候調査"]
        has_cancel_kw = any(kw in all_html for kw in cancel_kws)
        empty_title = (sp_title.startswith("|") or pc_title.startswith("|") or (not sp_title and not pc_title))
        if has_cancel_kw or (empty_title and len(orders) == 0):
            is_cancelled = True

    if race_name and not race_name.startswith("Race "):
        race_name = clean_pure_race_name(race_name, race_id)
    else:
        # デフォルト値 "Race 2026..." ではなく、keiba_scheduled_races から正規のレース名を取得
        try:
            from app.core.database import get_supabase_client
            sb = get_supabase_client()
            if sb:
                sr = sb.table("keiba_scheduled_races").select("race_name").eq("race_id", race_id).limit(1).execute()
                if sr.data and sr.data[0].get("race_name"):
                    race_name = clean_pure_race_name(sr.data[0]["race_name"], race_id)
        except Exception:
            pass

    if not race_name or race_name.startswith("Race "):
        raise ValueError(f"確定結果のレース名が取得できませんでした (race_id: {race_id})。デフォルト値の混入を避けるため処理を中断します。")

    if is_cancelled:
        return {
            "race_name": race_name,
            "orders": {},
            "payouts": payouts,
            "is_cancelled": True
        }

    if not is_confirmed:
        return None

    return normalize_race_res({
        "race_name": race_name,
        "orders": orders,
        "payouts": payouts,
        "is_cancelled": False
    })
