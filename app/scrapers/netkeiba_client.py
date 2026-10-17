# -*- coding: utf-8 -*-
"""
app/scrapers/netkeiba_client.py

Netkeiba HTTP クライアント基盤
- 要件 FR-SCR-05: 指数バックオフ付き最大3回リトライ (1s -> 2s -> 4s)
- タイムアウト制御、User-Agentヘッダー
"""

import time
import sys
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from app.core.config import settings


class NetkeibaClient:
    """Netkeiba耐障害HTTPセッションクライアント"""

    BASE_URL = "https://race.netkeiba.com"

    def __init__(self):
        self.session = requests.Session()
        retries = Retry(
            total=settings.SCRAPER_MAX_RETRIES,
            backoff_factor=settings.SCRAPER_BACKOFF_FACTOR,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def get_soup(self, url: str, timeout: int = 10, encoding: str = "euc-jp") -> Optional[BeautifulSoup]:
        """
        URLからHTMLを取得し、BeautifulSoupオブジェクトを生成する。
        3回リトライに失敗した場合は None を返し、例外でサーバーを落とさない (Fail-Safe)。
        """
        for attempt in range(1, settings.SCRAPER_MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    try:
                        content_str = resp.content.decode(encoding, errors="replace")
                    except Exception:
                        content_str = resp.text
                    return BeautifulSoup(content_str, "html.parser")
                else:
                    print(f"[NetkeibaClient] ⚠️ HTTP {resp.status_code} ({url}) attempt {attempt}/{settings.SCRAPER_MAX_RETRIES}", file=sys.stderr)
            except Exception as e:
                print(f"[NetkeibaClient] ⚠️ 通信例外 ({url}): {e} attempt {attempt}/{settings.SCRAPER_MAX_RETRIES}", file=sys.stderr)

            if attempt < settings.SCRAPER_MAX_RETRIES:
                time.sleep(settings.SCRAPER_BACKOFF_FACTOR * (2 ** (attempt - 1)))

        print(f"[NetkeibaClient] ❌ 最大リトライ超過により取得失敗: {url}", file=sys.stderr)
        return None

