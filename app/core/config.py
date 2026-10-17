# -*- coding: utf-8 -*-
"""
app/core/config.py

環境変数、システム定数、安全ガードレールしきい値設定
"""

import os
from typing import Optional


class Settings:
    # 実行環境
    ENV: str = os.environ.get("ENV", "production")
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"

    # Supabase 設定
    SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

    # 即PAT 設定
    IPAT_USER_NUM: str = os.environ.get("IPAT_USER_NUM", "")
    IPAT_PIN: str = os.environ.get("IPAT_PIN", "")
    IPAT_PARS: str = os.environ.get("IPAT_PARS", "")

    # UMACAスマート 設定
    UMACA_CARD_NUMBER: str = os.environ.get("UMACA_CARD_NUMBER", "")
    UMACA_PIN: str = os.environ.get("UMACA_PIN", "")
    UMACA_BIRTHDAY: str = os.environ.get("UMACA_BIRTHDAY", "")

    # Firebase Admin SDK 設定
    FIREBASE_KEY_PATH: str = os.environ.get("FIREBASE_KEY_PATH", "firebase_service_account.json")
    FIREBASE_SERVICE_ACCOUNT_KEY: str = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY", "")

    # 安全ガードレール定数（決定済み仕様）
    SLIPPAGE_ROI_BORDER: float = 105.0        # 期待回収率 105.0% 未満でスリッページ停止
    MAX_DELAY_MINUTES: int = 30               # 当初発走時刻から30分以上の遅延で自動投票中止
    DEFAULT_MAX_BET_PER_RACE: int = 2500      # 1レース最大上限額（デフォルト）

    # スクレイピング設定
    SCRAPER_MAX_RETRIES: int = 3              # Netkeiba最大リトライ回数
    SCRAPER_BACKOFF_FACTOR: float = 1.0       # 指数バックオフ係数 (1s, 2s, 4s)


settings = Settings()

