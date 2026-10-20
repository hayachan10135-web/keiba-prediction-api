# -*- coding: utf-8 -*-
"""
app/core/logging_config.py

常駐バックグラウンドサーバー向けファイルロギング＆ローテーション設定
- コンソール出力 (標準出力/標準エラー出力) と完全連動
- tmp/logs/keiba_server.log へ日次ローテーションで自動保存 (直近14日分保持)
"""

import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "tmp", "logs"))
LOG_FILE = os.path.join(LOG_DIR, "keiba_server.log")


def setup_server_logging(logger_name: str = "keiba_ai") -> logging.Logger:
    """サーバー全体のファイルロギングハンドラーを構成"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 既存ハンドラーとの重複追加を防止
    if any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        return logger

    # フォーマット
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 日次ローテーションファイルハンドラー (深夜0時切替, 14世代保持)
    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. コンソール (標準出力) ハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
