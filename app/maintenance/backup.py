# -*- coding: utf-8 -*-
"""
app/maintenance/backup.py

ローカル SQLite データベースの自動バックアップ・世代管理モジュール
- SQLite の 'VACUUM INTO' 構文を活用し、オンライン無停止・整合性100%のスナップショットを生成
- Google ドライブ (G:\マイドライブ\keiba_data\backup) へ自動退避
- 直近 14 世代のローテーション保持 (超過分は自動削除)
"""

import os
import sys
import glob
import sqlite3
import datetime
from contextlib import closing

# ルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.sqlite_bridge import DEFAULT_DB_PATH

# バックアップ先ディレクトリ (Google ドライブ)
BACKUP_DRIVE_DIR = r"G:\マイドライブ\keiba_data\backup"
MAX_BACKUP_GENERATIONS = 14


def backup_database_to_drive() -> bool:
    """
    ローカル SQLite データベースを Google ドライブへ安全にスナップショット保存。
    戻り値: 成功時 True, 失敗時 False
    """
    if not os.path.exists(DEFAULT_DB_PATH):
        print(f"[Backup] [WARN] バックアップ元DBが存在しません: {DEFAULT_DB_PATH}")
        return False

    try:
        os.makedirs(BACKUP_DRIVE_DIR, exist_ok=True)
    except Exception as e:
        print(f"[Backup] [ERROR] Google ドライブバックアップ先作成エラー: {e}")
        return False

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_filename = f"keiba_local_{now_str}.db"
    target_path = os.path.join(BACKUP_DRIVE_DIR, target_filename)

    try:
        # VACUUM INTO による無停止・整合性保証スナップショット
        # パスのバックスラッシュをスラッシュに正規化
        safe_target_path = target_path.replace("\\", "/")
        with closing(sqlite3.connect(DEFAULT_DB_PATH)) as conn:
            conn.execute("VACUUM INTO ?", (safe_target_path,))

        file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[Backup] [SUCCESS] Google ドライブへの自動バックアップ成功: {target_filename} ({file_size_mb:.2f} MB)")

        # 世代管理 (古いバックアップの削除)
        cleanup_old_backups()
        return True
    except Exception as e:
        print(f"[Backup] [ERROR] バックアップ実行エラー: {e}")
        return False


def cleanup_old_backups(keep_count: int = MAX_BACKUP_GENERATIONS):
    """古いバックアップファイルを削除して指定世代数を維持"""
    try:
        pattern = os.path.join(BACKUP_DRIVE_DIR, "keiba_local_*.db")
        files = glob.glob(pattern)
        if len(files) <= keep_count:
            return

        # 作成日時昇順にソート (古い順)
        files.sort(key=lambda x: os.path.getmtime(x))
        delete_targets = files[:-keep_count]

        for df in delete_targets:
            try:
                os.remove(df)
                print(f"[Backup] 世代管理: 古いバックアップを削除しました -> {os.path.basename(df)}")
            except Exception as e:
                print(f"[Backup] [WARN] 削除エラー ({df}): {e}")
    except Exception as e:
        print(f"[Backup] [WARN] 世代管理処理エラー: {e}")


if __name__ == "__main__":
    sys.exit(0 if backup_database_to_drive() else 1)
