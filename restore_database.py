# -*- coding: utf-8 -*-
"""
backend-render/restore_database.py

Google ドライブ上のバックアップから SQLite データベースを安全に復元（リストア）するスクリプト
- 復元前に現在の本番DBを自動で .bak 退避（事故防止）
- 復元元ファイルの整合性（PRAGMA integrity_check）を事前検証
- 復元後に WAL モードおよびメモリ最適化 PRAGMA を再適用
- テーブル件数と健全性をサマリー表示
"""

import os
import sys
import glob
import shutil
import sqlite3
import datetime
import argparse

BACKUP_DIR = r"G:\マイドライブ\keiba_data\backup"
DEFAULT_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
TARGET_DB_PATH = os.path.join(DEFAULT_DB_DIR, "keiba_local.db")


def restore(backup_path: str = None) -> bool:
    print("=" * 65)
    print("🔄 KEIBA AI - DATABASE RESTORE UTILITY")
    print("=" * 65)

    # 1. バックアップファイルの特定
    if not os.path.exists(BACKUP_DIR):
        print(f"❌ Google ドライブのバックアップディレクトリが見つかりません: {BACKUP_DIR}")
        return False

    backup_files = glob.glob(os.path.join(BACKUP_DIR, "keiba_local_*.db"))
    if not backup_files:
        print(f"❌ 有効なバックアップファイルが見つかりません: {BACKUP_DIR}")
        return False

    backup_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if not backup_path:
        selected_backup = backup_files[0]
        print(f"[1] 最新バックアップを自動選択しました:\n    -> {selected_backup}")
    else:
        selected_backup = backup_path
        if not os.path.exists(selected_backup):
            print(f"❌ 指定されたバックアップファイルが存在しません: {selected_backup}")
            return False
        print(f"[1] 指定されたバックアップを選択しました:\n    -> {selected_backup}")

    file_size_mb = os.path.getsize(selected_backup) / (1024 * 1024)
    print(f"    サイズ: {file_size_mb:.2f} MB")

    # 2. バックアップ元の事前整合性チェック
    print("\n[2] バックアップファイルの整合性を検証中 (PRAGMA integrity_check)...")
    try:
        check_conn = sqlite3.connect(selected_backup)
        check_cur = check_conn.cursor()
        check_cur.execute("PRAGMA integrity_check;")
        res = check_cur.fetchall()
        check_conn.close()
        if res != [("ok",)]:
            print(f"❌ バックアップファイルが破損しています: {res}")
            return False
        print("    ✅ 整合性チェック合格 (100% 健全)")
    except Exception as e:
        print(f"❌ バックアップ検証中に例外が発生しました: {e}")
        return False

    # 3. 現在の本番DBを .bak に安全退避
    os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
    if os.path.exists(TARGET_DB_PATH):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_path = f"{TARGET_DB_PATH}.bak_{ts}"
        try:
            # WALファイルがある場合を考慮し shutil.copy2
            shutil.copy2(TARGET_DB_PATH, bak_path)
            print(f"\n[3] 現在の本番DBを安全に退避しました:\n    -> {bak_path}")
        except Exception as e:
            print(f"⚠️ 現在のDBの退避に失敗しましたが続行します: {e}")

    # 4. データベースの復元展開
    print(f"\n[4] データベースを本番パスへ復元中:\n    -> {TARGET_DB_PATH}")
    try:
        # WAL / SHM ファイルの古い残存があればクリーンアップ
        for ext in ["-wal", "-shm"]:
            w_file = TARGET_DB_PATH + ext
            if os.path.exists(w_file):
                try:
                    os.remove(w_file)
                except Exception:
                    pass

        shutil.copy2(selected_backup, TARGET_DB_PATH)
        print("    ✅ データベースファイルのコピーが完了しました。")
    except Exception as e:
        print(f"❌ 復元コピーに失敗しました: {e}")
        return False

    # 5. 最適化 PRAGMA の再適用 ＆ テーブル件数確認
    print("\n[5] SQLite WALモード・メモリ最適化 PRAGMA を再設定中...")
    try:
        conn = sqlite3.connect(TARGET_DB_PATH)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA mmap_size = 268435456;")
        cur.execute("PRAGMA temp_store = MEMORY;")

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]

        print(f"    検出テーブル ({len(tables)}件):")
        for t in tables:
            cur.execute(f'SELECT count(*) FROM "{t}"')
            cnt = cur.fetchone()[0]
            print(f"      - {t}: {cnt:,} 件")

        conn.close()
    except Exception as e:
        print(f"⚠️ PRAGMA設定またはテーブル確認中にエラー: {e}")

    print("\n" + "=" * 65)
    print("🎉 データベースの復元が正常に完了しました！")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keiba AI Database Restore Tool")
    parser.add_argument("--backup", type=str, help="復元元の特定バックアップファイルパス (省略時は最新)")
    args = parser.parse_args()
    success = restore(args.backup)
    sys.exit(0 if success else 1)
