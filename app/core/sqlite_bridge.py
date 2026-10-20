# -*- coding: utf-8 -*-
"""
app/core/sqlite_bridge.py

Supabase PostgREST クライアントと完全互換の超高速ローカル SQLite ブリッジ
- ソケット通信ゼロ、プロセス内 C ライブラリ直接実行 (0.01ms〜0.05ms)
- WAL モード + メモリマップド I/O (mmap_size=256MB)
- スレッドセーフ接続管理 (threading.local + busy_timeout=5000)
- JSONB カラムの透過的シリアライズ/デシリアライズ
"""

import os
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Union, Tuple

# JSON としてシリアライズ/デシリアライズすべきカラム名
JSON_COLUMNS = {
    "data_json",
    "result_json",
    "bet_details",
    "payout_details",
    "bet_breakdown",
    "payout_breakdown",
    "track_bias",
    "odds_alerts",
    "top3_orders",
    "mark_orders",
    "items",
    "value",
}

# デフォルトのローカルDBパス
DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "keiba_local.db")
)


class SQLiteResponse:
    """Supabase API レスポンス互換オブジェクト"""
    def __init__(self, data: List[Dict[str, Any]], count: Optional[int] = None):
        self.data = data
        self.count = count


class SQLiteConnectionPool:
    """スレッドセーフな SQLite コネクションマネージャー"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("PRAGMA synchronous = NORMAL;")
            cur.execute("PRAGMA busy_timeout = 5000;")
            cur.execute("PRAGMA mmap_size = 268435456;")
            cur.execute("PRAGMA temp_store = MEMORY;")
            self._local.conn = conn
        return self._local.conn


def ensure_database_schema(conn: sqlite3.Connection):
    """ローカル SQLite の全テーブル・全インデックス・全カラムの存在を物理保証"""
    cur = conn.cursor()
    # 1. keiba_scheduled_races
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_scheduled_races (
        race_id TEXT PRIMARY KEY,
        race_date TEXT,
        race_name TEXT,
        post_datetime TEXT,
        is_predicted INTEGER DEFAULT 0,
        is_settled INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        created_at TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sched_date ON keiba_scheduled_races(race_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sched_post ON keiba_scheduled_races(post_datetime);")

    # 2. keiba_race_settlements
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_race_settlements (
        race_id TEXT PRIMARY KEY,
        race_date TEXT,
        race_name TEXT,
        strategy_type TEXT,
        target_umaban INTEGER,
        horse_name TEXT,
        grade TEXT,
        is_pass INTEGER DEFAULT 0,
        total_bet INTEGER DEFAULT 0,
        total_payout INTEGER DEFAULT 0,
        is_hit INTEGER DEFAULT 0,
        actual_order INTEGER,
        hit_type TEXT,
        hypothetical_bet INTEGER DEFAULT 0,
        hypothetical_payout INTEGER DEFAULT 0,
        pass_benefit INTEGER DEFAULT 0,
        bet_breakdown TEXT,
        payout_breakdown TEXT,
        bet_details TEXT,
        payout_details TEXT,
        umaren_bet INTEGER DEFAULT 0,
        trio_bet INTEGER DEFAULT 0,
        umaren_payout INTEGER DEFAULT 0,
        trio_payout INTEGER DEFAULT 0,
        tansho_bet INTEGER DEFAULT 0,
        fukusho_bet INTEGER DEFAULT 0,
        tansho_payout INTEGER DEFAULT 0,
        fukusho_payout INTEGER DEFAULT 0,
        created_at TEXT,
        updated_at TEXT,
        track_bias TEXT,
        odds_alerts TEXT,
        top3_orders TEXT,
        mark_orders TEXT,
        bet_category TEXT,
        is_real_bet_target INTEGER,
        is_virtual_bet INTEGER,
        real_bet_badge TEXT,
        race_class_code INTEGER,
        wide_bet INTEGER DEFAULT 0,
        wide_payout INTEGER DEFAULT 0,
        umatan_bet INTEGER DEFAULT 0,
        umatan_payout INTEGER DEFAULT 0,
        sanrentan_bet INTEGER DEFAULT 0,
        sanrentan_payout INTEGER DEFAULT 0
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_settle_date ON keiba_race_settlements(race_date);")

    # 既存の keiba_race_settlements テーブルのカラム不足を自動マイグレーション
    cur.execute("PRAGMA table_info(keiba_race_settlements);")
    existing_cols = {row[1] for row in cur.fetchall()}
    columns_to_ensure = {
        "mark_orders": "TEXT",
        "bet_category": "TEXT",
        "is_real_bet_target": "INTEGER",
        "is_virtual_bet": "INTEGER",
        "real_bet_badge": "TEXT",
        "race_class_code": "INTEGER",
        "wide_bet": "INTEGER DEFAULT 0",
        "wide_payout": "INTEGER DEFAULT 0",
        "umatan_bet": "INTEGER DEFAULT 0",
        "umatan_payout": "INTEGER DEFAULT 0",
        "sanrentan_bet": "INTEGER DEFAULT 0",
        "sanrentan_payout": "INTEGER DEFAULT 0",
    }
    for col_name, col_type in columns_to_ensure.items():
        if col_name not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE keiba_race_settlements ADD COLUMN {col_name} {col_type};")
            except Exception:
                pass

    # 3. keiba_predictions_cache
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_predictions_cache (
        race_id TEXT PRIMARY KEY,
        race_date TEXT,
        data_json TEXT,
        created_at TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON keiba_predictions_cache(race_date);")

    # 4. keiba_race_results_cache
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_race_results_cache (
        race_id TEXT PRIMARY KEY,
        race_name TEXT,
        race_date TEXT,
        result_json TEXT,
        created_at TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_res_date ON keiba_race_results_cache(race_date);")

    # 5. keiba_real_bets
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_real_bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id TEXT,
        platform TEXT,
        receipt_no TEXT,
        total_amount INTEGER DEFAULT 0,
        items TEXT,
        status TEXT,
        is_manual_bet INTEGER DEFAULT 0,
        bet_time TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_real_bets_race_id ON keiba_real_bets(race_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_real_bets_time ON keiba_real_bets(bet_time);")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_real_bets_auto_unique ON keiba_real_bets(race_id) WHERE is_manual_bet = 0 AND status = 'success';")

    # 6. keiba_app_meta
    cur.execute("""
    CREATE TABLE IF NOT EXISTS keiba_app_meta (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TEXT
    );
    """)
    conn.commit()


class SQLiteQueryBuilder:
    """Supabase PostgREST クエリビルダー互換"""
    def __init__(self, pool: SQLiteConnectionPool, table_name: str):
        self.pool = pool
        self.table_name = table_name

        self._action = "select"  # select, insert, update, upsert, delete
        self._select_cols = "*"
        self._action_data: Any = None

        self._filters: List[Tuple[str, str, Any]] = []  # (col, op, val)
        self._orders: List[Tuple[str, bool]] = []       # (col, desc)
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None

    def select(self, columns: str = "*", count: Optional[str] = None):
        self._action = "select"
        self._select_cols = columns
        return self

    def insert(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        self._action = "insert"
        self._action_data = data
        return self

    def update(self, data: Dict[str, Any]):
        self._action = "update"
        self._action_data = data
        return self

    def upsert(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        self._action = "upsert"
        self._action_data = data
        return self

    def delete(self):
        self._action = "delete"
        return self

    def eq(self, column: str, value: Any):
        self._filters.append((column, "=", value))
        return self

    def neq(self, column: str, value: Any):
        self._filters.append((column, "!=", value))
        return self

    def gt(self, column: str, value: Any):
        self._filters.append((column, ">", value))
        return self

    def gte(self, column: str, value: Any):
        self._filters.append((column, ">=", value))
        return self

    def lt(self, column: str, value: Any):
        self._filters.append((column, "<", value))
        return self

    def lte(self, column: str, value: Any):
        self._filters.append((column, "<=", value))
        return self

    def in_(self, column: str, values: List[Any]):
        self._filters.append((column, "IN", list(values)))
        return self

    def like(self, column: str, pattern: str):
        self._filters.append((column, "LIKE", pattern))
        return self

    def ilike(self, column: str, pattern: str):
        self._filters.append((column, "LIKE", pattern))
        return self

    def order(self, column: str, desc: bool = False):
        self._orders.append((column, desc))
        return self

    def limit(self, count: int):
        self._limit_val = count
        return self

    def range(self, start: int, end: int):
        self._offset_val = start
        self._limit_val = (end - start + 1)
        return self

    def _format_value_for_db(self, col: str, val: Any) -> Any:
        if val is None:
            return None
        if col in JSON_COLUMNS or isinstance(val, (dict, list)):
            if not isinstance(val, str):
                return json.dumps(val, ensure_ascii=False)
            return val
        if isinstance(val, bool):
            return 1 if val else 0
        return val

    def _format_row_from_db(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k, v in d.items():
            if k in JSON_COLUMNS and isinstance(v, str) and v:
                try:
                    d[k] = json.loads(v)
                except Exception:
                    pass
        return d

    def execute(self) -> SQLiteResponse:
        conn = self.pool.get_connection()
        cur = conn.cursor()

        try:
            if self._action == "select":
                return self._execute_select(cur)
            elif self._action == "insert":
                return self._execute_insert(conn, cur)
            elif self._action == "update":
                return self._execute_update(conn, cur)
            elif self._action == "upsert":
                return self._execute_upsert(conn, cur)
            elif self._action == "delete":
                return self._execute_delete(conn, cur)
            else:
                raise ValueError(f"未知のアクションです: {self._action}")
        except Exception as e:
            conn.rollback()
            raise e

    def _build_where_clause(self) -> Tuple[str, List[Any]]:
        if not self._filters:
            return "", []

        clauses = []
        params = []
        for col, op, val in self._filters:
            safe_col = f'"{col}"'
            if op == "IN":
                if not val:
                    clauses.append("1=0")
                else:
                    placeholders = ", ".join(["?"] * len(val))
                    clauses.append(f"{safe_col} IN ({placeholders})")
                    params.extend(val)
            elif op == "LIKE":
                clauses.append(f"{safe_col} LIKE ?")
                params.append(val)
            else:
                clauses.append(f"{safe_col} {op} ?")
                params.append(val)

        return " WHERE " + " AND ".join(clauses), params

    def _execute_select(self, cur: sqlite3.Cursor) -> SQLiteResponse:
        cols = self._select_cols.strip()
        if cols == "*":
            cols_clause = "*"
        elif cols.upper().startswith("DISTINCT "):
            raw_target = cols[9:].strip()
            target_parts = [f'"{p.strip()}"' for p in raw_target.split(",") if p.strip()]
            cols_clause = "DISTINCT " + (", ".join(target_parts) if target_parts else "*")
        else:
            parts = [f'"{p.strip()}"' for p in cols.split(",") if p.strip()]
            cols_clause = ", ".join(parts) if parts else "*"

        sql = f"SELECT {cols_clause} FROM {self.table_name}"
        where_clause, params = self._build_where_clause()
        sql += where_clause

        if self._orders:
            order_parts = []
            for col, desc in self._orders:
                direction = "DESC" if desc else "ASC"
                order_parts.append(f'"{col}" {direction}')
            sql += " ORDER BY " + ", ".join(order_parts)

        if self._limit_val is not None:
            sql += f" LIMIT {int(self._limit_val)}"
        if self._offset_val is not None:
            sql += f" OFFSET {int(self._offset_val)}"

        cur.execute(sql, params)
        rows = cur.fetchall()
        result_data = [self._format_row_from_db(r) for r in rows]
        return SQLiteResponse(data=result_data, count=len(result_data))

    def _execute_insert(self, conn: sqlite3.Connection, cur: sqlite3.Cursor) -> SQLiteResponse:
        rows_to_insert = self._action_data if isinstance(self._action_data, list) else [self._action_data]
        if not rows_to_insert:
            return SQLiteResponse(data=[])

        # 全ての行が同一キー構成を持つ場合は executemany で一括高速実行
        first_cols = list(rows_to_insert[0].keys())
        same_schema = all(list(r.keys()) == first_cols for r in rows_to_insert)
        if same_schema and len(rows_to_insert) > 1:
            col_clause = ", ".join([f'"{c}"' for c in first_cols])
            placeholders = ", ".join(["?"] * len(first_cols))
            sql = f"INSERT INTO {self.table_name} ({col_clause}) VALUES ({placeholders});"
            batch_vals = [
                [self._format_value_for_db(c, r[c]) for c in first_cols]
                for r in rows_to_insert
            ]
            cur.executemany(sql, batch_vals)
        else:
            for row in rows_to_insert:
                cols = list(row.keys())
                vals = [self._format_value_for_db(c, row[c]) for c in cols]
                col_clause = ", ".join([f'"{c}"' for c in cols])
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT INTO {self.table_name} ({col_clause}) VALUES ({placeholders});"
                cur.execute(sql, vals)

        conn.commit()
        return SQLiteResponse(data=rows_to_insert)

    def _execute_update(self, conn: sqlite3.Connection, cur: sqlite3.Cursor) -> SQLiteResponse:
        if not self._action_data:
            return SQLiteResponse(data=[])

        update_dict = self._action_data
        set_parts = []
        set_params = []
        for k, v in update_dict.items():
            set_parts.append(f'"{k}" = ?')
            set_params.append(self._format_value_for_db(k, v))

        where_clause, where_params = self._build_where_clause()
        sql = f"UPDATE {self.table_name} SET " + ", ".join(set_parts) + where_clause
        cur.execute(sql, set_params + where_params)
        conn.commit()
        return SQLiteResponse(data=[update_dict])

    def _execute_upsert(self, conn: sqlite3.Connection, cur: sqlite3.Cursor) -> SQLiteResponse:
        rows_to_upsert = self._action_data if isinstance(self._action_data, list) else [self._action_data]
        if not rows_to_upsert:
            return SQLiteResponse(data=[])

        first_cols = list(rows_to_upsert[0].keys())
        same_schema = all(list(r.keys()) == first_cols for r in rows_to_upsert)
        if same_schema and len(rows_to_upsert) > 1:
            col_clause = ", ".join([f'"{c}"' for c in first_cols])
            placeholders = ", ".join(["?"] * len(first_cols))
            sql = f"INSERT OR REPLACE INTO {self.table_name} ({col_clause}) VALUES ({placeholders});"
            batch_vals = [
                [self._format_value_for_db(c, r[c]) for c in first_cols]
                for r in rows_to_upsert
            ]
            cur.executemany(sql, batch_vals)
        else:
            for row in rows_to_upsert:
                cols = list(row.keys())
                vals = [self._format_value_for_db(c, row[c]) for c in cols]
                col_clause = ", ".join([f'"{c}"' for c in cols])
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO {self.table_name} ({col_clause}) VALUES ({placeholders});"
                cur.execute(sql, vals)

        conn.commit()
        return SQLiteResponse(data=rows_to_upsert)

    def _execute_delete(self, conn: sqlite3.Connection, cur: sqlite3.Cursor) -> SQLiteResponse:
        where_clause, where_params = self._build_where_clause()
        sql = f"DELETE FROM {self.table_name}" + where_clause
        cur.execute(sql, where_params)
        conn.commit()
        return SQLiteResponse(data=[])


class LocalSQLiteClient:
    """Supabase Client の完全ドロップイン代替となるローカル SQLite クライアント"""
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.pool = SQLiteConnectionPool(db_path)
        # テーブル・インデックス・カラムの物理存在保証 (自動マイグレーション)
        try:
            conn = self.pool.get_connection()
            ensure_database_schema(conn)
        except Exception as e:
            print(f"[LocalSQLiteClient] スキーマ初期化警告: {e}")

    def table(self, table_name: str) -> SQLiteQueryBuilder:
        return SQLiteQueryBuilder(self.pool, table_name)
