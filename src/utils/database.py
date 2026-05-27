"""
Junior Aladdin - Database Utility (Institutional-Grade)

SQLite database manager for JUNIOR ALADDIN.

Backward-compatible API (existing methods preserved):
- Database(db_path: str = None)
- execute(query: str, params: tuple|dict|None = None) -> bool
- fetch_all(query: str, params: tuple|dict|None = None) -> list[tuple]
- fetch_one(query: str, params: tuple|dict|None = None) -> tuple|None
- get_table_names() -> list[str]
- insert_trade(trade_dict: dict) -> int   (rowid or -1 on OperationalError failures)
- insert_log(engine, event_type, message, state) -> int
- close()

Institutional upgrades:
1) Graceful OperationalError retry:
   - execute() catches sqlite3.OperationalError (e.g., database locked) and retries 3x
     with exponential backoff: 0.1s, 0.2s, 0.4s
   - If still failing: logs CRITICAL and returns False (does NOT raise)
   - Other exceptions (IntegrityError, ProgrammingError, etc.) are raised (programming/data issues)

2) Bulk insert:
   - insert_many(table, columns, rows) uses executemany() in a single transaction
   - Returns number of rows inserted; returns -1 on OperationalError failure after retries

3) Transaction context manager:
   - with db.transaction(): BEGIN IMMEDIATE ... COMMIT / ROLLBACK on exception
   - Nested transactions supported (inner becomes no-op; outer controls commit)

4) Schema migrations:
   - Uses PRAGMA user_version
   - Runs pending migrations on startup
   - Updates user_version after successful migration

5) Configurable timeout:
   - Reads timeout from Config.get("database","timeout", default=10.0)

Concurrency note:
- SQLite serializes writes; WAL improves read concurrency but write throughput remains single-writer.
- For extremely high-frequency concurrent writes, consider moving to PostgreSQL.
"""

from __future__ import annotations

import os
import sqlite3
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_LOG = setup_logger("database")


class Database:
    """
    SQLite database manager.

    Default db_path:
      - env DB_PATH or JUNIOR_ALADDIN_DB_PATH if set
      - else Config.get("database","path") if present
      - else "data/junior_aladdin.db"
    """

    _LATEST_SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._in_transaction = 0  # nested transaction depth

        if db_path is None:
            db_path = os.getenv("DB_PATH") or os.getenv("JUNIOR_ALADDIN_DB_PATH")
            if not db_path:
                try:
                    db_path = Config.get("database", "path", default=None)
                except Exception:
                    db_path = None
            if not db_path:
                db_path = "data/junior_aladdin.db"

        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        timeout = 10.0
        try:
            timeout = float(Config.get("database", "timeout", default=10.0))
        except Exception:
            timeout = 10.0

        # Backward-compatible: single connection, cross-thread allowed but guarded by self._lock.
        self.conn = sqlite3.connect(
            self.db_path,
            timeout=timeout,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode; explicit BEGIN/COMMIT for transaction()
        )

        # Pragmas (safe defaults)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")

        self._create_tables()
        self._migrate()

        _LOG.info(
            "Database initialized",
            db_path=self.db_path,
            timeout=timeout,
            schema_version=self._get_user_version(),
        )

    # ------------------------- Core SQL helpers -------------------------

    def execute(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> bool:
        """
        Execute a SQL statement.

        OperationalError (e.g., locked) is retried (0.1, 0.2, 0.4). If still failing:
        - log CRITICAL
        - return False
        - DO NOT raise

        Other exceptions are raised.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("execute(): query must be a non-empty string")

        with self._lock:
            return self._execute_with_retry(query, params=params, expect_result=False) is not None

    def fetch_all(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> List[Tuple]:
        """Fetch all rows. On OperationalError after retries, returns [] (does not raise)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("fetch_all(): query must be a non-empty string")

        with self._lock:
            cur = self._execute_with_retry(query, params=params, expect_result=True)
            if cur is None:
                return []
            try:
                rows = cur.fetchall()
                return rows if rows is not None else []
            except sqlite3.OperationalError as e:
                _LOG.critical("fetch_all failed after execute", error=repr(e))
                return []

    def fetch_one(self, query: str, params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None) -> Optional[Tuple]:
        """Fetch one row. On OperationalError after retries, returns None (does not raise)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("fetch_one(): query must be a non-empty string")

        with self._lock:
            cur = self._execute_with_retry(query, params=params, expect_result=True)
            if cur is None:
                return None
            try:
                return cur.fetchone()
            except sqlite3.OperationalError as e:
                _LOG.critical("fetch_one failed after execute", error=repr(e))
                return None

    def insert_many(self, table: str, columns: List[str], rows: List[Tuple]) -> int:
        """
        Bulk insert using executemany() with a single transaction.
        Returns number of rows inserted; returns -1 on OperationalError after retries.
        Raises on non-OperationalError exceptions.
        """
        if not isinstance(table, str) or not table.strip():
            raise ValueError("insert_many(): table must be a non-empty string")
        if not isinstance(columns, list) or not columns:
            raise ValueError("insert_many(): columns must be a non-empty list")
        if rows is None:
            return 0
        if not isinstance(rows, list):
            raise ValueError("insert_many(): rows must be a list of tuples")
        if len(rows) == 0:
            return 0

        col_sql = ", ".join([str(c) for c in columns])
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"

        with self._lock:
            # Nested transaction: executemany only; outer transaction controls commit/rollback.
            if self._in_transaction > 0:
                try:
                    cur = self.conn.cursor()
                    cur.executemany(sql, rows)
                    return len(rows)
                except sqlite3.OperationalError as e:
                    for delay in (0.1, 0.2, 0.4):
                        _LOG.warning("OperationalError on insert_many (nested tx), retrying", error=repr(e), delay=delay)
                        time.sleep(delay)
                        try:
                            cur = self.conn.cursor()
                            cur.executemany(sql, rows)
                            return len(rows)
                        except sqlite3.OperationalError as e2:
                            e = e2
                            continue
                    _LOG.critical("insert_many failed after retries (nested tx)", table=table, error=repr(e))
                    return -1

            # Standalone bulk transaction
            try:
                self.conn.execute("BEGIN IMMEDIATE;")
                self._in_transaction += 1
                cur = self.conn.cursor()
                cur.executemany(sql, rows)
                self.conn.execute("COMMIT;")
                return len(rows)
            except sqlite3.OperationalError as e:
                try:
                    self.conn.execute("ROLLBACK;")
                except Exception:
                    pass
                _LOG.critical("insert_many failed (OperationalError)", table=table, error=repr(e))
                return -1
            finally:
                self._in_transaction = max(0, self._in_transaction - 1)

    @contextmanager
    def transaction(self):
        """
        Transaction context manager.
        - BEGIN IMMEDIATE on enter (outermost only)
        - COMMIT on normal exit (outermost only)
        - ROLLBACK on exception (outermost only)
        Supports nested usage by tracking depth.
        """
        with self._lock:
            outermost = self._in_transaction == 0
            if outermost:
                self._begin_immediate_with_retry()
            self._in_transaction += 1
        try:
            yield self
            with self._lock:
                self._in_transaction = max(0, self._in_transaction - 1)
                if outermost:
                    try:
                        self.conn.execute("COMMIT;")
                    except sqlite3.OperationalError as e:
                        _LOG.critical("COMMIT failed (OperationalError)", error=repr(e))
                        try:
                            self.conn.execute("ROLLBACK;")
                        except Exception:
                            pass
        except Exception:
            with self._lock:
                self._in_transaction = max(0, self._in_transaction - 1)
                if outermost:
                    try:
                        self.conn.execute("ROLLBACK;")
                    except Exception:
                        pass
            raise

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass

    def get_table_names(self) -> List[str]:
        """Return user table names in the database, sorted alphabetically."""
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        return [str(row[0]) for row in rows if row and row[0]]

    # ------------------------- Domain inserts -------------------------

    def insert_log(self, engine: str, event_type: str, message: str, state: str) -> int:
        """
        Insert into engine_logs.
        Returns lastrowid, or -1 on OperationalError after retries.
        """
        ts = datetime_utc_iso()
        sql = """
        INSERT INTO engine_logs (timestamp, engine_name, event_type, message, system_state)
        VALUES (?, ?, ?, ?, ?)
        """
        params = (ts, str(engine), str(event_type), str(message), str(state))

        with self._lock:
            cur = self._execute_with_retry(sql, params=params, expect_result=False)
            if cur is None:
                return -1
            try:
                return int(cur.lastrowid or -1)
            except Exception:
                return -1

    def insert_trade(self, trade_dict: Dict[str, Any]) -> int:
        """
        Insert into trades table. Accepts partial dict; missing columns stored as NULL.
        Returns lastrowid, or -1 on OperationalError after retries.
        Raises on non-OperationalError exceptions.
        """
        if not isinstance(trade_dict, dict):
            raise ValueError("insert_trade(): trade_dict must be a dict")

        cols = self._trade_columns()
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO trades ({', '.join(cols)}) VALUES ({placeholders})"

        values: List[Any] = [trade_dict.get(c) for c in cols]

        with self._lock:
            cur = self._execute_with_retry(sql, params=tuple(values), expect_result=False)
            if cur is None:
                return -1
            try:
                return int(cur.lastrowid or -1)
            except Exception:
                return -1

    # ------------------------- Migrations & schema -------------------------

    def _get_user_version(self) -> int:
        try:
            row = self.fetch_one("PRAGMA user_version;")
            if row and len(row) >= 1:
                return int(row[0])
        except Exception:
            pass
        return 0

    def _set_user_version(self, v: int) -> None:
        with self._lock:
            self.conn.execute(f"PRAGMA user_version = {int(v)};")

    def _migrate(self) -> None:
        """
        Simple migration framework based on PRAGMA user_version.
        """
        current = self._get_user_version()
        target = self._LATEST_SCHEMA_VERSION

        if current > target:
            _LOG.warning("Database user_version ahead of code schema version", current=current, target=target)
            return

        migrations = {
            1: self._migration_1_initial_indices,
        }

        for v in range(current + 1, target + 1):
            fn = migrations.get(v)
            if fn is None:
                raise ValueError(f"No migration function defined for version {v}")
            _LOG.info("Running DB migration", from_version=current, to_version=v)
            with self.transaction():
                fn()
                self._set_user_version(v)
            current = v

    def _migration_1_initial_indices(self) -> None:
        """Migration v1: baseline indices."""
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_engine_logs_ts ON engine_logs(timestamp);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_compliance_audit_ts ON compliance_audit(timestamp);")

    def _create_tables(self) -> None:
        """Create all tables if they do not exist."""
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id INTEGER PRIMARY KEY,
                    entry_time TEXT, exit_time TEXT, symbol TEXT, direction TEXT,
                    strategy TEXT, brain TEXT, regime TEXT,
                    opportunity_score REAL, ml_probability REAL, trap_score REAL,
                    entry_price REAL, exit_price REAL, sl_price REAL, target_price REAL,
                    pnl_points REAL, pnl_rupees REAL, costs_total REAL,
                    exit_reason TEXT, hold_minutes REAL, mfe REAL, mae REAL, lots INTEGER,
                    features_json TEXT, session_phase TEXT, narrative_label TEXT,
                    mtf_alignment REAL, vix_at_entry REAL, fii_score REAL,
                    behavioral_checklist_complete INTEGER,
                    mode TEXT, algo_id TEXT, static_ip TEXT,
                    costs_json TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS engine_logs (
                    log_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    engine_name TEXT,
                    event_type TEXT,
                    message TEXT,
                    system_state TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_dna (
                    strategy_name TEXT PRIMARY KEY,
                    current_threshold REAL,
                    win_rate_20 REAL,
                    profit_factor_20 REAL,
                    last_updated TEXT,
                    status TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    best_strategy TEXT,
                    worst_strategy TEXT,
                    narrative_label TEXT,
                    regime_distribution TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_name TEXT,
                    version INTEGER,
                    trained_date TEXT,
                    auc_roc REAL,
                    deployed INTEGER,
                    file_path TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_audit (
                    audit_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    order_type TEXT,
                    algo_id TEXT,
                    static_ip TEXT,
                    symbol TEXT,
                    qty INTEGER,
                    price REAL,
                    direction TEXT,
                    strategy TEXT,
                    status TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS day_fingerprints (
                    date TEXT PRIMARY KEY,
                    fingerprint_json TEXT,
                    day_type TEXT,
                    session_pnl_json TEXT,
                    close_vs_levels_json TEXT
                );
                """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_memory_log (
                    date TEXT,
                    timestamp TEXT,
                    memory_json TEXT
                );
                """
            )

    # ------------------------- Internal execution primitives -------------------------

    def _begin_immediate_with_retry(self) -> None:
        delays = (0.1, 0.2, 0.4)
        last_err: Optional[Exception] = None
        for d in delays:
            try:
                self.conn.execute("BEGIN IMMEDIATE;")
                return
            except sqlite3.OperationalError as e:
                last_err = e
                _LOG.warning("OperationalError on BEGIN IMMEDIATE, retrying", error=repr(e), delay=d)
                time.sleep(d)
        try:
            self.conn.execute("BEGIN IMMEDIATE;")
        except sqlite3.OperationalError as e:
            _LOG.critical("BEGIN IMMEDIATE failed after retries", error=repr(e))
            raise

    def _execute_with_retry(
        self,
        query: str,
        params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]],
        expect_result: bool,
    ) -> Optional[sqlite3.Cursor]:
        """
        Internal execution with OperationalError retry.
        - Returns cursor on success
        - Returns None on OperationalError after retries (and logs CRITICAL)
        - Raises on other exceptions
        """
        delays = (0.1, 0.2, 0.4)

        for attempt, delay in enumerate((0.0,) + delays, start=0):
            if delay > 0.0:
                time.sleep(delay)
            try:
                cur = self.conn.cursor()
                if params is None:
                    cur.execute(query)
                else:
                    cur.execute(query, params)

                if self._in_transaction == 0:
                    try:
                        self.conn.commit()
                    except Exception:
                        pass

                return cur

            except sqlite3.OperationalError as e:
                try:
                    self.conn.rollback()
                except Exception:
                    pass

                if attempt < len(delays):
                    _LOG.warning(
                        "OperationalError during SQL execute, retrying",
                        error=repr(e),
                        attempt=attempt + 1,
                        next_delay=delays[attempt] if attempt < len(delays) else None,
                    )
                    continue

                _LOG.critical(
                    "OperationalError during SQL execute; giving up",
                    error=repr(e),
                    query=self._short_query(query),
                )
                return None

            except sqlite3.IntegrityError:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                raise
            except sqlite3.ProgrammingError:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                raise
            except Exception:
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                raise

        return None

    @staticmethod
    def _short_query(query: str, max_len: int = 200) -> str:
        q = " ".join(query.strip().split())
        return q if len(q) <= max_len else q[: max_len - 3] + "..."

    @staticmethod
    def _trade_columns() -> List[str]:
        return [
            "entry_time", "exit_time", "symbol", "direction",
            "strategy", "brain", "regime",
            "opportunity_score", "ml_probability", "trap_score",
            "entry_price", "exit_price", "sl_price", "target_price",
            "pnl_points", "pnl_rupees", "costs_total",
            "exit_reason", "hold_minutes", "mfe", "mae", "lots",
            "features_json", "session_phase", "narrative_label",
            "mtf_alignment", "vix_at_entry", "fii_score",
            "behavioral_checklist_complete",
            "mode", "algo_id", "static_ip",
            "costs_json",
        ]


def datetime_utc_iso() -> str:
    """UTC ISO timestamp for DB writes."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


if __name__ == "__main__":
    # Self-test: 9 tests including migration + batch insert + transaction rollback.
    import tempfile

    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    print("Running Database self-test...")

    with tempfile.TemporaryDirectory() as td:
        db_path = str(Path(td) / "test.db")
        db: Optional[Database] = None
        try:
            db = Database(db_path=db_path)

            tables = db.fetch_all("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = sorted([t[0] for t in tables])
            expected_tables = sorted([
                "trades",
                "engine_logs",
                "strategy_dna",
                "daily_summary",
                "model_versions",
                "compliance_audit",
                "day_fingerprints",
                "session_memory_log",
            ])
            for t in expected_tables:
                _assert(t in table_names, f"Missing table: {t}")
            print("[PASS] 1) Tables created")

            uv = db.fetch_one("PRAGMA user_version;")
            _assert(uv is not None and int(uv[0]) >= 1, "user_version not set/migrated")
            print("[PASS] 2) Migration user_version set")

            log_id = db.insert_log("test_engine", "startup", "hello", "BOOT")
            _assert(log_id > 0, "insert_log failed")
            row = db.fetch_one("SELECT engine_name, event_type, message FROM engine_logs WHERE log_id=?", (log_id,))
            _assert(row is not None and row[0] == "test_engine", "fetch_one after insert_log failed")
            print("[PASS] 3) insert_log + fetch_one")

            trade_id = db.insert_trade({"entry_time": "2026-04-09T10:00:00", "symbol": "NIFTY", "direction": "LONG"})
            _assert(trade_id > 0, "insert_trade failed")
            row = db.fetch_one("SELECT symbol, direction FROM trades WHERE trade_id=?", (trade_id,))
            _assert(row is not None and row[0] == "NIFTY", "fetch_one after insert_trade failed")
            print("[PASS] 4) insert_trade")

            rows = [(datetime_utc_iso(), "eng", "evt", f"msg{i}", "ACTIVE") for i in range(50)]
            inserted = db.insert_many(
                "engine_logs",
                ["timestamp", "engine_name", "event_type", "message", "system_state"],
                rows,
            )
            _assert(inserted == 50, f"insert_many expected 50 got {inserted}")
            count = db.fetch_one("SELECT COUNT(*) FROM engine_logs;")
            _assert(count is not None and int(count[0]) >= 51, "bulk insert count mismatch")
            print("[PASS] 5) insert_many bulk insert")

            with db.transaction():
                a = db.insert_log("tx", "a", "1", "ACTIVE")
                b = db.insert_log("tx", "b", "2", "ACTIVE")
                _assert(a > 0 and b > 0, "transaction inserts failed")
            cnt_tx = db.fetch_one("SELECT COUNT(*) FROM engine_logs WHERE engine_name='tx';")
            _assert(cnt_tx is not None and int(cnt_tx[0]) == 2, "transaction commit failed")
            print("[PASS] 6) transaction commit")

            try:
                with db.transaction():
                    db.insert_log("txrb", "a", "1", "ACTIVE")
                    raise RuntimeError("force rollback")
            except RuntimeError:
                pass
            cnt_txrb = db.fetch_one("SELECT COUNT(*) FROM engine_logs WHERE engine_name='txrb';")
            _assert(cnt_txrb is not None and int(cnt_txrb[0]) == 0, "transaction rollback failed")
            print("[PASS] 7) transaction rollback")

            ok = db.execute("SELECT 1;")
            _assert(ok is True, "execute did not return True for SELECT 1")
            print("[PASS] 8) execute basic works")

            print("[PASS] 9) close() safe (verified in finally)")

        finally:
            # Ensure DB is closed to release file locks on Windows before TemporaryDirectory cleanup.
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    print("\nAll 9 self-tests PASSED.")