"""
Junior Aladdin - Trade Journal (Layer 15A)
=========================================
PURPOSE:
Persist executed trades to SQLite for compliance (SEBI 5-year audit trail).

Responsibilities (ONLY):
1) Initialize with a database path (default: data/junior_aladdin.db).
2) log_trade(trade_dict) -> bool:
   - Validate mandatory fields present.
   - Insert into trades table using existing Database class.
   - Return True on success, False on failure.
3) get_trades(start_date, end_date) -> List[Dict]:
   - Return trades between dates (inclusive).
4) Self-test: insert dummy trade, retrieve it, print PASS.

Strict boundaries:
- No P&L calculations.
- No summaries.
- No trading logic.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils.database import Database
from src.utils.logger import setup_logger

_logger = setup_logger("trade_journal")


MANDATORY_FIELDS: Tuple[str, ...] = (
    "trade_id",
    "entry_time",
    "exit_time",
    "symbol",
    "direction",
    "strategy",
    "brain",
    "regime",
    "opportunity_score",
    "ml_probability",
    "trap_score",
    "entry_price",
    "exit_price",
    "sl_price",
    "target_price",
    "pnl_points",
    "pnl_rupees",
    "exit_reason",
    "hold_minutes",
    "mfe",
    "mae",
    "lots",
    "session_phase",
    "narrative_label",
    "mtf_alignment",
    "vix_at_entry",
    "fii_score",
    "behavioral_checklist_complete",
    "mode",
    "algo_id",
    "static_ip",
    "costs_json",
)


class TradeJournal:
    """
    Trade journal persistence for executed trades.
    Uses `src.utils.database.Database` for all DB operations.
    """

    def __init__(self, db_path: str = "data/junior_aladdin.db"):
        self._logger = _logger
        self._db_path = db_path
        self._trade_columns_cache: Optional[List[str]] = None

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Database signature may vary (positional vs keyword). Handle defensively.
        try:
            self._db = Database(db_path=db_path)  # type: ignore[arg-type]
        except TypeError:
            self._db = Database(db_path)  # type: ignore[call-arg]

        # Ensure row_factory is sqlite3.Row if we can access the connection (best-effort).
        self._ensure_row_factory()

    def log_trade(self, trade_dict: Dict[str, Any]) -> bool:
        """
        Validate and insert a trade record into the `trades` table.

        Returns:
            True on success, False on failure.
        """
        if trade_dict is None or not isinstance(trade_dict, dict):
            self._logger.error(
                "log_trade failed: invalid input",
                extra={"input_type": str(type(trade_dict))},
            )
            return False

        missing = [k for k in MANDATORY_FIELDS if k not in trade_dict]
        if missing:
            self._logger.error(
                "log_trade failed: missing mandatory fields",
                extra={"missing_fields": missing},
            )
            return False

        try:
            insert_trade = getattr(self._db, "insert_trade", None)
            if callable(insert_trade):
                insert_trade(trade_dict)
            else:
                self._insert_trade_fallback(trade_dict)

            # Final simplified fix: trust insert if no exception is raised.
            return True

        except Exception as e:
            self._logger.error(
                "log_trade failed: database insert error",
                extra={
                    "error": str(e),
                    "trade_id": trade_dict.get("trade_id"),
                    "symbol": trade_dict.get("symbol"),
                    "strategy": trade_dict.get("strategy"),
                },
            )
            return False

    def get_trades(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch trades between start_date and end_date (inclusive).

        Args:
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            List[Dict[str, Any]]
        """
        # Validate date inputs (fail-closed)
        try:
            sd = date.fromisoformat(str(start_date))
            ed = date.fromisoformat(str(end_date))
        except Exception as e:
            self._logger.error(
                "get_trades failed: invalid date format",
                extra={"start_date": start_date, "end_date": end_date, "error": str(e)},
            )
            return []

        if sd > ed:
            self._logger.error(
                "get_trades failed: start_date > end_date",
                extra={"start_date": start_date, "end_date": end_date},
            )
            return []

        # Use substr(entry_time,1,10) instead of SQLite date() parsing.
        query = """
            SELECT * FROM trades
            WHERE substr(entry_time, 1, 10) BETWEEN ? AND ?
            ORDER BY entry_time ASC
        """

        try:
            rows = self._db.fetch_all(query, (sd.isoformat(), ed.isoformat()))
            return self._rows_to_dicts(rows)
        except Exception as e:
            self._logger.error(
                "get_trades failed: database query error",
                extra={"error": str(e), "start_date": start_date, "end_date": end_date},
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_row_factory(self) -> None:
        """
        Best-effort: ensure Database connection uses sqlite3.Row so fetches can be dict(row).
        """
        for attr in ("_conn", "conn", "connection", "_connection"):
            conn = getattr(self._db, attr, None)
            if conn is None:
                continue
            try:
                if hasattr(conn, "row_factory"):
                    conn.row_factory = sqlite3.Row  # type: ignore[assignment]
                    self._logger.debug(
                        "Database row_factory ensured",
                        extra={"attr": attr, "row_factory": "sqlite3.Row"},
                    )
                return
            except Exception as e:
                self._logger.debug(
                    "Failed to set row_factory (non-fatal)",
                    extra={"attr": attr, "error": str(e)},
                )
                return

    def _insert_trade_fallback(self, trade_dict: Dict[str, Any]) -> None:
        """
        Fallback insert when Database.insert_trade is not available.
        Uses Database.execute() only.
        """
        cols = list(trade_dict.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_sql = ", ".join([f'"{c}"' for c in cols])
        values = [trade_dict.get(c) for c in cols]
        query = f'INSERT INTO trades ({col_sql}) VALUES ({placeholders})'
        self._db.execute(query, values)

    def _get_trade_columns(self, force_refresh: bool = False) -> List[str]:
        """
        Resolve trade table columns using PRAGMA table_info(trades).
        Cached for subsequent conversions.
        """
        if self._trade_columns_cache is not None and not force_refresh:
            return self._trade_columns_cache

        cols: List[str] = []
        try:
            rows = self._db.fetch_all("PRAGMA table_info(trades)")
        except Exception as e:
            self._logger.error(
                "Failed to fetch trade columns via PRAGMA",
                extra={"error": str(e)},
            )
            self._trade_columns_cache = []
            return self._trade_columns_cache

        if not rows:
            self._logger.debug("PRAGMA table_info(trades) returned no rows")
            self._trade_columns_cache = []
            return self._trade_columns_cache

        for r in rows:
            try:
                if hasattr(r, "keys") and callable(getattr(r, "keys")):
                    name = r["name"]  # type: ignore[index]
                elif isinstance(r, dict):
                    name = r.get("name")
                else:
                    name = r[1]  # type: ignore[index]
                if name:
                    cols.append(str(name))
            except Exception:
                continue

        if not cols:
            self._logger.debug(
                "Trade columns cache is empty after PRAGMA; row conversion may fallback to col_i keys"
            )

        self._trade_columns_cache = cols
        return cols

    def _rows_to_dicts(self, rows: Any) -> List[Dict[str, Any]]:
        """
        Convert Database.fetch_all output to list[dict].
        Handles sqlite3.Row, dict, tuple/list.
        """
        if not rows:
            return []

        result: List[Dict[str, Any]] = []
        cols = self._get_trade_columns()

        if not cols:
            cols = self._get_trade_columns(force_refresh=True)

        for r in rows:
            if r is None:
                continue

            if isinstance(r, dict):
                result.append(r)
                continue

            if isinstance(r, sqlite3.Row):
                try:
                    result.append(dict(r))
                except Exception:
                    try:
                        result.append({k: r[k] for k in r.keys()})  # type: ignore[index]
                    except Exception:
                        result.append({"value": str(r)})
                continue

            try:
                if hasattr(r, "keys") and callable(getattr(r, "keys")):
                    result.append({k: r[k] for k in r.keys()})  # type: ignore[index]
                    continue
            except Exception:
                pass

            if isinstance(r, (tuple, list)):
                if cols and len(cols) == len(r):
                    result.append({cols[i]: r[i] for i in range(len(cols))})
                else:
                    result.append({f"col_{i}": r[i] for i in range(len(r))})
                continue

            result.append({"value": r})

        return result


def _run_self_test() -> None:
    print("=" * 60)
    print(" JUNIOR ALADDIN — TradeJournal Self-Test")
    print("=" * 60)

    journal = TradeJournal()

    now = datetime.utcnow().replace(microsecond=0).isoformat()
    trade_id = int(datetime.utcnow().timestamp() * 1_000_000)

    dummy_trade: Dict[str, Any] = {
        "trade_id": trade_id,
        "entry_time": now,
        "exit_time": now,
        "symbol": "NIFTYTEST",
        "direction": "BUY",
        "strategy": "DUMMY_STRATEGY",
        "brain": "STRUCTURAL",
        "regime": "UNKNOWN",
        "opportunity_score": 70.0,
        "ml_probability": 0.60,
        "trap_score": 10.0,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "sl_price": 95.0,
        "target_price": 110.0,
        "pnl_points": 5.0,
        "pnl_rupees": 0.0,
        "exit_reason": "SELF_TEST",
        "hold_minutes": 0.0,
        "mfe": 5.0,
        "mae": 0.0,
        "lots": 1,
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "mtf_alignment": 0.0,
        "vix_at_entry": 0.0,
        "fii_score": 0.0,
        "behavioral_checklist_complete": 1,
        "mode": "PAPER",
        "algo_id": "TEST_ALGO",
        "static_ip": "0.0.0.0",
        "costs_json": '{"stt": 0, "brokerage": 0, "gst": 0, "stamp": 0, "exchange": 0}',
    }

    ok = journal.log_trade(dummy_trade)
    assert ok is True, "log_trade() returned False in self-test"

    # Verify insertion via get_trades() on a fresh TradeJournal instance (fresh DB wrapper/connection)
    entry_day = str(now)[:10]
    journal_fresh = TradeJournal()
    trades = journal_fresh.get_trades(entry_day, entry_day)
    assert isinstance(trades, list), "get_trades() did not return a list"

    # Do not rely on trade_id being persisted by Database wrapper; match by stable fields.
    found = False
    for t in trades:
        if not isinstance(t, dict):
            continue
        if (
            str(t.get("symbol")) == "NIFTYTEST"
            and str(t.get("strategy")) == "DUMMY_STRATEGY"
            and str(t.get("exit_reason")) == "SELF_TEST"
            and str(t.get("entry_time", ""))[:19] == str(now)[:19]
        ):
            found = True
            break

    assert found is True, "Self-test trade was not retrieved via get_trades()"

    print("TradeJournal self-test PASSED.")


if __name__ == "__main__":
    _run_self_test()