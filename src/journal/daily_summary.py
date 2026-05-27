"""
Junior Aladdin - Daily Summary (Layer 15A)
=========================================
PURPOSE:
Generate an end-of-day performance summary from the trade journal.

STRICT BOUNDARIES:
- Do NOT modify the database schema.
- Do NOT contain any trading logic.
- Use the existing TradeJournal class for all data access.

OUTPUT:
JSON-serializable dictionary (basic Python types only).
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple, Union

from src.journal.trade_journal import TradeJournal
from src.utils.logger import setup_logger

_logger = setup_logger("daily_summary")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        v = int(float(value))
        return v
    except Exception:
        return default


def _sum_costs(costs_json: Any) -> float:
    """
    Sum numeric values inside costs_json.
    Accepts dict or JSON string. Returns 0.0 on parse failure.
    """
    try:
        if costs_json is None:
            return 0.0
        if isinstance(costs_json, dict):
            return float(
                sum(_safe_float(v, 0.0) for v in costs_json.values() if isinstance(v, (int, float, str)))
            )
        if isinstance(costs_json, str):
            s = costs_json.strip()
            if not s:
                return 0.0
            obj = json.loads(s)
            if isinstance(obj, dict):
                return float(
                    sum(_safe_float(v, 0.0) for v in obj.values() if isinstance(v, (int, float, str)))
                )
    except Exception:
        return 0.0
    return 0.0


class DailySummary:
    """
    End-of-day performance summary generator.

    Initialize with:
      - a TradeJournal instance, OR
      - a database path (will create an internal TradeJournal)
    """

    def __init__(self, trade_journal: Optional[TradeJournal] = None, db_path: str = "data/junior_aladdin.db"):
        self._logger = _logger
        self._journal = trade_journal if trade_journal is not None else TradeJournal(db_path=db_path)

    def generate(self, date_str: str) -> Dict[str, Any]:
        """
        Generate summary for a date (YYYY-MM-DD).

        Returns JSON-serializable dict with keys:
          - total_trades
          - winning_trades, losing_trades
          - win_rate
          - gross_pnl_points, gross_pnl_rupees
          - net_pnl_points, net_pnl_rupees (after costs)
          - total_costs
          - profit_factor
          - largest_win, largest_loss
          - average_win, average_loss
          - best_strategy, worst_strategy
          - performance_by_regime, performance_by_session
        """
        try:
            # Validate date format
            _ = date.fromisoformat(str(date_str))
        except Exception as e:
            self._logger.error("DailySummary.generate invalid date_str", extra={"date_str": date_str, "error": str(e)})
            return self._empty_summary()

        trades = self._journal.get_trades(date_str, date_str)
        if not trades:
            return self._empty_summary()

        total_trades = len(trades)

        net_pnl_points = 0.0
        net_pnl_rupees = 0.0
        gross_pnl_points = 0.0
        gross_pnl_rupees = 0.0
        total_costs = 0.0

        wins: List[float] = []
        losses: List[float] = []

        strategy_pnl: Dict[str, float] = {}
        regime_stats: Dict[str, Dict[str, Any]] = {}
        session_stats: Dict[str, Dict[str, Any]] = {}

        for t in trades:
            if not isinstance(t, dict):
                continue

            pnl_points = _safe_float(t.get("pnl_points"), 0.0)
            pnl_rupees = _safe_float(t.get("pnl_rupees"), 0.0)
            costs = _sum_costs(t.get("costs_json"))

            net_pnl_points += pnl_points
            net_pnl_rupees += pnl_rupees
            gross_pnl_points += pnl_points
            gross_pnl_rupees += pnl_rupees + costs
            total_costs += costs

            if pnl_rupees > 0:
                wins.append(pnl_rupees)
            elif pnl_rupees < 0:
                losses.append(pnl_rupees)

            strategy = str(t.get("strategy", "UNKNOWN"))
            strategy_pnl[strategy] = strategy_pnl.get(strategy, 0.0) + pnl_rupees

            regime = str(t.get("regime", "UNKNOWN"))
            session = str(t.get("session_phase", "UNKNOWN"))

            self._accumulate_group(regime_stats, regime, pnl_rupees)
            self._accumulate_group(session_stats, session, pnl_rupees)

        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = round((winning_trades / total_trades) * 100.0, 2) if total_trades > 0 else 0.0

        gross_profit = sum(wins)
        gross_loss_abs = abs(sum(losses))  # losses are negative
        profit_factor: Optional[float]
        if gross_loss_abs <= 0:
            profit_factor = None
        else:
            profit_factor = round(gross_profit / gross_loss_abs, 3)

        largest_win = round(max(wins), 2) if wins else 0.0
        largest_loss = round(min(losses), 2) if losses else 0.0  # most negative
        average_win = round(sum(wins) / len(wins), 2) if wins else 0.0
        average_loss = round(sum(losses) / len(losses), 2) if losses else 0.0

        best_strategy = None
        worst_strategy = None
        if strategy_pnl:
            best_strategy = max(strategy_pnl.items(), key=lambda kv: kv[1])[0]
            worst_strategy = min(strategy_pnl.items(), key=lambda kv: kv[1])[0]

        performance_by_regime = self._finalize_group(regime_stats)
        performance_by_session = self._finalize_group(session_stats)

        return {
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "win_rate": float(win_rate),
            "gross_pnl_points": round(float(gross_pnl_points), 2),
            "gross_pnl_rupees": round(float(gross_pnl_rupees), 2),
            "net_pnl_points": round(float(net_pnl_points), 2),
            "net_pnl_rupees": round(float(net_pnl_rupees), 2),
            "total_costs": round(float(total_costs), 2),
            "profit_factor": profit_factor,
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "average_win": float(average_win),
            "average_loss": float(average_loss),
            "best_strategy": best_strategy,
            "worst_strategy": worst_strategy,
            "performance_by_regime": performance_by_regime,
            "performance_by_session": performance_by_session,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "gross_pnl_points": 0.0,
            "gross_pnl_rupees": 0.0,
            "net_pnl_points": 0.0,
            "net_pnl_rupees": 0.0,
            "total_costs": 0.0,
            "profit_factor": None,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "best_strategy": None,
            "worst_strategy": None,
            "performance_by_regime": {},
            "performance_by_session": {},
        }

    def _accumulate_group(self, store: Dict[str, Dict[str, Any]], key: str, pnl_rupees: float) -> None:
        if key not in store:
            store[key] = {"trades": 0, "wins": 0, "losses": 0, "net_pnl_rupees": 0.0}
        store[key]["trades"] += 1
        store[key]["net_pnl_rupees"] += float(pnl_rupees)
        if pnl_rupees > 0:
            store[key]["wins"] += 1
        elif pnl_rupees < 0:
            store[key]["losses"] += 1

    def _finalize_group(self, store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in store.items():
            trades = _safe_int(v.get("trades"), 0)
            wins = _safe_int(v.get("wins"), 0)
            win_rate = round((wins / trades) * 100.0, 2) if trades > 0 else 0.0
            out[str(k)] = {
                "trades": int(trades),
                "wins": int(wins),
                "losses": int(_safe_int(v.get("losses"), 0)),
                "win_rate": float(win_rate),
                "net_pnl_rupees": round(float(_safe_float(v.get("net_pnl_rupees"), 0.0)), 2),
            }
        return out


def _run_self_test() -> None:
    print("=" * 60)
    print(" JUNIOR ALADDIN — DailySummary Self-Test")
    print("=" * 60)

    test_db_path = "data/daily_summary_selftest.db"
    # Start fresh
    try:
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
    except Exception:
        pass

    journal = TradeJournal(db_path=test_db_path)
    summary_engine = DailySummary(trade_journal=journal)

    today = date.today().isoformat()
    t_entry_1 = f"{today}T10:00:00"
    t_exit_1 = f"{today}T10:05:00"
    t_entry_2 = f"{today}T11:00:00"
    t_exit_2 = f"{today}T11:10:00"
    t_entry_3 = f"{today}T13:15:00"
    t_exit_3 = f"{today}T13:25:00"

    base_trade: Dict[str, Any] = {
        "entry_time": t_entry_1,
        "exit_time": t_exit_1,
        "symbol": "NIFTYTEST",
        "direction": "BUY",
        "strategy": "VWAP_PULLBACK",
        "brain": "STRUCTURAL",
        "regime": "TRENDING",
        "opportunity_score": 70.0,
        "ml_probability": 0.60,
        "trap_score": 10.0,
        "entry_price": 100.0,
        "exit_price": 110.0,
        "sl_price": 95.0,
        "target_price": 115.0,
        "pnl_points": 10.0,
        "pnl_rupees": 900.0,
        "exit_reason": "SELF_TEST",
        "hold_minutes": 5.0,
        "mfe": 12.0,
        "mae": -2.0,
        "lots": 1,
        "session_phase": "GOLDEN_AM",
        "narrative_label": "NEUTRAL",
        "mtf_alignment": 4.5,
        "vix_at_entry": 14.0,
        "fii_score": 0.0,
        "behavioral_checklist_complete": 1,
        "mode": "PAPER",
        "algo_id": "TEST_ALGO",
        "static_ip": "0.0.0.0",
        "costs_json": '{"stt": 10, "brokerage": 20, "gst": 3.6, "stamp": 2, "exchange": 1}',
    }

    trades_to_insert = [
        dict(base_trade, trade_id=1),
        dict(
            base_trade,
            trade_id=2,
            entry_time=t_entry_2,
            exit_time=t_exit_2,
            strategy="SR_REJECTION",
            regime="RANGE",
            session_phase="LUNCH_LULL",
            pnl_rupees=-500.0,
            pnl_points=-6.0,
            costs_json='{"stt": 8, "brokerage": 20, "gst": 3.6, "stamp": 2, "exchange": 1}',
        ),
        dict(
            base_trade,
            trade_id=3,
            entry_time=t_entry_3,
            exit_time=t_exit_3,
            strategy="STOP_HUNT_RECLAIM",
            regime="VOLATILE",
            session_phase="GOLDEN_PM",
            pnl_rupees=300.0,
            pnl_points=4.0,
            costs_json='{"stt": 6, "brokerage": 20, "gst": 3.6, "stamp": 2, "exchange": 1}',
        ),
    ]

    for td in trades_to_insert:
        ok = journal.log_trade(td)
        assert ok is True, "Failed to insert dummy trade during self-test"

    summary = summary_engine.generate(today)
    required_keys = {
        "total_trades",
        "winning_trades",
        "losing_trades",
        "win_rate",
        "gross_pnl_points",
        "gross_pnl_rupees",
        "net_pnl_points",
        "net_pnl_rupees",
        "total_costs",
        "profit_factor",
        "largest_win",
        "largest_loss",
        "average_win",
        "average_loss",
        "best_strategy",
        "worst_strategy",
        "performance_by_regime",
        "performance_by_session",
    }

    missing = [k for k in required_keys if k not in summary]
    assert not missing, f"Summary missing keys: {missing}"
    assert isinstance(summary["performance_by_regime"], dict)
    assert isinstance(summary["performance_by_session"], dict)
    assert summary["total_trades"] >= 3

    print("DailySummary self-test PASSED.")


if __name__ == "__main__":
    _run_self_test()