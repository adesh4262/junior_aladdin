"""
TEMPORARY VALIDATION TOOL — FAKE SNAPSHOT PUBLISHER
====================================================

PURPOSE:
    This is a SAFE TEMPORARY fake backend that produces HOT/WARM/COLD binary
    frames over TCP so the dashboard can validate its dataflow pipeline.

    It generates realistic-looking but entirely artificial MarketState snapshots
    so panels render LIVE values instead of DEGRADED/empty states.

ARCHITECTURE:
    - Standalone TCP server (listens on 127.0.0.1:18765)
    - Produces HOT frames every 200ms, WARM every 1000ms, COLD every 5000ms
    - Uses the REAL binary_frame.pack_frame() — same contract as production
    - Simulates realistic market data with slow drift/random walk

ISOLATION RULES:
    - ZERO modifications to existing dashboard files
    - ZERO modifications to src/ backend files
    - Fully contained in this single file
    - File clearly marked as TEMPORARY in docstring

CLEANUP (when real backend IPC is ready):
    1. Stop running `tools/fake_snapshot_publisher.py`
    2. Delete this file
    3. Delete `tools/run_fake_validation.py`
    4. Done — no other files touched

USAGE:
    # Terminal 1 — Start fake publisher
    python -m tools.fake_snapshot_publisher

    # Terminal 2 — Start dashboard (connects automatically)
    python -m dashboard.main
"""

from __future__ import annotations

import itertools
import math
import random
import socket
import struct
import sys
import threading
import time
from typing import Any, Dict

# Add project root to path
sys.path.insert(0, ".")

from dashboard.core.binary_frame import pack_frame, KIND_HOT, KIND_WARM, KIND_COLD

# ─── Configuration ───────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 18765
HOT_INTERVAL_S = 0.200   # 200ms
WARM_INTERVAL_S = 1.0    # 1s
COLD_INTERVAL_S = 5.0    # 5s

_LOG = print  # plain print to avoid logger dependency

# ─── State Generator ─────────────────────────────────────────────────────────

class FakeMarketGenerator:
    """Generates drifting fake market state values."""

    def __init__(self) -> None:
        self._spot = 24650.0
        self._prev_close = 24600.0
        self._seq_hot = 0
        self._seq_warm = 0
        self._seq_cold = 0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._tick_count = 0
        self._poc = 24650.0
        self._vah = 24700.0
        self._val = 24600.0
        self._cvd = 0.0
        self._imbalance = 0.0
        self._cycle = 0

        # Heavyweights for Component Guard
        self._heavyweights = [
            {"symbol": "RELIANCE", "price": 3125.0, "change_pct": 0.0, "contribution_ratio": 0.12, "veto_status": "OK"},
            {"symbol": "HDFCBANK", "price": 1780.0, "change_pct": 0.0, "contribution_ratio": 0.10, "veto_status": "OK"},
            {"symbol": "ICICIBANK", "price": 1285.0, "change_pct": 0.0, "contribution_ratio": 0.08, "veto_status": "OK"},
            {"symbol": "INFY",     "price": 1645.0, "change_pct": 0.0, "contribution_ratio": 0.07, "veto_status": "OK"},
            {"symbol": "TCS",      "price": 3920.0, "change_pct": 0.0, "contribution_ratio": 0.06, "veto_status": "OK"},
        ]

    def _drift(self, value: float, amplitude: float = 2.0, mean_revert: float = 0.01) -> float:
        """Random walk with mild mean reversion."""
        noise = random.gauss(0, amplitude)
        reversion = (self._spot - value) * mean_revert if value != self._spot else 0.0
        return value + noise + reversion

    def _make_volume_profile(self) -> Dict[str, Any]:
        """Generate fake volume profile levels around POC."""
        poc_int = int(round(self._poc / 5) * 5)
        levels: Dict[str, float] = {}
        for offset in range(-40, 41, 5):
            price = poc_int + offset
            dist = abs(price - poc_int)
            volume = max(100, 10000 * math.exp(-0.5 * (dist / 15) ** 2) * random.uniform(0.8, 1.2))
            levels[str(price)] = round(volume, 2)
        return {
            "profile": levels,
            "poc": self._poc,
            "poc_volume": max(levels.values()),
            "vah": self._vah,
            "val": self._val,
            "bucket_size": 5,
            "hvn_count": 3,
            "lvn_count": 2,
            "hvn_levels": [self._poc - 5, self._poc, self._poc + 5],
            "lvn_levels": [self._poc - 20, self._poc + 20],
        }

    def hot_snapshot(self) -> Dict[str, Any]:
        """Generate HOT-tier snapshot (200ms cadence)."""
        self._seq_hot += 1
        self._tick_count += 1
        self._cycle += 1

        # Drift spot price with occasional impulses
        if self._cycle % 50 == 0:
            self._spot += random.choice([-10, 10])
        else:
            self._spot += random.gauss(0, 1.5)

        spot_change = self._spot - self._prev_close
        trend = "BULLISH" if spot_change > 0 else "BEARISH"

        # Simulate changing feeds/state
        feed_health = random.choices(
            ["HEALTHY", "HEALTHY", "HEALTHY", "HEALTHY", "DELAYED"],
            weights=[80, 10, 5, 3, 2],
        )[0]

        system_state = random.choices(
            ["ACTIVE", "ACTIVE", "ACTIVE", "CAUTIOUS", "SAFE"],
            weights=[70, 15, 10, 4, 1],
        )[0]

        data_quality = max(0, min(100, 85 + random.gauss(0, 8)))

        self._cvd += random.gauss(0, 500)

        return {
            "system_state": system_state,
            "mode": "PAPER",
            "feed_health": feed_health,
            "data_quality_score": round(data_quality, 1),
            "ticks_per_second": round(15 + random.gauss(0, 3), 1),
            "feed_lag_ms": round(max(0, 20 + random.gauss(0, 15)), 1),
            "using_fallback": feed_health == "DELAYED",
            "last_update_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "spot": round(self._spot, 2),
            "previous_close": self._prev_close,
            "capital": 50000.0,
            "daily_pnl": round(spot_change * 65, 2),
            "drawdown_pct": round(max(0, -spot_change * 0.001), 4),
            "trades_today": min(5, self._trades_today),
            "consecutive_losses": self._consecutive_losses,
            "tilt_score": round(random.uniform(0, 30), 1),
            "risk_state": "NORMAL",
            "session_phase": random.choice(["GOLDEN_AM", "LUNCH_LULL", "GOLDEN_PM"]),
            "day_type": random.choice(["TREND_DAY", "RANGE_DAY", "VOLATILE_DAY"]),
            "regime": random.choice(["TRENDING", "RANGE", "VOLATILE"]),
            "narrative_label": random.choice(["MILD_BULLISH", "STRONG_BULLISH", "NEUTRAL"]),
            "active_brains": random.choice([["structural"], ["tactical"], ["structural", "institutional"], ["adaptive"]]),
            "regime_transition_prob": round(random.uniform(0, 0.4), 3),
            "component_guard_heavyweights": self._heavyweights,
            "heavyweights": self._heavyweights,
            "engine_health": {
                "data": {"alive": True, "status": "ok"},
                "captain": {"alive": True, "status": "ok"},
                "features": {"alive": True, "status": "ok"},
                "risk": {"alive": True, "status": "ok"},
                "execution": {"alive": True, "status": "ok"},
            },
        }

    def warm_snapshot(self) -> Dict[str, Any]:
        """Generate WARM-tier snapshot (1s cadence)."""
        self._seq_warm += 1

        for hw in self._heavyweights:
            hw["price"] = round(hw["price"] + random.gauss(0, 5), 2)
            hw["change_pct"] = round(((hw["price"] - 3000) / 3000) * 100, 2) if "RELIANCE" in hw["symbol"] else \
                               round(random.uniform(-1.5, 1.5), 2)

        return {
            "narrative_score": round(50 + random.gauss(0, 15), 1),
            "narrative_label": random.choice(["MILD_BULLISH", "STRONG_BULLISH", "NEUTRAL", "MILD_BEARISH"]),
            "narrative_fit_factors": {
                "long_fit": round(random.uniform(0.4, 1.2), 2),
                "short_fit": round(random.uniform(0.1, 0.8), 2),
            },
            "regime": random.choice(["TRENDING", "RANGE", "VOLATILE"]),
            "regime_confidence": round(random.uniform(0.3, 0.95), 3),
            "regime_transition_prob": round(random.uniform(0, 0.5), 3),
            "session_phase": "GOLDEN_AM",
            "day_type": random.choice(["TREND_DAY", "RANGE_DAY"]),
            "day_personality": {"day_type": "TREND_DAY"},
            "historical_match_score": round(random.uniform(0.4, 0.9), 3),
            "session_memory": {
                "levels_defended": [round(self._spot - 30, 2)],
                "levels_broken": [],
                "failed_breakouts": random.choices([0, 1, 2], weights=[60, 30, 10])[0],
                "traps_detected": random.choices([0, 1], weights=[70, 30])[0],
                "dominant_direction_morning": random.choice(["UP", "DOWN"]),
                "momentum_decay_started": False,
                "largest_move_size": round(random.uniform(15, 40), 1),
                "volume_profile_shift": "STABLE",
            },
            "or_high": round(self._spot + 60, 2),
            "or_low": round(self._spot - 40, 2),
            "ib_high": round(self._spot + 90, 2),
            "ib_low": round(self._spot - 70, 2),
            "ib_width": 160,
            "session_size_multiplier": 1.0,
            "active_brains": ["structural"],
            "brain_confidence": {"structural": round(random.uniform(0.6, 0.95), 3)},
            "features": {
                "1m": {"rsi": round(55 + random.gauss(0, 5), 1), "atr": round(18 + random.gauss(0, 2), 1), "ema_9": round(self._spot - 10, 2), "ema_21": round(self._spot - 25, 2)},
                "3m": {"rsi": round(52 + random.gauss(0, 4), 1), "atr": round(22 + random.gauss(0, 2), 1)},
                "5m": {"rsi": round(50 + random.gauss(0, 3), 1), "atr": round(25 + random.gauss(0, 2), 1)},
                "15m": {"rsi": round(48 + random.gauss(0, 3), 1), "atr": round(30 + random.gauss(0, 3), 1)},
            },
            "options_summary": {
                "pcr_oi": round(random.uniform(0.6, 1.4), 3),
                "atm_iv": round(random.uniform(0.10, 0.18), 3),
                "max_pain": round(self._spot - random.randint(0, 50), 2),
                "highest_ce_oi_strike": round(self._spot + 200, 2),
                "highest_pe_oi_strike": round(self._spot - 200, 2),
            },
            "options_features": {
                "pcr_oi": round(random.uniform(0.6, 1.4), 3),
                "atm_iv": round(random.uniform(0.10, 0.18), 3),
            },
            "smart_money_summary": {
                "sm_direction_score": round(random.uniform(-20, 40), 1),
                "total_fvgs": random.randint(0, 5),
                "bullish_fvgs": random.randint(0, 3),
                "bearish_fvgs": random.randint(0, 3),
            },
            "smart_money": {
                "fvg_zones": [],
                "order_blocks": [],
                "liquidity_pools": [{"price": round(self._spot + 100, 2), "type": "sell"}],
                "bos_points": [],
            },
            "volume_profile": self._make_volume_profile(),
            "microstructure": {
                "cumulative_volume_delta": round(self._cvd, 0),
                "bid_ask_imbalance": round(random.uniform(-0.5, 0.5), 3),
                "spread_bps": round(random.uniform(0.5, 3.0), 2),
                "trade_intensity": round(random.uniform(0.3, 1.5), 2),
            },
            "mtf_candles": self._fake_candles(),
            "vwap_bands": {"vwap": [round(self._spot - 15 + random.gauss(0, 3), 2) for _ in range(20)]},
            "active_timeframe": "5m",
            "timeframe": "5m",
        }

    def cold_snapshot(self) -> Dict[str, Any]:
        """Generate COLD-tier snapshot (5s cadence)."""
        self._seq_cold += 1

        self._trades_today = random.randint(0, 5)
        self._consecutive_losses = random.choices([0, 1, 2], weights=[60, 30, 10])[0]

        return {
            "raw_opportunities": [{"id": i, "strategy": s, "score": round(random.uniform(50, 90), 1)} for i, s in zip(range(random.randint(3, 8)), ["vwap_pullback", "trend_cont", "sr_rej", "fvg_retest", "stop_hunt"])],
            "trapped_opportunities": [{"id": i, "reason": r} for i, r in zip(range(random.randint(1, 3)), ["volume_low", "oi_contradiction"])],
            "scored_opportunities": [{"id": i, "score": round(random.uniform(55, 85), 1)} for i in range(random.randint(2, 5))],
            "ml_filtered": [{"id": i, "probability": round(random.uniform(0.1, 0.45), 3)} for i in range(random.randint(0, 2))],
            "behavioral_filtered": [{"id": i, "reason": r} for i, r in zip(range(random.randint(0, 1)), ["tilt"])],
            "approved_opportunities": [{"id": i, "direction": d, "score": round(random.uniform(65, 95), 1)} for i, d in zip(range(random.randint(1, 3)), ["BUY", "BUY", "SELL"])],
            "open_positions": [
                {"symbol": "NIFTY", "direction": "BUY", "qty": 65, "entry_price": round(self._spot - 20, 2), "current_price": round(self._spot, 2), "pnl": round(random.uniform(-200, 500), 2)}
            ] if self._trades_today > 0 else [],
            "engine_health": {
                "data": {"alive": True, "status": "ok", "last_error": None, "uptime_hours": 2.5},
                "captain": {"alive": True, "status": "ok", "last_error": None},
                "features": {"alive": True, "status": "ok", "last_error": None},
                "risk": {"alive": True, "status": "ok"},
                "execution": {"alive": True, "status": "ok"},
                "ml": {"alive": True, "status": "ok", "model_loaded": True},
            },
            "feed_health": "HEALTHY",
            "data_quality_score": round(85 + random.gauss(0, 8), 1),
            "ticks_per_second": round(15 + random.gauss(0, 3), 1),
            "feed_lag_ms": round(max(0, 20 + random.gauss(0, 15)), 1),
            "using_fallback": False,
            "kill_switch_state": "SAFE",
            "snapshot_age_seconds": 0.0,
            "last_update_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }

    @staticmethod
    def _fake_candles() -> Dict[str, list]:
        """Generate fake MTF candles for chart validation."""
        now = time.time()
        candles: Dict[str, list] = {}
        for tf, count, interval_s in [("1m", 60, 60), ("3m", 40, 180), ("5m", 30, 300), ("15m", 15, 900)]:
            rows: list[Dict] = []
            base = 24650.0
            for i in range(count):
                ts = now - (count - i) * interval_s
                o = base + random.gauss(0, 5)
                h = o + abs(random.gauss(0, 8))
                l = o - abs(random.gauss(0, 8))
                c = random.uniform(l, h)
                rows.append({
                    "timestamp": ts,
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(l, 2),
                    "close": round(c, 2),
                    "volume": int(random.uniform(1000, 50000)),
                })
                base = c
            candles[tf] = rows
        return candles


    def next_hot_payload(self) -> bytes:
        return pack_frame(self.hot_snapshot(), kind=KIND_HOT, seq=self._seq_hot)

    def next_warm_payload(self) -> bytes:
        return pack_frame(self.warm_snapshot(), kind=KIND_WARM, seq=self._seq_warm)

    def next_cold_payload(self) -> bytes:
        return pack_frame(self.cold_snapshot(), kind=KIND_COLD, seq=self._seq_cold)


# ─── TCP Server ──────────────────────────────────────────────────────────────

class FakePublisherServer:
    """TCP server that pushes HOT/WARM/COLD frames to connected clients."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._generator = FakeMarketGenerator()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, port))
        self._server.listen(5)
        self._server.settimeout(1.0)  # allow clean shutdown
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True
        _LOG(f"[FAKE] Publisher listening on {self.host}:{self.port}")
        _LOG("[FAKE] Press Ctrl+C to stop")

        # Accept connections in main thread
        accept_thread = threading.Thread(target=self._accept_loop, daemon=True, name="accept")
        accept_thread.start()

        # Push frames in producer threads
        hot_thread = threading.Thread(target=self._push_loop, args=(KIND_HOT, HOT_INTERVAL_S, self._generator.next_hot_payload), daemon=True, name="hot_push")
        warm_thread = threading.Thread(target=self._push_loop, args=(KIND_WARM, WARM_INTERVAL_S, self._generator.next_warm_payload), daemon=True, name="warm_push")
        cold_thread = threading.Thread(target=self._push_loop, args=(KIND_COLD, COLD_INTERVAL_S, self._generator.next_cold_payload), daemon=True, name="cold_push")

        hot_thread.start()
        warm_thread.start()
        cold_thread.start()

        hot_thread.join()
        warm_thread.join()
        cold_thread.join()
        accept_thread.join()

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except Exception:
                    pass
            self._clients.clear()
        try:
            self._server.close()
        except Exception:
            pass

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, addr = self._server.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _LOG(f"[FAKE] Client connected: {addr}")
                with self._lock:
                    self._clients.append(client)
            except socket.timeout:
                continue
            except OSError:
                break

    def _push_loop(self, kind: int, interval_s: float, payload_fn) -> None:
        kind_name = {KIND_HOT: "HOT", KIND_WARM: "WARM", KIND_COLD: "COLD"}.get(kind, "?")
        while self._running:
            try:
                frame = payload_fn()
                self._broadcast(frame)
            except Exception as exc:
                _LOG(f"[FAKE] {kind_name} push error: {exc}")
            time.sleep(interval_s)

    def _broadcast(self, data: bytes) -> None:
        with self._lock:
            dead: list[socket.socket] = []
            for c in self._clients:
                try:
                    # Length-prefix for SnapshotStreamClient
                    prefix = struct.pack("!I", len(data))
                    c.sendall(prefix + data)
                except Exception:
                    dead.append(c)
            for c in dead:
                try:
                    c.close()
                except Exception:
                    pass
                self._clients.remove(c)


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    _LOG("=" * 60)
    _LOG("  FAKE SNAPSHOT PUBLISHER (TEMPORARY VALIDATION TOOL)")
    _LOG("=" * 60)
    _LOG("")
    _LOG("  This is temporary validation infrastructure.")
    _LOG("  When real backend IPC is ready, delete this file.")
    _LOG("")
    _LOG(f"  Publishing on tcp://{HOST}:{PORT}")
    _LOG(f"  HOT  frames every {HOT_INTERVAL_S*1000:.0f}ms")
    _LOG(f"  WARM frames every {WARM_INTERVAL_S*1000:.0f}ms")
    _LOG(f"  COLD frames every {COLD_INTERVAL_S*1000:.0f}ms")
    _LOG("")
    _LOG("  Start dashboard in another terminal:")
    _LOG("    python -m dashboard.main")
    _LOG("")
    _LOG("=" * 60)

    server = FakePublisherServer(HOST, PORT)
    try:
        server.start()
    except KeyboardInterrupt:
        _LOG("\n[FAKE] Shutting down...")
    finally:
        server.stop()
        _LOG("[FAKE] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())