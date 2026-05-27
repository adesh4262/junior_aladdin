# src/core/market_state.py

"""
Junior Aladdin - MarketState Dataclass (Institutional Grade)
============================================================

PURPOSE:
    Single shared state object that ALL engines read and write.
    This is the DATA CONTRACT of the entire system.

INSTITUTIONAL HARDENING:
    - Atomic bulk update with validate-all-then-apply (no partial updates).
    - Declarative schema for field types and basic range checks.
    - Deep-copy snapshot for dashboard/journal (no live references).
    - Thread-safe getters returning copies of mutable objects.
    - last_update initialized in __init__ (never None).
    - last_updated_fields tracks per-field update timestamps.

PUBLIC API (unchanged signatures):
    update(**kwargs) -> None
    snapshot() -> Dict[str, Any]
    reset_daily() -> None
    get_candles(timeframe) -> deque
    get_last_candle(timeframe) -> Optional[Dict]
    get_feature(timeframe, feature_name, default=None)
    is_market_active() -> bool
    can_trade() -> bool
    get_field_count() -> int
"""

from __future__ import annotations

import copy
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import setup_logger

# ============================================
# IST Timezone
# ============================================
IST = timezone(timedelta(hours=5, minutes=30))


class MarketState:
    """
    Central shared state object for the entire trading system.

    Thread safety:
        All reads/writes are guarded by self._lock.
        Getters return copies for mutable objects.
    """

    # -------------------------------
    # Allowed enum values (broad)
    # -------------------------------
    @staticmethod
    def _allowed_values() -> Dict[str, set]:
        return {
            # FeedHealthMonitor can emit market-phase states in addition to pure
            # feed-health states; keep the state contract permissive enough to
            # accept those live values without rejecting the whole update.
            "feed_health": {"HEALTHY", "DELAYED", "STALE", "DOWN", "PRE_MARKET", "MARKET", "POST_MARKET"},
            "regime": {"TRENDING", "RANGE", "VOLATILE", "CHOP", "EVENT", "UNKNOWN"},
            "system_state": {"BOOT", "OBSERVE", "ACTIVE", "CAUTIOUS", "SAFE", "LOCKED", "SHUTDOWN"},
            "mode": {"ALERT", "OBSERVE", "PAPER", "LIVE"},
            "session_phase": {
                "PRE_MARKET",
                "OPENING_AUCTION",
                "OR_FORMATION",
                "INITIAL_BALANCE",
                "GOLDEN_AM",
                "GOLDEN_MORNING",
                "LUNCH_LULL",
                "GOLDEN_PM",
                "GOLDEN_AFTERNOON",
                "CLOSING_SESSION",
                "LAST_MINUTES",
            },
            "risk_state": {"NORMAL", "CAUTION", "REDUCED", "LOCKED"},
            "day_type": {"UNKNOWN", "TREND_DAY", "RANGE_DAY", "VOLATILE_DAY", "QUIET_DAY", "EXPIRY_DAY", "EVENT_DAY"},
        }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._logger = setup_logger("market_state")

        # last_update must never be None (I4/C5)
        self.last_update: datetime = datetime.now(IST)
        # per-field staleness tracking (I6)
        self.last_updated_fields: Dict[str, datetime] = {}

        # ══════════════════════════════════════
        # LAYER 1 — Raw Data
        # ══════════════════════════════════════
        self.timestamp: Optional[datetime] = None
        self.spot: float = 0.0
        self.previous_close: float = 0.0

        self.candles: Dict[str, deque] = {
            "1min": deque(maxlen=400),
            "3min": deque(maxlen=140),
            "5min": deque(maxlen=80),
            "15min": deque(maxlen=30),
        }

        self.option_chain: Dict = {}
        self.market_depth: Dict = {"bids": [], "asks": []}

        self.feed_health: str = "DOWN"  # HEALTHY / DELAYED / STALE / DOWN
        self.feed_lag_ms: float = 0.0
        self.ticks_per_second: float = 0.0
        self.data_quality_score: float = 0.0
        self.using_fallback: bool = False

        # ══════════════════════════════════════
        # LAYER 2 — Features
        # ══════════════════════════════════════
        self.features: Dict[str, Dict] = {"1min": {}, "3min": {}, "5min": {}, "15min": {}}
        self.options_features: Dict = {}
        self.microstructure: Dict = {}
        self.key_levels: Dict = {}
        self.smart_money: Dict = {}
        self.candle_patterns: List[Dict] = []

        # ══════════════════════════════════════
        # LAYER 3 — Context
        # ══════════════════════════════════════
        self.narrative_score: float = 0.0
        self.narrative_label: str = "NEUTRAL"
        self.narrative_fit_factors: Dict = {}

        self.regime: str = "UNKNOWN"
        self.regime_confidence: float = 0.0
        self.regime_transition_prob: float = 0.0

        self.session_phase: str = "PRE_MARKET"
        self.session_size_multiplier: float = 0.0
        self.day_type: str = "UNKNOWN"

        self.or_high: float = 0.0
        self.or_low: float = 0.0
        self.ib_high: float = 0.0
        self.ib_low: float = 0.0
        self.ib_width: float = 0.0

        # ══════════════════════════════════════
        # LAYER 4 — Market DNA
        # ══════════════════════════════════════
        self.day_personality: Dict = {}
        self.historical_match_score: float = 0.0
        self.session_memory: Dict = {
            "levels_defended": [],
            "levels_broken": [],
            "failed_breakouts": [],
            "traps_detected": 0,
            "dominant_direction_morning": "",
            "momentum_decay_started": False,
            "largest_move_size": 0.0,
            "volume_profile_shift": "STABLE",
        }

        # ══════════════════════════════════════
        # LAYER 5 — Brain State
        # ══════════════════════════════════════
        self.active_brains: List[str] = []
        self.brain_confidence: Dict[str, float] = {}

        # ══════════════════════════════════════
        # Opportunities Pipeline
        # ══════════════════════════════════════
        self.raw_opportunities: List[Dict] = []
        self.trapped_opportunities: List[Dict] = []
        self.scored_opportunities: List[Dict] = []
        self.ml_filtered: List[Dict] = []
        self.behavioral_filtered: List[Dict] = []
        self.approved_opportunities: List[Dict] = []

        # ══════════════════════════════════════
        # Risk State
        # ══════════════════════════════════════
        self.capital: float = 50000.0
        self.daily_pnl: float = 0.0
        self.drawdown_pct: float = 0.0
        self.trades_today: int = 0
        self.consecutive_losses: int = 0
        self.tilt_score: float = 0.0
        self.risk_state: str = "NORMAL"

        # ══════════════════════════════════════
        # System State
        # ══════════════════════════════════════
        self.system_state: str = "BOOT"
        self.mode: str = "ALERT"
        self.engine_health: Dict[str, Any] = {}
        self.kill_switch_state: str = "UNKNOWN"
        self.snapshot_age_seconds: float = 0.0

        # ══════════════════════════════════════
        # MTF Chart / Overlay State (for dashboard projection)
        # ══════════════════════════════════════
        self.mtf_candles: Dict = {}
        self.candles_by_tf: Dict = {}
        self.vwap_bands: Dict = {}
        self.or_levels: Dict = {}
        self.ib_levels: Dict = {}
        self.active_timeframe: str = "5m"
        self.timeframe: str = "5m"

        # ══════════════════════════════════════
        # Open Positions
        # ══════════════════════════════════════
        self.open_positions: List[Dict] = []

        self._logger.info("MarketState initialized", last_update=self.last_update.isoformat())

    # ============================================
    # Validation Schema (I2)
    # ============================================
    @staticmethod
    def _field_types() -> Dict[str, Tuple[type, ...]]:
        """
        Declarative schema for type validation.
        - Keep broad to avoid breaking evolution
        - Still blocks obvious corruption (e.g., capital='50000', open_positions=None)
        """
        return {
            # timestamps
            "last_update": (datetime,),
            "timestamp": (datetime, type(None)),
            # numerics
            "spot": (int, float),
            "previous_close": (int, float),
            "feed_lag_ms": (int, float),
            "ticks_per_second": (int, float),
            "data_quality_score": (int, float),
            "narrative_score": (int, float),
            "regime_confidence": (int, float),
            "regime_transition_prob": (int, float),
            "session_size_multiplier": (int, float),
            "or_high": (int, float),
            "or_low": (int, float),
            "ib_high": (int, float),
            "ib_low": (int, float),
            "ib_width": (int, float),
            "historical_match_score": (int, float),
            "capital": (int, float),
            "daily_pnl": (int, float),
            "drawdown_pct": (int, float),
            "trades_today": (int,),
            "consecutive_losses": (int,),
            "tilt_score": (int, float),
            # enums/strings
            "feed_health": (str,),
            "narrative_label": (str,),
            "regime": (str,),
            "session_phase": (str,),
            "day_type": (str,),
            "risk_state": (str,),
            "system_state": (str,),
            "mode": (str,),
            # containers
            "candles": (dict,),
            "option_chain": (dict,),
            "market_depth": (dict,),
            "features": (dict,),
            "options_features": (dict,),
            "microstructure": (dict,),
            "key_levels": (dict,),
            "smart_money": (dict,),
            "candle_patterns": (list,),
            "narrative_fit_factors": (dict,),
            "day_personality": (dict,),
            "session_memory": (dict,),
            "active_brains": (list,),
            "brain_confidence": (dict,),
            "raw_opportunities": (list,),
            "trapped_opportunities": (list,),
            "scored_opportunities": (list,),
            "ml_filtered": (list,),
            "behavioral_filtered": (list,),
            "approved_opportunities": (list,),
            "engine_health": (dict,),
            "open_positions": (list,),
            # MTF Chart / Overlay fields
            "mtf_candles": (dict,),
            "candles_by_tf": (dict,),
            "vwap_bands": (dict,),
            "or_levels": (dict,),
            "ib_levels": (dict,),
            "active_timeframe": (str,),
            "timeframe": (str,),
            "last_updated_fields": (dict,),
            "using_fallback": (bool,),
            "kill_switch_state": (str,),
            "snapshot_age_seconds": (int, float),
        }

    @staticmethod
    def _numeric_ranges() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """
        Simple range checks for numeric fields.
        """
        return {
            "spot": (0.0, None),
            "previous_close": (0.0, None),
            "feed_lag_ms": (0.0, None),
            "ticks_per_second": (0.0, None),
            "data_quality_score": (0.0, 100.0),
            "capital": (0.0, None),
            "drawdown_pct": (0.0, 1.0),
            "session_size_multiplier": (0.0, 2.0),
            "tilt_score": (0.0, 100.0),
            "regime_confidence": (0.0, 1.0),
            "regime_transition_prob": (0.0, 1.0),
        }

    def _normalize_enum(self, key: str, value: str) -> str:
        v = value.strip().upper()
        if key in ("session_phase",):
            v = v.replace(" ", "_").replace("-", "_")
        return v

    def _validate_field(self, key: str, value: Any) -> Tuple[bool, Any, str]:
        """
        Returns (ok, normalized_value, error_msg).
        """
        types = self._field_types()
        allowed = self._allowed_values()
        ranges = self._numeric_ranges()

        if key not in self.__dict__ or key.startswith("_"):
            return False, value, "unknown_field"

        expected = types.get(key)
        if expected is not None and not isinstance(value, expected):
            return False, value, f"type_mismatch expected={expected} got={type(value)}"

        # enum normalization/validation
        if key in allowed:
            if not isinstance(value, str):
                return False, value, "enum_not_string"
            v = self._normalize_enum(key, value)
            if v not in allowed[key]:
                return False, value, f"enum_invalid value={v}"
            return True, v, ""

        # numeric range validation
        if key in ranges and isinstance(value, (int, float)):
            v = float(value)
            lo, hi = ranges[key]
            if lo is not None and v < lo:
                return False, value, f"range_low value={v} lo={lo}"
            if hi is not None and v > hi:
                return False, value, f"range_high value={v} hi={hi}"
            # normalize ints where expected
            if key in ("trades_today", "consecutive_losses"):
                return True, int(v), ""
            return True, v, ""

        return True, value, ""

    # ============================================
    # Atomic Bulk Update (I1)
    # ============================================
    def update(self, **kwargs) -> None:
        """
        Atomic bulk update:
          - validate all fields first
          - if ANY invalid => reject ENTIRE update
          - apply all changes together under lock
          - update last_update and last_updated_fields
        """
        if not kwargs:
            return

        # validate-all first (no mutation)
        validated: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for k, v in kwargs.items():
            ok, norm, err = self._validate_field(k, v)
            if not ok:
                errors[k] = err
            else:
                validated[k] = norm

        if errors:
            self._logger.error(
                "Atomic MarketState update rejected (one or more fields invalid)",
                extra={"errors": errors, "keys": list(kwargs.keys())[:50]},
            )
            return

        now = datetime.now(IST)
        with self._lock:
            for k, v in validated.items():
                setattr(self, k, v)
                self.last_updated_fields[k] = now
            self.last_update = now

    # ============================================
    # Snapshot (I3)
    # ============================================
    def snapshot(self) -> Dict[str, Any]:
        """
        Deep copy snapshot (fully independent).
        Deques are converted to lists for serialization.
        """
        with self._lock:
            snap: Dict[str, Any] = {}
            for key, value in self.__dict__.items():
                if key.startswith("_"):
                    continue

                if isinstance(value, dict) and key == "candles":
                    converted: Dict[str, Any] = {}
                    for k, v in value.items():
                        if isinstance(v, deque):
                            converted[k] = list(v)
                        else:
                            converted[k] = copy.deepcopy(v)
                    snap[key] = converted
                elif isinstance(value, deque):
                    snap[key] = list(value)
                else:
                    snap[key] = copy.deepcopy(value)
            return snap

    # ============================================
    # Daily Reset (C4/I7)
    # ============================================
    def reset_daily(self) -> None:
        """
        Reset daily counters and pipeline fields.
        Updates last_update at the end (after all changes).
        """
        now = datetime.now(IST)
        with self._lock:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.tilt_score = 0.0
            self.risk_state = "NORMAL"
            self.drawdown_pct = 0.0

            self.raw_opportunities = []
            self.trapped_opportunities = []
            self.scored_opportunities = []
            self.ml_filtered = []
            self.behavioral_filtered = []
            self.approved_opportunities = []

            self.session_memory = {
                "levels_defended": [],
                "levels_broken": [],
                "failed_breakouts": [],
                "traps_detected": 0,
                "dominant_direction_morning": "",
                "momentum_decay_started": False,
                "largest_move_size": 0.0,
                "volume_profile_shift": "STABLE",
            }

            self.or_high = 0.0
            self.or_low = 0.0
            self.ib_high = 0.0
            self.ib_low = 0.0
            self.ib_width = 0.0

            self.candle_patterns = []

            self.day_type = "UNKNOWN"
            self.day_personality = {}

            self.open_positions = []

            self.features = {"1min": {}, "3min": {}, "5min": {}, "15min": {}}
            self.options_features = {}
            self.microstructure = {}
            self.key_levels = {}
            self.smart_money = {}

            self.system_state = "BOOT"
            self.session_phase = "PRE_MARKET"
            self.session_size_multiplier = 0.0

            # staleness tracking update
            for k in (
                "daily_pnl",
                "trades_today",
                "consecutive_losses",
                "tilt_score",
                "risk_state",
                "drawdown_pct",
                "raw_opportunities",
                "trapped_opportunities",
                "scored_opportunities",
                "ml_filtered",
                "behavioral_filtered",
                "approved_opportunities",
                "session_memory",
                "or_high",
                "or_low",
                "ib_high",
                "ib_low",
                "ib_width",
                "candle_patterns",
                "day_type",
                "day_personality",
                "open_positions",
                "features",
                "options_features",
                "microstructure",
                "key_levels",
                "smart_money",
                "system_state",
                "session_phase",
                "session_size_multiplier",
            ):
                self.last_updated_fields[k] = now

            self.last_update = now

    # ============================================
    # Thread-safe Getters (I5)
    # ============================================
    def get_candles(self, timeframe: str) -> deque:
        """
        Returns a COPY of the candle deque for the given timeframe to prevent external mutation.
        """
        with self._lock:
            dq = self.candles.get(timeframe)
            if dq is None:
                return deque()
            return deque(dq)  # copy

    def get_last_candle(self, timeframe: str) -> Optional[Dict]:
        with self._lock:
            dq = self.candles.get(timeframe)
            if dq and len(dq) > 0:
                return copy.deepcopy(dq[-1])
            return None

    def get_feature(self, timeframe: str, feature_name: str, default=None):
        with self._lock:
            tf_features = self.features.get(timeframe, {})
            if not isinstance(tf_features, dict):
                return default
            return copy.deepcopy(tf_features.get(feature_name, default))

    def is_market_active(self) -> bool:
        with self._lock:
            return self.system_state in ("ACTIVE", "CAUTIOUS")

    def can_trade(self) -> bool:
        with self._lock:
            return (
                self.system_state == "ACTIVE"
                and self.feed_health in ("HEALTHY", "DELAYED")
                and float(self.session_size_multiplier) > 0.0
                and self.risk_state != "LOCKED"
            )

    def get_field_count(self) -> int:
        with self._lock:
            return len([k for k in self.__dict__ if not k.startswith("_")])


# ============================================
# Module Self-Test
# ============================================
if __name__ == "__main__":
    import concurrent.futures

    print("=" * 60)
    print("  JUNIOR ALADDIN — MarketState Test (Institutional)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    print("  [Test 1] Create MarketState...")
    try:
        state = MarketState()
        print(f"    ✅ Created with {state.get_field_count()} fields")
        passed += 1
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        failed += 1
        raise

    print("\n  [Test 2] Atomic update rejection...")
    try:
        # invalid update should reject entire update
        state.update(spot=25000.0, capital="50000")  # invalid capital type
        if state.spot != 25000.0:
            print("    ✅ Atomic rejection worked (spot not partially updated)")
            passed += 1
        else:
            print("    ❌ Partial update happened (should not)")
            failed += 1
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        failed += 1

    print("\n  [Test 3] Valid update...")
    try:
        state.update(spot=24500.0, feed_health="HEALTHY", system_state="ACTIVE", session_size_multiplier=1.0)
        if state.spot == 24500.0 and state.feed_health == "HEALTHY" and state.system_state == "ACTIVE":
            print("    ✅ Valid update applied")
            passed += 1
        else:
            print("    ❌ Valid update not applied correctly")
            failed += 1
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        failed += 1

    print("\n  [Test 4] Snapshot deep copy independence...")
    try:
        snap = state.snapshot()
        assert isinstance(snap, dict)
        snap["spot"] = 1.0
        if state.spot == 24500.0:
            print("    ✅ Snapshot is independent")
            passed += 1
        else:
            print("    ❌ Snapshot mutated live state")
            failed += 1
    except Exception as e:
        print(f"    ❌ Exception: {e}")
        failed += 1

    print("\n  [Test 5] Thread safety stress...")
    errors = []

    def writer(val: float):
        for _ in range(200):
            state.update(spot=val)

    def reader():
        for _ in range(200):
            _ = state.spot
            _ = state.can_trade()
            _ = state.is_market_active()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [
            ex.submit(writer, 24600.0),
            ex.submit(writer, 24700.0),
            ex.submit(writer, 24800.0),
            ex.submit(reader),
            ex.submit(reader),
            ex.submit(reader),
        ]
        for f in concurrent.futures.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errors.append(str(e))

    if not errors:
        print("    ✅ Concurrent read/write completed without errors")
        passed += 1
    else:
        print(f"    ❌ Errors: {errors}")
        failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)