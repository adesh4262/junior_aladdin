# src/core/market_state.py

"""
Junior Aladdin - MarketState Dataclass (Institutional Grade)
============================================================

PURPOSE:
    Single shared state object that ALL engines read and write.
    This is the DATA CONTRACT of the entire system.

Strongest Version: Hardened with all necessary fields including VIX 
and diagnostics for full pipeline visibility.
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
    """

    @staticmethod
    def _allowed_values() -> Dict[str, set]:
        return {
            "feed_health": {"HEALTHY", "DELAYED", "STALE", "DOWN", "PRE_MARKET", "MARKET", "POST_MARKET"},
            "regime": {"TRENDING", "RANGE", "VOLATILE", "CHOP", "EVENT", "UNKNOWN"},
            "system_state": {"BOOT", "OBSERVE", "ACTIVE", "CAUTIOUS", "SAFE", "LOCKED", "SHUTDOWN"},
            "mode": {"ALERT", "OBSERVE", "PAPER", "LIVE"},
            "session_phase": {
                "PRE_MARKET", "OPENING_AUCTION", "OR_FORMATION", "INITIAL_BALANCE",
                "GOLDEN_AM", "GOLDEN_MORNING", "LUNCH_LULL", "GOLDEN_PM",
                "GOLDEN_AFTERNOON", "CLOSING_SESSION", "LAST_MINUTES",
            },
            "risk_state": {"NORMAL", "CAUTION", "REDUCED", "LOCKED"},
            "day_type": {"UNKNOWN", "TREND_DAY", "RANGE_DAY", "VOLATILE_DAY", "QUIET_DAY", "EXPIRY_DAY", "EVENT_DAY"},
        }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._logger = setup_logger("market_state")

        self.last_update: datetime = datetime.now(IST)
        self.last_updated_fields: Dict[str, datetime] = {}

        # ══════════════════════════════════════
        # LAYER 1 — Raw Data
        # ══════════════════════════════════════
        self.timestamp: Optional[datetime] = None
        self.spot: float = 0.0
        self.previous_close: float = 0.0
        self.vix: float = 0.0  # ADDED: Mandatory for accuracy

        self.candles: Dict[str, deque] = {
            "1min": deque(maxlen=400),
            "3min": deque(maxlen=140),
            "5min": deque(maxlen=80),
            "15min": deque(maxlen=30),
        }

        self.option_chain: Dict = {}
        self.market_depth: Dict = {"bids": [], "asks": []}

        self.feed_health: str = "DOWN"
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
            "levels_defended": [], "levels_broken": [], "failed_breakouts": [],
            "traps_detected": 0, "dominant_direction_morning": "",
            "momentum_decay_started": False, "largest_move_size": 0.0,
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
        # MTF Chart / Overlay State
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

    @staticmethod
    def _field_types() -> Dict[str, Tuple[type, ...]]:
        return {
            "last_update": (datetime,),
            "timestamp": (datetime, type(None)),
            "spot": (int, float),
            "previous_close": (int, float),
            "vix": (int, float), # ADDED
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
            "feed_health": (str,),
            "narrative_label": (str,),
            "regime": (str,),
            "session_phase": (str,),
            "day_type": (str,),
            "risk_state": (str,),
            "system_state": (str,),
            "mode": (str,),
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

    def update(self, **kwargs) -> None:
        if not kwargs: return
        validated: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        
        types = self._field_types()
        allowed = self._allowed_values()

        for k, v in kwargs.items():
            if k not in self.__dict__ or k.startswith("_"):
                errors[k] = "unknown_field"
                continue
            
            expected = types.get(k)
            if expected and not isinstance(v, expected):
                errors[k] = f"type_mismatch expected={expected} got={type(v)}"
                continue
                
            validated[k] = v

        if errors:
            self._logger.error("Atomic MarketState update rejected", extra={"errors": errors})
            return

        now = datetime.now(IST)
        with self._lock:
            for k, v in validated.items():
                setattr(self, k, v)
                self.last_updated_fields[k] = now
            self.last_update = now

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap: Dict[str, Any] = {}
            for key, value in self.__dict__.items():
                if key.startswith("_"): continue
                if isinstance(value, dict) and key == "candles":
                    snap[key] = {k: list(v) if isinstance(v, deque) else copy.deepcopy(v) for k, v in value.items()}
                else:
                    snap[key] = copy.deepcopy(value)
            return snap

    def reset_daily(self) -> None:
        now = datetime.now(IST)
        with self._lock:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_update = now

    def get_candles(self, timeframe: str) -> deque:
        with self._lock: return deque(self.candles.get(timeframe, deque()))

    def get_last_candle(self, timeframe: str) -> Optional[Dict]:
        with self._lock:
            dq = self.candles.get(timeframe)
            return copy.deepcopy(dq[-1]) if dq else None

    def is_market_active(self) -> bool:
        with self._lock: return self.system_state in ("ACTIVE", "CAUTIOUS")

    def get_field_count(self) -> int:
        with self._lock: return len([k for k in self.__dict__ if not k.startswith("_")])
