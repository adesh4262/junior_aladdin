"""
Junior Aladdin - Event Overlay
==============================
PURPOSE:
Apply event-specific restrictions and overlay logic when major events are near.

This file exists because the roadmap expects:
    src/brains/event_overlay.py

RESPONSIBILITIES:
- detect whether event overlay should be active
- reduce size
- restrict normal directional trading
- allow/route pre-event straddle opportunities
- provide consistent event-mode output for Captain / brain pipeline

THIS FILE DOES NOT:
- place orders
- replace main brain selection
- score trades

CONNECTS TO:
- Narrative Engine
- Time Context
- Fundamental Features
- Pre-Event Straddle Strategy
- Captain / Brain Engine
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger
from src.strategies.pre_event_straddle import PreEventStraddleStrategy

_logger = setup_logger("event_overlay")


@dataclass
class EventOverlayDecision:
    active: bool
    block_directional_trades: bool
    size_factor: float
    allow_straddle: bool
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    straddle_opportunities: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "block_directional_trades": self.block_directional_trades,
            "size_factor": self.size_factor,
            "allow_straddle": self.allow_straddle,
            "reason": self.reason,
            "notes": self.notes,
            "straddle_opportunities": self.straddle_opportunities,
        }


class EventOverlay:
    """
    Event overlay controller.
    """

    def __init__(self):
        self._logger = _logger
        self._straddle_strategy = PreEventStraddleStrategy()

    def evaluate(
        self,
        features_1m: Optional[Dict] = None,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> EventOverlayDecision:
        context = context or {}

        fundamental = context.get("fundamental", {}) or {}
        session_phase = str(context.get("session_phase", "UNKNOWN"))
        narrative_label = str(context.get("narrative_label", "NEUTRAL"))
        day_type = str(context.get("day_type", "UNKNOWN"))

        # accept both nested and flattened event fields
        event_severity = self._safe_int(
            fundamental.get("event_severity", context.get("event_severity", 0)),
            0,
        )
        event_days_away = self._safe_int(
            fundamental.get("event_days_away", context.get("event_days_away", 999)),
            999,
        )
        event_name = str(
            fundamental.get("event_name", context.get("event_name", "NONE"))
        )

        active = (
            narrative_label == "EVENT_RISK"
            or day_type == "EVENT_DAY"
            or (event_severity >= 2 and event_days_away <= 1)
        )

        if not active:
            return EventOverlayDecision(
                active=False,
                block_directional_trades=False,
                size_factor=1.0,
                allow_straddle=False,
                reason="no_major_event_overlay",
            )

        block_directional = True
        size_factor = 0.3
        allow_straddle = session_phase not in (
            "PRE_MARKET",
            "OPENING_AUCTION",
            "LAST_MINUTES",
            "POST_MARKET",
        )

        straddle_opps: List[Dict[str, Any]] = []
        if allow_straddle:
            try:
                results = self._straddle_strategy.scan(
                    features_1m=features_1m or {},
                    features_5m=features_5m,
                    features_15m=features_15m,
                    context=context,
                )
                for opp in results:
                    straddle_opps.append(opp.to_dict() if hasattr(opp, "to_dict") else opp)
            except Exception as e:
                self._logger.error(
                    "Pre-event straddle scan failed",
                    extra={"error": str(e)},
                )

        decision = EventOverlayDecision(
            active=True,
            block_directional_trades=block_directional,
            size_factor=size_factor,
            allow_straddle=allow_straddle,
            reason=f"major_event:{event_name}",
            notes=[
                "major_event_overlay_active",
                "reduce_size_to_30pct",
                "directional_trades_should_be_restricted",
            ],
            straddle_opportunities=straddle_opps,
        )

        self._logger.info(
            "Event overlay evaluated",
            extra={
                "active": decision.active,
                "event_name": event_name,
                "event_severity": event_severity,
                "event_days_away": event_days_away,
                "block_directional_trades": decision.block_directional_trades,
                "size_factor": decision.size_factor,
                "allow_straddle": decision.allow_straddle,
                "straddle_count": len(decision.straddle_opportunities),
            },
        )

        return decision

    def get_status(self, context: Optional[Dict] = None) -> Dict[str, Any]:
        return self.evaluate(context=context).to_dict()

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Event Overlay Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    overlay = EventOverlay()

    print(" [Test 1] No event -> inactive...")
    ctx1 = {
        "fundamental": {
            "event_severity": 0,
            "event_days_away": 999,
            "event_name": "NONE",
        },
        "session_phase": "GOLDEN_AM",
    }
    r1 = overlay.evaluate(context=ctx1)
    if not r1.active and r1.size_factor == 1.0:
        print(f" ✅ Inactive event overlay: {r1.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Unexpected inactive result: {r1.to_dict()}")
        failed += 1

    print("\n [Test 2] Major event activates overlay...")
    ctx2 = {
        "fundamental": {
            "event_severity": 2,
            "event_days_away": 0,
            "event_name": "RBI MPC Decision",
        },
        "session_phase": "GOLDEN_AM",
        "feed_health": "HEALTHY",
        "data_quality_score": 80,
        "spot_price": 22425.0,
        "capital": 50000.0,
        "event_minutes_away": 90,
        "options": {
            "atm_strike_used": 22450,
            "atm_iv_pct": 18.5,
            "iv_rank_session": 35.0,
        },
        "option_chain": {
            22450: {
                "ce": {"ltp": 95.0},
                "pe": {"ltp": 92.0},
            }
        },
        "microstructure": {
            "spread_zscore": 0.5,
        },
    }
    r2 = overlay.evaluate(context=ctx2)
    if r2.active and r2.block_directional_trades and r2.size_factor == 0.3:
        print(f" ✅ Major event overlay active: {r2.to_dict()}")
        passed += 1
    else:
        print(f" ❌ Major event overlay failed: {r2.to_dict()}")
        failed += 1

    print("\n [Test 3] Straddle allowed in valid session...")
    if r2.allow_straddle:
        print(" ✅ Straddle allowed")
        passed += 1
    else:
        print(" ❌ Straddle should be allowed")
        failed += 1

    print("\n [Test 4] Invalid session blocks straddle...")
    ctx4 = dict(ctx2)
    ctx4["session_phase"] = "LAST_MINUTES"
    r4 = overlay.evaluate(context=ctx4)
    if r4.active and not r4.allow_straddle:
        print(" ✅ Invalid session blocks straddle")
        passed += 1
    else:
        print(f" ❌ Invalid-session logic failed: {r4.to_dict()}")
        failed += 1

    print("\n [Test 5] Flattened context also works...")
    ctx5 = {
        "event_severity": 2,
        "event_days_away": 0,
        "event_name": "Flattened Event",
        "session_phase": "GOLDEN_PM",
        "feed_health": "HEALTHY",
        "data_quality_score": 80,
        "spot_price": 22425.0,
        "capital": 50000.0,
        "event_minutes_away": 90,
        "options": {
            "atm_strike_used": 22450,
            "atm_iv_pct": 18.5,
            "iv_rank_session": 35.0,
        },
        "option_chain": {
            22450: {
                "ce": {"ltp": 95.0},
                "pe": {"ltp": 92.0},
            }
        },
        "microstructure": {
            "spread_zscore": 0.5,
        },
    }
    r5 = overlay.evaluate(context=ctx5)
    if r5.active:
        print(" ✅ Flattened event context works")
        passed += 1
    else:
        print(f" ❌ Flattened event context failed: {r5.to_dict()}")
        failed += 1

    print("\n [Test 6] Status helper...")
    st6 = overlay.get_status(ctx2)
    if isinstance(st6, dict) and "active" in st6 and "size_factor" in st6:
        print(f" ✅ Status helper works: {st6}")
        passed += 1
    else:
        print(f" ❌ Status helper failed: {st6}")
        failed += 1

    print("\n [Test 7] Empty context safe...")
    r7 = overlay.evaluate(context={})
    if isinstance(r7, EventOverlayDecision):
        print(f" ✅ Empty context safe: {r7.to_dict()}")
        passed += 1
    else:
        print(" ❌ Empty context failed")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Event Overlay working perfectly!")
        print(" ✅ Current-phase missing event overlay completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()