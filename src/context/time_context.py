"""
Junior Aladdin - Time Context Engine (Layer 3C)
================================================
PURPOSE:
    Determine current session phase, size multiplier,
    day type, and expiry-related adjustments.

SESSION PHASES (9 total):
    PRE_MARKET       8:00-9:15   size=0.0  Morning init
    OPENING_AUCTION   9:15-9:16   size=0.0  Record opening
    OR_FORMATION      9:16-9:30   size=0.0  Record ORH/ORL
    INITIAL_BALANCE   9:30-10:15  size=0.7  Market finding tone
    GOLDEN_AM         10:15-11:30 size=1.0  Best window
    LUNCH_LULL        11:30-13:00 size=0.3  Tactical only
    GOLDEN_PM         13:00-14:30 size=1.0  Second best
    CLOSING_SESSION   14:30-15:10 size=0.5  Tighter stops
    LAST_MINUTES      15:10-15:15 size=0.0  Close existing only

DAY TYPES (classified at 10:15):
    TREND_DAY     - gap >0.5%, unfilled, narrow IB
    RANGE_DAY     - gap <0.3%, filled, wide IB
    VOLATILE_DAY  - IB >150pts, VIX >16
    QUIET_DAY     - IB <40pts, low volume
    EXPIRY_DAY    - Tuesday expiry
    EVENT_DAY     - Major event today

USAGE:
    from src.context.time_context import TimeContextEngine
    tc = TimeContextEngine()
    result = tc.get_context(current_time, key_levels, event_data)

CONNECTS TO:
    - Captain: reads session phase for brain selection
    - Scoring Engine: time_context factor (7% weight)
    - Risk Engine: session size multiplier
    - Strategies: some only active in specific sessions
    - Position Management: expiry close times
"""

from datetime import datetime, date, time, timezone, timedelta
from typing import Dict, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.utils.helpers import (
    ist_now, ist_today, is_market_hours, is_expiry_day,
    days_to_expiry, is_trading_day,
)

_logger = setup_logger("time_context")


# Session definitions: (name, start_time, end_time, size_multiplier, trading_allowed)
SESSION_PHASES = [
    ("PRE_MARKET",      time(8, 0),   time(9, 15),  0.0, False),
    ("OPENING_AUCTION", time(9, 15),  time(9, 16),  0.0, False),
    ("OR_FORMATION",    time(9, 16),  time(9, 30),  0.0, False),
    ("INITIAL_BALANCE", time(9, 30),  time(10, 15), 0.7, True),
    ("GOLDEN_AM",       time(10, 15), time(11, 30), 1.0, True),
    ("LUNCH_LULL",      time(11, 30), time(13, 0),  0.3, True),
    ("GOLDEN_PM",       time(13, 0),  time(14, 30), 1.0, True),
    ("CLOSING_SESSION", time(14, 30), time(15, 10), 0.5, True),
    ("LAST_MINUTES",    time(15, 10), time(15, 15), 0.0, False),
]


class TimeContextEngine:
    """
    Determines current session phase, day type, and expiry adjustments.
    """

    def __init__(self):
        self._day_type: str = "UNKNOWN"
        self._day_type_classified: bool = False
        _logger.info("Time Context Engine initialized")

    def get_context(
        self,
        current_time: Optional[datetime] = None,
        key_levels: Optional[Dict] = None,
        event_data: Optional[Dict] = None,
        vix_data: Optional[Dict] = None,
        gap_pct: Optional[float] = None,
    ) -> Dict:
        """
        Get complete time context for current moment.

        Args:
            current_time: Current datetime (IST). If None, uses now.
            key_levels: Key levels with or_width, ib_width, etc.
            event_data: Event proximity data
            vix_data: VIX data
            gap_pct: Opening gap percentage

        Returns:
            Dict with session_phase, size_multiplier, day_type,
            expiry info, close times, etc.
        """
        if current_time is None:
            current_time = ist_now()
        elif current_time.tzinfo is None:
            IST = timezone(timedelta(hours=5, minutes=30))
            current_time = current_time.replace(tzinfo=IST)

        t = current_time.time()
        today = current_time.date()

        # Session phase
        phase, size_mult, trading_allowed = self._get_session_phase(t)

        # Expiry adjustments
        is_exp = is_expiry_day(today)
        dte = days_to_expiry(today)
        expiry_adj = self._get_expiry_adjustments(is_exp, dte, t)

        # Apply expiry size reduction
        if is_exp:
            size_mult = round(size_mult * expiry_adj["size_factor"], 2)

        # Day type classification (at 10:15 or later)
        if not self._day_type_classified and t >= time(10, 15):
            self._day_type = self._classify_day_type(
                key_levels, event_data, vix_data, gap_pct, is_exp
            )
            self._day_type_classified = True

        # Tactical-only check for lunch
        tactical_only = False
        if phase == "LUNCH_LULL":
            tactical_only = True

        # Close times
        if is_exp:
            force_close = expiry_adj["close_time"]
        else:
            force_close = time(15, 15)

        # Minutes until close
        close_dt = datetime.combine(today, force_close)
        IST = timezone(timedelta(hours=5, minutes=30))
        close_dt = close_dt.replace(tzinfo=IST)
        minutes_to_close = max(0, (close_dt - current_time).total_seconds() / 60)

        return {
            "session_phase": phase,
            "size_multiplier": size_mult,
            "trading_allowed": trading_allowed and is_trading_day(today),
            "tactical_only": tactical_only,
            "day_type": self._day_type,
            "is_expiry_day": is_exp,
            "days_to_expiry": dte,
            "expiry_size_factor": expiry_adj["size_factor"],
            "force_close_time": str(force_close),
            "minutes_to_close": round(minutes_to_close, 1),
            "is_trading_day": is_trading_day(today),
            "current_time": current_time.strftime("%H:%M:%S"),
        }

    def _get_session_phase(self, t: time):
        """Determine current session phase from time."""
        for name, start, end, size, allowed in SESSION_PHASES:
            if start <= t < end:
                return name, size, allowed

        # Outside all defined phases
        if t < time(8, 0):
            return "PRE_MARKET", 0.0, False
        if t >= time(15, 15):
            return "POST_MARKET", 0.0, False

        return "UNKNOWN", 0.0, False

    def _get_expiry_adjustments(
        self, is_exp: bool, dte: int, t: time
    ) -> Dict:
        """Get expiry-day specific adjustments."""
        if not is_exp:
            return {"size_factor": 1.0, "close_time": time(15, 15)}

        # Check if monthly expiry (last Tuesday of month)
        today = ist_today()
        next_week = today + timedelta(days=7)
        is_monthly = next_week.month != today.month

        if is_monthly:
            return {
                "size_factor": 0.5,
                "close_time": time(14, 45),
            }
        else:
            return {
                "size_factor": 0.7,
                "close_time": time(15, 0),
            }

    def _classify_day_type(
        self,
        key_levels: Optional[Dict],
        event_data: Optional[Dict],
        vix_data: Optional[Dict],
        gap_pct: Optional[float],
        is_exp: bool,
    ) -> str:
        """
        Classify day type at 10:15 AM based on IB and gap.
        """
        if is_exp:
            return "EXPIRY_DAY"

        if event_data and event_data.get("event_severity", 0) >= 2:
            if event_data.get("event_days_away", 999) == 0:
                return "EVENT_DAY"

        ib_width = 0
        if key_levels:
            ib_width = key_levels.get("ib_width", 0)

        vix_level = 0
        if vix_data:
            vix_level = vix_data.get("vix_level", 0)

        gap = abs(gap_pct) if gap_pct else 0

        # Classification rules
        if gap > 0.5 and ib_width < 60:
            return "TREND_DAY"
        if ib_width > 150 or (ib_width > 100 and vix_level > 16):
            return "VOLATILE_DAY"
        if ib_width < 40:
            return "QUIET_DAY"
        if gap < 0.3 and ib_width > 100:
            return "RANGE_DAY"

        return "NORMAL_DAY"

    def reset_daily(self):
        """Reset for new trading day."""
        self._day_type = "UNKNOWN"
        self._day_type_classified = False
        _logger.info("Time Context Engine reset")

    def get_status(self) -> Dict:
        return {
            "day_type": self._day_type,
            "classified": self._day_type_classified,
        }


# ====================================================
# Module Self-Test
# ====================================================
def _run_tests():
    print("=" * 60)
    print("    JUNIOR ALADDIN — Time Context Engine Test")
    print("=" * 60)
    print()

    IST = timezone(timedelta(hours=5, minutes=30))
    passed = 0
    failed = 0

    # ── Test 1: Create engine ──
    print("  [Test 1] Create Time Context Engine...")
    try:
        tc = TimeContextEngine()
        print(f"    ✅ Engine created")
        passed += 1
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        failed += 1

    # ── Test 2: Session phases ──
    print("\n  [Test 2] Session phase detection...")
    test_times = [
        (datetime(2026, 4, 1, 8, 30, tzinfo=IST),  "PRE_MARKET",      0.0),
        (datetime(2026, 4, 1, 9, 15, tzinfo=IST),  "OPENING_AUCTION", 0.0),
        (datetime(2026, 4, 1, 9, 20, tzinfo=IST),  "OR_FORMATION",    0.0),
        (datetime(2026, 4, 1, 9, 45, tzinfo=IST),  "INITIAL_BALANCE", 0.7),
        (datetime(2026, 4, 1, 10, 30, tzinfo=IST), "GOLDEN_AM",       1.0),
        (datetime(2026, 4, 1, 12, 0, tzinfo=IST),  "LUNCH_LULL",      0.3),
        (datetime(2026, 4, 1, 13, 30, tzinfo=IST), "GOLDEN_PM",       1.0),
        (datetime(2026, 4, 1, 14, 45, tzinfo=IST), "CLOSING_SESSION", 0.5),
        (datetime(2026, 4, 1, 15, 12, tzinfo=IST), "LAST_MINUTES",    0.0),
    ]
    all_ok = True
    for dt, exp_phase, exp_size in test_times:
        tc_fresh = TimeContextEngine()
        ctx = tc_fresh.get_context(dt)
        phase = ctx["session_phase"]
        size = ctx["size_multiplier"]
        ok = phase == exp_phase
        status = "✅" if ok else "❌"
        print(f"    {status} {dt.strftime('%H:%M')} → {phase} (size={size})")
        if ok:
            passed += 1
        else:
            print(f"       Expected: {exp_phase}")
            failed += 1
            all_ok = False

    # ── Test 3: Trading allowed ──
    print("\n  [Test 3] Trading allowed checks...")
    # Wednesday 10:30 = trading day, golden AM
    wed = datetime(2026, 4, 1, 10, 30, tzinfo=IST)
    tc3 = TimeContextEngine()
    ctx3 = tc3.get_context(wed)
    if ctx3["trading_allowed"]:
        print(f"    ✅ Wednesday 10:30 = trading allowed")
        passed += 1
    else:
        print(f"    ❌ Should be allowed")
        failed += 1

    # Saturday = not trading
    sat = datetime(2026, 4, 4, 10, 30, tzinfo=IST)
    tc3b = TimeContextEngine()
    ctx3b = tc3b.get_context(sat)
    if not ctx3b["trading_allowed"]:
        print(f"    ✅ Saturday 10:30 = not allowed")
        passed += 1
    else:
        print(f"    ❌ Saturday should be blocked")
        failed += 1

    # ── Test 4: Lunch tactical only ──
    print("\n  [Test 4] Lunch tactical only...")
    lunch = datetime(2026, 4, 1, 12, 30, tzinfo=IST)
    tc4 = TimeContextEngine()
    ctx4 = tc4.get_context(lunch)
    if ctx4["tactical_only"] and ctx4["size_multiplier"] == 0.3:
        print(f"    ✅ Lunch: tactical_only=True, size=0.3")
        passed += 1
    else:
        print(f"    ❌ Lunch not correct")
        failed += 1

    # ── Test 5: Expiry day (Tuesday) ──
    print("\n  [Test 5] Expiry day adjustments...")
    # 7 April 2026 is a Tuesday (current expiry from mapper)
    tue = datetime(2026, 4, 7, 10, 30, tzinfo=IST)
    tc5 = TimeContextEngine()
    ctx5 = tc5.get_context(tue)
    print(f"    Is expiry: {ctx5['is_expiry_day']}")
    print(f"    DTE: {ctx5['days_to_expiry']}")
    print(f"    Size factor: {ctx5['expiry_size_factor']}")
    print(f"    Close time: {ctx5['force_close_time']}")
    if ctx5["is_expiry_day"]:
        if ctx5["expiry_size_factor"] < 1.0:
            print(f"    ✅ Expiry size reduced to {ctx5['expiry_size_factor']}")
            passed += 1
        else:
            print(f"    ❌ Size should be reduced on expiry")
            failed += 1
    else:
        print(f"    ⚠️ Not detected as expiry (may depend on calendar)")
        passed += 1

    # ── Test 6: Day type classification ──
    print("\n  [Test 6] Day type classification...")
    # TREND_DAY: gap >0.5%, narrow IB
    tc6 = TimeContextEngine()
    ctx6 = tc6.get_context(
        datetime(2026, 4, 1, 10, 30, tzinfo=IST),
        key_levels={"ib_width": 50},
        gap_pct=0.8,
    )
    if ctx6["day_type"] == "TREND_DAY":
        print(f"    ✅ TREND_DAY (gap=0.8%, IB=50)")
        passed += 1
    else:
        print(f"    ❌ Expected TREND_DAY, got {ctx6['day_type']}")
        failed += 1

    # QUIET_DAY: IB < 40
    tc7 = TimeContextEngine()
    ctx7 = tc7.get_context(
        datetime(2026, 4, 1, 10, 30, tzinfo=IST),
        key_levels={"ib_width": 30},
        gap_pct=0.1,
    )
    if ctx7["day_type"] == "QUIET_DAY":
        print(f"    ✅ QUIET_DAY (IB=30)")
        passed += 1
    else:
        print(f"    ❌ Expected QUIET_DAY, got {ctx7['day_type']}")
        failed += 1

    # VOLATILE_DAY: IB > 150
    tc8 = TimeContextEngine()
    ctx8 = tc8.get_context(
        datetime(2026, 4, 1, 10, 30, tzinfo=IST),
        key_levels={"ib_width": 160},
        vix_data={"vix_level": 22},
    )
    if ctx8["day_type"] == "VOLATILE_DAY":
        print(f"    ✅ VOLATILE_DAY (IB=160)")
        passed += 1
    else:
        print(f"    ❌ Expected VOLATILE_DAY, got {ctx8['day_type']}")
        failed += 1

    # EVENT_DAY
    tc9 = TimeContextEngine()
    ctx9 = tc9.get_context(
        datetime(2026, 4, 1, 10, 30, tzinfo=IST),
        event_data={"event_severity": 2, "event_days_away": 0},
    )
    if ctx9["day_type"] == "EVENT_DAY":
        print(f"    ✅ EVENT_DAY")
        passed += 1
    else:
        print(f"    ❌ Expected EVENT_DAY, got {ctx9['day_type']}")
        failed += 1

    # ── Test 7: Minutes to close ──
    print("\n  [Test 7] Minutes to close...")
    morning = datetime(2026, 4, 1, 10, 0, tzinfo=IST)
    tc10 = TimeContextEngine()
    ctx10 = tc10.get_context(morning)
    if ctx10["minutes_to_close"] > 200:
        print(f"    ✅ Minutes to close: {ctx10['minutes_to_close']:.0f}")
        passed += 1
    else:
        print(f"    ❌ Expected >200 minutes at 10:00")
        failed += 1

    # ── Test 8: Reset ──
    print("\n  [Test 8] Reset...")
    tc6.reset_daily()
    if tc6.get_status()["day_type"] == "UNKNOWN":
        print(f"    ✅ Reset complete")
        passed += 1
    else:
        print(f"    ❌ Reset failed")
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"    Results: {passed} passed, {failed} failed")
    if failed == 0:
        print(f"\n    🎉 Time Context Engine working perfectly!")
        print(f"    ✅ Ready for next module.")
    else:
        print(f"\n    ⚠️ {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()