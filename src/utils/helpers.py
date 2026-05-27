"""
Junior Aladdin - Helper Utilities Module
==========================================
PURPOSE:
 Common utility functions used across the entire system.
 Centralizes time handling, price rounding, formatting,
 and market calendar logic.

USAGE:
 from src.utils.helpers import (
     ist_now, is_market_hours, is_pre_market,
     round_to_strike, format_rupees, pct_change,
     next_expiry_date, is_expiry_day,
 )

 now = ist_now()
 if is_market_hours():
     strike = round_to_strike(24537)  # → 24550
 print(format_rupees(1234.56))  # → ₹1,234.56

CONNECTS TO:
 Every module in the system uses these helpers:
 - Data Engine: is_market_hours(), ist_now()
 - Strategies: round_to_strike()
 - Risk Engine: pct_change()
 - Journal: format_rupees()
 - Time Context: next_expiry_date(), is_expiry_day()
 - Captain: is_market_hours(), is_pre_market()
"""

import json
import math
import os
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, List


# ==============================================
# IST Timezone Constant
# ==============================================
IST = timezone(timedelta(hours=5, minutes=30))


# ==============================================
# Time Functions
# ==============================================
def ist_now() -> datetime:
    """
    Get current date and time in IST (Indian Standard Time).

    Returns:
        datetime: Current time with IST timezone info
    """
    return datetime.now(IST)


def ist_today() -> date:
    """
    Get today's date in IST.

    Returns:
        date: Today's date in IST
    """
    return datetime.now(IST).date()


def ist_time_now() -> time:
    """
    Get current time only (no date) in IST.

    Returns:
        time: Current time in IST
    """
    return datetime.now(IST).time()


def is_market_hours(check_time: Optional[datetime] = None) -> bool:
    """
    Check if current time (or given datetime) is during market hours
    AND it is a valid trading day.

    Market hours: 9:15 AM to 3:30 PM IST
    Trading day: Not Saturday, not Sunday, not holiday

    Args:
        check_time: Optional datetime to check. If None, uses current IST time.

    Returns:
        bool: True only if both conditions are true:
              1. It is a trading day
              2. Time is between 9:15 and 15:30

    Examples:
        Saturday 14:00 → False
        Monday 08:30   → False
        Monday 10:00   → True
    """
    if check_time is not None:
        if isinstance(check_time, datetime):
            dt_obj = check_time
            if dt_obj.tzinfo is not None:
                dt_obj = dt_obj.astimezone(IST)
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=IST)
            check_date = dt_obj.date()
            check_t = dt_obj.time()
        else:
            # If someone passes only a time object, use today's date
            check_date = ist_today()
            check_t = check_time
    else:
        now = ist_now()
        check_date = now.date()
        check_t = now.time()

    # First: must be a trading day
    if not is_trading_day(check_date):
        return False

    # Second: must be in market time window
    market_open = time(9, 15)
    market_close = time(15, 30)
    return market_open <= check_t <= market_close


def is_pre_market(check_time: Optional[datetime] = None) -> bool:
    """
    Check if current time (or given datetime) is during pre-market hours
    AND it is a valid trading day.

    Pre-market: 8:00 AM to 9:15 AM IST

    Args:
        check_time: Optional datetime to check.

    Returns:
        bool: True if in pre-market window on a trading day.
    """
    if check_time is not None:
        if isinstance(check_time, datetime):
            dt_obj = check_time
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=IST)
            check_date = dt_obj.date()
            check_t = dt_obj.time()
        else:
            check_date = ist_today()
            check_t = check_time
    else:
        now = ist_now()
        check_date = now.date()
        check_t = now.time()

    if not is_trading_day(check_date):
        return False

    pre_open = time(8, 0)
    market_open = time(9, 15)
    return pre_open <= check_t < market_open


def is_post_market(check_time: Optional[datetime] = None) -> bool:
    """
    Check if current time is after market close on a trading day.

    Post-market: after 3:30 PM IST

    Args:
        check_time: Optional datetime to check.

    Returns:
        bool: True if after market hours on a trading day.
    """
    if check_time is not None:
        if isinstance(check_time, datetime):
            dt_obj = check_time
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=IST)
            check_date = dt_obj.date()
            check_t = dt_obj.time()
        else:
            check_date = ist_today()
            check_t = check_time
    else:
        now = ist_now()
        check_date = now.date()
        check_t = now.time()

    if not is_trading_day(check_date):
        return False

    market_close = time(15, 30)
    return check_t > market_close


def is_weekend(check_date: Optional[date] = None) -> bool:
    """
    Check if a date is Saturday or Sunday.

    Args:
        check_date: Date to check. If None, uses today.

    Returns:
        bool: True if Saturday (5) or Sunday (6)
    """
    d = check_date if check_date is not None else ist_today()
    return d.weekday() >= 5


# ==============================================
# Market Holiday Calendar
# ==============================================
_holidays_cache = None
_holidays_cache_date = None


def _load_holidays() -> List[str]:
    """
    Load market holidays from economic calendar JSON.

    Returns:
        List of date strings in YYYY-MM-DD format where severity == 0
        (indicating market holiday).
    """
    global _holidays_cache, _holidays_cache_date

    today = ist_today()
    if _holidays_cache is not None and _holidays_cache_date == today:
        return _holidays_cache

    calendar_path = os.path.join("data", "calendar", "economic_calendar.json")
    holidays = []

    if os.path.isfile(calendar_path):
        try:
            with open(calendar_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for event in data.get("events", []):
                if event.get("severity") == 0:
                    holidays.append(event["date"])

        except (json.JSONDecodeError, KeyError):
            pass

    _holidays_cache = holidays
    _holidays_cache_date = today
    return holidays


def is_trading_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a date is a trading day (not weekend, not holiday).

    Args:
        check_date: Date to check. If None, uses today.

    Returns:
        bool: True if it's a trading day
    """
    d = check_date if check_date is not None else ist_today()

    # Weekend check
    if is_weekend(d):
        return False

    # Holiday check
    holidays = _load_holidays()
    date_str = d.strftime("%Y-%m-%d")
    if date_str in holidays:
        return False

    return True


# ==============================================
# Price Functions
# ==============================================
def round_to_strike(price: float, step: int = 50) -> int:
    """
    Round a price to the nearest NIFTY strike price.

    Uses normal .5-up rounding, not banker's rounding.

    Examples:
        round_to_strike(24537) → 24550
        round_to_strike(24520) → 24500
        round_to_strike(24525) → 24550
    """
    if price < 0:
        print(f"WARNING: round_to_strike called with negative price={price}; returning 0")
        return 0
    return int(math.floor(price / step + 0.5)) * step


def pct_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change from old to new value.

    Returns:
        float: Decimal percentage change
               0.05 means +5%
               -0.05 means -5%
    """
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / abs(old_value)


def points_to_rupees(points: float, lot_size: int = 65) -> float:
    """
    Convert NIFTY points to Rupees.
    """
    return points * lot_size


def rupees_to_points(rupees: float, lot_size: int = 65) -> float:
    """
    Convert Rupees to NIFTY points.
    """
    if lot_size == 0:
        return 0.0
    return rupees / lot_size


# ==============================================
# Formatting Functions
# ==============================================
def format_rupees(amount: float) -> str:
    """
    Format a number as Indian Rupees with comma separation.

    Examples:
        format_rupees(1234.56) → "₹1,234.56"
        format_rupees(100000)  → "₹1,00,000.00"
    """
    negative = amount < 0
    amount = abs(amount)

    integer_part = int(amount)
    decimal_part = round(amount - integer_part, 2)
    decimal_str = f"{decimal_part:.2f}"[1:]

    int_str = str(integer_part)
    if len(int_str) <= 3:
        formatted = int_str
    else:
        last_three = int_str[-3:]
        remaining = int_str[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted = ",".join(groups) + "," + last_three

    sign = "-" if negative else ""
    return f"{sign}\u20B9{formatted}{decimal_str}"


def format_points(points: float) -> str:
    """
    Format NIFTY points with sign and 1 decimal.
    """
    if points > 0:
        return f"+{points:.1f} pts"
    elif points < 0:
        return f"{points:.1f} pts"
    else:
        return "0.0 pts"


def format_pct(value: float) -> str:
    """
    Format a decimal as percentage string.
    """
    return f"{value * 100:.2f}%"


# ==============================================
# Expiry & Calendar Functions
# ==============================================
def next_expiry_date(from_date: Optional[date] = None) -> date:
    """
    Find the next NIFTY weekly expiry date (Tuesday).

    If Tuesday is holiday, expiry moves to previous trading day.
    """
    d = from_date if from_date is not None else ist_today()

    # Tuesday = weekday 1
    days_until_tuesday = (1 - d.weekday()) % 7

    if days_until_tuesday == 0 and d.weekday() == 1:
        next_tuesday = d
    elif days_until_tuesday == 0:
        next_tuesday = d + timedelta(days=7)
    else:
        next_tuesday = d + timedelta(days=days_until_tuesday)

    # Holiday adjustment
    holidays = _load_holidays()
    tuesday_str = next_tuesday.strftime("%Y-%m-%d")
    if tuesday_str in holidays:
        monday = next_tuesday - timedelta(days=1)
        monday_str = monday.strftime("%Y-%m-%d")
        if monday_str in holidays or monday.weekday() >= 5:
            friday = next_tuesday - timedelta(days=3)
            return friday
        return monday

    return next_tuesday


def is_expiry_day(check_date: Optional[date] = None) -> bool:
    """
    Check if a given date is expiry day.
    """
    d = check_date if check_date is not None else ist_today()
    expiry = next_expiry_date(d)
    return d == expiry


def days_to_expiry(from_date: Optional[date] = None) -> int:
    """
    Calculate days until next expiry.
    """
    d = from_date if from_date is not None else ist_today()
    expiry = next_expiry_date(d)
    return (expiry - d).days


def trading_days_between(start_date: date, end_date: date) -> int:
    """
    Count trading days between two dates (inclusive),
    excluding weekends and holidays.
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    holidays = _load_holidays()
    count = 0
    current = start_date

    while current <= end_date:
        if current.weekday() < 5:
            current_str = current.strftime("%Y-%m-%d")
            if current_str not in holidays:
                count += 1
        current += timedelta(days=1)

    return count


def time_to_expiry_years(from_date: Optional[date] = None) -> float:
    """
    Calculate time to expiry in years (for Black-Scholes).
    Minimum = 0.5 day / 365 to avoid divide-by-zero issues.
    """
    days = days_to_expiry(from_date)
    return max(0.5, days) / 365.0


# ==============================================
# Module Self-Test
# ==============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  JUNIOR ALADDIN — Helpers Module Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: IST Time ──
    print("  [Test 1] IST Time Functions...")
    now = ist_now()
    today = ist_today()
    current_time = ist_time_now()
    print(f"    IST Now : {now}")
    print(f"    IST Today: {today}")
    print(f"    IST Time : {current_time}")
    if now.tzinfo is not None:
        print(f"    ✅ Timezone aware: {now.tzinfo}")
        passed += 1
    else:
        print(f"    ❌ Not timezone aware")
        failed += 1

    # ── Test 2: Market Hours Check with trading day awareness ──
    print("\n  [Test 2] Market Hours Check...")
    from datetime import datetime as dt

    # Friday = trading day
    friday_tests = [
        (dt(2026, 3, 27, 8, 0, tzinfo=IST), False, "Friday 8:00 AM - before market"),
        (dt(2026, 3, 27, 9, 15, tzinfo=IST), True, "Friday 9:15 AM - market open"),
        (dt(2026, 3, 27, 12, 0, tzinfo=IST), True, "Friday 12:00 PM - midday"),
        (dt(2026, 3, 27, 15, 30, tzinfo=IST), True, "Friday 3:30 PM - close"),
        (dt(2026, 3, 27, 15, 31, tzinfo=IST), False, "Friday 3:31 PM - after close"),
    ]

    # Saturday = not trading day
    saturday_tests = [
        (dt(2026, 3, 28, 10, 0, tzinfo=IST), False, "Saturday 10:00 AM - weekend"),
        (dt(2026, 3, 28, 14, 13, tzinfo=IST), False, "Saturday 2:13 PM - weekend"),
    ]

    all_market_ok = True
    for test_time, expected, description in friday_tests + saturday_tests:
        result = is_market_hours(test_time)
        status = "✅" if result == expected else "❌"
        print(f"    {status} {description}: {result}")
        if result == expected:
            passed += 1
        else:
            failed += 1
            all_market_ok = False

    # ── Test 3: Pre-Market Check ──
    print("\n  [Test 3] Pre-Market Check...")
    pre_tests = [
        (dt(2026, 3, 27, 7, 59, tzinfo=IST), False, "Friday 7:59 AM"),
        (dt(2026, 3, 27, 8, 0, tzinfo=IST), True, "Friday 8:00 AM"),
        (dt(2026, 3, 27, 8, 30, tzinfo=IST), True, "Friday 8:30 AM"),
        (dt(2026, 3, 27, 9, 14, tzinfo=IST), True, "Friday 9:14 AM"),
        (dt(2026, 3, 27, 9, 15, tzinfo=IST), False, "Friday 9:15 AM"),
        (dt(2026, 3, 28, 8, 30, tzinfo=IST), False, "Saturday 8:30 AM"),
    ]
    for test_time, expected, description in pre_tests:
        result = is_pre_market(test_time)
        status = "✅" if result == expected else "❌"
        print(f"    {status} {description}: is_pre_market={result}")
        if result == expected:
            passed += 1
        else:
            failed += 1

    # ── Test 4: Round to Strike ──
    print("\n  [Test 4] Round to Strike...")
    strike_tests = [
        (24537, 24550),
        (24520, 24500),
        (24525, 24550),
        (24500, 24500),
        (24524, 24500),
        (24575, 24600),
        (24550, 24550),
        (24499, 24500),
        (24501, 24500),
    ]
    for price, expected in strike_tests:
        result = round_to_strike(price)
        status = "✅" if result == expected else "❌"
        print(f"    {status} round_to_strike({price}) = {result} (expected {expected})")
        if result == expected:
            passed += 1
        else:
            failed += 1

    # ── Test 5: Percentage Change ──
    print("\n  [Test 5] Percentage Change...")
    pct_tests = [
        (100, 105, 0.05),
        (100, 95, -0.05),
        (24500, 24600, 100 / 24500),
        (0, 100, 0.0),
    ]
    for old, new, expected in pct_tests:
        result = pct_change(old, new)
        status = "✅" if abs(result - expected) < 0.0001 else "❌"
        print(f"    {status} pct_change({old}, {new}) = {result:.6f}")
        if abs(result - expected) < 0.0001:
            passed += 1
        else:
            failed += 1

    # ── Test 6: Points ↔ Rupees ──
    print("\n  [Test 6] Points to Rupees...")
    pts_tests = [
        (10, 650.0),
        (-5, -325.0),
        (0, 0.0),
        (1, 65.0),
    ]
    for points, expected in pts_tests:
        result = points_to_rupees(points)
        status = "✅" if result == expected else "❌"
        print(f"    {status} {points} pts = {format_rupees(result)}")
        if result == expected:
            passed += 1
        else:
            failed += 1

    # ── Test 7: Format Rupees ──
    print("\n  [Test 7] Format Rupees...")
    fmt_tests = [
        (1234.56, "\u20B91,234.56"),
        (0, "\u20B90.00"),
        (-500, "-\u20B9500.00"),
        (100000, "\u20B91,00,000.00"),
        (65, "\u20B965.00"),
        (1234567.89, "\u20B912,34,567.89"),
    ]
    for amount, expected in fmt_tests:
        result = format_rupees(amount)
        status = "✅" if result == expected else "❌"
        print(f"    {status} format_rupees({amount}) = {result}")
        if result == expected:
            passed += 1
        else:
            print(f"       Expected: {expected}")
            failed += 1

    # ── Test 8: Format Points ──
    print("\n  [Test 8] Format Points...")
    fpt_tests = [
        (10.5, "+10.5 pts"),
        (-3.0, "-3.0 pts"),
        (0, "0.0 pts"),
    ]
    for pts, expected in fpt_tests:
        result = format_points(pts)
        status = "✅" if result == expected else "❌"
        print(f"    {status} format_points({pts}) = {result}")
        if result == expected:
            passed += 1
        else:
            failed += 1

    # ── Test 9: Expiry Functions ──
    print("\n  [Test 9] Expiry Functions...")
    test_date_tue = date(2026, 3, 24)
    test_date_wed = date(2026, 3, 25)
    test_date_mon = date(2026, 3, 23)

    exp_from_tue = next_expiry_date(test_date_tue)
    print(f"    From Tuesday {test_date_tue}: next expiry = {exp_from_tue} ({exp_from_tue.strftime('%A')})")
    if exp_from_tue == test_date_tue:
        print(f"    ✅ Tuesday IS expiry day")
        passed += 1
    else:
        print(f"    ❌ Expected {test_date_tue}")
        failed += 1

    exp_from_wed = next_expiry_date(test_date_wed)
    print(f"    From Wednesday {test_date_wed}: next expiry = {exp_from_wed} ({exp_from_wed.strftime('%A')})")
    if exp_from_wed == date(2026, 3, 31):
        print(f"    ✅ Correctly found next Tuesday")
        passed += 1
    else:
        print(f"    ❌ Expected 2026-03-31")
        failed += 1

    exp_from_mon = next_expiry_date(test_date_mon)
    print(f"    From Monday {test_date_mon}: next expiry = {exp_from_mon} ({exp_from_mon.strftime('%A')})")
    if exp_from_mon == test_date_tue:
        print(f"    ✅ Correctly found upcoming Tuesday")
        passed += 1
    else:
        print(f"    ❌ Expected {test_date_tue}")
        failed += 1

    is_exp_tue = is_expiry_day(test_date_tue)
    is_exp_wed = is_expiry_day(test_date_wed)
    if is_exp_tue:
        print(f"    ✅ is_expiry_day({test_date_tue}) = True")
        passed += 1
    else:
        print(f"    ❌ Tuesday should be expiry")
        failed += 1

    if not is_exp_wed:
        print(f"    ✅ is_expiry_day({test_date_wed}) = False")
        passed += 1
    else:
        print(f"    ❌ Wednesday should not be expiry")
        failed += 1

    dte = days_to_expiry(test_date_mon)
    if dte == 1:
        print(f"    ✅ days_to_expiry({test_date_mon}) = 1")
        passed += 1
    else:
        print(f"    ❌ Expected 1 day, got {dte}")
        failed += 1

    # ── Test 10: Trading Days ──
    print("\n  [Test 10] Trading Days Between...")
    mon = date(2026, 3, 23)
    fri = date(2026, 3, 27)
    td = trading_days_between(mon, fri)
    print(f"    {mon} to {fri}: {td} trading days")
    if td == 5:
        print(f"    ✅ Correct (Mon-Fri = 5)")
        passed += 1
    else:
        print(f"    ❌ Expected 5")
        failed += 1

    mon1 = date(2026, 3, 23)
    mon2 = date(2026, 3, 30)
    td2 = trading_days_between(mon1, mon2)
    print(f"    {mon1} to {mon2}: {td2} trading days")
    if td2 == 6:
        print(f"    ✅ Correct (Mon-Mon with weekend removed = 6)")
        passed += 1
    else:
        print(f"    ❌ Expected 6, got {td2}")
        failed += 1

    # ── Test 11: Weekend Check ──
    print("\n  [Test 11] Weekend Check...")
    sat = date(2026, 3, 28)
    sun = date(2026, 3, 29)
    wed = date(2026, 3, 25)

    if is_weekend(sat):
        print(f"    ✅ Saturday is weekend")
        passed += 1
    else:
        print(f"    ❌ Saturday should be weekend")
        failed += 1

    if is_weekend(sun):
        print(f"    ✅ Sunday is weekend")
        passed += 1
    else:
        print(f"    ❌ Sunday should be weekend")
        failed += 1

    if not is_weekend(wed):
        print(f"    ✅ Wednesday is not weekend")
        passed += 1
    else:
        print(f"    ❌ Wednesday should not be weekend")
        failed += 1

    # ── Test 12: Time to Expiry (Years) ──
    print("\n  [Test 12] Time to Expiry (Years)...")
    tte = time_to_expiry_years(test_date_mon)
    print(f"    time_to_expiry_years({test_date_mon}) = {tte:.6f}")
    if tte > 0:
        print(f"    ✅ Positive value for Black-Scholes")
        passed += 1
    else:
        print(f"    ❌ Should be positive")
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n  🎉 Helpers Module working perfectly!")
        print("  ✅ Ready for next module.")
    else:
        print(f"\n  ⚠️  {failed} tests failed. Check the logic.")
    print("=" * 60)