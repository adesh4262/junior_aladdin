"""
Junior Aladdin - Pre-Event Straddle Strategy (Hardened Version)
===============================================================
PURPOSE:
Trade volatility expansion before major events using an ATM straddle.

This is an EVENT overlay strategy.
It is NOT directional.
It buys both ATM CE and ATM PE only when:
- a major event is close enough to matter
- IV is still acceptable
- both legs are available
- total debit fits the risk budget
- the session/feed/data state is safe

This hardened version improves:
- stricter event-window validation
- stronger debit-budget protection
- stronger session/feed/data quality gating
- cleaner synthetic risk model for downstream compatibility
- better consistency with the shared Opportunity pipeline

RISK MODEL:
- total debit is the actual capital at risk
- synthetic stop is modeled using debit retention
- target is modeled as debit expansion
- this preserves compatibility with downstream shared execution/scoring structures

CONNECTS TO:
- Event overlay
- Options features / option chain
- Risk engine
- Captain / execution
- StrategyBase / Opportunity
"""

from typing import Dict, List, Optional

from src.strategies.strategy_base import StrategyBase, Opportunity
from src.utils.logger import setup_logger


class PreEventStraddleStrategy(StrategyBase):
    """
    Hardened pre-event volatility expansion strategy.
    """

    @property
    def name(self) -> str:
        return "PRE_EVENT_STRADDLE"

    @property
    def brain(self) -> str:
        return "EVENT"

    def __init__(self):
        self._max_iv_rank_pct = 60.0
        self._max_atm_iv_pct = 30.0

        self._event_window_min = 120
        self._event_window_start_min = 60

        self._max_spread_zscore = 2.0
        self._max_risk_pct = 0.004  # 0.4% of capital

        self._target_debit_expansion_pct = 50.0
        self._synthetic_stop_debit_retention = 0.75

        super().__init__()
        self._logger = setup_logger("strategy_pre_event_straddle")

    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        if not context:
            return []

        opp = self._check_setup(context)
        return [opp] if opp else []

    def _check_setup(self, ctx: Dict) -> Optional[Opportunity]:
        fundamental = ctx.get("fundamental", {})
        options = ctx.get("options", {})
        option_chain = ctx.get("option_chain", {})
        micro = ctx.get("microstructure", {})

        session = str(ctx.get("session_phase", ""))
        feed_health = str(ctx.get("feed_health", "UNKNOWN"))
        data_quality_score = float(ctx.get("data_quality_score", 0) or 0)
        spot_price = float(ctx.get("spot_price", 0.0) or 0.0)
        capital = float(ctx.get("capital", 50000.0) or 50000.0)

        event_severity = int(fundamental.get("event_severity", 0) or 0)
        event_name = str(fundamental.get("event_name", "NONE"))
        event_days_away = int(fundamental.get("event_days_away", 999) or 999)
        event_minutes_away = ctx.get("event_minutes_away")

        atm_strike = int(options.get("atm_strike_used", 0) or 0)
        atm_iv_pct = float(options.get("atm_iv_pct", 0.0) or 0.0)
        iv_rank = options.get("iv_rank_session")
        spread_zscore = micro.get("spread_zscore")

        if atm_strike <= 0 and spot_price > 0:
            atm_strike = round(spot_price / 50) * 50

        ce_data = option_chain.get(atm_strike, {}).get("ce", {})
        pe_data = option_chain.get(atm_strike, {}).get("pe", {})

        ce_ltp = float(self._safe_get(ce_data, "ltp", 0.0) or 0.0)
        pe_ltp = float(self._safe_get(pe_data, "ltp", 0.0) or 0.0)

        total_debit = round(ce_ltp + pe_ltp, 2)
        max_risk_rupees = round(capital * self._max_risk_pct, 2)

        event_time_ok = False
        if event_minutes_away is not None:
            event_time_ok = (
                self._event_window_start_min
                <= float(event_minutes_away)
                <= self._event_window_min
            )
        else:
            # fallback only if same-day high-impact event
            event_time_ok = event_severity == 2 and event_days_away == 0

        conditions = {
            "major_event": event_severity == 2,
            "event_time_ok": event_time_ok,
            "session_ok": session not in (
                "PRE_MARKET",
                "OPENING_AUCTION",
                "LAST_MINUTES",
                "POST_MARKET",
            ),
            "feed_ok": feed_health not in ("DOWN", "STALE"),
            "data_quality_ok": data_quality_score >= 40,
            "atm_strike_ok": atm_strike > 0,
            "ce_available": ce_ltp > 0,
            "pe_available": pe_ltp > 0,
            "atm_iv_ok": atm_iv_pct > 0 and atm_iv_pct <= self._max_atm_iv_pct,
            "iv_rank_ok": iv_rank is None or float(iv_rank) <= self._max_iv_rank_pct,
            "spread_ok": spread_zscore is None or float(spread_zscore) < self._max_spread_zscore,
            "debit_positive": total_debit > 0,
            "risk_budget_ok": total_debit <= max_risk_rupees,
        }

        if not self._all_conditions(conditions):
            return None

        # synthetic but pipeline-safe risk model
        entry_price = total_debit
        sl_price = round(total_debit * self._synthetic_stop_debit_retention, 2)
        target_price = round(
            total_debit * (1 + self._target_debit_expansion_pct / 100.0),
            2,
        )

        synthetic_risk = abs(entry_price - sl_price)
        synthetic_reward = abs(target_price - entry_price)

        if synthetic_risk <= 0:
            return None

        synthetic_rr = synthetic_reward / synthetic_risk
        if synthetic_rr < 1.0:
            return None

        score = 55
        if iv_rank is not None and float(iv_rank) <= 40:
            score += 10
        if atm_iv_pct <= 20:
            score += 10
        if data_quality_score >= 70:
            score += 5
        if session in ("INITIAL_BALANCE", "GOLDEN_AM", "GOLDEN_PM"):
            score += 5
        if event_days_away == 0:
            score += 5

        thesis = (
            f"Pre-event ATM straddle for {event_name}: "
            f"ATM={atm_strike}, CE={ce_ltp}, PE={pe_ltp}, "
            f"IV={atm_iv_pct}%, IVRank={iv_rank}, debit={total_debit}, "
            f"syntheticRR={round(synthetic_rr, 2)}"
        )

        opp = Opportunity(
            strategy=self.name,
            direction="BUY",
            entry_price=entry_price,
            sl_price=sl_price,
            target_price=target_price,
            raw_score=score,
            thesis=thesis,
            timeframe="1min",
            brain=self.brain,
            conditions_met=conditions,
        )

        # attach straddle-specific metadata
        opp.legs = {
            "ce_strike": atm_strike,
            "ce_ltp": ce_ltp,
            "pe_strike": atm_strike,
            "pe_ltp": pe_ltp,
        }
        opp.instrument_type = "STRADDLE"
        opp.event_name = event_name
        opp.event_severity = event_severity
        opp.max_risk_rupees = max_risk_rupees
        opp.target_leg_gain_pct = self._target_debit_expansion_pct
        opp.total_debit = total_debit

        return opp


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Pre-Event Straddle Strategy Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    strategy = PreEventStraddleStrategy()

    print(" [Test 1] Create strategy...")
    if strategy.name == "PRE_EVENT_STRADDLE" and strategy.brain == "EVENT":
        print(f" ✅ Name={strategy.name}, Brain={strategy.brain}")
        passed += 1
    else:
        print(" ❌ Wrong strategy metadata")
        failed += 1

    print("\n [Test 2] Perfect valid pre-event setup...")
    ctx = {
        "session_phase": "GOLDEN_AM",
        "feed_health": "HEALTHY",
        "data_quality_score": 80,
        "spot_price": 22425.0,
        "capital": 50000.0,
        "event_minutes_away": 90,
        "fundamental": {
            "event_severity": 2,
            "event_name": "RBI MPC Decision",
            "event_days_away": 0,
        },
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
    results = strategy.safe_scan({}, context=ctx)
    if len(results) >= 1:
        opp = results[0]
        print(
            f" ✅ Signal created: debit={opp.entry_price}, "
            f"synthetic_sl={opp.sl_price}, target={opp.target_price}, rr={opp.risk_reward}"
        )
        print(f" Thesis: {opp.thesis}")
        if opp.risk_reward >= 1.0:
            passed += 1
        else:
            print(" ❌ Invalid synthetic RR")
            failed += 1
    else:
        print(" ❌ No signal")
        failed += 1

    print("\n [Test 3] No signal if event not major...")
    ctx_minor = {
        **ctx,
        "fundamental": {
            "event_severity": 1,
            "event_name": "Minor Event",
            "event_days_away": 0,
        },
    }
    r3 = strategy.safe_scan({}, context=ctx_minor)
    if len(r3) == 0:
        print(" ✅ No signal (event severity not 2)")
        passed += 1
    else:
        print(" ❌ Should block minor events")
        failed += 1

    print("\n [Test 4] No signal if IV too high...")
    ctx_high_iv = {
        **ctx,
        "options": {
            "atm_strike_used": 22450,
            "atm_iv_pct": 38.0,
            "iv_rank_session": 85.0,
        },
    }
    r4 = strategy.safe_scan({}, context=ctx_high_iv)
    if len(r4) == 0:
        print(" ✅ No signal (IV too expensive)")
        passed += 1
    else:
        print(" ❌ Should block high IV")
        failed += 1

    print("\n [Test 5] No signal if option legs missing...")
    ctx_missing_legs = {
        **ctx,
        "option_chain": {},
    }
    r5 = strategy.safe_scan({}, context=ctx_missing_legs)
    if len(r5) == 0:
        print(" ✅ No signal (option legs unavailable)")
        passed += 1
    else:
        print(" ❌ Should block missing legs")
        failed += 1

    print("\n [Test 6] No signal if debit exceeds risk budget...")
    ctx_expensive = {
        **ctx,
        "option_chain": {
            22450: {
                "ce": {"ltp": 130.0},
                "pe": {"ltp": 120.0},
            }
        },
    }
    r6 = strategy.safe_scan({}, context=ctx_expensive)
    if len(r6) == 0:
        print(" ✅ No signal (debit exceeds risk budget)")
        passed += 1
    else:
        print(" ❌ Should block expensive straddle")
        failed += 1

    print("\n [Test 7] No signal during invalid session...")
    ctx_bad_session = {**ctx, "session_phase": "LAST_MINUTES"}
    r7 = strategy.safe_scan({}, context=ctx_bad_session)
    if len(r7) == 0:
        print(" ✅ No signal (invalid session)")
        passed += 1
    else:
        print(" ❌ Should block LAST_MINUTES")
        failed += 1

    print("\n [Test 8] Empty context...")
    r8 = strategy.safe_scan({})
    if len(r8) == 0:
        print(" ✅ No crash with empty input")
        passed += 1
    else:
        print(" ❌ Should return empty")
        failed += 1

    print("\n [Test 9] Stats tracking...")
    stats = strategy.get_stats()
    if stats["scan_count"] >= 7:
        print(f" ✅ Scans={stats['scan_count']}, Signals={stats['signal_count']}")
        passed += 1
    else:
        print(f" ❌ Stats: {stats}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Pre-Event Straddle Strategy (Hardened) working perfectly!")
        print(" ✅ Ready for next strategy.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()