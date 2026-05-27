"""
Junior Aladdin - Strategy Base Class
======================================
PURPOSE:
    Define the standard interface for all 13 trading strategies.
    Each strategy scans market conditions and returns Opportunities.

OPPORTUNITY OUTPUT:
    Every strategy returns a list of Opportunity dicts:
    {
        "strategy": str,          # Strategy name
        "direction": str,         # "BUY" or "SELL"
        "entry_price": float,     # Suggested entry
        "sl_price": float,        # Stop loss (thesis invalidation)
        "target_price": float,    # Target price
        "raw_score": int,         # 0-100 quality score
        "thesis": str,            # Human-readable reason
        "timeframe": str,         # Primary timeframe used
        "brain": str,             # Which brain this belongs to
        "conditions_met": dict,   # Which conditions passed/failed
        "timestamp": str,         # When signal was generated
    }

RULES (from plan):
    - Pure functions — no side effects, no broker access
    - Every condition must be True for signal to fire
    - SL is contextual (thesis invalidation level, not arbitrary ATR)
    - Target based on structure (swing high/low, not fixed points)
    - Each strategy has a config-defined minimum score threshold

CONNECTS TO:
    - Brain Engine: selects which strategies to scan
    - Trap Detection: filters opportunities before scoring
    - Scoring Engine: scores each opportunity
    - Captain: makes final trade decision
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from src.utils.logger import setup_logger
from src.utils.helpers import ist_now

IST = timezone(timedelta(hours=5, minutes=30))


class Opportunity:
    """
    Standard trade opportunity output.
    
    Every strategy produces these. They flow through:
    Strategy → Trap Detection → Scoring → ML Filter → 
    Behavioral → Risk → Captain → Execution
    """

    def __init__(
        self,
        strategy: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        target_price: float,
        raw_score: int,
        thesis: str,
        timeframe: str = "1min",
        brain: str = "STRUCTURAL",
        conditions_met: Optional[Dict] = None,
    ):
        self.strategy = strategy
        self.direction = direction.upper()
        self.entry_price = round(entry_price, 2)
        self.sl_price = round(sl_price, 2)
        self.target_price = round(target_price, 2)
        self.raw_score = max(0, min(100, raw_score))
        self.thesis = thesis
        self.timeframe = timeframe
        self.brain = brain
        self.conditions_met = conditions_met or {}
        self.timestamp = ist_now().isoformat()

        # Computed fields
        self.risk_points = round(abs(entry_price - sl_price), 2)
        self.reward_points = round(abs(target_price - entry_price), 2)
        self.risk_reward = (
            round(self.reward_points / self.risk_points, 2)
            if self.risk_points > 0 else 0.0
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for pipeline processing."""
        return {
            "strategy": self.strategy,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "target_price": self.target_price,
            "raw_score": self.raw_score,
            "thesis": self.thesis,
            "timeframe": self.timeframe,
            "brain": self.brain,
            "conditions_met": self.conditions_met,
            "timestamp": self.timestamp,
            "risk_points": self.risk_points,
            "reward_points": self.reward_points,
            "risk_reward": self.risk_reward,
        }

    def __repr__(self):
        return (
            f"Opportunity({self.strategy} {self.direction} "
            f"@{self.entry_price} SL={self.sl_price} "
            f"TGT={self.target_price} score={self.raw_score})"
        )


class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    
    Subclasses must implement:
        - scan() method that returns list of Opportunities
        - name property
        - brain property
    
    Usage:
        class MyStrategy(StrategyBase):
            @property
            def name(self): return "MY_STRATEGY"
            
            @property
            def brain(self): return "STRUCTURAL"
            
            def scan(self, features, context) -> List[Opportunity]:
                # Check conditions, return opportunities
                ...
    """

    def __init__(self):
        self._logger = setup_logger(f"strategy_{self.name.lower()}")
        self._scan_count = 0
        self._signal_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., 'VWAP_PULLBACK')."""
        pass

    @property
    @abstractmethod
    def brain(self) -> str:
        """Which brain this strategy belongs to."""
        pass

    @abstractmethod
    def scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        """
        Scan for trading opportunities.
        
        Args:
            features_1m: 1-min timeframe features (price+momentum+volatility+volume)
            features_5m: 5-min features (optional)
            features_15m: 15-min features (optional)
            context: Dict with narrative, regime, time_context, key_levels, 
                     options, smart_money, microstructure, mtf, session_memory
        
        Returns:
            List of Opportunity objects (empty if no signal)
        """
        pass

    def safe_scan(
        self,
        features_1m: Dict,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        context: Optional[Dict] = None,
    ) -> List[Opportunity]:
        """
        Safe wrapper around scan() — catches exceptions.
        One strategy failing must NOT crash the pipeline.
        """
        self._scan_count += 1
        try:
            opportunities = self.scan(features_1m, features_5m, features_15m, context)
            if opportunities:
                self._signal_count += len(opportunities)
                for opp in opportunities:
                    self._logger.info(
                        f"Signal: {opp.direction} @{opp.entry_price}",
                        extra={
                            "strategy": self.name,
                            "score": opp.raw_score,
                            "rr": opp.risk_reward,
                        },
                    )
            return opportunities or []
        except Exception as e:
            self._logger.error(
                f"Scan error: {e}",
                extra={"strategy": self.name, "scan_count": self._scan_count},
            )
            return []

    def get_stats(self) -> Dict:
        """Get strategy performance stats."""
        return {
            "name": self.name,
            "brain": self.brain,
            "scan_count": self._scan_count,
            "signal_count": self._signal_count,
            "signal_rate": (
                round(self._signal_count / self._scan_count * 100, 2)
                if self._scan_count > 0 else 0.0
            ),
        }

    # ════════════════════════════════════
    # Helper methods for subclasses
    # ════════════════════════════════════

    @staticmethod
    def _safe_get(features: Optional[Dict], key: str, default=None):
        """Safely get a feature value."""
        if features is None:
            return default
        return features.get(key, default)

    @staticmethod
    def _all_conditions(conditions: Dict[str, bool]) -> bool:
        """Check if ALL conditions are True."""
        return all(conditions.values())

    @staticmethod
    def _count_true(conditions: Dict[str, bool]) -> int:
        """Count how many conditions are True."""
        return sum(1 for v in conditions.values() if v)


# ════════════════════════════════════════
# Module Self-Test
# ════════════════════════════════════════

def _run_tests():
    print("=" * 60)
    print("  JUNIOR ALADDIN — Strategy Base Class Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    # ── Test 1: Opportunity creation ──
    print("  [Test 1] Opportunity creation...")
    opp = Opportunity(
        strategy="TEST_STRATEGY",
        direction="BUY",
        entry_price=23200.0,
        sl_price=23185.0,
        target_price=23230.0,
        raw_score=72,
        thesis="Test signal for validation",
        brain="STRUCTURAL",
    )
    if (opp.risk_points == 15.0 
            and opp.reward_points == 30.0
            and opp.risk_reward == 2.0):
        print(f"    ✅ Risk={opp.risk_points}, Reward={opp.reward_points}, RR={opp.risk_reward}")
        passed += 1
    else:
        print(f"    ❌ Risk/Reward computation wrong")
        failed += 1

    # ── Test 2: Opportunity to_dict ──
    print("\n  [Test 2] Opportunity to_dict...")
    d = opp.to_dict()
    expected_keys = [
        "strategy", "direction", "entry_price", "sl_price",
        "target_price", "raw_score", "thesis", "risk_reward",
    ]
    missing = [k for k in expected_keys if k not in d]
    if not missing:
        print(f"    ✅ All keys present")
        passed += 1
    else:
        print(f"    ❌ Missing: {missing}")
        failed += 1

    # ── Test 3: Score clamping ──
    print("\n  [Test 3] Score clamping...")
    opp_high = Opportunity("T", "BUY", 100, 95, 110, 150, "test")
    opp_low = Opportunity("T", "SELL", 100, 105, 90, -10, "test")
    if opp_high.raw_score == 100 and opp_low.raw_score == 0:
        print(f"    ✅ Scores clamped: 150→100, -10→0")
        passed += 1
    else:
        print(f"    ❌ Clamping failed")
        failed += 1

    # ── Test 4: Concrete strategy subclass ──
    print("\n  [Test 4] Concrete strategy subclass...")

    class TestStrategy(StrategyBase):
        @property
        def name(self): return "TEST"
        
        @property
        def brain(self): return "STRUCTURAL"
        
        def scan(self, f1m, f5m=None, f15m=None, ctx=None):
            close = self._safe_get(f1m, "last_close", 0)
            if close > 23000:
                return [Opportunity(
                    self.name, "BUY", close, close - 15, close + 25,
                    65, "Test buy signal"
                )]
            return []

    try:
        ts = TestStrategy()
        result = ts.safe_scan({"last_close": 23100})
        if len(result) == 1 and result[0].direction == "BUY":
            print(f"    ✅ Strategy produced signal: {result[0]}")
            passed += 1
        else:
            print(f"    ❌ Expected 1 BUY signal")
            failed += 1
    except Exception as e:
        print(f"    ❌ Error: {e}")
        failed += 1

    # ── Test 5: No signal when conditions not met ──
    print("\n  [Test 5] No signal when conditions not met...")
    result2 = ts.safe_scan({"last_close": 22000})
    if len(result2) == 0:
        print(f"    ✅ No signal (price below threshold)")
        passed += 1
    else:
        print(f"    ❌ Should be empty")
        failed += 1

    # ── Test 6: Safe scan handles errors ──
    print("\n  [Test 6] Safe scan error handling...")

    class BrokenStrategy(StrategyBase):
        @property
        def name(self): return "BROKEN"
        @property
        def brain(self): return "TEST"
        def scan(self, f1m, f5m=None, f15m=None, ctx=None):
            raise ValueError("Intentional error")

    bs = BrokenStrategy()
    result3 = bs.safe_scan({})
    if result3 == []:
        print(f"    ✅ Error caught, returned empty list")
        passed += 1
    else:
        print(f"    ❌ Should return empty on error")
        failed += 1

    # ── Test 7: Stats tracking ──
    print("\n  [Test 7] Stats tracking...")
    stats = ts.get_stats()
    if stats["scan_count"] == 2 and stats["signal_count"] == 1:
        print(f"    ✅ Stats: scans={stats['scan_count']}, signals={stats['signal_count']}")
        passed += 1
    else:
        print(f"    ❌ Stats: {stats}")
        failed += 1

    # ── Test 8: Helper methods ──
    print("\n  [Test 8] Helper methods...")
    conditions = {"cond_a": True, "cond_b": True, "cond_c": False}
    if not StrategyBase._all_conditions(conditions):
        print(f"    ✅ _all_conditions=False (one is False)")
        passed += 1
    else:
        failed += 1
    if StrategyBase._count_true(conditions) == 2:
        print(f"    ✅ _count_true=2")
        passed += 1
    else:
        failed += 1
    if StrategyBase._safe_get(None, "key", 42) == 42:
        print(f"    ✅ _safe_get handles None")
        passed += 1
    else:
        failed += 1

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print(f"\n  🎉 Strategy Base Class working perfectly!")
        print(f"  ✅ Ready for concrete strategies.")
    else:
        print(f"\n  ⚠️ {failed} tests failed.")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()