"""
Junior Aladdin - Option Chain Manager
=====================================
PURPOSE:
Provide a dedicated roadmap-aligned core module for option-chain management.

This file exists because the roadmap / folder architecture explicitly includes:
    src/core/option_chain.py

The system already has a working polling implementation in:
    src/core/option_chain_poller.py

So this module acts as the production-grade orchestration layer around the
poller, giving the rest of the system a stable interface for:
- polling the option chain
- caching latest snapshots
- exposing ATM IV / PCR / metadata
- tracking snapshot history
- integrating into Data Engine / Feature Engine cleanly

DESIGN GOALS:
- preserve existing tested poller logic
- no duplication of core polling math
- roadmap completeness without breaking working code
- strong validation, metadata, and history tracking

CONNECTS TO:
- AuthManager
- InstrumentMapper
- OptionChainPoller
- DataEngine
- Feature Engine / Options Features
"""

from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any

from src.utils.logger import setup_logger
from src.core.option_chain_poller import OptionChainPoller

IST = timezone(timedelta(hours=5, minutes=30))
_logger = setup_logger("option_chain_manager")


class OptionChainManager:
    """
    High-level manager for option-chain lifecycle.

    Responsibilities:
    - wrap OptionChainPoller
    - maintain latest chain
    - maintain snapshot history
    - expose convenience summaries
    """

    def __init__(
        self,
        auth_manager,
        instrument_mapper,
        max_history: int = 200,
    ):
        self._logger = _logger
        self._auth = auth_manager
        self._mapper = instrument_mapper
        self._poller = OptionChainPoller(auth_manager, instrument_mapper)

        self._current_chain: Dict = {}
        self._last_spot_price: float = 0.0
        self._last_poll_time: Optional[datetime] = None
        self._snapshot_history: deque = deque(maxlen=max_history)

        self._logger.info("Option Chain Manager initialized")

    # ------------------------------------------------------------------
    # Core polling
    # ------------------------------------------------------------------
    def poll(self, spot_price: float) -> Dict:
        """
        Poll fresh option-chain data and store snapshot.
        """
        if spot_price <= 0:
            self._logger.warning("Option chain poll skipped: invalid spot price", extra={"spot_price": spot_price})
            return self._current_chain

        try:
            chain = self._poller.poll(spot_price)
            self._current_chain = chain or {}
            self._last_spot_price = float(spot_price)
            self._last_poll_time = datetime.now(IST)

            snapshot = {
                "timestamp": self._last_poll_time.isoformat(),
                "spot_price": self._last_spot_price,
                "strike_count": len(self._current_chain),
                "atm_iv": self.get_atm_iv().get("avg_iv", 0.0),
                "pcr_oi": self.get_pcr().get("pcr_oi", 0.0),
            }
            self._snapshot_history.append(snapshot)

            self._logger.info(
                "Option chain manager poll complete",
                extra={
                    "spot_price": self._last_spot_price,
                    "strike_count": len(self._current_chain),
                    "history_size": len(self._snapshot_history),
                },
            )
            return self._current_chain

        except Exception as e:
            self._logger.error(
                "Option chain manager poll failed",
                extra={"error": str(e)},
            )
            return self._current_chain

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_current_chain(self) -> Dict:
        return self._current_chain

    def get_last_poll_time(self) -> Optional[str]:
        return self._last_poll_time.isoformat() if self._last_poll_time else None

    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        return list(self._snapshot_history)

    def get_atm_iv(self) -> Dict:
        if self._last_spot_price <= 0:
            return {
                "ce_iv": 0.0,
                "pe_iv": 0.0,
                "avg_iv": 0.0,
                "ce_iv_pct": 0.0,
                "pe_iv_pct": 0.0,
                "avg_iv_pct": 0.0,
            }
        try:
            return self._poller.get_atm_iv(self._last_spot_price)
        except Exception as e:
            self._logger.warning("ATM IV fetch failed", extra={"error": str(e)})
            return {
                "ce_iv": 0.0,
                "pe_iv": 0.0,
                "avg_iv": 0.0,
                "ce_iv_pct": 0.0,
                "pe_iv_pct": 0.0,
                "avg_iv_pct": 0.0,
            }

    def get_pcr(self) -> Dict:
        try:
            return self._poller.get_pcr()
        except Exception as e:
            self._logger.warning("PCR fetch failed", extra={"error": str(e)})
            return {
                "pcr_oi": 0.0,
                "pcr_volume": 0.0,
                "total_ce_oi": 0,
                "total_pe_oi": 0,
            }

    def get_status(self) -> Dict:
        """
        Full manager status for dashboard / engine diagnostics.
        """
        poller_status = self._poller.get_status()
        return {
            "has_chain": len(self._current_chain) > 0,
            "strike_count": len(self._current_chain),
            "last_spot_price": self._last_spot_price,
            "last_poll_time": self.get_last_poll_time(),
            "snapshot_history_count": len(self._snapshot_history),
            "poller_status": poller_status,
        }

    def reset_daily(self):
        """
        Reset daily option-chain state.
        """
        self._current_chain = {}
        self._last_spot_price = 0.0
        self._last_poll_time = None
        self._snapshot_history.clear()
        self._logger.info("Option Chain Manager reset")


# ============================================================================
# Module self-test
# ============================================================================
def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Option Chain Manager Test")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    class MockAuth:
        def get_smart_api(self):
            return None

    class MockMapper:
        def get_current_expiry(self):
            return None

    # ------------------------------------------------------------------
    # Test 1: Create manager
    # ------------------------------------------------------------------
    print(" [Test 1] Create manager...")
    try:
        mgr = OptionChainManager(MockAuth(), MockMapper())
        print(" ✅ Manager created")
        passed += 1
    except Exception as e:
        print(f" ❌ Failed to create manager: {e}")
        failed += 1
        print("\n" + "=" * 60)
        print(f" Results: {passed} passed, {failed} failed")
        return

    # ------------------------------------------------------------------
    # Test 2: Initial status
    # ------------------------------------------------------------------
    print("\n [Test 2] Initial status...")
    st = mgr.get_status()
    if not st["has_chain"] and st["strike_count"] == 0:
        print(f" ✅ Initial status correct: {st}")
        passed += 1
    else:
        print(f" ❌ Initial status wrong: {st}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 3: Invalid spot poll safe
    # ------------------------------------------------------------------
    print("\n [Test 3] Invalid spot poll safety...")
    result = mgr.poll(0)
    if result == {}:
        print(" ✅ Invalid spot safely skipped")
        passed += 1
    else:
        print(f" ❌ Unexpected result: {result}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 4: Snapshot history empty initially
    # ------------------------------------------------------------------
    print("\n [Test 4] Snapshot history...")
    hist = mgr.get_snapshot_history()
    if hist == []:
        print(" ✅ History empty initially")
        passed += 1
    else:
        print(f" ❌ History should be empty: {hist}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 5: ATM IV safe fallback
    # ------------------------------------------------------------------
    print("\n [Test 5] ATM IV fallback...")
    iv = mgr.get_atm_iv()
    if iv["avg_iv"] == 0.0:
        print(f" ✅ ATM IV fallback works: {iv}")
        passed += 1
    else:
        print(f" ❌ Unexpected ATM IV fallback: {iv}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 6: PCR fallback
    # ------------------------------------------------------------------
    print("\n [Test 6] PCR fallback...")
    pcr = mgr.get_pcr()
    if pcr["pcr_oi"] == 0.0:
        print(f" ✅ PCR fallback works: {pcr}")
        passed += 1
    else:
        print(f" ❌ Unexpected PCR fallback: {pcr}")
        failed += 1

    # ------------------------------------------------------------------
    # Test 7: Reset
    # ------------------------------------------------------------------
    print("\n [Test 7] Reset...")
    mgr.reset_daily()
    st2 = mgr.get_status()
    if not st2["has_chain"] and st2["snapshot_history_count"] == 0:
        print(" ✅ Reset works")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st2}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Option Chain Manager working perfectly!")
        print(" ✅ Missing roadmap file completed safely.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()