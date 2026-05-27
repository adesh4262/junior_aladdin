"""
Junior Aladdin - Market DNA Fingerprint Engine (Hardened Version)
=================================================================
PURPOSE:
Compute the personality of the current day, match it to similar historical
days, and maintain robust session memory.

This hardened version improves:
- stricter fingerprint normalization
- safer DTE handling
- stronger malformed-history protection
- cleaner session-memory updates
- deterministic similarity behavior

FINGERPRINT COMPONENTS:
1. gap_direction
2. gap_magnitude
3. gap_fill_status
4. ib_width
5. ib_direction
6. opening_volume_ratio
7. vix_change_pct
8. fii_flow_direction
9. day_of_week
10. days_to_expiry
11. global_cue_direction
12. regime_at_1015

CONNECTS TO:
- Key Levels
- Fundamental Features
- Regime Engine
- Time Context
- Database
- Captain / Scoring / Trap layers
"""

import math
import json
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

from src.utils.logger import setup_logger
from src.utils.helpers import ist_today, days_to_expiry
from src.utils.database import Database

_logger = setup_logger("market_dna")


class MarketDNAEngine:
    """
    Hardened Market DNA engine with:
    - deterministic fingerprint creation
    - safe historical matching
    - session-memory tracking
    """

    def __init__(self):
        self._fingerprint: Optional[Dict] = None
        self._fingerprint_computed: bool = False
        self._similar_days: List[Dict] = []
        self._historical_match_score: float = 0.0
        self._session_memory: Dict = self._empty_session_memory()

        _logger.info("Market DNA Engine initialized")

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------
    def compute_fingerprint(
        self,
        key_levels: Optional[Dict] = None,
        fundamental_data: Optional[Dict] = None,
        regime: str = "UNKNOWN",
        gap_pct: float = 0.0,
        previous_close: float = 0.0,
        opening_price: float = 0.0,
        current_price: float = 0.0,
        opening_volume_ratio: float = 1.0,
    ) -> Dict:
        """
        Compute normalized day fingerprint.
        Should be called around 10:15 AM after IB forms.
        """
        today = ist_today()

        # 1. gap direction
        if gap_pct > 0.1:
            gap_direction = 1
        elif gap_pct < -0.1:
            gap_direction = -1
        else:
            gap_direction = 0

        # 2. gap magnitude
        gap_magnitude = round(abs(gap_pct), 4)

        # 3. gap fill status
        gap_fill_status = self._compute_gap_fill_status(
            gap_direction=gap_direction,
            previous_close=previous_close,
            opening_price=opening_price,
            current_price=current_price,
        )

        # 4,5 IB
        ib_width = 0.0
        ib_direction = 0
        if key_levels:
            ib_width = float(key_levels.get("ib_width", 0.0) or 0.0)
            ib_direction = int(key_levels.get("ib_direction", 0) or 0)

        # 6 opening volume ratio
        vol_ratio = round(max(0.0, float(opening_volume_ratio or 0.0)), 2)

        # 7 VIX change
        vix_change = 0.0
        if fundamental_data:
            vix_change = float(fundamental_data.get("vix_change_pct", 0.0) or 0.0)

        # 8 FII flow direction
        fii_direction = 0
        if fundamental_data:
            fii_score = float(fundamental_data.get("fii_score", 0) or 0)
            if fii_score > 30:
                fii_direction = 1
            elif fii_score < -30:
                fii_direction = -1

        # 9 weekday
        day_of_week = today.weekday()

        # 10 DTE (never negative in fingerprint)
        dte = max(0, int(days_to_expiry(today)))

        # 11 global cue direction
        global_direction = 0
        if fundamental_data:
            global_score = float(fundamental_data.get("global_score", 0) or 0)
            if global_score > 15:
                global_direction = 1
            elif global_score < -15:
                global_direction = -1

        # 12 regime
        regime_at_1015 = str(regime or "UNKNOWN")

        self._fingerprint = {
            "date": today.strftime("%Y-%m-%d"),
            "gap_direction": gap_direction,
            "gap_magnitude": gap_magnitude,
            "gap_fill_status": gap_fill_status,
            "ib_width": round(ib_width, 2),
            "ib_direction": ib_direction,
            "opening_volume_ratio": vol_ratio,
            "vix_change_pct": round(vix_change, 2),
            "fii_flow_direction": fii_direction,
            "day_of_week": day_of_week,
            "days_to_expiry": dte,
            "global_cue_direction": global_direction,
            "regime_at_1015": regime_at_1015,
        }

        self._fingerprint_computed = True

        _logger.info(
            "Day fingerprint computed",
            extra={
                "gap": f"{gap_direction}({gap_magnitude:.2%})",
                "ib_width": ib_width,
                "regime": regime_at_1015,
                "dte": dte,
            },
        )

        self._find_similar_days()
        return self._fingerprint

    def _compute_gap_fill_status(
        self,
        gap_direction: int,
        previous_close: float,
        opening_price: float,
        current_price: float,
    ) -> str:
        if previous_close <= 0 or opening_price <= 0 or current_price <= 0:
            return "unknown"

        if gap_direction == 1:
            if current_price <= previous_close:
                return "filled"
            elif current_price < opening_price:
                return "partial"
            return "unfilled"

        if gap_direction == -1:
            if current_price >= previous_close:
                return "filled"
            elif current_price > opening_price:
                return "partial"
            return "unfilled"

        return "no_gap"

    # ------------------------------------------------------------------
    # Historical matching
    # ------------------------------------------------------------------
    def _find_similar_days(self):
        if not self._fingerprint:
            return

        try:
            db = Database()
            rows = db.fetch_all_as_dicts(
                "SELECT date, fingerprint_json, day_type, session_pnl_json "
                "FROM day_fingerprints ORDER BY date DESC LIMIT 180"
            )
            db.close()
        except Exception as e:
            _logger.debug(f"No historical fingerprints available: {e}")
            self._similar_days = []
            self._historical_match_score = 0.0
            return

        if not rows:
            self._similar_days = []
            self._historical_match_score = 0.0
            return

        current_vec = self._fingerprint_to_vector(self._fingerprint)
        similarities: List[Dict] = []

        for row in rows:
            try:
                hist_fp = json.loads(row.get("fingerprint_json", "{}"))
                if not isinstance(hist_fp, dict):
                    continue

                hist_vec = self._fingerprint_to_vector(hist_fp)
                sim = self._cosine_similarity(current_vec, hist_vec)

                similarities.append(
                    {
                        "date": row.get("date", ""),
                        "similarity": round(sim, 4),
                        "day_type": row.get("day_type", "UNKNOWN"),
                        "fingerprint": hist_fp,
                    }
                )
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        self._similar_days = similarities[:10]
        self._historical_match_score = (
            self._similar_days[0]["similarity"] if self._similar_days else 0.0
        )

    def _fingerprint_to_vector(self, fp: Dict) -> List[float]:
        """
        Convert fingerprint into normalized numeric vector.
        """
        regime_map = {
            "TRENDING": 1.0,
            "RANGE": 0.0,
            "VOLATILE": -1.0,
            "CHOP": -0.5,
            "EVENT": 0.5,
            "UNKNOWN": 0.0,
        }

        gap_fill = fp.get("gap_fill_status", "unknown")
        gap_fill_num = {
            "unfilled": 1.0,
            "partial": 0.5,
            "filled": 0.0,
            "no_gap": 0.0,
            "unknown": 0.0,
        }.get(gap_fill, 0.0)

        return [
            float(fp.get("gap_direction", 0)),
            float(fp.get("gap_magnitude", 0)) * 100.0,
            gap_fill_num,
            float(fp.get("ib_width", 0)) / 50.0,
            float(fp.get("ib_direction", 0)),
            float(fp.get("opening_volume_ratio", 1.0)),
            float(fp.get("vix_change_pct", 0)) / 5.0,
            float(fp.get("fii_flow_direction", 0)),
            float(fp.get("day_of_week", 0)) / 4.0,
            float(max(0, int(fp.get("days_to_expiry", 0)))) / 5.0,
            float(fp.get("global_cue_direction", 0)),
            regime_map.get(str(fp.get("regime_at_1015", "UNKNOWN")), 0.0),
        ]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    # ------------------------------------------------------------------
    # Session memory
    # ------------------------------------------------------------------
    def update_session_memory(self, event_type: str, data: Optional[Dict] = None):
        data = data or {}
        mem = self._session_memory

        if event_type == "LEVEL_DEFENDED":
            level = data.get("level", 0)
            if level > 0 and level not in mem["levels_defended"]:
                mem["levels_defended"].append(level)

        elif event_type == "LEVEL_BROKEN":
            level = data.get("level", 0)
            if level > 0 and level not in mem["levels_broken"]:
                mem["levels_broken"].append(level)

        elif event_type == "FAILED_BREAKOUT":
            level = data.get("level", 0)
            if level > 0:
                mem["failed_breakouts"].append(level)
            mem["traps_detected"] += 1

        elif event_type == "TRAP_DETECTED":
            mem["traps_detected"] += 1

        elif event_type == "DIRECTION_SET":
            mem["dominant_direction_morning"] = data.get("direction", "")

        elif event_type == "MOMENTUM_DECAY":
            mem["momentum_decay_started"] = True

        elif event_type == "LARGE_MOVE":
            size = float(data.get("size", 0) or 0)
            if size > mem["largest_move_size"]:
                mem["largest_move_size"] = size

        elif event_type == "VP_SHIFT":
            mem["volume_profile_shift"] = data.get("direction", "STABLE")

    def get_session_memory(self) -> Dict:
        return self._session_memory.copy()

    def get_trap_penalty(self) -> int:
        fb_count = len(self._session_memory["failed_breakouts"])
        if fb_count >= 2:
            return 15
        if fb_count >= 1:
            return 8
        return 0

    def get_defended_level_bonus(self, price: float, tolerance: float = 5.0) -> int:
        for level in self._session_memory["levels_defended"]:
            if abs(price - level) <= tolerance:
                return 15
        return 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def store_fingerprint(
        self,
        day_type: str = "UNKNOWN",
        session_pnl: Optional[Dict] = None,
    ):
        if not self._fingerprint:
            return

        try:
            db = Database()
            db.execute(
                """
                INSERT OR REPLACE INTO day_fingerprints
                (date, fingerprint_json, day_type, session_pnl_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._fingerprint["date"],
                    json.dumps(self._fingerprint),
                    day_type,
                    json.dumps(session_pnl or {}),
                ),
            )
            db.close()
            _logger.info("Fingerprint stored in database")
        except Exception as e:
            _logger.error(f"Failed to store fingerprint: {e}")

    # ------------------------------------------------------------------
    # Status / reset
    # ------------------------------------------------------------------
    def get_fingerprint(self) -> Optional[Dict]:
        return self._fingerprint

    def get_similar_days(self) -> List[Dict]:
        return self._similar_days

    def get_historical_match_score(self) -> float:
        return self._historical_match_score

    def get_status(self) -> Dict:
        return {
            "fingerprint_computed": self._fingerprint_computed,
            "similar_days_count": len(self._similar_days),
            "historical_match_score": self._historical_match_score,
            "traps_today": self._session_memory["traps_detected"],
            "levels_defended": len(self._session_memory["levels_defended"]),
            "failed_breakouts": len(self._session_memory["failed_breakouts"]),
        }

    def reset_daily(self):
        self._fingerprint = None
        self._fingerprint_computed = False
        self._similar_days = []
        self._historical_match_score = 0.0
        self._session_memory = self._empty_session_memory()
        _logger.info("Market DNA Engine reset")

    def _empty_session_memory(self) -> Dict:
        return {
            "levels_defended": [],
            "levels_broken": [],
            "failed_breakouts": [],
            "traps_detected": 0,
            "dominant_direction_morning": "",
            "momentum_decay_started": False,
            "largest_move_size": 0.0,
            "volume_profile_shift": "STABLE",
        }


def _run_tests():
    print("=" * 60)
    print(" JUNIOR ALADDIN — Market DNA Engine Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    dna = MarketDNAEngine()

    print(" [Test 1] Create engine...")
    if dna is not None:
        print(" ✅ Engine created")
        passed += 1
    else:
        print(" ❌ Engine creation failed")
        failed += 1

    print("\n [Test 2] Compute fingerprint...")
    fp = dna.compute_fingerprint(
        key_levels={"ib_width": 85, "ib_direction": 1},
        fundamental_data={
            "fii_score": 60,
            "global_score": 20,
            "vix_change_pct": -1.5,
        },
        regime="TRENDING",
        gap_pct=0.6,
        previous_close=23200,
        opening_price=23340,
        current_price=23380,
        opening_volume_ratio=1.3,
    )
    expected_keys = [
        "gap_direction",
        "gap_magnitude",
        "gap_fill_status",
        "ib_width",
        "ib_direction",
        "opening_volume_ratio",
        "vix_change_pct",
        "fii_flow_direction",
        "day_of_week",
        "days_to_expiry",
        "global_cue_direction",
        "regime_at_1015",
    ]
    missing = [k for k in expected_keys if k not in fp]
    if not missing:
        print(" ✅ All 12 fingerprint fields present")
        passed += 1
    else:
        print(f" ❌ Missing keys: {missing}")
        failed += 1

    print("\n [Test 3] Session memory updates...")
    dna.update_session_memory("LEVEL_DEFENDED", {"level": 23300})
    dna.update_session_memory("FAILED_BREAKOUT", {"level": 23450})
    dna.update_session_memory("FAILED_BREAKOUT", {"level": 23460})
    mem = dna.get_session_memory()
    if len(mem["levels_defended"]) == 1 and mem["traps_detected"] == 2:
        print(f" ✅ Session memory works: {mem}")
        passed += 1
    else:
        print(f" ❌ Session memory failed: {mem}")
        failed += 1

    print("\n [Test 4] Trap penalty...")
    if dna.get_trap_penalty() == 15:
        print(" ✅ Trap penalty works")
        passed += 1
    else:
        print(" ❌ Trap penalty failed")
        failed += 1

    print("\n [Test 5] Defended level bonus...")
    if dna.get_defended_level_bonus(23302, tolerance=5) == 15:
        print(" ✅ Defended level bonus works")
        passed += 1
    else:
        print(" ❌ Defended bonus failed")
        failed += 1

    print("\n [Test 6] Cosine similarity...")
    same = dna._cosine_similarity([1, 2, 3], [1, 2, 3])
    if abs(same - 1.0) < 0.001:
        print(" ✅ Cosine similarity works")
        passed += 1
    else:
        print(" ❌ Cosine similarity failed")
        failed += 1

    print("\n [Test 7] Reset...")
    dna.reset_daily()
    st = dna.get_status()
    if not st["fingerprint_computed"] and st["traps_today"] == 0:
        print(f" ✅ Reset works: {st}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Market DNA Engine (Hardened) working perfectly!")
        print(" ✅ Ready for next hardening step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    from src.utils.database import Database
    _run_tests()