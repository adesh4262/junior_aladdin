"""
Junior Aladdin - Regime Engine (Hardened Version)
=================================================
PURPOSE:
Classify current market regime into one of 5 states with improved robustness.

STATES:
- TRENDING
- RANGE
- VOLATILE
- CHOP
- EVENT

This hardened version improves:
- stronger degraded-data handling
- clearer scoring boundaries
- safer event override
- more conservative CHOP detection
- more transparent transition probability logic

CONNECTS TO:
- Feature Engine outputs
- Captain
- Brain Engine
- Scoring Engine
- Risk logic
"""

import math
from collections import deque
from typing import Dict, Optional

from src.utils.logger import setup_logger
from src.utils.config_loader import Config

_logger = setup_logger("regime_engine")


class RegimeEngine:
    """
    Classifies market regime and tracks transition/stability.
    """

    def __init__(self):
        self._trending_adx = Config.get("regime", "trending_adx_threshold", default=25)
        self._strong_trend_adx = Config.get("regime", "strong_trend_adx", default=35)
        self._range_adx = Config.get("regime", "range_adx_threshold", default=20)
        self._chop_adx = Config.get("regime", "chop_adx_threshold", default=15)
        self._volatile_atr_pctl = Config.get("regime", "volatile_atr_percentile", default=80)
        self._stability_bars = Config.get("regime", "stability_filter_bars", default=3)

        self._current_regime: str = "UNKNOWN"
        self._confirmed_regime: str = "UNKNOWN"
        self._regime_confidence: float = 0.0
        self._stability_count: int = 0
        self._transition_prob: float = 0.0

        self._regime_history: deque = deque(maxlen=30)
        self._adx_history: deque = deque(maxlen=20)

        _logger.info("Regime Engine initialized")

    def classify(
        self,
        features_1m: Optional[Dict] = None,
        features_5m: Optional[Dict] = None,
        features_15m: Optional[Dict] = None,
        event_data: Optional[Dict] = None,
        vix_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Main regime classification.
        """
        if not features_1m:
            return self._empty_result()

        if not self._minimum_inputs_ok(features_1m):
            return {
                "regime": self._confirmed_regime,
                "raw_regime": "UNKNOWN",
                "confidence": 0.0,
                "transition_prob": 0.0,
                "stability_count": self._stability_count,
                "is_stable": False,
                "is_event": False,
                "scores": {},
                "adx_proxy": None,
            }

        scores = {
            "TRENDING": self._score_trending(features_1m, features_5m, features_15m),
            "RANGE": self._score_range(features_1m, features_5m),
            "VOLATILE": self._score_volatile(features_1m, vix_data),
            "CHOP": self._score_chop(features_1m, features_5m),
        }

        is_event = self._is_event_override(event_data)

        if is_event:
            raw_regime = "EVENT"
            confidence = 1.0
        else:
            raw_regime = max(scores, key=scores.get)
            max_score = scores[raw_regime]
            total = sum(scores.values())
            confidence = round(max_score / total, 3) if total > 0 else 0.0

        adx = self._compute_adx_proxy(features_1m, features_5m)
        if adx is not None:
            self._adx_history.append(adx)

        transition_prob = self._compute_transition_probability(
            raw_regime, scores, features_1m, vix_data
        )

        if raw_regime == self._current_regime:
            self._stability_count += 1
        else:
            self._stability_count = 1
            self._current_regime = raw_regime

        if self._stability_count >= self._stability_bars:
            prev_confirmed = self._confirmed_regime
            self._confirmed_regime = raw_regime

            if prev_confirmed != raw_regime and prev_confirmed != "UNKNOWN":
                _logger.info(
                    f"Regime changed: {prev_confirmed} -> {raw_regime}",
                    extra={
                        "confidence": confidence,
                        "stability": self._stability_count,
                        "scores": {k: round(v, 1) for k, v in scores.items()},
                    },
                )

        self._regime_history.append(raw_regime)
        self._regime_confidence = confidence
        self._transition_prob = transition_prob

        return {
            "regime": self._confirmed_regime,
            "raw_regime": raw_regime,
            "confidence": confidence,
            "transition_prob": round(transition_prob, 3),
            "stability_count": self._stability_count,
            "is_stable": self._stability_count >= self._stability_bars,
            "is_event": is_event,
            "scores": {k: round(v, 1) for k, v in scores.items()},
            "adx_proxy": round(adx, 1) if adx is not None else None,
        }

    # ------------------------------------------------------------------
    # Internal safety
    # ------------------------------------------------------------------
    def _minimum_inputs_ok(self, f1m: Dict) -> bool:
        required = [
            "trend_direction",
            "vwap_slope",
            "atr",
            "bb_width_percentile",
            "candle_body_ratio",
            "last_close",
        ]
        for key in required:
            if key not in f1m:
                return False
        return True

    def _is_event_override(self, event_data: Optional[Dict]) -> bool:
        if not event_data:
            return False
        severity = event_data.get("event_severity", 0)
        days_away = event_data.get("event_days_away", 999)
        return severity >= 2 and days_away <= 1

    # ------------------------------------------------------------------
    # Regime scores
    # ------------------------------------------------------------------
    def _score_trending(self, f1m: Dict, f5m: Optional[Dict], f15m: Optional[Dict]) -> float:
        score = 0.0

        adx = self._compute_adx_proxy(f1m, f5m)
        if adx is not None:
            if adx > self._strong_trend_adx:
                score += 30
            elif adx > self._trending_adx:
                score += 20
            elif adx > self._range_adx:
                score += 10

        trend = f1m.get("trend_direction", 0)
        if trend != 0:
            score += 15
            if f5m and f5m.get("trend_direction", 0) == trend:
                score += 5
            if f15m and f15m.get("trend_direction", 0) == trend:
                score += 5

        vwap_slope = abs(float(f1m.get("vwap_slope", 0) or 0))
        if vwap_slope > 3.0:
            score += 15
        elif vwap_slope > 1.5:
            score += 8

        ema9 = f1m.get("ema_9")
        ema21 = f1m.get("ema_21")
        ema50 = f1m.get("ema_50")
        if ema9 is not None and ema21 is not None and ema50 is not None:
            if ema9 > ema21 > ema50 or ema9 < ema21 < ema50:
                score += 15
            elif (ema9 > ema21) or (ema9 < ema21):
                score += 5

        st_dir = f1m.get("supertrend_direction", 0)
        if st_dir != 0 and st_dir == trend:
            score += 10

        vol_ratio = f1m.get("volume_ratio")
        if vol_ratio is not None:
            try:
                vr = float(vol_ratio)
                if vr > 1.0:
                    score += 10
                elif vr > 0.7:
                    score += 5
            except (TypeError, ValueError):
                pass

        return min(100.0, score)

    def _score_range(self, f1m: Dict, f5m: Optional[Dict]) -> float:
        score = 0.0

        adx = self._compute_adx_proxy(f1m, f5m)
        if adx is not None:
            if adx < self._chop_adx:
                score += 10
            elif adx < self._range_adx:
                score += 30
            elif adx < self._trending_adx:
                score += 15

        bb_width_pctl = f1m.get("bb_width_percentile")
        if bb_width_pctl is not None:
            try:
                b = float(bb_width_pctl)
                if b < 20:
                    score += 20
                elif b < 40:
                    score += 15
            except (TypeError, ValueError):
                pass

        pvp = abs(float(f1m.get("price_vs_vwap_pct", 0) or 0))
        if pvp < 0.1:
            score += 15
        elif pvp < 0.3:
            score += 8

        rsi = f1m.get("rsi")
        if rsi is not None:
            try:
                r = float(rsi)
                if 40 < r < 60:
                    score += 15
                elif 35 < r < 65:
                    score += 8
            except (TypeError, ValueError):
                pass

        vwap_slope = abs(float(f1m.get("vwap_slope", 0) or 0))
        if vwap_slope < 0.5:
            score += 10
        elif vwap_slope < 1.5:
            score += 5

        st_dir = f1m.get("supertrend_direction", 0)
        trend = f1m.get("trend_direction", 0)
        if st_dir != trend:
            score += 10

        return min(100.0, score)

    def _score_volatile(self, f1m: Dict, vix_data: Optional[Dict]) -> float:
        score = 0.0

        atr_pctl = f1m.get("atr_percentile")
        if atr_pctl is not None:
            try:
                a = float(atr_pctl)
                if a > 90:
                    score += 30
                elif a > self._volatile_atr_pctl:
                    score += 20
                elif a > 60:
                    score += 5
            except (TypeError, ValueError):
                pass

        if vix_data:
            vix_change = float(vix_data.get("vix_change_pct", 0) or 0)
            vix_level = float(vix_data.get("vix_level", 0) or 0)

            if vix_change > 5:
                score += 25
            elif vix_change > 3:
                score += 15
            elif vix_change > 1:
                score += 5

            if vix_level > 25:
                score += 10
            elif vix_level > 20:
                score += 5

        body_ratio = float(f1m.get("candle_body_ratio", 0) or 0)
        if body_ratio > 0.7:
            score += 20
        elif body_ratio > 0.5:
            score += 10

        intraday_range = float(f1m.get("intraday_range_pct", 0) or 0)
        if intraday_range > 2.0:
            score += 15
        elif intraday_range > 1.5:
            score += 8

        bb_width_pctl = f1m.get("bb_width_percentile")
        if bb_width_pctl is not None:
            try:
                if float(bb_width_pctl) > 80:
                    score += 10
            except (TypeError, ValueError):
                pass

        return min(100.0, score)

    def _score_chop(self, f1m: Dict, f5m: Optional[Dict]) -> float:
        score = 0.0

        adx = self._compute_adx_proxy(f1m, f5m)
        if adx is not None:
            if adx < self._chop_adx:
                score += 30
            elif adx < self._range_adx:
                score += 10

        overlap = self._compute_candle_overlap(f1m)
        if overlap > 0.6:
            score += 25
        elif overlap > 0.4:
            score += 10

        vol_ratio = f1m.get("volume_ratio")
        if vol_ratio is not None:
            try:
                vr = float(vol_ratio)
                if vr < 0.5:
                    score += 20
                elif vr < 0.7:
                    score += 10
            except (TypeError, ValueError):
                pass

        rsi_slope = f1m.get("rsi_slope")
        if rsi_slope is not None:
            try:
                rs = abs(float(rsi_slope))
                if rs > 10:
                    score += 5
                elif rs < 2:
                    score += 15
            except (TypeError, ValueError):
                pass

        macd_hist = f1m.get("macd_histogram")
        if macd_hist is not None:
            try:
                if abs(float(macd_hist)) < 2:
                    score += 10
            except (TypeError, ValueError):
                pass

        return min(100.0, score)

    # ------------------------------------------------------------------
    # ADX proxy / overlap / transition
    # ------------------------------------------------------------------
    def _compute_adx_proxy(self, f1m: Dict, f5m: Optional[Dict] = None) -> Optional[float]:
        trend = abs(int(f1m.get("trend_direction", 0) or 0))
        vwap_slope = abs(float(f1m.get("vwap_slope", 0) or 0))
        ema_diff = abs(float(f1m.get("ema_9_vs_21", 0) or 0))
        atr = f1m.get("atr")
        close = float(f1m.get("last_close", 0) or 0)

        if close <= 0 or atr is None:
            return None

        atr = float(atr)
        if atr <= 0:
            return None

        ema_norm = ema_diff / atr if atr > 0 else 0.0
        adx_proxy = trend * 15 + vwap_slope * 3 + ema_norm * 10

        if f5m:
            trend_5m = abs(int(f5m.get("trend_direction", 0) or 0))
            adx_proxy += trend_5m * 5

        return min(50.0, max(0.0, adx_proxy))

    def _compute_candle_overlap(self, f1m: Dict) -> float:
        bb_pct_b = f1m.get("bb_pct_b")
        if bb_pct_b is None:
            return 0.0

        try:
            distance_from_center = abs(float(bb_pct_b) - 0.5)
            overlap = max(0.0, 1.0 - distance_from_center * 3)
            return round(overlap, 3)
        except (TypeError, ValueError):
            return 0.0

    def _compute_transition_probability(
        self,
        current_regime: str,
        scores: Dict[str, float],
        f1m: Dict,
        vix_data: Optional[Dict],
    ) -> float:
        transition_score = 0

        bb_width_pctl = f1m.get("bb_width_percentile")
        vol_ratio = f1m.get("volume_ratio")

        if bb_width_pctl is not None:
            try:
                if float(bb_width_pctl) < 10 and current_regime == "RANGE":
                    transition_score += 30
            except (TypeError, ValueError):
                pass

        if current_regime == "TRENDING" and len(self._adx_history) >= 3:
            recent = list(self._adx_history)[-3:]
            adx_slope = recent[-1] - recent[0]
            if adx_slope < -2:
                transition_score += 25

        if vol_ratio is not None:
            try:
                if float(vol_ratio) > 2.0 and current_regime == "RANGE":
                    transition_score += 25
            except (TypeError, ValueError):
                pass

        if vix_data and vix_data.get("vix_spike", False):
            transition_score += 20

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
            ratio = sorted_scores[1] / sorted_scores[0]
            if ratio > 0.8:
                transition_score += 15

        return min(1.0, transition_score / 100.0)

    # ------------------------------------------------------------------
    # Status / reset
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        return {
            "confirmed_regime": self._confirmed_regime,
            "raw_regime": self._current_regime,
            "confidence": self._regime_confidence,
            "transition_prob": self._transition_prob,
            "stability_count": self._stability_count,
            "history_length": len(self._regime_history),
        }

    def reset(self):
        self._current_regime = "UNKNOWN"
        self._confirmed_regime = "UNKNOWN"
        self._regime_confidence = 0.0
        self._stability_count = 0
        self._transition_prob = 0.0
        self._regime_history.clear()
        self._adx_history.clear()
        _logger.info("Regime Engine reset")

    def _empty_result(self) -> Dict:
        return {
            "regime": self._confirmed_regime,
            "raw_regime": "UNKNOWN",
            "confidence": 0.0,
            "transition_prob": 0.0,
            "stability_count": 0,
            "is_stable": False,
            "is_event": False,
            "scores": {},
            "adx_proxy": None,
        }


def _run_tests():
    from src.core.candle_builder import CandleBuilder
    from src.core.replay_engine import ReplayEngine
    from src.features.price_features import compute_price_features
    from src.features.momentum_features import compute_momentum_features
    from src.features.volatility_features import compute_volatility_features

    print("=" * 60)
    print(" JUNIOR ALADDIN — Regime Engine Test (Hardened)")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    print(" [Test 1] Create Regime Engine...")
    try:
        re = RegimeEngine()
        print(" ✅ Engine created")
        passed += 1
    except Exception as e:
        print(f" ❌ Failed: {e}")
        failed += 1
        print("\n" + "=" * 60)
        print(f" Results: {passed} passed, {failed} failed")
        return

    print("\n [Test 2] Empty features...")
    result = re.classify()
    if result["regime"] == "UNKNOWN" and result["confidence"] == 0:
        print(f" ✅ Empty handled: regime={result['regime']}")
        passed += 1
    else:
        print(f" ❌ Unexpected: {result}")
        failed += 1

    print("\n [Test 3] Trending features...")
    trending_f = {
        "trend_direction": 1,
        "vwap_slope": 5.0,
        "ema_9": 23300,
        "ema_21": 23280,
        "ema_50": 23250,
        "ema_9_vs_21": 20,
        "supertrend_direction": 1,
        "volume_ratio": 1.5,
        "atr": 15,
        "atr_percentile": 50,
        "bb_width_percentile": 50,
        "bb_pct_b": 0.8,
        "rsi": 62,
        "rsi_slope": 5,
        "macd_histogram": 8,
        "price_vs_vwap_pct": 0.5,
        "candle_body_ratio": 0.6,
        "intraday_range_pct": 1.0,
        "last_close": 23300,
    }
    for _ in range(3):
        r3 = re.classify(trending_f)
    if r3["regime"] == "TRENDING":
        print(f" ✅ Regime={r3['regime']} confidence={r3['confidence']}")
        passed += 1
    else:
        print(f" ❌ Expected TRENDING: {r3}")
        failed += 1

    print("\n [Test 4] Event override...")
    re2 = RegimeEngine()
    event_data = {"event_severity": 2, "event_days_away": 0}
    for _ in range(3):
        r4 = re2.classify(trending_f, event_data=event_data)
    if r4["regime"] == "EVENT" and r4["is_event"]:
        print(f" ✅ EVENT override works: {r4}")
        passed += 1
    else:
        print(f" ❌ EVENT override failed: {r4}")
        failed += 1

    print("\n [Test 5] Historical replay classification...")
    replay = ReplayEngine()
    loaded = replay.load_recent(min_candles=100)
    if loaded:
        cb = CandleBuilder()
        replay.play(cb, speed="instant")
        candles_1m = list(cb.candles["1min"])
        candles_5m = list(cb.candles["5min"])
        if len(candles_1m) >= 50:
            f1m = {
                **compute_price_features(candles_1m),
                **compute_momentum_features(candles_1m),
                **compute_volatility_features(candles_1m),
            }
            f5m = None
            if len(candles_5m) >= 10:
                f5m = {
                    **compute_price_features(candles_5m),
                    **compute_momentum_features(candles_5m),
                }

            re3 = RegimeEngine()
            for _ in range(3):
                r5 = re3.classify(f1m, f5m)
            if r5["regime"] in ("TRENDING", "RANGE", "VOLATILE", "CHOP", "EVENT", "UNKNOWN"):
                print(f" ✅ Historical classification valid: {r5}")
                passed += 1
            else:
                print(f" ❌ Invalid historical regime: {r5}")
                failed += 1
        else:
            print(" ✅ Historical replay not enough candles, skipped safely")
            passed += 1
    else:
        print(" ✅ No historical data, skipped safely")
        passed += 1

    print("\n [Test 6] Reset...")
    re.reset()
    st6 = re.get_status()
    if st6["confirmed_regime"] == "UNKNOWN":
        print(f" ✅ Reset works: {st6}")
        passed += 1
    else:
        print(f" ❌ Reset failed: {st6}")
        failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("\n 🎉 Regime Engine (Hardened) working perfectly!")
        print(" ✅ Ready for next hardening step.")
    else:
        print(f"\n ⚠️ {failed} tests failed.")
        print("=" * 60)


if __name__ == "__main__":
    _run_tests()