"""Read-only snapshot projection helpers for dashboard refresh tiers.

This module transforms raw snapshot dictionaries into panel-ready flat
structures for HOT/WARM/COLD update paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from dashboard.core.binary_frame import KIND_COLD, KIND_HOT, KIND_WARM


def _snapshot_timestamp(snap: Dict[str, Any]) -> str | None:
    """Return backend-provided freshness timestamp without inventing one.

    Pre-Week-9 stabilization: projections must not use wall-clock time as a
    fallback freshness signal.  If the backend did not publish timestamp truth,
    downstream UI must render stale/unknown rather than fake freshness.
    """
    for key in ("last_update_timestamp", "timestamp", "last_update"):
        value = snap.get(key) if isinstance(snap, dict) else None
        if isinstance(value, str) and value.strip():
            return value
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_mtf_candles(value: Any, *, max_points: int = 5000) -> Dict[str, List[Any]]:
    """Pass through prepared MTF candle arrays without aggregation.

    Pre-Week-8 stabilization (P0 chart integration fix): MtfChartPanel expects
    backend/projected truth under mtf_candles.  The dashboard must never build
    candles here, so this helper only copies lists supplied by the backend and
    clamps them to the chart performance window.
    """
    if not isinstance(value, Mapping):
        return {}
    out: Dict[str, List[Any]] = {}
    for key, raw in value.items():
        if isinstance(raw, list):
            out[str(key)] = list(raw)[-max_points:]
    return out

def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any, default: List[Any] | None = None) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    return list(default or [])


def _as_dict(value: Any, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return dict(default or {})


def _lookup_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _first_present(value: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        candidate = _lookup_path(value, path)
        if candidate is not None:
            return candidate
    return default


def project_hot_snapshot(snapshot: dict) -> dict:
    """Project a raw snapshot into HOT-tier UI fields."""
    snap = snapshot if isinstance(snapshot, dict) else {}
    timestamp = _snapshot_timestamp(snap)

    return {
        "system_state": str(snap.get("system_state", "UNKNOWN")),
        "mode": str(snap.get("mode", "ALERT")),
        "feed_health": str(snap.get("feed_health", "DOWN")),
        "data_quality_score": _to_float(snap.get("data_quality_score", 0.0), 0.0),
        "ticks_per_second": _to_float(snap.get("ticks_per_second", 0.0), 0.0),
        "feed_lag_ms": _to_float(snap.get("feed_lag_ms", 0.0), 0.0),
        "using_fallback": bool(snap.get("using_fallback", False)),
        "last_update_timestamp": timestamp,
        "timestamp": timestamp,
        "spot": _to_float(snap.get("spot", 0.0), 0.0),
        "previous_close": _to_float(snap.get("previous_close", 0.0), 0.0),
        "capital": _to_float(snap.get("capital", 0.0), 0.0),
        "daily_pnl": _to_float(snap.get("daily_pnl", 0.0), 0.0),
        "drawdown_pct": _to_float(snap.get("drawdown_pct", 0.0), 0.0),
        "trades_today": _to_int(snap.get("trades_today", 0), 0),
        "consecutive_losses": _to_int(snap.get("consecutive_losses", 0), 0),
        "tilt_score": _to_float(snap.get("tilt_score", 0.0), 0.0),
        "risk_state": str(snap.get("risk_state", "NORMAL")),
        "session_phase": str(snap.get("session_phase", "UNKNOWN")),
        "day_type": str(snap.get("day_type", "UNKNOWN")),
        # HOT tier stays intentionally thin.  Heavy feature/SMC/options maps are
        # WARM-tier fields; keeping them out of HOT protects the 200ms loop from
        # accidental payload growth as Week 9+ panels are added.
        "regime": str(snap.get("regime", "UNKNOWN")),
        "narrative_label": str(snap.get("narrative_label", "NEUTRAL")),
        "active_brains": _as_list(snap.get("active_brains", []))[:4],
        # Week 06 Global Vitals / Component Guard fields.
        # These are pass-through dashboard contract fields: the backend owns
        # computation; the dashboard only renders known values and degrades when
        # absent.
        "regime_transition_prob": _to_float(snap.get("regime_transition_prob", 0.0), 0.0),
        "component_guard_heavyweights": _as_list(
            snap.get("component_guard_heavyweights", snap.get("heavyweights", []))
        )[:5],
        "heavyweights": _as_list(snap.get("heavyweights", snap.get("component_guard_heavyweights", [])))[:5],
    }


def project_warm_snapshot(snapshot: dict) -> dict:
    """Project a raw snapshot into WARM-tier UI fields."""
    snap = snapshot if isinstance(snapshot, dict) else {}
    timestamp = _snapshot_timestamp(snap)

    options_summary = {
        "pcr_oi": _to_float(snap.get("pcr_oi", 0.0), 0.0),
        "atm_iv": _to_float(snap.get("atm_iv", 0.0), 0.0),
        "max_pain": _to_float(snap.get("max_pain", 0.0), 0.0),
        "highest_ce_oi_strike": _to_float(snap.get("highest_ce_oi_strike", 0.0), 0.0),
        "highest_pe_oi_strike": _to_float(snap.get("highest_pe_oi_strike", 0.0), 0.0),
    }

    smart_money_src = _as_dict(snap.get("smart_money_5m", None))
    if not smart_money_src:
        smart_money_src = _as_dict(snap.get("smart_money_15m", None))
    smart_money_summary = {
        "sm_direction_score": _to_float(smart_money_src.get("sm_direction_score", 0.0), 0.0),
        "total_fvgs": _to_int(smart_money_src.get("total_fvgs", 0), 0),
        "bullish_fvgs": _to_int(smart_money_src.get("bullish_fvgs", 0), 0),
        "bearish_fvgs": _to_int(smart_money_src.get("bearish_fvgs", 0), 0),
    }

    features = _as_dict(snap.get("features", snap.get("features_1m", {})))
    features_1m = _as_dict(snap.get("features_1m", features))
    options_features = _as_dict(snap.get("options_features", options_summary))
    smart_money = _as_dict(snap.get("smart_money", smart_money_src))

    return {
        "narrative_score": _to_float(snap.get("narrative_score", 0.0), 0.0),
        "narrative_label": str(snap.get("narrative_label", "NEUTRAL")),
        "regime": str(snap.get("regime", "UNKNOWN")),
        "regime_confidence": _to_float(snap.get("regime_confidence", 0.0), 0.0),
        "regime_transition_prob": _to_float(snap.get("regime_transition_prob", 0.0), 0.0),
        "session_phase": str(snap.get("session_phase", "UNKNOWN")),
        "day_type": str(snap.get("day_type", "UNKNOWN")),
        "day_personality": _as_dict(snap.get("day_personality", {"day_type": str(snap.get("day_type", "UNKNOWN"))})),
        "historical_match_score": _to_float(snap.get("historical_match_score", 0.0), 0.0),
        "session_memory": _as_dict(snap.get("session_memory", {})),
        "or_high": _maybe_float(snap.get("or_high")),
        "or_low": _maybe_float(snap.get("or_low")),
        "ib_high": _maybe_float(snap.get("ib_high")),
        "ib_low": _maybe_float(snap.get("ib_low")),
        "ib_width": _maybe_float(snap.get("ib_width")),
        "session_size_multiplier": _to_float(snap.get("session_size_multiplier", 1.0), 1.0),
        "active_brains": _as_list(snap.get("active_brains", [])),
        "brain_confidence": _as_dict(snap.get("brain_confidence", {})),
        "features": features,
        "features_1m": features_1m,
        "options_summary": options_summary,
        "options_features": options_features,
        "smart_money_summary": smart_money_summary,
        "smart_money": smart_money,
        "narrative_fit_factors": _as_dict(snap.get("narrative_fit_factors", {})),
        # Week 09 Volume Profile / Order Flow input contract.  These are
        # pass-through containers only; the dashboard must not derive CVD, POC,
        # VAH/VAL, imbalance, absorption, or exhaustion locally.
        "volume_profile": _as_dict(snap.get("volume_profile", {})),
        "session_volume_profile": _as_dict(snap.get("session_volume_profile", {})),
        "microstructure": _as_dict(snap.get("microstructure", {})),
        "order_flow": _as_dict(snap.get("order_flow", {})),
        "cvd": snap.get("cvd"),
        "imbalance": snap.get("imbalance"),
        "absorption_alerts": _as_list(snap.get("absorption_alerts", []))[-20:],
        "exhaustion_alerts": _as_list(snap.get("exhaustion_alerts", []))[-20:],
        "session_profile_anchors": _as_dict(snap.get("session_profile_anchors", {})),
        "poc": snap.get("poc"),
        "vah": snap.get("vah"),
        "val": snap.get("val"),
        # Week-7 MTF chart pass-through.  Backend owns candle preparation and
        # overlay computation; projection only preserves already-known fields.
        "mtf_candles": _as_mtf_candles(snap.get("mtf_candles", snap.get("candles_by_tf", {}))),
        "candles_by_tf": _as_mtf_candles(snap.get("candles_by_tf", snap.get("mtf_candles", {}))),
        "vwap_bands": _as_dict(snap.get("vwap_bands", {})),
        "or_levels": _as_dict(snap.get("or_levels", {})),
        "ib_levels": _as_dict(snap.get("ib_levels", {})),
        "active_timeframe": str(snap.get("active_timeframe", snap.get("timeframe", "5m"))),
        "timeframe": str(snap.get("timeframe", snap.get("active_timeframe", "5m"))),
        "timestamp": timestamp,
        "last_update_timestamp": timestamp,
    }


def project_cold_snapshot(snapshot: dict) -> dict:
    """Project a raw snapshot into COLD-tier UI fields.

    Issue #3 fix: Added feed_health, data_quality_score, ticks_per_second,
    feed_lag_ms, using_fallback, kill_switch_state, snapshot_age_seconds,
    and last_update_timestamp so SystemHealthPanelAdapter receives complete
    health diagnostic data instead of UNKNOWN defaults.
    """
    snap = snapshot if isinstance(snapshot, dict) else {}
    timestamp = _snapshot_timestamp(snap)

    explicit_snapshot_age = _maybe_float(_first_present(snap, "snapshot_age_seconds"))
    if explicit_snapshot_age is not None:
        snapshot_age_seconds = max(0.0, explicit_snapshot_age)
    else:
        # Derive snapshot_age_seconds from timestamp if available.
        snapshot_age_seconds = None
        if timestamp:
            try:
                from datetime import datetime, timezone

                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                snapshot_age_seconds = max(0.0, (now - ts).total_seconds())
            except Exception:
                snapshot_age_seconds = None

    return {
        "raw_opportunities": _as_list(snap.get("raw_opportunities", []))[-20:],
        "trapped_opportunities": _as_list(snap.get("trapped_opportunities", []))[-10:],
        "scored_opportunities": _as_list(snap.get("scored_opportunities", []))[-10:],
        "ml_filtered": _as_list(snap.get("ml_filtered", []))[-10:],
        "behavioral_filtered": _as_list(snap.get("behavioral_filtered", []))[-10:],
        "approved_opportunities": _as_list(snap.get("approved_opportunities", []))[-10:],
        "open_positions": _as_list(snap.get("open_positions", [])),
        "engine_health": _as_dict(snap.get("engine_health", {})),
        # Issue #3: Add health diagnostic fields for SystemHealthPanelAdapter
        "feed_health": str(_first_present(snap, "feed_health", default="UNKNOWN")),
        "data_quality_score": _to_float(
            _first_present(snap, "data_quality_score", "data_quality", default=0.0),
            0.0,
        ),
        "ticks_per_second": _to_float(snap.get("ticks_per_second", 0.0), 0.0),
        "feed_lag_ms": _to_float(snap.get("feed_lag_ms", 0.0), 0.0),
        "using_fallback": bool(snap.get("using_fallback", False)),
        "kill_switch_state": str(_first_present(snap, "kill_switch_state", default="UNKNOWN")),
        "snapshot_age_seconds": snapshot_age_seconds,
        "last_update_timestamp": timestamp,
        "timestamp": timestamp,
    }


def project_snapshot(snapshot: dict, kind: int) -> dict:
    if kind == KIND_HOT:
        return project_hot_snapshot(snapshot)
    if kind == KIND_WARM:
        return project_warm_snapshot(snapshot)
    if kind == KIND_COLD:
        return project_cold_snapshot(snapshot)
    return dict(snapshot) if isinstance(snapshot, dict) else {}


if __name__ == "__main__":
    dummy_snapshot = {
        "system_state": "ACTIVE",
        "mode": "PAPER",
        "feed_health": "HEALTHY",
        "data_quality_score": 91.5,
        "ticks_per_second": 122.0,
        "feed_lag_ms": 14.2,
        "using_fallback": False,
        "spot": 24670.5,
        "daily_pnl": 1520.0,
        "drawdown_pct": 0.45,
        "trades_today": 3,
        "consecutive_losses": 1,
        "tilt_score": 18.0,
        "risk_state": "NORMAL",
        "narrative_score": 62.0,
        "narrative_label": "RISK_ON",
        "regime": "TRENDING",
        "regime_confidence": 0.78,
        "regime_transition_prob": 0.12,
        "session_phase": "GOLDEN_MORNING",
        "day_type": "NORMAL",
        "or_high": 24710,
        "or_low": 24610,
        "ib_high": 24740,
        "ib_low": 24590,
        "ib_width": 150,
        "active_brains": ["structural", "institutional"],
        "features": {"rsi": 58.1, "atr": 21.3, "vwap": 24650.2},
        "features_1m": {"rsi": 58.1, "atr": 21.3, "vwap": 24650.2},
        "pcr_oi": 1.08,
        "atm_iv": 0.14,
        "max_pain": 24600,
        "highest_ce_oi_strike": 24800,
        "highest_pe_oi_strike": 24500,
        "smart_money_5m": {"sm_direction_score": 14.0, "total_fvgs": 3, "bullish_fvgs": 2, "bearish_fvgs": 1},
        "raw_opportunities": [{"id": i} for i in range(30)],
        "trapped_opportunities": [{"id": i} for i in range(15)],
        "scored_opportunities": [{"id": i} for i in range(12)],
        "ml_filtered": [{"id": i} for i in range(11)],
        "behavioral_filtered": [{"id": i} for i in range(13)],
        "approved_opportunities": [{"id": i} for i in range(16)],
        "open_positions": [{"symbol": "NIFTY", "qty": 1}],
        "engine_health": {"data": "ok", "captain": "ok"},
    }

    hot = project_hot_snapshot(dummy_snapshot)
    warm = project_warm_snapshot(dummy_snapshot)
    cold = project_cold_snapshot(dummy_snapshot)

    print("HOT keys:", sorted(hot.keys())[:5], "...")
    print("WARM keys:", sorted(warm.keys())[:5], "...")
    print("COLD keys:", sorted(cold.keys())[:5], "...")

    assert "system_state" in hot
    assert "feed_health" in hot
    assert hot["data_quality_score"] is not None
    assert hot["ticks_per_second"] is not None
    assert "options_summary" in warm
    assert "smart_money_summary" in warm
    assert "features" in warm
    assert "brain_confidence" in warm
    assert isinstance(cold["raw_opportunities"], list)
    assert isinstance(cold["engine_health"], dict)

    print("state_projection self-test passed")