"""
Default dashboard panel catalog.

The registry contract lives in dashboard.panels.__init__.
This module only provides concrete panels and the default registry builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from . import PanelBase, PanelRegistry, PanelResult, PanelStatus


def _compact(value: Any, *, max_items: int = 8, depth: int = 2) -> Any:
    if depth < 0:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        items = list(value.items())
        if len(items) <= max_items:
            return {str(k): _compact(v, max_items=max_items, depth=depth - 1) for k, v in items}
        return {
            "size": len(items),
            "keys": [str(k) for k, _ in items[:max_items]],
            "sample": {str(k): _compact(v, max_items=max_items, depth=depth - 1) for k, v in items[:max_items]},
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) <= max_items:
            return [_compact(v, max_items=max_items, depth=depth - 1) for v in items]
        return {
            "size": len(items),
            "sample": [_compact(v, max_items=max_items, depth=depth - 1) for v in items[:max_items]],
        }
    return str(value)


@dataclass
class _SimplePanel(PanelBase):
	panel_id_value: str
	title_value: str
	priority_value: int
	tags_value: List[str]
	required_keys_value: List[str]
	section_key: str
	section_title: str

	@property
	def panel_id(self) -> str:
		return self.panel_id_value

	@property
	def title(self) -> str:
		return self.title_value

	@property
	def priority(self) -> int:
		return self.priority_value

	@property
	def tags(self) -> List[str]:
		return list(self.tags_value)

	def required_keys(self) -> List[str]:
		return list(self.required_keys_value)

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		section = _compact(snapshot.get(self.section_key, {}))
		return PanelResult(
			panel_id=self.panel_id,
			title=self.title,
			status=PanelStatus.OK.value,
			generated_at=now.isoformat(),
			render_ms=0.0,
			payload={self.section_title: section},
			warnings=[],
			errors=[],
			meta={"section_key": self.section_key},
		)


class SystemStatusPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "system_status"

	@property
	def title(self) -> str:
		return "System Status"

	@property
	def priority(self) -> int:
		return 10

	@property
	def tags(self) -> List[str]:
		return ["system", "risk"]

	def required_keys(self) -> List[str]:
		return ["feed_health", "using_fallback", "capital", "daily_pnl", "drawdown_pct", "risk_state", "system_state", "mode"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"system_state": snapshot.get("system_state"),
			"mode": snapshot.get("mode"),
			"feed_health": snapshot.get("feed_health"),
			"using_fallback": snapshot.get("using_fallback"),
			"capital": snapshot.get("capital"),
			"daily_pnl": snapshot.get("daily_pnl"),
			"drawdown_pct": snapshot.get("drawdown_pct"),
			"risk_state": snapshot.get("risk_state"),
			"trades_today": snapshot.get("trades_today"),
			"engine_health": _compact(snapshot.get("engine_health", {})),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "status"})


class MarketOverviewPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "market_overview"

	@property
	def title(self) -> str:
		return "Market Overview"

	@property
	def priority(self) -> int:
		return 20

	@property
	def tags(self) -> List[str]:
		return ["market", "live"]

	def required_keys(self) -> List[str]:
		return ["spot", "feed_lag_ms", "ticks_per_second"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"spot": snapshot.get("spot"),
			"previous_close": snapshot.get("previous_close"),
			"feed_lag_ms": snapshot.get("feed_lag_ms"),
			"ticks_per_second": snapshot.get("ticks_per_second"),
			"data_quality_score": snapshot.get("data_quality_score"),
			"session_phase": snapshot.get("session_phase"),
			"day_type": snapshot.get("day_type"),
			"timestamp": _compact(snapshot.get("timestamp")),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "overview"})


class NarrativePanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "narrative"

	@property
	def title(self) -> str:
		return "Morning Briefing"

	@property
	def priority(self) -> int:
		return 30

	@property
	def tags(self) -> List[str]:
		return ["context", "narrative"]

	def required_keys(self) -> List[str]:
		return ["narrative_score", "narrative_label", "regime", "session_phase", "day_type"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"narrative_score": snapshot.get("narrative_score"),
			"narrative_label": snapshot.get("narrative_label"),
			"regime": snapshot.get("regime"),
			"session_phase": snapshot.get("session_phase"),
			"day_type": snapshot.get("day_type"),
			"fit_factors": _compact(snapshot.get("narrative_fit_factors", {})),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "narrative"})


class FeatureSnapshotPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "feature_snapshot"

	@property
	def title(self) -> str:
		return "Feature Snapshot"

	@property
	def priority(self) -> int:
		return 40

	@property
	def tags(self) -> List[str]:
		return ["features", "mtf"]

	def required_keys(self) -> List[str]:
		return ["features"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		features = snapshot.get("features", {})
		payload = {
			"timeframes": _compact(features),
			"options_features": _compact(snapshot.get("options_features", {})),
			"microstructure": _compact(snapshot.get("microstructure", {})),
			"key_levels": _compact(snapshot.get("key_levels", {})),
			"smart_money": _compact(snapshot.get("smart_money", {})),
			"candle_patterns": _compact(snapshot.get("candle_patterns", [])),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "features"})


class OptionChainPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "option_chain"

	@property
	def title(self) -> str:
		return "Option Chain"

	@property
	def priority(self) -> int:
		return 50

	@property
	def tags(self) -> List[str]:
		return ["options", "chain"]

	def required_keys(self) -> List[str]:
		return ["option_chain", "spot"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"spot": snapshot.get("spot"),
			"option_chain": _compact(snapshot.get("option_chain", {})),
			"market_depth": _compact(snapshot.get("market_depth", {})),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "chain"})


class OpportunityPipelinePanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "opportunity_pipeline"

	@property
	def title(self) -> str:
		return "Opportunity Pipeline"

	@property
	def priority(self) -> int:
		return 60

	@property
	def tags(self) -> List[str]:
		return ["decision", "scoring"]

	def required_keys(self) -> List[str]:
		return ["raw_opportunities", "approved_opportunities"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"raw_count": len(snapshot.get("raw_opportunities", []) or []),
			"trapped_count": len(snapshot.get("trapped_opportunities", []) or []),
			"scored_count": len(snapshot.get("scored_opportunities", []) or []),
			"ml_filtered_count": len(snapshot.get("ml_filtered", []) or []),
			"behavioral_filtered_count": len(snapshot.get("behavioral_filtered", []) or []),
			"approved_count": len(snapshot.get("approved_opportunities", []) or []),
			"approved_preview": _compact((snapshot.get("approved_opportunities", []) or [])[:3]),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "pipeline"})


class BrainStatusPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "brain_status"

	@property
	def title(self) -> str:
		return "Brain Status"

	@property
	def priority(self) -> int:
		return 70

	@property
	def tags(self) -> List[str]:
		return ["brains", "context"]

	def required_keys(self) -> List[str]:
		return ["active_brains", "brain_confidence"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"active_brains": _compact(snapshot.get("active_brains", [])),
			"brain_confidence": _compact(snapshot.get("brain_confidence", {})),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "brains"})


class PositionPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "positions"

	@property
	def title(self) -> str:
		return "Positions"

	@property
	def priority(self) -> int:
		return 80

	@property
	def tags(self) -> List[str]:
		return ["positions", "risk"]

	def required_keys(self) -> List[str]:
		return ["open_positions", "trades_today", "daily_pnl"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		open_positions = snapshot.get("open_positions", []) or []
		payload = {
			"open_count": len(open_positions),
			"trades_today": snapshot.get("trades_today"),
			"daily_pnl": snapshot.get("daily_pnl"),
			"positions_preview": _compact(open_positions[:5]),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "positions"})


class EngineHealthPanel(PanelBase):
	@property
	def panel_id(self) -> str:
		return "engine_health"

	@property
	def title(self) -> str:
		return "Engine Health"

	@property
	def priority(self) -> int:
		return 90

	@property
	def tags(self) -> List[str]:
		return ["health", "diagnostics"]

	def required_keys(self) -> List[str]:
		return ["engine_health"]

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"engine_health": _compact(snapshot.get("engine_health", {})),
			"last_update": _compact(snapshot.get("last_update")),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "health"})


class SessionStatePanel(_SimplePanel):
	def __init__(self) -> None:
		super().__init__(
			panel_id_value="session_state",
			title_value="Session State",
			priority_value=25,
			tags_value=["market", "session"],
			required_keys_value=["session_phase", "day_type", "session_size_multiplier"],
			section_key="session_phase",
			section_title="session_phase",
		)

	def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
		payload = {
			"session_phase": snapshot.get("session_phase"),
			"session_size_multiplier": snapshot.get("session_size_multiplier"),
			"day_type": snapshot.get("day_type"),
			"or_high": snapshot.get("or_high"),
			"or_low": snapshot.get("or_low"),
			"ib_high": snapshot.get("ib_high"),
			"ib_low": snapshot.get("ib_low"),
			"ib_width": snapshot.get("ib_width"),
		}
		return PanelResult(self.panel_id, self.title, PanelStatus.OK.value, now.isoformat(), 0.0, payload, [], [], {"kind": "session"})


class StatusPanelAdapter(PanelBase):
    """
    Bridge adapter: wraps StatusPanel into PanelBase contract for registry rendering.

    The actual StatusPanel (QFrame-based) is used in the live GUI.
    This adapter produces a comparable PanelResult for headless/registry rendering
    without requiring PyQt6. The adapter mirrors StatusPanel.update_hot() logic.
    """

    @property
    def panel_id(self) -> str:
        return "status"

    @property
    def title(self) -> str:
        return "STATUS"

    @property
    def priority(self) -> int:
        return 5

    @property
    def tags(self) -> List[str]:
        return ["system", "health", "hot"]

    def required_keys(self) -> List[str]:
        return [
            "feed_health",
            "system_state",
            "mode",
            "data_quality_score",
            "ticks_per_second",
            "feed_lag_ms",
            "using_fallback",
            "last_update_timestamp",
            "risk_state",
            "trades_today",
            "consecutive_losses",
        ]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        payload = {
            "feed_health": snapshot.get("feed_health"),
            "system_state": snapshot.get("system_state"),
            "mode": snapshot.get("mode"),
            "data_quality_score": snapshot.get("data_quality_score"),
            "ticks_per_second": snapshot.get("ticks_per_second"),
            "feed_lag_ms": snapshot.get("feed_lag_ms"),
            "using_fallback": snapshot.get("using_fallback"),
            "last_update_timestamp": _compact(snapshot.get("last_update_timestamp")),
            "risk_state": snapshot.get("risk_state"),
            "trades_today": snapshot.get("trades_today"),
            "consecutive_losses": snapshot.get("consecutive_losses"),
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=PanelStatus.OK.value,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=[],
            errors=[],
            meta={"kind": "status", "refresh_class": "hot"},
        )


class GlobalVitalsPanelAdapter(PanelBase):
    """Registry/headless adapter for Week 06 GlobalVitalsPanel.

    The PyQt GlobalVitalsPanel owns the visible cockpit widget and Component
    Guard signal behavior. This adapter keeps the default PanelRegistry and
    headless render path backend-truth-only and PyQt-free, mirroring the same
    hot-snapshot fields without constructing UI widgets.
    """

    @property
    def panel_id(self) -> str:
        return "global_vitals"

    @property
    def title(self) -> str:
        return "GLOBAL VITALS"

    @property
    def priority(self) -> int:
        return 9

    @property
    def tags(self) -> List[str]:
        return ["market", "global", "guard", "hot"]

    def required_keys(self) -> List[str]:
        return [
            "spot",
            "previous_close",
            "regime",
            "narrative_label",
            "mode",
            "feed_health",
            "data_quality_score",
            "regime_transition_prob",
            "drawdown_pct",
            "session_phase",
            "day_type",
            "active_brains",
        ]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        heavyweights = snapshot.get("component_guard_heavyweights", snapshot.get("heavyweights", []))
        if not isinstance(heavyweights, list):
            heavyweights = []

        veto_count = 0
        watch_count = 0
        ok_count = 0
        for item in heavyweights:
            if not isinstance(item, dict):
                continue
            state = str(item.get("veto_status", "")).upper().strip()
            if state == "VETO":
                veto_count += 1
            elif state == "WATCH":
                watch_count += 1
            elif state == "OK":
                ok_count += 1

        payload = {
            "spot": snapshot.get("spot"),
            "previous_close": snapshot.get("previous_close"),
            "regime": snapshot.get("regime"),
            "narrative_label": snapshot.get("narrative_label"),
            "mode": snapshot.get("mode"),
            "feed_health": snapshot.get("feed_health"),
            "data_quality_score": snapshot.get("data_quality_score"),
            "regime_transition_prob": snapshot.get("regime_transition_prob"),
            "drawdown_pct": snapshot.get("drawdown_pct"),
            "session_phase": snapshot.get("session_phase"),
            "day_type": snapshot.get("day_type"),
            "active_brains": _compact(snapshot.get("active_brains", [])),
            "component_guard": {
                "count": len(heavyweights),
                "ok_count": ok_count,
                "watch_count": watch_count,
                "veto_count": veto_count,
                "heavyweights": _compact(heavyweights[:5]),
            },
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=PanelStatus.OK.value,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=[],
            errors=[],
            meta={"kind": "global_vitals", "refresh_class": "hot"},
        )


class MtfChartPanelAdapter(PanelBase):
    """Registry/headless adapter for Week 07 MTF chart panel shell.

    The PyQt MtfChartPanel embeds MtfChart for visible runtime use. This adapter
    keeps the PanelRegistry/headless path PyQt-free and verifies that prepared
    candle data is present without computing or aggregating market data.
    """

    @property
    def panel_id(self) -> str:
        return "mtf_chart"

    @property
    def title(self) -> str:
        return "MTF CHART"

    @property
    def priority(self) -> int:
        return 45

    @property
    def tags(self) -> List[str]:
        return ["chart", "mtf", "warm"]

    def required_keys(self) -> List[str]:
        return ["mtf_candles"]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        src = snapshot.get("mtf_candles", snapshot.get("candles_by_tf", snapshot.get("candles", {})))
        candles_by_tf = dict(src) if isinstance(src, dict) else {}
        counts: Dict[str, int] = {}
        latest: Dict[str, Any] = {}
        for tf in ("1m", "3m", "5m", "15m"):
            rows = candles_by_tf.get(tf, [])
            if isinstance(rows, list):
                counts[tf] = len(rows)
                if rows and isinstance(rows[-1], dict):
                    latest[tf] = _compact(rows[-1])
            else:
                counts[tf] = 0

        active_tf = snapshot.get("active_timeframe", snapshot.get("timeframe", "5m"))
        vwap_bands = snapshot.get("vwap_bands", {})
        active_vwap = {}
        if isinstance(vwap_bands, dict):
            active_vwap = vwap_bands.get(str(active_tf), vwap_bands) if isinstance(vwap_bands.get(str(active_tf), vwap_bands), dict) else {}

        vwap_points = (
            len(active_vwap.get("vwap", snapshot.get("vwap", [])))
            if isinstance(active_vwap.get("vwap", snapshot.get("vwap", [])), list)
            else 0
        )
        maxpoints_limit = 5000
        active_count = counts.get(str(active_tf), 0)
        total_points = sum(int(v or 0) for v in counts.values())
        performance_profile = {
            # Week 08 Thu/Fri profiling foundation: this is UI payload-size
            # telemetry derived only from supplied snapshot data.  It does not
            # create market values or alpha; it helps detect chart DOM pressure
            # before live repaint issues reach the operator.
            "active_timeframe": active_tf,
            "active_candle_points": active_count,
            "total_candle_points": total_points,
            "vwap_points": vwap_points,
            "maxpoints_limit": maxpoints_limit,
            "dom_window_bounded": active_count <= maxpoints_limit,
            "profile_source": "snapshot_payload",
        }
        warnings: List[str] = []
        status = PanelStatus.OK.value
        if active_count > maxpoints_limit:
            status = PanelStatus.DEGRADED.value
            warnings.append("active_timeframe_exceeds_maxpoints_limit")

        payload = {
            "active_timeframe": active_tf,
            "timeframes": counts,
            "latest_candle": latest,
            "overlays": {
                "has_vwap": "vwap" in snapshot or "vwap_bands" in snapshot,
                "vwap_points": vwap_points,
                "or_high": snapshot.get("or_high", (snapshot.get("or_levels") or {}).get("high") if isinstance(snapshot.get("or_levels"), dict) else None),
                "or_low": snapshot.get("or_low", (snapshot.get("or_levels") or {}).get("low") if isinstance(snapshot.get("or_levels"), dict) else None),
                "ib_high": snapshot.get("ib_high", (snapshot.get("ib_levels") or {}).get("high") if isinstance(snapshot.get("ib_levels"), dict) else None),
                "ib_low": snapshot.get("ib_low", (snapshot.get("ib_levels") or {}).get("low") if isinstance(snapshot.get("ib_levels"), dict) else None),
            },
            "performance_profile": performance_profile,
            # Backward-compatible flags used by existing Week 07 Wednesday tests.
            "has_vwap": "vwap" in snapshot or "vwap_bands" in snapshot,
            "has_or_levels": "or_high" in snapshot or "or_low" in snapshot or "or_levels" in snapshot,
            "has_ib_levels": "ib_high" in snapshot or "ib_low" in snapshot or "ib_levels" in snapshot,
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=status,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=warnings,
            errors=[],
            meta={"kind": "mtf_chart", "refresh_class": "warm", "performance_profile": performance_profile},
        )


class SystemHealthPanelAdapter(PanelBase):
    """
    Bridge adapter: wraps SystemHealthPanel into PanelBase contract for registry rendering.

    The actual SystemHealthPanel (QFrame-based) is used in the live GUI.
    This adapter produces a comparable PanelResult for headless/registry rendering
    without requiring PyQt6. The adapter mirrors SystemHealthPanel.update_cold() logic.
    """

    @property
    def panel_id(self) -> str:
        return "system_health"

    @property
    def title(self) -> str:
        return "SYSTEM HEALTH"

    @property
    def priority(self) -> int:
        return 50

    @property
    def tags(self) -> List[str]:
        return ["health", "diagnostics", "cold"]

    def required_keys(self) -> List[str]:
        return [
            "engine_health",
            "feed_health",
            "data_quality_score",
            "ticks_per_second",
            "feed_lag_ms",
            "using_fallback",
            "kill_switch_state",
            "snapshot_age_seconds",
            "last_update_timestamp",
        ]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        engine_health = snapshot.get("engine_health", {}) or {}
        engine_summary = {}
        if isinstance(engine_health, dict):
            for k, v in engine_health.items():
                if isinstance(v, dict):
                    engine_summary[k] = {
                        "alive": v.get("alive"),
                        "status": v.get("status"),
                        "last_error": v.get("last_error"),
                    }
                else:
                    engine_summary[k] = str(v)

        payload = {
            "engine_summary": _compact(engine_summary),
            "feed_health": snapshot.get("feed_health"),
            "data_quality_score": snapshot.get("data_quality_score"),
            "ticks_per_second": snapshot.get("ticks_per_second"),
            "feed_lag_ms": snapshot.get("feed_lag_ms"),
            "using_fallback": snapshot.get("using_fallback"),
            "kill_switch_state": snapshot.get("kill_switch_state"),
            "snapshot_age_seconds": snapshot.get("snapshot_age_seconds"),
            "last_update_timestamp": _compact(snapshot.get("last_update_timestamp")),
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=PanelStatus.OK.value,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=[],
            errors=[],
            meta={"kind": "system_health", "refresh_class": "cold"},
        )


class VolumeProfilePanelAdapter(PanelBase):
    """Registry/headless adapter for Week 09 Volume Profile panel.

    The PyQt VolumeProfilePanel owns the visible widget. This adapter keeps the
    PanelRegistry/headless path PyQt-free and mirrors same snapshot fields.
    """

    @property
    def panel_id(self) -> str:
        return "volume_profile"

    @property
    def title(self) -> str:
        return "VOLUME PROFILE"

    @property
    def priority(self) -> int:
        return 42

    @property
    def tags(self) -> List[str]:
        return ["volume", "profile", "warm"]

    def required_keys(self) -> List[str]:
        return ["volume_profile", "microstructure"]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        # Keep raw reference for field detection; _compact truncates large dicts.
        raw_vp = snapshot.get("volume_profile", {})
        if not isinstance(raw_vp, dict):
            raw_vp = {}
        vp = _compact(raw_vp)
        ms = _compact(snapshot.get("microstructure", {}))

        # Read poc/vah/val from nested volume_profile first, then fall back to
        # top-level snapshot keys (warm projection may copy them to both places).
        poc = snapshot.get("poc")
        if poc is None:
            poc = raw_vp.get("poc")
        vah = snapshot.get("vah")
        if vah is None:
            vah = raw_vp.get("vah")
        val = snapshot.get("val")
        if val is None:
            val = raw_vp.get("val")

        payload = {
            "volume_profile": vp,
            "microstructure": ms,
            "has_profile": bool(raw_vp and raw_vp.get("profile")),
            "poc": poc,
            "vah": vah,
            "val": val,
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=PanelStatus.OK.value,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=[],
            errors=[],
            meta={"kind": "volume_profile", "refresh_class": "warm"},
        )


class BriefingPanelAdapter(PanelBase):
    """
    Bridge adapter: wraps BriefingPanel into PanelBase contract for registry rendering.

    The actual BriefingPanel (QFrame-based) is used in the live GUI.
    This adapter produces a comparable PanelResult for headless/registry rendering
    without requiring PyQt6. The adapter mirrors BriefingPanel.update_warm() logic.
    """

    @property
    def panel_id(self) -> str:
        return "briefing"

    @property
    def title(self) -> str:
        return "BRIEFING"

    @property
    def priority(self) -> int:
        return 8

    @property
    def tags(self) -> List[str]:
        return ["context", "narrative", "warm"]

    def required_keys(self) -> List[str]:
        return [
            "narrative_label",
            "narrative_score",
            "narrative_fit_factors",
            "day_personality",
            "historical_match_score",
            "session_memory",
            "regime",
            "regime_confidence",
            "session_phase",
            "day_type",
        ]

    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
        # Keep adapter defensive: backend projection may temporarily provide
        # strings/None for nested warm context while upstream wiring stabilizes.
        # We render known values and never raise from shape mismatches.
        fit = snapshot.get("narrative_fit_factors", {}) or {}
        fit = fit if isinstance(fit, dict) else {}
        day_personality = snapshot.get("day_personality", {}) or {}
        day_personality = day_personality if isinstance(day_personality, dict) else {"day_type": day_personality}
        session_memory = snapshot.get("session_memory", {}) or {}
        session_memory = session_memory if isinstance(session_memory, dict) else {"summary": session_memory}

        payload = {
            "narrative_label": snapshot.get("narrative_label"),
            "narrative_score": snapshot.get("narrative_score"),
            "long_fit": fit.get("long_fit"),
            "short_fit": fit.get("short_fit"),
            "day_personality": day_personality.get("day_type"),
            "historical_match_score": snapshot.get("historical_match_score"),
            "session_memory_summary": _compact(session_memory),
            "regime": snapshot.get("regime"),
            "regime_confidence": snapshot.get("regime_confidence"),
            "session_phase": snapshot.get("session_phase"),
            "day_type": snapshot.get("day_type"),
        }
        return PanelResult(
            panel_id=self.panel_id,
            title=self.title,
            status=PanelStatus.OK.value,
            generated_at=now.isoformat(),
            render_ms=0.0,
            payload=payload,
            warnings=[],
            errors=[],
            meta={"kind": "briefing", "refresh_class": "warm"},
        )


def build_default_panels() -> List[PanelBase]:
	"""Build the default panel list.
	
	Duplicates removed for Week 9 cockpit cleanup:
	- SystemStatusPanel (duplicates real STATUS widget)
	- NarrativePanel (duplicates real BRIEFING/BriefingPanelAdapter widget)
	
	Real widget panels are in _create_real_panel_widget() in main_window.py.
	Registry adapters here support headless rendering and tab fallback.
	"""
	return [
		StatusPanelAdapter(),       # Cockpit — real widget exists
		BriefingPanelAdapter(),     # Cockpit — real widget exists
		GlobalVitalsPanelAdapter(), # Cockpit — real widget exists
		VolumeProfilePanelAdapter(),# Markets — real widget exists
		MarketOverviewPanel(),      # Markets — headless adapter only
		SessionStatePanel(),        # Risk — headless adapter only
		FeatureSnapshotPanel(),     # Systems — headless adapter only
		MtfChartPanelAdapter(),     # Markets — real widget exists
		OptionChainPanel(),         # Markets — headless adapter (Week 10)
		SystemHealthPanelAdapter(), # Cockpit — real widget exists
		OpportunityPipelinePanel(), # Systems — headless adapter only
		BrainStatusPanel(),         # Systems — headless adapter only
		PositionPanel(),            # Risk — headless adapter (Week 13)
		EngineHealthPanel(),        # Systems — headless adapter only
	]


def build_default_registry() -> PanelRegistry:
	registry = PanelRegistry()
	for panel in build_default_panels():
		registry.register(panel)
	return registry


__all__ = ["build_default_panels", "build_default_registry"]
