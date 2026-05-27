# FILE: dashboard/panels/__init__.py
"""
Junior Aladdin — Dashboard Panels Contract & Registry
=====================================================

This module defines the *panel contract* and a *safe registry* to render dashboard
panels deterministically from a MarketState snapshot.

Hard requirements satisfied:
- PanelStatus enum
- PanelResult dataclass (standardized payload)
- PanelBase contract (render/required_keys/validate_snapshot)
- PanelRegistry (register/list/render_all/metrics, failure isolation, rate-limited logging)
- Snapshot safety (read-only; invalid snapshot handled safely)
- JSON safety (payload is coerced to JSON-friendly primitives)
- Performance safety (O(N panels), no blocking I/O, bounded rolling metrics)
- Minimal __main__ self-test (no external deps, no UI imports)

Architecture (brief):
- Panels are independent renderers that accept `snapshot: dict` and return PanelResult.
- PanelRegistry orchestrates rendering:
  - validates snapshot keys
  - isolates panel exceptions
  - records per-panel metrics
  - rate-limits error logs per panel

Modified file list:
- dashboard/panels/__init__.py (only)

Runtime impact:
- Dashboard becomes deterministic and observable.
- Panel failures cannot break other panels or the registry.

Self-test:
    python -m dashboard.panels
"""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from src.utils.logger import setup_logger

_log = setup_logger("dashboard_panels")


class PanelStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    ERROR = "ERROR"
    DISABLED = "DISABLED"


@dataclass
class PanelResult:
    panel_id: str
    title: str
    status: str
    generated_at: str
    render_ms: float
    payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class PanelBase(ABC):
    """
    Base contract for dashboard panels.

    Rules:
    - Panel code should be pure, fast, and read-only.
    - Panel.render SHOULD NOT raise; registry still guards against exceptions.
    """

    @property
    @abstractmethod
    def panel_id(self) -> str: ...

    @property
    @abstractmethod
    def title(self) -> str: ...

    @property
    @abstractmethod
    def priority(self) -> int: ...

    @property
    @abstractmethod
    def tags(self) -> List[str]: ...

    @abstractmethod
    def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult: ...

    @abstractmethod
    def required_keys(self) -> List[str]: ...

    def stale_after_seconds(self) -> Optional[float]:
        """Optional freshness SLA for this panel.

        Week 04 integration note:
        Panels may opt into stale-state handling by returning a positive
        threshold in seconds.  The registry then checks a backend-provided
        timestamp (``timestamp``, ``last_update``, or
        ``last_update_timestamp``) and marks the panel STALE if the snapshot is
        older than the threshold.

        Default is ``None`` so existing/future panels are not forced to invent
        freshness where the backend has not published a timestamp.  This keeps
        the dashboard roadmap rule intact: render known state only; never guess.
        """
        return None

    def validate_snapshot(self, snapshot: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that required_keys() exist in snapshot.

        Supports dotted paths (e.g., "meta.total_computations") for nested dicts.
        """
        missing: List[str] = []
        for key in self.required_keys() or []:
            if not _has_key_path(snapshot, key):
                missing.append(key)
        return (len(missing) == 0), missing


# ---------------------------
# Registry internals
# ---------------------------

_METRIC_WINDOW = 200
_ERROR_LOG_RATE_LIMIT_SEC = 60.0


@dataclass
class _PanelMetrics:
    last_render_at: Optional[str] = None
    last_success_at: Optional[str] = None
    last_error_at: Optional[str] = None
    error_count: int = 0
    last_status: str = PanelStatus.STALE.value
    render_ms_window: Deque[float] = field(default_factory=lambda: deque(maxlen=_METRIC_WINDOW))
    last_error_log_mono: float = 0.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    try:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return _utc_now().isoformat()


def _has_key_path(d: Dict[str, Any], key_path: str) -> bool:
    if not isinstance(d, dict):
        return False
    kp = str(key_path).strip()
    if not kp:
        return False
    if "." not in kp:
        return kp in d
    cur: Any = d
    for part in kp.split("."):
        if not isinstance(cur, dict):
            return False
        if part not in cur:
            return False
        cur = cur.get(part)
    return True


def _jsonable(obj: Any, *, _depth: int = 0, _max_depth: int = 6) -> Any:
    """
    Coerce common types into JSON-serializable primitives.
    Depth-limited to avoid runaway recursion.
    """
    if _depth > _max_depth:
        return str(obj)

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return _iso(obj)

    if isinstance(obj, Enum):
        return str(obj.value)

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                ks = str(k)
            except Exception:
                ks = repr(k)
            out[ks] = _jsonable(v, _depth=_depth + 1, _max_depth=_max_depth)
        return out

    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x, _depth=_depth + 1, _max_depth=_max_depth) for x in obj]

    # Fallback: stringify unknown objects
    return str(obj)


def _ensure_payload_dict(payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (payload_dict, error_string_if_any).
    Ensures payload is a dict and JSON-friendly.
    """
    if payload is None:
        return {}, None
    if not isinstance(payload, dict):
        return None, f"payload_not_dict(type={type(payload).__name__})"
    safe = _jsonable(payload)
    # Validate JSON serializability quickly
    try:
        json.dumps(safe)
    except Exception as e:
        return None, f"payload_not_json_serializable(error={str(e)[:200]})"
    return safe, None


def _coerce_snapshot_datetime(value: Any) -> Optional[datetime]:
    """Best-effort timestamp parser for stale-state handling.

    Accepts aware/naive datetimes, ISO-8601 strings, and numeric epoch values
    (seconds / milliseconds / nanoseconds). Returns timezone-aware UTC or None.
    """
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            raw = float(value)
            # Heuristic: ns > ms > seconds. Backend HOT frames commonly carry
            # timestamp_ns; snapshots may carry seconds. We do not mutate input.
            if raw > 1e17:
                raw = raw / 1e9
            elif raw > 1e12:
                raw = raw / 1e3
            dt = datetime.fromtimestamp(raw, tz=timezone.utc)
        elif isinstance(value, str) and value.strip():
            raw_s = value.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(raw_s)
        else:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _snapshot_age_seconds(snapshot: Dict[str, Any], now_dt: datetime) -> Optional[float]:
    """Return age in seconds when the backend provided a known timestamp."""
    if not isinstance(snapshot, dict):
        return None
    for key in ("timestamp", "last_update", "last_update_timestamp", "timestamp_ns"):
        if key not in snapshot:
            continue
        dt = _coerce_snapshot_datetime(snapshot.get(key))
        if dt is None:
            continue
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        age = (now_dt.astimezone(timezone.utc) - dt).total_seconds()
        return max(0.0, age)
    return None


class PanelRegistry:
    """
    Panel registry and deterministic renderer.

    - register/unregister by panel_id
    - list_panels sorted by (priority asc, panel_id asc)
    - render_all isolates failures per panel and records metrics
    """

    def __init__(self):
        self._panels: Dict[str, PanelBase] = {}
        self._metrics: Dict[str, _PanelMetrics] = {}

    def register(self, panel: PanelBase) -> None:
        """
        Register (or replace) a panel by panel_id.
        Deduplicates and replaces older entry with a warning.
        """
        try:
            pid = str(getattr(panel, "panel_id", "")).strip()
            if not pid:
                _log.error("Panel registration failed: empty panel_id", panel_type=type(panel).__name__)
                return

            replaced = pid in self._panels
            self._panels[pid] = panel
            self._metrics.setdefault(pid, _PanelMetrics())

            if replaced:
                _log.warning("Duplicate panel_id registered; replaced older panel", panel_id=pid, panel_type=type(panel).__name__)
            else:
                _log.info("Panel registered", panel_id=pid, panel_type=type(panel).__name__)
        except Exception as e:
            # Registry must never throw
            _log.error("PanelRegistry.register exception", error=str(e)[:300])

    def unregister(self, panel_id: str) -> None:
        pid = str(panel_id).strip()
        if not pid:
            return
        try:
            existed = pid in self._panels
            self._panels.pop(pid, None)
            # Keep metrics for diagnostics unless explicitly desired to purge.
            if existed:
                _log.info("Panel unregistered", panel_id=pid)
        except Exception as e:
            _log.error("PanelRegistry.unregister exception", panel_id=pid, error=str(e)[:300])

    def list_panels(self) -> List[PanelBase]:
        try:
            return sorted(
                list(self._panels.values()),
                key=lambda p: (int(getattr(p, "priority", 10**9)), str(getattr(p, "panel_id", ""))),
            )
        except Exception as e:
            _log.error("PanelRegistry.list_panels exception", error=str(e)[:300])
            return list(self._panels.values())

    @staticmethod
    def _panel_refresh_class(panel: PanelBase) -> str:
        try:
            tags = {str(t).lower() for t in (getattr(panel, "tags", []) or [])}
            for candidate in ("hot", "warm", "cold"):
                if candidate in tags:
                    return candidate
        except Exception:
            pass
        # Untagged legacy/headless placeholders are COLD by default so they do
        # not silently join the 200ms HOT loop as Week 9+ panels get heavier.
        return "cold"

    def _filter_panels_by_refresh_class(self, panels: List[PanelBase], refresh_class: Optional[str]) -> List[PanelBase]:
        if refresh_class is None:
            return list(panels)
        rc = str(refresh_class).lower().strip()
        if rc not in {"hot", "warm", "cold"}:
            return list(panels)
        return [p for p in panels if self._panel_refresh_class(p) == rc]

    def render_all(
        self,
        snapshot: Any,
        now: Optional[datetime] = None,
        *,
        refresh_class: Optional[str] = None,
    ) -> List[PanelResult]:
        """
        Render panels with isolation.

        Pre-Week-9 tier-purity note:
        ``refresh_class`` may be "hot", "warm", or "cold".  When provided,
        only panels tagged for that class render.  This prevents the HOT loop
        from rendering all adapters every 200ms while preserving the default
        render-all behavior for headless reports/tests.

        Snapshot safety:
        - snapshot must be a dict (read-only).
        - if invalid snapshot: returns ERROR results (per-panel if panels exist,
          otherwise a single registry-level error result).
        """
        try:
            now_dt = now or _utc_now()
            generated_at = _iso(now_dt)

            panels = self._filter_panels_by_refresh_class(self.list_panels(), refresh_class)
            if not isinstance(snapshot, dict):
                msg = "invalid_snapshot"
                if not panels:
                    return [
                        PanelResult(
                            panel_id="__registry__",
                            title="Registry",
                            status=PanelStatus.ERROR.value,
                            generated_at=generated_at,
                            render_ms=0.0,
                            payload={},
                            warnings=[],
                            errors=[msg],
                            meta={"reason": msg, "snapshot_type": type(snapshot).__name__},
                        )
                    ]
                results: List[PanelResult] = []
                for p in panels:
                    results.append(
                        self._make_error_result(
                            panel=p,
                            now_dt=now_dt,
                            render_ms=0.0,
                            errors=[msg],
                            warnings=[],
                            meta={"reason": msg, "snapshot_type": type(snapshot).__name__},
                            log_exception=False,
                        )
                    )
                return results

            results: List[PanelResult] = []
            for panel in panels:
                pid = str(getattr(panel, "panel_id", "")).strip() or "UNKNOWN_PANEL"
                metrics = self._metrics.setdefault(pid, _PanelMetrics())

                t0 = time.monotonic()
                try:
                    ok, missing = panel.validate_snapshot(snapshot)
                    warnings: List[str] = []
                    if not ok and missing:
                        warnings.append(f"missing_required_keys={missing}")

                    # Render panel (panel should ideally not raise)
                    pr = panel.render(snapshot, now_dt)
                    render_ms = (time.monotonic() - t0) * 1000.0

                    # Ensure PanelResult contract
                    if not isinstance(pr, PanelResult):
                        raise TypeError(f"render() must return PanelResult, got {type(pr).__name__}")

                    # Enforce id/title and timestamps
                    pr.panel_id = pid
                    pr.title = str(getattr(panel, "title", pr.title) or pr.title)
                    pr.generated_at = pr.generated_at or generated_at
                    pr.render_ms = float(render_ms)

                    # Degraded if missing keys
                    if warnings and pr.status == PanelStatus.OK.value:
                        pr.status = PanelStatus.DEGRADED.value
                    pr.warnings = list(pr.warnings or []) + warnings

                    # JSON safety of payload
                    payload_dict, payload_err = _ensure_payload_dict(pr.payload)
                    if payload_err:
                        raise TypeError(payload_err)
                    pr.payload = payload_dict or {}

                    # Week 04 stale-state support:
                    # If a panel opts into a freshness SLA and the backend gave
                    # us a recognizable timestamp, downgrade OK/DEGRADED to
                    # STALE once the snapshot exceeds the SLA. We intentionally
                    # do nothing when no timestamp exists; missing data is
                    # already represented through validate_snapshot warnings and
                    # must not be guessed.
                    stale_warning = self._stale_warning(panel, snapshot, now_dt)
                    if stale_warning and pr.status not in (PanelStatus.ERROR.value, PanelStatus.DISABLED.value):
                        pr.status = PanelStatus.STALE.value
                        pr.warnings = list(pr.warnings or []) + [stale_warning]

                    # Update metrics
                    self._update_metrics(metrics, pr, now_dt)

                    results.append(pr)

                except Exception as e:
                    render_ms = (time.monotonic() - t0) * 1000.0
                    tb = traceback.format_exc(limit=6)
                    err_str = f"{type(e).__name__}: {str(e)}"
                    pr_err = self._make_error_result(
                        panel=panel,
                        now_dt=now_dt,
                        render_ms=render_ms,
                        errors=[err_str],
                        warnings=[],
                        meta={"traceback": tb},
                        log_exception=True,
                    )
                    results.append(pr_err)

            return results

        except Exception as e:
            # Registry must never throw
            _log.critical("PanelRegistry.render_all fatal exception", error=str(e)[:300])
            now_dt = now or _utc_now()
            return [
                PanelResult(
                    panel_id="__registry__",
                    title="Registry",
                    status=PanelStatus.ERROR.value,
                    generated_at=_iso(now_dt),
                    render_ms=0.0,
                    payload={},
                    warnings=[],
                    errors=[f"registry_error:{type(e).__name__}:{str(e)[:200]}"],
                    meta={},
                )
            ]

    def _stale_warning(self, panel: PanelBase, snapshot: Dict[str, Any], now_dt: datetime) -> Optional[str]:
        try:
            threshold = panel.stale_after_seconds()
            if threshold is None:
                return None
            threshold_f = float(threshold)
            if threshold_f <= 0:
                return None
            age = _snapshot_age_seconds(snapshot, now_dt)
            if age is None or age <= threshold_f:
                return None
            return f"stale_snapshot_age_sec={age:.1f};threshold_sec={threshold_f:.1f}"
        except Exception as exc:
            # Stale calculation is observability, not business logic. Never let
            # it break panel rendering. The warning remains local to logs.
            _log.warning(
                "Panel stale check failed",
                panel_id=str(getattr(panel, "panel_id", "UNKNOWN_PANEL")),
                error=str(exc)[:200],
            )
            return None

    def _make_error_result(
        self,
        *,
        panel: PanelBase,
        now_dt: datetime,
        render_ms: float,
        errors: List[str],
        warnings: List[str],
        meta: Dict[str, Any],
        log_exception: bool,
    ) -> PanelResult:
        pid = str(getattr(panel, "panel_id", "")).strip() or "UNKNOWN_PANEL"
        title = str(getattr(panel, "title", "Panel") or "Panel")

        metrics = self._metrics.setdefault(pid, _PanelMetrics())
        pr = PanelResult(
            panel_id=pid,
            title=title,
            status=PanelStatus.ERROR.value,
            generated_at=_iso(now_dt),
            render_ms=float(render_ms),
            payload={},
            warnings=list(warnings or []),
            errors=list(errors or []),
            meta=_jsonable(meta or {}),
        )

        self._update_metrics(metrics, pr, now_dt, is_error=True)

        if log_exception:
            self._rate_limited_error_log(panel_id=pid, title=title, errors=pr.errors, meta=pr.meta, metrics=metrics)

        return pr

    def _rate_limited_error_log(self, *, panel_id: str, title: str, errors: List[str], meta: Dict[str, Any], metrics: _PanelMetrics) -> None:
        now_m = time.monotonic()
        if (now_m - metrics.last_error_log_mono) < _ERROR_LOG_RATE_LIMIT_SEC:
            return
        metrics.last_error_log_mono = now_m
        try:
            _log.error("Panel render failed", panel_id=panel_id, title=title, errors=errors[:3], meta=meta)
        except Exception:
            # last-resort; never throw
            pass

    def _update_metrics(self, metrics: _PanelMetrics, pr: PanelResult, now_dt: datetime, *, is_error: bool = False) -> None:
        ts = _iso(now_dt)
        metrics.last_render_at = ts
        metrics.last_status = str(pr.status)

        # Rolling average window
        try:
            metrics.render_ms_window.append(float(pr.render_ms))
        except Exception:
            pass

        if str(pr.status) == PanelStatus.OK.value and not pr.errors and not is_error:
            metrics.last_success_at = ts

        if str(pr.status) == PanelStatus.ERROR.value or is_error:
            metrics.error_count += 1
            metrics.last_error_at = ts

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns per-panel metrics + registry summary.

        Per-panel:
            last_render_at, last_success_at, last_error_at, error_count,
            avg_render_ms (rolling), last_status
        Summary:
            total_panels, ok_count, error_count, degraded_count, stale_count, disabled_count
        """
        try:
            per_panel: Dict[str, Any] = {}
            status_counts = {
                PanelStatus.OK.value: 0,
                PanelStatus.ERROR.value: 0,
                PanelStatus.DEGRADED.value: 0,
                PanelStatus.STALE.value: 0,
                PanelStatus.DISABLED.value: 0,
            }

            for pid, m in self._metrics.items():
                avg_ms: Optional[float]
                if m.render_ms_window:
                    avg_ms = round(sum(m.render_ms_window) / max(len(m.render_ms_window), 1), 3)
                else:
                    avg_ms = None

                per_panel[pid] = {
                    "last_render_at": m.last_render_at,
                    "last_success_at": m.last_success_at,
                    "last_error_at": m.last_error_at,
                    "error_count": int(m.error_count),
                    "avg_render_ms": avg_ms,
                    "last_status": m.last_status,
                }

                if m.last_status in status_counts:
                    status_counts[m.last_status] += 1

            summary = {
                "total_panels": len(self._panels),
                "ok_count": status_counts[PanelStatus.OK.value],
                "error_count": status_counts[PanelStatus.ERROR.value],
                "degraded_count": status_counts[PanelStatus.DEGRADED.value],
                "stale_count": status_counts[PanelStatus.STALE.value],
                "disabled_count": status_counts[PanelStatus.DISABLED.value],
            }

            return {"summary": summary, "per_panel": per_panel}
        except Exception as e:
            _log.error("PanelRegistry.get_metrics exception", error=str(e)[:300])
            return {"summary": {"total_panels": len(self._panels), "error": str(e)[:200]}, "per_panel": {}}


__all__ = [
    "PanelStatus",
    "PanelResult",
    "PanelBase",
    "PanelRegistry",
    "build_default_panels",
    "build_default_registry",
]


def build_default_panels() -> List[PanelBase]:
    from .catalog import build_default_panels as _build_default_panels

    return _build_default_panels()


def build_default_registry() -> PanelRegistry:
    from .catalog import build_default_registry as _build_default_registry

    return _build_default_registry()


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    class _OkPanel(PanelBase):
        @property
        def panel_id(self) -> str:
            return "panel_ok"

        @property
        def title(self) -> str:
            return "OK Panel"

        @property
        def priority(self) -> int:
            return 10

        @property
        def tags(self) -> List[str]:
            return ["system", "health"]

        def required_keys(self) -> List[str]:
            return ["system_state", "spot"]

        def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
            payload = {"system_state": snapshot.get("system_state"), "spot": snapshot.get("spot")}
            return PanelResult(
                panel_id=self.panel_id,
                title=self.title,
                status=PanelStatus.OK.value,
                generated_at=_iso(now),
                render_ms=0.0,
                payload=payload,
                warnings=[],
                errors=[],
                meta={},
            )

    class _BoomPanel(PanelBase):
        @property
        def panel_id(self) -> str:
            return "panel_boom"

        @property
        def title(self) -> str:
            return "Boom Panel"

        @property
        def priority(self) -> int:
            return 20

        @property
        def tags(self) -> List[str]:
            return ["debug"]

        def required_keys(self) -> List[str]:
            return ["system_state"]

        def render(self, snapshot: Dict[str, Any], now: datetime) -> PanelResult:
            raise RuntimeError("boom")

    reg = PanelRegistry()
    reg.register(_OkPanel())
    reg.register(_BoomPanel())

    snap = {"system_state": "ACTIVE", "spot": 24500.0}

    out = reg.render_all(snap, now=_utc_now())
    statuses = {r.panel_id: r.status for r in out}

    ok = statuses.get("panel_ok") == PanelStatus.OK.value
    boom = statuses.get("panel_boom") == PanelStatus.ERROR.value

    print("=== Self-test Results ===")
    print("panel_ok:", statuses.get("panel_ok"))
    print("panel_boom:", statuses.get("panel_boom"))
    print("metrics:", reg.get_metrics()["summary"])

    if ok and boom:
        print("PASS")
        raise SystemExit(0)
    print("FAIL")
    raise SystemExit(1)