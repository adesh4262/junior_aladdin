"""
dashboard/charts/qwebengine_plotly_bridge.py

Reusable bridge between PyQt6 QWebEngineView and Plotly.js.

Purpose
-------
- Load a minimal HTML page that hosts Plotly charts.
- Provide Python methods to initialize/update/extend traces via runJavaScript().
- Queue commands until the page is ready (HTML loaded + Plotly available).
- Batch high-frequency extendTraces payloads before crossing the Python->JS boundary.
- Coalesce browser-side Plotly writes with requestAnimationFrame.
- Reuse chart DOM with Plotly.react on reinitialization to reduce churn.

Non-goals
---------
- No chart business logic (OHLC aggregation, data sourcing, strategy logic).
- No backend calls.
- No threads/async; relies on Qt timers and asynchronous JavaScript execution.

Notes
-----
- QWebEnginePage.runJavaScript is asynchronous; this bridge logs errors via callbacks
  but does not block.
- For safety and to avoid JS injection pitfalls, arguments are passed as JSON strings
  and parsed inside JavaScript (no eval).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import json
import logging
from pathlib import Path
import time

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtWebEngineWidgets import QWebEngineView

from src.utils.logger import setup_logger
try:
    from src.utils.config_loader import Config
except Exception:  # pragma: no cover
    Config = None  # type: ignore


@dataclass(frozen=True)
class _PendingCommand:
    js: str
    desc: str
    created_ns: int


@dataclass(frozen=True)
class _PendingExtendTrace:
    div_id: str
    trace_index: int
    update: Dict[str, Any]
    max_points: int
    created_ns: int


class PlotlyBridge(QObject):
    """
    Plotly bridge for QWebEngineView.

    Signals
    -------
    page_ready:
        Emitted once when HTML is loaded and Plotly is available.
    log_message(msg, level):
        Optional signal for external log capture/telemetry.
    error_occurred(msg):
        Emitted on serious, actionable issues (page load failure, etc.).
    """

    page_ready = pyqtSignal()
    log_message = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, web_view: QWebEngineView, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._log = self._get_logger()
        self._view = web_view

        self._page_loaded: bool = False
        self._ready: bool = False

        self._pending: List[_PendingCommand] = []
        self._pending_max: int = 5000  # defensive cap for replay/high-frequency updates

        # Week 08 Mon — Python->JS payload batching for high-frequency
        # extendTraces calls.  We coalesce multiple updates for the same
        # (div_id, trace_index) into one command and flush on a short Qt timer.
        # This reduces runJavaScript/JSON.parse churn without changing the
        # dashboard truth contract: callers still provide prepared chart data.
        self._extend_batches: Dict[Tuple[str, int], _PendingExtendTrace] = {}
        self._extend_batch_scheduled: bool = False
        self._extend_batch_window_ms: int = 50
        self._extend_batch_max_items: int = 250
        self._extend_batch_flush_count: int = 0
        self._extend_batch_coalesced_count: int = 0
        self._extend_batch_dropped_count: int = 0
        self._extend_batch_max_depth_seen: int = 0

        # Chart initialization bookkeeping (best-effort; JS is source of truth)
        self._chart_initialized: Dict[str, bool] = {}

        # Readiness polling
        self._ready_poll_attempts: int = 0
        self._ready_poll_max_attempts: int = 120  # ~12s at 100ms interval
        self._ready_poll_interval_ms: int = 100

        try:
            self._view.loadFinished.connect(self._on_load_finished)  # type: ignore[attr-defined]
        except Exception as e:
            # If we can't connect the signal, bridge cannot function, but must not crash.
            msg = f"Failed to connect loadFinished for QWebEngineView: {e!r}"
            self._log.error(msg, extra={"dashboard_component": "plotly_bridge"})
            self.error_occurred.emit(msg)

        self._load_html_template()

    # --------------------------
    # Public methods (API)
    # --------------------------

    def initialize_chart(self, div_id: str, traces: List[Any], layout: Mapping[str, Any]) -> None:
        """
        Initialize a new chart in div_id using Plotly.newPlot.

        - traces: Plotly traces list (JSON-serializable).
        - layout: Plotly layout dict (JSON-serializable).
        """
        div = self._norm_div_id(div_id)
        payload = {"divId": div, "traces": traces, "layout": dict(layout)}
        js = self._js_call("initializeChart", payload)
        self._chart_initialized[div] = True  # optimistic (JS may still fail; errors logged via callback)
        self._enqueue_or_run(js, desc=f"initialize_chart(div_id={div})")

    def update_trace(
        self,
        div_id: str,
        trace_index: int,
        x_data: Any,
        y_data: Any,
        max_points: int = 5000,
    ) -> None:
        """
        Replace/overwrite trace data (full update).

        This uses Plotly.update internally. For complex trace types (candlestick),
        use update_dict APIs (extend_trace with update dict) from the caller.

        x_data and y_data should be list-like.
        """
        div = self._norm_div_id(div_id)
        payload = {
            "divId": div,
            "traceIndex": int(trace_index),
            "x": x_data,
            "y": y_data,
            "maxPoints": int(max_points),
        }
        js = self._js_call("updateTrace", payload)
        self._enqueue_or_run(js, desc=f"update_trace(div_id={div}, idx={trace_index})", requires_init=True, div_id=div)

    def extend_trace(
        self,
        div_id: str,
        trace_index: int,
        update_dict: Mapping[str, Any],
        max_points: int = 5000,
    ) -> None:
        """
        Append data to an existing trace using Plotly.extendTraces.

        Parameters
        ----------
        update_dict:
            Dict of arrays to append, e.g. for scatter:
                {"x": [[x_point]], "y": [[y_point]]}
            For candlestick:
                {"x": [[ts]], "open": [[o]], "high": [[h]], "low": [[l]], "close": [[c]]}

            Note: Plotly expects each property to be a list-of-lists matching trace indices.
        """
        div = self._norm_div_id(div_id)
        payload = {
            "divId": div,
            "traceIndex": int(trace_index),
            "update": dict(update_dict),
            "maxPoints": int(max_points),
        }
        desc = f"extend_trace(div_id={div}, idx={trace_index})"
        # Week 08 Mon: when ready, high-frequency append traffic goes through
        # the batching lane.  Before page/chart readiness we preserve existing
        # pending-command behavior so initialization ordering stays deterministic.
        if self._ready and self._chart_initialized.get(div, False):
            self._enqueue_extend_batch(payload, desc=desc)
            return

        js = self._js_call("extendTrace", payload)
        self._enqueue_or_run(js, desc=desc, requires_init=True, div_id=div)

    def update_layout(self, div_id: str, layout_updates: Mapping[str, Any]) -> None:
        """Update chart layout via Plotly.relayout."""
        div = self._norm_div_id(div_id)
        payload = {"divId": div, "layout": dict(layout_updates)}
        js = self._js_call("updateLayout", payload)
        self._enqueue_or_run(js, desc=f"update_layout(div_id={div})", requires_init=True, div_id=div)

    def resize(self, div_id: Optional[str] = None) -> None:
        """
        Trigger chart resize.

        If div_id is None, resizes all charts known to the JS page.
        """
        payload = {"divId": self._norm_div_id(div_id) if div_id else None}
        js = self._js_call("resize", payload)
        self._enqueue_or_run(js, desc=f"resize(div_id={payload['divId']})")

    # --------------------------
    # HTML + readiness handling
    # --------------------------

    def _load_html_template(self) -> None:
        try:
            asset_path, asset_name = self._resolve_plotly_bundle_path()
            html = self._html_template(plotly_asset_name=asset_name)
            # Use the assets directory as baseUrl so the HTML can reference the
            # vendored Plotly file by relative path. This avoids embedding a
            # multi-megabyte inline script blob, which was causing unreliable
            # QWebEngine page load failures in the full dashboard runtime.
            self._view.setHtml(html, baseUrl=self._assets_base_url(asset_path.parent))
            self._log.info(
                "PlotlyBridge HTML template loaded into QWebEngineView",
                extra={"dashboard_component": "plotly_bridge", "plotly_asset": asset_name},
            )
        except Exception as e:
            msg = f"Failed to load Plotly HTML template: {e!r}"
            self._log.error(msg, extra={"dashboard_component": "plotly_bridge"})
            self.error_occurred.emit(msg)

    def _assets_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "assets"

    @staticmethod
    def _assets_base_url(assets_dir: Path) -> QUrl:
        base = str(assets_dir.resolve())
        if not base.endswith(("/", "\\")):
            base += "/"
        return QUrl.fromLocalFile(base)

    def _resolve_plotly_bundle_path(self) -> tuple[Path, str]:
        """Resolve a vendored local Plotly bundle path.

        Part 2B hardening rule: the dashboard chart runtime must not depend on
        CDN availability. We therefore load Plotly from a local asset file via a
        relative <script src=...> reference rather than injecting a huge inline
        bundle into the page.
        """
        assets_dir = self._assets_dir()
        configured_path = None
        if Config is not None:
            try:
                configured = Config.get("dashboard", "plotly_bundle", default=None)
                if isinstance(configured, str) and configured.strip():
                    configured_path = Path(configured.strip())
                    if not configured_path.is_absolute():
                        configured_path = Path(__file__).resolve().parents[2] / configured_path
            except Exception:
                configured_path = None

        candidates = []
        if configured_path is not None:
            candidates.append(configured_path)
        candidates.extend(sorted(assets_dir.glob("plotly-*.min.js"), reverse=True))

        seen = set()
        for asset_path in candidates:
            try:
                resolved_key = str(asset_path.resolve())
            except Exception:
                resolved_key = str(asset_path)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            if not asset_path.exists() or not asset_path.is_file():
                continue
            if asset_path.stat().st_size <= 0:
                continue
            head = asset_path.read_text(encoding="utf-8", errors="ignore")[:4096]
            if "plotly" not in head.lower():
                continue
            return asset_path, asset_path.name

        raise FileNotFoundError("No valid local Plotly bundle found under dashboard/assets")

    def _on_load_finished(self, ok: bool) -> None:
        self._page_loaded = bool(ok)
        self._ready = False
        self._ready_poll_attempts = 0

        if not ok:
            msg = "QWebEngineView reported loadFinished=False (HTML page failed to load)."
            self._log.error(msg, extra={"dashboard_component": "plotly_bridge"})
            self.error_occurred.emit(msg)
            return

        self._log.info("QWebEngineView loadFinished=True; polling Plotly readiness", extra={"dashboard_component": "plotly_bridge"})
        self._poll_ready()

    def _poll_ready(self) -> None:
        """
        Poll JS environment until Plotly is loaded and bridge functions are installed.

        This avoids racing where loadFinished is True but external script is still loading.
        """
        if not self._page_loaded:
            return
        if self._ready:
            return

        self._ready_poll_attempts += 1
        if self._ready_poll_attempts > self._ready_poll_max_attempts:
            msg = "PlotlyBridge readiness timeout: Plotly did not become available."
            self._log.error(msg, extra={"dashboard_component": "plotly_bridge"})
            self.error_occurred.emit(msg)
            return

        js = "window.DashboardPlotlyBridge && window.DashboardPlotlyBridge.isReady ? window.DashboardPlotlyBridge.isReady() : false;"
        self._run_js(js, desc="poll_ready()", on_result=self._on_ready_polled)

    def _on_ready_polled(self, result: Any) -> None:
        if bool(result):
            self._ready = True
            self._log.info("PlotlyBridge ready (Plotly loaded + bridge installed)", extra={"dashboard_component": "plotly_bridge"})
            try:
                self.page_ready.emit()
            except Exception:
                # Do not crash if signal emission misbehaves in host environment.
                pass
            self._flush_pending()
            return

        QTimer.singleShot(self._ready_poll_interval_ms, self._poll_ready)

    # --------------------------
    # Queueing and execution
    # --------------------------

    def _enqueue_extend_batch(self, payload: Mapping[str, Any], desc: str) -> None:
        """Coalesce extendTrace payloads for a short batching window.

        Week 08 Mon performance contract:
        - no backend/chart business logic here;
        - no data aggregation beyond concatenating already-prepared Plotly
          append arrays for the same trace;
        - bounded queue protects replay/high-frequency bursts from unbounded
          memory growth.
        """
        div = self._norm_div_id(str(payload.get("divId", "chart-div")))
        trace_index = int(payload.get("traceIndex", 0))
        max_points = int(payload.get("maxPoints", 5000))
        update = self._normalise_extend_update(payload.get("update", {}))
        if not update:
            return

        key = (div, trace_index)
        existing = self._extend_batches.get(key)
        if existing is None:
            if len(self._extend_batches) >= self._extend_batch_max_items:
                # Drop the oldest batch to preserve freshest operator-visible data.
                oldest_key = min(
                    self._extend_batches,
                    key=lambda k: self._extend_batches[k].created_ns,
                )
                dropped = self._extend_batches.pop(oldest_key, None)
                self._extend_batch_dropped_count += 1
                self._log.warning(
                    "PlotlyBridge extend batch overflow; dropped oldest batch div=%s idx=%s",
                    dropped.div_id if dropped else oldest_key[0],
                    dropped.trace_index if dropped else oldest_key[1],
                    extra={"dashboard_component": "plotly_bridge"},
                )
            self._extend_batches[key] = _PendingExtendTrace(
                div_id=div,
                trace_index=trace_index,
                update=update,
                max_points=max_points,
                created_ns=time.time_ns(),
            )
        else:
            merged = self._merge_extend_updates(existing.update, update)
            self._extend_batches[key] = _PendingExtendTrace(
                div_id=existing.div_id,
                trace_index=existing.trace_index,
                update=merged,
                max_points=max(existing.max_points, max_points),
                created_ns=existing.created_ns,
            )
            self._extend_batch_coalesced_count += 1

        self._extend_batch_max_depth_seen = max(self._extend_batch_max_depth_seen, len(self._extend_batches))
        if not self._extend_batch_scheduled:
            self._extend_batch_scheduled = True
            QTimer.singleShot(self._extend_batch_window_ms, self._flush_extend_batches)

    def _flush_extend_batches(self) -> None:
        if not self._extend_batches:
            self._extend_batch_scheduled = False
            return
        if not self._ready:
            # Page became unavailable between enqueue and flush; defer until it
            # is ready again.  We keep the coalesced payloads instead of emitting
            # stale/fake chart data.
            QTimer.singleShot(self._ready_poll_interval_ms, self._flush_extend_batches)
            return

        batches = list(self._extend_batches.values())
        self._extend_batches = {}
        self._extend_batch_scheduled = False
        self._extend_batch_flush_count += 1

        payload = {
            "items": [
                {
                    "divId": item.div_id,
                    "traceIndex": item.trace_index,
                    "update": item.update,
                    "maxPoints": item.max_points,
                }
                for item in batches
            ],
            "batchId": self._extend_batch_flush_count,
        }
        js = self._js_call("batchExtendTraces", payload)
        self._run_js(js, desc=f"batch_extend_traces(count={len(batches)})")

    @staticmethod
    def _normalise_extend_update(update: Any) -> Dict[str, Any]:
        if not isinstance(update, Mapping):
            return {}
        out: Dict[str, Any] = {}
        for key, value in update.items():
            # Plotly.extendTraces expects list-of-lists per property.  Preserve
            # values already in that shape; wrap flat lists conservatively.
            if isinstance(value, list):
                if value and all(isinstance(v, list) for v in value):
                    out[str(key)] = [list(v) for v in value]
                else:
                    out[str(key)] = [list(value)]
            else:
                out[str(key)] = [[value]]
        return out

    @staticmethod
    def _merge_extend_updates(base: Mapping[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
        merged = PlotlyBridge._normalise_extend_update(base)
        inc = PlotlyBridge._normalise_extend_update(incoming)
        for prop, value in inc.items():
            if prop not in merged:
                merged[prop] = value
                continue
            # The bridge batches per single trace, so index 0 is the only list
            # used.  Preserve additional lists if callers ever provide them.
            existing_lists = merged[prop]
            incoming_lists = value
            max_len = max(len(existing_lists), len(incoming_lists))
            while len(existing_lists) < max_len:
                existing_lists.append([])
            for idx, vals in enumerate(incoming_lists):
                if isinstance(vals, list):
                    existing_lists[idx].extend(vals)
                else:
                    existing_lists[idx].append(vals)
        return merged

    def get_performance_stats(self) -> Dict[str, Any]:
        """Return lightweight Python-side bridge batching telemetry."""
        return {
            "pending_command_count": len(self._pending),
            "extend_batch_depth": len(self._extend_batches),
            "extend_batch_flush_count": self._extend_batch_flush_count,
            "extend_batch_coalesced_count": self._extend_batch_coalesced_count,
            "extend_batch_dropped_count": self._extend_batch_dropped_count,
            "extend_batch_max_depth_seen": self._extend_batch_max_depth_seen,
            "extend_batch_window_ms": self._extend_batch_window_ms,
        }

    def request_js_performance_stats(self, callback: Optional[Callable[[Any], None]] = None) -> None:
        """Request browser-side RAF/repaint metrics asynchronously.

        Week 08 Thu/Fri profiling hook: QWebEngine JavaScript execution is async,
        so runtime callers can provide a callback for the JS getPerformanceStats
        result.  If the page is not ready, the request is queued through the
        normal bridge path instead of blocking the Qt event loop.
        """
        js = self._js_call("getPerformanceStats", {})
        if not self._ready:
            self._enqueue(js, desc="get_js_performance_stats() [queued: page_not_ready]")
            return
        self._run_js(js, desc="get_js_performance_stats()", on_result=callback)

    def _enqueue_or_run(self, js: str, desc: str, requires_init: bool = False, div_id: Optional[str] = None) -> None:
        """
        Enqueue the JS command if page not ready (or chart not initialized), otherwise run.

        For commands requiring init, if we don't believe the chart is initialized yet,
        queue and retry later (flush occurs on page_ready and after initialize_chart).
        """
        if not self._ready:
            self._enqueue(js, desc=f"{desc} [queued: page_not_ready]")
            return

        if requires_init and div_id:
            if not self._chart_initialized.get(div_id, False):
                self._enqueue(js, desc=f"{desc} [queued: chart_not_initialized]")
                return

        self._run_js(js, desc=desc)

    def _enqueue(self, js: str, desc: str) -> None:
        if len(self._pending) >= self._pending_max:
            # Drop oldest to preserve recent updates in replay/high-frequency mode.
            dropped = self._pending.pop(0)
            self._log.warning(
                "PlotlyBridge pending queue overflow; dropped oldest command: %s",
                dropped.desc,
                extra={"dashboard_component": "plotly_bridge"},
            )
        self._pending.append(_PendingCommand(js=js, desc=desc, created_ns=time.time_ns()))

    def _flush_pending(self) -> None:
        if not self._ready:
            return
        # Flush any coalesced streaming payloads first so chart append state
        # catches up before older generic commands are replayed.
        if self._extend_batches:
            self._flush_extend_batches()
        if not self._pending:
            return

        pending = self._pending
        self._pending = []

        self._log.info(
            "Flushing %d pending PlotlyBridge commands",
            len(pending),
            extra={"dashboard_component": "plotly_bridge"},
        )

        for cmd in pending:
            self._run_js(cmd.js, desc=f"{cmd.desc} [flushed]")

    def _run_js(self, js: str, desc: str, on_result: Optional[Callable[[Any], None]] = None) -> None:
        """
        Run JS on the web page asynchronously and log errors via callback.

        The JS snippets in this module return either:
          - {"ok": true, "result": ...}
          - {"ok": false, "error": "..."}
          - or a raw value (for simple probes)
        """
        try:
            page = self._view.page()
        except Exception as e:
            self._log.error(
                "Cannot access QWebEngineView.page(); dropping command. desc=%s err=%s",
                desc,
                repr(e),
                extra={"dashboard_component": "plotly_bridge"},
            )
            return

        def _callback(result: Any) -> None:
            try:
                # If our JS wrapper returns a dict with ok/error fields, honor it.
                if isinstance(result, dict) and result.get("ok") is False:
                    err = result.get("error", "unknown JS error")
                    self._log.error(
                        "PlotlyBridge JS error. desc=%s err=%s",
                        desc,
                        err,
                        extra={"dashboard_component": "plotly_bridge"},
                    )
                    try:
                        self.log_message.emit(f"{desc}: {err}", "ERROR")
                    except Exception:
                        pass
                if on_result is not None:
                    on_result(result)
            except Exception as e:
                self._log.warning(
                    "PlotlyBridge JS callback handler error. desc=%s err=%s",
                    desc,
                    repr(e),
                    extra={"dashboard_component": "plotly_bridge"},
                )

        try:
            page.runJavaScript(js, _callback)
        except Exception as e:
            self._log.error(
                "runJavaScript failed; command dropped. desc=%s err=%s",
                desc,
                repr(e),
                extra={"dashboard_component": "plotly_bridge"},
            )

    # --------------------------
    # JS + HTML generation
    # --------------------------

    def _js_call(self, method: str, payload_obj: Mapping[str, Any]) -> str:
        """
        Build a safe JS call that:
        - JSON.parse() the payload string
        - calls DashboardPlotlyBridge[method](payload)
        - returns {ok, result/error}
        """
        payload_json = self._safe_json(payload_obj)
        payload_literal = json.dumps(payload_json)  # JS string literal containing JSON
        method_literal = json.dumps(str(method))

        # No eval; only JSON.parse.
        return f"""
(function() {{
  try {{
    const payload = JSON.parse({payload_literal});
    const m = {method_literal};
    if (!window.DashboardPlotlyBridge || typeof window.DashboardPlotlyBridge[m] !== "function") {{
      return {{ok: false, error: "DashboardPlotlyBridge method not available: " + m}};
    }}
    return window.DashboardPlotlyBridge[m](payload);
  }} catch (e) {{
    return {{ok: false, error: String(e)}};
  }}
}})();
""".strip()

    def _html_template(self, *, plotly_asset_name: str) -> str:
        """
        Minimal HTML template that:
        - loads the vendored local Plotly bundle by relative file path
        - installs window.DashboardPlotlyBridge with required methods
        - supports multiple charts via ensureDiv()
        """
        # Keep HTML/JS self-contained except for the local Plotly asset file.
        # This dramatically reduces page size and avoids the unreliable
        # loadFinished=False behavior caused by injecting the whole bundle inline.
        template = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard Plotly Bridge</title>
  <style>
    html, body {
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      background: #111;
      overflow: hidden;
      font-family: sans-serif;
    }
    #root {
      width: 100%;
      height: 100%;
      display: flex;
      flex-direction: column;
      align-items: stretch;
      justify-content: stretch;
    }
    .chart-container {
      width: 100%;
      height: 100%;
      min-height: 100px;
    }
  </style>
  <script src="{plotly_asset_name}"></script>
  <script>
  (function() {
    // Simple internal registry
    const charts = {}; // divId -> true
    // Week 08 Tue — requestAnimationFrame batching.
    // Python batches reduce runJavaScript calls; this browser-side queue then
    // coalesces all extendTraces writes that arrive before the next paint.  It
    // keeps DOM/repaint churn bounded while preserving backend-provided chart
    // truth exactly as received.
    const rafQueue = [];
    const RAF_QUEUE_MAX_ITEMS = 500;
    let rafScheduled = false;
    const rafStats = {
      queuedBatches: 0,
      queuedItems: 0,
      appliedBatches: 0,
      appliedItems: 0,
      droppedItems: 0,
      maxQueueDepth: 0,
      lastError: null,
      lastFlushMs: 0
    };
    const lifecycleStats = {
      newPlotCount: 0,
      reactCount: 0,
      relayoutCount: 0,
      resizeCount: 0
    };
    const scheduleFrame = window.requestAnimationFrame || function(cb) { return window.setTimeout(cb, 16); };

    function normalizeUpdate(update) {
      const out = {};
      update = update || {};
      for (const prop in update) {
        const value = update[prop];
        if (Array.isArray(value)) {
          if (value.length > 0 && value.every(Array.isArray)) {
            out[prop] = value.map(function(v) { return v.slice(); });
          } else {
            out[prop] = [value.slice()];
          }
        } else {
          out[prop] = [[value]];
        }
      }
      return out;
    }

    function mergeUpdate(base, incoming) {
      const merged = normalizeUpdate(base);
      const inc = normalizeUpdate(incoming);
      for (const prop in inc) {
        if (!merged[prop]) {
          merged[prop] = inc[prop];
          continue;
        }
        const existingLists = merged[prop];
        const incomingLists = inc[prop];
        while (existingLists.length < incomingLists.length) existingLists.push([]);
        for (let i = 0; i < incomingLists.length; i++) {
          existingLists[i].push.apply(existingLists[i], incomingLists[i]);
        }
      }
      return merged;
    }

    function scheduleExtendItems(items, batchId) {
      if (!Array.isArray(items)) {
        throw new Error("batchExtendTraces requires items array");
      }
      for (let i = 0; i < items.length; i++) {
        const item = items[i] || {};
        const divId = item.divId;
        ensureDiv(divId);
        if (!charts[divId]) {
          throw new Error("Chart not initialized for divId=" + divId);
        }
        rafQueue.push({
          divId: divId,
          traceIndex: item.traceIndex,
          update: normalizeUpdate(item.update || {}),
          maxPoints: item.maxPoints,
          batchId: batchId || null
        });
      }
      if (rafQueue.length > RAF_QUEUE_MAX_ITEMS) {
        const dropCount = rafQueue.length - RAF_QUEUE_MAX_ITEMS;
        rafQueue.splice(0, dropCount);
        rafStats.droppedItems += dropCount;
      }
      rafStats.queuedBatches += 1;
      rafStats.queuedItems += items.length;
      rafStats.maxQueueDepth = Math.max(rafStats.maxQueueDepth, rafQueue.length);
      if (!rafScheduled) {
        rafScheduled = true;
        scheduleFrame(flushRafQueue);
      }
      return {queued: items.length, rafQueueDepth: rafQueue.length, batchId: batchId || null};
    }

    function flushRafQueue() {
      const started = (window.performance && window.performance.now) ? window.performance.now() : Date.now();
      rafScheduled = false;
      if (!rafQueue.length) return;
      const Plotly = safePlotly();
      const items = rafQueue.splice(0, rafQueue.length);
      const grouped = new Map();
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const key = item.divId + "::" + item.traceIndex;
        const existing = grouped.get(key);
        if (!existing) {
          grouped.set(key, {
            divId: item.divId,
            traceIndex: item.traceIndex,
            update: normalizeUpdate(item.update),
            maxPoints: item.maxPoints
          });
        } else {
          existing.update = mergeUpdate(existing.update, item.update);
          existing.maxPoints = Math.max(existing.maxPoints || 0, item.maxPoints || 0) || item.maxPoints;
        }
      }
      try {
        grouped.forEach(function(item) {
          Plotly.extendTraces(item.divId, item.update, [item.traceIndex], item.maxPoints);
          rafStats.appliedItems += 1;
        });
        rafStats.appliedBatches += 1;
        rafStats.lastError = null;
      } catch (e) {
        rafStats.lastError = String(e);
        try { console.error("DashboardPlotlyBridge RAF flush error", e); } catch (_err) {}
      } finally {
        const ended = (window.performance && window.performance.now) ? window.performance.now() : Date.now();
        rafStats.lastFlushMs = Math.max(0, ended - started);
      }
    }

    function ensureDiv(divId) {
      if (!divId || typeof divId !== "string") {
        throw new Error("Invalid divId");
      }
      let el = document.getElementById(divId);
      if (!el) {
        const root = document.getElementById("root");
        if (!root) throw new Error("Root container not found");
        el = document.createElement("div");
        el.id = divId;
        el.className = "chart-container";
        root.appendChild(el);
      }
      return el;
    }

    function safePlotly() {
      if (!window.Plotly) {
        throw new Error("Plotly not loaded");
      }
      return window.Plotly;
    }

    window.DashboardPlotlyBridge = {
      isReady: function() {
        return !!window.Plotly && !!window.Plotly.newPlot && !!window.Plotly.extendTraces && typeof scheduleExtendItems === "function";
      },

      initializeChart: function(payload) {
        try {
          const Plotly = safePlotly();
          const divId = payload.divId;
          const traces = payload.traces || [];
          const layout = payload.layout || {};
          const el = ensureDiv(divId);

          // Add responsive config; allow panels to override later via updateLayout.
          const config = {
            displaylogo: false,
            responsive: true,
            scrollZoom: true,
            doubleClick: "reset"
          };

          if (charts[divId] && Plotly.react) {
            // Week 08 Weekend: reduce DOM churn on timeframe switches/reinit by
            // reusing the existing plot container instead of tearing it down.
            Plotly.react(el, traces, layout, config);
            lifecycleStats.reactCount += 1;
          } else {
            Plotly.newPlot(el, traces, layout, config);
            lifecycleStats.newPlotCount += 1;
          }
          charts[divId] = true;
          return {ok: true, result: true};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      updateTrace: function(payload) {
        try {
          const Plotly = safePlotly();
          const divId = payload.divId;
          const idx = payload.traceIndex;
          const x = payload.x;
          const y = payload.y;
          const maxPoints = payload.maxPoints;

          ensureDiv(divId);
          if (!charts[divId]) {
            throw new Error("Chart not initialized for divId=" + divId);
          }

          // Basic update. For advanced traces, use extendTrace with update dict from Python.
          const update = {x: [x], y: [y]};
          Plotly.update(divId, update, {}, [idx]);

          // maxPoints not enforced here; for streaming use extendTrace which supports maxPoints.
          return {ok: true, result: true};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      extendTrace: function(payload) {
        try {
          const Plotly = safePlotly();
          const divId = payload.divId;
          const idx = payload.traceIndex;
          const update = payload.update || {};
          const maxPoints = payload.maxPoints;

          ensureDiv(divId);
          if (!charts[divId]) {
            throw new Error("Chart not initialized for divId=" + divId);
          }

          const scheduled = scheduleExtendItems([{divId: divId, traceIndex: idx, update: update, maxPoints: maxPoints}], null);
          return {ok: true, result: scheduled};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      batchExtendTraces: function(payload) {
        try {
          const scheduled = scheduleExtendItems(payload.items || [], payload.batchId || null);
          return {ok: true, result: scheduled};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      getPerformanceStats: function(_payload) {
        try {
          return {ok: true, result: {
            rafQueueDepth: rafQueue.length,
            rafQueueMaxItems: RAF_QUEUE_MAX_ITEMS,
            rafScheduled: rafScheduled,
            queuedBatches: rafStats.queuedBatches,
            queuedItems: rafStats.queuedItems,
            appliedBatches: rafStats.appliedBatches,
            appliedItems: rafStats.appliedItems,
            droppedItems: rafStats.droppedItems,
            maxQueueDepth: rafStats.maxQueueDepth,
            lastError: rafStats.lastError,
            lastFlushMs: rafStats.lastFlushMs,
            newPlotCount: lifecycleStats.newPlotCount,
            reactCount: lifecycleStats.reactCount,
            relayoutCount: lifecycleStats.relayoutCount,
            resizeCount: lifecycleStats.resizeCount
          }};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      updateLayout: function(payload) {
        try {
          const Plotly = safePlotly();
          const divId = payload.divId;
          const layout = payload.layout || {};

          ensureDiv(divId);
          if (!charts[divId]) {
            throw new Error("Chart not initialized for divId=" + divId);
          }

          Plotly.relayout(divId, layout);
          lifecycleStats.relayoutCount += 1;
          return {ok: true, result: true};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      },

      resize: function(payload) {
        try {
          const Plotly = safePlotly();
          const divId = payload.divId;

          if (divId) {
            ensureDiv(divId);
            if (!charts[divId]) return {ok: true, result: true};
            Plotly.Plots.resize(divId);
            lifecycleStats.resizeCount += 1;
            return {ok: true, result: true};
          }

          // Resize all known charts
          for (const id in charts) {
            try { Plotly.Plots.resize(id); lifecycleStats.resizeCount += 1; } catch (e) {}
          }
          return {ok: true, result: true};
        } catch (e) {
          return {ok: false, error: String(e)};
        }
      }
    };

    // Resize handler (best-effort)
    window.addEventListener("resize", function() {
      try {
        if (window.DashboardPlotlyBridge && window.DashboardPlotlyBridge.resize) {
          window.DashboardPlotlyBridge.resize({divId: null});
        }
      } catch (e) {}
    });
  })();
  </script>
</head>
<body>
  <div id="root">
    <!-- Default chart container (optional); caller may use any divId -->
    <div id="chart-div" class="chart-container"></div>
  </div>
</body>
</html>
""".strip()
        return template.replace("{plotly_asset_name}", plotly_asset_name)

    # --------------------------
    # JSON safety / utilities
    # --------------------------

    def _safe_json(self, obj: Any) -> str:
        """
        Serialize to JSON defensively.

        - Handles common non-JSON types (numpy scalars/arrays) without importing numpy.
        - Falls back to str() as last resort.
        """
        try:
            return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=self._json_default)
        except Exception as e:
            self._log.error(
                "JSON serialization failed; falling back to string payload. err=%s type=%s",
                repr(e),
                type(obj).__name__,
                extra={"dashboard_component": "plotly_bridge"},
            )
            # Last resort: wrap in string so JS side still receives something.
            return json.dumps({"_serialization_error": str(e), "_payload": str(obj)}, ensure_ascii=False)

    @staticmethod
    def _json_default(o: Any) -> Any:
        # numpy scalar: .item()
        if hasattr(o, "item") and callable(getattr(o, "item")):
            try:
                return o.item()
            except Exception:
                pass
        # numpy array / pandas series: .tolist()
        if hasattr(o, "tolist") and callable(getattr(o, "tolist")):
            try:
                return o.tolist()
            except Exception:
                pass
        # dataclasses / objects with __dict__
        if hasattr(o, "__dict__"):
            try:
                return dict(o.__dict__)
            except Exception:
                pass
        return str(o)

    @staticmethod
    def _norm_div_id(div_id: Optional[str]) -> str:
        if div_id is None:
            return "chart-div"
        s = str(div_id).strip()
        return s if s else "chart-div"

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        chart bridge telemetry in the normal project log files while retaining a
        stdlib fallback if logger creation fails.
        """
        name = "dashboard_charts_plotly_bridge"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    view = QWebEngineView()
    win.setCentralWidget(view)
    win.resize(1000, 700)
    win.show()

    bridge = PlotlyBridge(view)
    bridge.page_ready.connect(lambda: print("Page ready"))

    def after_ready() -> None:
        bridge.initialize_chart(
            "chart",
            [{"x": [1, 2, 3], "y": [4, 5, 6], "type": "scatter", "mode": "lines+markers", "name": "demo"}],
            {"title": "Test", "paper_bgcolor": "#111", "plot_bgcolor": "#111", "font": {"color": "#eaeaea"}},
        )

    # Give the web view time to load + Plotly to become available.
    QTimer.singleShot(1000, after_ready)

    raise SystemExit(app.exec())