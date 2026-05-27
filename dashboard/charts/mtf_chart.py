"""
dashboard/charts/mtf_chart.py

Multi-Timeframe (MTF) Candlestick Chart component (Roadmap Week 07).

This widget owns:
- A QWebEngineView hosting Plotly.js
- A PlotlyBridge for safe, queued, async JS calls
- Lightweight UI controls (timeframe selector buttons + status)

It is intentionally "dumb" about market data:
- No aggregation
- No backend queries
- Caller provides prepared OHLC lists and overlays via public methods

Primary contract
----------------
The parent panel (e.g., MtfChartPanel) calls:
- set_timeframe(tf)
- update_candles(tf, ohlc, max_points)
- update_vwap(vwap, upper1, lower1, upper2, lower2)
- update_or_levels(or_high, or_low)
- update_ib_levels(ib_high, ib_low)
- update_smc_objects(...)
- update_cursor(timestamp_str)

Performance strategy
--------------------
- Candles/volume: extendTraces when new candles append.
- Layout overlays (OR/IB/cursor/SMC): relayout(shapes) coalesced via a short QTimer.

Limitations (intentional for Week 07 shell)
-------------------------------------------
- Candlestick "in-progress" updates are not handled; update_candles assumes candles are
  appended on close. If the last candle changes in-place, we will trigger a redraw.
- SMC rendering is best-effort and shape-based; objects without clear rectangle/line
  coordinates are ignored.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import logging
import math

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel

from src.utils.logger import setup_logger


def _load_qt_webengine_runtime() -> tuple[Any, Any]:
    """Load MTF chart runtime dependencies with precise failure reason.

    Why this exists:
    The earlier broad import guard collapsed *any* import/runtime exception into a
    false "QtWebEngine unavailable" message. In practice, that hid real causes
    coming from the Plotly bridge or other import-time failures. We now resolve
    runtime dependencies lazily and preserve the original exception text.
    """
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView as _QWebEngineView
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PyQt6 QtWebEngine import failed: {exc}") from exc

    try:
        from dashboard.charts.qwebengine_plotly_bridge import PlotlyBridge as _PlotlyBridge
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PlotlyBridge import failed: {exc}") from exc

    return _QWebEngineView, _PlotlyBridge


@dataclass
class _Bands:
    vwap: List[float]
    upper1: List[float]
    lower1: List[float]
    upper2: List[float]
    lower2: List[float]


class MtfChart(QWidget):
    runtime_ready = pyqtSignal()
    runtime_error = pyqtSignal(str)

    """
    Multi-timeframe candlestick chart widget.

    Timeframes are symbolic strings. The widget accepts common aliases, e.g.:
      "1m", "1min", "1min", "1"
      "3m", "3min", "3"
      "5m", "5min", "5"
      "15m", "15min", "15"

    Internally stored as: "1m", "3m", "5m", "15m"
    """

    _DIV_ID = "chart-div"  # use the default container provided by PlotlyBridge HTML template

    _TF_ORDER: Tuple[str, ...] = ("1m", "3m", "5m", "15m")

    # Trace indices (fixed at initialization)
    _IDX_CANDLE = 0
    _IDX_VOLUME = 1
    _IDX_VWAP = 2
    _IDX_U1 = 3
    _IDX_L1 = 4
    _IDX_U2 = 5
    _IDX_L2 = 6

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        try:
            QWebEngineView, PlotlyBridge = _load_qt_webengine_runtime()
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        self._log = self._get_logger()
        self._log.info("MtfChart created", extra={"dashboard_component": "mtf_chart"})

        self._bridge_ready: bool = False
        self._chart_initialized: bool = False
        self._bridge_last_error: Optional[str] = None
        self._init_epoch: int = 0  # incremented on timeframe switch to gate stale operations

        self._current_tf: str = "5m"

        # Data caches per timeframe
        self._candles_by_tf: Dict[str, List[Mapping[str, Any]]] = {}
        self._bands_by_tf: Dict[str, _Bands] = {}

        # Overlays/state (not per-tf unless caller chooses; levels often session-level)
        self._or_levels: Optional[Tuple[Optional[float], Optional[float]]] = None
        self._ib_levels: Optional[Tuple[Optional[float], Optional[float]]] = None
        self._cursor_ts: Optional[str] = None

        self._smc_state: Dict[str, Any] = {
            "fvgs": [],
            "obs": [],
            "liquidity_pools": [],
            "bos_choch": [],
            "traps": [],
        }

        # Coalescing timers to prevent relayout spam (cursor at ~200ms)
        self._shape_update_timer = QTimer(self)
        self._shape_update_timer.setSingleShot(True)
        self._shape_update_timer.timeout.connect(self._apply_shapes_now)

        self._pending_shape_apply: bool = False

        # UI
        root = QVBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Toolbar (timeframe selector + status)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(8)

        title = QLabel("MTF CHART")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(max(10, title_font.pointSize()))
        title.setFont(title_font)
        title.setToolTip("Multi-timeframe candlestick chart (Plotly in QWebEngineView).")
        toolbar.addWidget(title, stretch=0)

        toolbar.addSpacing(10)

        self._tf_buttons: Dict[str, QPushButton] = {}
        for tf in self._TF_ORDER:
            b = QPushButton(tf)
            b.setCheckable(True)
            b.setToolTip(f"Switch to {tf} candles")
            b.clicked.connect(lambda _checked: self.set_timeframe(tf))  # late-binding bug avoided below
            self._tf_buttons[tf] = b
            toolbar.addWidget(b, stretch=0)

        # Fix lambda late-binding: reconnect properly
        for tf, b in self._tf_buttons.items():
            try:
                b.clicked.disconnect()
            except Exception:
                pass
            b.clicked.connect(lambda _checked, _tf=tf: self.set_timeframe(_tf))

        toolbar.addStretch(1)

        self._status_label = QLabel("Loading…")
        status_font = QFont("Monospace")
        status_font.setStyleHint(QFont.StyleHint.Monospace)
        status_font.setPointSize(max(9, status_font.pointSize()))
        self._status_label.setFont(status_font)
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setToolTip("Chart status")
        toolbar.addWidget(self._status_label, stretch=0)

        root.addLayout(toolbar)

        # Web view + bridge
        self._view = QWebEngineView(self)
        root.addWidget(self._view, stretch=1)

        self.setLayout(root)

        self._bridge = PlotlyBridge(self._view, parent=self)
        self._bridge.page_ready.connect(self._on_bridge_ready)
        self._bridge.error_occurred.connect(self._on_bridge_error)

        # Initial button state
        self._sync_tf_button_state()
        self._set_status("Loading Plotly…")

    # -----------------------------
    # Public API
    # -----------------------------

    def set_timeframe(self, tf: str) -> None:
        """
        Switch active timeframe and reinitialize chart using cached data.

        Safe under missing data: shows empty chart until update_candles provides candles.
        """
        new_tf = self._normalize_tf(tf)
        if new_tf == self._current_tf:
            return

        self._log.info(
            "MTF timeframe change %s -> %s",
            self._current_tf,
            new_tf,
            extra={"dashboard_component": "mtf_chart"},
        )

        self._current_tf = new_tf
        self._init_epoch += 1
        self._chart_initialized = False
        self._pending_shape_apply = False
        self._shape_update_timer.stop()

        self._sync_tf_button_state()
        self._set_status(f"Switching to {new_tf}…")

        # Reinitialize if bridge ready
        if self._bridge_ready:
            self._initialize_chart(epoch=self._init_epoch)

    def update_candles(self, timeframe: str, ohlc: Sequence[Mapping[str, Any]], max_points: int = 5000) -> None:
        """
        Update candles for a timeframe.

        Parameters
        ----------
        timeframe:
            Timeframe key/alias (e.g., "1m", "5m", "15m").
        ohlc:
            List of dicts with keys:
              timestamp (ISO string), open, high, low, close, volume
        max_points:
            Plotly max points (kept on JS side); we also clamp caches to this size.
        """
        tf = self._normalize_tf(timeframe)

        if not isinstance(ohlc, Sequence):
            self._log.warning(
                "update_candles: ohlc is not a sequence; ignored. tf=%s type=%s",
                tf,
                type(ohlc).__name__,
                extra={"dashboard_component": "mtf_chart"},
            )
            return

        # Clamp to last max_points to avoid unbounded memory
        ohlc_list = [c for c in ohlc if isinstance(c, Mapping)]
        if max_points > 0 and len(ohlc_list) > max_points:
            ohlc_list = ohlc_list[-max_points:]

        prev = self._candles_by_tf.get(tf, [])
        self._candles_by_tf[tf] = ohlc_list

        # Only render if this is the active TF
        if tf != self._current_tf:
            return

        # If bridge not ready yet, initialize later via _on_bridge_ready.
        if not self._bridge_ready:
            self._set_status(f"Waiting for Plotly… ({tf})")
            return

        # If not initialized, initialize full chart (includes vwap/bands if cached).
        if not self._chart_initialized:
            self._initialize_chart(epoch=self._init_epoch)
            return

        # Incremental extension when possible, else redraw
        appended = self._compute_appended(prev, ohlc_list)
        if appended is None:
            # incompatible history -> redraw
            self._log.info(
                "update_candles: history mismatch; reinitializing. tf=%s prev=%d new=%d",
                tf,
                len(prev),
                len(ohlc_list),
                extra={"dashboard_component": "mtf_chart"},
            )
            self._initialize_chart(epoch=self._init_epoch)
            return

        if len(appended) == 0:
            # no new candles
            self._set_status(f"{tf}: {len(ohlc_list)} candles")
            return

        # Extend candles + volume traces
        try:
            xs, opens, highs, lows, closes, vols = self._candle_arrays(appended)
        except Exception as e:
            self._log.warning(
                "update_candles: failed to extract arrays; reinitializing. err=%s",
                repr(e),
                extra={"dashboard_component": "mtf_chart"},
            )
            self._initialize_chart(epoch=self._init_epoch)
            return

        if not xs:
            return

        candle_update = {
            "x": [xs],
            "open": [opens],
            "high": [highs],
            "low": [lows],
            "close": [closes],
        }
        vol_update = {"x": [xs], "y": [vols]}

        self._bridge.extend_trace(self._DIV_ID, self._IDX_CANDLE, candle_update, max_points=max_points)
        self._bridge.extend_trace(self._DIV_ID, self._IDX_VOLUME, vol_update, max_points=max_points)

        # VWAP/bands: if cached for this tf and lengths align, extend the new segment too.
        self._extend_bands_if_possible(tf=tf, start_index=max(0, len(ohlc_list) - len(appended)), max_points=max_points)

        self._set_status(f"{tf}: {len(ohlc_list)} candles (+{len(appended)})")

    def update_vwap(
        self,
        vwap_values: Sequence[Any],
        upper1: Sequence[Any],
        lower1: Sequence[Any],
        upper2: Sequence[Any],
        lower2: Sequence[Any],
    ) -> None:
        """
        Update VWAP and band arrays for the current timeframe.

        Caller should provide arrays aligned to the current timeframe candles.
        """
        tf = self._current_tf
        bands = _Bands(
            vwap=self._to_float_list(vwap_values),
            upper1=self._to_float_list(upper1),
            lower1=self._to_float_list(lower1),
            upper2=self._to_float_list(upper2),
            lower2=self._to_float_list(lower2),
        )
        self._bands_by_tf[tf] = bands

        if not self._bridge_ready:
            return

        if not self._chart_initialized:
            # Will be included on initialization.
            self._initialize_chart(epoch=self._init_epoch)
            return

        # If candles exist, we can try to extend or replace. Prefer extend if lengths increased.
        candles = self._candles_by_tf.get(tf, [])
        xs = [self._as_str(c.get("timestamp"), default="") for c in candles if isinstance(c, Mapping)]
        xs = [x for x in xs if x]

        if not xs:
            self._log.warning(
                "update_vwap: no candles/timestamps available; storing only. tf=%s",
                tf,
                extra={"dashboard_component": "mtf_chart"},
            )
            return

        # Clamp bands to candle length
        n = min(len(xs), len(bands.vwap), len(bands.upper1), len(bands.lower1), len(bands.upper2), len(bands.lower2))
        xs = xs[:n]
        bands = _Bands(
            vwap=bands.vwap[:n],
            upper1=bands.upper1[:n],
            lower1=bands.lower1[:n],
            upper2=bands.upper2[:n],
            lower2=bands.lower2[:n],
        )
        self._bands_by_tf[tf] = bands

        # Extend from previous length if possible
        prev = self._bands_by_tf.get(tf)
        # Note: prev is same object now; we need previous length tracking. Keep a shadow length per tf.
        # Minimal approach: store last rendered length on instance.
        last_len_attr = f"_bands_rendered_len_{tf}"
        prev_len = getattr(self, last_len_attr, 0)
        new_len = n

        if isinstance(prev_len, int) and prev_len > 0 and new_len > prev_len:
            seg_x = xs[prev_len:new_len]
            if seg_x:
                self._bridge.extend_trace(self._DIV_ID, self._IDX_VWAP, {"x": [seg_x], "y": [bands.vwap[prev_len:new_len]]})
                self._bridge.extend_trace(self._DIV_ID, self._IDX_U1, {"x": [seg_x], "y": [bands.upper1[prev_len:new_len]]})
                self._bridge.extend_trace(self._DIV_ID, self._IDX_L1, {"x": [seg_x], "y": [bands.lower1[prev_len:new_len]]})
                self._bridge.extend_trace(self._DIV_ID, self._IDX_U2, {"x": [seg_x], "y": [bands.upper2[prev_len:new_len]]})
                self._bridge.extend_trace(self._DIV_ID, self._IDX_L2, {"x": [seg_x], "y": [bands.lower2[prev_len:new_len]]})
                setattr(self, last_len_attr, new_len)
                return

        # Otherwise: replace full series using update_trace (scatter traces only).
        self._bridge.update_trace(self._DIV_ID, self._IDX_VWAP, xs, bands.vwap, max_points=new_len)
        self._bridge.update_trace(self._DIV_ID, self._IDX_U1, xs, bands.upper1, max_points=new_len)
        self._bridge.update_trace(self._DIV_ID, self._IDX_L1, xs, bands.lower1, max_points=new_len)
        self._bridge.update_trace(self._DIV_ID, self._IDX_U2, xs, bands.upper2, max_points=new_len)
        self._bridge.update_trace(self._DIV_ID, self._IDX_L2, xs, bands.lower2, max_points=new_len)
        setattr(self, last_len_attr, new_len)

    def update_or_levels(self, or_high: Optional[float], or_low: Optional[float]) -> None:
        """Update Opening Range levels (horizontal lines)."""
        self._or_levels = (self._to_float(or_high, default=None), self._to_float(or_low, default=None))
        self._schedule_shapes_apply()

    def update_ib_levels(self, ib_high: Optional[float], ib_low: Optional[float]) -> None:
        """Update Initial Balance levels (horizontal lines)."""
        self._ib_levels = (self._to_float(ib_high, default=None), self._to_float(ib_low, default=None))
        self._schedule_shapes_apply()

    def update_smc_objects(
        self,
        fvgs: Sequence[Any],
        obs: Sequence[Any],
        liquidity_pools: Sequence[Any],
        bos_choch: Sequence[Any],
        traps: Sequence[Any],
    ) -> None:
        """
        Store SMC objects for best-effort rendering as shapes.

        Expected: list-like of dicts. Supported formats are best-effort (see _smc_to_shapes()).
        """
        self._smc_state = {
            "fvgs": list(fvgs) if isinstance(fvgs, Sequence) else [],
            "obs": list(obs) if isinstance(obs, Sequence) else [],
            "liquidity_pools": list(liquidity_pools) if isinstance(liquidity_pools, Sequence) else [],
            "bos_choch": list(bos_choch) if isinstance(bos_choch, Sequence) else [],
            "traps": list(traps) if isinstance(traps, Sequence) else [],
        }
        self._schedule_shapes_apply()

    def update_cursor(self, timestamp_str: Optional[str]) -> None:
        """
        Update the time cursor (vertical line) for the current chart.

        Coalesced via a short timer to prevent relayout spam.
        """
        self._cursor_ts = self._as_str(timestamp_str, default="") or None
        self._schedule_shapes_apply(coalesce_ms=60)

    # -----------------------------
    # Qt overrides
    # -----------------------------

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        try:
            if self._bridge_ready:
                self._bridge.resize(self._DIV_ID)
        except Exception:
            # Never crash UI due to resize events.
            pass

    # -----------------------------
    # Internal: initialization
    # -----------------------------

    def _on_bridge_ready(self) -> None:
        self._bridge_ready = True
        self._bridge_last_error = None
        self._set_status("Plotly ready; initializing…")
        try:
            self.runtime_ready.emit()
        except Exception:
            pass
        self._initialize_chart(epoch=self._init_epoch)

    def _on_bridge_error(self, message: str) -> None:
        self._bridge_last_error = str(message)
        self._bridge_ready = False
        self._chart_initialized = False
        self._set_status("Plotly runtime load failed")
        try:
            self.runtime_error.emit(self._bridge_last_error)
        except Exception:
            pass

    def is_runtime_ready(self) -> bool:
        return bool(self._bridge_ready and self._chart_initialized and not self._bridge_last_error)

    def last_runtime_error(self) -> Optional[str]:
        return self._bridge_last_error

    def _initialize_chart(self, epoch: int) -> None:
        """
        Initialize (or reinitialize) Plotly chart for current timeframe.

        epoch is used to guard against re-entrant timeframe switches.
        """
        if epoch != self._init_epoch:
            return
        if not self._bridge_ready:
            return

        tf = self._current_tf
        candles = self._candles_by_tf.get(tf, [])
        bands = self._bands_by_tf.get(tf)

        traces = self._build_traces(tf=tf, candles=candles, bands=bands)
        layout = self._build_layout(tf=tf, candles=candles)

        self._bridge.initialize_chart(self._DIV_ID, traces, layout)
        self._chart_initialized = True

        # Reset rendered vwap length tracker for this TF
        n_c = len(candles)
        setattr(self, f"_bands_rendered_len_{tf}", min(n_c, len(bands.vwap) if bands else 0))

        self._set_status(f"{tf}: {len(candles)} candles (init)")
        # Apply shapes once more after init (OR/IB/SMC/cursor)
        self._schedule_shapes_apply(coalesce_ms=120)

    def _build_traces(self, tf: str, candles: Sequence[Mapping[str, Any]], bands: Optional[_Bands]) -> List[Dict[str, Any]]:
        xs, opens, highs, lows, closes, vols = self._candle_arrays(candles)

        # VWAP/bands aligned to candle timestamps
        if bands and xs:
            n = min(len(xs), len(bands.vwap), len(bands.upper1), len(bands.lower1), len(bands.upper2), len(bands.lower2))
            bx = xs[:n]
            vwap = bands.vwap[:n]
            u1 = bands.upper1[:n]
            l1 = bands.lower1[:n]
            u2 = bands.upper2[:n]
            l2 = bands.lower2[:n]
        else:
            bx, vwap, u1, l1, u2, l2 = [], [], [], [], [], []

        candle_trace: Dict[str, Any] = {
            "type": "candlestick",
            "name": f"Price ({tf})",
            "x": xs,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "increasing": {"line": {"color": "#26A69A"}},
            "decreasing": {"line": {"color": "#EF5350"}},
            "xaxis": "x",
            "yaxis": "y",
        }

        volume_trace: Dict[str, Any] = {
            "type": "bar",
            "name": "Volume",
            "x": xs,
            "y": vols,
            "marker": {"color": "#607D8B"},
            "opacity": 0.35,
            "xaxis": "x",
            "yaxis": "y2",
        }

        vwap_trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "name": "VWAP",
            "x": bx,
            "y": vwap,
            "line": {"color": "#FFD54F", "width": 1.4},
            "xaxis": "x",
            "yaxis": "y",
        }
        u1_trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "name": "VWAP +1σ",
            "x": bx,
            "y": u1,
            "line": {"color": "#FFECB3", "width": 1, "dash": "dot"},
            "xaxis": "x",
            "yaxis": "y",
        }
        l1_trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "name": "VWAP -1σ",
            "x": bx,
            "y": l1,
            "line": {"color": "#FFECB3", "width": 1, "dash": "dot"},
            "xaxis": "x",
            "yaxis": "y",
        }
        u2_trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "name": "VWAP +2σ",
            "x": bx,
            "y": u2,
            "line": {"color": "#FFE082", "width": 1, "dash": "dash"},
            "xaxis": "x",
            "yaxis": "y",
        }
        l2_trace: Dict[str, Any] = {
            "type": "scatter",
            "mode": "lines",
            "name": "VWAP -2σ",
            "x": bx,
            "y": l2,
            "line": {"color": "#FFE082", "width": 1, "dash": "dash"},
            "xaxis": "x",
            "yaxis": "y",
        }

        return [candle_trace, volume_trace, vwap_trace, u1_trace, l1_trace, u2_trace, l2_trace]

    def _build_layout(self, tf: str, candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        # Dark institutional theme
        shapes = self._build_shapes()

        annotations: List[Dict[str, Any]] = []
        if not candles:
            annotations.append(
                {
                    "text": "No candle data",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"color": "#BDBDBD", "size": 16},
                }
            )

        layout: Dict[str, Any] = {
            "title": {"text": f"MTF Candles — {tf}", "font": {"color": "#EAEAEA", "size": 14}},
            "paper_bgcolor": "#111111",
            "plot_bgcolor": "#111111",
            "margin": {"l": 45, "r": 20, "t": 35, "b": 25},
            "xaxis": {
                "showgrid": True,
                "gridcolor": "#222222",
                "zeroline": False,
                "rangeslider": {"visible": False},
                "tickfont": {"color": "#CFCFCF"},
            },
            # Price axis on top domain
            "yaxis": {
                "domain": [0.25, 1.0],
                "showgrid": True,
                "gridcolor": "#222222",
                "zeroline": False,
                "tickfont": {"color": "#CFCFCF"},
            },
            # Volume axis at bottom domain
            "yaxis2": {
                "domain": [0.0, 0.20],
                "showgrid": True,
                "gridcolor": "#1A1A1A",
                "zeroline": False,
                "tickfont": {"color": "#9E9E9E"},
                "title": {"text": "Vol", "font": {"color": "#9E9E9E", "size": 10}},
            },
            "legend": {"orientation": "h", "y": 1.02, "x": 0, "font": {"color": "#CFCFCF"}},
            "shapes": shapes,
            "annotations": annotations,
        }
        return layout

    # -----------------------------
    # Shapes (OR/IB/Cursor/SMC)
    # -----------------------------

    def _schedule_shapes_apply(self, coalesce_ms: int = 100) -> None:
        if not self._bridge_ready or not self._chart_initialized:
            return
        self._pending_shape_apply = True
        if self._shape_update_timer.isActive():
            return
        self._shape_update_timer.start(max(1, int(coalesce_ms)))

    def _apply_shapes_now(self) -> None:
        if not self._pending_shape_apply:
            return
        self._pending_shape_apply = False
        if not (self._bridge_ready and self._chart_initialized):
            return
        shapes = self._build_shapes()
        self._bridge.update_layout(self._DIV_ID, {"shapes": shapes})

    def _build_shapes(self) -> List[Dict[str, Any]]:
        shapes: List[Dict[str, Any]] = []

        # OR / IB levels: horizontal lines spanning the plot
        def _hline(y: float, color: str, dash: str, width: int, name: str) -> Dict[str, Any]:
            return {
                "type": "line",
                "xref": "paper",
                "x0": 0.0,
                "x1": 1.0,
                "yref": "y",
                "y0": y,
                "y1": y,
                "line": {"color": color, "width": width, "dash": dash},
                "opacity": 0.9,
                "name": name,
            }

        if self._or_levels:
            hi, lo = self._or_levels
            if isinstance(hi, (int, float)) and math.isfinite(float(hi)):
                shapes.append(_hline(float(hi), color="#29B6F6", dash="dash", width=1, name="OR High"))
            if isinstance(lo, (int, float)) and math.isfinite(float(lo)):
                shapes.append(_hline(float(lo), color="#29B6F6", dash="dash", width=1, name="OR Low"))

        if self._ib_levels:
            hi, lo = self._ib_levels
            if isinstance(hi, (int, float)) and math.isfinite(float(hi)):
                shapes.append(_hline(float(hi), color="#AB47BC", dash="dot", width=1, name="IB High"))
            if isinstance(lo, (int, float)) and math.isfinite(float(lo)):
                shapes.append(_hline(float(lo), color="#AB47BC", dash="dot", width=1, name="IB Low"))

        # Cursor: vertical line at current timestamp
        if self._cursor_ts:
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "x0": self._cursor_ts,
                    "x1": self._cursor_ts,
                    "yref": "paper",
                    "y0": 0.0,
                    "y1": 1.0,
                    "line": {"color": "#BDBDBD", "width": 1, "dash": "dot"},
                    "opacity": 0.6,
                    "name": "Cursor",
                }
            )

        # SMC objects: best-effort rectangle/line shapes
        shapes.extend(self._smc_to_shapes(max_shapes=200))

        return shapes

    def _smc_to_shapes(self, max_shapes: int = 200) -> List[Dict[str, Any]]:
        """
        Best-effort conversion from SMC objects to Plotly shapes.

        Supported object formats (dict-like):
        - Rectangle zones: {x0, x1, y0, y1, color?, opacity?, label?}
          where x0/x1 are timestamps and y0/y1 are price levels.
        - Horizontal liquidity lines: {x0, x1, y, ...}
        """
        out: List[Dict[str, Any]] = []

        def _rect(x0: Any, x1: Any, y0: Any, y1: Any, color: str, opacity: float) -> Dict[str, Any]:
            return {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "line": {"color": color, "width": 1},
                "fillcolor": color,
                "opacity": float(max(0.0, min(1.0, opacity))),
                "layer": "below",
            }

        def _line(x0: Any, x1: Any, y: Any, color: str, dash: str, opacity: float) -> Dict[str, Any]:
            return {
                "type": "line",
                "xref": "x",
                "yref": "y",
                "x0": x0,
                "x1": x1,
                "y0": y,
                "y1": y,
                "line": {"color": color, "width": 1, "dash": dash},
                "opacity": float(max(0.0, min(1.0, opacity))),
            }

        # Color palette by category
        palette = {
            "fvgs": ("#00E5FF", 0.12),  # cyan translucent
            "obs": ("#FF6D00", 0.10),   # orange translucent
            "liquidity_pools": ("#76FF03", 0.10),  # green translucent
            "bos_choch": ("#FF1744", 0.10),  # red/pink translucent
            "traps": ("#FFD600", 0.10),  # yellow translucent
        }

        total = 0
        for key, items in self._smc_state.items():
            if not isinstance(items, list):
                continue
            color, default_op = palette.get(key, ("#9E9E9E", 0.08))

            for obj in items:
                if total >= max_shapes:
                    self._log.warning(
                        "SMC shapes capped at %d; additional shapes ignored.",
                        max_shapes,
                        extra={"dashboard_component": "mtf_chart"},
                    )
                    return out
                if not isinstance(obj, Mapping):
                    continue

                # rectangle
                if all(k in obj for k in ("x0", "x1", "y0", "y1")):
                    out.append(
                        _rect(
                            obj.get("x0"),
                            obj.get("x1"),
                            obj.get("y0"),
                            obj.get("y1"),
                            color=str(obj.get("color", color)),
                            opacity=float(obj.get("opacity", default_op)),
                        )
                    )
                    total += 1
                    continue

                # liquidity line
                if all(k in obj for k in ("x0", "x1", "y")):
                    out.append(
                        _line(
                            obj.get("x0"),
                            obj.get("x1"),
                            obj.get("y"),
                            color=str(obj.get("color", color)),
                            dash=str(obj.get("dash", "dot")),
                            opacity=float(obj.get("opacity", 0.8)),
                        )
                    )
                    total += 1
                    continue

                # Unsupported format: ignore (no heavy inference/business logic)
        return out

    # -----------------------------
    # VWAP extension helper
    # -----------------------------

    def _extend_bands_if_possible(self, tf: str, start_index: int, max_points: int) -> None:
        bands = self._bands_by_tf.get(tf)
        if not bands:
            return

        candles = self._candles_by_tf.get(tf, [])
        xs = [self._as_str(c.get("timestamp"), default="") for c in candles if isinstance(c, Mapping)]
        xs = [x for x in xs if x]
        if not xs:
            return

        n = min(len(xs), len(bands.vwap), len(bands.upper1), len(bands.lower1), len(bands.upper2), len(bands.lower2))
        if start_index >= n:
            return

        seg_x = xs[start_index:n]
        if not seg_x:
            return

        self._bridge.extend_trace(self._DIV_ID, self._IDX_VWAP, {"x": [seg_x], "y": [bands.vwap[start_index:n]]}, max_points=max_points)
        self._bridge.extend_trace(self._DIV_ID, self._IDX_U1, {"x": [seg_x], "y": [bands.upper1[start_index:n]]}, max_points=max_points)
        self._bridge.extend_trace(self._DIV_ID, self._IDX_L1, {"x": [seg_x], "y": [bands.lower1[start_index:n]]}, max_points=max_points)
        self._bridge.extend_trace(self._DIV_ID, self._IDX_U2, {"x": [seg_x], "y": [bands.upper2[start_index:n]]}, max_points=max_points)
        self._bridge.extend_trace(self._DIV_ID, self._IDX_L2, {"x": [seg_x], "y": [bands.lower2[start_index:n]]}, max_points=max_points)

        setattr(self, f"_bands_rendered_len_{tf}", n)

    # -----------------------------
    # Utils
    # -----------------------------

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        MTF chart telemetry in the normal project log files while retaining a
        stdlib fallback if logger creation fails.
        """
        name = "dashboard_charts_mtf_chart"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)

    def _sync_tf_button_state(self) -> None:
        for tf, b in self._tf_buttons.items():
            b.setChecked(tf == self._current_tf)

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    @classmethod
    def _normalize_tf(cls, tf: str) -> str:
        s = str(tf).strip().lower()
        # allow "1", "1m", "1min", "1minute"
        if s in {"1", "1m", "1min", "1mins", "1minute", "1minutes"}:
            return "1m"
        if s in {"3", "3m", "3min", "3mins", "3minute", "3minutes"}:
            return "3m"
        if s in {"5", "5m", "5min", "5mins", "5minute", "5minutes"}:
            return "5m"
        if s in {"15", "15m", "15min", "15mins", "15minute", "15minutes"}:
            return "15m"
        # fallback: accept already normalized or unknown
        if s in set(cls._TF_ORDER):
            return s
        return "5m"

    @staticmethod
    def _as_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        try:
            s = str(value)
            return s if s else default
        except Exception:
            return default

    @staticmethod
    def _to_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        if value is None:
            return default
        try:
            v = float(value)
            if not math.isfinite(v):
                return default
            return v
        except Exception:
            return default

    @staticmethod
    def _to_float_list(seq: Sequence[Any]) -> List[float]:
        out: List[float] = []
        if not isinstance(seq, Sequence):
            return out
        for x in seq:
            try:
                v = float(x)
                if math.isfinite(v):
                    out.append(v)
            except Exception:
                out.append(float("nan"))
        return out

    @staticmethod
    def _candle_arrays(candles: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[float], List[float], List[float], List[float], List[float]]:
        xs: List[str] = []
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        vols: List[float] = []

        for c in candles:
            if not isinstance(c, Mapping):
                continue
            ts = c.get("timestamp", None)
            if ts is None:
                continue
            tss = str(ts)
            if not tss:
                continue

            def _f(k: str) -> float:
                v = c.get(k, None)
                try:
                    return float(v)
                except Exception:
                    return float("nan")

            xs.append(tss)
            opens.append(_f("open"))
            highs.append(_f("high"))
            lows.append(_f("low"))
            closes.append(_f("close"))
            vols.append(_f("volume"))

        return xs, opens, highs, lows, closes, vols

    @staticmethod
    def _compute_appended(prev: Sequence[Mapping[str, Any]], new: Sequence[Mapping[str, Any]]) -> Optional[List[Mapping[str, Any]]]:
        """
        Decide if `new` is `prev` with appended candles.

        Returns:
            - list of appended candle dicts (possibly empty) if append-compatible
            - None if histories appear incompatible (requires redraw)
        """
        if not prev:
            # treat as full init, not incremental
            return list(new)

        if not new:
            return None

        # If new shorter than prev: not append-compatible
        if len(new) < len(prev):
            return None

        # Compare first timestamp to ensure same window (cheap guard)
        try:
            if str(new[0].get("timestamp")) != str(prev[0].get("timestamp")):
                return None
        except Exception:
            return None

        # Check that the last item of prev matches the corresponding index in new.
        try:
            if str(new[len(prev) - 1].get("timestamp")) != str(prev[-1].get("timestamp")):
                return None
        except Exception:
            return None

        # Append segment
        if len(new) == len(prev):
            return []
        return list(new[len(prev) :])

    # -----------------------------
    # Self-test harness
    # -----------------------------


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    chart = MtfChart(win)
    win.setCentralWidget(chart)
    win.resize(1200, 750)
    win.show()

    def test() -> None:
        # Provide 5m candles
        chart.update_candles(
            "5m",
            [
                {"timestamp": "2026-05-21T10:00:00", "open": 24500, "high": 24520, "low": 24490, "close": 24510, "volume": 1000},
                {"timestamp": "2026-05-21T10:05:00", "open": 24510, "high": 24530, "low": 24500, "close": 24525, "volume": 1200},
            ],
        )
        chart.update_vwap(
            [24505, 24512],
            [24510, 24518],
            [24500, 24506],
            [24515, 24522],
            [24495, 24502],
        )
        chart.update_or_levels(24525, 24495)
        chart.update_ib_levels(24520, 24500)
        chart.update_cursor("2026-05-21T10:05:00")

        # Simulate next candle append
        chart.update_candles(
            "5m",
            [
                {"timestamp": "2026-05-21T10:00:00", "open": 24500, "high": 24520, "low": 24490, "close": 24510, "volume": 1000},
                {"timestamp": "2026-05-21T10:05:00", "open": 24510, "high": 24530, "low": 24500, "close": 24525, "volume": 1200},
                {"timestamp": "2026-05-21T10:10:00", "open": 24525, "high": 24540, "low": 24510, "close": 24518, "volume": 900},
            ],
        )
        chart.update_vwap(
            [24505, 24512, 24514],
            [24510, 24518, 24522],
            [24500, 24506, 24508],
            [24515, 24522, 24528],
            [24495, 24502, 24504],
        )
        chart.update_cursor("2026-05-21T10:10:00")

    QTimer.singleShot(2000, test)

    raise SystemExit(app.exec())