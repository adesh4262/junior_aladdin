"""MTF chart panel shell for Week 07 Wednesday.

This file embeds ``dashboard.charts.mtf_chart.MtfChart`` in a dashboard panel
surface.  It intentionally stays UI-only:

- no backend calls
- no candle aggregation
- no alpha/strategy calculations
- caller provides already-prepared MTF candle snapshots

The registry/headless adapter lives in ``dashboard.panels.catalog`` so default
registry rendering stays PyQt-free in headless environments.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.utils.logger import setup_logger

try:
    from dashboard.charts.mtf_chart import MtfChart
except Exception:  # pragma: no cover - depends on PyQt6-WebEngine/system libs
    MtfChart = None  # type: ignore[assignment,misc]


_ON_BG_INSET = "#102131"
_ON_BG_TEXT = "#EAF2FF"
_ON_BG_AMBER = "#2B2000"
_ON_FG_AMBER = "#FFB020"
_ON_BG_RED = "#2B0000"
_ON_FG_RED = "#FF6161"
_ON_BORDER = "#163043"


class MtfChartPanel(QWidget):
    """Visible PyQt panel wrapper around the MTF chart widget."""

    panel_id: str = "mtf_chart"
    title: str = "MTF CHART"
    refresh_class: str = "warm"
    required_snapshot_keys = ["mtf_candles"]
    default_visible: bool = True

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._log = self._get_logger()

        self.setObjectName("MtfChartPanel")
        self.setStyleSheet(
            "QWidget#MtfChartPanel {"
            f" background-color: {_ON_BG_INSET};"
            f" border: 1px solid {_ON_BORDER};"
            " border-radius: 8px;"
            " }"
        )

        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QLabel("MTF CHART SHELL")
        header_font = QFont()
        header_font.setBold(True)
        header_font.setPointSize(max(11, header_font.pointSize()))
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setStyleSheet(f"color: {_ON_BG_TEXT};")
        root.addWidget(header)

        self._status_label = QLabel("Waiting for MTF candle snapshot")
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setStyleSheet(self._css_status(_ON_BG_AMBER, _ON_FG_AMBER))
        root.addWidget(self._status_label)

        self._chart: Any = None
        self._chart_runtime_error: Optional[str] = None
        if MtfChart is None:
            # Keep preview/degraded operation possible on machines missing
            # PyQt6-WebEngine.  The panel surface remains visible and explicit.
            fallback = QLabel("PyQt6-WebEngine unavailable — chart widget disabled")
            fallback.setWordWrap(True)
            fallback.setStyleSheet(self._css_status(_ON_BG_RED, _ON_FG_RED))
            root.addWidget(fallback, stretch=1)
            self._log.warning(
                "MtfChartPanel created without MtfChart (QtWebEngine unavailable)",
                dashboard_component="panel_mtf_chart",
            )
        else:
            try:
                self._chart = MtfChart(self)
                if hasattr(self._chart, "runtime_error"):
                    self._chart.runtime_error.connect(self._on_chart_runtime_error)
                if hasattr(self._chart, "runtime_ready"):
                    self._chart.runtime_ready.connect(self._on_chart_runtime_ready)
                root.addWidget(self._chart, stretch=1)
            except Exception as exc:
                fallback = QLabel(f"MTF chart failed to initialize: {exc}")
                fallback.setWordWrap(True)
                fallback.setStyleSheet(self._css_status(_ON_BG_RED, _ON_FG_RED))
                root.addWidget(fallback, stretch=1)
                self._chart = None
                self._log.error(
                    "MtfChartPanel chart initialization failed",
                    dashboard_component="panel_mtf_chart",
                    error=str(exc),
                )

        self.setLayout(root)
        self._log.info("MtfChartPanel created", dashboard_component="panel_mtf_chart")

    def _on_chart_runtime_error(self, message: str) -> None:
        self._chart_runtime_error = str(message)
        self._set_status("Plotly runtime load failed", critical=True)

    def _on_chart_runtime_ready(self) -> None:
        self._chart_runtime_error = None
        self._set_status("Plotly runtime ready", critical=False)

    def update_warm(self, data: Any) -> None:
        """Update chart from projected warm snapshot data.

        Expected shape:
            {"mtf_candles": {"1m": [...], "3m": [...], "5m": [...], "15m": [...]}}

        The panel only forwards prepared candle lists to MtfChart.  It never
        aggregates candles or queries backend state.
        """
        try:
            if not isinstance(data, Mapping):
                self._set_status("Invalid warm snapshot for MTF chart", critical=True)
                return

            candles_by_tf = self._extract_candles_by_tf(data)
            if not candles_by_tf:
                self._set_status("MTF candles missing", critical=False)
                return

            if self._chart is None:
                self._set_status("Chart widget unavailable", critical=True)
                return

            active_tf = str(data.get("active_timeframe", data.get("timeframe", "5m")))
            self._chart.set_timeframe(active_tf)

            updated = 0
            for tf, candles in candles_by_tf.items():
                if isinstance(candles, list):
                    self._chart.update_candles(tf, candles)
                    updated += len(candles)

            # Week 07 Thursday overlay wiring:
            # Backend/projection owns VWAP/OR/IB computation.  This panel only
            # forwards already prepared arrays/levels to MtfChart.  Missing
            # overlays are left absent; we never invent chart levels.
            overlays_applied = []
            bands = self._extract_vwap_bands(data, active_tf)
            if bands is not None:
                self._chart.update_vwap(*bands)
                overlays_applied.append("VWAP")

            or_levels = self._extract_level_pair(data, "or_high", "or_low", container_key="or_levels")
            if or_levels is not None:
                self._chart.update_or_levels(or_levels[0], or_levels[1])
                overlays_applied.append("OR")

            ib_levels = self._extract_level_pair(data, "ib_high", "ib_low", container_key="ib_levels")
            if ib_levels is not None:
                self._chart.update_ib_levels(ib_levels[0], ib_levels[1])
                overlays_applied.append("IB")

            if hasattr(self._chart, "is_runtime_ready") and not self._chart.is_runtime_ready():
                runtime_error = None
                if hasattr(self._chart, "last_runtime_error"):
                    runtime_error = self._chart.last_runtime_error()
                if runtime_error:
                    self._set_status("Plotly runtime load failed", critical=True)
                else:
                    self._set_status("Waiting for Plotly runtime", critical=False)
                return

            overlay_text = f" overlays={','.join(overlays_applied)}" if overlays_applied else ""
            self._set_status(f"MTF chart updated — {updated} candles across {len(candles_by_tf)} TFs{overlay_text}")
        except Exception as exc:
            self._log.error(
                "MtfChartPanel.update_warm failed",
                dashboard_component="panel_mtf_chart",
                error=str(exc),
            )
            self._set_status("MTF chart update failed", critical=True)

    @staticmethod
    def _extract_candles_by_tf(data: Mapping[str, Any]) -> dict[str, Any]:
        src = data.get("mtf_candles", data.get("candles_by_tf", data.get("candles", {})))
        return dict(src) if isinstance(src, Mapping) else {}


    @staticmethod
    def _extract_vwap_bands(data: Mapping[str, Any], active_tf: str) -> Optional[tuple[list[Any], list[Any], list[Any], list[Any], list[Any]]]:
        """Extract prepared VWAP/band arrays for the active timeframe.

        Supported backend-friendly shapes:
        - ``vwap_bands: {"5m": {"vwap": [...], "upper1": [...], ...}}``
        - ``vwap_bands: {"vwap": [...], "upper1": [...], ...}``
        - flat keys: ``vwap``, ``vwap_upper1``, ``vwap_lower1``,
          ``vwap_upper2``, ``vwap_lower2``
        """
        src = data.get("vwap_bands", None)
        if isinstance(src, Mapping):
            tf_src = src.get(active_tf, src.get(str(active_tf).replace("m", "min"), None))
            if isinstance(tf_src, Mapping):
                src = tf_src
            if isinstance(src, Mapping):
                vwap = src.get("vwap", src.get("values", None))
                upper1 = src.get("upper1", src.get("vwap_upper1", src.get("u1", None)))
                lower1 = src.get("lower1", src.get("vwap_lower1", src.get("l1", None)))
                upper2 = src.get("upper2", src.get("vwap_upper2", src.get("u2", None)))
                lower2 = src.get("lower2", src.get("vwap_lower2", src.get("l2", None)))
                if all(isinstance(x, list) for x in (vwap, upper1, lower1, upper2, lower2)):
                    return (list(vwap), list(upper1), list(lower1), list(upper2), list(lower2))

        vwap = data.get("vwap", None)
        upper1 = data.get("vwap_upper1", data.get("upper1", None))
        lower1 = data.get("vwap_lower1", data.get("lower1", None))
        upper2 = data.get("vwap_upper2", data.get("upper2", None))
        lower2 = data.get("vwap_lower2", data.get("lower2", None))
        if all(isinstance(x, list) for x in (vwap, upper1, lower1, upper2, lower2)):
            return (list(vwap), list(upper1), list(lower1), list(upper2), list(lower2))
        return None

    @staticmethod
    def _extract_level_pair(data: Mapping[str, Any], high_key: str, low_key: str, *, container_key: str) -> Optional[tuple[Any, Any]]:
        container = data.get(container_key, None)
        if isinstance(container, Mapping):
            high = container.get("high", container.get(high_key, None))
            low = container.get("low", container.get(low_key, None))
            if high is not None or low is not None:
                return (high, low)
        if high_key in data or low_key in data:
            return (data.get(high_key), data.get(low_key))
        return None

    def _set_status(self, text: str, *, critical: bool = False) -> None:
        self._status_label.setText(str(text))
        if critical:
            self._status_label.setStyleSheet(self._css_status(_ON_BG_RED, _ON_FG_RED))
        else:
            self._status_label.setStyleSheet(self._css_status(_ON_BG_AMBER, _ON_FG_AMBER))

    @staticmethod
    def _css_status(bg: str, fg: str) -> str:
        return (
            "QLabel {"
            f" background-color: {bg};"
            f" color: {fg};"
            " border-radius: 6px;"
            " padding: 4px 8px;"
            " }"
        )

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers."""
        name = "dashboard_panels_mtf_chart"
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
    panel = MtfChartPanel(win)
    win.setCentralWidget(panel)
    win.resize(1200, 760)
    win.show()
    raise SystemExit(app.exec())
