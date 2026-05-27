"""
dashboard/panels/system_health_panel.py

Panel 14 — SYSTEM HEALTH (Roadmap §4.14)

Displays runtime health of engines and overall pipeline metrics using projected
cold snapshot data pushed by the caller via:

    SystemHealthPanel.update_cold(data: dict)

Architecture:
- UI-only: no backend calls, no SnapshotBus/state_projection imports.
- Defensive: update_cold never raises; malformed/missing data shows placeholders.
- Fast: table + label updates are lightweight (<10ms target for ~20 engines).
- Visual: Obsidian Night theme (roadmap §1.2, §1.3). Colors match theme.json.
- PanelBase bridge: SystemHealthPanelAdapter in catalog.py makes this registrable
  in PanelRegistry for headless rendering.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QBrush
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.utils.logger import setup_logger


# Obsidian Night color palette (roadmap §1.2 — centralized in theme.json)
# Inline constants used here to keep this file self-contained for self-testing.

_ON_BG_INSET = "#102131"       # BG_3 deep inset — neutral pill background
_ON_BG_TEXT = "#EAF2FF"        # text primary — neutral pill text
_ON_BG_GREEN = "#0F2B1E"      # bullish bg (rgba 22/212/138 0.12 area)
_ON_FG_GREEN = "#22D48A"       # bullish text — primary positive
_ON_BG_AMBER = "#2B2000"       # caution bg
_ON_FG_AMBER = "#FFB020"       # caution text
_ON_BG_ORANGE = "#2B1200"      # alert bg
_ON_FG_ORANGE = "#FF7A18"      # alert text
_ON_BG_RED = "#2B0000"         # critical bg
_ON_FG_RED = "#FF6161"         # bearish / critical text

# QTableWidget will use QColor for dot markers (alive/dead indicators).
_ON_QCOLOR_OK = "#22D48A"
_ON_QCOLOR_WARN = "#FFB020"
_ON_QCOLOR_ERROR = "#FF6161"
_ON_QCOLOR_UNKNOWN = "#4C5B6D"


class SystemHealthPanel(QFrame):
    panel_id: str = "system_health"
    title: str = "SYSTEM HEALTH"
    refresh_class: str = "cold"
    required_snapshot_keys = [
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
    default_visible: bool = True

    _COL_ENGINE = 0
    _COL_STATUS = 1
    _COL_HEARTBEAT = 2
    _COL_ERROR = 3

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)

        self._log = self._get_logger()
        self._log.info(
            "SystemHealthPanel created",
            extra={"dashboard_component": "panel_system_health"},
        )

        self.setObjectName("SystemHealthPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        self._title_font = QFont()
        self._title_font.setPointSize(max(12, self._title_font.pointSize()))
        self._title_font.setBold(True)

        self._label_font = QFont()
        self._label_font.setPointSize(max(10, self._label_font.pointSize()))
        self._label_font.setBold(True)

        self._value_font = QFont("Monospace")
        self._value_font.setStyleHint(QFont.StyleHint.Monospace)
        self._value_font.setPointSize(max(11, self._value_font.pointSize()))
        self._value_font.setBold(True)

        root = QVBoxLayout()
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("SYSTEM HEALTH")
        title.setFont(self._title_font)
        title.setToolTip("Runtime health of engines and pipeline metrics (cold snapshot).")
        root.addWidget(title)

        # Engine health matrix
        matrix_title = QLabel("Engine Health Matrix")
        matrix_title.setFont(self._label_font)
        root.addWidget(matrix_title)

        self.engine_table = QTableWidget(0, 4)
        self.engine_table.setHorizontalHeaderLabels(["Engine", "Status", "Last Heartbeat", "Last Error"])
        self.engine_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.engine_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.engine_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.engine_table.setAlternatingRowColors(True)
        self.engine_table.verticalHeader().setVisible(False)
        self.engine_table.setSortingEnabled(False)

        # Obsidian Night table styling via stylesheet
        self.engine_table.setStyleSheet(
            "QTableWidget {"
            f" background-color: {_ON_BG_INSET};"
            f" alternate-background-color: #14283A;"
            f" color: {_ON_BG_TEXT};"
            " gridline-color: #163043;"
            " border: 1px solid #163043;"
            " }"
            "QHeaderView::section {"
            f" background-color: #0B1724;"
            f" color: {_ON_BG_TEXT};"
            " border: 1px solid #163043;"
            " padding: 4px;"
            " }"
        )

        header = self.engine_table.horizontalHeader()
        header.setSectionResizeMode(self._COL_ENGINE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._COL_HEARTBEAT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._COL_ERROR, QHeaderView.ResizeMode.Stretch)

        self.engine_table.setToolTip("Hover over 'Last Error' cells for full error text.")
        root.addWidget(self.engine_table, stretch=2)

        # Global metrics grid
        metrics_title = QLabel("Global Health Metrics")
        metrics_title.setFont(self._label_font)
        root.addWidget(metrics_title)

        self._metrics_grid = QGridLayout()
        self._metrics_grid.setHorizontalSpacing(14)
        self._metrics_grid.setVerticalSpacing(8)

        self.feed_health_label = self._add_metric(0, 0, "Feed Health", "Market data feed state.")
        self.data_quality_label = self._add_metric(0, 2, "Data Quality", "0–100 score; lower indicates degraded data.")
        self.tps_label = self._add_metric(0, 4, "Ticks / sec", "Approx ticks per second processed by the pipeline.")

        self.feed_lag_label = self._add_metric(1, 0, "Feed Lag", "Feed lag (ms).")
        self.fallback_label = self._add_metric(1, 2, "Using Fallback", "YES if fallback/degraded mode is active.")
        self.kill_switch_label = self._add_metric(1, 4, "Kill-Switch", "ARMED or TRIGGERED.")

        self.snapshot_age_label = self._add_metric(2, 0, "Snapshot Age", "Age of the latest cold snapshot frame.")
        self.last_update_label = self._add_metric(2, 2, "Last Update", "Timestamp of the last cold snapshot update.")

        # Stretch value columns
        self._metrics_grid.setColumnStretch(1, 1)
        self._metrics_grid.setColumnStretch(3, 1)
        self._metrics_grid.setColumnStretch(5, 1)

        root.addLayout(self._metrics_grid, stretch=0)
        self.setLayout(root)

        self.clear()

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        system-health audit/warning events in the normal project log files while
        retaining a stdlib fallback if logger construction fails.
        """
        name = "dashboard_panels_system_health"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)

    # -----------------------------
    # UI helpers
    # -----------------------------

    def _add_metric(self, row: int, col: int, label_text: str, tooltip: str) -> QLabel:
        key = QLabel(label_text)
        key.setFont(self._label_font)
        key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        key.setToolTip(tooltip)

        val = QLabel("\u2013")
        val.setFont(self._value_font)
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        val.setTextFormat(Qt.TextFormat.PlainText)
        val.setMinimumWidth(140)
        val.setToolTip(tooltip)
        val.setStyleSheet(self._css_neutral())

        self._metrics_grid.addWidget(key, row, col)
        self._metrics_grid.addWidget(val, row, col + 1)
        return val

    # -----------------------------
    # Public API
    # -----------------------------

    def clear(self) -> None:
        """Reset table and metric labels to neutral placeholders."""
        self.engine_table.setRowCount(0)

        for lbl in (
            self.feed_health_label,
            self.data_quality_label,
            self.tps_label,
            self.feed_lag_label,
            self.fallback_label,
            self.kill_switch_label,
            self.snapshot_age_label,
            self.last_update_label,
        ):
            lbl.setText("\u2013")
            lbl.setStyleSheet(self._css_neutral())

    def update_cold(self, data: Any) -> None:
        """
        Update the panel using the projected cold snapshot dict.

        Never raises; logs warnings for malformed input.
        """
        try:
            if not isinstance(data, Mapping):
                self._log.warning(
                    "SystemHealthPanel.update_cold received non-mapping; clearing. type=%s",
                    type(data).__name__,
                    extra={"dashboard_component": "panel_system_health"},
                )
                self._set_unknown_all()
                return

            # Soft missing-key warning for observability.
            expected = (
                "engine_health",
                "feed_health",
                "data_quality_score",
                "ticks_per_second",
                "feed_lag_ms",
                "using_fallback",
                "kill_switch_state",
                "snapshot_age_seconds",
                "last_update_timestamp",
            )
            missing = [k for k in expected if k not in data]
            if missing:
                self._log.warning(
                    "SystemHealthPanel.update_cold missing keys=%s",
                    missing,
                    extra={"dashboard_component": "panel_system_health"},
                )

            # Engine health matrix
            engine_health = data.get("engine_health", {}) or {}
            self._update_engine_table(engine_health)

            # Global metrics
            feed_health = self._as_str(data.get("feed_health", "UNKNOWN"), default="UNKNOWN").upper().strip()
            dqs = self._clamp_float(data.get("data_quality_score", 0.0), 0.0, 100.0)
            tps = self._to_float(data.get("ticks_per_second", 0.0), default=0.0)
            lag_ms = self._to_float(data.get("feed_lag_ms", 0.0), default=0.0)
            using_fallback = bool(data.get("using_fallback", False))
            kill_switch_state = self._as_str(data.get("kill_switch_state", "UNKNOWN"), default="UNKNOWN").upper().strip()
            snapshot_age_sec = self._to_float(data.get("snapshot_age_seconds", 0.0), default=0.0)
            last_update_ts = self._as_str(data.get("last_update_timestamp", "never"), default="never")

            self.feed_health_label.setText(self._clip(feed_health, 16))
            self.data_quality_label.setText(f"{dqs:.0f}/100")
            self.tps_label.setText(f"{tps:.1f}")
            self.feed_lag_label.setText(self._format_lag(lag_ms))
            self.fallback_label.setText("YES" if using_fallback else "NO")
            self.kill_switch_label.setText(self._clip(kill_switch_state, 16))
            self.snapshot_age_label.setText(self._format_age(snapshot_age_sec))
            self.last_update_label.setText(self._clip(last_update_ts, 40))

            # Styling (Obsidian Night palette)
            self._style_feed_health(self.feed_health_label, feed_health)
            self._style_data_quality(self.data_quality_label, dqs)
            self._style_fallback(self.fallback_label, using_fallback)
            self._style_kill_switch(self.kill_switch_label, kill_switch_state)

            # Keep these neutral (avoid adding policy/business logic).
            self.tps_label.setStyleSheet(self._css_neutral())
            self.feed_lag_label.setStyleSheet(self._css_neutral())
            self.snapshot_age_label.setStyleSheet(self._css_neutral())
            self.last_update_label.setStyleSheet(self._css_neutral())

        except Exception as e:
            self._log.warning(
                "SystemHealthPanel.update_cold failed; showing unknown. err=%s",
                repr(e),
                extra={"dashboard_component": "panel_system_health"},
            )
            self._set_unknown_all()

    # -----------------------------
    # Engine table update
    # -----------------------------

    def _update_engine_table(self, engine_health: Any) -> None:
        self.engine_table.setSortingEnabled(False)
        self.engine_table.setUpdatesEnabled(False)
        try:
            self.engine_table.setRowCount(0)

            if not isinstance(engine_health, Mapping) or not engine_health:
                self._add_engine_row(
                    engine="(no data)",
                    status="UNKNOWN",
                    last_heartbeat="never",
                    last_error="No engine health data",
                    color=self._color_unknown(),
                )
                return

            # Stable order: alphabetical by engine name for operator scanning.
            for engine_name in sorted(engine_health.keys(), key=lambda x: str(x)):
                entry = engine_health.get(engine_name, None)

                if not isinstance(entry, Mapping):
                    # Malformed entry -> ERROR row
                    self._add_engine_row(
                        engine=str(engine_name),
                        status="ERROR",
                        last_heartbeat="unknown",
                        last_error=f"Malformed engine entry type={type(entry).__name__}",
                        color=self._color_error(),
                    )
                    continue

                alive = entry.get("alive", None)
                status_raw = entry.get("status", None)
                last_heartbeat = self._as_str(entry.get("last_heartbeat", "never"), default="never")
                last_error = self._as_str(entry.get("last_error", ""), default="")

                status = self._derive_engine_status(status_raw=status_raw, alive=alive)
                color = self._status_color(status)

                self._add_engine_row(
                    engine=str(engine_name),
                    status=status,
                    last_heartbeat=last_heartbeat if last_heartbeat else "never",
                    last_error=last_error if last_error else "\u2013",
                    color=color,
                    tooltip=last_error if last_error else "",
                )
        finally:
            self.engine_table.setUpdatesEnabled(True)

    def _add_engine_row(
        self,
        engine: str,
        status: str,
        last_heartbeat: str,
        last_error: str,
        color: QColor,
        tooltip: str = "",
    ) -> None:
        r = self.engine_table.rowCount()
        self.engine_table.insertRow(r)

        item_engine = QTableWidgetItem(self._clip(engine, 48))
        item_engine.setToolTip(engine)
        item_engine.setFlags(item_engine.flags() & ~Qt.ItemFlag.ItemIsEditable)

        # Status uses a colored dot + text for quick scan.
        dot = "\u25cf"
        item_status = QTableWidgetItem(f"{dot} {status}")
        item_status.setFlags(item_status.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item_status.setForeground(QBrush(color))

        item_hb = QTableWidgetItem(self._clip(last_heartbeat, 48))
        item_hb.setToolTip(last_heartbeat)
        item_hb.setFlags(item_hb.flags() & ~Qt.ItemFlag.ItemIsEditable)

        display_error = self._clip(last_error, 120)
        item_err = QTableWidgetItem(display_error)
        item_err.setFlags(item_err.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if tooltip:
            item_err.setToolTip(tooltip)
        else:
            item_err.setToolTip(last_error if last_error and last_error != "\u2013" else "")

        self.engine_table.setItem(r, self._COL_ENGINE, item_engine)
        self.engine_table.setItem(r, self._COL_STATUS, item_status)
        self.engine_table.setItem(r, self._COL_HEARTBEAT, item_hb)
        self.engine_table.setItem(r, self._COL_ERROR, item_err)

    @staticmethod
    def _derive_engine_status(status_raw: Any, alive: Any) -> str:
        if isinstance(status_raw, str) and status_raw.strip():
            s = status_raw.strip().upper()
            if s in {"OK", "WARNING", "ERROR", "UNKNOWN"}:
                return s
            # tolerate variants
            if s in {"WARN", "DEGRADED"}:
                return "WARNING"
            if s in {"DEAD", "DOWN", "FAIL"}:
                return "ERROR"

        if alive is True:
            return "OK"
        if alive is False:
            return "ERROR"
        return "UNKNOWN"

    @staticmethod
    def _status_color(status: str) -> QColor:
        s = (status or "").strip().upper()
        if s == "OK":
            return SystemHealthPanel._color_ok()
        if s == "WARNING":
            return SystemHealthPanel._color_warn()
        if s == "ERROR":
            return SystemHealthPanel._color_error()
        return SystemHealthPanel._color_unknown()

    @staticmethod
    def _color_ok() -> QColor:
        return QColor(_ON_QCOLOR_OK)

    @staticmethod
    def _color_warn() -> QColor:
        return QColor(_ON_QCOLOR_WARN)

    @staticmethod
    def _color_error() -> QColor:
        return QColor(_ON_QCOLOR_ERROR)

    @staticmethod
    def _color_unknown() -> QColor:
        return QColor(_ON_QCOLOR_UNKNOWN)

    # -----------------------------
    # Formatting + styling
    # -----------------------------

    def _set_unknown_all(self) -> None:
        self.engine_table.setRowCount(0)
        self._add_engine_row(
            engine="(unknown)",
            status="UNKNOWN",
            last_heartbeat="never",
            last_error="Invalid/missing cold snapshot data",
            color=self._color_unknown(),
        )
        for lbl in (
            self.feed_health_label,
            self.data_quality_label,
            self.tps_label,
            self.feed_lag_label,
            self.fallback_label,
            self.kill_switch_label,
            self.snapshot_age_label,
            self.last_update_label,
        ):
            lbl.setText("?")
            lbl.setStyleSheet(self._css_critical())

    @staticmethod
    def _as_str(value: Any, default: str = "unknown") -> str:
        if value is None:
            return default
        try:
            s = str(value)
            return s if s else default
        except Exception:
            return default

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float) -> float:
        v = SystemHealthPanel._to_float(value, default=lo)
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    @staticmethod
    def _clip(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 1)] + "\u2026"

    @staticmethod
    def _format_lag(ms: float) -> str:
        try:
            v = float(ms)
        except Exception:
            return "???"
        if v < 0:
            v = 0.0
        if v >= 1000.0:
            return f"{v / 1000.0:.1f}s"
        return f"{v:.0f}ms"

    @staticmethod
    def _format_age(sec: float) -> str:
        try:
            v = float(sec)
        except Exception:
            return "???"
        if v < 0:
            v = 0.0
        if v < 1.0:
            return f"{v * 1000.0:.0f}ms"
        if v < 60.0:
            return f"{v:.1f}s"
        if v < 3600.0:
            return f"{v / 60.0:.1f}m"
        return f"{v / 3600.0:.1f}h"

    # CSS "pill" labels for metrics (Obsidian Night palette)
    @staticmethod
    def _css_base(fg: str, bg: str) -> str:
        return (
            "QLabel {"
            f" color: {fg};"
            f" background-color: {bg};"
            " padding: 3px 7px;"
            " border-radius: 6px;"
            " }"
        )

    @classmethod
    def _css_neutral(cls) -> str:
        return cls._css_base(fg=_ON_BG_TEXT, bg=_ON_BG_INSET)

    @classmethod
    def _css_ok(cls) -> str:
        return cls._css_base(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN)

    @classmethod
    def _css_warn(cls) -> str:
        return cls._css_base(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER)

    @classmethod
    def _css_caution(cls) -> str:
        return cls._css_base(fg=_ON_FG_ORANGE, bg=_ON_BG_ORANGE)

    @classmethod
    def _css_critical(cls) -> str:
        return cls._css_base(fg=_ON_FG_RED, bg=_ON_BG_RED)

    def _style_feed_health(self, label: QLabel, value: str) -> None:
        v = (value or "").strip().upper()
        if v == "HEALTHY":
            label.setStyleSheet(self._css_ok())
        elif v == "DELAYED":
            label.setStyleSheet(self._css_warn())
        elif v == "STALE":
            label.setStyleSheet(self._css_caution())
        elif v == "DOWN":
            label.setStyleSheet(self._css_critical())
        else:
            label.setStyleSheet(self._css_neutral())

    def _style_data_quality(self, label: QLabel, score: float) -> None:
        if score >= 80.0:
            label.setStyleSheet(self._css_ok())
        elif score >= 60.0:
            label.setStyleSheet(self._css_warn())
        else:
            label.setStyleSheet(self._css_critical())

    def _style_fallback(self, label: QLabel, using: bool) -> None:
        label.setStyleSheet(self._css_caution() if using else self._css_ok())

    def _style_kill_switch(self, label: QLabel, state: str) -> None:
        s = (state or "").strip().upper()
        if s == "ARMED":
            label.setStyleSheet(self._css_ok())
        elif s == "TRIGGERED":
            label.setStyleSheet(self._css_critical())
        elif s in {"UNKNOWN", ""}:
            label.setStyleSheet(self._css_neutral())
        else:
            label.setStyleSheet(self._css_warn())


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    panel = SystemHealthPanel(win)
    win.setCentralWidget(panel)
    win.resize(1100, 650)
    win.show()

    test_data = {
        "engine_health": {
            "data_engine": {"alive": True, "last_heartbeat": "2026-05-21T10:30:00", "last_error": ""},
            "feature_engine": {"alive": True, "last_heartbeat": "2026-05-21T10:30:00", "last_error": ""},
            "risk_engine": {"alive": False, "last_heartbeat": "2026-05-21T10:25:00", "last_error": "Circuit breaker tripped"},
            "ml_engine": {"alive": True, "last_heartbeat": "2026-05-21T10:29:00", "last_error": "SHAP model missing"},
        },
        "feed_health": "HEALTHY",
        "data_quality_score": 92.0,
        "ticks_per_second": 15.2,
        "feed_lag_ms": 120.0,
        "using_fallback": False,
        "kill_switch_state": "ARMED",
        "snapshot_age_seconds": 0.5,
        "last_update_timestamp": "2026-05-21T10:30:02",
    }
    panel.update_cold(test_data)

    raise SystemExit(app.exec())