"""
dashboard/panels/status_panel.py

Panel 01 — STATUS (Roadmap §4.1)

High-density system health monitor for operators. This panel renders projected
hot snapshot data pushed by the caller (MainWindow), via:

    StatusPanel.update_hot(data: dict)

Architecture:
- UI-only: no backend calls, no SnapshotBus/state_projection imports.
- Defensive: never raises from update_hot; missing/invalid data renders as "???".
- Fast: update_hot does minimal formatting and label updates (<5ms target).
- Visual: Obsidian Night theme (roadmap §1.2, §1.3). Colors match theme.json.
- PanelBase bridge: StatusPanelAdapter in catalog.py makes this registrable in
  PanelRegistry for headless rendering.

Expected (soft) hot keys:
- feed_health: str (HEALTHY/DELAYED/STALE/DOWN/...)
- system_state: str (ACTIVE/CAUTIOUS/SAFE/LOCKED/...)
- mode: str (ALERT/PAPER/LIVE/...)
- data_quality_score: float|int (0..100)
- ticks_per_second: float|int
- feed_lag_ms: float|int
- using_fallback: bool
- last_update_timestamp: str|int|float (already projected by caller)
- risk_state: str (NORMAL/CAUTION/REDUCED/LOCKED/...)
- trades_today: int
- consecutive_losses: int
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel

from src.utils.logger import setup_logger


# Obsidian Night color palette (roadmap §1.2 — centralized in theme.json)
# Inline constants used here to keep this file self-contained for self-testing.
# At runtime the theme.json palette takes effect at app level; these CSS pills
# provide high-contrast readability on the dark panel surface.

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


class StatusPanel(QFrame):
    panel_id: str = "status"
    title: str = "STATUS"
    refresh_class: str = "hot"
    required_snapshot_keys = [
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
    default_visible: bool = True

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)

        self._log = self._get_logger()
        self._log.info(
            "StatusPanel created",
            extra={"dashboard_component": "panel_status"},
        )

        self.setObjectName("StatusPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        self._grid = QGridLayout()
        self._grid.setContentsMargins(10, 10, 10, 10)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(8)

        # Fonts: keep readable from distance.
        self._label_font = QFont()
        self._label_font.setPointSize(max(10, self._label_font.pointSize()))
        self._label_font.setBold(True)

        self._value_font = QFont("Monospace")
        self._value_font.setStyleHint(QFont.StyleHint.Monospace)
        self._value_font.setPointSize(max(11, self._value_font.pointSize()))
        self._value_font.setBold(True)

        # Row 0
        self.feed_health_label = self._add_pair(0, 0, "Feed Health", tooltip=(
            "Market data feed health.\n"
            "HEALTHY: on-time\n"
            "DELAYED: lagging\n"
            "STALE: no fresh ticks beyond threshold\n"
            "DOWN: disconnected"
        ))[1]
        self.system_state_label = self._add_pair(0, 2, "System State", tooltip=(
            "Overall system state (risk/controls).\n"
            "LOCKED implies no LIVE trading permitted."
        ))[1]
        self.mode_label = self._add_pair(0, 4, "Mode", tooltip=(
            "Operator mode.\n"
            "ALERT: monitoring\n"
            "PAPER: simulated execution\n"
            "LIVE: real trading (requires handshake)"
        ))[1]

        # Row 1
        self.data_quality_label = self._add_pair(1, 0, "Data Quality", tooltip=(
            "0–100 score.\n"
            ">=80: good\n"
            "60–79: caution\n"
            "<60: poor (expect degraded strategy behavior)"
        ), numeric=True)[1]
        self.tps_label = self._add_pair(1, 2, "Ticks / sec", tooltip=(
            "Approx ticks per second processed by the pipeline."
        ), numeric=True)[1]
        self.feed_lag_label = self._add_pair(1, 4, "Feed Lag", tooltip=(
            "Feed lag (ms). Large values indicate delayed data."
        ), numeric=True)[1]

        # Row 2
        self.fallback_label = self._add_pair(2, 0, "Using Fallback", tooltip=(
            "YES if system is using fallback / degraded data source or logic."
        ))[1]
        self.last_update_label = self._add_pair(2, 2, "Last Update", tooltip=(
            "Timestamp of last hot snapshot update (projected by caller)."
        ))[1]
        self.risk_state_label = self._add_pair(2, 4, "Risk State", tooltip=(
            "Risk posture.\n"
            "LOCKED: trading locked\n"
            "REDUCED: reduced exposure\n"
            "CAUTION: caution mode\n"
            "NORMAL: normal operation"
        ))[1]

        # Row 3 (two items)
        self.trades_today_label = self._add_pair(3, 0, "Trades Today", tooltip=(
            "Number of trades executed today."
        ), numeric=True)[1]
        self.consecutive_losses_label = self._add_pair(3, 2, "Consecutive Losses", tooltip=(
            "Consecutive losing trades today (if tracked)."
        ), numeric=True)[1]

        # Stretch: allow value columns to expand.
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self._grid.setColumnStretch(5, 1)

        self.setLayout(self._grid)
        self.clear()

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        status-panel audit/warning events in the normal project log files while
        retaining a stdlib fallback if logger construction fails.
        """
        name = "dashboard_panels_status"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)

    def _add_pair(
        self,
        row: int,
        col: int,
        label_text: str,
        tooltip: str = "",
        numeric: bool = False,
    ) -> Tuple[QLabel, QLabel]:
        """
        Add a (descriptor, value) pair into the grid at (row, col) and (row, col+1).
        """
        key = QLabel(label_text)
        key.setFont(self._label_font)
        key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            key.setToolTip(tooltip)

        val = QLabel("–")
        val.setFont(self._value_font if numeric else self._label_font)
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        val.setTextFormat(Qt.TextFormat.PlainText)
        val.setMinimumWidth(140)  # stable, readable columns
        val.setToolTip(tooltip if tooltip else label_text)

        # Base style for value fields: padded pill (Obsidian Night deep inset).
        val.setStyleSheet(self._css_neutral())

        self._grid.addWidget(key, row, col)
        self._grid.addWidget(val, row, col + 1)
        return key, val

    # -----------------------------
    # Public API
    # -----------------------------

    def clear(self) -> None:
        """
        Reset all values to a neutral placeholder.
        """
        placeholders = {
            self.feed_health_label: "–",
            self.system_state_label: "–",
            self.mode_label: "–",
            self.data_quality_label: "–",
            self.tps_label: "–",
            self.feed_lag_label: "–",
            self.fallback_label: "–",
            self.last_update_label: "–",
            self.risk_state_label: "–",
            self.trades_today_label: "–",
            self.consecutive_losses_label: "–",
        }
        for lbl, txt in placeholders.items():
            lbl.setText(txt)
            lbl.setStyleSheet(self._css_neutral())

    def update_hot(self, data: Any) -> None:
        """
        Update the panel using the projected hot snapshot dict.

        This method must be safe under malformed input and must not raise.
        """
        try:
            if not isinstance(data, Mapping):
                self._log.warning(
                    "StatusPanel.update_hot received non-mapping; clearing. type=%s",
                    type(data).__name__,
                    extra={"dashboard_component": "panel_status"},
                )
                self._set_unknown_all()
                return

            # Soft missing-key warning (avoid heavy enforcement; just observability).
            expected = (
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
            )
            missing = [k for k in expected if k not in data]
            if missing:
                self._log.warning(
                    "StatusPanel.update_hot missing keys=%s",
                    missing,
                    extra={"dashboard_component": "panel_status"},
                )

            feed_health = self._as_str(data.get("feed_health", "UNKNOWN"), default="UNKNOWN")
            system_state = self._as_str(data.get("system_state", "UNKNOWN"), default="UNKNOWN")
            mode = self._as_str(data.get("mode", "UNKNOWN"), default="UNKNOWN")

            dqs = self._clamp_float(data.get("data_quality_score", 0.0), 0.0, 100.0)
            tps = self._to_float(data.get("ticks_per_second", 0.0), default=0.0)
            feed_lag_ms = self._to_float(data.get("feed_lag_ms", 0.0), default=0.0)

            using_fallback = bool(data.get("using_fallback", False))
            last_update_ts = data.get("last_update_timestamp", "never")
            risk_state = self._as_str(data.get("risk_state", "NORMAL"), default="NORMAL")

            trades_today = self._to_int(data.get("trades_today", 0), default=0)
            consecutive_losses = self._to_int(data.get("consecutive_losses", 0), default=0)

            # Text updates (compact, readable)
            self.feed_health_label.setText(self._clip(feed_health, 24))
            self.system_state_label.setText(self._clip(system_state, 24))
            self.mode_label.setText(self._clip(mode, 12))

            self.data_quality_label.setText(f"{dqs:.0f}/100")
            self.tps_label.setText(f"{tps:.1f}")
            self.feed_lag_label.setText(self._format_lag(feed_lag_ms))

            self.fallback_label.setText("YES" if using_fallback else "NO")
            self.last_update_label.setText(self._clip(self._as_str(last_update_ts, default="never"), 40))
            self.risk_state_label.setText(self._clip(risk_state, 24))

            self.trades_today_label.setText(str(trades_today))
            self.consecutive_losses_label.setText(str(consecutive_losses))

            # Styling (severity-based — Obsidian Night palette)
            self._style_feed_health(self.feed_health_label, feed_health)
            self._style_system_state(self.system_state_label, system_state)
            self._style_mode(self.mode_label, mode)
            self._style_data_quality(self.data_quality_label, dqs)
            self._style_fallback(self.fallback_label, using_fallback)
            self._style_risk_state(self.risk_state_label, risk_state)

            # Keep numeric fields neutral (readability)
            self.tps_label.setStyleSheet(self._css_neutral())
            self.feed_lag_label.setStyleSheet(self._css_neutral())
            self.last_update_label.setStyleSheet(self._css_neutral())
            self.trades_today_label.setStyleSheet(self._css_neutral())
            self.consecutive_losses_label.setStyleSheet(self._css_neutral())

        except Exception as e:
            # Never crash the dashboard UI on a bad frame.
            self._log.warning(
                "StatusPanel.update_hot failed; clearing. err=%s",
                repr(e),
                extra={"dashboard_component": "panel_status"},
            )
            self._set_unknown_all()

    # -----------------------------
    # Formatting + styling
    # -----------------------------

    @staticmethod
    def _clip(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 1)] + "\u2026"

    @staticmethod
    def _as_str(value: Any, default: str = "???") -> str:
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
    def _to_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float) -> float:
        v = StatusPanel._to_float(value, default=lo)
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    @staticmethod
    def _format_lag(ms: float) -> str:
        """
        If > 1000ms, display seconds with 1 decimal; else show ms.
        """
        try:
            v = float(ms)
        except Exception:
            return "???"
        if v < 0:
            v = 0.0
        if v >= 1000.0:
            return f"{v / 1000.0:.1f}s"
        return f"{v:.0f}ms"

    def _set_unknown_all(self) -> None:
        # Set to "???" to signal invalid frame
        labels = (
            self.feed_health_label,
            self.system_state_label,
            self.mode_label,
            self.data_quality_label,
            self.tps_label,
            self.feed_lag_label,
            self.fallback_label,
            self.last_update_label,
            self.risk_state_label,
            self.trades_today_label,
            self.consecutive_losses_label,
        )
        for lbl in labels:
            lbl.setText("???")
            lbl.setStyleSheet(self._css_critical())

    # CSS helpers (Obsidian Night — high-contrast pills on deep inset background)
    # Matches theme.json: BG_3 (#102131), bullish (#22D48A), bearish (#FF6161),
    # caution (#FFB020), alert (#FF7A18), critical (#FF3D57).

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

    def _style_system_state(self, label: QLabel, value: str) -> None:
        v = (value or "").strip().upper()
        if v == "ACTIVE":
            label.setStyleSheet(self._css_ok())
        elif v == "CAUTIOUS":
            label.setStyleSheet(self._css_warn())
        elif v == "SAFE":
            label.setStyleSheet(self._css_caution())
        elif v == "LOCKED":
            label.setStyleSheet(self._css_critical())
        else:
            label.setStyleSheet(self._css_neutral())

    def _style_risk_state(self, label: QLabel, value: str) -> None:
        v = (value or "").strip().upper()
        if v == "NORMAL":
            label.setStyleSheet(self._css_ok())
        elif v == "CAUTION":
            label.setStyleSheet(self._css_warn())
        elif v == "REDUCED":
            label.setStyleSheet(self._css_caution())
        elif v == "LOCKED":
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
        # Fallback is caution (not necessarily critical, but attention-worthy).
        label.setStyleSheet(self._css_caution() if using else self._css_ok())

    def _style_mode(self, label: QLabel, value: str) -> None:
        # Mode isn't inherently "bad" but LIVE should stand out.
        v = (value or "").strip().upper()
        if v == "LIVE":
            label.setStyleSheet(self._css_warn())
        elif v == "PAPER":
            label.setStyleSheet(self._css_ok())
        elif v == "ALERT":
            label.setStyleSheet(self._css_neutral())
        else:
            label.setStyleSheet(self._css_neutral())


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    panel = StatusPanel(win)
    win.setCentralWidget(panel)
    win.resize(980, 220)
    win.show()

    # Simulate a hot frame
    test_data = {
        "feed_health": "HEALTHY",
        "system_state": "ACTIVE",
        "mode": "PAPER",
        "data_quality_score": 85.0,
        "ticks_per_second": 12.5,
        "feed_lag_ms": 250.0,
        "using_fallback": False,
        "last_update_timestamp": "2026-05-21T10:30:00",
        "risk_state": "NORMAL",
        "trades_today": 2,
        "consecutive_losses": 0,
    }
    panel.update_hot(test_data)

    raise SystemExit(app.exec())