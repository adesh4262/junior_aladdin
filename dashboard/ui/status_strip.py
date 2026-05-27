from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel, QMainWindow, QStatusBar

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)

try:
    from src.utils.helpers import ist_now, IST
except Exception:  # pragma: no cover
    from datetime import datetime, timezone, timedelta

    def ist_now() -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    IST = timezone(timedelta(hours=5, minutes=30))


class StatusStrip(QStatusBar):
    def __init__(self, parent: QMainWindow | None = None) -> None:
        super().__init__(parent)
        self.log = setup_logger("dashboard_ui_statusstrip")

        self._last_timestamp: str | None = None
        self._theme = self._load_theme()
        self._base_stylesheet = self._theme.get("stylesheet", "") if isinstance(self._theme, dict) else ""

        self.backend_label = QLabel("Backend: -")
        self.feed_health_label = QLabel("Feed: -")
        self.data_quality_label = QLabel("DQ: -")
        self.system_state_label = QLabel("State: -")
        self.mode_label = QLabel("Mode: -")
        self.clock_label = QLabel("Clock: -")
        self.panel_health_label = QLabel("Panels: -")
        self.last_update_age_label = QLabel("Last: -")

        self.addPermanentWidget(self.backend_label)
        self.addPermanentWidget(self.feed_health_label)
        self.addPermanentWidget(self.data_quality_label)
        self.addPermanentWidget(self.system_state_label)
        self.addPermanentWidget(self.mode_label)
        self.addPermanentWidget(self.panel_health_label)
        self.addPermanentWidget(self.clock_label)
        self.addPermanentWidget(self.last_update_age_label)

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start()

        self._tick_clock()

    def _theme_path(self) -> Path:
        # Locate theme relative to project layout
        base = Path(__file__).resolve().parents[2]
        return base / "dashboard" / "assets" / "theme.json"

    def _load_theme(self) -> dict:
        try:
            p = self._theme_path()
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            self.log.warning("Failed to load theme.json", error=str(exc))
        # Fallback minimal palette
        return {"palette": {}}

    def _load_theme_color(self, key: str, fallback: str) -> str:
        try:
            pal = self._theme.get("palette", {}) if isinstance(self._theme, dict) else {}
            return pal.get(key, fallback)
        except Exception:
            return fallback

    def _truncate(self, text: Any, max_len: int = 24) -> str:
        raw = str(text) if text is not None else "-"
        return raw if len(raw) <= max_len else f"{raw[: max_len - 3]}..."

    def _tick_clock(self) -> None:
        now = ist_now().strftime("%H:%M:%S IST")
        self.clock_label.setText(f"Clock: {now}")

        if self._last_timestamp:
            age_text = self._compute_age(self._last_timestamp)
            self.last_update_age_label.setText(f"Last: {age_text}")

    def _compute_age(self, timestamp_str: str) -> str:
        try:
            from datetime import datetime

            last = datetime.fromisoformat(timestamp_str)
            if last.tzinfo is None:
                # assume IST if no timezone provided
                last = last.replace(tzinfo=IST)
            age_seconds = max(0.0, (ist_now() - last).total_seconds())
            return f"{age_seconds:.1f}s"
        except Exception:
            return "invalid"

    def _apply_feed_color(self, feed_health: str) -> None:
        fh = (str(feed_health) or "").upper()
        # map feed health to theme palette keys
        mapping = {
            "HEALTHY": ["bullish", "veto_green"],
            "DELAYED": ["caution", "veto_amber"],
            "STALE": ["alert"],
            "DOWN": ["critical", "veto_red"],
        }
        fallbacks = {
            "bullish": "#166534",
            "caution": "#A16207",
            "alert": "#C2410C",
            "critical": "#991B1B",
            "veto_green": "#21D28B",
            "veto_amber": "#F5A524",
            "veto_red": "#FF4B5C",
        }

        candidates = mapping.get(fh, [])
        color = None
        for cand in candidates:
            color = self._load_theme_color(cand, fallbacks.get(cand, ""))
            if color:
                break
        if not color:
            color = "#374151"

        self.feed_health_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: 600; }}")

    def _apply_data_quality_color(self, score: float) -> None:
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s < 40.0:
            key = "critical"
            fallback = "#FF3D57"
        elif s < 70.0:
            key = "caution"
            fallback = "#FFB020"
        else:
            key = "bullish"
            fallback = "#22D48A"
        color = self._load_theme_color(key, fallback)
        self.data_quality_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: 600; }}")

    def _apply_panel_health(self, summary: Any) -> None:
        if not isinstance(summary, dict):
            self.panel_health_label.setText("Panels: -")
            self.panel_health_label.setStyleSheet("")
            return

        try:
            total = int(summary.get("total_panels", 0) or 0)
            ok = int(summary.get("ok_count", 0) or 0)
            degraded = int(summary.get("degraded_count", 0) or 0)
            stale = int(summary.get("stale_count", 0) or 0)
            error = int(summary.get("error_count", 0) or 0)
            warnings_count = int(summary.get("warnings_count", 0) or 0)
        except Exception:
            self.panel_health_label.setText("Panels: invalid")
            self.panel_health_label.setStyleSheet(f"QLabel {{ color: {self._load_theme_color('critical', '#FF3D57')}; font-weight: 700; }}")
            return

        self.panel_health_label.setText(
            f"Panels: {ok}/{total} OK D:{degraded} S:{stale} E:{error} W:{warnings_count}"
        )

        if error > 0:
            color = self._load_theme_color("critical", "#FF3D57")
            weight = 700
        elif stale > 0:
            color = self._load_theme_color("alert", "#FF7A18")
            weight = 700
        elif degraded > 0 or warnings_count > 0:
            color = self._load_theme_color("caution", "#FFB020")
            weight = 600
        else:
            color = self._load_theme_color("bullish", "#22D48A")
            weight = 600

        self.panel_health_label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: {weight}; }}")

    def update_status(self, data: dict) -> None:
        if not isinstance(data, dict):
            self.log.warning("update_status received non-dict payload")
            return

        required_keys = ("feed_health", "system_state", "mode", "last_update_timestamp")
        missing = [k for k in required_keys if k not in data]
        if missing:
            self.log.warning("Status update missing keys", missing_keys=missing)

        raw_feed_health = data.get("feed_health", self.feed_health_label.text().replace("Feed: ", ""))
        feed_health = self._truncate(raw_feed_health)
        system_state = self._truncate(data.get("system_state", self.system_state_label.text().replace("State: ", "")))
        mode = self._truncate(data.get("mode", self.mode_label.text().replace("Mode: ", "")))
        last_update_timestamp = data.get("last_update_timestamp", self._last_timestamp)
        dq = data.get("data_quality_score", data.get("data_quality", 0.0))

        # Backend connection state
        backend_state = data.get("backend_state", "DISCONNECTED")
        backend_text = str(backend_state).upper().strip()
        if backend_text in ("CONNECTED", "ACTIVE"):
            self.backend_label.setText("Backend: CONNECTED")
            bc = self._load_theme_color("bullish", "#22D48A")
            self.backend_label.setStyleSheet(f"QLabel {{ color: {bc}; font-weight: 700; }}")
        elif backend_text in ("WAITING", "RECONNECTING"):
            self.backend_label.setText("Backend: WAITING")
            bc = self._load_theme_color("caution", "#FFB020")
            self.backend_label.setStyleSheet(f"QLabel {{ color: {bc}; font-weight: 600; }}")
        elif backend_text in ("STALE", "DEGRADED"):
            self.backend_label.setText("Backend: STALE")
            bc = self._load_theme_color("alert", "#FF7A18")
            self.backend_label.setStyleSheet(f"QLabel {{ color: {bc}; font-weight: 700; }}")
        else:
            self.backend_label.setText("Backend: DISCONNECTED")
            bc = self._load_theme_color("text_disabled", "#4C5B6D")
            self.backend_label.setStyleSheet(f"QLabel {{ color: {bc}; font-weight: 600; }}")

        self.feed_health_label.setText(f"Feed: {feed_health}")
        self.system_state_label.setText(f"State: {system_state}")
        self.mode_label.setText(f"Mode: {mode}")
        # data quality
        try:
            dq_val = float(dq)
        except Exception:
            dq_val = 0.0
        self.data_quality_label.setText(f"DQ: {dq_val:.1f}")
        self._apply_data_quality_color(dq_val)
        self._apply_feed_color(str(raw_feed_health))
        self._apply_panel_health(data.get("panel_status_summary"))

        if isinstance(last_update_timestamp, str) and last_update_timestamp:
            self._last_timestamp = last_update_timestamp
            self.last_update_age_label.setText(f"Last: {self._compute_age(last_update_timestamp)}")

        emergency = bool(data.get("emergency", False))
        panel_summary = data.get("panel_status_summary") if isinstance(data.get("panel_status_summary"), dict) else {}
        warning_count = int(panel_summary.get("warnings_count", 0) or 0) if isinstance(panel_summary, dict) else 0
        error_count = int(panel_summary.get("error_count", 0) or 0) if isinstance(panel_summary, dict) else 0
        stale_count = int(panel_summary.get("stale_count", 0) or 0) if isinstance(panel_summary, dict) else 0

        if emergency:
            critical_color = self._load_theme_color("critical", "#7F1D1D")
            self.setStyleSheet(self._base_stylesheet + f" QStatusBar {{ background-color: {critical_color}; color: #FFFFFF; }}")
            self.showMessage("EMERGENCY")
            self.log.error("Emergency status set")
        else:
            self.setStyleSheet(self._base_stylesheet)
            if error_count > 0:
                self.showMessage(f"PANEL ERRORS: {error_count}")
            elif stale_count > 0:
                self.showMessage(f"STALE PANELS: {stale_count}")
            elif warning_count > 0:
                self.showMessage(f"PANEL WARNINGS: {warning_count}")
            else:
                self.clearMessage()


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    from datetime import datetime

    app = QApplication(sys.argv)
    window = QMainWindow()
    strip = StatusStrip(window)
    window.setStatusBar(strip)
    window.resize(900, 120)
    window.show()

    strip.update_status(
        {
            "feed_health": "HEALTHY",
            "system_state": "ACTIVE",
            "mode": "PAPER",
            "last_update_timestamp": ist_now().isoformat(),
            "data_quality_score": 88.2,
            "emergency": False,
        }
    )

    raise SystemExit(app.exec())