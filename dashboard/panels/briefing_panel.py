"""
dashboard/panels/briefing_panel.py

Panel 02 — BRIEFING (Roadmap §4.2)

Operator-friendly summary of macro narrative, day personality, regime, and session
memory. Receives *projected warm snapshot* dicts from the caller via:

    BriefingPanel.update_warm(data: dict)

Architecture:
- UI-only: no backend calls, no SnapshotBus/state_projection imports.
- Defensive: never raises from update_warm; missing/invalid data renders as unknown.
- Fast: update_warm does minimal formatting and label updates (<5ms target).
- Visual: Obsidian Night theme (roadmap §1.2, §1.3). Colors match theme.json.
- PanelBase bridge: BriefingPanelAdapter in catalog.py makes this registrable in
  PanelRegistry for headless rendering.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

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
_ON_BG_BLUE = "#001C2B"        # info bg
_ON_FG_BLUE = "#48B9FF"        # index follow-through blue
_ON_BG_PURPLE = "#1A002B"      # event bg
_ON_FG_PURPLE = "#C084FC"      # event text


class BriefingPanel(QFrame):
    panel_id: str = "briefing"
    title: str = "BRIEFING"
    refresh_class: str = "warm"
    required_snapshot_keys = [
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
    default_visible: bool = True

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)

        self._log = self._get_logger()
        self._log.info(
            "BriefingPanel created",
            extra={"dashboard_component": "panel_briefing"},
        )

        self.setObjectName("BriefingPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        # Fonts: readable and compact.
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

        title = QLabel("MORNING BRIEFING")
        title.setFont(self._title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setToolTip("High-level context from warm snapshot projection.")
        root.addWidget(title)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(8)

        # Row 0: Narrative Label + Narrative Score bar
        self.narrative_label = self._add_pair(
            0,
            0,
            "Narrative",
            tooltip=(
                "Macro narrative label.\n"
                "Used as context and guardrails for trading permission."
            ),
            numeric=False,
            wide_value=True,
        )
        self.narrative_score_bar = QProgressBar()
        self.narrative_score_bar.setRange(0, 100)
        self.narrative_score_bar.setValue(0)
        self.narrative_score_bar.setTextVisible(True)
        self.narrative_score_bar.setFormat("%p%")
        self.narrative_score_bar.setToolTip("Narrative score (0–100). Higher indicates stronger narrative alignment.")
        self.narrative_score_bar.setFixedHeight(18)
        self.narrative_score_bar.setStyleSheet(self._css_progress_neutral())
        self._grid.addWidget(QLabel("Narrative Score"), 0, 2)
        self._grid.itemAtPosition(0, 2).widget().setFont(self._label_font)  # type: ignore[union-attr]
        self._grid.addWidget(self.narrative_score_bar, 0, 3)

        # Row 1: Fit factors
        self.long_fit_label = self._add_pair(
            1,
            0,
            "Long Fit",
            tooltip="Narrative fit factor for long setups (e.g., 1.2x supports long bias).",
            numeric=True,
        )
        self.short_fit_label = self._add_pair(
            1,
            2,
            "Short Fit",
            tooltip="Narrative fit factor for short setups (e.g., 0.4x discourages shorts).",
            numeric=True,
        )

        # Row 2: Day personality + historical match
        self.day_personality_label = self._add_pair(
            2,
            0,
            "Day Personality",
            tooltip="Market DNA day type classification (e.g., TREND_DAY, RANGE_DAY).",
            numeric=False,
            wide_value=True,
        )
        self.historical_match_label = self._add_pair(
            2,
            2,
            "Historical Match",
            tooltip="Similarity to historical days (0–100%).",
            numeric=True,
        )

        # Row 3: Session memory (one-line summary)
        self.session_memory_label = self._add_pair(
            3,
            0,
            "Session Memory",
            tooltip="One-line summary of notable session patterns (defended levels, failed breakouts, traps).",
            numeric=False,
            wide_value=True,
            span_to_end=True,
        )

        # Row 4: Regime + confidence
        self.regime_label = self._add_pair(
            4,
            0,
            "Regime",
            tooltip="Regime classification (TRENDING/RANGE/VOLATILE/CHOP/EVENT).",
            numeric=False,
        )
        self.regime_confidence_label = self._add_pair(
            4,
            2,
            "Regime Conf.",
            tooltip="Regime confidence (0–100%).",
            numeric=True,
        )

        # Row 5: Session phase + day type
        self.session_phase_label = self._add_pair(
            5,
            0,
            "Session Phase",
            tooltip="Session phase marker (e.g., OPENING, GOLDEN_AM, MIDDAY, CLOSE).",
            numeric=False,
            wide_value=True,
        )
        self.day_type_label = self._add_pair(
            5,
            2,
            "Day Type",
            tooltip="Projected day type (may mirror day personality).",
            numeric=False,
            wide_value=True,
        )

        # Stretch columns to keep layout stable.
        self._grid.setColumnStretch(1, 2)
        self._grid.setColumnStretch(3, 3)

        root.addLayout(self._grid)
        self.setLayout(root)

        self.clear()

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        briefing-panel audit/warning events in the normal project log files while
        retaining a stdlib fallback if logger construction fails.
        """
        name = "dashboard_panels_briefing"
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

    def _add_pair(
        self,
        row: int,
        col: int,
        label_text: str,
        tooltip: str = "",
        numeric: bool = False,
        wide_value: bool = False,
        span_to_end: bool = False,
    ) -> QLabel:
        """
        Adds a label/value pair to grid and returns the value QLabel.
        If span_to_end is True, value spans remaining columns to keep memory line readable.
        """
        key = QLabel(label_text)
        key.setFont(self._label_font)
        key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if tooltip:
            key.setToolTip(tooltip)

        val = QLabel("\u2013")
        val.setFont(self._value_font if numeric else self._label_font)
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        val.setWordWrap(wide_value or span_to_end)
        val.setTextFormat(Qt.TextFormat.PlainText)
        val.setToolTip(tooltip if tooltip else label_text)

        # Visual pill style for values (Obsidian Night deep inset).
        val.setStyleSheet(self._css_neutral())

        self._grid.addWidget(key, row, col)

        if span_to_end:
            # span value across remaining columns (col+1..end)
            self._grid.addWidget(val, row, col + 1, 1, 3)
        else:
            self._grid.addWidget(val, row, col + 1)

        return val

    # -----------------------------
    # Public API
    # -----------------------------

    def clear(self) -> None:
        self.narrative_label.setText("\u2013")
        self.narrative_label.setStyleSheet(self._css_neutral())
        self.narrative_score_bar.setValue(0)
        self.narrative_score_bar.setStyleSheet(self._css_progress_neutral())

        self.long_fit_label.setText("\u2013")
        self.long_fit_label.setStyleSheet(self._css_neutral())
        self.short_fit_label.setText("\u2013")
        self.short_fit_label.setStyleSheet(self._css_neutral())

        self.day_personality_label.setText("\u2013")
        self.day_personality_label.setStyleSheet(self._css_neutral())
        self.historical_match_label.setText("\u2013")
        self.historical_match_label.setStyleSheet(self._css_neutral())

        self.session_memory_label.setText("\u2013")
        self.session_memory_label.setStyleSheet(self._css_neutral())

        self.regime_label.setText("\u2013")
        self.regime_label.setStyleSheet(self._css_neutral())
        self.regime_confidence_label.setText("\u2013")
        self.regime_confidence_label.setStyleSheet(self._css_neutral())

        self.session_phase_label.setText("\u2013")
        self.session_phase_label.setStyleSheet(self._css_neutral())
        self.day_type_label.setText("\u2013")
        self.day_type_label.setStyleSheet(self._css_neutral())

    def update_warm(self, data: Any) -> None:
        """
        Update the panel using the projected warm snapshot dict.

        This method must not raise.
        """
        try:
            if not isinstance(data, Mapping):
                self._log.warning(
                    "BriefingPanel.update_warm received non-mapping; showing unknown. type=%s",
                    type(data).__name__,
                    extra={"dashboard_component": "panel_briefing"},
                )
                self._set_unknown_all()
                return

            expected = (
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
            )
            missing = [k for k in expected if k not in data]
            if missing:
                self._log.warning(
                    "BriefingPanel.update_warm missing keys=%s",
                    missing,
                    extra={"dashboard_component": "panel_briefing"},
                )

            narrative_label = self._as_str(data.get("narrative_label", "NEUTRAL"), default="NEUTRAL").upper().strip()
            narrative_score = self._clamp_float(data.get("narrative_score", 0.0), 0.0, 100.0)

            fit = data.get("narrative_fit_factors", {}) or {}
            long_fit = self._to_float(fit.get("long_fit", 0.8), default=0.8)
            short_fit = self._to_float(fit.get("short_fit", 0.8), default=0.8)

            day_personality = data.get("day_personality", {}) or {}
            day_personality_label = self._as_str(day_personality.get("day_type", "UNKNOWN"), default="UNKNOWN").strip()

            historical_match = self._to_float(data.get("historical_match_score", 0.0), default=0.0)
            historical_match_pct = self._to_pct(historical_match)

            session_memory = data.get("session_memory", {}) or {}
            session_memory_summary = self._format_session_memory(session_memory)

            regime = self._as_str(data.get("regime", "UNKNOWN"), default="UNKNOWN").upper().strip()
            regime_confidence = self._to_float(data.get("regime_confidence", 0.0), default=0.0)
            regime_conf_pct = self._to_pct(regime_confidence)

            session_phase = self._as_str(data.get("session_phase", "UNKNOWN"), default="UNKNOWN").strip()
            day_type = self._as_str(data.get("day_type", "UNKNOWN"), default="UNKNOWN").strip()

            # Apply texts
            self.narrative_label.setText(self._clip(narrative_label, 28))
            self.narrative_score_bar.setValue(int(round(narrative_score)))

            self.long_fit_label.setText(f"{long_fit:.2f}x")
            self.short_fit_label.setText(f"{short_fit:.2f}x")

            self.day_personality_label.setText(self._clip(day_personality_label, 28))
            self.historical_match_label.setText(f"{historical_match_pct:.0f}%")

            self.session_memory_label.setText(self._clip(session_memory_summary, 120))

            self.regime_label.setText(self._clip(regime, 18))
            self.regime_confidence_label.setText(f"{regime_conf_pct:.0f}%")

            self.session_phase_label.setText(self._clip(session_phase, 28))
            self.day_type_label.setText(self._clip(day_type, 28))

            # Apply styles (Obsidian Night palette)
            self._style_narrative(self.narrative_label, narrative_label)
            self._style_narrative_bar(self.narrative_score_bar, narrative_label, narrative_score)

            self.long_fit_label.setStyleSheet(self._css_neutral())
            self.short_fit_label.setStyleSheet(self._css_neutral())

            self.day_personality_label.setStyleSheet(self._css_neutral())
            self.historical_match_label.setStyleSheet(self._css_neutral())

            self.session_memory_label.setStyleSheet(self._css_neutral())

            self._style_regime(self.regime_label, regime)
            self.regime_confidence_label.setStyleSheet(self._css_neutral())

            self.session_phase_label.setStyleSheet(self._css_neutral())
            self.day_type_label.setStyleSheet(self._css_neutral())

        except Exception as e:
            self._log.warning(
                "BriefingPanel.update_warm failed; showing unknown. err=%s",
                repr(e),
                extra={"dashboard_component": "panel_briefing"},
            )
            self._set_unknown_all()

    # -----------------------------
    # Formatting + styling
    # -----------------------------

    def _set_unknown_all(self) -> None:
        labels = (
            self.narrative_label,
            self.long_fit_label,
            self.short_fit_label,
            self.day_personality_label,
            self.historical_match_label,
            self.session_memory_label,
            self.regime_label,
            self.regime_confidence_label,
            self.session_phase_label,
            self.day_type_label,
        )
        for lbl in labels:
            lbl.setText("?")
            lbl.setStyleSheet(self._css_critical())
        self.narrative_score_bar.setValue(0)
        self.narrative_score_bar.setStyleSheet(self._css_progress_neutral())

    @staticmethod
    def _clip(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max(0, max_len - 1)] + "\u2026"

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
        v = BriefingPanel._to_float(value, default=lo)
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    @staticmethod
    def _to_pct(value: float) -> float:
        """
        Convert 0..1 or 0..100 float into 0..100 percent (clamped).
        """
        try:
            v = float(value)
        except Exception:
            return 0.0
        if 0.0 <= v <= 1.0:
            v *= 100.0
        if v < 0.0:
            v = 0.0
        if v > 100.0:
            v = 100.0
        return v

    @staticmethod
    def _countish(x: Any) -> int:
        """
        Interpret x as a count:
        - list/tuple/set -> len
        - int/float/str -> int(float)
        - None -> 0
        """
        if x is None:
            return 0
        if isinstance(x, (list, tuple, set)):
            return len(x)
        try:
            return int(float(x))
        except Exception:
            return 0

    def _format_session_memory(self, mem: Any) -> str:
        """
        Format session memory dict to a one-line summary.
        Expected keys: levels_defended, failed_breakouts, traps_detected
        """
        if not isinstance(mem, Mapping) or not mem:
            return "No memory"

        defended = self._countish(mem.get("levels_defended", 0))
        failed = self._countish(mem.get("failed_breakouts", 0))
        traps = self._countish(mem.get("traps_detected", 0))

        # Keep crisp, operator-friendly.
        parts = [
            f"Defended: {defended}",
            f"Traps: {traps}",
            f"Failed breakouts: {failed}",
        ]
        return ", ".join(parts)

    # CSS helpers: pill labels + progress bar (Obsidian Night palette)
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
    def _css_good(cls) -> str:
        return cls._css_base(fg=_ON_FG_GREEN, bg="#0F2B1E")  # lighter green bg

    @classmethod
    def _css_warn(cls) -> str:
        return cls._css_base(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER)

    @classmethod
    def _css_caution(cls) -> str:
        return cls._css_base(fg=_ON_FG_ORANGE, bg=_ON_BG_ORANGE)

    @classmethod
    def _css_critical(cls) -> str:
        return cls._css_base(fg=_ON_FG_RED, bg=_ON_BG_RED)

    @classmethod
    def _css_info(cls) -> str:
        return cls._css_base(fg=_ON_FG_BLUE, bg=_ON_BG_BLUE)

    @classmethod
    def _css_event(cls) -> str:
        return cls._css_base(fg=_ON_FG_PURPLE, bg=_ON_BG_PURPLE)

    @staticmethod
    def _css_progress_neutral() -> str:
        return (
            "QProgressBar {"
            " border: 1px solid #163043;"
            " border-radius: 6px;"
            " text-align: center;"
            f" color: {_ON_BG_TEXT};"
            f" background: {_ON_BG_INSET};"
            " }"
            "QProgressBar::chunk {"
            " border-radius: 6px;"
            " background-color: #4C5B6D;"
            " }"
        )

    @staticmethod
    def _css_progress(color: str) -> str:
        return (
            "QProgressBar {"
            " border: 1px solid #163043;"
            " border-radius: 6px;"
            " text-align: center;"
            f" color: {_ON_BG_TEXT};"
            f" background: {_ON_BG_INSET};"
            " }"
            "QProgressBar::chunk {"
            " border-radius: 6px;"
            f" background-color: {color};"
            " }"
        )

    def _style_narrative(self, label: QLabel, narrative_label: str) -> None:
        v = (narrative_label or "").strip().upper()
        if v == "STRONG_BULLISH":
            label.setStyleSheet(self._css_ok())
        elif v == "MILD_BULLISH":
            label.setStyleSheet(self._css_good())
        elif v == "NEUTRAL":
            label.setStyleSheet(self._css_neutral())
        elif v == "MILD_BEARISH":
            label.setStyleSheet(self._css_caution())
        elif v == "STRONG_BEARISH":
            label.setStyleSheet(self._css_critical())
        elif v == "EVENT_RISK":
            label.setStyleSheet(self._css_event())
        else:
            label.setStyleSheet(self._css_neutral())

    def _style_narrative_bar(self, bar: QProgressBar, narrative_label: str, score: float) -> None:
        v = (narrative_label or "").strip().upper()
        if v in {"STRONG_BULLISH", "MILD_BULLISH"}:
            bar.setStyleSheet(self._css_progress(_ON_FG_GREEN))
        elif v in {"MILD_BEARISH", "STRONG_BEARISH"}:
            bar.setStyleSheet(self._css_progress(_ON_FG_RED))
        elif v == "EVENT_RISK":
            bar.setStyleSheet(self._css_progress(_ON_FG_PURPLE))
        else:
            if score >= 70.0:
                bar.setStyleSheet(self._css_progress(_ON_FG_GREEN))
            elif score >= 40.0:
                bar.setStyleSheet(self._css_progress(_ON_FG_AMBER))
            else:
                bar.setStyleSheet(self._css_progress(_ON_FG_ORANGE))

    def _style_regime(self, label: QLabel, regime: str) -> None:
        v = (regime or "").strip().upper()
        if v == "TRENDING":
            label.setStyleSheet(self._css_ok())
        elif v == "RANGE":
            label.setStyleSheet(self._css_info())
        elif v == "VOLATILE":
            label.setStyleSheet(self._css_caution())
        elif v == "CHOP":
            label.setStyleSheet(self._css_critical())
        elif v == "EVENT":
            label.setStyleSheet(self._css_event())
        else:
            label.setStyleSheet(self._css_neutral())


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    panel = BriefingPanel(win)
    win.setCentralWidget(panel)
    win.resize(980, 320)
    win.show()

    test_data = {
        "narrative_label": "MILD_BULLISH",
        "narrative_score": 68.0,
        "narrative_fit_factors": {"long_fit": 1.0, "short_fit": 0.4},
        "day_personality": {"day_type": "TREND_DAY"},
        "historical_match_score": 0.85,
        "session_memory": {"levels_defended": [23050], "failed_breakouts": [23200], "traps_detected": 1},
        "regime": "TRENDING",
        "regime_confidence": 0.92,
        "session_phase": "GOLDEN_AM",
        "day_type": "TREND_DAY",
    }
    panel.update_warm(test_data)

    raise SystemExit(app.exec())