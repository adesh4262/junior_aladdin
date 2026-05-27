"""
dashboard/panels/global_vitals_panel.py

Panel 16 — GLOBAL VITALS (Roadmap §4.16)

Highest-level sovereign summary of system + market, and hosts the Component Guard.

Data sources:
- Hot snapshot projection (pushed by caller): update_hot(data: dict)
- Heavyweights/component guard feed (pushed by caller): update_component_guard(heavyweights: list)

This panel is UI-only:
- No backend calls
- No SnapshotBus/state_projection imports
- Emits veto_state_changed(bool) when any heavyweight is in VETO (non-replay mode)
- Visual: Obsidian Night theme (roadmap §1.2, §1.3). Colors match theme.json.

Replay note:
- In replay mode, heavyweight data may be absent. The panel will display N/A and
  will not emit veto signals (caller should already prevent LIVE in replay).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple
import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.utils.logger import setup_logger


# Obsidian Night color palette (roadmap §1.2 — centralized in theme.json)
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
_ON_BG_GREY = "#14283A"        # hover overlay — disabled/unavailable bg
_ON_FG_GREY = "#4C5B6D"        # text disabled — unavailable text
_ON_BG_DARK = "#0B1724"        # elevated panel — container neutral
_ON_BORDER = "#163043"         # border default


class GlobalVitalsPanel(QFrame):
    panel_id: str = "global_vitals"
    title: str = "GLOBAL VITALS"
    refresh_class: str = "hot"
    required_snapshot_keys = [
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
    default_visible: bool = True

    veto_state_changed = pyqtSignal(bool)

    # ---------------------------------
    # Component Guard Item
    # ---------------------------------

    class ComponentGuardItem(QWidget):
        """
        Small compact widget for a single heavyweight.
        Displays: symbol, price, change%, weight, veto status, sparkline text.
        """

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)

            self._symbol = QLabel("\u2014")
            self._price = QLabel("\u2014")
            self._chg = QLabel("\u2014")
            self._weight = QLabel("\u2014")
            self._veto = QLabel("DISABLED")
            self._spark = QLabel("")

            self._symbol.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._spark.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            sym_font = QFont()
            sym_font.setBold(True)
            sym_font.setPointSize(max(10, sym_font.pointSize()))
            self._symbol.setFont(sym_font)

            num_font = QFont("Monospace")
            num_font.setStyleHint(QFont.StyleHint.Monospace)
            num_font.setPointSize(max(9, num_font.pointSize()))
            self._price.setFont(num_font)
            self._chg.setFont(num_font)
            self._weight.setFont(num_font)

            veto_font = QFont()
            veto_font.setBold(True)
            veto_font.setPointSize(max(9, veto_font.pointSize()))
            self._veto.setFont(veto_font)

            self._spark.setFont(num_font)
            self._spark.setStyleSheet(f"color: {_ON_BG_TEXT};")

            root = QVBoxLayout()
            root.setContentsMargins(8, 6, 8, 6)
            root.setSpacing(4)

            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(6)
            top.addWidget(self._symbol, stretch=1)
            top.addWidget(self._veto, stretch=0)

            mid = QHBoxLayout()
            mid.setContentsMargins(0, 0, 0, 0)
            mid.setSpacing(8)
            mid.addWidget(self._price, stretch=0)
            mid.addWidget(self._chg, stretch=0)
            mid.addStretch(1)
            mid.addWidget(self._weight, stretch=0)

            root.addLayout(top)
            root.addLayout(mid)
            root.addWidget(self._spark)

            self.setLayout(root)
            self.setObjectName("ComponentGuardItem")
            self.setStyleSheet(self._css_container_neutral())

            self.set_unavailable()

        def set_unavailable(self, reason: str = "Component Guard data unavailable") -> None:
            self._symbol.setText("N/A")
            self._price.setText("\u2014")
            self._chg.setText("\u2014")
            self._weight.setText("\u2014")
            self._veto.setText("DISABLED")
            self._veto.setStyleSheet(self._css_veto("DISABLED"))
            self._spark.setText("no data")
            self.setToolTip(reason)
            self.setStyleSheet(self._css_container_neutral())

        def clear(self) -> None:
            self._symbol.setText("\u2014")
            self._price.setText("\u2014")
            self._chg.setText("\u2014")
            self._weight.setText("\u2014")
            self._veto.setText("DISABLED")
            self._veto.setStyleSheet(self._css_veto("DISABLED"))
            self._spark.setText("")
            self.setToolTip("")
            self.setStyleSheet(self._css_container_neutral())

        def update_item(self, hw: Mapping[str, Any]) -> None:
            symbol = self._as_str(hw.get("symbol", "UNKNOWN"), default="UNKNOWN").upper()
            price = self._to_float(hw.get("price", None), default=None)
            chg = self._to_float(hw.get("change_pct", None), default=None)
            weight = self._to_float(hw.get("contribution_ratio", None), default=None)
            veto = self._as_str(hw.get("veto_status", "UNKNOWN"), default="UNKNOWN").upper()
            spark_data = hw.get("sparkline_data", None)

            self._symbol.setText(self._clip(symbol, 14))
            self._price.setText(self._format_price(price))
            self._chg.setText(self._format_pct(chg, signed=True))
            self._weight.setText(self._format_weight(weight))

            self._veto.setText(self._clip(veto, 10))
            self._veto.setStyleSheet(self._css_veto(veto))

            self._spark.setText(self._sparkline_text(spark_data))
            self.setStyleSheet(self._css_container_for_veto(veto))

            tooltip = (
                f"<b>{self._escape_html(symbol)}</b><br/>"
                f"Price: {self._escape_html(self._format_price(price))}<br/>"
                f"Change: {self._escape_html(self._format_pct(chg, signed=True))}<br/>"
                f"Weight: {self._escape_html(self._format_weight(weight))}<br/>"
                f"Veto: <b>{self._escape_html(veto)}</b>"
            )
            self.setToolTip(tooltip)

        @staticmethod
        def _css_container_neutral() -> str:
            return (
                "QWidget#ComponentGuardItem {"
                f" background-color: {_ON_BG_DARK};"
                f" border: 1px solid {_ON_BORDER};"
                " border-radius: 8px;"
                " }"
            )

        @staticmethod
        def _css_container_for_veto(veto: str) -> str:
            v = (veto or "").strip().upper()
            if v == "VETO":
                return (
                    "QWidget#ComponentGuardItem {"
                    f" background-color: {_ON_BG_RED};"
                    " border: 2px solid " + _ON_FG_RED + ";"
                    " border-radius: 8px;"
                    " }"
                )
            if v == "WATCH":
                return (
                    "QWidget#ComponentGuardItem {"
                    f" background-color: {_ON_BG_AMBER};"
                    " border: 2px solid " + _ON_FG_AMBER + ";"
                    " border-radius: 8px;"
                    " }"
                )
            if v == "OK":
                return (
                    "QWidget#ComponentGuardItem {"
                    f" background-color: {_ON_BG_GREEN};"
                    " border: 1px solid " + _ON_FG_GREEN + ";"
                    " border-radius: 8px;"
                    " }"
                )
            return GlobalVitalsPanel.ComponentGuardItem._css_container_neutral()

        @staticmethod
        def _css_veto(veto: str) -> str:
            v = (veto or "").strip().upper()
            if v == "OK":
                return GlobalVitalsPanel._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN)
            if v == "WATCH":
                return GlobalVitalsPanel._css_pill(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER)
            if v == "VETO":
                return GlobalVitalsPanel._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED)
            if v == "DISABLED":
                return GlobalVitalsPanel._css_pill(fg=_ON_FG_GREY, bg=_ON_BG_DARK)
            return GlobalVitalsPanel._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET)

        @staticmethod
        def _sparkline_text(spark_data: Any, max_len: int = 20) -> str:
            if spark_data is None:
                return "no data"
            if not isinstance(spark_data, (list, tuple)) or len(spark_data) == 0:
                return "no data"

            vals: List[float] = []
            for x in spark_data:
                try:
                    vals.append(float(x))
                except Exception:
                    continue

            if len(vals) < 2:
                return "no data"

            treat_as_prices = any(abs(v) > 50.0 for v in vals)
            if treat_as_prices:
                diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
            else:
                diffs = vals

            out: List[str] = []
            for d in diffs[-max_len:]:
                if d > 0:
                    out.append("\u2191")
                elif d < 0:
                    out.append("\u2193")
                else:
                    out.append("\u2192")
            return "".join(out)

        @staticmethod
        def _format_price(v: Optional[float]) -> str:
            if v is None:
                return "\u2014"
            if abs(v) >= 1000:
                return f"{v:,.1f}"
            return f"{v:.2f}"

        @staticmethod
        def _format_pct(v: Optional[float], signed: bool = False) -> str:
            if v is None:
                return "\u2014"
            s = f"{v:+.2f}%" if signed else f"{v:.2f}%"
            return s

        @staticmethod
        def _format_weight(v: Optional[float]) -> str:
            if v is None:
                return "w:?%"
            w = float(v)
            if 0.0 <= w <= 1.0:
                w *= 100.0
            if w < 0.0:
                w = 0.0
            if w > 100.0:
                w = 100.0
            return f"w:{w:.1f}%"

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
        def _to_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if value is None:
                return default
            try:
                return float(value)
            except Exception:
                return default

        @staticmethod
        def _escape_html(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
            )

    # ---------------------------------
    # Panel init
    # ---------------------------------

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)

        self._log = self._get_logger()
        self._log.info("GlobalVitalsPanel created", extra={"dashboard_component": "panel_global_vitals"})

        self.setObjectName("GlobalVitalsPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        self._replay_mode: bool = False
        self._last_veto_active: Optional[bool] = None
        self._cg_last_warn_time: float = 0.0

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

        title = QLabel("GLOBAL VITALS")
        title.setFont(self._title_font)
        title.setToolTip("Sovereign top-level system + market status (hot snapshot).")
        root.addWidget(title)

        # Top grid: spot, prev close, regime, narrative, mode
        self._top_grid = QGridLayout()
        self._top_grid.setHorizontalSpacing(14)
        self._top_grid.setVerticalSpacing(8)

        self.spot_label = self._add_metric(self._top_grid, 0, 0, "Spot", "Spot/index price.")
        self.prev_close_label = self._add_metric(self._top_grid, 0, 2, "Prev Close", "Previous close.")
        self.regime_label = self._add_metric(self._top_grid, 0, 4, "Regime", "Regime classification.")
        self.narrative_label = self._add_metric(self._top_grid, 1, 0, "Narrative", "Macro narrative label.")
        self.mode_label = self._add_metric(self._top_grid, 1, 2, "Mode", "System mode (ALERT/PAPER/LIVE).")

        self.breadth_badge = QLabel("Breadth: \u2014")
        self.breadth_badge.setFont(self._label_font)
        self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
        self.breadth_badge.setToolTip("Component Guard aggregate breadth approval.")
        self._top_grid.addWidget(QLabel("Breadth"), 1, 4)
        self._top_grid.itemAtPosition(1, 4).widget().setFont(self._label_font)  # type: ignore[union-attr]
        self._top_grid.addWidget(self.breadth_badge, 1, 5)

        self._top_grid.setColumnStretch(1, 1)
        self._top_grid.setColumnStretch(3, 1)
        self._top_grid.setColumnStretch(5, 1)

        root.addLayout(self._top_grid)

        # Second grid: feed health, data quality, transition prob, drawdown, session phase, day type
        self._mid_grid = QGridLayout()
        self._mid_grid.setHorizontalSpacing(14)
        self._mid_grid.setVerticalSpacing(8)

        self.feed_health_label = self._add_metric(self._mid_grid, 0, 0, "Feed Health", "Market data feed state.")
        self.data_quality_label = self._add_metric(self._mid_grid, 0, 2, "Data Quality", "0\u2013100 score (projection).")
        self.transition_prob_label = self._add_metric(self._mid_grid, 0, 4, "Regime P(\u0394)", "Regime transition probability.")
        self.drawdown_label = self._add_metric(self._mid_grid, 1, 0, "Drawdown", "Drawdown percentage (0\u2013100% or 0\u20131).")
        self.session_phase_label = self._add_metric(self._mid_grid, 1, 2, "Session Phase", "Session phase marker.")
        self.day_type_label = self._add_metric(self._mid_grid, 1, 4, "Day Type", "Projected day type.")

        self._mid_grid.setColumnStretch(1, 1)
        self._mid_grid.setColumnStretch(3, 1)
        self._mid_grid.setColumnStretch(5, 1)

        root.addLayout(self._mid_grid)

        # Active brains row
        brains_row = QHBoxLayout()
        brains_row.setContentsMargins(0, 0, 0, 0)
        brains_row.setSpacing(10)

        brains_key = QLabel("Active Brains")
        brains_key.setFont(self._label_font)
        brains_key.setToolTip("List of active strategy brains/components.")
        self.active_brains_label = QLabel("\u2014")
        self.active_brains_label.setFont(self._value_font)
        self.active_brains_label.setTextFormat(Qt.TextFormat.PlainText)
        self.active_brains_label.setWordWrap(True)
        self.active_brains_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
        self.active_brains_label.setToolTip("Active brain list (projection).")

        brains_row.addWidget(brains_key, stretch=0)
        brains_row.addWidget(self.active_brains_label, stretch=1)
        root.addLayout(brains_row)

        # Component Guard strip
        guard_title = QLabel("Component Guard \u2014 Top Heavyweights")
        guard_title.setFont(self._label_font)
        guard_title.setToolTip("Veto indicators for top heavyweight participation/health.")
        root.addWidget(guard_title)

        self._guard_strip = QWidget()
        self._guard_layout = QHBoxLayout()
        self._guard_layout.setContentsMargins(0, 0, 0, 0)
        self._guard_layout.setSpacing(10)
        self._guard_strip.setLayout(self._guard_layout)

        self._guard_items: List[GlobalVitalsPanel.ComponentGuardItem] = []
        for _ in range(5):
            item = GlobalVitalsPanel.ComponentGuardItem(self._guard_strip)
            self._guard_items.append(item)
            self._guard_layout.addWidget(item, stretch=1)

        root.addWidget(self._guard_strip)

        self.setLayout(root)
        self.clear()

    @staticmethod
    def _get_logger() -> Any:
        """Return project logger without dropping BoundLogger wrappers.

        Integration fix: ``setup_logger`` returns the Junior Aladdin BoundLogger
        wrapper, not a raw ``logging.Logger``. Accepting logger-like objects keeps
        global-vitals and Component Guard audit/warning events in the normal
        project log files while retaining a stdlib fallback if logger creation
        fails.
        """
        name = "dashboard_panels_global_vitals"
        try:
            log = setup_logger(name)
            if all(hasattr(log, method) for method in ("info", "warning", "error")):
                return log
        except Exception:
            pass
        return logging.getLogger(name)

    # ---------------------------------
    # Public API
    # ---------------------------------

    def clear(self) -> None:
        """Reset all fields to neutral placeholders."""
        for lbl in (
            self.spot_label,
            self.prev_close_label,
            self.regime_label,
            self.narrative_label,
            self.mode_label,
            self.feed_health_label,
            self.data_quality_label,
            self.transition_prob_label,
            self.drawdown_label,
            self.session_phase_label,
            self.day_type_label,
        ):
            lbl.setText("\u2014")
            lbl.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

        self.breadth_badge.setText("Breadth: \u2014")
        self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
        self.active_brains_label.setText("\u2014")

        for it in self._guard_items:
            it.set_unavailable()

        self._set_veto_active(None, reason="clear()")

    def update_hot(self, data: Any) -> None:
        """
        Update panel from projected hot snapshot dict (UI-only; never raises).
        """
        try:
            if not isinstance(data, Mapping):
                self._log.warning(
                    "GlobalVitalsPanel.update_hot received non-mapping; clearing. type=%s",
                    type(data).__name__,
                    extra={"dashboard_component": "panel_global_vitals"},
                )
                self._set_unknown_hot()
                return

            self._replay_mode = bool(data.get("replay_mode", data.get("is_replay", False)))

            expected = (
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
            )
            missing = [k for k in expected if k not in data]
            if missing:
                self._log.warning(
                    "GlobalVitalsPanel.update_hot missing keys=%s",
                    missing,
                    extra={"dashboard_component": "panel_global_vitals"},
                )

            spot = self._to_float(data.get("spot", None), default=None)
            prev_close = self._to_float(data.get("previous_close", None), default=None)
            regime = self._as_str(data.get("regime", "UNKNOWN"), default="UNKNOWN").upper().strip()
            narrative = self._as_str(data.get("narrative_label", "NEUTRAL"), default="NEUTRAL").upper().strip()
            mode = self._as_str(data.get("mode", "UNKNOWN"), default="UNKNOWN").upper().strip()

            feed_health = self._as_str(data.get("feed_health", "UNKNOWN"), default="UNKNOWN").upper().strip()
            dqs = self._clamp_float(data.get("data_quality_score", 0.0), 0.0, 100.0)
            trans_prob = self._to_float(data.get("regime_transition_prob", 0.0), default=0.0)
            drawdown = self._to_float(data.get("drawdown_pct", 0.0), default=0.0)
            session_phase = self._as_str(data.get("session_phase", "UNKNOWN"), default="UNKNOWN").strip()
            day_type = self._as_str(data.get("day_type", "UNKNOWN"), default="UNKNOWN").strip()

            brains = data.get("active_brains", [])
            brains_text = self._format_active_brains(brains)

            # Text updates
            self.spot_label.setText(self._format_price(spot))
            self.prev_close_label.setText(self._format_price(prev_close))
            self.regime_label.setText(self._clip(regime, 16))
            self.narrative_label.setText(self._clip(narrative, 22))
            self.mode_label.setText(self._clip(mode, 10))

            self.feed_health_label.setText(self._clip(feed_health, 12))
            self.data_quality_label.setText(f"{dqs:.0f}/100")
            self.transition_prob_label.setText(self._format_pct(self._pct01_to_pct(trans_prob), signed=False))
            self.drawdown_label.setText(self._format_pct(self._pct01_to_pct(drawdown), signed=False))
            self.session_phase_label.setText(self._clip(session_phase, 18))
            self.day_type_label.setText(self._clip(day_type, 18))

            self.active_brains_label.setText(brains_text)

            # Styling (Obsidian Night palette)
            self._style_feed_health(self.feed_health_label, feed_health)
            self._style_data_quality(self.data_quality_label, dqs)
            self._style_regime(self.regime_label, regime)
            self._style_narrative(self.narrative_label, narrative)
            self._style_mode(self.mode_label, mode)

            # Others neutral
            self.spot_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
            self.prev_close_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
            self.transition_prob_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
            self.drawdown_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
            self.session_phase_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
            self.day_type_label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

            if self._replay_mode:
                self.setToolTip("Replay mode: Component Guard veto signalling is disabled.")
            else:
                self.setToolTip("")

        except Exception as e:
            self._log.warning(
                "GlobalVitalsPanel.update_hot failed; showing unknown. err=%s",
                repr(e),
                extra={"dashboard_component": "panel_global_vitals"},
            )
            self._set_unknown_hot()

    def update_component_guard(self, heavyweights: Any) -> None:
        """
        Update Component Guard strip.

        Expected heavyweights: list[dict] with keys:
            symbol, price, change_pct, contribution_ratio, veto_status, sparkline_data (optional)
        """
        try:
            if self._replay_mode:
                self._update_guard_items_best_effort(heavyweights, replay=True)
                self._update_breadth_badge_from_items(heavyweights, replay=True)
                self._set_veto_active(None, reason="replay_mode")
                return

            if not isinstance(heavyweights, list):
                if self._cg_should_log():
                    self._log.warning(
                        "Component Guard data malformed (not a list). type=%s",
                        type(heavyweights).__name__,
                        extra={"dashboard_component": "panel_global_vitals"},
                    )
                self._set_guard_unavailable("Component Guard unavailable (malformed payload)")
                self._set_veto_active(True, reason="component_guard_malformed")
                return

            if len(heavyweights) == 0:
                if self._cg_should_log():
                    self._log.warning(
                        "Component Guard data empty; blocking (safe).",
                        extra={"dashboard_component": "panel_global_vitals"},
                    )
                self._set_guard_unavailable("Component Guard unavailable (empty list)")
                self._set_veto_active(True, reason="component_guard_empty")
                return

            veto_active = False
            any_watch = False
            any_ok = False

            for i, item in enumerate(self._guard_items):
                if i < len(heavyweights) and isinstance(heavyweights[i], Mapping):
                    hw = heavyweights[i]
                    item.update_item(hw)
                    status = self._as_str(hw.get("veto_status", "UNKNOWN"), default="UNKNOWN").upper().strip()
                    if status == "VETO":
                        veto_active = True
                    elif status == "WATCH":
                        any_watch = True
                    elif status == "OK":
                        any_ok = True
                else:
                    item.set_unavailable("Missing heavyweight entry")
                    veto_active = True

            self._update_breadth_badge(veto_active=veto_active, any_watch=any_watch, any_ok=any_ok)
            self._set_veto_active(veto_active, reason="component_guard_update")

        except Exception as e:
            if self._cg_should_log():
                self._log.warning(
                    "update_component_guard failed; blocking (safe). err=%s",
                    repr(e),
                    extra={"dashboard_component": "panel_global_vitals"},
                )
            self._set_guard_unavailable("Component Guard unavailable (exception)")
            if not self._replay_mode:
                self._set_veto_active(True, reason="component_guard_exception")

    # ---------------------------------
    # Component Guard rate-limit helper
    # ---------------------------------

    def _cg_should_log(self, interval_sec: float = 10.0) -> bool:
        now = time.monotonic()
        if now - self._cg_last_warn_time >= interval_sec:
            self._cg_last_warn_time = now
            return True
        return False

    # ---------------------------------
    # Internal helpers
    # ---------------------------------

    def _set_unknown_hot(self) -> None:
        for lbl in (
            self.spot_label,
            self.prev_close_label,
            self.regime_label,
            self.narrative_label,
            self.mode_label,
            self.feed_health_label,
            self.data_quality_label,
            self.transition_prob_label,
            self.drawdown_label,
            self.session_phase_label,
            self.day_type_label,
        ):
            lbl.setText("?")
            lbl.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))
        self.active_brains_label.setText("?")
        self.breadth_badge.setText("Breadth: ?")
        self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))

    def _set_guard_unavailable(self, reason: str) -> None:
        for it in self._guard_items:
            it.set_unavailable(reason)
        self.breadth_badge.setText("Breadth: UNAVAILABLE")
        self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_GREY, bg=_ON_BG_DARK))
        self.breadth_badge.setToolTip(reason)

    def _update_guard_items_best_effort(self, heavyweights: Any, replay: bool) -> None:
        if not isinstance(heavyweights, list) or len(heavyweights) == 0:
            self._set_guard_unavailable("Component Guard unavailable (replay/no data)")
            return

        for i, item in enumerate(self._guard_items):
            if i < len(heavyweights) and isinstance(heavyweights[i], Mapping):
                item.update_item(heavyweights[i])
            else:
                item.set_unavailable("Missing heavyweight entry")

    def _update_breadth_badge(self, veto_active: bool, any_watch: bool, any_ok: bool) -> None:
        if veto_active:
            self.breadth_badge.setText("Breadth: BLOCKED")
            self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))
            self.breadth_badge.setToolTip("At least one heavyweight is VETO or data is incomplete.")
        elif any_watch:
            self.breadth_badge.setText("Breadth: WATCH")
            self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER))
            self.breadth_badge.setToolTip("No veto, but at least one heavyweight is in WATCH state.")
        elif any_ok:
            self.breadth_badge.setText("Breadth: APPROVED")
            self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
            self.breadth_badge.setToolTip("All available heavyweights are OK (no WATCH/VETO).")
        else:
            self.breadth_badge.setText("Breadth: UNAVAILABLE")
            self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_GREY, bg=_ON_BG_DARK))
            self.breadth_badge.setToolTip("No OK symbols available.")

    def _update_breadth_badge_from_items(self, heavyweights: Any, replay: bool) -> None:
        if not isinstance(heavyweights, list) or not heavyweights:
            self.breadth_badge.setText("Breadth: N/A")
            self.breadth_badge.setStyleSheet(self._css_pill(fg=_ON_FG_GREY, bg=_ON_BG_DARK))
            self.breadth_badge.setToolTip("Replay mode: Component Guard not enforced.")
            return

        veto = any(
            isinstance(hw, Mapping) and str(hw.get("veto_status", "")).upper().strip() == "VETO"
            for hw in heavyweights
        )
        watch = any(
            isinstance(hw, Mapping) and str(hw.get("veto_status", "")).upper().strip() == "WATCH"
            for hw in heavyweights
        )
        ok = any(
            isinstance(hw, Mapping) and str(hw.get("veto_status", "")).upper().strip() == "OK"
            for hw in heavyweights
        )
        self._update_breadth_badge(veto_active=veto, any_watch=watch, any_ok=ok)
        self.breadth_badge.setToolTip("Replay mode: Component Guard not enforced.")

    def _set_veto_active(self, veto_active: Optional[bool], reason: str) -> None:
        if veto_active is None:
            self._last_veto_active = None
            return

        if self._last_veto_active is veto_active:
            return

        self._last_veto_active = veto_active
        self._log.info(
            "Component Guard veto_state_changed=%s reason=%s",
            veto_active,
            reason,
            extra={"dashboard_component": "panel_global_vitals"},
        )
        try:
            self.veto_state_changed.emit(bool(veto_active))
        except Exception as e:
            self._log.warning(
                "Failed to emit veto_state_changed. err=%s",
                repr(e),
                extra={"dashboard_component": "panel_global_vitals"},
            )

    def _add_metric(self, grid: QGridLayout, row: int, col: int, label_text: str, tooltip: str) -> QLabel:
        key = QLabel(label_text)
        key.setFont(self._label_font)
        key.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        key.setToolTip(tooltip)

        val = QLabel("\u2014")
        val.setFont(self._value_font)
        val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        val.setTextFormat(Qt.TextFormat.PlainText)
        val.setMinimumWidth(140)
        val.setToolTip(tooltip)
        val.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

        grid.addWidget(key, row, col)
        grid.addWidget(val, row, col + 1)
        return val

    # ---------------------------------
    # Formatting / parsing
    # ---------------------------------

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
    def _to_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _clamp_float(value: Any, lo: float, hi: float) -> float:
        try:
            v = float(value)
        except Exception:
            v = lo
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    @staticmethod
    def _format_price(v: Optional[float]) -> str:
        if v is None:
            return "\u2014"
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        return f"{v:.2f}"

    @staticmethod
    def _pct01_to_pct(v: float) -> float:
        try:
            x = float(v)
        except Exception:
            return 0.0
        if 0.0 <= x <= 1.0:
            x *= 100.0
        if x < 0.0:
            x = 0.0
        if x > 100.0:
            x = 100.0
        return x

    @staticmethod
    def _format_pct(v: float, signed: bool = False) -> str:
        if signed:
            return f"{v:+.2f}%"
        return f"{v:.2f}%"

    @staticmethod
    def _format_active_brains(brains: Any) -> str:
        if brains is None:
            return "\u2014"
        if isinstance(brains, str):
            return brains.strip() if brains.strip() else "\u2014"
        if isinstance(brains, (list, tuple, set)):
            items = [str(x).strip() for x in brains if x is not None and str(x).strip()]
            return ", ".join(items) if items else "\u2014"
        return str(brains)

    # ---------------------------------
    # Styling (Obsidian Night palette)
    # ---------------------------------

    @staticmethod
    def _css_pill(fg: str, bg: str) -> str:
        return (
            "QLabel {"
            f" color: {fg};"
            f" background-color: {bg};"
            " padding: 3px 7px;"
            " border-radius: 6px;"
            " }"
        )

    def _style_feed_health(self, label: QLabel, value: str) -> None:
        v = (value or "").strip().upper()
        if v == "HEALTHY":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif v == "DELAYED":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER))
        elif v == "STALE":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_ORANGE, bg=_ON_BG_ORANGE))
        elif v == "DOWN":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))
        else:
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

    def _style_data_quality(self, label: QLabel, score: float) -> None:
        if score >= 80.0:
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif score >= 60.0:
            label.setStyleSheet(self._css_pill(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER))
        else:
            label.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))

    def _style_regime(self, label: QLabel, regime: str) -> None:
        v = (regime or "").strip().upper()
        if v == "TRENDING":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif v == "RANGE":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_BLUE, bg=_ON_BG_BLUE))
        elif v == "VOLATILE":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_ORANGE, bg=_ON_BG_ORANGE))
        elif v == "CHOP":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))
        elif v == "EVENT":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_PURPLE, bg=_ON_BG_PURPLE))
        else:
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

    def _style_narrative(self, label: QLabel, narrative: str) -> None:
        v = (narrative or "").strip().upper()
        if v == "STRONG_BULLISH":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif v == "MILD_BULLISH":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif v == "NEUTRAL":
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
        elif v == "MILD_BEARISH":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_ORANGE, bg=_ON_BG_ORANGE))
        elif v == "STRONG_BEARISH":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_RED, bg=_ON_BG_RED))
        elif v == "EVENT_RISK":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_PURPLE, bg=_ON_BG_PURPLE))
        else:
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))

    def _style_mode(self, label: QLabel, mode: str) -> None:
        v = (mode or "").strip().upper()
        if v == "LIVE":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_AMBER, bg=_ON_BG_AMBER))
        elif v == "PAPER":
            label.setStyleSheet(self._css_pill(fg=_ON_FG_GREEN, bg=_ON_BG_GREEN))
        elif v == "ALERT":
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))
        else:
            label.setStyleSheet(self._css_pill(fg=_ON_BG_TEXT, bg=_ON_BG_INSET))


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    panel = GlobalVitalsPanel(win)
    win.setCentralWidget(panel)
    win.resize(1200, 520)
    win.show()

    def on_veto(v: bool) -> None:
        print("veto_state_changed:", v)

    panel.veto_state_changed.connect(on_veto)

    hot_data = {
        "spot": 24500.5,
        "previous_close": 24480.0,
        "regime": "TRENDING",
        "narrative_label": "MILD_BULLISH",
        "mode": "PAPER",
        "feed_health": "HEALTHY",
        "data_quality_score": 88.0,
        "regime_transition_prob": 0.15,
        "drawdown_pct": 0.02,
        "session_phase": "GOLDEN_AM",
        "day_type": "TREND_DAY",
        "active_brains": ["structural", "institutional"],
    }
    panel.update_hot(hot_data)

    heavyweights = [
        {"symbol": "RELIANCE", "price": 2850.5, "change_pct": 0.8, "contribution_ratio": 0.12, "veto_status": "OK", "sparkline_data": [0.5, 0.6, 0.7, 0.6, 0.8]},
        {"symbol": "HDFC", "price": 1680.0, "change_pct": 0.3, "contribution_ratio": 0.10, "veto_status": "OK", "sparkline_data": []},
        {"symbol": "INFY", "price": 1650.0, "change_pct": -0.2, "contribution_ratio": 0.08, "veto_status": "WATCH", "sparkline_data": None},
        {"symbol": "ICICI", "price": 1150.0, "change_pct": 1.2, "contribution_ratio": 0.09, "veto_status": "OK", "sparkline_data": [0.2, 0.3, 0.4]},
        {"symbol": "TCS", "price": 3950.0, "change_pct": -0.5, "contribution_ratio": 0.11, "veto_status": "VETO", "sparkline_data": []},
    ]
    panel.update_component_guard(heavyweights)

    raise SystemExit(app.exec())