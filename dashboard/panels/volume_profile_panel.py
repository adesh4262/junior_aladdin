"""
dashboard/panels/volume_profile_panel.py

Panel 04 — VOLUME PROFILE (Roadmap §4.4, Week 09)

Displays session volume profile statistics and a compact horizontal histogram
(volume by price bucket). Data is provided via cold snapshot projection and
fed through `update_cold(data: dict)`.

Design goals:
- Production-grade defensive handling of missing/partial data.
- Fast update path; paint is lightweight and scales to ~100 levels.
- Replay-safe (purely driven by snapshot payloads; no backend calls).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PyQt6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QGridLayout,
    QWidget,
)
from PyQt6.QtCore import Qt, QSize, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QFontMetrics

from src.utils.logger import setup_logger

_log = setup_logger("dashboard_panels_volume_profile")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None


def _safe_number(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or isinstance(x, bool):
            return default
        return float(x)
    except Exception:
        return default


def _format_price(p: Any) -> str:
    f = _safe_float(p)
    if f is None:
        return "—"
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f))}"
    return f"{f:.2f}"


def _format_number(x: Any) -> str:
    f = _safe_float(x)
    if f is None:
        return "—"
    af = abs(f)
    if af >= 1e9 or (af > 0 and af < 1e-3):
        return f"{f:.3e}"
    if af >= 1e6:
        return f"{f/1e6:.3f}M"
    if af >= 1e3:
        return f"{f/1e3:.3f}K"
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f))}"
    return f"{f:.3f}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class _ProfilePrepared:
    levels: List[Tuple[float, float]]
    max_volume: float
    has_data: bool


class ProfileHistogramWidget(QWidget):
    """Custom histogram widget: horizontal bars for volume by price bucket."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)

        self._prepared: _ProfilePrepared = _ProfilePrepared(levels=[], max_volume=1.0, has_data=False)
        self._poc: Optional[float] = None
        self._vah: Optional[float] = None
        self._val: Optional[float] = None
        self._hvn_levels: Sequence[float] = ()
        self._lvn_levels: Sequence[float] = ()

        self._no_data_text: str = "No volume profile data"
        self._last_profile_hash: Optional[int] = None

        self._bg = QColor(18, 18, 20)
        self._fg = QColor(230, 230, 235)
        self._grid = QColor(70, 70, 80)
        self._bar = QColor(70, 130, 180)
        self._bar_dim = QColor(70, 130, 180, 130)
        self._poc_color = QColor(0, 220, 220)
        self._va_color = QColor(0, 200, 90)
        self._hvn_color = QColor(240, 200, 0)
        self._lvn_color = QColor(255, 80, 80)

        self.setAutoFillBackground(False)

    def sizeHint(self) -> QSize:
        return QSize(520, 260)

    def set_profile(self, profile: Mapping[Any, Any], poc: Any, vah: Any, val: Any) -> None:
        if profile is None:
            profile = {}
        profile_hash: Optional[int]
        try:
            items = tuple(sorted((str(k), str(v)) for k, v in profile.items()))
            profile_hash = hash(items + (str(poc), str(vah), str(val)))
        except Exception:
            profile_hash = None
        if profile_hash is not None and profile_hash == self._last_profile_hash:
            return
        self._last_profile_hash = profile_hash
        self._poc = _safe_float(poc)
        self._vah = _safe_float(vah)
        self._val = _safe_float(val)
        levels: List[Tuple[float, float]] = []
        max_vol = 0.0
        if isinstance(profile, Mapping):
            for k, v in profile.items():
                pk = _safe_float(k)
                vv = _safe_number(v, default=0.0)
                if pk is None:
                    continue
                if not math.isfinite(pk) or not math.isfinite(vv):
                    continue
                if vv < 0:
                    vv = 0.0
                levels.append((pk, vv))
                if vv > max_vol:
                    max_vol = vv
        levels.sort(key=lambda t: t[0])
        has_data = len(levels) > 0 and max_vol > 0.0
        if not has_data:
            self._prepared = _ProfilePrepared(levels=[], max_volume=1.0, has_data=False)
        else:
            self._prepared = _ProfilePrepared(levels=levels, max_volume=max(1e-12, max_vol), has_data=True)
        self.update()

    def set_nodes(self, hvn_levels: Sequence[Any] = (), lvn_levels: Sequence[Any] = ()) -> None:
        def _clean(seq: Sequence[Any]) -> List[float]:
            out: List[float] = []
            for x in seq or ():
                fx = _safe_float(x)
                if fx is None or not math.isfinite(fx):
                    continue
                out.append(fx)
            out.sort()
            return out
        self._hvn_levels = _clean(hvn_levels)
        self._lvn_levels = _clean(lvn_levels)
        self.update()

    def _y_for_price(self, price: float, rect: QRectF, levels: Sequence[Tuple[float, float]]) -> Optional[float]:
        if not levels:
            return None
        prices = [p for p, _ in levels]
        if price <= prices[0]:
            idx = 0
        elif price >= prices[-1]:
            idx = len(prices) - 1
        else:
            lo, hi = 0, len(prices) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if prices[mid] < price:
                    lo = mid + 1
                elif prices[mid] > price:
                    hi = mid - 1
                else:
                    lo = mid
                    break
            idx_hi = _clamp(float(lo), 0.0, float(len(prices) - 1))
            idx_lo = int(max(0, min(len(prices) - 1, int(idx_hi) - 1)))
            idx_hi_i = int(max(0, min(len(prices) - 1, int(idx_hi))))
            idx = idx_hi_i if abs(prices[idx_hi_i] - price) < abs(prices[idx_lo] - price) else idx_lo
        n = len(levels)
        if n <= 0:
            return None
        row_h = rect.height() / n
        return rect.top() + (idx + 0.5) * row_h

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.fillRect(self.rect(), self._bg)
            prepared = self._prepared
            if not prepared.has_data:
                p.setPen(QPen(self._fg))
                fm = QFontMetrics(p.font())
                text = self._no_data_text
                tw = fm.horizontalAdvance(text)
                th = fm.height()
                p.drawText(int((self.width() - tw) / 2), int((self.height() + th) / 2), text)
                return
            levels = prepared.levels
            max_vol = prepared.max_volume
            left_pad, right_pad, top_pad, bottom_pad = 64, 10, 8, 8
            inner = QRectF(float(left_pad), float(top_pad), float(max(1, self.width() - left_pad - right_pad)), float(max(1, self.height() - top_pad - bottom_pad)))
            n = len(levels)
            row_h = inner.height() / max(1, n)
            bar_h = max(1.0, row_h * 0.72)
            p.setPen(QPen(self._grid, 1, Qt.PenStyle.DotLine))
            for frac in (0.25, 0.5, 0.75, 1.0):
                x = inner.left() + inner.width() * frac
                p.drawLine(int(x), int(inner.top()), int(x), int(inner.bottom()))
            max_labels = 10
            stride = max(1, int(math.ceil(n / max_labels)))
            fm = QFontMetrics(p.font())
            p.setPen(QPen(self._fg))
            for i, (price, vol) in enumerate(levels):
                y_center = inner.top() + (i + 0.5) * row_h
                w = 0.0 if max_vol <= 0 else (vol / max_vol) * inner.width()
                x0, y0 = inner.left(), y_center - bar_h / 2.0
                is_poc_bucket = self._poc is not None and abs(price - self._poc) < 1e-9
                brush = QBrush(self._bar if is_poc_bucket else self._bar_dim)
                p.fillRect(QRectF(x0, y0, max(0.0, w), bar_h), brush)
                if i % stride == 0 or i == n - 1:
                    label = _format_price(price)
                    tw = fm.horizontalAdvance(label)
                    p.drawText(int(inner.left() - 8 - tw), int(y_center + fm.ascent() / 2), label)
            if self._vah is not None and self._val is not None and self._vah >= self._val:
                y_vah = self._y_for_price(self._vah, inner, levels)
                y_val = self._y_for_price(self._val, inner, levels)
                if y_vah is not None and y_val is not None:
                    y_top, y_bot = min(y_vah, y_val), max(y_vah, y_val)
                    shade = QColor(self._va_color); shade.setAlpha(35)
                    p.fillRect(QRectF(inner.left(), y_top, inner.width(), max(1.0, y_bot - y_top)), QBrush(shade))
            def draw_level_line(price: Optional[float], color: QColor, label: str) -> None:
                if price is None or not math.isfinite(price): return
                y = self._y_for_price(price, inner, levels)
                if y is None: return
                p.setPen(QPen(color, 2))
                p.drawLine(int(inner.left()), int(y), int(inner.right()), int(y))
                p.setPen(QPen(color))
                p.drawText(int(inner.left() + 6), int(y - 3), f"{label} {_format_price(price)}")
            draw_level_line(self._poc, self._poc_color, "POC")
            draw_level_line(self._vah, self._va_color, "VAH")
            draw_level_line(self._val, self._va_color, "VAL")
            def draw_ticks(levels_seq: Sequence[float], color: QColor) -> None:
                if not levels_seq: return
                p.setPen(QPen(color, 2))
                for pr in levels_seq:
                    y = self._y_for_price(pr, inner, levels)
                    if y is None: continue
                    p.drawLine(int(inner.right() - 8), int(y), int(inner.right()), int(y))
            draw_ticks(self._hvn_levels, self._hvn_color)
            draw_ticks(self._lvn_levels, self._lvn_color)
        finally:
            p.end()


class VolumeProfilePanel(QFrame):
    """
    Panel 04 — Volume Profile (Roadmap §4.4, Week 09).

    Public API:
    - update_cold(self, data: dict) -> None

    Panel contract:
    - panel_id: "volume_profile"
    - title: "VOLUME PROFILE"
    - refresh_class: "warm"
    - required_snapshot_keys: ("volume_profile", "microstructure")
    """

    panel_id: str = "volume_profile"
    title: str = "VOLUME PROFILE"
    refresh_class: str = "warm"
    required_snapshot_keys: tuple = ("volume_profile", "microstructure")
    default_visible: bool = True

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.setObjectName("VolumeProfilePanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        self._profile_data: Dict[float, float] = {}
        self._poc: Optional[float] = None
        self._vah: Optional[float] = None
        self._val: Optional[float] = None
        self._bucket_size: int = 5

        self._last_exhaustion: bool = False
        self._last_absorption: bool = False
        self._missing_logged: bool = False

        self._exhaustion_count: int = 0
        self._absorption_count: int = 0
        self._last_exhaustion_ts: Optional[str] = None
        self._last_absorption_ts: Optional[str] = None

        self._cvd_max_abs_seen: float = 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("VOLUME PROFILE", self)
        title.setStyleSheet("font-weight: 700; letter-spacing: 0.5px;")
        title_row.addWidget(title)
        title_row.addStretch(1)

        self._status_label = QLabel("", self)
        self._status_label.setStyleSheet("color: #A0A0A8;")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(self._status_label)
        root.addLayout(title_row)

        self.histogram_widget = ProfileHistogramWidget(self)
        root.addWidget(self.histogram_widget, stretch=1)

        # Stats grid
        stats = QGridLayout()
        stats.setHorizontalSpacing(12)
        stats.setVerticalSpacing(4)

        def k(label: str) -> QLabel:
            w = QLabel(label, self)
            w.setStyleSheet("color: #A0A0A8;")
            return w

        def v(initial: str = "—") -> QLabel:
            w = QLabel(initial, self)
            w.setStyleSheet("color: #E8E8EE; font-weight: 600;")
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return w

        self.poc_label = v()
        self.poc_vol_label = v()
        self.vah_label = v()
        self.val_label = v()
        self.hvn_label = v()
        self.lvn_label = v()
        self.bucket_label = v()

        stats.addWidget(k("POC"), 0, 0); stats.addWidget(self.poc_label, 0, 1)
        stats.addWidget(k("POC VOL"), 0, 2); stats.addWidget(self.poc_vol_label, 0, 3)
        stats.addWidget(k("VAH"), 1, 0); stats.addWidget(self.vah_label, 1, 1)
        stats.addWidget(k("VAL"), 1, 2); stats.addWidget(self.val_label, 1, 3)
        stats.addWidget(k("HVN"), 2, 0); stats.addWidget(self.hvn_label, 2, 1)
        stats.addWidget(k("LVN"), 2, 2); stats.addWidget(self.lvn_label, 2, 3)
        stats.addWidget(k("BUCKET"), 3, 0); stats.addWidget(self.bucket_label, 3, 1)
        root.addLayout(stats)

        # Microstructure row: CVD + imbalance + badges
        micro = QGridLayout()
        micro.setHorizontalSpacing(12)
        micro.setVerticalSpacing(4)

        self.cvd_text = QLabel("CVD", self)
        self.cvd_text.setStyleSheet("color: #A0A0A8; font-weight: 600;")
        micro.addWidget(self.cvd_text, 0, 0, 1, 1)

        self.cvd_value = QLabel("—", self)
        self.cvd_value.setStyleSheet("color: #E8E8EE; font-weight: 600;")
        self.cvd_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        micro.addWidget(self.cvd_value, 0, 1, 1, 1)

        self.cvd_bar = QProgressBar(self)
        self.cvd_bar.setRange(-100, 100)
        self.cvd_bar.setValue(0)
        self.cvd_bar.setTextVisible(False)
        self.cvd_bar.setFixedHeight(10)
        self.cvd_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #3A3A44; border-radius: 4px; background: #141418; }"
            "QProgressBar::chunk { border-radius: 3px; }"
        )
        micro.addWidget(self.cvd_bar, 0, 2, 1, 4)

        micro.addWidget(QLabel("IMB", self), 1, 0)
        self.imbalance_chip = QLabel("—", self)
        self.imbalance_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imbalance_chip.setStyleSheet("background: #14283A; color: #9FB0C3; font-weight: 600; padding: 2px 10px; border-radius: 6px;")
        self.imbalance_chip.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        micro.addWidget(self.imbalance_chip, 1, 1, 1, 2)

        # Exhaustion badge — shorter label "EXH" with tooltip
        self.exhaustion_badge = QLabel("EXH", self)
        self.exhaustion_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exhaustion_badge.setStyleSheet("background: #2B1200; color: #FF7A18; font-weight: 600; padding: 2px 8px; border-radius: 6px;")
        self.exhaustion_badge.setToolTip("Exhaustion alert: CVD divergence, potential reversal")
        micro.addWidget(self.exhaustion_badge, 1, 2)

        # Absorption badge — shorter label "ABS" with tooltip
        self.absorption_badge = QLabel("ABS", self)
        self.absorption_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.absorption_badge.setStyleSheet("background: #001C2B; color: #48B9FF; font-weight: 600; padding: 2px 8px; border-radius: 6px;")
        self.absorption_badge.setToolTip("Absorption alert: large volume at key level")
        micro.addWidget(self.absorption_badge, 1, 3)

        root.addLayout(micro)

        # Default placeholder
        self.histogram_widget.set_profile({}, 0, 0, 0)

    def update_cold(self, data: Dict[str, Any]) -> None:
        """Update the panel from projected cold snapshot data."""
        if not isinstance(data, dict):
            _log.warning("update_cold received non-dict payload: %r", type(data))
            return

        volume_profile = data.get("volume_profile") or {}
        microstructure = data.get("microstructure") or {}
        session_anchors = data.get("session_profile_anchors") or {}
        session_volume_profile = data.get("session_volume_profile") or {}

        if not isinstance(volume_profile, dict):
            volume_profile = {}
        if not isinstance(microstructure, dict):
            microstructure = {}

        if not volume_profile:
            self._status_label.setText("No volume profile data")
            if not self._missing_logged:
                _log.warning("Missing volume_profile in cold snapshot projection.")
                self._missing_logged = True
            self.histogram_widget.set_profile({}, 0, 0, 0)
            self.poc_label.setText("—")
            self.poc_vol_label.setText("—")
            self.vah_label.setText("—")
            self.val_label.setText("—")
            self.hvn_label.setText("—")
            self.lvn_label.setText("—")
            self.bucket_label.setText("—")
        else:
            self._missing_logged = False
            self._status_label.setText("")

            profile_raw = volume_profile.get("profile") or {}
            poc = volume_profile.get("poc", 0)
            poc_vol = volume_profile.get("poc_volume", None)
            vah = volume_profile.get("vah", 0)
            val = volume_profile.get("val", 0)
            bucket_size = volume_profile.get("bucket_size", self._bucket_size)

            self._poc = _safe_float(poc)
            self._vah = _safe_float(vah)
            self._val = _safe_float(val)

            bs_i = _safe_int(bucket_size)
            if bs_i is None or bs_i <= 0:
                bs_i = self._bucket_size
            self._bucket_size = bs_i

            hvn_levels = volume_profile.get("hvn_levels") or ()
            lvn_levels = volume_profile.get("lvn_levels") or ()
            hvn_count = volume_profile.get("hvn_count", None)
            lvn_count = volume_profile.get("lvn_count", None)

            if hvn_count is None:
                try:
                    hvn_count = len(hvn_levels) if isinstance(hvn_levels, (list, tuple)) else 0
                except Exception:
                    hvn_count = 0
            if lvn_count is None:
                try:
                    lvn_count = len(lvn_levels) if isinstance(lvn_levels, (list, tuple)) else 0
                except Exception:
                    lvn_count = 0

            self.poc_label.setText(_format_price(self._poc))
            self.poc_vol_label.setText(_format_number(poc_vol))
            self.vah_label.setText(_format_price(self._vah))
            self.val_label.setText(_format_price(self._val))
            self.hvn_label.setText(str(_safe_int(hvn_count) or 0))
            self.lvn_label.setText(str(_safe_int(lvn_count) or 0))
            self.bucket_label.setText(str(self._bucket_size))

            if not isinstance(profile_raw, Mapping):
                profile_raw = {}
            self.histogram_widget.set_profile(profile_raw, self._poc or 0, self._vah or 0, self._val or 0)
            if isinstance(hvn_levels, (list, tuple)) or isinstance(lvn_levels, (list, tuple)):
                self.histogram_widget.set_nodes(hvn_levels, lvn_levels)

        # --- Session anchors (Thursday) ---
        # Read separate session profile anchors from snapshot (if provided).
        # These are key levels computed per session, not from volume_profile dict.
        session_poc = _safe_float(session_anchors.get("poc", session_anchors.get("anchor_poc", None)))
        session_vah = _safe_float(session_anchors.get("vah", session_anchors.get("anchor_vah", None)))
        session_val = _safe_float(session_anchors.get("val", session_anchors.get("anchor_val", None)))

        # Also read session_volume_profile if present (fallback for anchors)
        if session_poc is None:
            session_poc = _safe_float(session_volume_profile.get("poc", None))
        if session_vah is None:
            session_vah = _safe_float(session_volume_profile.get("vah", None))
        if session_val is None:
            session_val = _safe_float(session_volume_profile.get("val", None))

        # If we have session anchors and no volume_profile data was present, push to histogram
        if not volume_profile and session_poc is not None:
            self._status_label.setText("Session profile")
            if session_vah is not None and session_val is not None:
                self.histogram_widget.set_profile({}, session_poc, session_vah, session_val)
                self.poc_label.setText(_format_price(session_poc))
                self.vah_label.setText(_format_price(session_vah))
                self.val_label.setText(_format_price(session_val))

        # --- Microstructure: CVD ---
        cvd = _safe_number(microstructure.get("cumulative_volume_delta", None), default=float("nan"))
        if not math.isfinite(cvd):
            self.cvd_value.setText("—")
            self.cvd_bar.setValue(0)
        else:
            self.cvd_value.setText(_format_number(cvd))
            abs_cvd = min(abs(cvd), 1e18)
            self._cvd_max_abs_seen = max(1.0, max(abs_cvd, self._cvd_max_abs_seen * 0.98))
            norm = 0.0 if self._cvd_max_abs_seen <= 0 else (cvd / self._cvd_max_abs_seen)
            norm = _clamp(norm, -1.0, 1.0)
            self.cvd_bar.setValue(int(round(norm * 100.0)))

        # --- Microstructure: Imbalance chip ---
        imbalance = _safe_number(microstructure.get("bid_ask_imbalance", None), default=float("nan"))
        if not math.isfinite(imbalance):
            self.imbalance_chip.setText("—")
            self.imbalance_chip.setStyleSheet("background: #14283A; color: #9FB0C3; font-weight: 600; padding: 2px 10px; border-radius: 6px;")
        else:
            self.imbalance_chip.setText(f"{imbalance:+.3f}")
            if imbalance > 0.3:
                self.imbalance_chip.setStyleSheet("background: #0F2B1E; color: #22D48A; font-weight: 700; padding: 2px 10px; border-radius: 6px;")
            elif imbalance < -0.3:
                self.imbalance_chip.setStyleSheet("background: #2B0000; color: #FF6161; font-weight: 700; padding: 2px 10px; border-radius: 6px;")
            else:
                self.imbalance_chip.setStyleSheet("background: #14283A; color: #FFB020; font-weight: 600; padding: 2px 10px; border-radius: 6px;")

        # --- Microstructure: Exhaustion + Absorption (Wednesday improvement) ---
        # Read alerts lists (richer data than boolean flags alone)
        exhaustion_alerts = microstructure.get("exhaustion_alerts", microstructure.get("exhaustion_detected", False))
        absorption_alerts = microstructure.get("absorption_alerts", microstructure.get("absorption_detected", False))

        # Parse exhaustion — supports both list and bool
        if isinstance(exhaustion_alerts, list) and len(exhaustion_alerts) > 0:
            exhaustion = True
            self._exhaustion_count += len(exhaustion_alerts)
            latest = exhaustion_alerts[-1]
            if isinstance(latest, dict):
                self._last_exhaustion_ts = str(latest.get("timestamp", ""))
        elif isinstance(exhaustion_alerts, bool):
            exhaustion = exhaustion_alerts
            if exhaustion:
                self._exhaustion_count += 1
        else:
            exhaustion = bool(exhaustion_alerts)

        # Parse absorption — supports both list and bool
        if isinstance(absorption_alerts, list) and len(absorption_alerts) > 0:
            absorption = True
            self._absorption_count += len(absorption_alerts)
            latest = absorption_alerts[-1]
            if isinstance(latest, dict):
                self._last_absorption_ts = str(latest.get("timestamp", ""))
        elif isinstance(absorption_alerts, bool):
            absorption = absorption_alerts
            if absorption:
                self._absorption_count += 1
        else:
            absorption = bool(absorption_alerts)

        # Update badge text with counts
        exh_text = f"EXH x{self._exhaustion_count}" if self._exhaustion_count > 0 else "EXH"
        abs_text = f"ABS x{self._absorption_count}" if self._absorption_count > 0 else "ABS"

        # Style: active vs inactive
        if exhaustion:
            self.exhaustion_badge.setText(exh_text)
            self.exhaustion_badge.setStyleSheet("background: #7A1C1C; color: #FFF; font-weight: 800; padding: 3px 8px; border-radius: 6px;")
        else:
            self.exhaustion_badge.setText("EXH")
            self.exhaustion_badge.setStyleSheet("background: #2B1200; color: #FF7A18; font-weight: 600; padding: 2px 8px; border-radius: 6px;")
        self.exhaustion_badge.setVisible(exhaustion or self._exhaustion_count > 0)

        if absorption:
            self.absorption_badge.setText(abs_text)
            self.absorption_badge.setStyleSheet("background: #1C3D7A; color: #FFF; font-weight: 800; padding: 3px 8px; border-radius: 6px;")
        else:
            self.absorption_badge.setText("ABS")
            self.absorption_badge.setStyleSheet("background: #001C2B; color: #48B9FF; font-weight: 600; padding: 2px 8px; border-radius: 6px;")
        self.absorption_badge.setVisible(absorption or self._absorption_count > 0)

        # Log only on state change
        if exhaustion and not self._last_exhaustion:
            _log.warning("Order-flow exhaustion detected", count=self._exhaustion_count, ts=self._last_exhaustion_ts)
        if absorption and not self._last_absorption:
            _log.info("Absorption detected", count=self._absorption_count, ts=self._last_absorption_ts)

        # Update tooltips
        self.exhaustion_badge.setToolTip(
            f"Exhaustion alerts: {self._exhaustion_count} total\n"
            f"Latest: {self._last_exhaustion_ts or 'now'}\n"
            "CVD divergence — potential reversal zone."
        )
        self.absorption_badge.setToolTip(
            f"Absorption alerts: {self._absorption_count} total\n"
            f"Latest: {self._last_absorption_ts or 'now'}\n"
            "Large volume absorbed at key level — accumulation."
        )

        self._last_exhaustion = exhaustion
        self._last_absorption = absorption


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)
    win = QMainWindow()
    panel = VolumeProfilePanel(win)
    win.setCentralWidget(panel)
    win.resize(860, 600)
    win.show()

    test_data = {
        "volume_profile": {
            "profile": {23050: 5000, 23055: 12000, 23060: 8000, 23065: 3000},
            "poc": 23055, "poc_volume": 12000,
            "vah": 23065, "val": 23050, "bucket_size": 5,
            "hvn_count": 2, "lvn_count": 1,
            "hvn_levels": [23055, 23060], "lvn_levels": [23065],
        },
        "microstructure": {
            "cumulative_volume_delta": 125000,
            "bid_ask_imbalance": 0.32,
            "exhaustion_detected": False,
            "absorption_detected": True,
        },
    }
    panel.update_cold(test_data)
    sys.exit(app.exec())