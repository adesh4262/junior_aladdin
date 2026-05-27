"""Periodic refresh clock for dashboard HOT/WARM/COLD update cycles."""

from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

try:
    from src.utils.logger import setup_logger
except Exception:  # pragma: no cover
    import logging

    def setup_logger(name: str):  # type: ignore
        return logging.getLogger(name)


class DashboardClock(QObject):
    """Emit periodic tiered ticks using Qt timers."""

    hot_tick = pyqtSignal()
    warm_tick = pyqtSignal()
    cold_tick = pyqtSignal()

    def __init__(
        self,
        hot_interval_ms: int,
        warm_interval_ms: int,
        cold_interval_ms: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.log = setup_logger("dashboard_core_dashboard_clock")

        self.hot_interval_ms = self._validate_interval("hot", hot_interval_ms)
        self.warm_interval_ms = self._validate_interval("warm", warm_interval_ms)
        self.cold_interval_ms = self._validate_interval("cold", cold_interval_ms)

        self._hot_timer = QTimer(self)
        self._warm_timer = QTimer(self)
        self._cold_timer = QTimer(self)

        self._hot_timer.setInterval(self.hot_interval_ms)
        self._warm_timer.setInterval(self.warm_interval_ms)
        self._cold_timer.setInterval(self.cold_interval_ms)

        self._hot_timer.timeout.connect(self.hot_tick.emit)
        self._warm_timer.timeout.connect(self.warm_tick.emit)
        self._cold_timer.timeout.connect(self.cold_tick.emit)

    def start(self) -> None:
        self._hot_timer.start()
        self._warm_timer.start()
        self._cold_timer.start()
        self.log.info(
            "Dashboard clock started",
            dashboard_component="dashboard_clock",
            hot_interval_ms=self.hot_interval_ms,
            warm_interval_ms=self.warm_interval_ms,
            cold_interval_ms=self.cold_interval_ms,
        )

    def stop(self) -> None:
        self._hot_timer.stop()
        self._warm_timer.stop()
        self._cold_timer.stop()
        self.log.info("Dashboard clock stopped", dashboard_component="dashboard_clock")

    def set_hot_interval(self, ms: int) -> None:
        self._set_interval("hot", int(ms), self._hot_timer)

    def set_warm_interval(self, ms: int) -> None:
        self._set_interval("warm", int(ms), self._warm_timer)

    def set_cold_interval(self, ms: int) -> None:
        self._set_interval("cold", int(ms), self._cold_timer)

    @staticmethod
    def _validate_interval(name: str, ms: int) -> int:
        """Coerce to int and reject non-positive intervals.

        QTimer.setInterval(<=0) fires as fast as the event loop allows, which
        freezes the UI and is almost always a config or call-site bug. Fail
        loudly with a ValueError so the bad value surfaces immediately.
        """
        ms_int = int(ms)
        if ms_int <= 0:
            raise ValueError(
                f"DashboardClock {name}_interval_ms must be > 0 (got {ms_int})"
            )
        return ms_int

    def _set_interval(self, name: str, ms: int, timer: QTimer) -> None:
        ms = self._validate_interval(name, ms)
        was_running = timer.isActive()
        if was_running:
            timer.stop()

        timer.setInterval(ms)
        if name == "hot":
            self.hot_interval_ms = ms
        elif name == "warm":
            self.warm_interval_ms = ms
        elif name == "cold":
            self.cold_interval_ms = ms

        if was_running:
            timer.start()

        self.log.info(
            "Dashboard clock interval updated",
            dashboard_component="dashboard_clock",
            interval_type=name,
            interval_ms=ms,
            timer_running=was_running,
        )


if __name__ == "__main__":
    from PyQt6.QtCore import QCoreApplication

    app = QCoreApplication([])

    hot_count = {"n": 0}
    warm_count = {"n": 0}
    cold_count = {"n": 0}

    clock = DashboardClock(hot_interval_ms=100, warm_interval_ms=500, cold_interval_ms=2000)

    def on_hot() -> None:
        hot_count["n"] += 1
        print("hot")

    def on_warm() -> None:
        warm_count["n"] += 1
        print("warm")

    def on_cold() -> None:
        cold_count["n"] += 1
        print("cold")

    clock.hot_tick.connect(on_hot)
    clock.warm_tick.connect(on_warm)
    clock.cold_tick.connect(on_cold)

    clock.start()

    def finish() -> None:
        clock.stop()
        print(
            "tick_counts",
            {
                "hot": hot_count["n"],
                "warm": warm_count["n"],
                "cold": cold_count["n"],
            },
        )
        app.quit()

    QTimer.singleShot(3000, finish)
    raise SystemExit(app.exec())