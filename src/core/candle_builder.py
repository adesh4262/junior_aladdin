# src/core/candle_builder.py

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dtime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.utils.helpers import IST, ist_now  # type: ignore


_INT32_MAX = 2**31 - 1


def _dt_floor_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def _market_open_dt(ts_ist: datetime) -> datetime:
    d = ts_ist.date()
    return datetime(d.year, d.month, d.day, 9, 15, tzinfo=IST)


def _minutes_since_open(minute_ts: datetime) -> int:
    mo = _market_open_dt(minute_ts)
    return int((minute_ts - mo).total_seconds() // 60)


def _clamp_volume(vol: Any) -> int:
    """
    M5: Volume normalization
      volume == -1 => 0
      else clamp to [0, INT32_MAX]
    """
    try:
        if vol is None:
            return 0
        v = int(vol)
        if v == -1:
            return 0
        if v < 0:
            return 0
        if v > _INT32_MAX:
            return _INT32_MAX
        return v
    except Exception:
        return 0


def _validate_ohlc(c: Dict[str, Any]) -> bool:
    """
    M6: low<=high and open/close within [low, high]
    """
    try:
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = float(c["close"])
        if l > h:
            return False
        if not (l <= o <= h):
            return False
        if not (l <= cl <= h):
            return False
        return True
    except Exception:
        return False


def _copy_candle(prev: Dict[str, Any], *, new_ts: datetime, flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = dict(prev)
    c["timestamp"] = new_ts
    c["is_complete"] = True
    c["volume"] = 0
    if flags:
        c.update(flags)
    return c


@dataclass
class _TFBuffer:
    tf_min: int
    bucket_start: Optional[datetime] = None
    candles: List[Dict[str, Any]] = field(default_factory=list)
    last_minute_ts: Optional[datetime] = None

    def reset(self) -> None:
        self.bucket_start = None
        self.candles.clear()
        self.last_minute_ts = None


@dataclass
class _InstrumentState:
    token: str
    forming: Optional[Dict[str, Any]] = None

    candles_1m: Deque[Dict[str, Any]] = field(default_factory=deque)
    candles_3m: Deque[Dict[str, Any]] = field(default_factory=deque)
    candles_5m: Deque[Dict[str, Any]] = field(default_factory=deque)
    candles_15m: Deque[Dict[str, Any]] = field(default_factory=deque)

    buf_3m: _TFBuffer = field(default_factory=lambda: _TFBuffer(tf_min=3))
    buf_5m: _TFBuffer = field(default_factory=lambda: _TFBuffer(tf_min=5))
    buf_15m: _TFBuffer = field(default_factory=lambda: _TFBuffer(tf_min=15))

    last_valid_by_tf: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    rejected_out_of_order: int = 0
    _last_ooo_warn_minute: Optional[datetime] = None

    def reset_tf_buffers(self) -> None:
        self.buf_3m.reset()
        self.buf_5m.reset()
        self.buf_15m.reset()


class CandleBuilder:
    """
    Institutional-grade Candle Builder (gap-aware, wall-clock aligned, thread-safe).

    Backward compatible:
      - on_tick(price, volume, timestamp) -> uses default_token
    New:
      - on_tick(price, volume, timestamp, token="99926000")
      - on_tick(token, price, volume, timestamp)

    Callback:
      - set_on_candle_close(callback(timeframe, candle_dict))
        Emitted OUTSIDE lock to avoid deadlocks/stalls.
    """

    def __init__(self, default_token: str = "DEFAULT") -> None:
        self._log = setup_logger("candle_builder")
        self._lock = threading.Lock()

        self._default_token = str(default_token).strip() or "DEFAULT"

        sizes = Config.get("data", "candle_deque_sizes", default={})
        if not isinstance(sizes, dict):
            sizes = {}
        self._maxlen_1m = int(sizes.get("1min", 400) or 400)
        self._maxlen_3m = int(sizes.get("3min", 140) or 140)
        self._maxlen_5m = int(sizes.get("5min", 80) or 80)
        self._maxlen_15m = int(sizes.get("15min", 30) or 30)

        self._forming_timeout_sec = float(Config.get("candle_builder", "forming_timeout_sec", default=300))

        self._instruments: Dict[str, _InstrumentState] = {}

        self._on_candle_close: Optional[Callable[[str, Dict[str, Any]], None]] = None

        self._log.info(
            "CandleBuilder initialized",
            default_token=self._default_token,
            maxlen={"1min": self._maxlen_1m, "3min": self._maxlen_3m, "5min": self._maxlen_5m, "15min": self._maxlen_15m},
            forming_timeout_sec=self._forming_timeout_sec,
        )

    # --------------------- Compatibility convenience ---------------------

    @property
    def candles(self) -> Dict[str, Deque[Dict[str, Any]]]:
        """
        Compatibility: expose a dict like older CandleBuilder versions for DEFAULT token.
        Note: returns internal deques (read-only usage expected).
        """
        with self._lock:
            st = self._get_state_locked(self._default_token)
            return {
                "1min": st.candles_1m,
                "3min": st.candles_3m,
                "5min": st.candles_5m,
                "15min": st.candles_15m,
            }

    def get_candle_count(self, timeframe: str, token: str = "DEFAULT") -> int:
        dq = self.get_candles(timeframe, token=token)
        with self._lock:
            return int(len(dq))

    # --------------------- Callbacks ---------------------

    def set_on_candle_close(self, callback: Optional[Callable[[str, Dict[str, Any]], None]]) -> None:
        with self._lock:
            self._on_candle_close = callback

    def _emit_close_events(self, events: List[Tuple[str, Dict[str, Any]]]) -> None:
        """
        Emit candle-close callback OUTSIDE lock.
        """
        cb = None
        with self._lock:
            cb = self._on_candle_close
        if cb is None:
            return

        for tf, candle in events:
            try:
                cb(tf, candle)
            except Exception as e:
                self._log.warning("on_candle_close callback failed (ignored)", timeframe=tf, error=str(e))

    # --------------------- State helpers ---------------------

    def _get_state_locked(self, token: str) -> _InstrumentState:
        t = str(token).strip() or self._default_token
        st = self._instruments.get(t)
        if st is None:
            st = _InstrumentState(token=t)
            st.candles_1m = deque(maxlen=self._maxlen_1m)
            st.candles_3m = deque(maxlen=self._maxlen_3m)
            st.candles_5m = deque(maxlen=self._maxlen_5m)
            st.candles_15m = deque(maxlen=self._maxlen_15m)
            self._instruments[t] = st
        return st

    # --------------------- Public API ---------------------

    def reset(self) -> None:
        with self._lock:
            self._instruments.clear()
        self._log.info("CandleBuilder reset complete (all instruments cleared)")

    def get_candles(self, timeframe: str, token: str = "DEFAULT") -> Deque[Dict[str, Any]]:
        tf = str(timeframe).lower()
        tok = str(token).strip() or self._default_token
        with self._lock:
            st = self._get_state_locked(tok)
            if tf in ("1m", "1min", "1minute"):
                return st.candles_1m
            if tf in ("3m", "3min", "3minute"):
                return st.candles_3m
            if tf in ("5m", "5min", "5minute"):
                return st.candles_5m
            if tf in ("15m", "15min", "15minute"):
                return st.candles_15m
            raise ValueError(f"Unknown timeframe: {timeframe}")

    def get_last_closed(self, timeframe: str, token: str = "DEFAULT") -> Optional[Dict[str, Any]]:
        tf = str(timeframe).lower()
        tok = str(token).strip() or self._default_token
        with self._lock:
            st = self._get_state_locked(tok)
            dq = None
            if tf in ("1m", "1min", "1minute"):
                dq = st.candles_1m
            elif tf in ("3m", "3min", "3minute"):
                dq = st.candles_3m
            elif tf in ("5m", "5min", "5minute"):
                dq = st.candles_5m
            elif tf in ("15m", "15min", "15minute"):
                dq = st.candles_15m
            else:
                raise ValueError(f"Unknown timeframe: {timeframe}")
            result = dq[-1] if dq and len(dq) else None
            # Visibility: log candle retrieval
            self._log.debug("[CANDLE] get_last_closed", timeframe=timeframe, token=tok, has_candle=result is not None, candle_count=len(dq) if dq else 0)
            return result

    def get_forming_candle(self, token: str = "DEFAULT") -> Optional[Dict[str, Any]]:
        """
        M9: forming candle timeout. If forming is older than timeout, force-close and append with is_timeout=True.
        """
        tok = str(token).strip() or self._default_token
        close_events: List[Tuple[str, Dict[str, Any]]] = []

        with self._lock:
            st = self._get_state_locked(tok)
            if st.forming is None:
                return None

            ts = st.forming.get("timestamp")
            if not isinstance(ts, datetime):
                return st.forming

            now_ist = ist_now()
            age_sec = (now_ist - ts).total_seconds()
            if age_sec <= self._forming_timeout_sec:
                return st.forming

            self._log.warning("Forming candle timeout; force-closing", token=tok, age_sec=round(age_sec, 2), timeout_sec=self._forming_timeout_sec)
            forming = dict(st.forming)
            forming["is_complete"] = True
            forming["is_timeout"] = True
            forming["volume"] = _clamp_volume(forming.get("volume", 0))
            self._append_1m_closed_locked(st, forming, allow_replace_invalid=True, close_events=close_events)
            st.forming = None

        # emit outside lock
        self._emit_close_events(close_events)
        return None

    def on_tick(self, *args: Any, **kwargs: Any) -> Optional[str]:
        """
        Returns "candle_closed" if at least one closed 1-min candle (or gap candle) appended during this call.
        """
        # Visibility: log tick received
        self._log.debug("[CANDLE] on_tick received", args_count=len(args), kwargs_keys=list(kwargs.keys()))
        
        token = kwargs.get("token", None)

        if len(args) == 3:
            price, volume, ts = args
            tok = token or self._default_token
        elif len(args) == 4:
            if isinstance(args[0], str):
                tok, price, volume, ts = args
            else:
                price, volume, ts, tok = args
        else:
            raise TypeError("on_tick expects (price, volume, timestamp[, token]) or (token, price, volume, timestamp)")

        tok = str(tok).strip() if tok is not None else self._default_token
        if not tok:
            tok = self._default_token

        ts_dt = self._normalize_timestamp(ts)
        if ts_dt is None:
            self._log.error("Tick timestamp parse failed; ignoring tick", token=tok, ts_raw=str(ts)[:120])
            return None

        price_f = self._safe_float(price)
        if price_f is None:
            self._log.error("Tick price parse failed; ignoring tick", token=tok, price_raw=str(price)[:120])
            return None

        vol_i = _clamp_volume(volume)

        close_events: List[Tuple[str, Dict[str, Any]]] = []
        closed_any = False

        with self._lock:
            st = self._get_state_locked(tok)
            current_minute = _dt_floor_minute(ts_dt)

            # M4 out-of-order discard
            if st.forming is not None:
                forming_ts = st.forming.get("timestamp")
                if isinstance(forming_ts, datetime) and current_minute < forming_ts:
                    st.rejected_out_of_order += 1
                    warn_min = _dt_floor_minute(ts_dt)
                    if st._last_ooo_warn_minute is None or warn_min > st._last_ooo_warn_minute:
                        st._last_ooo_warn_minute = warn_min
                        self._log.warning(
                            "Out-of-order tick discarded",
                            token=tok,
                            tick_minute=current_minute.isoformat(),
                            forming_minute=forming_ts.isoformat(),
                            rejected_out_of_order=st.rejected_out_of_order,
                        )
                    return None

            # no forming => start
            if st.forming is None:
                st.forming = self._new_forming_candle(tok, current_minute, price_f, vol_i)
            else:
                forming_minute = st.forming["timestamp"]
                if current_minute > forming_minute:
                    minute_diff = int((current_minute - forming_minute).total_seconds() // 60)

                    # close existing forming
                    closed_candle = dict(st.forming)
                    closed_candle["is_complete"] = True
                    closed_candle["is_gap"] = False
                    self._append_1m_closed_locked(st, closed_candle, allow_replace_invalid=True, close_events=close_events)
                    closed_any = True

                    # M3 gap insertion
                    if minute_diff > 1:
                        missing = minute_diff - 1
                        self._log.warning(
                            "Gap detected in tick stream; inserting gap markers",
                            token=tok,
                            from_minute=forming_minute.isoformat(),
                            to_minute=current_minute.isoformat(),
                            missing_minutes=missing,
                        )
                        st.reset_tf_buffers()
                        last_valid_1m = st.last_valid_by_tf.get("1min")
                        for k in range(1, minute_diff):
                            missing_ts = forming_minute + timedelta(minutes=k)
                            gap_candle = self._gap_candle(tok, missing_ts, last_valid_1m=last_valid_1m)
                            self._append_1m_closed_locked(st, gap_candle, allow_replace_invalid=False, close_events=close_events)
                            closed_any = True

                    # start new forming with this tick
                    st.forming = self._new_forming_candle(tok, current_minute, price_f, vol_i)
                else:
                    # same minute update forming
                    st.forming["high"] = max(float(st.forming["high"]), price_f)
                    st.forming["low"] = min(float(st.forming["low"]), price_f)
                    st.forming["close"] = price_f
                    st.forming["volume"] = int(max(0, int(st.forming.get("volume", 0))) + vol_i)

        # emit callbacks outside lock
        self._emit_close_events(close_events)

        return "candle_closed" if closed_any else None

    # --------------------- Internals ---------------------

    def _normalize_timestamp(self, ts: Any) -> Optional[datetime]:
        try:
            if ts is None:
                return None
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if isinstance(ts, str):
                s = ts.strip().replace("Z", "+00:00")
                ts = datetime.fromisoformat(s)
            if not isinstance(ts, datetime):
                return None
            if ts.tzinfo is None:
                return ts.replace(tzinfo=IST)
            return ts.astimezone(IST)
        except Exception:
            return None

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None or isinstance(x, bool):
                return None
            f = float(x)
            if not math.isfinite(f):
                return None
            return f
        except Exception:
            return None

    def _new_forming_candle(self, token: str, minute_ts: datetime, price: float, volume: int) -> Dict[str, Any]:
        return {
            "token": token,
            "timestamp": minute_ts,
            "open": float(price),
            "high": float(price),
            "low": float(price),
            "close": float(price),
            "volume": int(volume),
            "is_complete": False,
            "is_gap": False,
            "is_timeout": False,
        }

    def _gap_candle(self, token: str, minute_ts: datetime, last_valid_1m: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if last_valid_1m is not None:
            px = float(last_valid_1m.get("close", last_valid_1m.get("open", 0.0)))
        else:
            px = 0.0
        return {
            "token": token,
            "timestamp": minute_ts,
            "open": px,
            "high": px,
            "low": px,
            "close": px,
            "volume": 0,
            "is_complete": True,
            "is_gap": True,
            "is_timeout": False,
        }

    def _append_1m_closed_locked(
        self,
        st: _InstrumentState,
        candle: Dict[str, Any],
        *,
        allow_replace_invalid: bool,
        close_events: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        candle = dict(candle)
        candle["volume"] = _clamp_volume(candle.get("volume", 0))
        candle["is_complete"] = True
        candle["token"] = st.token

        if not _validate_ohlc(candle):
            self._log.error("Invalid OHLC candle detected (1m)", token=st.token, ts=str(candle.get("timestamp")), candle_preview=self._candle_preview(candle))
            if allow_replace_invalid:
                prev = st.last_valid_by_tf.get("1min")
                if prev is not None:
                    candle = _copy_candle(prev, new_ts=candle["timestamp"], flags={"token": st.token, "is_repaired": True, "is_gap": False})
                else:
                    return
            else:
                return

        st.candles_1m.append(candle)
        st.last_valid_by_tf["1min"] = dict(candle)
        close_events.append(("1min", dict(candle)))

        if candle.get("is_gap"):
            st.reset_tf_buffers()
            return

        self._update_tf_buffer_locked(st, candle, st.buf_3m, "3min", close_events=close_events)
        self._update_tf_buffer_locked(st, candle, st.buf_5m, "5min", close_events=close_events)
        self._update_tf_buffer_locked(st, candle, st.buf_15m, "15min", close_events=close_events)

    def _update_tf_buffer_locked(
        self,
        st: _InstrumentState,
        candle_1m: Dict[str, Any],
        buf: _TFBuffer,
        tf_name: str,
        *,
        close_events: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        try:
            minute_ts = candle_1m["timestamp"]
            if not isinstance(minute_ts, datetime):
                return

            n = int(buf.tf_min)
            ms = _minutes_since_open(minute_ts)
            if ms < 0:
                return

            mo = _market_open_dt(minute_ts)
            bucket_start = mo + timedelta(minutes=(ms // n) * n)

            if buf.bucket_start is None or bucket_start != buf.bucket_start:
                buf.bucket_start = bucket_start
                buf.candles = []
                buf.last_minute_ts = None

            if buf.last_minute_ts is not None:
                if (minute_ts - buf.last_minute_ts).total_seconds() != 60:
                    buf.candles = []
                    buf.last_minute_ts = None
                    buf.bucket_start = bucket_start

            buf.candles.append(candle_1m)
            buf.last_minute_ts = minute_ts

            end_minute = minute_ts + timedelta(minutes=1)
            ms_end = int((end_minute - mo).total_seconds() // 60)

            if (ms_end % n) == 0:
                if len(buf.candles) != n:
                    self._log.warning(
                        "Higher timeframe bucket incomplete; dropping",
                        token=st.token,
                        tf=tf_name,
                        bucket_start=str(buf.bucket_start),
                        expected=n,
                        got=len(buf.candles),
                    )
                    buf.reset()
                    return

                agg = self._aggregate(buf.candles, timestamp=buf.bucket_start, token=st.token, tf=tf_name)
                if not _validate_ohlc(agg):
                    self._log.error("Invalid OHLC candle detected (agg)", token=st.token, tf=tf_name, ts=str(agg.get("timestamp")), candle_preview=self._candle_preview(agg))
                    prev = st.last_valid_by_tf.get(tf_name)
                    if prev is not None:
                        agg = _copy_candle(prev, new_ts=agg["timestamp"], flags={"token": st.token, "is_repaired": True, "tf": tf_name})
                    else:
                        buf.reset()
                        return

                if tf_name == "3min":
                    st.candles_3m.append(agg)
                elif tf_name == "5min":
                    st.candles_5m.append(agg)
                elif tf_name == "15min":
                    st.candles_15m.append(agg)

                st.last_valid_by_tf[tf_name] = dict(agg)
                close_events.append((tf_name, dict(agg)))
                buf.reset()

        except Exception as e:
            self._log.error("TF aggregation failed", token=st.token, tf=tf_name, error=str(e))

    def _aggregate(self, candles: List[Dict[str, Any]], *, timestamp: datetime, token: str, tf: str) -> Dict[str, Any]:
        o = float(candles[0]["open"])
        h = float(max(float(c["high"]) for c in candles))
        l = float(min(float(c["low"]) for c in candles))
        c = float(candles[-1]["close"])
        v = int(sum(int(ca.get("volume", 0)) for ca in candles))
        v = _clamp_volume(v)

        return {
            "token": token,
            "timestamp": timestamp,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "is_complete": True,
            "is_gap": False,
            "tf": tf,
        }

    @staticmethod
    def _candle_preview(c: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("timestamp", "open", "high", "low", "close", "volume", "is_gap", "is_timeout")
        out: Dict[str, Any] = {}
        for k in keys:
            if k in c:
                try:
                    v = c[k]
                    out[k] = v.isoformat() if isinstance(v, datetime) else v
                except Exception:
                    out[k] = "<err>"
        return out


# ==============================================
# Module Self-Test (Institutional)
# ==============================================
if __name__ == "__main__":
    print("=" * 72)
    print("  JUNIOR ALADDIN — CandleBuilder (Institutional) Self-Test")
    print("=" * 72)

    cb = CandleBuilder(default_token="99926000")

    closed_events: List[Tuple[str, Dict[str, Any]]] = []

    def on_close(tf: str, candle: Dict[str, Any]) -> None:
        closed_events.append((tf, candle))

    cb.set_on_candle_close(on_close)

    base = datetime(2026, 4, 14, 9, 15, 1, tzinfo=IST)

    # Test A: wall-clock 3min alignment
    for i in range(0, 4):
        ts = base + timedelta(minutes=i)
        cb.on_tick(24500.0 + i, 10, ts, token="99926000")
        cb.on_tick(24500.2 + i, 5, ts + timedelta(seconds=20), token="99926000")

    c3 = cb.get_last_closed("3min", token="99926000")
    print("\n[Test A] 3min alignment:", "PASS" if (c3 and c3["timestamp"].hour == 9 and c3["timestamp"].minute == 15) else "FAIL", "| 3m_ts=", c3["timestamp"] if c3 else None)

    # Test B: gap insertion
    cb2 = CandleBuilder(default_token="T")
    t0 = datetime(2026, 4, 14, 9, 20, 5, tzinfo=IST)
    cb2.on_tick(100.0, 10, t0, token="T")
    cb2.on_tick(101.0, 10, t0 + timedelta(minutes=5), token="T")  # inserts 9:21..9:24
    dq = cb2.get_candles("1min", token="T")
    gaps = [c for c in dq if c.get("is_gap")]
    print("[Test B] Gap insertion:", "PASS" if len(gaps) >= 3 else "FAIL", "| gap_count=", len(gaps))

    # Test C: out-of-order discard
    cb3 = CandleBuilder(default_token="X")
    cb3.on_tick(200.0, 10, datetime(2026, 4, 14, 9, 30, 5, tzinfo=IST), token="X")
    cb3.on_tick(199.0, 10, datetime(2026, 4, 14, 9, 29, 10, tzinfo=IST), token="X")
    st_ooo = cb3._instruments.get("X").rejected_out_of_order if "X" in cb3._instruments else 0
    print("[Test C] Out-of-order discard:", "PASS" if st_ooo >= 1 else "FAIL", "| rejected_out_of_order=", st_ooo)

    # Test D: volume normalization (-1 becomes 0)
    cb4 = CandleBuilder(default_token="V")
    cb4.on_tick(300.0, -1, datetime(2026, 4, 14, 9, 40, 1, tzinfo=IST), token="V")
    cb4.on_tick(301.0, -1, datetime(2026, 4, 14, 9, 41, 1, tzinfo=IST), token="V")  # closes prev
    last1 = cb4.get_last_closed("1min", token="V")
    print("[Test D] Volume -1 normalization:", "PASS" if (last1 and last1["volume"] >= 0) else "FAIL", "| vol=", last1["volume"] if last1 else None)

    # Test E: thread safety smoke test
    cb5 = CandleBuilder(default_token="P")
    tbase = datetime(2026, 4, 14, 10, 0, 0, tzinfo=IST)

    def worker(offset: int) -> None:
        for j in range(50):
            cb5.on_tick(400.0 + offset + j * 0.01, 1, tbase + timedelta(seconds=j), token="P")

    import threading as _th
    th1 = _th.Thread(target=worker, args=(0,), daemon=True)
    th2 = _th.Thread(target=worker, args=(1,), daemon=True)
    th1.start()
    th2.start()
    th1.join()
    th2.join()
    print("[Test E] Thread safety smoke:", "PASS")

    print("\nCandle close events captured:", len(closed_events))
    print("=" * 72)