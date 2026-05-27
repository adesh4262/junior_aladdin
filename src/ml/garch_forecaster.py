"""
Junior Aladdin — GARCH(1,1) Volatility Forecaster
=================================================

PURPOSE
- Forecast next-day volatility using GARCH(1,1) on daily True Range (TR) expressed as % of prev close.
- Used by Risk Engine for size multiplier:
    if forecast_vol > (median_vol * threshold_mult) -> high vol regime -> reduce size.

DATA
- Reads daily candles parquet (default from config):
    Config.get("historical", "daily_candles_path",
               default="data/historical/candles/NIFTY_daily.parquet")

- Computes daily True Range:
    tr = max(high-low, abs(high-prev_close), abs(low-prev_close))
  and converts to percentage volatility:
    tr_pct = tr / prev_close
  (first row uses close as denominator to avoid NaN).

MODEL
- Uses arch_model(tr_pct * 100, vol="Garch", p=1, q=1) for numerical stability.
- If fit fails or emits warnings -> EWMA fallback (span=20) on tr_pct.

RUN (terminal)
- Self-test:
    python -m src.ml.garch_forecaster
"""

from __future__ import annotations

import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import setup_logger

log = setup_logger("garch_forecaster")


@dataclass
class _FitArtifacts:
    model_type: str  # "GARCH" | "EWMA" | "NONE"
    lookback_days: int
    points_used: int
    fitted_at: Optional[datetime]
    last_train_value: Optional[float]
    ewma_span: Optional[int]


class GARCHForecaster:
    """
    Production-grade GARCH(1,1) volatility forecaster with safe fallbacks.

    Public methods:
      - fit(lookback_days: int = 60) -> bool
      - forecast(horizon: int = 1) -> Optional[float]
      - get_median_volatility() -> float
      - is_high_vol_regime() -> bool
      - get_status() -> Dict[str, Any]
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        default_path = Config.get(
            "historical",
            "daily_candles_path",
            default="data/historical/candles/NIFTY_daily.parquet",
        )
        self._data_path = str(data_path or default_path)

        self._lookback_days_cfg = int(Config.get("ml", "garch_lookback_days", default=60))
        self._high_vol_threshold_mult = float(
            Config.get("risk", "garch_high_vol_multiplier_threshold", default=1.5)
        )

        # Fitted objects/state
        self._fitted: bool = False
        self._use_garch: bool = False
        self._garch_result = None  # arch.univariate.base.ARCHModelResult
        self._ewma_last: Optional[float] = None
        self._median_vol: float = 0.0
        self._last_forecast: Optional[float] = None

        self._artifacts = _FitArtifacts(
            model_type="NONE",
            lookback_days=self._lookback_days_cfg,
            points_used=0,
            fitted_at=None,
            last_train_value=None,
            ewma_span=None,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, lookback_days: int = 60) -> bool:
        """
        Fit GARCH(1,1) model on last `lookback_days` of daily TR% (or EWMA fallback).
        Returns True if fitted successfully (either GARCH or EWMA).
        """
        lookback = int(lookback_days or self._lookback_days_cfg)
        lookback = max(1, lookback)

        path = Path(self._data_path)
        if not path.exists():
            log.critical("Daily candles parquet missing; GARCH forecaster disabled", path=str(path))
            self._reset_fit_state(model_type="NONE", lookback_days=lookback)
            return False

        try:
            df, tr_pct = self._load_and_compute_tr_pct(path)
            if tr_pct is None or tr_pct.empty:
                log.critical("Failed to compute TR% series (empty); GARCH forecaster disabled", path=str(path))
                self._reset_fit_state(model_type="NONE", lookback_days=lookback)
                return False

            # Median volatility baseline: entire history or last 252
            self._median_vol = self._compute_median_volatility(tr_pct)

            # Training subset
            train_series = tr_pct.dropna().astype(float)
            train_series = train_series[np.isfinite(train_series)]
            n_total = int(train_series.shape[0])

            if n_total <= 0:
                log.critical("No valid TR% values available; GARCH forecaster disabled", path=str(path))
                self._reset_fit_state(model_type="NONE", lookback_days=lookback)
                return False

            # Use last lookback values
            train = train_series.iloc[-min(lookback, n_total):].copy()
            points_used = int(train.shape[0])

            # Insufficient data -> EWMA fallback
            if points_used < 30:
                log.warning(
                    "Insufficient data for GARCH fit; using EWMA fallback",
                    points_used=points_used,
                    lookback_days=lookback,
                )
                self._fit_ewma(train, lookback)
                return True

            # Try GARCH fit
            ok = self._fit_garch(train, lookback)
            if ok:
                return True

            # Fallback EWMA
            log.warning("GARCH fit failed; using EWMA fallback", lookback_days=lookback, points_used=points_used)
            self._fit_ewma(train, lookback)
            return True

        except Exception as e:
            log.critical(
                "GARCHForecaster.fit failed with exception; disabling forecaster",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._reset_fit_state(model_type="NONE", lookback_days=lookback)
            return False

    def forecast(self, horizon: int = 1) -> Optional[float]:
        """
        Forecast next-day volatility (TR% as decimal, e.g., 0.015 for 1.5%).
        - If GARCH fitted: sqrt(variance) / 100 (undo scaling).
        - If EWMA fallback: latest EWMA value (already in decimal).
        - If not fitted: None.
        """
        if not self._fitted:
            return None

        h = int(horizon or 1)
        h = max(1, h)

        try:
            if self._use_garch and self._garch_result is not None:
                # Lazy import for type + potential warnings; safe if arch not installed at runtime.
                # NOTE: variance is in (scaled units)^2 because we fit on (tr_pct * 100).
                f = self._garch_result.forecast(horizon=h, reindex=False)
                var_df = f.variance
                if var_df is None or var_df.empty:
                    raise RuntimeError("GARCH forecast produced empty variance")

                # Take the last available forecast for the farthest horizon
                # DataFrame columns may be [h.1, h.2, ...] or numeric; use last column.
                var_last = float(var_df.iloc[-1, -1])
                if not np.isfinite(var_last) or var_last < 0:
                    raise RuntimeError(f"Invalid variance from GARCH forecast: {var_last}")

                vol_scaled = float(np.sqrt(var_last))
                vol_decimal = vol_scaled / 100.0  # undo *100 scaling
                self._last_forecast = float(max(0.0, vol_decimal))
                return self._last_forecast

            # EWMA fallback
            if self._ewma_last is not None and np.isfinite(self._ewma_last):
                self._last_forecast = float(max(0.0, float(self._ewma_last)))
                return self._last_forecast

            return None

        except Exception as e:
            log.error(
                "GARCHForecaster.forecast failed; returning None",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return None

    def get_median_volatility(self) -> float:
        """
        Median daily volatility baseline (TR% as decimal).
        Computed from entire history or last 252 days if longer.
        """
        return float(max(0.0, self._median_vol or 0.0))

    def is_high_vol_regime(self) -> bool:
        """
        Returns True if forecast > median_vol * threshold_mult.
        Safe default: False if forecast unavailable or median=0.
        """
        try:
            forecast_val = self.forecast(horizon=1)
            if forecast_val is None:
                return False

            median = self.get_median_volatility()
            if median <= 0:
                return False

            thr_mult = float(self._high_vol_threshold_mult)
            if not np.isfinite(thr_mult) or thr_mult <= 0:
                thr_mult = 1.5

            return float(forecast_val) > float(median) * float(thr_mult)
        except Exception:
            log.error("is_high_vol_regime failed; returning False", traceback=traceback.format_exc())
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "data_path": self._data_path,
            "fitted": bool(self._fitted),
            "model_type": self._artifacts.model_type,
            "use_garch": bool(self._use_garch),
            "lookback_days": int(self._artifacts.lookback_days),
            "points_used": int(self._artifacts.points_used),
            "fitted_at": self._artifacts.fitted_at.isoformat() if self._artifacts.fitted_at else None,
            "median_volatility": float(self.get_median_volatility()),
            "last_forecast": float(self._last_forecast) if self._last_forecast is not None else None,
            "high_vol_threshold_mult": float(self._high_vol_threshold_mult),
            "is_high_vol_regime": bool(self.is_high_vol_regime()),
            "fallback_ewma_span": self._artifacts.ewma_span,
        }

    # ---------------------------------------------------------------------
    # Internals: data loading / TR% computation
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Accept typical candle schemas; normalize to lower-case columns.
        df2 = df.copy()
        df2.columns = [str(c).strip().lower() for c in df2.columns]
        return df2

    def _load_and_compute_tr_pct(self, path: Path) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Reads daily candles parquet and computes TR% (decimal).
        """
        df = pd.read_parquet(path)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df, None

        df = self._normalize_columns(df)

        required = {"high", "low", "close"}
        missing = sorted(list(required - set(df.columns)))
        if missing:
            log.critical(
                "Daily parquet missing required columns",
                path=str(path),
                missing_columns=missing,
                available_columns=sorted(list(df.columns)),
            )
            return df, None

        # Some data includes 'timestamp' or index datetime; not required for TR%.
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        prev_close = close.shift(1)
        # True Range:
        # tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        hl = (high - low).abs()
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

        # Convert to percent of prev close (decimal)
        denom = prev_close.copy()
        # For first row or missing prev_close, use current close
        denom = denom.where(denom.notna() & (denom != 0), close.where(close.notna() & (close != 0), np.nan))
        tr_pct = tr / denom

        tr_pct = pd.to_numeric(tr_pct, errors="coerce")
        tr_pct = tr_pct.replace([np.inf, -np.inf], np.nan).dropna()

        # Basic sanity clipping: TR% should be within [0, 0.2] typically for index (0-20%)
        # We don't hard reject; we clip outliers to keep model stable.
        tr_pct = tr_pct.clip(lower=0.0, upper=0.20)

        return df, tr_pct

    @staticmethod
    def _compute_median_volatility(tr_pct: pd.Series) -> float:
        s = tr_pct.dropna().astype(float)
        s = s[np.isfinite(s)]
        if s.empty:
            return 0.0
        if s.shape[0] > 252:
            s = s.iloc[-252:]
        med = float(np.median(s.values))
        return float(max(0.0, med))

    # ---------------------------------------------------------------------
    # Internals: fitting
    # ---------------------------------------------------------------------
    def _fit_garch(self, train: pd.Series, lookback: int) -> bool:
        """
        Attempt to fit GARCH(1,1) with warning-to-failure behavior.
        Returns True if model fitted and stored.
        """
        try:
            # Lazy import
            from arch import arch_model  # type: ignore

            y = (train.astype(float).values * 100.0)  # scaling by 100
            y = y[np.isfinite(y)]
            if y.size < 30:
                return False

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                am = arch_model(y, vol="Garch", p=1, q=1, rescale=False)
                res = am.fit(disp="off")

                # If any warnings during fit, treat as failure per spec.
                if w:
                    # Some warnings are harmless; per requirement we fallback when warning happens.
                    log.warning(
                        "GARCH fit emitted warnings; treating as failure",
                        warnings=[str(x.message) for x in w][:5],
                    )
                    return False

            self._garch_result = res
            self._use_garch = True
            self._fitted = True
            self._ewma_last = None

            self._artifacts = _FitArtifacts(
                model_type="GARCH",
                lookback_days=int(lookback),
                points_used=int(len(train)),
                fitted_at=datetime.utcnow(),
                last_train_value=float(train.iloc[-1]) if len(train) > 0 else None,
                ewma_span=None,
            )

            log.info(
                "GARCH model fitted",
                lookback_days=lookback,
                points_used=len(train),
                median_vol=self._median_vol,
            )
            return True

        except Exception as e:
            log.warning(
                "GARCH fit failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return False

    def _fit_ewma(self, train: pd.Series, lookback: int, span: int = 20) -> None:
        s = train.dropna().astype(float)
        s = s[np.isfinite(s)]
        if s.empty:
            self._reset_fit_state(model_type="NONE", lookback_days=lookback)
            return

        ewma = s.ewm(span=int(span), adjust=False).mean()
        last = float(ewma.iloc[-1])
        if not np.isfinite(last):
            self._reset_fit_state(model_type="NONE", lookback_days=lookback)
            return

        self._garch_result = None
        self._use_garch = False
        self._fitted = True
        self._ewma_last = float(max(0.0, last))

        self._artifacts = _FitArtifacts(
            model_type="EWMA",
            lookback_days=int(lookback),
            points_used=int(len(s)),
            fitted_at=datetime.utcnow(),
            last_train_value=float(s.iloc[-1]),
            ewma_span=int(span),
        )

        log.info(
            "EWMA fallback fitted",
            span=span,
            lookback_days=lookback,
            points_used=len(s),
            ewma_last=self._ewma_last,
            median_vol=self._median_vol,
        )

    def _reset_fit_state(self, model_type: str, lookback_days: int) -> None:
        self._fitted = False
        self._use_garch = False
        self._garch_result = None
        self._ewma_last = None
        self._last_forecast = None
        # keep _median_vol as-is (may be 0.0)
        self._artifacts = _FitArtifacts(
            model_type=str(model_type),
            lookback_days=int(lookback_days),
            points_used=0,
            fitted_at=None,
            last_train_value=None,
            ewma_span=None,
        )


# -------------------------------------------------------------------------
# Self-test (MANDATORY)
# -------------------------------------------------------------------------
def _run_tests() -> None:
    import tempfile

    print("Running GARCHForecaster self-tests...")

    # Create synthetic daily candles parquet
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "NIFTY_daily.parquet"

        n = 260  # > 252 for median logic
        rng = np.random.default_rng(7)

        # Synthetic close series around 24500 with mild daily noise
        close = 24500.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=n))
        # Create realistic intraday ranges so TR% is ~0.8% to 2.5%
        base_range = np.clip(rng.normal(0.012, 0.004, size=n), 0.004, 0.05)  # fraction of price
        high = close * (1.0 + base_range / 2.0)
        low = close * (1.0 - base_range / 2.0)
        open_ = close * (1.0 + rng.normal(0.0, 0.0008, size=n))
        vol = rng.integers(1_000_000, 5_000_000, size=n)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=n, freq="B"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )

        df.to_parquet(p, index=False)

        f = GARCHForecaster(data_path=str(p))
        ok = f.fit(lookback_days=60)
        assert ok is True, "fit() should succeed via GARCH or EWMA"

        vol_fc = f.forecast(horizon=1)
        assert vol_fc is not None, "forecast() should return a float when fitted"
        assert 0.005 <= float(vol_fc) <= 0.05, f"forecast out of expected range: {vol_fc}"

        med = f.get_median_volatility()
        assert 0.002 <= float(med) <= 0.05, f"median out of expected range: {med}"

        hv = f.is_high_vol_regime()
        assert isinstance(hv, bool)

        status = f.get_status()
        assert isinstance(status, dict)
        assert status["fitted"] is True
        assert status["model_type"] in {"GARCH", "EWMA"}

        print("Status:", status)
        print("GARCHForecaster self-tests PASSED.")


if __name__ == "__main__":
    _run_tests()