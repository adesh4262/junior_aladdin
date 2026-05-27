"""
Junior Aladdin — Isolation Forest Anomaly Detector (Production Grade)
====================================================================

PURPOSE
- Detect anomalous 85-feature vectors indicating data corruption, feed issues,
  or regime breaks.
- Output anomaly_score in [0,1] where higher => more anomalous.

Captain actions (handled elsewhere):
- score > anomaly_pause_threshold (default 0.70) -> recommend PAUSE
- score > anomaly_safe_threshold  (default 0.90) -> recommend SAFE

MODEL
- sklearn.ensemble.IsolationForest

FEATURES
- Reuses the exact same 85-feature vector built by:
    LightGBMFilter.build_feature_vector(opportunity, context)

FALLBACK
- If model not fitted / any failure -> return NORMAL with anomaly_score=0.0
  (never block trades due to detector failure).

SELF-TEST
- python -m src.ml.anomaly_detector
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

from src.utils.logger import setup_logger
from src.utils.config_loader import Config
from src.ml.lightgbm_filter import LightGBMFilter

log = setup_logger("anomaly_detector")


@dataclass(frozen=True)
class AnomalyDecision:
    anomaly_score: float
    action: str  # "NORMAL" | "PAUSE" | "SAFE"
    pause_recommended: bool
    safe_recommended: bool
    reason: str


class AnomalyDetector:
    """
    Production-grade Isolation Forest anomaly detector.

    Methods:
      - fit(feature_matrix, contamination=0.05) -> bool
      - predict(features) -> Optional[float]  (returns anomaly_score [0,1])
      - evaluate(opportunity, context) -> AnomalyDecision
      - save_model(path) -> bool
      - load_model(path) -> bool
      - get_status() -> dict
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_dir = str(Config.get("ml", "model_dir", default="models"))
        self._default_model_name = str(Config.get("ml", "anomaly_model_name", default="isolation_forest_anomaly.joblib"))
        self._default_model_path = str(Path(self._model_dir) / self._default_model_name)

        self._pause_thr = float(Config.get("ml", "anomaly_pause_threshold", default=0.70))
        self._safe_thr = float(Config.get("ml", "anomaly_safe_threshold", default=0.90))
        self._contamination_cfg = float(Config.get("ml", "anomaly_contamination", default=0.05))

        self._historical_feature_path = str(
            Config.get(
                "ml",
                "historical_feature_matrix_path",
                default="data/historical/features/daily_features.parquet",
            )
        )

        self._model: Optional[IsolationForest] = None
        self._fitted: bool = False

        # For 0-1 scaling of decision_function output
        self._train_decision_min: Optional[float] = None
        self._train_decision_max: Optional[float] = None

        self._points_used: int = 0
        self._last_score: Optional[float] = None
        self._last_reason: str = "not_initialized"

        # Feature builder (shared with LightGBM filter)
        self._feature_builder = LightGBMFilter()

        # Attempt to load model if available
        load_path = str(model_path or self._default_model_path)
        if Path(load_path).exists():
            ok = self.load_model(load_path)
            if ok:
                log.info("AnomalyDetector loaded persisted model", path=load_path)
            else:
                log.warning("AnomalyDetector failed to load persisted model; starting unfitted", path=load_path)
        else:
            log.info("AnomalyDetector: no persisted model found; starting unfitted", expected_path=load_path)

    # ---------------------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------------------
    def fit(self, feature_matrix: np.ndarray, contamination: float = 0.05) -> bool:
        """
        Fit Isolation Forest on provided feature matrix.

        feature_matrix: shape (n_samples, 85)
        contamination: expected anomaly fraction (default from config if invalid)
        """
        try:
            X = self._ensure_2d_float(feature_matrix, expected_dim=85)
            X = self._clean_matrix(X)
            if X.shape[0] < 50:
                log.warning("AnomalyDetector.fit: low sample count; model may be unstable", samples=int(X.shape[0]))

            cont = float(contamination) if contamination is not None else float(self._contamination_cfg)
            if (not np.isfinite(cont)) or cont <= 0 or cont >= 0.5:
                cont = float(self._contamination_cfg)

            self._model = IsolationForest(
                n_estimators=int(Config.get("ml", "anomaly_n_estimators", default=300)),
                contamination=cont,
                random_state=int(Config.get("ml", "anomaly_random_state", default=42)),
                bootstrap=bool(Config.get("ml", "anomaly_bootstrap", default=False)),
                n_jobs=int(Config.get("ml", "anomaly_n_jobs", default=-1)),
            )
            self._model.fit(X)

            # Compute decision_function range on training for score normalization
            decision_scores = self._model.decision_function(X)  # higher => more normal
            decision_scores = np.asarray(decision_scores, dtype=np.float64)
            finite = decision_scores[np.isfinite(decision_scores)]
            if finite.size == 0:
                raise RuntimeError("decision_function produced no finite values")

            self._train_decision_min = float(np.min(finite))
            self._train_decision_max = float(np.max(finite))
            self._fitted = True
            self._points_used = int(X.shape[0])
            self._last_reason = "fitted_ok"

            log.info(
                "AnomalyDetector fitted",
                samples=int(X.shape[0]),
                contamination=float(cont),
                decision_min=self._train_decision_min,
                decision_max=self._train_decision_max,
            )
            return True

        except Exception as e:
            log.critical(
                "AnomalyDetector.fit failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._model = None
            self._fitted = False
            self._train_decision_min = None
            self._train_decision_max = None
            self._points_used = 0
            self._last_reason = "fit_failed"
            return False

    def fit_from_parquet(self) -> bool:
        """
        Fit the model from historical parquet at:
          data/historical/features/daily_features.parquet (default)

        If missing/unreadable, fits on a synthetic baseline (fallback).
        """
        p = Path(self._historical_feature_path)
        if not p.exists():
            log.warning("Historical feature parquet missing; fitting synthetic baseline", path=str(p))
            X = self._synthetic_baseline(n=1200, d=85)
            return self.fit(X, contamination=self._contamination_cfg)

        try:
            df = pd.read_parquet(p)
            X = self._extract_matrix_from_parquet(df)
            if X is None:
                log.warning("Could not extract feature matrix from parquet; fitting synthetic baseline", path=str(p))
                X = self._synthetic_baseline(n=1200, d=85)
            return self.fit(X, contamination=self._contamination_cfg)
        except Exception as e:
            log.warning(
                "Failed to read/parse historical feature parquet; fitting synthetic baseline",
                path=str(p),
                error=str(e),
            )
            X = self._synthetic_baseline(n=1200, d=85)
            return self.fit(X, contamination=self._contamination_cfg)

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Returns anomaly_score in [0,1] using decision_function normalization.

        score = 1 - (decision_score - min) / (max - min)
        where decision_score higher => more normal.
        """
        if not self._fitted or self._model is None:
            self._last_reason = "model_not_fitted"
            return None

        try:
            x = self._ensure_2d_float(features, expected_dim=85)
            x = self._clean_matrix(x)

            decision_score = self._model.decision_function(x)  # array shape (1,)
            decision_score = float(np.asarray(decision_score, dtype=np.float64).reshape(-1)[0])

            if not np.isfinite(decision_score):
                raise RuntimeError("Non-finite decision_function output")

            dmin = self._train_decision_min
            dmax = self._train_decision_max
            if dmin is None or dmax is None or (not np.isfinite(dmin)) or (not np.isfinite(dmax)) or dmax <= dmin:
                # Can't normalize; safe default
                self._last_reason = "normalization_unavailable"
                score = 0.0
            else:
                score = 1.0 - (decision_score - dmin) / (dmax - dmin)

            score = float(np.clip(score, 0.0, 1.0))
            self._last_score = score
            self._last_reason = "predict_ok"
            return score

        except Exception as e:
            log.error(
                "AnomalyDetector.predict failed; returning None",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._last_reason = "predict_failed"
            return None

    def evaluate(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> AnomalyDecision:
        """
        Builds 85-feature vector using LightGBMFilter and returns decision.
        Graceful fallback if model/vector unavailable.
        """
        try:
            x = self._feature_builder.build_feature_vector(opportunity, context)
            if x is None:
                return AnomalyDecision(
                    anomaly_score=0.0,
                    action="NORMAL",
                    pause_recommended=False,
                    safe_recommended=False,
                    reason="feature_vector_unavailable",
                )

            score = self.predict(x)
            if score is None:
                return AnomalyDecision(
                    anomaly_score=0.0,
                    action="NORMAL",
                    pause_recommended=False,
                    safe_recommended=False,
                    reason="model_not_fitted",
                )

            pause_thr = float(self._pause_thr)
            safe_thr = float(self._safe_thr)
            if (not np.isfinite(pause_thr)) or pause_thr <= 0 or pause_thr >= 1:
                pause_thr = 0.70
            if (not np.isfinite(safe_thr)) or safe_thr <= 0 or safe_thr >= 1:
                safe_thr = 0.90
            if safe_thr <= pause_thr:
                safe_thr = max(pause_thr + 0.05, 0.90)

            if score > safe_thr:
                return AnomalyDecision(
                    anomaly_score=float(score),
                    action="SAFE",
                    pause_recommended=True,
                    safe_recommended=True,
                    reason="anomaly_score_above_safe_threshold",
                )

            if score > pause_thr:
                return AnomalyDecision(
                    anomaly_score=float(score),
                    action="PAUSE",
                    pause_recommended=True,
                    safe_recommended=False,
                    reason="anomaly_score_above_pause_threshold",
                )

            return AnomalyDecision(
                anomaly_score=float(score),
                action="NORMAL",
                pause_recommended=False,
                safe_recommended=False,
                reason="normal",
            )

        except Exception as e:
            log.error(
                "AnomalyDetector.evaluate failed; returning NORMAL",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return AnomalyDecision(
                anomaly_score=0.0,
                action="NORMAL",
                pause_recommended=False,
                safe_recommended=False,
                reason="evaluate_failed",
            )

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save_model(self, path: str) -> bool:
        try:
            if not self._fitted or self._model is None:
                log.warning("save_model called but model not fitted; skipping", path=path)
                return False

            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "model": self._model,
                "train_decision_min": self._train_decision_min,
                "train_decision_max": self._train_decision_max,
                "points_used": self._points_used,
                "pause_threshold": self._pause_thr,
                "safe_threshold": self._safe_thr,
                "contamination": self._contamination_cfg,
                "created_at": datetime.utcnow().isoformat(),
                "feature_dim": 85,
            }
            joblib.dump(payload, p)
            log.info("AnomalyDetector model saved", path=str(p))
            return True
        except Exception as e:
            log.error("Failed to save anomaly model", path=path, error=str(e), traceback=traceback.format_exc())
            return False

    def load_model(self, path: str) -> bool:
        try:
            p = Path(path)
            if not p.exists():
                log.warning("load_model: file missing", path=str(p))
                return False

            payload = joblib.load(p)
            if isinstance(payload, dict) and "model" in payload:
                model = payload.get("model")
                if not isinstance(model, IsolationForest):
                    log.warning("Loaded anomaly model is not IsolationForest", path=str(p), model_type=str(type(model)))
                    return False

                self._model = model
                self._train_decision_min = payload.get("train_decision_min")
                self._train_decision_max = payload.get("train_decision_max")
                self._points_used = int(payload.get("points_used", 0) or 0)
                self._fitted = True
                self._last_reason = "loaded_ok"

                log.info(
                    "AnomalyDetector model loaded",
                    path=str(p),
                    points_used=self._points_used,
                    decision_min=self._train_decision_min,
                    decision_max=self._train_decision_max,
                )
                return True

            # Backward compatibility: plain model saved
            if isinstance(payload, IsolationForest):
                self._model = payload
                self._train_decision_min = None
                self._train_decision_max = None
                self._points_used = 0
                self._fitted = True
                self._last_reason = "loaded_plain_model_ok"
                log.warning("Loaded plain IsolationForest without normalization range; scores may be conservative", path=str(p))
                return True

            log.warning("load_model: unrecognized payload", path=str(p), payload_type=str(type(payload)))
            return False

        except Exception as e:
            log.error("Failed to load anomaly model", path=path, error=str(e), traceback=traceback.format_exc())
            self._model = None
            self._fitted = False
            self._train_decision_min = None
            self._train_decision_max = None
            self._points_used = 0
            self._last_reason = "load_failed"
            return False

    # ---------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted": bool(self._fitted),
            "model_type": "IsolationForest" if self._model is not None else "NONE",
            "model_path_default": self._default_model_path,
            "historical_feature_path": self._historical_feature_path,
            "contamination_cfg": float(self._contamination_cfg),
            "pause_threshold": float(self._pause_thr),
            "safe_threshold": float(self._safe_thr),
            "points_used": int(self._points_used),
            "train_decision_min": float(self._train_decision_min) if isinstance(self._train_decision_min, (int, float)) else None,
            "train_decision_max": float(self._train_decision_max) if isinstance(self._train_decision_max, (int, float)) else None,
            "last_score": float(self._last_score) if self._last_score is not None else None,
            "last_reason": str(self._last_reason),
        }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_2d_float(arr: np.ndarray, expected_dim: int) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != expected_dim:
            raise ValueError(f"Expected shape (n,{expected_dim}) got {x.shape}")
        return x

    @staticmethod
    def _clean_matrix(x: np.ndarray) -> np.ndarray:
        # Replace NaN/inf with column medians (robust) then zeros if still non-finite.
        X = np.asarray(x, dtype=np.float64)
        X = X.copy()

        # Column-wise median fill
        for j in range(X.shape[1]):
            col = X[:, j]
            finite = col[np.isfinite(col)]
            med = float(np.median(finite)) if finite.size > 0 else 0.0
            bad = ~np.isfinite(col)
            if np.any(bad):
                col[bad] = med
            X[:, j] = col

        # Final clamp for any remaining
        X[~np.isfinite(X)] = 0.0
        return X

    @staticmethod
    def _synthetic_baseline(n: int, d: int) -> np.ndarray:
        rng = np.random.default_rng(123)
        X = rng.normal(loc=0.0, scale=1.0, size=(int(n), int(d)))
        return X.astype(np.float64)

    @staticmethod
    def _extract_matrix_from_parquet(df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Attempts to extract an (n,85) numeric matrix from a parquet dataframe.
        Supported layouts (best-effort):
          1) numeric columns >= 85 -> take first 85 numeric cols
          2) columns named f0..f84 -> use them in order
        """
        if df is None or df.empty:
            return None

        df2 = df.copy()
        df2.columns = [str(c) for c in df2.columns]

        # Preferred: f0..f84
        fcols = [f"f{i}" for i in range(85)]
        if all(c in df2.columns for c in fcols):
            X = df2[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            return X

        # Otherwise: take first 85 numeric columns
        numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        if len(numeric_cols) >= 85:
            use = numeric_cols[:85]
            X = df2[use].to_numpy(dtype=np.float64)
            return X

        # If types are object but convertible, attempt conversion and then select numeric
        conv = df2.apply(pd.to_numeric, errors="coerce")
        numeric_cols2 = [c for c in conv.columns if pd.api.types.is_numeric_dtype(conv[c])]
        if len(numeric_cols2) >= 85:
            use = numeric_cols2[:85]
            X = conv[use].to_numpy(dtype=np.float64)
            return X

        return None


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
def _run_tests() -> None:
    print("Running AnomalyDetector self-tests...")

    det = AnomalyDetector(model_path=None)

    # Synthetic "normal" data
    rng = np.random.default_rng(7)
    X_train = rng.normal(0.0, 1.0, size=(1500, 85))
    ok = det.fit(X_train, contamination=0.05)
    assert ok is True, "fit() must succeed on synthetic data"

    # Normal sample -> low anomaly score
    x_norm = rng.normal(0.0, 1.0, size=(1, 85))
    s_norm = det.predict(x_norm)
    assert s_norm is not None
    assert 0.0 <= s_norm <= 1.0

    # Outlier sample -> high anomaly score (very large magnitude)
    x_out = rng.normal(0.0, 1.0, size=(1, 85))
    x_out[:, :10] += 12.0
    s_out = det.predict(x_out)
    assert s_out is not None
    assert 0.0 <= s_out <= 1.0
    assert s_out > s_norm, f"Expected outlier score > normal score, got out={s_out}, norm={s_norm}"

    # Evaluate path (without needing LightGBM feature schema); should degrade gracefully
    decision = det.evaluate(opportunity={"direction": "LONG"}, context={})
    assert isinstance(decision, AnomalyDecision)
    assert decision.action in {"NORMAL", "PAUSE", "SAFE"}

    print("normal_score:", s_norm)
    print("outlier_score:", s_out)
    print("status:", det.get_status())
    print("AnomalyDetector self-tests PASSED.")


if __name__ == "__main__":
    _run_tests()