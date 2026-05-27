"""
Junior Aladdin — XGBoost Regime Classifier Backup
=================================================

PURPOSE
- Train an independent XGBoost classifier to predict market regime:
  TRENDING / RANGE / VOLATILE / CHOP / EVENT
- Runtime: provide predicted regime + confidence (max class probability).
- Integration (Captain): if XGBoost disagrees with rule-based regime -> size *= 0.7

FEATURES
- Uses the same 85-feature vector built by LightGBMFilter.build_feature_vector(opportunity, context)
  for runtime inference.
- For training, accepts a feature matrix (n,85) and labels; also supports fitting from parquet.

FALLBACK
- If model not loaded/fitted OR feature vector cannot be built:
  return rule-based regime with confidence=0.0.

SELF-TEST
- python -m src.ml.regime_classifier_backup
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import setup_logger
from src.ml.lightgbm_filter import LightGBMFilter

log = setup_logger("regime_classifier_backup")


REGIMES: Tuple[str, ...] = ("TRENDING", "RANGE", "VOLATILE", "CHOP", "EVENT")
REGIME_TO_ID: Dict[str, int] = {r: i for i, r in enumerate(REGIMES)}
ID_TO_REGIME: Dict[int, str] = {i: r for r, i in REGIME_TO_ID.items()}


@dataclass(frozen=True)
class RegimePrediction:
    predicted_regime: str
    confidence: float  # 0..1
    rule_regime: str
    agrees_with_rule: bool
    size_multiplier: float  # 1.0 if agree else 0.7 (per spec)
    reason: str


class RegimeClassifierBackup:
    """
    XGBoost regime classifier backup.

    Methods:
      - fit(feature_matrix, labels) -> bool
      - fit_from_parquet(path=None) -> bool
      - predict_from_vector(x) -> Optional[Tuple[str, float]]
      - predict(opportunity, context, rule_regime) -> RegimePrediction
      - save_model(path) -> bool
      - load_model(path) -> bool
      - get_status() -> Dict[str, Any]
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_dir = str(Config.get("ml", "model_dir", default="models"))
        self._model_name = str(Config.get("ml", "xgb_regime_model_name", default="xgb_regime_backup.joblib"))
        self._default_model_path = str(Path(self._model_dir) / self._model_name)

        self._training_parquet = str(
            Config.get(
                "ml",
                "regime_training_path",
                default="data/historical/features/daily_features.parquet",
            )
        )

        # Per spec: mismatch -> size *= 0.7
        self._mismatch_multiplier = float(Config.get("risk", "xgb_regime_mismatch_multiplier", default=0.7))

        self._model = None  # xgboost.XGBClassifier
        self._fitted: bool = False
        self._points_used: int = 0
        self._last_pred: Optional[RegimePrediction] = None

        self._feature_builder = LightGBMFilter()

        # Try loading persisted model
        path = str(model_path or self._default_model_path)
        if Path(path).exists():
            ok = self.load_model(path)
            if ok:
                log.info("Loaded XGBoost regime model", path=path)
            else:
                log.warning("Failed to load XGBoost regime model; starting unfitted", path=path)
        else:
            log.info("No XGBoost regime model found; starting unfitted", expected_path=path)

    # ---------------------------------------------------------------------
    # Training / fitting
    # ---------------------------------------------------------------------
    def fit(self, feature_matrix: np.ndarray, labels: Sequence[Any]) -> bool:
        """
        Fit XGBoost multi-class classifier.

        feature_matrix: shape (n,85)
        labels: sequence of regime strings or integer class ids
        """
        try:
            X = self._ensure_2d_float(feature_matrix, expected_dim=85)
            y = self._encode_labels(labels)

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X rows != y rows ({X.shape[0]} != {y.shape[0]})")
            if X.shape[0] < 100:
                log.warning("Low sample count for regime model; may be unstable", samples=int(X.shape[0]))

            # Lazy import per production best practice
            import xgboost as xgb  # type: ignore

            params = {
                "n_estimators": int(Config.get("ml", "xgb_regime_n_estimators", default=400)),
                "max_depth": int(Config.get("ml", "xgb_regime_max_depth", default=4)),
                "learning_rate": float(Config.get("ml", "xgb_regime_learning_rate", default=0.05)),
                "subsample": float(Config.get("ml", "xgb_regime_subsample", default=0.9)),
                "colsample_bytree": float(Config.get("ml", "xgb_regime_colsample_bytree", default=0.9)),
                "reg_lambda": float(Config.get("ml", "xgb_regime_reg_lambda", default=1.0)),
                "min_child_weight": float(Config.get("ml", "xgb_regime_min_child_weight", default=1.0)),
                "gamma": float(Config.get("ml", "xgb_regime_gamma", default=0.0)),
                "objective": "multi:softprob",
                "num_class": len(REGIMES),
                "eval_metric": "mlogloss",
                "random_state": int(Config.get("ml", "xgb_regime_random_state", default=42)),
                "n_jobs": int(Config.get("ml", "xgb_regime_n_jobs", default=-1)),
                # keep verbosity off
                "verbosity": 0,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X, y)

            self._model = model
            self._fitted = True
            self._points_used = int(X.shape[0])

            log.info(
                "XGBoost regime model fitted",
                samples=int(X.shape[0]),
                classes=len(REGIMES),
            )
            return True

        except Exception as e:
            log.critical(
                "RegimeClassifierBackup.fit failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._model = None
            self._fitted = False
            self._points_used = 0
            return False

    def fit_from_parquet(self, path: Optional[str] = None) -> bool:
        """
        Fit from historical parquet. Expected:
          - label column: 'regime' or 'regime_label' or 'label'
          - features: either f0..f84 OR >=85 numeric columns (excluding label)
        """
        p = Path(path or self._training_parquet)
        if not p.exists():
            log.warning("Regime training parquet missing; cannot fit", path=str(p))
            return False

        try:
            df = pd.read_parquet(p)
            X, y = self._extract_xy_from_parquet(df)
            if X is None or y is None:
                log.warning("Could not extract (X,y) from parquet; cannot fit", path=str(p))
                return False
            return self.fit(X, y)
        except Exception as e:
            log.critical(
                "fit_from_parquet failed",
                path=str(p),
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return False

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict_from_vector(self, x: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Predict regime from a ready feature vector of shape (1,85) or (n,85).
        Returns (predicted_regime, confidence_of_predicted_class).
        """
        if not self._fitted or self._model is None:
            return None

        try:
            X = self._ensure_2d_float(x, expected_dim=85)
            proba = self._model.predict_proba(X)
            proba = np.asarray(proba, dtype=np.float64)
            if proba.ndim != 2 or proba.shape[1] != len(REGIMES):
                raise RuntimeError(f"Unexpected proba shape: {proba.shape}")

            p = proba[0]
            if not np.all(np.isfinite(p)):
                raise RuntimeError("Non-finite probabilities")
            cls_id = int(np.argmax(p))
            conf = float(p[cls_id])

            regime = ID_TO_REGIME.get(cls_id, "UNKNOWN")
            conf = float(max(0.0, min(1.0, conf)))
            return regime, conf

        except Exception as e:
            log.error(
                "predict_from_vector failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return None

    def predict(self, opportunity: Dict[str, Any], context: Dict[str, Any], rule_regime: str) -> RegimePrediction:
        """
        Runtime prediction using LightGBMFilter.build_feature_vector for the same 85 features.
        If model unavailable/vector build fails -> return rule regime with confidence=0.0.
        """
        rr = str(rule_regime or "UNKNOWN").upper().strip()
        if rr not in REGIME_TO_ID:
            rr = "UNKNOWN"

        # Safe fallback if not fitted
        if not self._fitted or self._model is None:
            pred = RegimePrediction(
                predicted_regime=rr,
                confidence=0.0,
                rule_regime=rr,
                agrees_with_rule=True,
                size_multiplier=1.0,
                reason="model_not_loaded",
            )
            self._last_pred = pred
            return pred

        try:
            x = self._feature_builder.build_feature_vector(opportunity, context)
            if x is None:
                pred = RegimePrediction(
                    predicted_regime=rr,
                    confidence=0.0,
                    rule_regime=rr,
                    agrees_with_rule=True,
                    size_multiplier=1.0,
                    reason="feature_vector_unavailable",
                )
                self._last_pred = pred
                return pred

            out = self.predict_from_vector(x)
            if out is None:
                pred = RegimePrediction(
                    predicted_regime=rr,
                    confidence=0.0,
                    rule_regime=rr,
                    agrees_with_rule=True,
                    size_multiplier=1.0,
                    reason="prediction_failed",
                )
                self._last_pred = pred
                return pred

            predicted_regime, confidence = out
            agrees = (rr != "UNKNOWN") and (predicted_regime == rr)
            size_mult = 1.0 if agrees or rr == "UNKNOWN" else float(self._mismatch_multiplier)

            pred = RegimePrediction(
                predicted_regime=predicted_regime,
                confidence=float(confidence),
                rule_regime=rr,
                agrees_with_rule=bool(agrees) if rr != "UNKNOWN" else True,
                size_multiplier=size_mult,
                reason="ok" if rr == "UNKNOWN" else ("agree" if agrees else "mismatch"),
            )
            self._last_pred = pred

            log.info(
                "Regime backup prediction",
                predicted_regime=pred.predicted_regime,
                confidence=pred.confidence,
                rule_regime=pred.rule_regime,
                agrees=pred.agrees_with_rule,
                size_multiplier=pred.size_multiplier,
                reason=pred.reason,
            )
            return pred

        except Exception as e:
            log.error(
                "RegimeClassifierBackup.predict failed; returning rule regime",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            pred = RegimePrediction(
                predicted_regime=rr,
                confidence=0.0,
                rule_regime=rr,
                agrees_with_rule=True,
                size_multiplier=1.0,
                reason="exception_fallback",
            )
            self._last_pred = pred
            return pred

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def save_model(self, path: Optional[str] = None) -> bool:
        """
        Save model via joblib.
        """
        p = Path(path or self._default_model_path)
        try:
            if not self._fitted or self._model is None:
                log.warning("save_model called but model not fitted; skipping", path=str(p))
                return False
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self._model,
                "regimes": list(REGIMES),
                "points_used": int(self._points_used),
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }
            joblib.dump(payload, p)
            log.info("Saved regime backup model", path=str(p))
            return True
        except Exception as e:
            log.error("Failed to save regime backup model", path=str(p), error=str(e), traceback=traceback.format_exc())
            return False

    def load_model(self, path: str) -> bool:
        """
        Load model via joblib.
        """
        p = Path(path)
        try:
            if not p.exists():
                return False
            payload = joblib.load(p)

            if isinstance(payload, dict) and "model" in payload:
                self._model = payload["model"]
                self._fitted = True
                self._points_used = int(payload.get("points_used", 0) or 0)
                return True

            # Backward compatibility: model stored directly
            self._model = payload
            self._fitted = True
            self._points_used = 0
            return True

        except Exception as e:
            log.error("Failed to load regime backup model", path=str(p), error=str(e), traceback=traceback.format_exc())
            self._model = None
            self._fitted = False
            self._points_used = 0
            return False

    # ---------------------------------------------------------------------
    # Status / helpers
    # ---------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            "fitted": bool(self._fitted),
            "model_path_default": self._default_model_path,
            "training_parquet": self._training_parquet,
            "points_used": int(self._points_used),
            "mismatch_multiplier": float(self._mismatch_multiplier),
            "last_prediction": self._last_pred.__dict__ if self._last_pred is not None else None,
        }

    @staticmethod
    def _ensure_2d_float(x: np.ndarray, expected_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(f"Expected shape (n,{expected_dim}), got {arr.shape}")
        arr[~np.isfinite(arr)] = 0.0
        return arr

    @staticmethod
    def _encode_labels(labels: Sequence[Any]) -> np.ndarray:
        y_out: List[int] = []
        for v in labels:
            if isinstance(v, (int, np.integer)) and int(v) in ID_TO_REGIME:
                y_out.append(int(v))
                continue
            s = str(v).upper().strip()
            if s not in REGIME_TO_ID:
                raise ValueError(f"Unknown regime label in training data: {v!r}")
            y_out.append(REGIME_TO_ID[s])
        return np.asarray(y_out, dtype=np.int32)

    @staticmethod
    def _extract_xy_from_parquet(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if df is None or df.empty:
            return None, None

        cols = [str(c) for c in df.columns]
        lower_map = {str(c).lower(): str(c) for c in cols}

        label_col = None
        for cand in ("regime", "regime_label", "label"):
            if cand in lower_map:
                label_col = lower_map[cand]
                break
        if label_col is None:
            log.warning("No label column found in training parquet; expected one of: regime/regime_label/label")
            return None, None

        # Feature columns:
        fcols = [f"f{i}" for i in range(85)]
        if all(c in df.columns for c in fcols):
            X = df[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        else:
            # Use numeric columns excluding label col
            numeric_df = df.apply(pd.to_numeric, errors="coerce")
            num_cols = [c for c in numeric_df.columns if c != label_col and pd.api.types.is_numeric_dtype(numeric_df[c])]
            if len(num_cols) < 85:
                log.warning("Not enough numeric columns for 85 features", numeric_cols=len(num_cols))
                return None, None
            use = num_cols[:85]
            X = numeric_df[use].to_numpy(dtype=np.float32)

        y_raw = df[label_col].tolist()
        y = RegimeClassifierBackup._encode_labels(y_raw)
        return X, y


# -------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------
def _run_tests() -> None:
    print("Running RegimeClassifierBackup self-tests...")

    rng = np.random.default_rng(123)
    n = 2000
    X = rng.normal(0.0, 1.0, size=(n, 85)).astype(np.float32)

    # Create synthetic regimes by simple rules on a few features
    # (not meant to be realistic; only for sanity)
    y = np.zeros(n, dtype=np.int32)
    y[(X[:, 0] > 1.0) & (X[:, 1] > 0.5)] = REGIME_TO_ID["TRENDING"]
    y[(X[:, 2] < -1.0) & (np.abs(X[:, 3]) < 0.2)] = REGIME_TO_ID["RANGE"]
    y[(np.abs(X[:, 4]) > 2.0)] = REGIME_TO_ID["VOLATILE"]
    y[(np.abs(X[:, 5]) < 0.1) & (np.abs(X[:, 6]) < 0.1)] = REGIME_TO_ID["CHOP"]
    y[(X[:, 7] > 2.5)] = REGIME_TO_ID["EVENT"]

    clf = RegimeClassifierBackup(model_path="__nonexistent__.joblib")
    ok = clf.fit(X, y)
    assert ok is True, "fit() should succeed on synthetic data"

    # Predict on one sample
    x1 = X[0].reshape(1, -1)
    pred = clf.predict_from_vector(x1)
    assert pred is not None
    regime, conf = pred
    assert regime in REGIMES
    assert 0.0 <= conf <= 1.0

    # Fallback path (rule regime) should not crash even if LightGBM schema missing
    rp = clf.predict(opportunity={"direction": "LONG"}, context={}, rule_regime="RANGE")
    assert isinstance(rp, RegimePrediction)

    print("predict_from_vector:", pred)
    print("fallback predict:", rp)
    print("status:", clf.get_status())
    print("RegimeClassifierBackup self-tests PASSED.")


if __name__ == "__main__":
    _run_tests()