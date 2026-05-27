"""
Junior Aladdin — LightGBM Quality Filter (Production Hardened)
=============================================================

This module gates scored opportunities using an ML probability (0..1).
It runs AFTER Scoring Engine and BEFORE Behavioral Sentinel.

Key capabilities:
- Builds an 85-feature vector in model-expected order (strict; missing => None).
- Loads model in priority:
    1) ONNX (onnxruntime)   2) LightGBM .txt (lightgbm.Booster)   3) Pickle (.pkl via joblib)
- Evaluates an opportunity -> MLFilterDecision (REJECT/CAUTION/PASS) using config thresholds.
- Optional SHAP/top-contrib integration when available.
- Safe fallback behavior on any error: CAUTION with reduce_size=True, probability=0.50.

IMPORTANT: Feature schema/order
The filter MUST know the exact feature order used during training.
This is resolved by, in priority:
  - LightGBM Booster feature_name() (when .txt is loaded)
  - feature schema JSON in models/ (recommended)
  - config-driven feature list (ml.feature_order)
If none are available, feature vector construction fails (capital-safe fallback used).

CaptainEngine context compatibility:
- Training/spec may expect:
    context["smart_money_5m"], context["smart_money_15m"]
  but Captain provides:
    context["smart_money"] = {"5min": {...}, "15min": {...}}
- Training/spec may expect:
    context["features_1m"], context["features_5m"], context["features_15m"]
  but Captain provides:
    context["per_tf"] = {"1min": {...}, "5min": {...}, "15min": {...}}
This module implements strict backward-compatible fallback extraction for both.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from src.utils.config_loader import Config
from src.utils.logger import setup_logger

logger = setup_logger("lightgbm_filter")

Action = Literal["REJECT", "CAUTION", "PASS"]


@dataclass(frozen=True)
class MLFilterDecision:
    probability: float
    action: Action
    reduce_size: bool
    shap_values: Optional[Dict[str, float]]
    rejection_reason: str


@dataclass(frozen=True)
class FeatureSpec:
    """
    Feature specification for strict, deterministic extraction.

    source:
      - "opportunity" to read from opportunity dict
      - any other value to read from context[source] dict
    path:
      - dotted path within the selected dict (e.g., "rsi_14" or "vwap.bands.sigma1")
    """

    name: str
    source: str
    path: str


class LightGBMFilter:
    """
    Production-hardened ML filter.

    Public API:
      - build_feature_vector(opportunity, context) -> Optional[np.ndarray]
      - evaluate(opportunity, context) -> MLFilterDecision
    """

    def __init__(self) -> None:
        self._model_dir = str(Config.get("ml", "model_dir", default="models"))
        self._model_name = str(Config.get("ml", "lightgbm_model_name", default="lightgbm_quality_filter"))

        self._reject_thr = float(Config.get("ml", "lightgbm_reject_threshold", default=0.50))
        self._caution_thr = float(Config.get("ml", "lightgbm_caution_threshold", default=0.65))
        self._caution_reduction = float(Config.get("ml", "lightgbm_caution_size_reduction", default=0.20))

        # Model handles (initialized once)
        self._onnx_session = None
        self._onnx_input_name: Optional[str] = None
        self._onnx_output_name: Optional[str] = None

        self._lgbm_booster = None  # lightgbm.Booster if loaded
        self._pickle_model = None  # sklearn/lightgbm wrapper if loaded

        # Feature schema/order
        self._feature_specs: Optional[List[FeatureSpec]] = None
        self._feature_order: Optional[List[str]] = None

        # Optional SHAP explainer loaded from disk
        self._shap_explainer = None

        self._load_model_and_schema()

    # ---------------------------------------------------------------------
    # Feature vector construction (MANDATORY)
    # ---------------------------------------------------------------------
    def build_feature_vector(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Build (1, 85) float32 vector in the exact order the model expects.

        If any feature is missing/non-numeric/non-finite -> log CRITICAL and return None.
        """
        try:
            if not isinstance(opportunity, dict) or not isinstance(context, dict):
                logger.critical(
                    "build_feature_vector: invalid inputs",
                    extra={"opportunity_type": str(type(opportunity)), "context_type": str(type(context))},
                )
                return None

            # Prefer strict FeatureSpec schema; else fall back to feature_order names.
            if self._feature_specs is not None:
                specs = self._feature_specs
                if len(specs) != 85:
                    logger.critical("build_feature_vector: feature_specs count != 85", extra={"count": len(specs)})
                    return None

                values: List[float] = []
                for spec in specs:
                    v = self._extract_feature_by_spec(spec, opportunity, context)
                    if v is None:
                        logger.critical(
                            "build_feature_vector: missing feature",
                            extra={"feature": spec.name, "source": spec.source, "path": spec.path},
                        )
                        return None
                    values.append(v)

                return np.asarray(values, dtype=np.float32).reshape(1, 85)

            if self._feature_order is not None:
                names = self._feature_order
                if len(names) != 85:
                    logger.critical("build_feature_vector: feature_order count != 85", extra={"count": len(names)})
                    return None

                values = []
                for name in names:
                    v = self._extract_feature_by_name(name, opportunity, context)
                    if v is None:
                        logger.critical("build_feature_vector: missing feature", extra={"feature": name})
                        return None
                    values.append(v)

                return np.asarray(values, dtype=np.float32).reshape(1, 85)

            logger.critical(
                "build_feature_vector: no feature schema/order available; cannot build 85-feature vector",
                extra={
                    "model_name": self._model_name,
                    "model_dir": self._model_dir,
                    "hint": "Provide models/<model>_feature_schema.json (recommended) or ml.feature_order in config.",
                },
            )
            return None

        except Exception as e:
            logger.critical("build_feature_vector: exception", extra={"error": str(e), "traceback": traceback.format_exc()})
            return None

    # ---------------------------------------------------------------------
    # Prediction pipeline (MANDATORY)
    # ---------------------------------------------------------------------
    def evaluate(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> MLFilterDecision:
        """
        Evaluate opportunity using ML probability and config thresholds.
        Safe on failures: CAUTION, reduce_size=True, probability=0.50.
        """
        try:
            x = self.build_feature_vector(opportunity, context)
            if x is None:
                return self._safe_decision(reason="feature_vector_missing_or_incomplete", probability=0.50)

            prob = self.predict_proba_from_vector(x)
            if prob is None:
                return self._safe_decision(reason="model_prediction_failed", probability=0.50)

            # Optional SHAP values: top 5 features only
            shap_values = self._compute_shap_top5(x)

            # Threshold logic (MANDATORY)
            if prob < self._reject_thr:
                decision = MLFilterDecision(
                    probability=float(prob),
                    action="REJECT",
                    reduce_size=False,
                    shap_values=shap_values,
                    rejection_reason="ml_probability_below_reject_threshold",
                )
            elif prob < self._caution_thr:
                decision = MLFilterDecision(
                    probability=float(prob),
                    action="CAUTION",
                    reduce_size=True,
                    shap_values=shap_values,
                    rejection_reason="ml_probability_in_caution_band",
                )
            else:
                decision = MLFilterDecision(
                    probability=float(prob),
                    action="PASS",
                    reduce_size=False,
                    shap_values=shap_values,
                    rejection_reason="pass",
                )

            logger.info(
                "LightGBMFilter decision",
                extra={
                    "probability": decision.probability,
                    "action": decision.action,
                    "reduce_size": decision.reduce_size,
                    "reject_threshold": self._reject_thr,
                    "caution_threshold": self._caution_thr,
                    "model_loaded": self._model_loaded_kind(),
                    "shap_available": bool(decision.shap_values),
                },
            )
            return decision

        except Exception as e:
            logger.critical(
                "evaluate: exception; returning safe CAUTION",
                extra={"error": str(e), "traceback": traceback.format_exc()},
            )
            return self._safe_decision(reason="exception", probability=0.50)

    def predict_proba_from_vector(self, x: np.ndarray) -> Optional[float]:
        """
        Predict probability using the best-loaded model.
        Returns None if prediction fails.
        """
        try:
            x = self._ensure_2d_float32(x, expected_dim=85)

            # 1) ONNX
            if self._onnx_session is not None and self._onnx_input_name is not None:
                outputs = self._onnx_session.run(
                    [self._onnx_output_name] if self._onnx_output_name else None,
                    {self._onnx_input_name: x},
                )
                return self._coerce_probability(outputs[0])

            # 2) LightGBM Booster
            if self._lgbm_booster is not None:
                y = self._lgbm_booster.predict(x)
                return self._coerce_probability(y)

            # 3) Pickle model (sklearn/lightgbm wrapper)
            if self._pickle_model is not None:
                model = self._pickle_model
                if hasattr(model, "predict_proba"):
                    y = model.predict_proba(x)
                    y = np.asarray(y)
                    if y.ndim == 2 and y.shape[1] >= 2:
                        return self._coerce_probability(y[:, 1])
                    return self._coerce_probability(y)
                if hasattr(model, "predict"):
                    y = model.predict(x)
                    return self._coerce_probability(y)

            logger.warning(
                "No model loaded; using constant fallback probability=0.50",
                extra={"model_dir": self._model_dir, "model_name": self._model_name},
            )
            return 0.50

        except Exception as e:
            logger.critical(
                "predict_proba_from_vector: prediction failed",
                extra={"error": str(e), "traceback": traceback.format_exc()},
            )
            return None

    # ---------------------------------------------------------------------
    # Model loading / schema loading
    # ---------------------------------------------------------------------
    def _load_model_and_schema(self) -> None:
        model_dir = Path(self._model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Try to load optional SHAP explainer (best-effort)
        self._try_load_shap_explainer(model_dir)

        onnx_path, txt_path, pkl_path = self._candidate_paths(model_dir, self._model_name)

        # 1) ONNX
        if onnx_path and onnx_path.exists():
            if self._try_load_onnx(onnx_path):
                logger.info("Loaded ONNX model", extra={"path": str(onnx_path)})
            else:
                logger.error("Failed to load ONNX model", extra={"path": str(onnx_path)})

        # 2) LightGBM .txt
        if self._onnx_session is None and txt_path and txt_path.exists():
            if self._try_load_lgbm_txt(txt_path):
                logger.info("Loaded LightGBM .txt model", extra={"path": str(txt_path)})
            else:
                logger.error("Failed to load LightGBM .txt model", extra={"path": str(txt_path)})

        # 3) Pickle
        if self._onnx_session is None and self._lgbm_booster is None and pkl_path and pkl_path.exists():
            if self._try_load_pickle(pkl_path):
                logger.info("Loaded pickled model", extra={"path": str(pkl_path)})
            else:
                logger.error("Failed to load pickled model", extra={"path": str(pkl_path)})

        # Feature schema/order loading (strict; needed to build 85-vector)
        self._load_feature_schema_or_order(model_dir)

        # If we loaded a Booster and still no schema, try feature_name() as order
        if self._feature_specs is None and self._feature_order is None and self._lgbm_booster is not None:
            try:
                names = list(self._lgbm_booster.feature_name())
                if len(names) == 85:
                    self._feature_order = names
                    logger.info("Using LightGBM booster feature_name() as feature_order", extra={"count": len(names)})
                else:
                    logger.critical(
                        "LightGBM booster feature_name() count != 85; cannot use as order",
                        extra={"count": len(names)},
                    )
            except Exception as e:
                logger.error(
                    "Failed to read booster feature_name()",
                    extra={"error": str(e), "traceback": traceback.format_exc()},
                )

        logger.info(
            "LightGBMFilter initialization complete",
            extra={
                "model_loaded": self._model_loaded_kind(),
                "schema_loaded": self._feature_specs is not None,
                "feature_order_loaded": self._feature_order is not None,
            },
        )

    def _load_feature_schema_or_order(self, model_dir: Path) -> None:
        # 1) models/<model>_feature_schema.json
        schema_path = model_dir / f"{Path(self._model_name).stem}_feature_schema.json"
        cfg_schema_file = Config.get("ml", "feature_schema_file", default=None)
        if cfg_schema_file:
            cfg_path = Path(str(cfg_schema_file))
            schema_path = cfg_path if cfg_path.is_absolute() else (Path(".") / cfg_path)

        if schema_path.exists():
            try:
                specs = self._parse_feature_schema(schema_path)
                if len(specs) != 85:
                    logger.critical(
                        "Feature schema does not contain exactly 85 features",
                        extra={"path": str(schema_path), "count": len(specs)},
                    )
                else:
                    self._feature_specs = specs
                    logger.info("Loaded feature schema", extra={"path": str(schema_path), "count": len(specs)})
                    return
            except Exception as e:
                logger.critical(
                    "Failed to load/parse feature schema",
                    extra={"path": str(schema_path), "error": str(e), "traceback": traceback.format_exc()},
                )

        # 2) config: ml.feature_order
        feature_order = Config.get("ml", "feature_order", default=None)
        if isinstance(feature_order, list) and all(isinstance(x, str) for x in feature_order):
            if len(feature_order) != 85:
                logger.critical("Config ml.feature_order must contain exactly 85 names", extra={"count": len(feature_order)})
            else:
                self._feature_order = list(feature_order)
                logger.info("Loaded feature_order from config", extra={"count": len(feature_order)})
                return

        logger.warning(
            "No feature schema/order found; ML filter will fallback to safe CAUTION until provided",
            extra={"expected_schema": str(schema_path)},
        )

    # ---------------------------------------------------------------------
    # SHAP integration (OPTIONAL)
    # ---------------------------------------------------------------------
    def _try_load_shap_explainer(self, model_dir: Path) -> None:
        shap_file = Config.get("ml", "lightgbm_shap_explainer_name", default=None)
        candidates: List[Path] = []
        if shap_file:
            p = Path(str(shap_file))
            candidates.append(p if p.is_absolute() else (model_dir / p))
        candidates.append(model_dir / f"{Path(self._model_name).stem}_shap_explainer.pkl")
        candidates.append(model_dir / "shap_explainer.pkl")

        for p in candidates:
            if not p.exists():
                continue
            try:
                from joblib import load as joblib_load  # lazy import

                self._shap_explainer = joblib_load(p)
                logger.info("Loaded SHAP explainer", extra={"path": str(p)})
                return
            except Exception as e:
                logger.warning("Failed to load SHAP explainer (continuing without it)", extra={"path": str(p), "error": str(e)})

    def _compute_shap_top5(self, x: np.ndarray) -> Optional[Dict[str, float]]:
        try:
            # Booster contribution path (fast; no shap needed)
            if self._lgbm_booster is not None:
                contrib = self._lgbm_booster.predict(x, pred_contrib=True)
                contrib = np.asarray(contrib, dtype=np.float64)
                if contrib.ndim != 2 or contrib.shape[0] != 1:
                    return None

                try:
                    feat_names = list(self._lgbm_booster.feature_name())
                except Exception:
                    feat_names = [f"f{i}" for i in range(contrib.shape[1] - 1)]

                n_feat = min(len(feat_names), contrib.shape[1] - 1)  # exclude base value
                vals = contrib[0, :n_feat]
                idx = np.argsort(np.abs(vals))[::-1][:5]
                return {feat_names[i]: float(vals[i]) for i in idx}

            # External SHAP explainer path
            if self._shap_explainer is not None:
                try:
                    import shap  # noqa: F401
                except Exception:
                    return None

                explainer = self._shap_explainer
                shap_vals = explainer.shap_values(x)  # type: ignore[attr-defined]
                shap_vals = np.asarray(shap_vals)
                if shap_vals.ndim == 3:
                    shap_vals = shap_vals[-1, 0, :]
                elif shap_vals.ndim == 2:
                    shap_vals = shap_vals[0, :]
                else:
                    return None

                names = self._feature_names_for_reporting()
                idx = np.argsort(np.abs(shap_vals))[::-1][:5]
                return {names[i]: float(shap_vals[i]) for i in idx}

            return None
        except Exception:
            logger.debug("SHAP computation failed", extra={"traceback": traceback.format_exc()})
            return None

    def _feature_names_for_reporting(self) -> List[str]:
        if self._feature_specs is not None:
            return [fs.name for fs in self._feature_specs]
        if self._feature_order is not None:
            return list(self._feature_order)
        if self._lgbm_booster is not None:
            try:
                return list(self._lgbm_booster.feature_name())
            except Exception:
                pass
        return [f"f{i}" for i in range(85)]

    # ---------------------------------------------------------------------
    # Feature extraction helpers (UPDATED FOR CAPTAIN CONTEXT)
    # ---------------------------------------------------------------------
    def _extract_feature_by_spec(self, spec: FeatureSpec, opportunity: Dict[str, Any], context: Dict[str, Any]) -> Optional[float]:
        """
        Extract by strict feature spec.

        REQUIRED UPDATES:
        - If spec.source is smart_money_5m / smart_money_15m but context doesn't have direct key,
          fallback to context["smart_money"]["5min"/"15min"].
        - (Extra hardening) If spec.source is features_1m/5m/15m but context doesn't have direct key,
          fallback to context["per_tf"]["1min"/"5min"/"15min"].
        """
        if spec.source == "opportunity":
            base = opportunity
            val = self._get_by_dotted_path(base, spec.path)
            return self._coerce_float(val)

        # smart_money fallbacks
        if spec.source == "smart_money_5m":
            base = context.get("smart_money_5m")
            if base is None:
                base = (context.get("smart_money") or {}).get("5min")
                if base is not None:
                    logger.debug(
                        "Extracted feature via fallback from smart_money['5min']",
                        extra={"feature": spec.name, "path": spec.path},
                    )
            val = self._get_by_dotted_path(base, spec.path)
            return self._coerce_float(val)

        if spec.source == "smart_money_15m":
            base = context.get("smart_money_15m")
            if base is None:
                base = (context.get("smart_money") or {}).get("15min")
                if base is not None:
                    logger.debug(
                        "Extracted feature via fallback from smart_money['15min']",
                        extra={"feature": spec.name, "path": spec.path},
                    )
            val = self._get_by_dotted_path(base, spec.path)
            return self._coerce_float(val)

        # per_tf fallbacks for timeframe features
        if spec.source in ("features_1m", "features_5m", "features_15m"):
            base = context.get(spec.source)
            if base is None:
                per_tf = context.get("per_tf") or {}
                tf_map = {"features_1m": "1min", "features_5m": "5min", "features_15m": "15min"}
                tf_key = tf_map[spec.source]
                base = per_tf.get(tf_key) if isinstance(per_tf, dict) else None
                if base is not None:
                    logger.debug(
                        "Extracted feature via fallback from per_tf",
                        extra={"feature": spec.name, "source": spec.source, "tf": tf_key, "path": spec.path},
                    )
            val = self._get_by_dotted_path(base, spec.path)
            return self._coerce_float(val)

        # Default: context dict at spec.source
        base = context.get(spec.source)
        val = self._get_by_dotted_path(base, spec.path)
        return self._coerce_float(val)

    def _extract_feature_by_name(self, name: str, opportunity: Dict[str, Any], context: Dict[str, Any]) -> Optional[float]:
        """
        Name-based extraction for config-only feature_order.

        REQUIRED UPDATES:
        - smart_money_5m / smart_money_15m can be provided via:
            context["smart_money_5m"]  OR  context["smart_money"]["5min"/"15min"]
          Supports both plain keys and dotted names like "smart_money_5m.sm_direction_score".
        - features_1m / features_5m / features_15m can be provided via:
            context["features_1m"] OR context["per_tf"]["1min"] (same for 5/15)
          Supports dotted names like "features_1m.rsi".
        """
        # 1) direct scalars
        if name in opportunity:
            return self._coerce_float(opportunity.get(name))
        if name in context:
            return self._coerce_float(context.get(name))

        # 2) Special dotted handling for smart_money_* (Captain compat)
        if name.startswith("smart_money_5m."):
            subpath = name[len("smart_money_5m.") :]
            base = context.get("smart_money_5m")
            if base is None:
                base = (context.get("smart_money") or {}).get("5min")
                if base is not None:
                    logger.debug("Extracted feature via fallback from smart_money['5min']", extra={"feature": name})
            val = self._get_by_dotted_path(base, subpath)
            return self._coerce_float(val)

        if name.startswith("smart_money_15m."):
            subpath = name[len("smart_money_15m.") :]
            base = context.get("smart_money_15m")
            if base is None:
                base = (context.get("smart_money") or {}).get("15min")
                if base is not None:
                    logger.debug("Extracted feature via fallback from smart_money['15min']", extra={"feature": name})
            val = self._get_by_dotted_path(base, subpath)
            return self._coerce_float(val)

        # 3) Special dotted handling for features_* (Captain per_tf compat)
        if name.startswith("features_1m."):
            subpath = name[len("features_1m.") :]
            base = context.get("features_1m")
            if base is None:
                per_tf = context.get("per_tf") or {}
                base = per_tf.get("1min") if isinstance(per_tf, dict) else None
                if base is not None:
                    logger.debug("Extracted feature via fallback from per_tf['1min']", extra={"feature": name})
            val = self._get_by_dotted_path(base, subpath)
            return self._coerce_float(val)

        if name.startswith("features_5m."):
            subpath = name[len("features_5m.") :]
            base = context.get("features_5m")
            if base is None:
                per_tf = context.get("per_tf") or {}
                base = per_tf.get("5min") if isinstance(per_tf, dict) else None
                if base is not None:
                    logger.debug("Extracted feature via fallback from per_tf['5min']", extra={"feature": name})
            val = self._get_by_dotted_path(base, subpath)
            return self._coerce_float(val)

        if name.startswith("features_15m."):
            subpath = name[len("features_15m.") :]
            base = context.get("features_15m")
            if base is None:
                per_tf = context.get("per_tf") or {}
                base = per_tf.get("15min") if isinstance(per_tf, dict) else None
                if base is not None:
                    logger.debug("Extracted feature via fallback from per_tf['15min']", extra={"feature": name})
            val = self._get_by_dotted_path(base, subpath)
            return self._coerce_float(val)

        # 4) generic dotted path in context
        if "." in name:
            val = self._get_by_dotted_path(context, name)
            if val is not None:
                return self._coerce_float(val)

        # 5) heuristic timeframe suffixes (backward compat)
        lower = name.lower()
        for tf_key, ctx_key in (("1m", "features_1m"), ("5m", "features_5m"), ("15m", "features_15m")):
            suffix = f"_{tf_key}"
            if lower.endswith(suffix):
                base_key = name[: -len(suffix)]
                # First: direct keys; then per_tf fallback
                base = context.get(ctx_key)
                if base is None:
                    per_tf = context.get("per_tf") or {}
                    tf_map = {"features_1m": "1min", "features_5m": "5min", "features_15m": "15min"}
                    tf_src = tf_map[ctx_key]
                    base = per_tf.get(tf_src) if isinstance(per_tf, dict) else None
                    if base is not None:
                        logger.debug(
                            "Extracted feature via fallback from per_tf (suffix)",
                            extra={"feature": name, "tf": tf_src, "base_key": base_key},
                        )
                val = self._get_by_dotted_path(base, base_key)
                return self._coerce_float(val)

        # 6) try known sources by prefix convention (best-effort)
        prefix_map = {
            "vp_": "volume_profile",
            "kl_": "key_levels",
            "opt_": "options",
            "sm5_": "smart_money_5m",
            "sm15_": "smart_money_15m",
            "fund_": "fundamental",
            "micro_": "microstructure",
        }
        for pfx, src in prefix_map.items():
            if lower.startswith(pfx):
                key = name[len(pfx) :]
                # smart money sources are special-cased because Captain uses context["smart_money"]
                if src == "smart_money_5m":
                    base = context.get("smart_money_5m")
                    if base is None:
                        base = (context.get("smart_money") or {}).get("5min")
                        if base is not None:
                            logger.debug("Extracted feature via fallback from smart_money['5min']", extra={"feature": name})
                elif src == "smart_money_15m":
                    base = context.get("smart_money_15m")
                    if base is None:
                        base = (context.get("smart_money") or {}).get("15min")
                        if base is not None:
                            logger.debug("Extracted feature via fallback from smart_money['15min']", extra={"feature": name})
                else:
                    base = context.get(src)

                val = self._get_by_dotted_path(base, key)
                return self._coerce_float(val)

        return None

    @staticmethod
    def _get_by_dotted_path(obj: Any, path: str) -> Any:
        if obj is None:
            return None
        cur = obj
        for part in str(path).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    @staticmethod
    def _coerce_float(val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            f = float(val)
            if not np.isfinite(f):
                return None
            return f
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Schema parsing
    # ---------------------------------------------------------------------
    @staticmethod
    def _parse_feature_schema(path: Path) -> List[FeatureSpec]:
        """
        Parse schema JSON.

        Supported formats:
          A) {"features":[{"name":..,"source":..,"path":..}, ...]}  (recommended)
          B) [{"name":..,"source":..,"path":..}, ...]              (list of dicts)
        """
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and isinstance(payload.get("features"), list):
            payload = payload["features"]

        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            specs: List[FeatureSpec] = []
            for i, d in enumerate(payload):
                name = str(d.get("name", "")).strip()
                source = str(d.get("source", "")).strip()
                pth = str(d.get("path", "")).strip()
                if not name or not source or not pth:
                    raise ValueError(f"Invalid feature spec at index {i}: {d}")
                specs.append(FeatureSpec(name=name, source=source, path=pth))
            return specs

        raise ValueError("Unsupported feature schema JSON structure")

    # ---------------------------------------------------------------------
    # Model loading helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _candidate_paths(model_dir: Path, model_name: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        name_path = Path(model_name)
        stem = name_path.stem
        candidates = {
            ".onnx": model_dir / (model_name if name_path.suffix == ".onnx" else f"{stem}.onnx"),
            ".txt": model_dir / (model_name if name_path.suffix == ".txt" else f"{stem}.txt"),
            ".pkl": model_dir / (model_name if name_path.suffix == ".pkl" else f"{stem}.pkl"),
        }
        return candidates[".onnx"], candidates[".txt"], candidates[".pkl"]

    def _try_load_onnx(self, path: Path) -> bool:
        try:
            import onnxruntime as ort  # lazy import

            sess_options = ort.SessionOptions()
            providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)

            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if not inputs:
                raise RuntimeError("ONNX model has no inputs")
            if not outputs:
                raise RuntimeError("ONNX model has no outputs")

            self._onnx_session = session
            self._onnx_input_name = inputs[0].name
            self._onnx_output_name = outputs[0].name
            return True
        except Exception as e:
            logger.error("ONNX load failed", extra={"path": str(path), "error": str(e), "traceback": traceback.format_exc()})
            self._onnx_session = None
            self._onnx_input_name = None
            self._onnx_output_name = None
            return False

    def _try_load_lgbm_txt(self, path: Path) -> bool:
        try:
            import lightgbm as lgb  # lazy import

            self._lgbm_booster = lgb.Booster(model_file=str(path))
            return True
        except Exception as e:
            logger.error(
                "LightGBM .txt load failed",
                extra={"path": str(path), "error": str(e), "traceback": traceback.format_exc()},
            )
            self._lgbm_booster = None
            return False

    def _try_load_pickle(self, path: Path) -> bool:
        try:
            from joblib import load as joblib_load  # lazy import

            self._pickle_model = joblib_load(path)
            return True
        except Exception as e:
            logger.error("Pickle model load failed", extra={"path": str(path), "error": str(e), "traceback": traceback.format_exc()})
            self._pickle_model = None
            return False

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_2d_float32(x: np.ndarray, expected_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != expected_dim:
            raise ValueError(f"Expected shape (1,{expected_dim}) or (n,{expected_dim}), got {arr.shape}")
        return arr

    @staticmethod
    def _coerce_probability(y: Any) -> float:
        """
        Convert model output into a scalar probability in [0,1].

        Handles:
          - scalar
          - shape (1,)
          - shape (1,1)
          - shape (1,2) => take class-1 probability (col 1) by convention
        """
        arr = np.asarray(y, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("Empty model output")

        p: float
        if arr.ndim == 0:
            p = float(arr)
        elif arr.ndim == 1:
            # If model outputs two probs in 1D, take the last as class-1 best-effort
            p = float(arr[-1])
        else:
            # 2D or more
            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                p = float(arr.reshape(arr.shape[0], -1)[0, 1])
            else:
                p = float(arr.reshape(-1)[0])

        if not np.isfinite(p):
            raise ValueError("Non-finite probability")
        return float(max(0.0, min(1.0, p)))

    def _model_loaded_kind(self) -> str:
        if self._onnx_session is not None:
            return "ONNX"
        if self._lgbm_booster is not None:
            return "LGBM_TXT"
        if self._pickle_model is not None:
            return "PICKLE"
        return "FALLBACK_CONSTANT"

    def _safe_decision(self, reason: str, probability: float = 0.50) -> MLFilterDecision:
        return MLFilterDecision(
            probability=float(probability),
            action="CAUTION",
            reduce_size=True,
            shap_values=None,
            rejection_reason=reason,
        )


# -------------------------------------------------------------------------
# Self-test (UPDATED FOR ACTUAL CAPTAIN CONTEXT STRUCTURE)
# -------------------------------------------------------------------------
def _run_tests() -> None:
    print("Running LightGBMFilter self-tests...")

    f = LightGBMFilter()

    opportunity = {"direction": "BUY", "strategy": "TEST", "score": 70}

    # ACTUAL Captain-style structure (as per audit)
    context: Dict[str, Any] = {
        "narrative_score": 0.0,
        "weighted_mtf": 0.0,
        "smart_money": {
            "5min": {"sm_direction_score": 0.0, "total_fvgs": 0},
            "15min": {"sm_direction_score": 0.0, "total_fvgs": 0},
        },
        "per_tf": {
            "1min": {"rsi": 50.0, "atr": 10.0},
            "5min": {"rsi": 52.0, "atr": 12.0},
            "15min": {"rsi": 55.0, "atr": 15.0},
        },
        # other common keys that some schemas may reference
        "volume_profile": {},
        "key_levels": {},
        "options": {},
        "fundamental": {},
        "microstructure": {},
    }

    decision = f.evaluate(opportunity, context)
    assert isinstance(decision, MLFilterDecision)
    assert isinstance(decision.probability, float)
    assert decision.action in ("REJECT", "CAUTION", "PASS")
    assert isinstance(decision.reduce_size, bool)
    assert (decision.shap_values is None) or isinstance(decision.shap_values, dict)
    assert isinstance(decision.rejection_reason, str)

    print("Model kind:", f._model_loaded_kind())
    print("Decision:", decision)

    # If schema exists, ensure we can build a vector even when schema expects old keys
    if f._feature_specs is not None:
        # Populate required values in Captain-style structures when applicable, so fallback paths get exercised.
        for spec in f._feature_specs:
            if spec.source == "opportunity":
                _assign_dotted(opportunity, spec.path, 0.0)
                continue

            if spec.source == "smart_money_5m":
                context.setdefault("smart_money", {}).setdefault("5min", {})
                _assign_dotted(context["smart_money"]["5min"], spec.path, 0.0)
                continue

            if spec.source == "smart_money_15m":
                context.setdefault("smart_money", {}).setdefault("15min", {})
                _assign_dotted(context["smart_money"]["15min"], spec.path, 0.0)
                continue

            if spec.source == "features_1m":
                context.setdefault("per_tf", {}).setdefault("1min", {})
                _assign_dotted(context["per_tf"]["1min"], spec.path, 0.0)
                continue

            if spec.source == "features_5m":
                context.setdefault("per_tf", {}).setdefault("5min", {})
                _assign_dotted(context["per_tf"]["5min"], spec.path, 0.0)
                continue

            if spec.source == "features_15m":
                context.setdefault("per_tf", {}).setdefault("15min", {})
                _assign_dotted(context["per_tf"]["15min"], spec.path, 0.0)
                continue

            # default
            if spec.source not in context or not isinstance(context.get(spec.source), dict):
                context[spec.source] = {}
            _assign_dotted(context[spec.source], spec.path, 0.0)

        x = f.build_feature_vector(opportunity, context)
        assert x is not None and x.shape == (1, 85)
        p = f.predict_proba_from_vector(x)
        assert p is not None and 0.0 <= p <= 1.0
        print("Feature-spec build test: OK (shape (1,85))")

    elif f._feature_order is not None:
        # Populate by name-based resolution using Captain-style structures when possible.
        for name in f._feature_order:
            if name.startswith("smart_money_5m."):
                sub = name[len("smart_money_5m.") :]
                context.setdefault("smart_money", {}).setdefault("5min", {})
                _assign_dotted(context["smart_money"]["5min"], sub, 0.0)
            elif name.startswith("smart_money_15m."):
                sub = name[len("smart_money_15m.") :]
                context.setdefault("smart_money", {}).setdefault("15min", {})
                _assign_dotted(context["smart_money"]["15min"], sub, 0.0)
            elif name.startswith("features_1m."):
                sub = name[len("features_1m.") :]
                context.setdefault("per_tf", {}).setdefault("1min", {})
                _assign_dotted(context["per_tf"]["1min"], sub, 0.0)
            elif name.startswith("features_5m."):
                sub = name[len("features_5m.") :]
                context.setdefault("per_tf", {}).setdefault("5min", {})
                _assign_dotted(context["per_tf"]["5min"], sub, 0.0)
            elif name.startswith("features_15m."):
                sub = name[len("features_15m.") :]
                context.setdefault("per_tf", {}).setdefault("15min", {})
                _assign_dotted(context["per_tf"]["15min"], sub, 0.0)
            else:
                # direct scalar fallback
                context[name] = 0.0

        x = f.build_feature_vector(opportunity, context)
        assert x is not None and x.shape == (1, 85)
        p = f.predict_proba_from_vector(x)
        assert p is not None and 0.0 <= p <= 1.0
        print("Feature-order build test: OK (shape (1,85))")

    else:
        print("No feature schema/order available; build_feature_vector strict mode will fail (expected).")

    print("LightGBMFilter self-tests PASSED.")


def _assign_dotted(d: Dict[str, Any], path: str, value: Any) -> None:
    cur = d
    parts = str(path).split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


if __name__ == "__main__":
    _run_tests()