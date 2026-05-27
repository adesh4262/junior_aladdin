"""
Junior Aladdin - SHAP Explainer Module
=====================================

This module provides model explanation capabilities using SHAP (SHapley Additive exPlanations)
to explain ML model predictions and provide transparency in trading decisions.

Key capabilities:
- Feature importance explanation for individual predictions
- Global feature importance analysis
- Model interpretation for audit and compliance
- Integration with LightGBM and other ML models
- Explanation storage and retrieval
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ..utils.config_loader import Config


@dataclass
class FeatureExplanation:
    """Explanation for a single feature in a prediction"""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_pct: float
    impact_direction: str  # "positive" or "negative"


@dataclass
class PredictionExplanation:
    """Complete explanation for a single prediction"""
    prediction_id: str
    model_name: str
    prediction_timestamp: datetime
    predicted_value: float
    base_value: float
    feature_explanations: List[FeatureExplanation]
    top_features: List[str]
    explanation_confidence: float
    computation_time_ms: float


@dataclass
class GlobalFeatureImportance:
    """Global feature importance across multiple predictions"""
    feature_name: str
    mean_absolute_shap: float
    mean_shap: float
    importance_rank: int
    contribution_variance: float
    sample_count: int


class SHAPExplainer:
    """
    SHAP-based model explainer for trading ML models
    
    Provides explanations for individual predictions and global feature importance
    to support model transparency, audit, and compliance requirements.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config
        self.logger = logging.getLogger("junior_aladdin.shap_explainer")
        
        # Check SHAP availability
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available - explanations will be disabled")
            self.shap_available = False
        else:
            self.shap_available = True
        
        # Storage paths
        self.base_path = Path("data/explanations")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.explanations_path = self.base_path / "prediction_explanations.json"
        self.global_importance_path = self.base_path / "global_importance.json"
        self.models_path = self.base_path / "shap_models"
        self.models_path.mkdir(exist_ok=True)
        
        # Configuration
        self.top_features_count = self.config.get("ml", "shap_top_features", default=5)
        self.explanation_cache_size = 1000
        
        # Internal state
        self.explainers = {}  # model_name -> shap_explainer
        self.feature_names = []  # Cached feature names
        self.prediction_explanations: List[PredictionExplanation] = []
        self.global_importance: Dict[str, GlobalFeatureImportance] = {}
        
        # Load existing data
        self._load_explanation_data()
        
        self.logger.info(f"SHAPExplainer initialized (SHAP available: {self.shap_available})")
    
    def _load_explanation_data(self) -> None:
        """Load existing explanation data from disk"""
        try:
            # Load prediction explanations
            if self.explanations_path.exists():
                with open(self.explanations_path, 'r') as f:
                    data = json.load(f)
                    for exp_data in data:
                        exp_data['prediction_timestamp'] = datetime.fromisoformat(exp_data['prediction_timestamp'])
                        
                        # Reconstruct FeatureExplanation objects
                        feature_exps = []
                        for feat_data in exp_data['feature_explanations']:
                            feature_exps.append(FeatureExplanation(**feat_data))
                        exp_data['feature_explanations'] = feature_exps
                        
                        self.prediction_explanations.append(PredictionExplanation(**exp_data))
                
                # Keep only recent explanations
                if len(self.prediction_explanations) > self.explanation_cache_size:
                    self.prediction_explanations = self.prediction_explanations[-self.explanation_cache_size:]
                
                self.logger.info(f"Loaded {len(self.prediction_explanations)} prediction explanations")
            
            # Load global importance
            if self.global_importance_path.exists():
                with open(self.global_importance_path, 'r') as f:
                    data = json.load(f)
                    for name, imp_data in data.items():
                        self.global_importance[name] = GlobalFeatureImportance(**imp_data)
                
                self.logger.info(f"Loaded global importance for {len(self.global_importance)} features")
                
        except Exception as e:
            self.logger.warning(f"Error loading explanation data: {e}")
    
    def _save_explanation_data(self) -> None:
        """Save explanation data to disk"""
        try:
            # Save prediction explanations
            exp_data = []
            for exp in self.prediction_explanations:
                exp_dict = asdict(exp)
                exp_dict['prediction_timestamp'] = exp_dict['prediction_timestamp'].isoformat()
                exp_data.append(exp_dict)
            
            with open(self.explanations_path, 'w') as f:
                json.dump(exp_data, f, indent=2)
            
            # Save global importance
            imp_data = {name: asdict(imp) for name, imp in self.global_importance.items()}
            with open(self.global_importance_path, 'w') as f:
                json.dump(imp_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving explanation data: {e}")
    
    def register_model(self,
                      model_name: str,
                      model: Any,
                      feature_names: List[str],
                      background_data: Optional[np.ndarray] = None) -> bool:
        """
        Register a model for SHAP explanation
        
        Args:
            model_name: Name/identifier for the model
            model: The ML model object (LightGBM, scikit-learn, etc.)
            feature_names: List of feature names
            background_data: Background dataset for SHAP (optional)
            
        Returns:
            True if registration successful, False otherwise
        """
        if not self.shap_available:
            self.logger.warning("SHAP not available - cannot register model")
            return False
        
        try:
            self.feature_names = feature_names
            
            # Create appropriate SHAP explainer based on model type
            if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
                # Tree-based models (LightGBM, XGBoost, etc.)
                if hasattr(model, 'feature_importance'):
                    explainer = shap.TreeExplainer(model, data=background_data)
                else:
                    # Kernel explainer as fallback
                    if background_data is None:
                        self.logger.warning("No background data provided for KernelExplainer - using random data")
                        background_data = np.random.normal(0, 1, (100, len(feature_names)))
                    explainer = shap.KernelExplainer(model.predict, background_data)
            else:
                self.logger.error(f"Unsupported create SHAP explainer for model type: {type(model)}")
                return False
            
            self.explainers[model_name] = explainer
            
            # Save the explainer for future use
            explainer_path = self.models_path / f"{model_name}_explainer.pkl"
            with open(explainer_path, 'wb') as f:
                pickle.dump(explainer, f)
            
            self.logger.info(f"Registered model '{model_name}' for SHAP explanation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering model '{model_name}': {e}")
            return False
    
    def explain_prediction(self,
                          model_name: str,
                          features: Dict[str, float],
                          prediction_value: float,
                          prediction_id: Optional[str] = None) -> Optional[PredictionExplanation]:
        """
        Explain a single prediction using SHAP
        
        Args:
            model_name: Name of the model that made the prediction
            features: Feature values used for prediction
            prediction_value: The model's prediction output
            prediction_id: Unique identifier for this prediction
            
        Returns:
            PredictionExplanation object or None if explanation failed
        """
        if not self.shap_available or model_name not in self.explainers:
            self.logger.warning(f"Cannot explain prediction - SHAP unavailable or model '{model_name}' not registered")
            return None
        
        start_time = datetime.now()
        
        try:
            # Generate prediction ID if not provided
            if prediction_id is None:
                prediction_id = f"{model_name}_{int(start_time.timestamp())}"
            
            # Prepare feature array
            feature_array = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            # Get SHAP values
            explainer = self.explainers[model_name]
            shap_values = explainer.shap_values(feature_array)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class
            
            # Get base value
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[0]
            
            # Create feature explanations
            feature_explanations = []
            total_abs_shap = np.sum(np.abs(shap_values))
            
            for i, (feature_name, shap_val, feature_val) in enumerate(zip(self.feature_names, shap_values, feature_array)):
                contribution_pct = (abs(shap_val) / total_abs_shap * 100) if total_abs_shap > 0 else 0
                impact_direction = "positive" if shap_val > 0 else "negative"
                
                feature_exp = FeatureExplanation(
                    feature_name=feature_name,
                    feature_value=float(feature_val),
                    shap_value=float(shap_val),
                    contribution_pct=float(contribution_pct),
                    impact_direction=impact_direction
                )
                feature_explanations.append(feature_exp)
            
            # Sort by absolute SHAP value
            feature_explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
            
            # Get top features
            top_features = [exp.feature_name for exp in feature_explanations[:self.top_features_count]]
            
            # Calculate explanation confidence (simplified)
            explanation_confidence = min(1.0, total_abs_shap / np.std(shap_values)) if np.std(shap_values) > 0 else 0.5
            
            # Computation time
            computation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create explanation object
            explanation = PredictionExplanation(
                prediction_id=prediction_id,
                model_name=model_name,
                prediction_timestamp=start_time,
                predicted_value=float(prediction_value),
                base_value=float(base_value),
                feature_explanations=feature_explanations,
                top_features=top_features,
                explanation_confidence=float(explanation_confidence),
                computation_time_ms=float(computation_time)
            )
            
            # Store explanation
            self.prediction_explanations.append(explanation)
            
            # Keep cache size limited
            if len(self.prediction_explanations) > self.explanation_cache_size:
                self.prediction_explanations = self.prediction_explanations[-self.explanation_cache_size:]
            
            # Update global importance
            self._update_global_importance(feature_explanations)
            
            # Save data
            self._save_explanation_data()
            
            self.logger.debug(f"Generated SHAP explanation for prediction '{prediction_id}'")
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return None
    
    def _update_global_importance(self, feature_explanations: List[FeatureExplanation]) -> None:
        """Update global feature importance based on new explanation"""
        try:
            for feat_exp in feature_explanations:
                feature_name = feat_exp.feature_name
                shap_value = feat_exp.shap_value
                
                if feature_name not in self.global_importance:
                    self.global_importance[feature_name] = GlobalFeatureImportance(
                        feature_name=feature_name,
                        mean_absolute_shap=0.0,
                        mean_shap=0.0,
                        importance_rank=0,
                        contribution_variance=0.0,
                        sample_count=0
                    )
                
                # Update running statistics
                imp = self.global_importance[feature_name]
                old_count = imp.sample_count
                new_count = old_count + 1
                
                # Update mean absolute SHAP
                imp.mean_absolute_shap = (imp.mean_absolute_shap * old_count + abs(shap_value)) / new_count
                
                # Update mean SHAP
                imp.mean_shap = (imp.mean_shap * old_count + shap_value) / new_count
                
                # Update sample count
                imp.sample_count = new_count
            
            # Update importance ranks
            sorted_features = sorted(self.global_importance.items(), 
                                   key=lambda x: x[1].mean_absolute_shap, reverse=True)
            for rank, (feature_name, _) in enumerate(sorted_features, 1):
                self.global_importance[feature_name].importance_rank = rank
                
        except Exception as e:
            self.logger.error(f"Error updating global importance: {e}")
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> List[GlobalFeatureImportance]:
        """
        Get global feature importance ranking
        
        Args:
            top_n: Number of top features to return (None for all)
            
        Returns:
            List of GlobalFeatureImportance objects sorted by importance
        """
        features = list(self.global_importance.values())
        features.sort(key=lambda x: x.mean_absolute_shap, reverse=True)
        
        if top_n:
            features = features[:top_n]
        
        return features
    
    def get_explanation_summary(self, 
                              model_name: Optional[str] = None,
                              hours_back: Optional[int] = 24) -> Dict[str, Any]:
        """
        Get summary of recent explanations
        
        Args:
            model_name: Filter by model name (None for all)
            hours_back: Only consider explanations from last N hours
            
        Returns:
            Summary statistics and insights
        """
        try:
            # Filter explanations
            cutoff_time = datetime.now() - timedelta(hours=hours_back) if hours_back else None
            filtered_exps = []
            
            for exp in self.prediction_explanations:
                if model_name and exp.model_name != model_name:
                    continue
                if cutoff_time and exp.prediction_timestamp < cutoff_time:
                    continue
                filtered_exps.append(exp)
            
            if not filtered_exps:
                return {"error": "No explanations found matching criteria"}
            
            # Calculate statistics
            total_explanations = len(filtered_exps)
            avg_computation_time = np.mean([exp.computation_time_ms for exp in filtered_exps])
            avg_confidence = np.mean([exp.explanation_confidence for exp in filtered_exps])
            
            # Top features across all explanations
            feature_counts = {}
            for exp in filtered_exps:
                for feature in exp.top_features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Model breakdown
            model_counts = {}
            for exp in filtered_exps:
                model_counts[exp.model_name] = model_counts.get(exp.model_name, 0) + 1
            
            return {
                "total_explanations": total_explanations,
                "time_period_hours": hours_back,
                "avg_computation_time_ms": float(avg_computation_time),
                "avg_explanation_confidence": float(avg_confidence),
                "top_features": [{"feature": f, "count": c} for f, c in top_features],
                "model_breakdown": model_counts,
                "shap_available": self.shap_available
            }
            
        except Exception as e:
            self.logger.error(f"Error generating explanation summary: {e}")
            return {"error": str(e)}
    
    def export_explanations_for_audit(self,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Export explanations for audit and compliance purposes
        
        Args:
            start_date: Start date for export range
            end_date: End date for export range
            
        Returns:
            DataFrame with explanation data
        """
        try:
            # Filter explanations by date range
            filtered_exps = []
            for exp in self.prediction_explanations:
                if start_date and exp.prediction_timestamp < start_date:
                    continue
                if end_date and exp.prediction_timestamp > end_date:
                    continue
                filtered_exps.append(exp)
            
            # Convert to DataFrame
            rows = []
            for exp in filtered_exps:
                base_row = {
                    "prediction_id": exp.prediction_id,
                    "model_name": exp.model_name,
                    "prediction_timestamp": exp.prediction_timestamp,
                    "predicted_value": exp.predicted_value,
                    "base_value": exp.base_value,
                    "explanation_confidence": exp.explanation_confidence,
                    "computation_time_ms": exp.computation_time_ms,
                    "top_features": ", ".join(exp.top_features)
                }
                
                # Add feature explanations as separate columns
                for feat_exp in exp.feature_explanations[:self.top_features_count]:
                    base_row[f"feature_{feat_exp.feature_name}_value"] = feat_exp.feature_value
                    base_row[f"feature_{feat_exp.feature_name}_shap"] = feat_exp.shap_value
                    base_row[f"feature_{feat_exp.feature_name}_contribution_pct"] = feat_exp.contribution_pct
                
                rows.append(base_row)
            
            df = pd.DataFrame(rows)
            self.logger.info(f"Exported {len(df)} explanation records for audit")
            return df
            
        except Exception as e:
            self.logger.error(f"Error exporting explanations for audit: {e}")
            return pd.DataFrame()
    
    def clear_explanation_cache(self, older_than_hours: int = 168) -> int:
        """
        Clear old explanation data from cache
        
        Args:
            older_than_hours: Remove explanations older than this many hours (default: 7 days)
            
        Returns:
            Number of explanations removed
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            original_count = len(self.prediction_explanations)
            
            self.prediction_explanations = [
                exp for exp in self.prediction_explanations 
                if exp.prediction_timestamp >= cutoff_time
            ]
            
            removed_count = original_count - len(self.prediction_explanations)
            
            # Save updated data
            self._save_explanation_data()
            
            self.logger.info(f"Cleared {removed_count} old explanations from cache")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error clearing explanation cache: {e}")
            return 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the SHAP explainer"""
        return {
            "shap_available": self.shap_available,
            "registered_models": list(self.explainers.keys()),
            "total_explanations": len(self.prediction_explanations),
            "global_features_tracked": len(self.global_importance),
            "feature_names_count": len(self.feature_names),
            "cache_size_limit": self.explanation_cache_size
        }


# Self-test function
def _self_test():
    """Basic self-test of the SHAPExplainer module"""
    print("Running SHAPExplainer self-test...")
    
    explainer = SHAPExplainer()
    
    if not explainer.shap_available:
        print("⚠ SHAP not available - skipping detailed tests")
        return
    
    # Mock model for testing
    class MockModel:
        def predict(self, X):
            return np.sum(X, axis=1) * 0.1
    
    # Register model
    model = MockModel()
    feature_names = ["rsi", "macd", "volume", "atr", "bb_width"]
    background_data = np.random.normal(0, 1, (50, len(feature_names)))
    
    success = explainer.register_model("test_model", model, feature_names, background_data)
    print(f"✓ Model registration: {success}")
    
    # Explain prediction
    features = {name: np.random.normal() for name in feature_names}
    prediction = 0.75
    
    explanation = explainer.explain_prediction("test_model", features, prediction)
    if explanation:
        print(f"✓ Prediction explanation generated (confidence: {explanation.explanation_confidence:.2f})")
        print(f"  Top features: {explanation.top_features[:3]}")
    else:
        print("✗ Failed to generate prediction explanation")
    
    # Get feature importance
    importance = explainer.get_feature_importance(top_n=3)
    print(f"✓ Top {len(importance)} features by importance: {[imp.feature_name for imp in importance]}")
    
    # Get status
    status = explainer.get_status()
    print(f"✓ Status: {status['total_explanations']} explanations, {status['registered_models']} models")
    
    print("✓ SHAPExplainer self-test completed")


if __name__ == "__main__":
    _self_test()
