"""
Junior Aladdin - Primary Regime Classifier Module
==================================================

This module implements the primary regime classification system using machine learning
to identify market regimes (trending, ranging, volatile, chop) based on technical indicators
and market conditions.

Key capabilities:
- Multi-class regime classification
- Real-time regime detection
- Regime transition probability estimation
- Model confidence scoring
- Historical regime analysis
- Model retraining and adaptation
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.config_loader import Config


class MarketRegime(Enum):
    """Market regime enumeration"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CHOPPY = "CHOPPY"
    TRANSITIONAL = "TRANSITIONAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeClassification:
    """Result of regime classification"""
    regime: MarketRegime
    confidence: float
    probabilities: Dict[str, float]
    transition_probability: float
    model_version: str
    timestamp: datetime
    feature_contributions: Dict[str, float]
    classification_time_ms: float


@dataclass
class RegimeTransition:
    """Recorded regime transition"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float
    transition_duration_minutes: float
    market_conditions: Dict[str, Any]


@dataclass
class RegimeStatistics:
    """Statistical summary of regime behavior"""
    regime: MarketRegime
    occurrence_count: int
    avg_duration_hours: float
    typical_volatility: float
    typical_return_per_hour: float
    transition_probabilities: Dict[str, float]
    last_seen: Optional[datetime]


class RegimeClassifier:
    """
    Primary regime classifier using machine learning
    
    This class provides the main regime classification capability, complementing
    the backup classifier with more sophisticated ML-based detection.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config
        self.logger = logging.getLogger("junior_aladdin.regime_classifier")
        
        # Check scikit-learn availability
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available - using rule-based classification")
            self.ml_available = False
        else:
            self.ml_available = True
        
        # Storage paths
        self.base_path = Path("data/models/regime_classifier")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.base_path / "regime_model.pkl"
        self.scaler_path = self.base_path / "feature_scaler.pkl"
        self.history_path = self.base_path / "regime_history.json"
        self.stats_path = self.base_path / "regime_statistics.json"
        
        # Configuration parameters
        self.min_samples_for_training = self.config.get("regime", "min_samples_for_training", default=100)
        self.confidence_threshold = self.config.get("regime", "confidence_threshold", default=0.6)
        self.transition_alert_threshold = self.config.get("regime", "transition_alert_threshold", default=0.7)
        self.stability_filter_bars = self.config.get("regime", "stability_filter_bars", default=3)
        self.model_retrain_interval_days = self.config.get("regime", "model_retrain_interval_days", default=30)
        
        # Feature configuration
        self.feature_names = [
            'rsi_14', 'macd_signal', 'bb_position', 'atr_ratio', 'volume_ratio',
            'price_momentum_5', 'price_momentum_15', 'volatility_ratio', 'trend_strength',
            'vix_level', 'time_of_day', 'day_of_week', 'price_range_ratio'
        ]
        
        # Internal state
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.model_version = "1.0"
        self.last_training_date = None
        
        # Regime history and statistics
        self.regime_history: List[RegimeClassification] = []
        self.regime_transitions: List[RegimeTransition] = []
        self.regime_statistics: Dict[str, RegimeStatistics] = {}
        
        # Current state
        self.current_regime = MarketRegime.UNKNOWN
        self.current_confidence = 0.0
        self.regime_stability_counter = 0
        
        # Load existing model and data
        self._load_model()
        self._load_historical_data()
        
        self.logger.info(f"RegimeClassifier initialized (ML available: {self.ml_available}, Trained: {self.is_trained})")
    
    def _load_model(self) -> None:
        """Load trained model from disk"""
        try:
            if self.model_path.exists() and self.scaler_path.exists() and self.ml_available:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                self.logger.info("Loaded trained regime classification model")
            else:
                self.logger.info("No pre-trained model found - will use rule-based classification")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.is_trained = False
    
    def _load_historical_data(self) -> None:
        """Load historical regime data"""
        try:
            # Load regime history
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    for hist_data in data:
                        hist_data['timestamp'] = datetime.fromisoformat(hist_data['timestamp'])
                        hist_data['regime'] = MarketRegime(hist_data['regime'])
                        self.regime_history.append(RegimeClassification(**hist_data))
                
                self.logger.info(f"Loaded {len(self.regime_history)} historical regime classifications")
            
            # Load regime statistics
            if self.stats_path.exists():
                with open(self.stats_path, 'r') as f:
                    data = json.load(f)
                    for regime_name, stats_data in data.items():
                        stats_data['regime'] = MarketRegime(regime_name)
                        if stats_data['last_seen']:
                            stats_data['last_seen'] = datetime.fromisoformat(stats_data['last_seen'])
                        self.regime_statistics[regime_name] = RegimeStatistics(**stats_data)
                
                self.logger.info(f"Loaded statistics for {len(self.regime_statistics)} regimes")
                
        except Exception as e:
            self.logger.warning(f"Error loading historical data: {e}")
    
    def _save_model(self) -> None:
        """Save model and scaler to disk"""
        if not self.ml_available or not self.is_trained:
            return
        
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info("Saved regime classification model")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def _save_historical_data(self) -> None:
        """Save historical data to disk"""
        try:
            # Save regime history
            hist_data = []
            for hist in self.regime_history[-1000:]:  # Keep last 1000 records
                hist_dict = asdict(hist)
                hist_dict['regime'] = hist_dict['regime'].value
                hist_dict['timestamp'] = hist_dict['timestamp'].isoformat()
                hist_data.append(hist_dict)
            
            with open(self.history_path, 'w') as f:
                json.dump(hist_data, f, indent=2)
            
            # Save regime statistics
            stats_data = {}
            for regime_name, stats in self.regime_statistics.items():
                stats_dict = asdict(stats)
                stats_dict['regime'] = stats_dict['regime'].value
                if stats_dict['last_seen']:
                    stats_dict['last_seen'] = stats_dict['last_seen'].isoformat()
                stats_data[regime_name.value] = stats_dict
            
            with open(self.stats_path, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving historical data: {e}")
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data"""
        try:
            features = []
            
            # Technical indicators
            features.append(market_data.get('rsi_14', 50.0))
            features.append(market_data.get('macd_signal', 0.0))
            features.append(market_data.get('bb_position', 0.5))
            features.append(market_data.get('atr_ratio', 1.0))
            features.append(market_data.get('volume_ratio', 1.0))
            
            # Momentum indicators
            features.append(market_data.get('price_momentum_5', 0.0))
            features.append(market_data.get('price_momentum_15', 0.0))
            features.append(market_data.get('volatility_ratio', 1.0))
            features.append(market_data.get('trend_strength', 0.0))
            
            # Market context
            features.append(market_data.get('vix_level', 15.0))
            features.append(market_data.get('time_of_day', 12.0) / 24.0)  # Normalized
            features.append(market_data.get('day_of_week', 3.0) / 6.0)     # Normalized
            features.append(market_data.get('price_range_ratio', 0.02))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros((1, len(self.feature_names)))
    
    def _rule_based_classification(self, features: np.ndarray) -> Tuple[MarketRegime, float, Dict[str, float]]:
        """Fallback rule-based classification when ML is unavailable"""
        try:
            # Extract individual features for rule processing
            rsi = features[0, 0]
            macd_signal = features[0, 1]
            bb_position = features[0, 2]
            volatility_ratio = features[0, 7]
            trend_strength = features[0, 8]
            vix = features[0, 9]
            
            # Simple rule-based logic
            if trend_strength > 0.7 and rsi > 60:
                regime = MarketRegime.TRENDING_UP
                confidence = min(0.8, trend_strength)
            elif trend_strength > 0.7 and rsi < 40:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(0.8, trend_strength)
            elif volatility_ratio > 2.0 or vix > 25:
                regime = MarketRegime.VOLATILE
                confidence = min(0.7, volatility_ratio / 3.0)
            elif trend_strength < 0.3 and abs(macd_signal) < 0.1:
                regime = MarketRegime.CHOPPY
                confidence = 0.6
            else:
                regime = MarketRegime.RANGING
                confidence = 0.5
            
            # Create probability distribution (simplified)
            probabilities = {
                "TRENDING_UP": 0.2 if regime != MarketRegime.TRENDING_UP else confidence,
                "TRENDING_DOWN": 0.2 if regime != MarketRegime.TRENDING_DOWN else confidence,
                "RANGING": 0.2 if regime != MarketRegime.RANGING else confidence,
                "VOLATILE": 0.1 if regime != MarketRegime.VOLATILE else confidence,
                "CHOPPY": 0.1 if regime != MarketRegime.CHOPPY else confidence,
                "TRANSITIONAL": 0.1,
                "UNKNOWN": 0.1
            }
            
            # Normalize probabilities
            total = sum(probabilities.values())
            probabilities = {k: v/total for k, v in probabilities.items()}
            
            return regime, confidence, probabilities
            
        except Exception as e:
            self.logger.error(f"Error in rule-based classification: {e}")
            return MarketRegime.UNKNOWN, 0.1, {"UNKNOWN": 1.0}
    
    def classify_regime(self, market_data: Dict[str, Any]) -> RegimeClassification:
        """
        Classify current market regime
        
        Args:
            market_data: Dictionary containing market indicators and features
            
        Returns:
            RegimeClassification object with prediction and confidence
        """
        start_time = datetime.now()
        
        try:
            # Extract features
            features = self._extract_features(market_data)
            
            # Classification
            if self.ml_available and self.is_trained and self.model is not None:
                # ML-based classification
                features_scaled = self.scaler.transform(features)
                probabilities_raw = self.model.predict_proba(features_scaled)[0]
                
                # Get predicted class and confidence
                predicted_idx = np.argmax(probabilities_raw)
                confidence = probabilities_raw[predicted_idx]
                
                # Map to regime enum
                class_names = self.model.classes_
                regime_name = class_names[predicted_idx]
                regime = MarketRegime(regime_name)
                
                # Create probability dictionary
                probabilities = {}
                for i, class_name in enumerate(class_names):
                    probabilities[class_name] = float(probabilities_raw[i])
                
                # Feature contributions (simplified - would need SHAP for real contributions)
                feature_contributions = {}
                for i, feature_name in enumerate(self.feature_names):
                    feature_contributions[feature_name] = float(features[0, i] * 0.01)  # Placeholder
                
            else:
                # Rule-based fallback
                regime, confidence, probabilities = self._rule_based_classification(features)
                feature_contributions = {name: 0.01 for name in self.feature_names}
            
            # Calculate transition probability
            transition_probability = self._calculate_transition_probability(regime, confidence)
            
            # Apply stability filter
            if self._should_apply_stability_filter(regime, confidence):
                regime = MarketRegime.TRANSITIONAL
                confidence *= 0.8  # Reduce confidence for transitional periods
            
            # Create classification result
            classification = RegimeClassification(
                regime=regime,
                confidence=float(confidence),
                probabilities=probabilities,
                transition_probability=float(transition_probability),
                model_version=self.model_version,
                timestamp=start_time,
                feature_contributions=feature_contributions,
                classification_time_ms=float((datetime.now() - start_time).total_seconds() * 1000)
            )
            
            # Update state and history
            self._update_regime_state(classification)
            
            # Save historical data
            self._save_historical_data()
            
            self.logger.debug(f"Classified regime: {regime.value} (confidence: {confidence:.2f})")
            return classification
            
        except Exception as e:
            self.logger.error(f"Error classifying regime: {e}")
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                probabilities={"UNKNOWN": 1.0},
                transition_probability=0.0,
                model_version="error",
                timestamp=datetime.now(),
                feature_contributions={},
                classification_time_ms=0.0
            )
    
    def _calculate_transition_probability(self, new_regime: MarketRegime, confidence: float) -> float:
        """Calculate probability of regime transition"""
        try:
            if self.current_regime == MarketRegime.UNKNOWN:
                return 0.0
            
            if new_regime != self.current_regime:
                # Higher transition probability if confidence is high and it's a real change
                base_probability = 0.3
                confidence_boost = confidence * 0.4
                return min(0.9, base_probability + confidence_boost)
            else:
                # Low probability of transition if regime is stable
                return 0.05
                
        except Exception:
            return 0.1
    
    def _should_apply_stability_filter(self, regime: MarketRegime, confidence: float) -> bool:
        """Determine if stability filter should be applied"""
        try:
            # If regime changed recently, require stability confirmation
            if regime != self.current_regime:
                self.regime_stability_counter = 0
                return True
            
            # Increment stability counter for consistent regime
            if confidence > self.confidence_threshold:
                self.regime_stability_counter += 1
            else:
                self.regime_stability_counter = 0
            
            # Require multiple confirmations for regime changes
            return self.regime_stability_counter < self.stability_filter_bars
            
        except Exception:
            return False
    
    def _update_regime_state(self, classification: RegimeClassification) -> None:
        """Update internal regime state and statistics"""
        try:
            # Check for regime transition
            if (self.current_regime != MarketRegime.UNKNOWN and 
                classification.regime != self.current_regime):
                
                transition = RegimeTransition(
                    from_regime=self.current_regime,
                    to_regime=classification.regime,
                    timestamp=classification.timestamp,
                    confidence=classification.confidence,
                    transition_duration_minutes=0.0,  # Would need timing logic
                    market_conditions={}  # Would need market context
                )
                self.regime_transitions.append(transition)
                
                # Keep only recent transitions
                if len(self.regime_transitions) > 100:
                    self.regime_transitions = self.regime_transitions[-100:]
            
            # Update current state
            self.current_regime = classification.regime
            self.current_confidence = classification.confidence
            
            # Add to history
            self.regime_history.append(classification)
            
            # Keep history size manageable
            if len(self.regime_history) > 2000:
                self.regime_history = self.regime_history[-2000:]
            
            # Update statistics
            self._update_regime_statistics(classification)
            
        except Exception as e:
            self.logger.error(f"Error updating regime state: {e}")
    
    def _update_regime_statistics(self, classification: RegimeClassification) -> None:
        """Update regime statistics"""
        try:
            regime_name = classification.regime.value
            
            if regime_name not in self.regime_statistics:
                self.regime_statistics[regime_name] = RegimeStatistics(
                    regime=classification.regime,
                    occurrence_count=0,
                    avg_duration_hours=0.0,
                    typical_volatility=0.0,
                    typical_return_per_hour=0.0,
                    transition_probabilities={},
                    last_seen=None
                )
            
            stats = self.regime_statistics[regime_name]
            stats.occurrence_count += 1
            stats.last_seen = classification.timestamp
            
            # Update other statistics (simplified - would need more sophisticated tracking)
            
        except Exception as e:
            self.logger.error(f"Error updating regime statistics: {e}")
    
    def train_model(self, training_data: pd.DataFrame, regime_labels: pd.Series) -> bool:
        """
        Train the regime classification model
        
        Args:
            training_data: DataFrame with features
            regime_labels: Series with regime labels
            
        Returns:
            True if training successful, False otherwise
        """
        if not self.ml_available:
            self.logger.warning("Scikit-learn not available - cannot train model")
            return False
        
        try:
            if len(training_data) < self.min_samples_for_training:
                self.logger.warning(f"Insufficient training data: {len(training_data)} < {self.min_samples_for_training}")
                return False
            
            self.logger.info(f"Training regime classifier with {len(training_data)} samples")
            
            # Prepare features
            X = training_data[self.feature_names].values
            y = regime_labels.values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = np.mean(y_pred == y_test)
            
            self.logger.info(f"Model training completed. Test accuracy: {accuracy:.3f}")
            
            # Update model info
            self.is_trained = True
            self.model_version = f"trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.last_training_date = datetime.now()
            
            # Save model
            self._save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False
    
    def get_regime_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get summary of recent regime behavior
        
        Args:
            hours_back: Number of hours to look back
            
        Returns:
            Summary statistics and insights
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_classifications = [
                c for c in self.regime_history 
                if c.timestamp >= cutoff_time
            ]
            
            if not recent_classifications:
                return {"error": "No recent regime data available"}
            
            # Regime distribution
            regime_counts = {}
            for classification in recent_classifications:
                regime_name = classification.regime.value
                regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            
            # Average confidence
            avg_confidence = np.mean([c.confidence for c in recent_classifications])
            
            # Transition count
            transitions = len([t for t in self.regime_transitions if t.timestamp >= cutoff_time])
            
            # Current regime info
            current_info = {
                "regime": self.current_regime.value,
                "confidence": self.current_confidence,
                "stability_counter": self.regime_stability_counter
            }
            
            return {
                "period_hours": hours_back,
                "total_classifications": len(recent_classifications),
                "regime_distribution": regime_counts,
                "avg_confidence": float(avg_confidence),
                "transitions_detected": transitions,
                "current_regime": current_info,
                "model_trained": self.is_trained,
                "model_version": self.model_version
            }
            
        except Exception as e:
            self.logger.error(f"Error generating regime summary: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the regime classifier"""
        return {
            "ml_available": self.ml_available,
            "model_trained": self.is_trained,
            "model_version": self.model_version,
            "current_regime": self.current_regime.value,
            "current_confidence": self.current_confidence,
            "total_classifications": len(self.regime_history),
            "total_transitions": len(self.regime_transitions),
            "regimes_tracked": len(self.regime_statistics),
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None
        }


# Self-test function
def _self_test():
    """Basic self-test of the RegimeClassifier module"""
    print("Running RegimeClassifier self-test...")
    
    classifier = RegimeClassifier()
    
    # Test regime classification
    market_data = {
        'rsi_14': 65.0,
        'macd_signal': 0.2,
        'bb_position': 0.8,
        'atr_ratio': 1.2,
        'volume_ratio': 1.5,
        'price_momentum_5': 0.01,
        'price_momentum_15': 0.02,
        'volatility_ratio': 1.1,
        'trend_strength': 0.7,
        'vix_level': 18.0,
        'time_of_day': 14.0,
        'day_of_week': 3.0,
        'price_range_ratio': 0.015
    }
    
    classification = classifier.classify_regime(market_data)
    print(f"✓ Regime classification: {classification.regime.value} (confidence: {classification.confidence:.2f})")
    print(f"  Transition probability: {classification.transition_probability:.2f}")
    print(f"  Classification time: {classification.classification_time_ms:.1f}ms")
    
    # Test regime summary
    summary = classifier.get_regime_summary(hours_back=1)
    if "error" not in summary:
        print(f"✓ Regime summary generated for last {summary['period_hours']} hours")
        print(f"  Classifications: {summary['total_classifications']}")
    
    # Test status
    status = classifier.get_status()
    print(f"✓ Status: ML available={status['ml_available']}, Trained={status['model_trained']}")
    print(f"  Current regime: {status['current_regime']} (confidence: {status['current_confidence']:.2f})")
    
    print("✓ RegimeClassifier self-test completed")


if __name__ == "__main__":
    _self_test()
