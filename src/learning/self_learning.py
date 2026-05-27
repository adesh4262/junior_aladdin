"""
Junior Aladdin - Self Learning Module
=====================================

This module implements adaptive learning from trading history, strategy performance,
and market regime changes to continuously improve system parameters and decision making.

Key capabilities:
- Strategy performance tracking and threshold adaptation
- Regime transition learning and prediction
- Feature importance drift detection
- Post-trade analysis and pattern recognition
- Weekly/monthly learning jobs
"""

import logging
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from ..utils.config_loader import Config


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sharpe_ratio: float
    max_drawdown: float
    last_updated: datetime


@dataclass
class RegimeTransition:
    """Recorded regime transition with context"""
    from_regime: str
    to_regime: str
    timestamp: datetime
    transition_confidence: float
    market_conditions: Dict[str, Any]
    post_transition_performance: Dict[str, float]


@dataclass
class FeatureDriftReport:
    """Feature importance and distribution drift report"""
    feature_name: str
    historical_importance: float
    current_importance: float
    drift_score: float
    distribution_shift: float
    last_updated: datetime


class SelfLearning:
    """
    Self-learning system for continuous improvement
    
    This module analyzes trading history, strategy performance, and market patterns
    to adapt system parameters and improve decision quality over time.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config
        self.logger = logging.getLogger("junior_aladdin.learning")
        
        # Storage paths
        self.base_path = Path("data/learning")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.performance_path = self.base_path / "strategy_performance.json"
        self.regime_path = self.base_path / "regime_transitions.json"
        self.drift_path = self.base_path / "feature_drift.json"
        self.models_path = self.base_path / "learned_models"
        self.models_path.mkdir(exist_ok=True)
        
        # In-memory caches
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.regime_transitions: List[RegimeTransition] = []
        self.feature_drifts: Dict[str, FeatureDriftReport] = {}
        
        # Learning parameters from config
        self.min_trades_for_adjustment = self.config.get("learning", "min_trades_for_adjustment", default=20)
        self.bad_winrate_threshold = self.config.get("learning", "bad_winrate_threshold", default=0.40)
        self.good_winrate_threshold = self.config.get("learning", "good_winrate_threshold", default=0.70)
        self.threshold_increase_on_bad = self.config.get("learning", "threshold_increase_on_bad", default=5)
        self.threshold_decrease_on_good = self.config.get("learning", "threshold_decrease_on_good", default=3)
        self.min_strategy_threshold = self.config.get("learning", "min_strategy_threshold", default=55)
        self.max_threshold_change_per_week = self.config.get("learning", "max_threshold_change_per_week", default=5)
        self.rollback_after_bad_days = self.config.get("learning", "rollback_after_bad_days", default=5)
        
        # Load existing data
        self._load_historical_data()
        
        self.logger.info("SelfLearning module initialized")
    
    def _load_historical_data(self) -> None:
        """Load existing learning data from disk"""
        try:
            # Load strategy performance
            if self.performance_path.exists():
                with open(self.performance_path, 'r') as f:
                    data = json.load(f)
                    for name, perf_data in data.items():
                        perf_data['last_updated'] = datetime.fromisoformat(perf_data['last_updated'])
                        self.strategy_performance[name] = StrategyPerformance(**perf_data)
                self.logger.info(f"Loaded performance data for {len(self.strategy_performance)} strategies")
            
            # Load regime transitions
            if self.regime_path.exists():
                with open(self.regime_path, 'r') as f:
                    data = json.load(f)
                    for trans_data in data:
                        trans_data['timestamp'] = datetime.fromisoformat(trans_data['timestamp'])
                        self.regime_transitions.append(RegimeTransition(**trans_data))
                self.logger.info(f"Loaded {len(self.regime_transitions)} regime transitions")
            
            # Load feature drifts
            if self.drift_path.exists():
                with open(self.drift_path, 'r') as f:
                    data = json.load(f)
                    for name, drift_data in data.items():
                        drift_data['last_updated'] = datetime.fromisoformat(drift_data['last_updated'])
                        self.feature_drifts[name] = FeatureDriftReport(**drift_data)
                self.logger.info(f"Loaded drift data for {len(self.feature_drifts)} features")
                
        except Exception as e:
            self.logger.warning(f"Error loading historical learning data: {e}")
    
    def _save_data(self) -> None:
        """Save learning data to disk"""
        try:
            # Save strategy performance
            perf_data = {name: asdict(perf) for name, perf in self.strategy_performance.items()}
            for perf in perf_data.values():
                perf['last_updated'] = perf['last_updated'].isoformat()
            with open(self.performance_path, 'w') as f:
                json.dump(perf_data, f, indent=2)
            
            # Save regime transitions
            trans_data = [asdict(trans) for trans in self.regime_transitions]
            for trans in trans_data:
                trans['timestamp'] = trans['timestamp'].isoformat()
            with open(self.regime_path, 'w') as f:
                json.dump(trans_data, f, indent=2)
            
            # Save feature drifts
            drift_data = {name: asdict(drift) for name, drift in self.feature_drifts.items()}
            for drift in drift_data.values():
                drift['last_updated'] = drift['last_updated'].isoformat()
            with open(self.drift_path, 'w') as f:
                json.dump(drift_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
    
    def record_trade_outcome(self, 
                           strategy_name: str,
                           trade_result: Dict[str, Any]) -> None:
        """
        Record the outcome of a single trade for learning
        
        Args:
            strategy_name: Name of the strategy that generated the trade
            trade_result: Dictionary containing trade outcome data
        """
        try:
            if strategy_name not in self.strategy_performance:
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0.0,
                    avg_win=0.0,
                    avg_loss=0.0,
                    profit_factor=0.0,
                    max_consecutive_wins=0,
                    max_consecutive_losses=0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    last_updated=datetime.now()
                )
            
            perf = self.strategy_performance[strategy_name]
            
            # Update basic metrics
            perf.total_trades += 1
            perf.last_updated = datetime.now()
            
            # Extract trade P&L
            pnl = float(trade_result.get('pnl', 0.0))
            is_win = pnl > 0
            
            if is_win:
                perf.winning_trades += 1
                if perf.avg_win == 0:
                    perf.avg_win = abs(pnl)
                else:
                    perf.avg_win = (perf.avg_win * (perf.winning_trades - 1) + abs(pnl)) / perf.winning_trades
            else:
                perf.losing_trades += 1
                if perf.avg_loss == 0:
                    perf.avg_loss = abs(pnl)
                else:
                    perf.avg_loss = (perf.avg_loss * (perf.losing_trades - 1) + abs(pnl)) / perf.losing_trades
            
            # Update win rate
            perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0.0
            
            # Update profit factor
            if perf.avg_loss > 0:
                perf.profit_factor = (perf.avg_win * perf.winning_trades) / (perf.avg_loss * perf.losing_trades)
            
            # Update consecutive wins/losses (simplified)
            if is_win:
                perf.max_consecutive_wins = max(perf.max_consecutive_wins, 
                                              trade_result.get('consecutive_wins', 1))
            else:
                perf.max_consecutive_losses = max(perf.max_consecutive_losses,
                                                trade_result.get('consecutive_losses', 1))
            
            self.logger.debug(f"Recorded trade outcome for {strategy_name}: PnL={pnl}, Win={is_win}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade outcome: {e}")
    
    def record_regime_transition(self,
                               from_regime: str,
                               to_regime: str,
                               confidence: float,
                               market_conditions: Dict[str, Any]) -> None:
        """Record a regime transition for pattern learning"""
        try:
            transition = RegimeTransition(
                from_regime=from_regime,
                to_regime=to_regime,
                timestamp=datetime.now(),
                transition_confidence=confidence,
                market_conditions=market_conditions,
                post_transition_performance={}  # To be filled later
            )
            
            self.regime_transitions.append(transition)
            
            # Keep only recent transitions (last 100)
            if len(self.regime_transitions) > 100:
                self.regime_transitions = self.regime_transitions[-100:]
            
            self.logger.info(f"Recorded regime transition: {from_regime} -> {to_regime} (confidence: {confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error recording regime transition: {e}")
    
    def analyze_feature_drift(self,
                            current_features: Dict[str, float],
                            feature_importance: Dict[str, float]) -> Dict[str, FeatureDriftReport]:
        """
        Analyze feature drift compared to historical baselines
        
        Args:
            current_features: Current feature values
            feature_importance: Current feature importance scores
            
        Returns:
            Dictionary of feature drift reports
        """
        drift_reports = {}
        
        try:
            for feature_name, importance in feature_importance.items():
                historical_importance = 0.0
                if feature_name in self.feature_drifts:
                    historical_importance = self.feature_drifts[feature_name].historical_importance
                
                # Calculate drift score
                drift_score = abs(importance - historical_importance) / (historical_importance + 0.001)
                
                # Calculate distribution shift (simplified - would need historical data)
                distribution_shift = 0.0  # Placeholder
                
                drift_report = FeatureDriftReport(
                    feature_name=feature_name,
                    historical_importance=historical_importance,
                    current_importance=importance,
                    drift_score=drift_score,
                    distribution_shift=distribution_shift,
                    last_updated=datetime.now()
                )
                
                drift_reports[feature_name] = drift_report
                self.feature_drifts[feature_name] = drift_report
            
            self.logger.info(f"Analyzed drift for {len(drift_reports)} features")
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature drift: {e}")
        
        return drift_reports
    
    def suggest_strategy_threshold_adjustments(self) -> Dict[str, Dict[str, Any]]:
        """
        Suggest threshold adjustments based on recent performance
        
        Returns:
            Dictionary of strategy names and suggested adjustments
        """
        suggestions = {}
        
        try:
            for strategy_name, perf in self.strategy_performance.items():
                if perf.total_trades < self.min_trades_for_adjustment:
                    continue
                
                adjustment = {"current_threshold": 0.0, "suggested_threshold": 0.0, "reason": ""}
                
                # Get current threshold (simplified - would need to access actual strategy configs)
                current_threshold = self.min_strategy_threshold  # Placeholder
                adjustment["current_threshold"] = current_threshold
                
                # Analyze performance and suggest adjustment
                if perf.win_rate < self.bad_winrate_threshold:
                    # Poor performance - increase threshold
                    suggested = min(current_threshold + self.threshold_increase_on_bad, 95)
                    adjustment["suggested_threshold"] = suggested
                    adjustment["reason"] = f"Poor win rate ({perf.win_rate:.2f}) - increasing threshold"
                    
                elif perf.win_rate > self.good_winrate_threshold:
                    # Good performance - can decrease threshold slightly
                    suggested = max(current_threshold - self.threshold_decrease_on_good, self.min_strategy_threshold)
                    adjustment["suggested_threshold"] = suggested
                    adjustment["reason"] = f"Good win rate ({perf.win_rate:.2f}) - decreasing threshold"
                
                else:
                    adjustment["suggested_threshold"] = current_threshold
                    adjustment["reason"] = f"Acceptable performance ({perf.win_rate:.2f}) - no change"
                
                # Limit maximum change per week
                max_change = self.max_threshold_change_per_week
                if abs(adjustment["suggested_threshold"] - current_threshold) > max_change:
                    if adjustment["suggested_threshold"] > current_threshold:
                        adjustment["suggested_threshold"] = current_threshold + max_change
                    else:
                        adjustment["suggested_threshold"] = current_threshold - max_change
                    adjustment["reason"] += " (limited to max weekly change)"
                
                suggestions[strategy_name] = adjustment
            
            self.logger.info(f"Generated threshold suggestions for {len(suggestions)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error generating threshold suggestions: {e}")
        
        return suggestions
    
    def run_weekly_learning_job(self) -> Dict[str, Any]:
        """
        Run comprehensive weekly learning job
        
        Returns:
            Summary of learning outcomes and recommendations
        """
        self.logger.info("Starting weekly learning job")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "strategies_analyzed": 0,
            "threshold_adjustments": {},
            "regime_patterns": {},
            "feature_drift_alerts": [],
            "recommendations": []
        }
        
        try:
            # Analyze strategy performance
            results["strategies_analyzed"] = len(self.strategy_performance)
            results["threshold_adjustments"] = self.suggest_strategy_threshold_adjustments()
            
            # Analyze regime transitions
            if len(self.regime_transitions) >= 5:
                recent_transitions = self.regime_transitions[-20:]  # Last 20 transitions
                transition_patterns = {}
                for trans in recent_transitions:
                    key = f"{trans.from_regime}->{trans.to_regime}"
                    transition_patterns[key] = transition_patterns.get(key, 0) + 1
                results["regime_patterns"] = transition_patterns
            
            # Check for significant feature drift
            for feature_name, drift in self.feature_drifts.items():
                if drift.drift_score > 0.5:  # Significant drift threshold
                    results["feature_drift_alerts"].append({
                        "feature": feature_name,
                        "drift_score": drift.drift_score,
                        "importance_change": drift.current_importance - drift.historical_importance
                    })
            
            # Generate recommendations
            if results["threshold_adjustments"]:
                results["recommendations"].append("Consider applying suggested threshold adjustments")
            
            if results["feature_drift_alerts"]:
                results["recommendations"].append("Investigate features with significant drift")
            
            if len(self.regime_transitions) > 50:
                results["recommendations"].append("Consider retraining regime detection models")
            
            # Save updated data
            self._save_data()
            
            self.logger.info("Weekly learning job completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in weekly learning job: {e}")
            results["error"] = str(e)
        
        return results
    
    def run_monthly_learning_job(self) -> Dict[str, Any]:
        """
        Run comprehensive monthly learning job with deeper analysis
        
        Returns:
            Detailed monthly learning report
        """
        self.logger.info("Starting monthly learning job")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "monthly_performance_summary": {},
            "strategy_rankings": [],
            "regime_stability_analysis": {},
            "feature_importance_evolution": {},
            "model_retraining_recommendations": [],
            "system_health_metrics": {}
        }
        
        try:
            # Strategy performance summary
            for strategy_name, perf in self.strategy_performance.items():
                if perf.total_trades >= 10:  # Only include strategies with sufficient data
                    results["monthly_performance_summary"][strategy_name] = {
                        "win_rate": perf.win_rate,
                        "total_trades": perf.total_trades,
                        "profit_factor": perf.profit_factor,
                        "sharpe_ratio": perf.sharpe_ratio
                    }
            
            # Rank strategies by composite score
            strategy_scores = []
            for strategy_name, perf in self.strategy_performance.items():
                if perf.total_trades >= 10:
                    # Simple composite score (can be made more sophisticated)
                    score = perf.win_rate * 0.4 + min(perf.profit_factor, 3.0) / 3.0 * 0.3 + min(perf.sharpe_ratio, 2.0) / 2.0 * 0.3
                    strategy_scores.append((strategy_name, score))
            
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            results["strategy_rankings"] = [{"strategy": name, "score": score} for name, score in strategy_scores]
            
            # Regime stability analysis
            if len(self.regime_transitions) >= 10:
                regime_durations = {}
                for i, trans in enumerate(self.regime_transitions):
                    if i == 0:
                        continue
                    prev_trans = self.regime_transitions[i-1]
                    duration = (trans.timestamp - prev_trans.timestamp).total_seconds() / 3600  # hours
                    regime = prev_trans.to_regime
                    if regime not in regime_durations:
                        regime_durations[regime] = []
                    regime_durations[regime].append(duration)
                
                for regime, durations in regime_durations.items():
                    results["regime_stability_analysis"][regime] = {
                        "avg_duration_hours": np.mean(durations),
                        "min_duration_hours": np.min(durations),
                        "max_duration_hours": np.max(durations),
                        "transition_count": len(durations)
                    }
            
            # Feature importance evolution
            for feature_name, drift in self.feature_drifts.items():
                results["feature_importance_evolution"][feature_name] = {
                    "historical_importance": drift.historical_importance,
                    "current_importance": drift.current_importance,
                    "drift_magnitude": drift.drift_score
                }
            
            # Model retraining recommendations
            if results["feature_importance_evolution"]:
                high_drift_features = [f for f, data in results["feature_importance_evolution"].items() 
                                    if data["drift_magnitude"] > 0.3]
                if high_drift_features:
                    results["model_retraining_recommendations"].append(
                        f"High drift detected in {len(high_drift_features)} features - consider retraining ML models"
                    )
            
            # System health metrics
            total_trades = sum(perf.total_trades for perf in self.strategy_performance.values())
            avg_win_rate = np.mean([perf.win_rate for perf in self.strategy_performance.values() if perf.total_trades > 0])
            
            results["system_health_metrics"] = {
                "total_trades_all_strategies": total_trades,
                "average_win_rate": avg_win_rate,
                "number_of_active_strategies": len([p for p in self.strategy_performance.values() if p.total_trades > 0]),
                "regime_transitions_recorded": len(self.regime_transitions),
                "features_monitored": len(self.feature_drifts)
            }
            
            # Save updated data
            self._save_data()
            
            self.logger.info("Monthly learning job completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in monthly learning job: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current status of the learning system"""
        return {
            "strategies_tracked": len(self.strategy_performance),
            "regime_transitions_recorded": len(self.regime_transitions),
            "features_monitored": len(self.feature_drifts),
            "last_weekly_job": "Not run yet",
            "last_monthly_job": "Not run yet",
            "data_quality": "Good" if len(self.strategy_performance) > 0 else "Insufficient data"
        }
    
    def reset_learning_data(self, confirm: bool = False) -> bool:
        """Reset all learning data (use with caution)"""
        if not confirm:
            self.logger.warning("Reset not confirmed - use confirm=True to reset learning data")
            return False
        
        try:
            # Clear in-memory data
            self.strategy_performance.clear()
            self.regime_transitions.clear()
            self.feature_drifts.clear()
            
            # Remove files
            for file_path in [self.performance_path, self.regime_path, self.drift_path]:
                if file_path.exists():
                    file_path.unlink()
            
            self.logger.info("Learning data reset successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting learning data: {e}")
            return False


# Self-test function
def _self_test():
    """Basic self-test of the SelfLearning module"""
    print("Running SelfLearning self-test...")
    
    learning = SelfLearning()
    
    # Test trade recording
    learning.record_trade_outcome("test_strategy", {"pnl": 100.0, "consecutive_wins": 1})
    learning.record_trade_outcome("test_strategy", {"pnl": -50.0, "consecutive_losses": 1})
    
    # Test regime transition
    learning.record_regime_transition("TRENDING", "RANGE", 0.8, {"vix": 15.0, "volume": "high"})
    
    # Test feature drift
    features = {"rsi": 0.6, "macd": 0.3}
    importance = {"rsi": 0.8, "macd": 0.2}
    drifts = learning.analyze_feature_drift(features, importance)
    
    # Test threshold suggestions
    suggestions = learning.suggest_strategy_threshold_adjustments()
    
    # Test weekly job
    weekly_results = learning.run_weekly_learning_job()
    
    # Test status
    status = learning.get_learning_status()
    
    print(f"✓ Self-test completed. Status: {status['data_quality']}")
    print(f"  Strategies tracked: {status['strategies_tracked']}")
    print(f"  Regime transitions: {status['regime_transitions_recorded']}")
    print(f"  Features monitored: {status['features_monitored']}")


if __name__ == "__main__":
    _self_test()
