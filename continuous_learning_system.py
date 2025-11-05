"""
Real-Time Training System - Continuous Learning and Adaptation
Learns from every bet outcome to improve predictions and strategies
GOAL: MAXIMIZE LEARNING TO MAXIMIZE PROFITS
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
import asyncio
from collections import deque, defaultdict
import pickle
import json
import os
from dataclasses import dataclass, asdict
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class TrainingExample:
    """Single training example from bet outcome"""
    
    # Features (inputs)
    game_state: Dict
    predictions: List[Dict]
    strategy_used: str
    decision_confidence: float
    bet_amount: float
    multiplier: float
    side: str
    target_roll: float
    
    # Market conditions
    trend_indicators: Dict
    volatility_metrics: Dict
    entropy_analysis: Dict
    
    # Outcome (target)
    actual_roll: float
    bet_won: bool
    profit_loss: float
    
    # Metadata
    timestamp: datetime
    decision_id: str
    session_id: str

@dataclass
class ModelPerformance:
    """Track model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int
    recent_accuracy: float  # Last 100 predictions
    profit_correlation: float  # How well predicts profitable bets
    last_updated: datetime

class ContinuousLearningSystem:
    """Real-time learning system for betting optimization"""
    
    def __init__(self, model_save_path: str = "./models"):
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        # Training data storage
        self.training_examples = deque(maxlen=50000)  # Keep last 50k examples
        self.recent_outcomes = deque(maxlen=1000)
        
        # Models for different aspects
        self.models = {
            'roll_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
            'profit_predictor': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'confidence_calibrator': LogisticRegression(random_state=42),
            'strategy_selector': RandomForestClassifier(n_estimators=50, random_state=42),
            'volatility_predictor': LinearRegression()
        }
        
        # Model performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Scalers for numerical features
        self.scalers = {
            'features': StandardScaler(),
            'volatility': StandardScaler()
        }
        
        # Learning configuration
        self.min_samples_for_training = 100
        self.retrain_frequency = 50  # Retrain every 50 new examples
        self.performance_window = 100  # Window for recent performance
        
        # Load existing models if available
        self._load_models()
        
        logging.info("Continuous Learning System initialized")
    
    def add_training_example(self, example: TrainingExample):
        """Add new training example from bet outcome"""
        
        self.training_examples.append(example)
        self.recent_outcomes.append({
            'won': example.bet_won,
            'profit': example.profit_loss,
            'confidence': example.decision_confidence,
            'timestamp': example.timestamp
        })
        
        # Check if we should retrain models
        if len(self.training_examples) % self.retrain_frequency == 0:
            asyncio.create_task(self._retrain_models())
        
        logging.debug(f"Added training example: {example.decision_id}")
    
    async def _retrain_models(self):
        """Retrain all models with latest data"""
        
        if len(self.training_examples) < self.min_samples_for_training:
            logging.info("Insufficient data for training")
            return
        
        try:
            logging.info("Starting model retraining...")
            
            # Prepare training data
            features, targets = self._prepare_training_data()
            
            # Train each model
            await self._train_roll_predictor(features, targets)
            await self._train_profit_predictor(features, targets)
            await self._train_confidence_calibrator(features, targets)
            await self._train_strategy_selector(features, targets)
            await self._train_volatility_predictor(features, targets)
            
            # Update model performance metrics
            self._update_model_performance(features, targets)
            
            # Save updated models
            self._save_models()
            
            logging.info("Model retraining completed successfully")
            
        except Exception as e:
            logging.error(f"Model retraining failed: {e}")
    
    def _prepare_training_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Prepare training data from examples"""
        
        features_list = []
        targets = {
            'roll_outcome': [],  # Binary: over/under 50
            'exact_roll': [],    # Continuous: actual roll value
            'profit': [],        # Continuous: profit/loss
            'bet_won': [],       # Binary: bet won/lost
            'strategy': []       # Categorical: strategy used
        }
        
        for example in self.training_examples:
            # Extract features
            feature_row = self._extract_features(example)
            features_list.append(feature_row)
            
            # Extract targets
            targets['roll_outcome'].append(1 if example.actual_roll > 50 else 0)
            targets['exact_roll'].append(example.actual_roll)
            targets['profit'].append(example.profit_loss)
            targets['bet_won'].append(1 if example.bet_won else 0)
            targets['strategy'].append(example.strategy_used)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        return features_df, targets
    
    def _extract_features(self, example: TrainingExample) -> Dict:
        """Extract features from training example"""
        
        features = {}
        
        # Game state features
        features['server_seed_hash'] = hash(example.game_state.get('server_seed', '')) % 10000
        features['client_seed_hash'] = hash(example.game_state.get('client_seed', '')) % 10000
        features['nonce'] = example.game_state.get('nonce', 0)
        features['last_roll'] = example.game_state.get('last_roll', 50)
        features['api_latency'] = example.game_state.get('api_latency', 0)
        
        # Prediction features
        if example.predictions:
            pred_values = [p.get('prediction', 0.5) for p in example.predictions]
            conf_values = [p.get('confidence', 0.5) for p in example.predictions]
            
            features['pred_mean'] = np.mean(pred_values)
            features['pred_std'] = np.std(pred_values)
            features['pred_max'] = np.max(pred_values)
            features['pred_min'] = np.min(pred_values)
            features['conf_mean'] = np.mean(conf_values)
            features['conf_std'] = np.std(conf_values)
        else:
            features.update({
                'pred_mean': 0.5, 'pred_std': 0, 'pred_max': 0.5,
                'pred_min': 0.5, 'conf_mean': 0.5, 'conf_std': 0
            })
        
        # Decision features
        features['decision_confidence'] = example.decision_confidence
        features['bet_amount'] = example.bet_amount
        features['multiplier'] = example.multiplier
        features['side_over'] = 1 if example.side == 'over' else 0
        features['target_roll'] = example.target_roll
        
        # Trend features
        if example.trend_indicators and not example.trend_indicators.get('insufficient_data'):
            features['recent_average'] = example.trend_indicators.get('recent_average', 50)
            features['trend_slope'] = example.trend_indicators.get('trend_slope', 0)
            features['above_50_ratio'] = example.trend_indicators.get('above_50_ratio', 0.5)
            features['momentum_score'] = example.trend_indicators.get('momentum_score', 0)
        else:
            features.update({
                'recent_average': 50, 'trend_slope': 0,
                'above_50_ratio': 0.5, 'momentum_score': 0
            })
        
        # Volatility features
        if example.volatility_metrics and not example.volatility_metrics.get('insufficient_data'):
            features['session_volatility'] = example.volatility_metrics.get('session_volatility', 0.5)
            features['vol_percentile'] = example.volatility_metrics.get('vol_percentile', 50)
            features['short_term_vol'] = example.volatility_metrics.get('short_term_vol', 0.5)
        else:
            features.update({
                'session_volatility': 0.5, 'vol_percentile': 50, 'short_term_vol': 0.5
            })
        
        # Entropy features
        if example.entropy_analysis and not example.entropy_analysis.get('insufficient_data'):
            features['shannon_entropy'] = example.entropy_analysis.get('shannon_entropy', 1.0)
            features['randomness_score'] = example.entropy_analysis.get('randomness_score', 0.5)
        else:
            features.update({'shannon_entropy': 1.0, 'randomness_score': 0.5})
        
        # Time features
        features['hour_of_day'] = example.timestamp.hour
        features['day_of_week'] = example.timestamp.weekday()
        features['minute_of_hour'] = example.timestamp.minute
        
        return features
    
    async def _train_roll_predictor(self, features: pd.DataFrame, targets: Dict):
        """Train model to predict over/under 50"""
        
        X = features.copy()
        y = targets['roll_outcome']
        
        # Scale features
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Train model
        self.models['roll_predictor'].fit(X_scaled, y)
        
        logging.debug("Roll predictor retrained")
    
    async def _train_profit_predictor(self, features: pd.DataFrame, targets: Dict):
        """Train model to predict profitable bets"""
        
        X = features.copy()
        y = [1 if profit > 0 else 0 for profit in targets['profit']]
        
        # Scale features
        X_scaled = self.scalers['features'].transform(X)
        
        # Train model
        self.models['profit_predictor'].fit(X_scaled, y)
        
        logging.debug("Profit predictor retrained")
    
    async def _train_confidence_calibrator(self, features: pd.DataFrame, targets: Dict):
        """Train model to calibrate confidence scores"""
        
        # Features: original confidence + context features
        confidence_features = features[['decision_confidence', 'pred_mean', 'conf_mean', 
                                      'session_volatility', 'shannon_entropy']].copy()
        
        y = targets['bet_won']
        
        # Train calibration model
        self.models['confidence_calibrator'].fit(confidence_features, y)
        
        logging.debug("Confidence calibrator retrained")
    
    async def _train_strategy_selector(self, features: pd.DataFrame, targets: Dict):
        """Train model to select optimal strategy"""
        
        # Features for strategy selection
        strategy_features = features[['pred_mean', 'session_volatility', 'trend_slope',
                                    'above_50_ratio', 'shannon_entropy', 'vol_percentile']].copy()
        
        y = targets['strategy']
        
        # Only train if we have multiple strategies
        unique_strategies = set(y)
        if len(unique_strategies) > 1:
            # Encode strategies as numbers
            strategy_mapping = {strategy: i for i, strategy in enumerate(unique_strategies)}
            y_encoded = [strategy_mapping[strategy] for strategy in y]
            
            self.models['strategy_selector'].fit(strategy_features, y_encoded)
            
            # Store mapping for later use
            self.strategy_mapping = strategy_mapping
            self.reverse_strategy_mapping = {v: k for k, v in strategy_mapping.items()}
        
        logging.debug("Strategy selector retrained")
    
    async def _train_volatility_predictor(self, features: pd.DataFrame, targets: Dict):
        """Train model to predict future volatility"""
        
        # Features for volatility prediction
        vol_features = features[['session_volatility', 'short_term_vol', 'trend_slope',
                               'shannon_entropy', 'momentum_score']].copy()
        
        # Target: next period volatility (we'll approximate this)
        # For now, use current session volatility as proxy
        y = features['session_volatility'].values
        
        # Scale features
        vol_features_scaled = self.scalers['volatility'].fit_transform(vol_features)
        
        # Train model
        self.models['volatility_predictor'].fit(vol_features_scaled, y)
        
        logging.debug("Volatility predictor retrained")
    
    def _update_model_performance(self, features: pd.DataFrame, targets: Dict):
        """Update performance metrics for all models"""
        
        try:
            # Prepare test data (last 20% of samples)
            split_idx = int(len(features) * 0.8)
            X_test = features.iloc[split_idx:]
            
            # Test roll predictor
            X_test_scaled = self.scalers['features'].transform(X_test)
            y_test_roll = targets['roll_outcome'][split_idx:]
            
            if len(X_test) > 10:  # Minimum samples for meaningful metrics
                y_pred_roll = self.models['roll_predictor'].predict(X_test_scaled)
                
                self.model_performance['roll_predictor'] = ModelPerformance(
                    model_name='roll_predictor',
                    accuracy=accuracy_score(y_test_roll, y_pred_roll),
                    precision=precision_score(y_test_roll, y_pred_roll, average='weighted'),
                    recall=recall_score(y_test_roll, y_pred_roll, average='weighted'),
                    f1_score=f1_score(y_test_roll, y_pred_roll, average='weighted'),
                    total_samples=len(features),
                    recent_accuracy=self._calculate_recent_accuracy('roll_predictor'),
                    profit_correlation=self._calculate_profit_correlation('roll_predictor'),
                    last_updated=datetime.now()
                )
                
                # Update feature importance
                if hasattr(self.models['roll_predictor'], 'feature_importances_'):
                    self.feature_importance['roll_predictor'] = dict(
                        zip(features.columns, self.models['roll_predictor'].feature_importances_)
                    )
            
            logging.debug("Model performance metrics updated")
            
        except Exception as e:
            logging.error(f"Failed to update model performance: {e}")
    
    def _calculate_recent_accuracy(self, model_name: str) -> float:
        """Calculate accuracy on recent predictions"""
        if len(self.recent_outcomes) < 10:
            return 0.5
        
        # This would require storing model predictions, simplified for now
        recent_wins = sum(1 for outcome in list(self.recent_outcomes)[-50:] if outcome['won'])
        return recent_wins / min(50, len(self.recent_outcomes))
    
    def _calculate_profit_correlation(self, model_name: str) -> float:
        """Calculate correlation between model confidence and profit"""
        if len(self.recent_outcomes) < 10:
            return 0.0
        
        recent = list(self.recent_outcomes)[-100:]
        confidences = [outcome['confidence'] for outcome in recent]
        profits = [outcome['profit'] for outcome in recent]
        
        if len(set(confidences)) > 1 and len(set(profits)) > 1:
            correlation = np.corrcoef(confidences, profits)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    async def get_enhanced_predictions(self, context: Dict) -> Dict:
        """Get enhanced predictions using trained models"""
        
        if len(self.training_examples) < self.min_samples_for_training:
            return {'enhanced_predictions': None, 'reason': 'insufficient_training_data'}
        
        try:
            # Extract features from current context
            features = self._extract_features_from_context(context)
            features_df = pd.DataFrame([features])
            
            # Scale features
            features_scaled = self.scalers['features'].transform(features_df)
            
            # Get predictions from trained models
            predictions = {}
            
            # Roll prediction
            if 'roll_predictor' in self.models:
                roll_prob = self.models['roll_predictor'].predict_proba(features_scaled)[0]
                predictions['roll_over_50_probability'] = roll_prob[1] if len(roll_prob) > 1 else 0.5
            
            # Profit prediction
            if 'profit_predictor' in self.models:
                profit_prob = self.models['profit_predictor'].predict_proba(features_scaled)[0]
                predictions['profit_probability'] = profit_prob[1] if len(profit_prob) > 1 else 0.5
            
            # Confidence calibration
            if 'confidence_calibrator' in self.models:
                conf_features = features_df[['decision_confidence', 'pred_mean', 'conf_mean',
                                           'session_volatility', 'shannon_entropy']].fillna(0)
                calibrated_conf = self.models['confidence_calibrator'].predict_proba(conf_features)[0]
                predictions['calibrated_confidence'] = calibrated_conf[1] if len(calibrated_conf) > 1 else 0.5
            
            # Strategy recommendation
            if 'strategy_selector' in self.models and hasattr(self, 'reverse_strategy_mapping'):
                strategy_features = features_df[['pred_mean', 'session_volatility', 'trend_slope',
                                               'above_50_ratio', 'shannon_entropy', 'vol_percentile']].fillna(0)
                strategy_pred = self.models['strategy_selector'].predict(strategy_features)[0]
                predictions['recommended_strategy'] = self.reverse_strategy_mapping.get(strategy_pred, 'unknown')
            
            # Volatility prediction
            if 'volatility_predictor' in self.models:
                vol_features = features_df[['session_volatility', 'short_term_vol', 'trend_slope',
                                          'shannon_entropy', 'momentum_score']].fillna(0)
                vol_features_scaled = self.scalers['volatility'].transform(vol_features)
                predicted_volatility = self.models['volatility_predictor'].predict(vol_features_scaled)[0]
                predictions['predicted_volatility'] = predicted_volatility
            
            return {
                'enhanced_predictions': predictions,
                'model_confidence': self._calculate_ensemble_confidence(predictions),
                'feature_importance': self.feature_importance.get('roll_predictor', {}),
                'models_used': list(self.models.keys())
            }
            
        except Exception as e:
            logging.error(f"Enhanced prediction failed: {e}")
            return {'enhanced_predictions': None, 'error': str(e)}
    
    def _extract_features_from_context(self, context: Dict) -> Dict:
        """Extract features from current betting context"""
        
        # This should match the feature extraction in _extract_features
        features = {}
        
        # Game state
        game_state = context.get('game_state', {})
        features['server_seed_hash'] = hash(game_state.get('server_seed', '')) % 10000
        features['client_seed_hash'] = hash(game_state.get('client_seed', '')) % 10000
        features['nonce'] = game_state.get('nonce', 0)
        features['last_roll'] = game_state.get('last_roll', 50)
        features['api_latency'] = game_state.get('api_latency', 0)
        
        # Predictions (if available)
        predictions = context.get('predictions', [])
        if predictions:
            pred_values = [p.get('prediction', 0.5) for p in predictions]
            conf_values = [p.get('confidence', 0.5) for p in predictions]
            
            features['pred_mean'] = np.mean(pred_values)
            features['pred_std'] = np.std(pred_values)
            features['pred_max'] = np.max(pred_values)
            features['pred_min'] = np.min(pred_values)
            features['conf_mean'] = np.mean(conf_values)
            features['conf_std'] = np.std(conf_values)
        else:
            features.update({
                'pred_mean': 0.5, 'pred_std': 0, 'pred_max': 0.5,
                'pred_min': 0.5, 'conf_mean': 0.5, 'conf_std': 0
            })
        
        # Default decision features (will be updated when used)
        features['decision_confidence'] = context.get('decision_confidence', 50)
        features['bet_amount'] = context.get('bet_amount', 0)
        features['multiplier'] = context.get('multiplier', 2.0)
        features['side_over'] = 1  # Default
        features['target_roll'] = context.get('target_roll', 50)
        
        # Trend indicators
        trend_indicators = context.get('trend_indicators', {})
        if not trend_indicators.get('insufficient_data'):
            features['recent_average'] = trend_indicators.get('recent_average', 50)
            features['trend_slope'] = trend_indicators.get('trend_slope', 0)
            features['above_50_ratio'] = trend_indicators.get('above_50_ratio', 0.5)
            features['momentum_score'] = trend_indicators.get('momentum_score', 0)
        else:
            features.update({
                'recent_average': 50, 'trend_slope': 0,
                'above_50_ratio': 0.5, 'momentum_score': 0
            })
        
        # Volatility metrics
        volatility_metrics = context.get('volatility_metrics', {})
        if not volatility_metrics.get('insufficient_data'):
            features['session_volatility'] = volatility_metrics.get('session_volatility', 0.5)
            features['vol_percentile'] = volatility_metrics.get('vol_percentile', 50)
            features['short_term_vol'] = volatility_metrics.get('short_term_vol', 0.5)
        else:
            features.update({
                'session_volatility': 0.5, 'vol_percentile': 50, 'short_term_vol': 0.5
            })
        
        # Entropy analysis
        entropy_analysis = context.get('entropy_analysis', {})
        if not entropy_analysis.get('insufficient_data'):
            features['shannon_entropy'] = entropy_analysis.get('shannon_entropy', 1.0)
            features['randomness_score'] = entropy_analysis.get('randomness_score', 0.5)
        else:
            features.update({'shannon_entropy': 1.0, 'randomness_score': 0.5})
        
        # Time features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['minute_of_hour'] = now.minute
        
        return features
    
    def _calculate_ensemble_confidence(self, predictions: Dict) -> float:
        """Calculate overall confidence from ensemble predictions"""
        
        confidences = []
        
        # Collect confidence scores
        if 'roll_over_50_probability' in predictions:
            # Convert probability to confidence (distance from 0.5)
            prob = predictions['roll_over_50_probability']
            confidence = abs(prob - 0.5) * 2  # Scale to 0-1
            confidences.append(confidence)
        
        if 'profit_probability' in predictions:
            confidences.append(predictions['profit_probability'])
        
        if 'calibrated_confidence' in predictions:
            confidences.append(predictions['calibrated_confidence'])
        
        if not confidences:
            return 0.5
        
        # Average confidence with weights
        weights = [0.4, 0.4, 0.2][:len(confidences)]  # Adjust based on importance
        weighted_confidence = sum(c * w for c, w in zip(confidences, weights)) / sum(weights)
        
        return weighted_confidence
    
    def get_learning_insights(self) -> Dict:
        """Get insights from the learning system"""
        
        insights = {
            'training_data': {
                'total_examples': len(self.training_examples),
                'recent_examples': len([ex for ex in self.training_examples 
                                      if ex.timestamp > datetime.now() - timedelta(hours=24)]),
                'win_rate': self._calculate_overall_win_rate(),
                'avg_profit': self._calculate_avg_profit()
            },
            'model_performance': {name: asdict(perf) for name, perf in self.model_performance.items()},
            'feature_importance': self.feature_importance,
            'learning_trends': self._analyze_learning_trends(),
            'recommendations': self._generate_learning_recommendations()
        }
        
        return insights
    
    def _calculate_overall_win_rate(self) -> float:
        """Calculate overall win rate from training data"""
        if not self.training_examples:
            return 0.0
        
        wins = sum(1 for ex in self.training_examples if ex.bet_won)
        return wins / len(self.training_examples)
    
    def _calculate_avg_profit(self) -> float:
        """Calculate average profit per bet"""
        if not self.training_examples:
            return 0.0
        
        total_profit = sum(ex.profit_loss for ex in self.training_examples)
        return total_profit / len(self.training_examples)
    
    def _analyze_learning_trends(self) -> Dict:
        """Analyze trends in learning performance"""
        
        if len(self.training_examples) < 100:
            return {'status': 'insufficient_data'}
        
        # Analyze performance over time
        recent_examples = list(self.training_examples)[-100:]
        older_examples = list(self.training_examples)[-200:-100] if len(self.training_examples) >= 200 else []
        
        recent_win_rate = sum(1 for ex in recent_examples if ex.bet_won) / len(recent_examples)
        recent_avg_profit = sum(ex.profit_loss for ex in recent_examples) / len(recent_examples)
        
        trends = {
            'recent_win_rate': recent_win_rate,
            'recent_avg_profit': recent_avg_profit,
        }
        
        if older_examples:
            older_win_rate = sum(1 for ex in older_examples if ex.bet_won) / len(older_examples)
            older_avg_profit = sum(ex.profit_loss for ex in older_examples) / len(older_examples)
            
            trends['win_rate_trend'] = 'improving' if recent_win_rate > older_win_rate else 'declining'
            trends['profit_trend'] = 'improving' if recent_avg_profit > older_avg_profit else 'declining'
            trends['win_rate_change'] = recent_win_rate - older_win_rate
            trends['profit_change'] = recent_avg_profit - older_avg_profit
        
        return trends
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations based on learning analysis"""
        
        recommendations = []
        
        if len(self.training_examples) < 500:
            recommendations.append("Collect more training data for better model performance")
        
        if 'roll_predictor' in self.model_performance:
            accuracy = self.model_performance['roll_predictor'].accuracy
            if accuracy < 0.55:
                recommendations.append("Roll predictor accuracy is low - consider feature engineering")
            elif accuracy > 0.65:
                recommendations.append("Roll predictor performing well - consider increasing bet sizes")
        
        # Check recent performance
        if len(self.recent_outcomes) > 20:
            recent_win_rate = sum(1 for outcome in list(self.recent_outcomes)[-20:] if outcome['won']) / 20
            if recent_win_rate < 0.4:
                recommendations.append("Recent performance is poor - consider reducing bet sizes")
            elif recent_win_rate > 0.6:
                recommendations.append("Recent performance is good - consider more aggressive betting")
        
        # Feature importance recommendations
        if 'roll_predictor' in self.feature_importance:
            top_features = sorted(self.feature_importance['roll_predictor'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(f"Most important features: {', '.join([f[0] for f in top_features])}")
        
        return recommendations
    
    def _save_models(self):
        """Save trained models to disk"""
        
        try:
            # Save sklearn models
            for name, model in self.models.items():
                if hasattr(model, 'fit'):  # Check if model is trained
                    model_path = os.path.join(self.model_save_path, f"{name}.joblib")
                    joblib.dump(model, model_path)
            
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = os.path.join(self.model_save_path, f"scaler_{name}.joblib")
                joblib.dump(scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'model_performance': {k: asdict(v) for k, v in self.model_performance.items()},
                'feature_importance': self.feature_importance,
                'training_examples_count': len(self.training_examples),
                'last_saved': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.model_save_path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Failed to save models: {e}")
    
    def _load_models(self):
        """Load trained models from disk"""
        
        try:
            # Load sklearn models
            for name in self.models.keys():
                model_path = os.path.join(self.model_save_path, f"{name}.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    logging.debug(f"Loaded model: {name}")
            
            # Load scalers
            for name in self.scalers.keys():
                scaler_path = os.path.join(self.model_save_path, f"scaler_{name}.joblib")
                if os.path.exists(scaler_path):
                    self.scalers[name] = joblib.load(scaler_path)
                    logging.debug(f"Loaded scaler: {name}")
            
            # Load metadata
            metadata_path = os.path.join(self.model_save_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Restore model performance
                for name, perf_data in metadata.get('model_performance', {}).items():
                    perf_data['last_updated'] = datetime.fromisoformat(perf_data['last_updated'])
                    self.model_performance[name] = ModelPerformance(**perf_data)
                
                self.feature_importance = metadata.get('feature_importance', {})
                
                logging.info(f"Loaded models with {metadata.get('training_examples_count', 0)} training examples")
            
        except Exception as e:
            logging.error(f"Failed to load models: {e}")


# Global learning system instance
learning_system = None

def initialize_learning_system(model_save_path: str = "./models"):
    """Initialize the continuous learning system"""
    global learning_system
    learning_system = ContinuousLearningSystem(model_save_path)
    logging.info("Continuous Learning System initialized and ready for profit optimization!")
    return learning_system

def get_learning_system():
    """Get the global learning system instance"""
    global learning_system
    return learning_system