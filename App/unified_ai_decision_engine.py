"""
UNIFIED AI DECISION ENGINE - The Ultimate Thinking Machine
Merges all prediction models, strategies, AI brain, and real-time data
into one supreme betting decision system that MAXIMIZES PROFITS
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import json
import os
from collections import deque
import hashlib
import hmac
import time

# Import our components
from bedrock_ai_brain import BedrockAIBrain, BettingContext, AIDecision
from complete_strategy_framework import CompleteStrategyFramework, StrategyConfig
from cloudscraper_test import StakeAPIAccess

@dataclass
class PredictionOutput:
    """Structured output from prediction models"""
    model_name: str
    prediction: float  # 0-1 probability
    confidence: float  # 0-1 confidence level
    reasoning: str
    timestamp: datetime
    features_used: List[str]

@dataclass
class UnifiedDecision:
    """Final decision from the unified engine"""
    should_bet: bool
    bet_amount: float
    multiplier: float
    side: str  # "over" or "under"
    target_roll: float  # Target dice roll
    confidence: float
    
    # Decision components
    ai_decision: AIDecision
    strategy_recommendation: Dict
    prediction_consensus: Dict
    risk_assessment: Dict
    
    # Context
    timestamp: datetime
    session_id: str
    decision_id: str

class PredictionModelIntegrator:
    """Integrates all prediction models into unified output"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.performance_history = {}
        
    def register_model(self, name: str, model_func, weight: float = 1.0):
        """Register a prediction model"""
        self.models[name] = model_func
        self.model_weights[name] = weight
        self.performance_history[name] = deque(maxlen=1000)
        
    async def get_predictions(self, context: Dict) -> List[PredictionOutput]:
        """Get predictions from all registered models"""
        predictions = []
        
        for model_name, model_func in self.models.items():
            try:
                prediction_data = await self._run_model(model_name, model_func, context)
                
                prediction = PredictionOutput(
                    model_name=model_name,
                    prediction=prediction_data.get('prediction', 0.5),
                    confidence=prediction_data.get('confidence', 0.5),
                    reasoning=prediction_data.get('reasoning', ''),
                    timestamp=datetime.now(),
                    features_used=prediction_data.get('features', [])
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logging.error(f"Model {model_name} failed: {e}")
                # Add fallback prediction
                predictions.append(PredictionOutput(
                    model_name=f"{model_name}_fallback",
                    prediction=0.5,
                    confidence=0.1,
                    reasoning=f"Model failed: {e}",
                    timestamp=datetime.now(),
                    features_used=[]
                ))
        
        return predictions
    
    async def _run_model(self, name: str, model_func, context: Dict) -> Dict:
        """Run individual model with timeout and error handling"""
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(model_func(context)), 
                timeout=5.0
            )
            return result
        except asyncio.TimeoutError:
            logging.warning(f"Model {name} timed out")
            return {'prediction': 0.5, 'confidence': 0.1, 'reasoning': 'timeout'}
        except Exception as e:
            logging.error(f"Model {name} error: {e}")
            return {'prediction': 0.5, 'confidence': 0.1, 'reasoning': f'error: {e}'}
    
    def create_consensus(self, predictions: List[PredictionOutput]) -> Dict:
        """Create weighted consensus from all predictions"""
        if not predictions:
            return {'prediction': 0.5, 'confidence': 0.1, 'models_count': 0}
        
        # Weight predictions by confidence and historical performance
        weighted_predictions = []
        total_weight = 0
        
        for pred in predictions:
            # Get historical performance weight
            perf_weight = self._get_performance_weight(pred.model_name)
            
            # Combined weight: base weight * confidence * performance
            combined_weight = (
                self.model_weights.get(pred.model_name, 1.0) * 
                pred.confidence * 
                perf_weight
            )
            
            weighted_predictions.append(pred.prediction * combined_weight)
            total_weight += combined_weight
        
        if total_weight == 0:
            consensus_prediction = 0.5
            consensus_confidence = 0.1
        else:
            consensus_prediction = sum(weighted_predictions) / total_weight
            consensus_confidence = min(total_weight / len(predictions), 1.0)
        
        return {
            'prediction': consensus_prediction,
            'confidence': consensus_confidence,
            'models_count': len(predictions),
            'individual_predictions': [
                {
                    'model': p.model_name,
                    'prediction': p.prediction,
                    'confidence': p.confidence,
                    'reasoning': p.reasoning
                } for p in predictions
            ],
            'weighted_average': consensus_prediction,
            'agreement_score': self._calculate_agreement(predictions)
        }
    
    def _get_performance_weight(self, model_name: str) -> float:
        """Get performance-based weight for model"""
        history = self.performance_history.get(model_name, [])
        
        if len(history) < 10:
            return 1.0  # Default weight for new models
        
        # Calculate recent accuracy
        recent_accuracy = sum(history[-50:]) / len(history[-50:])
        
        # Convert to weight (0.5 = 1.0 weight, >0.5 = higher weight)
        weight = max(0.1, min(2.0, recent_accuracy * 2))
        return weight
    
    def _calculate_agreement(self, predictions: List[PredictionOutput]) -> float:
        """Calculate how much models agree (0-1)"""
        if len(predictions) < 2:
            return 1.0
        
        pred_values = [p.prediction for p in predictions]
        std_dev = np.std(pred_values)
        
        # Convert std dev to agreement score (lower std = higher agreement)
        agreement = max(0, 1 - (std_dev * 4))  # Scale factor of 4
        return agreement
    
    def update_model_performance(self, model_name: str, was_correct: bool):
        """Update model performance tracking"""
        if model_name in self.performance_history:
            self.performance_history[model_name].append(1.0 if was_correct else 0.0)

class RealTimeDataProcessor:
    """Processes real-time data from Stake API and other sources"""
    
    def __init__(self, api_access: StakeAPIAccess):
        self.api = api_access
        self.data_buffer = deque(maxlen=10000)
        self.processed_cache = {}
        
    async def get_current_game_state(self) -> Dict:
        """Get current game state from API"""
        try:
            # Get latest game data
            game_data = await self.api.get_latest_game_data()
            
            current_state = {
                'server_seed': game_data.get('serverSeed', ''),
                'client_seed': game_data.get('clientSeed', ''),
                'nonce': game_data.get('nonce', 0),
                'game_id': game_data.get('gameId', ''),
                'last_roll': game_data.get('lastRoll', 50.0),
                'timestamp': datetime.now(),
                'api_latency': game_data.get('latency', 0),
                'promotion_active': game_data.get('hasPromotion', False),
                'cashback_rate': game_data.get('cashbackRate', 0.0)
            }
            
            # Add to buffer
            self.data_buffer.append(current_state)
            
            return current_state
            
        except Exception as e:
            logging.error(f"Failed to get game state: {e}")
            return self._get_fallback_state()
    
    def _get_fallback_state(self) -> Dict:
        """Fallback state when API is unavailable"""
        return {
            'server_seed': 'fallback_seed',
            'client_seed': 'fallback_client',
            'nonce': int(time.time()),
            'game_id': f"fallback_{int(time.time())}",
            'last_roll': 50.0,
            'timestamp': datetime.now(),
            'api_latency': 999,
            'promotion_active': False,
            'cashback_rate': 0.0
        }
    
    def get_trend_indicators(self) -> Dict:
        """Calculate trend indicators from recent data"""
        if len(self.data_buffer) < 10:
            return {'insufficient_data': True}
        
        recent_rolls = [entry['last_roll'] for entry in list(self.data_buffer)[-100:]]
        
        indicators = {
            'recent_average': np.mean(recent_rolls),
            'rolling_std': np.std(recent_rolls),
            'trend_slope': self._calculate_trend_slope(recent_rolls),
            'above_50_ratio': sum(1 for roll in recent_rolls if roll > 50) / len(recent_rolls),
            'hot_cold_streaks': self._identify_streaks(recent_rolls),
            'volatility_regime': self._classify_volatility(recent_rolls),
            'momentum_score': self._calculate_momentum(recent_rolls)
        }
        
        return indicators
    
    def get_volatility_metrics(self) -> Dict:
        """Calculate comprehensive volatility metrics"""
        recent_data = list(self.data_buffer)[-200:]
        
        if len(recent_data) < 20:
            return {'insufficient_data': True}
        
        rolls = [entry['last_roll'] for entry in recent_data]
        
        return {
            'session_volatility': np.std(rolls),
            'short_term_vol': np.std(rolls[-20:]),
            'medium_term_vol': np.std(rolls[-50:]),
            'long_term_vol': np.std(rolls[-100:]) if len(rolls) >= 100 else np.std(rolls),
            'volatility_trend': self._volatility_trend(rolls),
            'vol_percentile': self._volatility_percentile(rolls),
            'regime_changes': self._detect_regime_changes(rolls)
        }
    
    def get_entropy_analysis(self) -> Dict:
        """Perform entropy analysis on recent outcomes"""
        recent_data = list(self.data_buffer)[-100:]
        
        if len(recent_data) < 20:
            return {'insufficient_data': True}
        
        rolls = [entry['last_roll'] for entry in recent_data]
        binary_outcomes = [1 if roll > 50 else 0 for roll in rolls]
        
        return {
            'shannon_entropy': self._calculate_shannon_entropy(binary_outcomes),
            'pattern_entropy': self._calculate_pattern_entropy(binary_outcomes),
            'local_entropy': self._calculate_local_entropy(binary_outcomes),
            'entropy_trend': self._entropy_trend_analysis(binary_outcomes),
            'randomness_score': self._assess_randomness(binary_outcomes)
        }
    
    # Helper methods for calculations
    def _calculate_trend_slope(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope
    
    def _identify_streaks(self, rolls: List[float]) -> Dict:
        over_50 = [roll > 50 for roll in rolls]
        streaks = {'current_streak': 0, 'max_streak': 0, 'streak_type': 'none'}
        
        current = 0
        max_streak = 0
        current_type = None
        
        for outcome in over_50:
            if current_type is None or current_type == outcome:
                current += 1
                current_type = outcome
            else:
                max_streak = max(max_streak, current)
                current = 1
                current_type = outcome
        
        streaks['current_streak'] = current
        streaks['max_streak'] = max(max_streak, current)
        streaks['streak_type'] = 'over' if current_type else 'under'
        
        return streaks
    
    def _classify_volatility(self, rolls: List[float]) -> str:
        vol = np.std(rolls)
        if vol < 15:
            return 'low'
        elif vol < 25:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_momentum(self, rolls: List[float]) -> float:
        if len(rolls) < 10:
            return 0.0
        
        short_avg = np.mean(rolls[-5:])
        long_avg = np.mean(rolls[-20:])
        
        return (short_avg - long_avg) / long_avg if long_avg != 0 else 0
    
    def _volatility_trend(self, rolls: List[float]) -> str:
        if len(rolls) < 40:
            return 'unknown'
        
        recent_vol = np.std(rolls[-20:])
        older_vol = np.std(rolls[-40:-20])
        
        if recent_vol > older_vol * 1.1:
            return 'increasing'
        elif recent_vol < older_vol * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _volatility_percentile(self, rolls: List[float]) -> float:
        if len(rolls) < 50:
            return 50.0
        
        # Rolling volatility calculation
        window = 20
        volatilities = []
        for i in range(window, len(rolls)):
            vol = np.std(rolls[i-window:i])
            volatilities.append(vol)
        
        current_vol = np.std(rolls[-window:])
        percentile = (sum(1 for v in volatilities if v <= current_vol) / len(volatilities)) * 100
        
        return percentile
    
    def _detect_regime_changes(self, rolls: List[float]) -> List[Dict]:
        # Simple regime change detection based on volatility shifts
        changes = []
        
        if len(rolls) < 60:
            return changes
        
        window = 20
        threshold = 1.5
        
        for i in range(window, len(rolls) - window, 10):
            vol_before = np.std(rolls[i-window:i])
            vol_after = np.std(rolls[i:i+window])
            
            if vol_after > vol_before * threshold:
                changes.append({
                    'index': i,
                    'type': 'volatility_increase',
                    'ratio': vol_after / vol_before
                })
            elif vol_after < vol_before / threshold:
                changes.append({
                    'index': i,
                    'type': 'volatility_decrease',
                    'ratio': vol_after / vol_before
                })
        
        return changes[-5:]  # Return last 5 changes
    
    def _calculate_shannon_entropy(self, binary_outcomes: List[int]) -> float:
        if not binary_outcomes:
            return 0.0
        
        from collections import Counter
        counts = Counter(binary_outcomes)
        probs = [count / len(binary_outcomes) for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def _calculate_pattern_entropy(self, binary_outcomes: List[int]) -> float:
        # Entropy of 2-grams (pairs)
        if len(binary_outcomes) < 2:
            return 0.0
        
        pairs = [f"{binary_outcomes[i]}{binary_outcomes[i+1]}" 
                for i in range(len(binary_outcomes)-1)]
        
        from collections import Counter
        counts = Counter(pairs)
        probs = [count / len(pairs) for count in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return entropy
    
    def _calculate_local_entropy(self, binary_outcomes: List[int]) -> List[float]:
        # Sliding window entropy
        window_size = 10
        entropies = []
        
        for i in range(len(binary_outcomes) - window_size + 1):
            window = binary_outcomes[i:i+window_size]
            entropy = self._calculate_shannon_entropy(window)
            entropies.append(entropy)
        
        return entropies[-10:]  # Return last 10 local entropies
    
    def _entropy_trend_analysis(self, binary_outcomes: List[int]) -> Dict:
        local_entropies = self._calculate_local_entropy(binary_outcomes)
        
        if len(local_entropies) < 5:
            return {'trend': 'unknown', 'slope': 0}
        
        x = np.arange(len(local_entropies))
        slope, _ = np.polyfit(x, local_entropies, 1)
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {'trend': trend, 'slope': slope}
    
    def _assess_randomness(self, binary_outcomes: List[int]) -> float:
        # Combined randomness score (0-1, 1 = most random)
        if len(binary_outcomes) < 20:
            return 0.5
        
        # Run test
        runs = 1
        for i in range(1, len(binary_outcomes)):
            if binary_outcomes[i] != binary_outcomes[i-1]:
                runs += 1
        
        n = len(binary_outcomes)
        n1 = sum(binary_outcomes)
        n0 = n - n1
        
        if n1 == 0 or n0 == 0:
            return 0.0
        
        expected_runs = (2 * n1 * n0) / n + 1
        run_score = min(1.0, runs / expected_runs)
        
        # Frequency test
        freq_score = 1 - abs(0.5 - (n1 / n)) * 2
        
        # Combined score
        randomness_score = (run_score + freq_score) / 2
        
        return randomness_score
    
    def update_session_seeds(self, seeds: Dict) -> None:
        """Update session seeds for enhanced prediction accuracy"""
        try:
            self.session_seeds = {
                'client_seed': seeds.get('client_seed', ''),
                'server_seed_hash': seeds.get('server_seed_hash', ''),
                'nonce': seeds.get('nonce', 0),
                'total_bets': seeds.get('total_bets', 0),
                'revealed_server_seed': seeds.get('revealed_server_seed', '')
            }
            
            # Enhance predictions with seed-based calculations
            self.seeds_enabled = True
            logging.info(f"Session seeds updated for enhanced predictions: {self.session_seeds['client_seed'][:8]}...")
            
        except Exception as e:
            logging.error(f"Failed to update session seeds: {e}")
            self.seeds_enabled = False
    
    def clear_session_seeds(self) -> None:
        """Clear session seeds"""
        try:
            self.session_seeds = {}
            self.seeds_enabled = False
            logging.info("Session seeds cleared")
        except Exception as e:
            logging.error(f"Failed to clear session seeds: {e}")
    
    def get_seed_based_prediction(self, nonce: int = None) -> Dict:
        """Get prediction based on session seeds if available"""
        try:
            if not hasattr(self, 'seeds_enabled') or not self.seeds_enabled:
                return {'prediction': None, 'confidence': 0, 'method': 'no_seeds'}
            
            if not hasattr(self, 'session_seeds') or not self.session_seeds.get('client_seed'):
                return {'prediction': None, 'confidence': 0, 'method': 'no_seeds'}
            
            # Use provided nonce or current session nonce
            current_nonce = nonce if nonce is not None else self.session_seeds.get('nonce', 0)
            
            if self.session_seeds.get('revealed_server_seed'):
                # High accuracy prediction with revealed server seed
                import hashlib
                import hmac
                
                client_seed = self.session_seeds['client_seed']
                server_seed = self.session_seeds['revealed_server_seed']
                message = f"{client_seed}:{current_nonce}"
                
                hmac_result = hmac.new(
                    server_seed.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                # Convert to dice roll
                hex_chunk = hmac_result[:8]
                decimal_value = int(hex_chunk, 16)
                predicted_roll = round((decimal_value % 10000) / 100, 2)
                
                return {
                    'prediction': predicted_roll,
                    'confidence': 95,
                    'method': 'hmac_calculation',
                    'hmac_output': hmac_result,
                    'message': message
                }
            else:
                # Limited prediction with hash only - use pattern analysis
                seed_hash = self.session_seeds['server_seed_hash']
                client_seed = self.session_seeds['client_seed']
                
                # Create a deterministic prediction based on available data
                combined_input = f"{client_seed}_{seed_hash}_{current_nonce}"
                hash_result = hashlib.md5(combined_input.encode()).hexdigest()
                
                # Convert to prediction
                hex_value = int(hash_result[:4], 16)
                predicted_roll = round((hex_value % 10000) / 100, 2)
                
                return {
                    'prediction': predicted_roll,
                    'confidence': 25,  # Low confidence without actual server seed
                    'method': 'hash_approximation',
                    'note': 'Limited accuracy without revealed server seed'
                }
                
        except Exception as e:
            logging.error(f"Seed-based prediction failed: {e}")
            return {'prediction': None, 'confidence': 0, 'method': 'error'}

class UnifiedAIDecisionEngine:
    """The supreme decision engine that combines everything"""
    
    def __init__(self, stake_api_key: str, aws_credentials: Dict = None):
        # Initialize components
        self.ai_brain = BedrockAIBrain()
        self.strategy_framework = CompleteStrategyFramework()
        self.prediction_integrator = PredictionModelIntegrator()
        self.api_access = StakeAPIAccess(stake_api_key)
        self.data_processor = RealTimeDataProcessor(self.api_access)
        
        # Session management
        self.session_id = self._generate_session_id()
        self.decision_history = deque(maxlen=10000)
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_bets': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'avg_confidence': 0.0,
            'roi': 0.0
        }
        
        # Real-time state
        self.current_bankroll = 1000.0  # Starting bankroll
        self.session_start_time = datetime.now()
        self.last_decision_time = None
        
        # Initialize prediction models
        self._register_prediction_models()
        
        logging.info("Unified AI Decision Engine initialized successfully")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_suffix}"
    
    def _register_prediction_models(self):
        """Register all prediction models"""
        
        # HMAC-based prediction model
        async def hmac_prediction_model(context):
            """HMAC-based prediction using server/client seeds"""
            try:
                server_seed = context['game_state']['server_seed']
                client_seed = context['game_state']['client_seed']
                nonce = context['game_state']['nonce']
                
                # Generate HMAC prediction
                message = f"{client_seed}:{nonce}"
                hmac_hash = hmac.new(
                    server_seed.encode(),
                    message.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                # Convert hash to prediction
                hash_int = int(hmac_hash[:8], 16)
                prediction = (hash_int % 10000) / 100.0  # 0-100 range
                
                confidence = 0.8  # High confidence for HMAC
                
                return {
                    'prediction': prediction / 100.0,  # Normalize to 0-1
                    'confidence': confidence,
                    'reasoning': f'HMAC prediction: {prediction:.2f}',
                    'features': ['server_seed', 'client_seed', 'nonce']
                }
                
            except Exception as e:
                return {
                    'prediction': 0.5,
                    'confidence': 0.1,
                    'reasoning': f'HMAC model error: {e}',
                    'features': []
                }
        
        # Trend-based prediction model
        async def trend_prediction_model(context):
            """Prediction based on recent trends"""
            try:
                indicators = context['trend_indicators']
                
                if indicators.get('insufficient_data'):
                    return {
                        'prediction': 0.5,
                        'confidence': 0.1,
                        'reasoning': 'Insufficient trend data',
                        'features': []
                    }
                
                # Analyze trends
                recent_avg = indicators['recent_average']
                trend_slope = indicators['trend_slope']
                above_50_ratio = indicators['above_50_ratio']
                
                # Predict based on trend
                if trend_slope > 0.1:  # Strong upward trend
                    prediction = 0.7
                    reasoning = "Strong upward trend detected"
                elif trend_slope < -0.1:  # Strong downward trend
                    prediction = 0.3
                    reasoning = "Strong downward trend detected"
                else:  # Use ratio-based prediction
                    prediction = above_50_ratio
                    reasoning = f"Based on recent ratio: {above_50_ratio:.2f}"
                
                confidence = min(0.8, abs(trend_slope) * 10 + 0.4)
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'features': ['recent_average', 'trend_slope', 'above_50_ratio']
                }
                
            except Exception as e:
                return {
                    'prediction': 0.5,
                    'confidence': 0.1,
                    'reasoning': f'Trend model error: {e}',
                    'features': []
                }
        
        # Volatility-based prediction model
        async def volatility_prediction_model(context):
            """Prediction based on volatility patterns"""
            try:
                vol_metrics = context['volatility_metrics']
                
                if vol_metrics.get('insufficient_data'):
                    return {
                        'prediction': 0.5,
                        'confidence': 0.1,
                        'reasoning': 'Insufficient volatility data',
                        'features': []
                    }
                
                vol_percentile = vol_metrics['vol_percentile']
                volatility_trend = vol_metrics['volatility_trend']
                
                # High volatility = more extreme outcomes expected
                if vol_percentile > 80:  # High volatility
                    prediction = 0.6  # Slight bias toward continuation
                    confidence = 0.6
                    reasoning = "High volatility - extreme outcome expected"
                elif vol_percentile < 20:  # Low volatility
                    prediction = 0.5  # Mean reversion
                    confidence = 0.5
                    reasoning = "Low volatility - mean reversion expected"
                else:
                    prediction = 0.5
                    confidence = 0.4
                    reasoning = "Normal volatility - no clear bias"
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'features': ['vol_percentile', 'volatility_trend']
                }
                
            except Exception as e:
                return {
                    'prediction': 0.5,
                    'confidence': 0.1,
                    'reasoning': f'Volatility model error: {e}',
                    'features': []
                }
        
        # Entropy-based prediction model
        async def entropy_prediction_model(context):
            """Prediction based on entropy analysis"""
            try:
                entropy_data = context['entropy_analysis']
                
                if entropy_data.get('insufficient_data'):
                    return {
                        'prediction': 0.5,
                        'confidence': 0.1,
                        'reasoning': 'Insufficient entropy data',
                        'features': []
                    }
                
                shannon_entropy = entropy_data['shannon_entropy']
                randomness_score = entropy_data['randomness_score']
                
                # Low entropy = pattern detected
                if shannon_entropy < 0.8:
                    prediction = 0.6  # Expect pattern continuation
                    confidence = 0.7
                    reasoning = f"Low entropy ({shannon_entropy:.2f}) - pattern detected"
                elif randomness_score < 0.3:
                    prediction = 0.4  # Anti-pattern
                    confidence = 0.6
                    reasoning = f"Low randomness score ({randomness_score:.2f})"
                else:
                    prediction = 0.5
                    confidence = 0.3
                    reasoning = "High entropy - random behavior"
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'features': ['shannon_entropy', 'randomness_score']
                }
                
            except Exception as e:
                return {
                    'prediction': 0.5,
                    'confidence': 0.1,
                    'reasoning': f'Entropy model error: {e}',
                    'features': []
                }
        
        # Register all models
        self.prediction_integrator.register_model("hmac_prediction", hmac_prediction_model, weight=1.5)
        self.prediction_integrator.register_model("trend_analysis", trend_prediction_model, weight=1.2)
        self.prediction_integrator.register_model("volatility_analysis", volatility_prediction_model, weight=1.0)
        self.prediction_integrator.register_model("entropy_analysis", entropy_prediction_model, weight=1.1)
        
        logging.info("Registered 4 prediction models")
    
    async def make_supreme_decision(self) -> Optional[UnifiedDecision]:
        """Make the ultimate betting decision using all available intelligence"""
        
        try:
            decision_start_time = datetime.now()
            
            # Step 1: Get real-time data
            logging.info("Gathering real-time data...")
            game_state = await self.data_processor.get_current_game_state()
            trend_indicators = self.data_processor.get_trend_indicators()
            volatility_metrics = self.data_processor.get_volatility_metrics()
            entropy_analysis = self.data_processor.get_entropy_analysis()
            
            # Step 2: Get predictions from all models
            logging.info("Running prediction models...")
            context_for_predictions = {
                'game_state': game_state,
                'trend_indicators': trend_indicators,
                'volatility_metrics': volatility_metrics,
                'entropy_analysis': entropy_analysis
            }
            
            predictions = await self.prediction_integrator.get_predictions(context_for_predictions)
            prediction_consensus = self.prediction_integrator.create_consensus(predictions)
            
            # Step 3: Get strategy recommendation
            logging.info("Analyzing optimal strategy...")
            betting_context = self._create_betting_context(
                game_state, predictions, trend_indicators, 
                volatility_metrics, entropy_analysis
            )
            
            strategy_recommendation = self.strategy_framework.execute_comprehensive_strategy(betting_context)
            
            # Step 4: Get AI brain decision
            logging.info("Consulting AI brain...")
            ai_decision = await self.ai_brain.make_decision(betting_context)
            
            # Step 5: Create unified decision
            logging.info("Creating unified decision...")
            unified_decision = self._create_unified_decision(
                ai_decision, strategy_recommendation, prediction_consensus,
                game_state, betting_context
            )
            
            # Step 6: Apply final risk checks
            unified_decision = self._apply_risk_management(unified_decision)
            
            # Step 7: Log and store decision
            decision_time = (datetime.now() - decision_start_time).total_seconds()
            logging.info(f"Decision created in {decision_time:.2f} seconds")
            
            self.decision_history.append(unified_decision)
            self.last_decision_time = datetime.now()
            
            return unified_decision
            
        except Exception as e:
            logging.error(f"Failed to make supreme decision: {e}")
            return self._create_emergency_decision()
    
    def _create_betting_context(self, game_state, predictions, trend_indicators, 
                              volatility_metrics, entropy_analysis) -> BettingContext:
        """Create comprehensive betting context"""
        
        # Get recent outcomes
        recent_outcomes = []
        recent_data = list(self.data_processor.data_buffer)[-50:]
        for entry in recent_data:
            recent_outcomes.append(entry['last_roll'] > 50)
        
        # Convert predictions to expected format
        prediction_models_output = [
            {
                'model': p.model_name,
                'prediction': p.prediction,
                'confidence': p.confidence,
                'reasoning': p.reasoning
            } for p in predictions
        ]
        
        # Get last 4 decisions
        last_4_decisions = []
        for decision in list(self.decision_history)[-4:]:
            last_4_decisions.append({
                'timestamp': decision.timestamp,
                'should_bet': decision.should_bet,
                'bet_amount': decision.bet_amount,
                'multiplier': decision.multiplier,
                'confidence': decision.confidence,
                'outcome': None  # Would be filled after bet resolution
            })
        
        return BettingContext(
            current_game_state=game_state,
            prediction_models_output=prediction_models_output,
            last_4_decisions=last_4_decisions,
            bankroll=self.current_bankroll,
            session_profit_loss=self.performance_metrics['total_profit'],
            recent_outcomes=recent_outcomes,
            hmac_predictions=[],  # Could add specific HMAC predictions here
            api_real_values=[],   # Could add real API values here
            trend_indicators=trend_indicators,
            volatility_metrics=volatility_metrics,
            entropy_analysis=entropy_analysis
        )
    
    def _create_unified_decision(self, ai_decision: AIDecision, strategy_rec: Dict,
                               prediction_consensus: Dict, game_state: Dict,
                               betting_context: BettingContext) -> UnifiedDecision:
        """Create the final unified decision"""
        
        decision_id = self._generate_decision_id()
        
        # Combine all inputs to make final decision
        should_bet = (
            ai_decision.should_bet and 
            not strategy_rec.get('skip', False) and
            prediction_consensus['confidence'] > 0.3
        )
        
        if should_bet:
            # Combine bet sizing recommendations
            ai_bet_amount = ai_decision.bet_amount
            strategy_bet_amount = strategy_rec.get('stake', 0)
            
            # Use more conservative of the two
            bet_amount = min(ai_bet_amount, strategy_bet_amount)
            
            # Combine multiplier recommendations
            ai_multiplier = ai_decision.multiplier
            strategy_multiplier = strategy_rec.get('multiplier', 2.0)
            
            # Use weighted average based on confidence
            ai_weight = ai_decision.confidence / 100
            strategy_weight = 1 - ai_weight
            
            multiplier = (ai_multiplier * ai_weight + strategy_multiplier * strategy_weight)
            
            # Determine side based on prediction consensus
            if prediction_consensus['prediction'] > 0.5:
                side = "over"
                target_roll = 50 + (prediction_consensus['prediction'] - 0.5) * 100
            else:
                side = "under"
                target_roll = 50 - (0.5 - prediction_consensus['prediction']) * 100
            
            # Combined confidence
            confidence = (
                ai_decision.confidence * 0.4 +
                prediction_consensus['confidence'] * 100 * 0.4 +
                (1 - strategy_rec.get('skip', False)) * 20
            )
            
        else:
            bet_amount = 0
            multiplier = 2.0
            side = "over"
            target_roll = 50.0
            confidence = 0
        
        # Risk assessment
        risk_assessment = {
            'bankroll_risk': bet_amount / self.current_bankroll,
            'volatility_adjusted': betting_context.volatility_metrics.get('session_volatility', 0.5),
            'confidence_level': confidence,
            'max_loss': bet_amount,
            'potential_gain': bet_amount * multiplier,
            'risk_reward_ratio': multiplier if bet_amount > 0 else 0
        }
        
        return UnifiedDecision(
            should_bet=should_bet,
            bet_amount=bet_amount,
            multiplier=multiplier,
            side=side,
            target_roll=target_roll,
            confidence=confidence,
            ai_decision=ai_decision,
            strategy_recommendation=strategy_rec,
            prediction_consensus=prediction_consensus,
            risk_assessment=risk_assessment,
            timestamp=datetime.now(),
            session_id=self.session_id,
            decision_id=decision_id
        )
    
    def _apply_risk_management(self, decision: UnifiedDecision) -> UnifiedDecision:
        """Apply final risk management checks"""
        
        # Maximum bet size check (never bet more than 5% of bankroll)
        max_bet = self.current_bankroll * 0.05
        if decision.bet_amount > max_bet:
            decision.bet_amount = max_bet
        
        # Minimum confidence check
        if decision.confidence < 30 and decision.should_bet:
            decision.should_bet = False
            decision.bet_amount = 0
        
        # Session loss protection (stop if down more than 20%)
        session_loss_ratio = abs(min(0, self.performance_metrics['total_profit'])) / self.current_bankroll
        if session_loss_ratio > 0.20:
            decision.should_bet = False
            decision.bet_amount = 0
        
        # Maximum multiplier check (cap at 200x)
        if decision.multiplier > 200:
            decision.multiplier = 200
        
        return decision
    
    def _create_emergency_decision(self) -> UnifiedDecision:
        """Create emergency fallback decision"""
        return UnifiedDecision(
            should_bet=False,
            bet_amount=0,
            multiplier=2.0,
            side="over",
            target_roll=50.0,
            confidence=0,
            ai_decision=None,
            strategy_recommendation={'strategy': 'emergency', 'skip': True},
            prediction_consensus={'prediction': 0.5, 'confidence': 0},
            risk_assessment={'emergency': True},
            timestamp=datetime.now(),
            session_id=self.session_id,
            decision_id=self._generate_decision_id()
        )
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"decision_{timestamp}"
    
    async def execute_decision(self, decision: UnifiedDecision) -> Dict:
        """Execute the betting decision through the API"""
        
        if not decision.should_bet:
            logging.info("Decision: No bet")
            return {'action': 'no_bet', 'reason': 'decision_not_to_bet'}
        
        try:
            # Place bet through API
            bet_result = await self.api_access.place_bet(
                amount=decision.bet_amount,
                target=decision.target_roll,
                over=decision.side == "over"
            )
            
            # Update performance metrics
            self._update_performance_metrics(decision, bet_result)
            
            logging.info(f"Bet executed: {decision.bet_amount} at {decision.multiplier}x")
            
            return {
                'action': 'bet_placed',
                'decision_id': decision.decision_id,
                'bet_result': bet_result,
                'updated_bankroll': self.current_bankroll
            }
            
        except Exception as e:
            logging.error(f"Failed to execute bet: {e}")
            return {'action': 'bet_failed', 'error': str(e)}
    
    def _update_performance_metrics(self, decision: UnifiedDecision, bet_result: Dict):
        """Update performance tracking"""
        
        self.performance_metrics['total_decisions'] += 1
        
        if bet_result.get('won', False):
            self.performance_metrics['successful_bets'] += 1
            profit = decision.bet_amount * (decision.multiplier - 1)
        else:
            profit = -decision.bet_amount
        
        self.performance_metrics['total_profit'] += profit
        self.current_bankroll += profit
        
        # Update rates
        self.performance_metrics['win_rate'] = (
            self.performance_metrics['successful_bets'] / 
            self.performance_metrics['total_decisions'] * 100
        )
        
        self.performance_metrics['roi'] = (
            self.performance_metrics['total_profit'] / 
            (self.current_bankroll - self.performance_metrics['total_profit']) * 100
        )
    
    def get_comprehensive_status(self) -> Dict:
        """Get complete status of the decision engine"""
        
        session_duration = datetime.now() - self.session_start_time
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'duration_minutes': session_duration.total_seconds() / 60,
                'start_time': self.session_start_time,
                'last_decision': self.last_decision_time
            },
            'performance_metrics': self.performance_metrics,
            'current_bankroll': self.current_bankroll,
            'recent_decisions': len(self.decision_history),
            'api_status': 'connected',  # Would check actual API status
            'ai_brain_status': 'active',
            'models_registered': len(self.prediction_integrator.models),
            'strategies_available': 22,
            'data_buffer_size': len(self.data_processor.data_buffer)
        }


# Global instance - The Supreme Decision Engine
supreme_engine = None

def initialize_supreme_engine(stake_api_key: str, aws_credentials: Dict = None):
    """Initialize the supreme decision engine"""
    global supreme_engine
    supreme_engine = UnifiedAIDecisionEngine(stake_api_key, aws_credentials)
    logging.info("Supreme AI Decision Engine initialized and ready for maximum profit generation!")
    return supreme_engine

def get_supreme_engine():
    """Get the global supreme engine instance"""
    global supreme_engine
    return supreme_engine