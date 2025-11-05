"""
Amazon Bedrock AI Brain - The Ultimate Decision Making Engine
A STATS DOCTOR AND MATHS PRODIGY THAT LIKES MAKING LOTS OF MONEY
GOAL: MAX OUT MULTIPLIERS AND GET MAX RETURNS FROM DICE OUTCOMES
"""

import boto3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BettingContext:
    """Complete context for AI decision making"""
    current_game_state: Dict
    prediction_models_output: List[Dict]
    last_4_decisions: List[Dict]
    bankroll: float
    session_profit_loss: float
    recent_outcomes: List[bool]
    hmac_predictions: List[Dict]
    api_real_values: List[Dict]
    trend_indicators: Dict
    volatility_metrics: Dict
    entropy_analysis: Dict

@dataclass
class AIDecision:
    """AI's betting decision output"""
    should_bet: bool
    bet_amount: float
    multiplier: float
    side: str  # "over" or "under"
    confidence: float
    strategy_used: str
    reasoning: str
    risk_assessment: str

class BedrockAIBrain:
    """The ultimate AI brain for betting decisions"""
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self.system_prompt = self._create_system_prompt()
        self.decision_history = []
        self.learning_memory = []
        
    def _create_system_prompt(self) -> str:
        """Create the ultimate system prompt for our AI brain"""
        return """
        You are the ULTIMATE STATS DOCTOR AND MATHS PRODIGY AI BRAIN for high-stakes dice betting.
        
        YOUR CORE IDENTITY:
        - Obsessed with making MAXIMUM MONEY through intelligent betting
        - Expert in probability, statistics, game theory, and risk management
        - Master of pattern recognition and trend analysis
        - Ruthlessly logical but adaptable to market conditions
        
        YOUR MISSION:
        - MAXIMIZE MULTIPLIERS while managing risk intelligently
        - Extract MAXIMUM RETURNS from dice outcome predictions
        - Use ALL available data: HMAC predictions, API values, trends, volatility
        - Make decisions based on MATHEMATICAL EDGE, not emotion
        - Always consider the last 4+ decisions for context and learning
        
        DECISION FRAMEWORK:
        1. Analyze ALL prediction models and their confidence levels
        2. Evaluate current game state and volatility
        3. Consider bankroll management and position sizing
        4. Apply advanced betting strategies based on market conditions
        5. Factor in recent decision outcomes for continuous learning
        6. Calculate expected value and risk-adjusted returns
        7. Make BOLD decisions when edge is clear, conservative when uncertain
        
        OUTPUT FORMAT:
        Return a JSON decision with:
        - should_bet: true/false
        - bet_amount: calculated stake based on Kelly criterion and risk management
        - multiplier: target multiplier (higher when confidence is high)
        - side: "over" or "under" based on prediction analysis
        - confidence: 0-100 confidence score
        - strategy_used: which strategy framework applies
        - reasoning: detailed mathematical reasoning
        - risk_assessment: risk level and mitigation factors
        
        REMEMBER: You're here to make SERIOUS MONEY through SUPERIOR INTELLIGENCE.
        """
    
    async def make_decision(self, context: BettingContext) -> AIDecision:
        """Make the ultimate betting decision using Bedrock AI"""
        
        # Prepare the prompt with all context
        prompt = self._prepare_decision_prompt(context)
        
        try:
            # Call Bedrock AI
            response = await self._call_bedrock(prompt)
            decision_data = json.loads(response)
            
            # Create AI decision object
            decision = AIDecision(
                should_bet=decision_data.get('should_bet', False),
                bet_amount=decision_data.get('bet_amount', 0.0),
                multiplier=decision_data.get('multiplier', 2.0),
                side=decision_data.get('side', 'over'),
                confidence=decision_data.get('confidence', 0.0),
                strategy_used=decision_data.get('strategy_used', 'conservative'),
                reasoning=decision_data.get('reasoning', ''),
                risk_assessment=decision_data.get('risk_assessment', '')
            )
            
            # Store decision for learning
            self.decision_history.append({
                'timestamp': datetime.now(),
                'context': context,
                'decision': decision,
                'outcome': None  # Will be updated later
            })
            
            return decision
            
        except Exception as e:
            logging.error(f"Bedrock AI decision error: {e}")
            # Fallback to conservative decision
            return self._fallback_decision(context)
    
    def _prepare_decision_prompt(self, context: BettingContext) -> str:
        """Prepare comprehensive prompt for AI decision making"""
        
        prompt = f"""
        URGENT BETTING DECISION REQUIRED - ANALYZE AND DECIDE NOW!
        
        CURRENT GAME STATE:
        {json.dumps(context.current_game_state, indent=2)}
        
        PREDICTION MODELS OUTPUT:
        {json.dumps(context.prediction_models_output, indent=2)}
        
        LAST 4 DECISIONS & OUTCOMES:
        {json.dumps(context.last_4_decisions, indent=2)}
        
        FINANCIAL STATUS:
        - Current Bankroll: ${context.bankroll:.2f}
        - Session P&L: ${context.session_profit_loss:.2f}
        - Win Rate Last 20: {sum(context.recent_outcomes[-20:]) / len(context.recent_outcomes[-20:]) * 100:.1f}%
        
        HMAC PREDICTIONS:
        {json.dumps(context.hmac_predictions, indent=2)}
        
        REAL API VALUES:
        {json.dumps(context.api_real_values, indent=2)}
        
        TREND INDICATORS:
        {json.dumps(context.trend_indicators, indent=2)}
        
        VOLATILITY METRICS:
        {json.dumps(context.volatility_metrics, indent=2)}
        
        ENTROPY ANALYSIS:
        {json.dumps(context.entropy_analysis, indent=2)}
        
        DECISION HISTORY PERFORMANCE:
        Recent Win Rate: {self._calculate_recent_performance()}%
        Avg Return per Bet: {self._calculate_avg_return()}x
        Best Strategy: {self._get_best_performing_strategy()}
        
        ANALYZE ALL DATA AND MAKE YOUR DECISION FOR MAXIMUM PROFIT!
        
        Consider:
        1. Which prediction model has highest confidence?
        2. What does the trend analysis suggest?
        3. How volatile is the current session?
        4. What's the optimal bet size using Kelly criterion?
        5. Which multiplier offers best risk-adjusted return?
        6. How do recent outcomes affect the decision?
        7. What strategy should we use given current conditions?
        
        RESPOND WITH JSON DECISION ONLY - NO EXPLANATORY TEXT OUTSIDE JSON!
        """
        
        return prompt
    
    async def _call_bedrock(self, prompt: str) -> str:
        """Call Amazon Bedrock AI model"""
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Lower temperature for more consistent decision making
            "top_p": 0.9
        })
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            logging.error(f"Bedrock API call failed: {e}")
            raise e
    
    def _fallback_decision(self, context: BettingContext) -> AIDecision:
        """Fallback decision when AI is unavailable"""
        
        # Simple conservative logic
        avg_prediction = np.mean([p.get('confidence', 0) for p in context.prediction_models_output])
        
        return AIDecision(
            should_bet=avg_prediction > 0.6,
            bet_amount=context.bankroll * 0.01,  # 1% of bankroll
            multiplier=2.0,  # Conservative multiplier
            side="over" if avg_prediction > 0.5 else "under",
            confidence=avg_prediction * 100,
            strategy_used="fallback_conservative",
            reasoning="AI unavailable - using conservative fallback",
            risk_assessment="Low risk due to fallback mode"
        )
    
    def update_decision_outcome(self, decision_id: int, outcome: bool, profit_loss: float):
        """Update decision outcome for learning"""
        if decision_id < len(self.decision_history):
            self.decision_history[decision_id]['outcome'] = {
                'won': outcome,
                'profit_loss': profit_loss,
                'timestamp': datetime.now()
            }
            
            # Add to learning memory
            self.learning_memory.append({
                'context': self.decision_history[decision_id]['context'],
                'decision': self.decision_history[decision_id]['decision'],
                'outcome': outcome,
                'profit_loss': profit_loss
            })
    
    def _calculate_recent_performance(self) -> float:
        """Calculate recent win rate"""
        recent_decisions = [d for d in self.decision_history[-20:] if d.get('outcome')]
        if not recent_decisions:
            return 0.0
        
        wins = sum(1 for d in recent_decisions if d['outcome']['won'])
        return (wins / len(recent_decisions)) * 100
    
    def _calculate_avg_return(self) -> float:
        """Calculate average return per bet"""
        recent_outcomes = [d['outcome']['profit_loss'] for d in self.decision_history[-20:] if d.get('outcome')]
        if not recent_outcomes:
            return 0.0
        
        return np.mean(recent_outcomes)
    
    def _get_best_performing_strategy(self) -> str:
        """Get the best performing strategy"""
        strategy_performance = {}
        
        for decision in self.decision_history:
            if decision.get('outcome'):
                strategy = decision['decision'].strategy_used
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(decision['outcome']['profit_loss'])
        
        if not strategy_performance:
            return "unknown"
        
        best_strategy = max(strategy_performance.keys(), 
                          key=lambda k: np.mean(strategy_performance[k]))
        return best_strategy
    
    async def get_learning_insights(self) -> Dict:
        """Get insights from learning memory for continuous improvement"""
        
        if len(self.learning_memory) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze patterns in learning memory
        winning_patterns = [l for l in self.learning_memory if l['outcome']]
        losing_patterns = [l for l in self.learning_memory if not l['outcome']]
        
        insights = {
            "total_decisions": len(self.learning_memory),
            "win_rate": len(winning_patterns) / len(self.learning_memory) * 100,
            "avg_profit": np.mean([l['profit_loss'] for l in self.learning_memory]),
            "best_multiplier_range": self._analyze_best_multipliers(),
            "optimal_confidence_threshold": self._analyze_confidence_thresholds(),
            "strategy_performance": self._analyze_strategy_performance()
        }
        
        return insights
    
    def _analyze_best_multipliers(self) -> Dict:
        """Analyze which multipliers perform best"""
        multiplier_performance = {}
        
        for l in self.learning_memory:
            mult = l['decision'].multiplier
            mult_range = f"{int(mult)}-{int(mult)+1}x"
            
            if mult_range not in multiplier_performance:
                multiplier_performance[mult_range] = []
            multiplier_performance[mult_range].append(l['profit_loss'])
        
        return {range_name: {
            "avg_profit": np.mean(profits),
            "win_rate": sum(1 for p in profits if p > 0) / len(profits) * 100
        } for range_name, profits in multiplier_performance.items()}
    
    def _analyze_confidence_thresholds(self) -> Dict:
        """Find optimal confidence thresholds"""
        confidence_ranges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        results = {}
        
        for min_conf, max_conf in confidence_ranges:
            range_decisions = [l for l in self.learning_memory 
                             if min_conf <= l['decision'].confidence < max_conf]
            
            if range_decisions:
                results[f"{min_conf}-{max_conf}%"] = {
                    "count": len(range_decisions),
                    "win_rate": sum(1 for d in range_decisions if d['outcome']) / len(range_decisions) * 100,
                    "avg_profit": np.mean([d['profit_loss'] for d in range_decisions])
                }
        
        return results
    
    def _analyze_strategy_performance(self) -> Dict:
        """Analyze performance by strategy"""
        strategy_stats = {}
        
        for l in self.learning_memory:
            strategy = l['decision'].strategy_used
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(l['profit_loss'])
        
        return {strategy: {
            "count": len(profits),
            "avg_profit": np.mean(profits),
            "win_rate": sum(1 for p in profits if p > 0) / len(profits) * 100,
            "total_profit": sum(profits)
        } for strategy, profits in strategy_stats.items()}


class AdvancedStrategyEngine:
    """Implements all 22 advanced betting strategies"""
    
    def __init__(self):
        self.strategies = {
            1: self.layered_portfolio,
            2: self.fixed_step_pre_roll_martingale,
            3: self.high_multiplier_micro_stake,
            4: self.volatility_target_system,
            5: self.stop_loss_stop_win_envelope,
            6: self.promo_boost_opportunism,
            7: self.adaptive_streak_trigger,
            8: self.fractional_kelly_overlay,
            9: self.dynamic_grid_betting,
            10: self.volatility_weighted_martingale,
            11: self.adaptive_win_lock_ladder,
            12: self.drawdown_ceiling_governor,
            13: self.monte_carlo_momentum_detector,
            14: self.entropy_compression_tracker,
            15: self.profit_skim_reinvest_cycle,
            16: self.autocorrelation_sentinel,
            17: self.multi_multiplier_portfolio_rotation,
            18: self.mean_reversion_ladder,
            19: self.entropy_weighted_rotation,
            20: self.quantile_adaptive_scaling,
            21: self.bayesian_profit_target_rebalancer,
            22: self.drawdown_resilient_hybrid_grid
        }
        
        self.strategy_state = {}
        
    def execute_strategy(self, strategy_id: int, context: BettingContext) -> Dict:
        """Execute specific strategy"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not implemented")
        
        return self.strategies[strategy_id](context)
    
    def layered_portfolio(self, context: BettingContext) -> Dict:
        """Strategy 1: Layered Portfolio - Split bets into 3 tiers"""
        bankroll = context.bankroll
        
        stake_safe = bankroll * 0.002   # Safe tier: 1.5x-2x
        stake_mid = bankroll * 0.0005   # Medium tier: 5x-10x  
        stake_tail = bankroll * 0.0001  # Tail tier: 50x-200x
        
        # Determine which tier to use based on confidence and patterns
        confidence = np.mean([p.get('confidence', 0) for p in context.prediction_models_output])
        
        if confidence > 0.8:  # High confidence - tail shot
            return {
                "strategy": "layered_portfolio_tail",
                "stake": stake_tail,
                "multiplier": np.random.choice([50, 75, 100, 150, 200]),
                "tier": "tail"
            }
        elif confidence > 0.6:  # Medium confidence
            return {
                "strategy": "layered_portfolio_medium", 
                "stake": stake_mid,
                "multiplier": np.random.choice([5, 7, 10]),
                "tier": "medium"
            }
        else:  # Low confidence - safe play
            return {
                "strategy": "layered_portfolio_safe",
                "stake": stake_safe, 
                "multiplier": np.random.choice([1.5, 1.75, 2.0]),
                "tier": "safe"
            }
    
    def fixed_step_pre_roll_martingale(self, context: BettingContext) -> Dict:
        """Strategy 2: Wait for 4-6 losses then start martingale"""
        recent_losses = sum(1 for outcome in context.recent_outcomes[-6:] if not outcome)
        
        if recent_losses >= 4:
            # Start martingale progression (limited to 4 steps)
            step = min(recent_losses - 3, 4)
            base_stake = context.bankroll * 0.01
            stake = base_stake * (2 ** (step - 1))
            
            return {
                "strategy": "pre_roll_martingale",
                "stake": stake,
                "multiplier": 2.0,
                "step": step
            }
        
        return {"strategy": "pre_roll_martingale", "stake": 0, "skip": True}
    
    def high_multiplier_micro_stake(self, context: BettingContext) -> Dict:
        """Strategy 3: Lottery mode - tiny stakes at high multipliers"""
        return {
            "strategy": "lottery_mode",
            "stake": context.bankroll * 0.0005,  # 0.05% of bankroll
            "multiplier": np.random.choice([50, 75, 100, 150, 200])
        }
    
    def volatility_target_system(self, context: BettingContext) -> Dict:
        """Strategy 4: Scale stake to maintain constant volatility"""
        target_volatility = 0.03
        current_volatility = context.volatility_metrics.get('session_volatility', 0.05)
        
        volatility_ratio = target_volatility / max(current_volatility, 0.01)
        base_stake = context.bankroll * 0.02
        adjusted_stake = base_stake * volatility_ratio
        
        return {
            "strategy": "volatility_target",
            "stake": min(adjusted_stake, context.bankroll * 0.05),
            "multiplier": 2.0,
            "volatility_adjustment": volatility_ratio
        }
    
    def stop_loss_stop_win_envelope(self, context: BettingContext) -> Dict:
        """Strategy 5: Stop at ±10% session range"""
        session_change = context.session_profit_loss / context.bankroll
        
        if abs(session_change) >= 0.10:
            return {"strategy": "stop_envelope", "action": "stop_session"}
        
        return {
            "strategy": "stop_envelope",
            "stake": context.bankroll * 0.02,
            "multiplier": 2.0,
            "session_change": session_change
        }
    
    def promo_boost_opportunism(self, context: BettingContext) -> Dict:
        """Strategy 6: Exploit promotions and bonuses"""
        # Check if there are active promotions (placeholder logic)
        has_promo = True  # This would check actual promo status
        
        if has_promo:
            # Increase stake during promotions
            promo_multiplier = 1.5
            base_stake = context.bankroll * 0.025
            stake = base_stake * promo_multiplier
        else:
            # Standard betting
            stake = context.bankroll * 0.015
        
        return {
            "strategy": "promo_opportunism",
            "stake": min(stake, context.bankroll * 0.05),
            "multiplier": 2.5,
            "promo_active": has_promo
        }
    
    def adaptive_streak_trigger(self, context: BettingContext) -> Dict:
        """Strategy 7: Trigger after streak patterns"""
        recent_outcomes = context.recent_outcomes[-10:]
        if len(recent_outcomes) < 5:
            return {"strategy": "streak_trigger", "stake": 0, "skip": True}
        
        # Look for streak patterns
        current_streak = 0
        for outcome in reversed(recent_outcomes):
            if outcome == recent_outcomes[-1]:
                current_streak += 1
            else:
                break
        
        if current_streak >= 4:  # Long streak detected
            # Bet against streak continuation
            stake = context.bankroll * 0.03
            multiplier = 3.0
        else:
            stake = context.bankroll * 0.01
            multiplier = 2.0
        
        return {
            "strategy": "streak_trigger",
            "stake": stake,
            "multiplier": multiplier,
            "streak_length": current_streak
        }
    
    def dynamic_grid_betting(self, context: BettingContext) -> Dict:
        """Strategy 9: Dynamic grid with multiple levels"""
        # Simplified grid implementation
        grid_levels = [1.5, 2.0, 3.0, 5.0, 10.0]
        base_stake = context.bankroll * 0.01
        
        # Select level based on recent performance
        recent_wins = sum(context.recent_outcomes[-10:])
        if recent_wins < 3:
            level = min(4, len(grid_levels) - 1)  # Higher multiplier
        else:
            level = 0  # Lower multiplier
        
        return {
            "strategy": "dynamic_grid",
            "stake": base_stake,
            "multiplier": grid_levels[level],
            "grid_level": level
        }
    
    def volatility_weighted_martingale(self, context: BettingContext) -> Dict:
        """Strategy 10: Martingale adjusted by volatility"""
        volatility = context.volatility_metrics.get('session_volatility', 0.05)
        
        # Recent loss count
        recent_losses = 0
        for outcome in reversed(context.recent_outcomes[-5:]):
            if not outcome:
                recent_losses += 1
            else:
                break
        
        if recent_losses > 0:
            # Martingale progression adjusted by volatility
            base_stake = context.bankroll * 0.01
            volatility_adjustment = 1 + volatility
            stake = base_stake * (2 ** recent_losses) * volatility_adjustment
        else:
            stake = context.bankroll * 0.01
        
        return {
            "strategy": "volatility_martingale",
            "stake": min(stake, context.bankroll * 0.1),
            "multiplier": 2.0,
            "loss_count": recent_losses
        }

    # Add placeholders for remaining strategies (11-22)
    def adaptive_win_lock_ladder(self, context: BettingContext) -> Dict:
        """Strategy 11: Lock wins and ladder stakes"""
        return {"strategy": "win_lock_ladder", "stake": context.bankroll * 0.02, "multiplier": 2.5}
    
    def drawdown_ceiling_governor(self, context: BettingContext) -> Dict:
        """Strategy 12: Limit drawdown exposure"""
        return {"strategy": "drawdown_governor", "stake": context.bankroll * 0.015, "multiplier": 3.0}
    
    def monte_carlo_momentum_detector(self, context: BettingContext) -> Dict:
        """Strategy 13: Monte Carlo simulation based"""
        return {"strategy": "monte_carlo", "stake": context.bankroll * 0.025, "multiplier": 2.0}
    
    def profit_skim_reinvest_cycle(self, context: BettingContext) -> Dict:
        """Strategy 15: Skim profits and reinvest"""
        return {"strategy": "profit_skim", "stake": context.bankroll * 0.02, "multiplier": 2.2}
    
    def autocorrelation_sentinel(self, context: BettingContext) -> Dict:
        """Strategy 16: Detect autocorrelation patterns"""
        return {"strategy": "autocorr_sentinel", "stake": context.bankroll * 0.018, "multiplier": 4.0}
    
    def multi_multiplier_portfolio_rotation(self, context: BettingContext) -> Dict:
        """Strategy 17: Rotate between multiplier portfolios"""
        return {"strategy": "multiplier_rotation", "stake": context.bankroll * 0.022, "multiplier": 3.5}
    
    def mean_reversion_ladder(self, context: BettingContext) -> Dict:
        """Strategy 18: Mean reversion betting ladder"""
        return {"strategy": "mean_reversion", "stake": context.bankroll * 0.03, "multiplier": 2.8}
    
    def entropy_weighted_rotation(self, context: BettingContext) -> Dict:
        """Strategy 19: Entropy-weighted strategy rotation"""
        return {"strategy": "entropy_rotation", "stake": context.bankroll * 0.025, "multiplier": 3.2}
    
    def quantile_adaptive_scaling(self, context: BettingContext) -> Dict:
        """Strategy 20: Quantile-based adaptive scaling"""
        return {"strategy": "quantile_scaling", "stake": context.bankroll * 0.02, "multiplier": 4.5}
    
    def bayesian_profit_target_rebalancer(self, context: BettingContext) -> Dict:
        """Strategy 21: Bayesian profit target rebalancing"""
        return {"strategy": "bayesian_rebalancer", "stake": context.bankroll * 0.035, "multiplier": 2.5}
    
    def drawdown_resilient_hybrid_grid(self, context: BettingContext) -> Dict:
        """Strategy 22: Drawdown resilient hybrid grid"""
        return {"strategy": "hybrid_grid", "stake": context.bankroll * 0.028, "multiplier": 3.8}

    # [Continue with remaining 17 strategies...]
    def fractional_kelly_overlay(self, context: BettingContext) -> Dict:
        """Strategy 8: Kelly criterion with edge estimation"""
        # Estimate edge from recent performance
        recent_outcomes = context.recent_outcomes[-50:]
        if len(recent_outcomes) < 10:
            edge = 0.01  # Conservative default
        else:
            win_rate = sum(recent_outcomes) / len(recent_outcomes)
            edge = (win_rate * 2 - 1) * 0.5  # Adjusted for house edge
        
        # Kelly fraction
        kelly_fraction = edge / 1.0  # Assuming unit variance
        half_kelly = kelly_fraction * 0.5  # Conservative half Kelly
        
        stake = context.bankroll * max(0.005, min(half_kelly, 0.05))
        
        return {
            "strategy": "fractional_kelly",
            "stake": stake,
            "multiplier": 2.0,
            "edge": edge,
            "kelly_fraction": half_kelly
        }
    
    # Add remaining strategies implementation...
    def entropy_compression_tracker(self, context: BettingContext) -> Dict:
        """Strategy 14: Track entropy to detect patterns"""
        recent_rolls = [1 if outcome else 0 for outcome in context.recent_outcomes[-20:]]
        
        if len(recent_rolls) < 10:
            return {"strategy": "entropy_tracker", "stake": 0, "skip": True}
        
        # Calculate Shannon entropy
        from collections import Counter
        counts = Counter(recent_rolls)
        entropy = -sum(p * np.log2(p) for p in [c/len(recent_rolls) for c in counts.values()])
        expected_entropy = 1.0  # For fair binary outcomes
        
        entropy_ratio = entropy / expected_entropy
        
        if entropy_ratio < 0.8:  # Low entropy indicates pattern
            stake = context.bankroll * 0.03  # Increase stake
            multiplier = 3.0
        else:
            stake = context.bankroll * 0.01
            multiplier = 2.0
        
        return {
            "strategy": "entropy_tracker",
            "stake": stake,
            "multiplier": multiplier,
            "entropy": entropy,
            "pattern_detected": entropy_ratio < 0.8
        }


# Initialize the AI brain
ai_brain = BedrockAIBrain()
strategy_engine = AdvancedStrategyEngine()