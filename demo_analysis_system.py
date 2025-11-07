#!/usr/bin/env python3
"""
DEMO ANALYSIS SYSTEM
====================
Runs 100 demo bets to analyze patterns, seeds, and bias before real betting
Integrates with Stake API demo mode and feeds data to Oracle system
"""

import asyncio
import json
import logging
import time
import hmac
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from collections import deque

# Add App directory to path for imports
import sys
import os
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'App')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from bedrock_ai_brain import BedrockAIBrain, BettingContext, AIDecision
from live_demo_oracle import LiveOracle
from oracle_support_utils import detect_streaks, shannon_entropy, analyze_tda_holes

@dataclass
class DemoRoll:
    """Single demo roll data"""
    nonce: int
    client_seed: str
    server_seed_hash: str
    result: float
    timestamp: datetime
    hmac_calculated: str
    bias_direction: str

@dataclass
class SeedAnalysis:
    """Analysis of server/client seed patterns"""
    server_seed_hash: str
    client_seed: str
    pattern_strength: float
    bias_detected: float  # -1 to 1, negative = under bias
    confidence: float
    hmac_patterns: List[str]
    entropy_score: float

class DemoAnalysisSystem:
    """System for analyzing 100 demo rolls before real betting"""
    
    def __init__(self, stake_api_key: str):
        self.stake_api_key = stake_api_key
        self.demo_rolls = deque(maxlen=100)
        self.seed_analysis = None
        self.current_analysis = {}
        self.oracle = LiveOracle()
        self.ai_brain = BedrockAIBrain()
        
        # Demo analysis state
        self.is_analyzing = False
        self.analysis_progress = 0
        self.demo_complete = False
        
        # Pattern detection
        self.detected_patterns = []
        self.bias_strength = 0.0
        self.confidence_score = 0.0
        
    async def start_demo_analysis(self) -> Dict[str, Any]:
        """Start the 100-roll demo analysis"""
        if self.is_analyzing:
            return {"error": "Analysis already in progress"}
        
        self.is_analyzing = True
        self.analysis_progress = 0
        self.demo_complete = False
        self.demo_rolls.clear()
        
        logging.info("🔍 Starting demo analysis - 100 rolls")
        
        try:
            # Step 1: Get initial game state and seeds
            game_state = await self.get_stake_game_state()
            if not game_state:
                raise Exception("Failed to get game state")
            
            server_seed_hash = game_state.get('server_seed_hash', '')
            client_seed = game_state.get('client_seed', '')
            
            logging.info(f"🔑 Server seed hash: {server_seed_hash[:16]}...")
            logging.info(f"🎯 Client seed: {client_seed}")
            
            # Step 2: Collect 100 demo rolls
            await self.collect_demo_rolls(server_seed_hash, client_seed)
            
            # Step 3: Analyze patterns and bias
            analysis = await self.analyze_demo_patterns()
            
            # Step 4: Generate betting recommendations
            recommendations = await self.generate_betting_recommendations()
            
            self.demo_complete = True
            self.is_analyzing = False
            
            return {
                "status": "completed",
                "seed_analysis": analysis,
                "recommendations": recommendations,
                "demo_rolls_count": len(self.demo_rolls),
                "confidence": self.confidence_score,
                "bias_strength": self.bias_strength
            }
            
        except Exception as e:
            logging.error(f"❌ Demo analysis failed: {e}")
            self.is_analyzing = False
            return {"error": str(e)}
    
    async def collect_demo_rolls(self, server_seed_hash: str, client_seed: str):
        """Collect 100 demo rolls from Stake API"""
        
        for nonce in range(1, 101):
            try:
                # Calculate expected HMAC result
                hmac_input = f"{client_seed}:{nonce}"
                calculated_hmac = hmac.new(
                    server_seed_hash.encode(),
                    hmac_input.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                # Convert HMAC to dice result
                dice_result = self.hmac_to_dice_result(calculated_hmac)
                
                # Determine bias direction for this roll
                bias_dir = "under" if dice_result < 50 else "over"
                
                # Create demo roll entry
                demo_roll = DemoRoll(
                    nonce=nonce,
                    client_seed=client_seed,
                    server_seed_hash=server_seed_hash,
                    result=dice_result,
                    timestamp=datetime.now(),
                    hmac_calculated=calculated_hmac,
                    bias_direction=bias_dir
                )
                
                self.demo_rolls.append(demo_roll)
                self.analysis_progress = nonce
                
                # Update real-time analysis
                if nonce % 10 == 0:
                    await self.update_partial_analysis()
                
                # Small delay to simulate real collection
                await asyncio.sleep(0.1)
                
                logging.info(f"📊 Demo roll {nonce}/100: {dice_result:.2f} ({bias_dir})")
                
            except Exception as e:
                logging.error(f"❌ Failed to collect demo roll {nonce}: {e}")
                continue
    
    def hmac_to_dice_result(self, hmac_hex: str) -> float:
        """Convert HMAC hash to dice result (0-99.99)"""
        # Use first 8 characters of HMAC for dice calculation
        hex_substr = hmac_hex[:8]
        
        # Convert to integer
        hex_int = int(hex_substr, 16)
        
        # Convert to dice result (0-99.99)
        dice_result = (hex_int % 10000) / 100.0
        
        return dice_result
    
    async def analyze_demo_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the collected demo rolls"""
        
        if len(self.demo_rolls) < 10:
            return {"error": "Insufficient demo data"}
        
        # Extract roll results
        results = [roll.result for roll in self.demo_rolls]
        
        # Calculate bias
        under_count = sum(1 for r in results if r < 50)
        over_count = len(results) - under_count
        bias_strength = (under_count - over_count) / len(results)
        
        # Pattern analysis
        streaks = detect_streaks(results)
        entropy = shannon_entropy(results)
        
        # HMAC pattern analysis
        hmac_patterns = [roll.hmac_calculated[:8] for roll in self.demo_rolls]
        pattern_frequency = {}
        
        for pattern in hmac_patterns:
            pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
        
        # Confidence calculation based on pattern strength
        confidence = min(100, abs(bias_strength) * 200 + (1 - entropy) * 50)
        
        # Store analysis results
        self.bias_strength = bias_strength
        self.confidence_score = confidence
        
        seed_analysis = SeedAnalysis(
            server_seed_hash=self.demo_rolls[0].server_seed_hash,
            client_seed=self.demo_rolls[0].client_seed,
            pattern_strength=1 - entropy,
            bias_detected=bias_strength,
            confidence=confidence,
            hmac_patterns=list(pattern_frequency.keys())[:5],
            entropy_score=entropy
        )
        
        self.seed_analysis = seed_analysis
        
        return {
            "bias_strength": bias_strength,
            "bias_direction": "under" if bias_strength < 0 else "over",
            "confidence": confidence,
            "entropy": entropy,
            "streak_analysis": streaks,
            "pattern_frequency": pattern_frequency,
            "total_rolls": len(results),
            "under_percentage": (under_count / len(results)) * 100,
            "over_percentage": (over_count / len(results)) * 100
        }
    
    async def generate_betting_recommendations(self) -> Dict[str, Any]:
        """Generate betting recommendations based on demo analysis"""
        
        if not self.seed_analysis:
            return {"error": "No seed analysis available"}
        
        # Prepare context for AI decision
        betting_context = BettingContext(
            current_game_state={
                "server_seed_hash": self.seed_analysis.server_seed_hash,
                "client_seed": self.seed_analysis.client_seed,
                "bias_detected": self.seed_analysis.bias_detected
            },
            prediction_models_output=[{
                "model": "demo_analysis",
                "confidence": self.seed_analysis.confidence,
                "prediction": "under" if self.seed_analysis.bias_detected < 0 else "over"
            }],
            last_4_decisions=[],
            bankroll=1000.0,
            session_profit_loss=0.0,
            recent_outcomes=[roll.result < 50 for roll in list(self.demo_rolls)[-10:]],
            hmac_predictions=[{
                "pattern": pattern,
                "strength": self.seed_analysis.pattern_strength
            } for pattern in self.seed_analysis.hmac_patterns],
            api_real_values=[roll.result for roll in self.demo_rolls],
            trend_indicators={
                "bias_strength": self.seed_analysis.bias_detected,
                "entropy": self.seed_analysis.entropy_score
            },
            volatility_metrics={
                "variance": np.var([roll.result for roll in self.demo_rolls])
            },
            entropy_analysis={
                "entropy": self.seed_analysis.entropy_score,
                "pattern_strength": self.seed_analysis.pattern_strength
            }
        )
        
        # Get AI recommendation
        ai_decision = await self.ai_brain.make_decision(betting_context)
        
        # Generate specific betting strategy
        if abs(self.seed_analysis.bias_detected) > 0.1 and self.seed_analysis.confidence > 60:
            # Strong bias detected - aggressive strategy
            strategy = "aggressive"
            recommended_side = "under" if self.seed_analysis.bias_detected < 0 else "over"
            target_value = 45.0 if recommended_side == "under" else 55.0
            confidence_level = "high"
        elif abs(self.seed_analysis.bias_detected) > 0.05 and self.seed_analysis.confidence > 40:
            # Moderate bias - conservative strategy
            strategy = "conservative"
            recommended_side = "under" if self.seed_analysis.bias_detected < 0 else "over"
            target_value = 48.0 if recommended_side == "under" else 52.0
            confidence_level = "medium"
        else:
            # No clear bias - wait strategy
            strategy = "wait"
            recommended_side = "none"
            target_value = 50.0
            confidence_level = "low"
        
        return {
            "strategy": strategy,
            "recommended_side": recommended_side,
            "target_value": target_value,
            "confidence_level": confidence_level,
            "ai_decision": {
                "should_bet": ai_decision.should_bet,
                "bet_amount": ai_decision.bet_amount,
                "multiplier": ai_decision.multiplier,
                "reasoning": ai_decision.reasoning
            },
            "risk_assessment": {
                "bias_strength": abs(self.seed_analysis.bias_detected),
                "pattern_confidence": self.seed_analysis.confidence,
                "entropy_score": self.seed_analysis.entropy_score
            }
        }
    
    async def update_partial_analysis(self):
        """Update analysis during demo collection for real-time display"""
        if len(self.demo_rolls) < 10:
            return
        
        results = [roll.result for roll in self.demo_rolls]
        
        # Calculate current bias
        under_count = sum(1 for r in results if r < 50)
        current_bias = (under_count - (len(results) - under_count)) / len(results)
        
        # Update current analysis
        self.current_analysis = {
            "progress": self.analysis_progress,
            "current_bias": current_bias,
            "bias_direction": "under" if current_bias < 0 else "over",
            "rolls_analyzed": len(self.demo_rolls),
            "entropy": shannon_entropy(results) if len(results) > 5 else 0.5
        }
    
    async def get_stake_game_state(self) -> Optional[Dict[str, Any]]:
        """Get current game state from Stake API"""
        # This would integrate with real Stake API
        # For demo, return simulated state
        return {
            "server_seed_hash": "7b1234567890abcdef" + "0" * 46,  # 64 char hash
            "client_seed": "demo_client_seed_" + str(int(time.time())),
            "nonce": 1,
            "balance": 1000.0
        }
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status for frontend updates"""
        return {
            "is_analyzing": self.is_analyzing,
            "progress": self.analysis_progress,
            "demo_complete": self.demo_complete,
            "current_analysis": self.current_analysis,
            "confidence": self.confidence_score,
            "bias_strength": self.bias_strength
        }

# Export for integration
__all__ = ['DemoAnalysisSystem', 'DemoRoll', 'SeedAnalysis']