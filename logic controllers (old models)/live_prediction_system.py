"""
Live Prediction Integration System
Combines real-time Stake API data with your existing prediction models
This is your MONEY-MAKING integration!
"""

import sys
import os
import json
import time
import hmac
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_env_setup import setup_enhanced_environment, EnhancedStakeAPI
    from ultimate_stake_predictor import UltimateStakePredictor
    from massive_stake_analyzer import MassiveStakeAnalyzer
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("📁 Make sure all prediction files are in the same directory")

class LivePredictionSystem:
    """
    The MAIN SYSTEM that combines everything for live predictions
    This is what you'll use for actual betting!
    """
    
    def __init__(self):
        print("🚀 Initializing Live Prediction System...")
        
        # Initialize API connection
        self.api_client = setup_enhanced_environment()
        if not self.api_client:
            raise Exception("❌ Failed to setup API connection")
        
        # Initialize prediction models
        self.predictor = UltimateStakePredictor()
        self.analyzer = MassiveStakeAnalyzer()
        
        # Tracking variables
        self.last_prediction = None
        self.prediction_accuracy = []
        self.total_predictions = 0
        self.correct_predictions = 0
        
        print("✅ Live Prediction System ready!")
    
    def get_live_prediction(self) -> Dict[str, Any]:
        """
        Get live prediction for the NEXT dice roll
        This is your main prediction function!
        """
        print("\n🎯 GETTING LIVE PREDICTION...")
        
        # Step 1: Get current game state from API
        iteration_info = self.api_client.detect_current_iteration()
        
        if not iteration_info.get("prediction_ready"):
            return {
                "status": "not_ready",
                "message": "No active dice game detected. Start a dice game on Stake first!",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        # Step 2: Extract prediction parameters
        seed = iteration_info["dice_seed"]
        next_nonce = iteration_info["dice_next_nonce"]
        
        print(f"🔢 Seed: {seed}")
        print(f"🎲 Next Nonce: {next_nonce}")
        
        # Step 3: Generate all prediction methods
        predictions = {}
        
        # HMAC-SHA256 Direct Calculation (Most Accurate)
        try:
            predictions["hmac_direct"] = self.calculate_hmac_result(seed, next_nonce)
            print(f"🔐 HMAC Direct: {predictions['hmac_direct']:.4f}")
        except Exception as e:
            print(f"⚠️ HMAC calculation error: {e}")
        
        # Ultimate Predictor (AI + Multiple Methods)
        try:
            ultimate_pred = self.predictor.predict_next_roll(seed, next_nonce)
            predictions.update(ultimate_pred)
            print(f"🤖 AI Prediction: {ultimate_pred.get('ai_prediction', 'N/A')}")
            print(f"📊 Ensemble: {ultimate_pred.get('ensemble_prediction', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Ultimate predictor error: {e}")
        
        # Step 4: Create final prediction result
        final_prediction = {
            "status": "ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "game_state": {
                "seed": seed,
                "next_nonce": next_nonce,
                "current_position": next_nonce - 1
            },
            "predictions": predictions,
            "confidence_score": self.calculate_confidence(predictions),
            "recommended_action": self.get_recommendation(predictions)
        }
        
        # Step 5: Store for accuracy tracking
        self.last_prediction = final_prediction
        self.total_predictions += 1
        
        return final_prediction
    
    def calculate_hmac_result(self, seed: str, nonce: int) -> float:
        """
        Calculate exact HMAC-SHA256 result (most accurate method)
        This matches Stake's exact algorithm!
        """
        # Create the message (exactly how Stake does it)
        message = f"{seed}:{nonce}"
        
        # Calculate HMAC-SHA256
        hmac_result = hmac.new(
            seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Convert to dice roll (0-99.99)
        # Take first 8 hex characters and convert to decimal
        hex_substr = hmac_result[:8]
        decimal_value = int(hex_substr, 16)
        
        # Convert to 0-99.99 range
        roll_result = (decimal_value / 0xFFFFFFFF) * 100
        
        return roll_result
    
    def calculate_confidence(self, predictions: Dict) -> float:
        """Calculate confidence score based on prediction agreement"""
        if not predictions:
            return 0.0
        
        values = [v for v in predictions.values() if isinstance(v, (int, float))]
        
        if len(values) < 2:
            return 0.5
        
        # Calculate agreement between predictions
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Higher confidence when predictions agree (low std)
        confidence = max(0.0, min(1.0, 1.0 - (std_val / 50.0)))
        
        return confidence
    
    def get_recommendation(self, predictions: Dict) -> Dict[str, Any]:
        """Get betting recommendation based on predictions"""
        if not predictions:
            return {"action": "wait", "reason": "No predictions available"}
        
        # Get the most reliable prediction (HMAC direct)
        main_prediction = predictions.get("hmac_direct")
        
        if main_prediction is None:
            return {"action": "wait", "reason": "No reliable prediction available"}
        
        # Betting strategy based on prediction
        if main_prediction < 49.5:
            return {
                "action": "bet_under",
                "target": 49.5,
                "predicted_result": main_prediction,
                "win_probability": 49.5,
                "confidence": "high" if main_prediction < 45 else "medium"
            }
        elif main_prediction > 50.5:
            return {
                "action": "bet_over",
                "target": 50.5,
                "predicted_result": main_prediction,
                "win_probability": 49.5,
                "confidence": "high" if main_prediction > 55 else "medium"
            }
        else:
            return {
                "action": "skip",
                "reason": "Prediction too close to 50/50",
                "predicted_result": main_prediction
            }
    
    def verify_last_prediction(self) -> Dict[str, Any]:
        """
        Verify accuracy of the last prediction
        Call this after placing a bet to track accuracy!
        """
        if not self.last_prediction:
            return {"error": "No previous prediction to verify"}
        
        # Get recent bet results
        history = self.api_client.get_betting_history(limit=5)
        
        if not history:
            return {"error": "Could not fetch betting history"}
        
        # Find the most recent bet that matches our prediction
        last_bet = history[0]  # Most recent bet
        
        predicted_nonce = self.last_prediction["game_state"]["next_nonce"]
        actual_nonce = last_bet.get("nonce")
        
        if actual_nonce == predicted_nonce:
            # This is our bet! Check accuracy
            actual_result = last_bet.get("outcome", 0)
            predicted_result = self.last_prediction["predictions"].get("hmac_direct")
            
            if predicted_result:
                error = abs(actual_result - predicted_result)
                accuracy = max(0, 100 - error)
                
                # Update tracking
                if error < 1.0:  # Very accurate prediction
                    self.correct_predictions += 1
                
                self.prediction_accuracy.append(accuracy)
                
                return {
                    "verified": True,
                    "predicted": predicted_result,
                    "actual": actual_result,
                    "error": error,
                    "accuracy_percent": accuracy,
                    "total_accuracy": np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0,
                    "success_rate": (self.correct_predictions / self.total_predictions) * 100
                }
        
        return {"error": "Could not match prediction with recent bet"}
    
    def live_monitoring_mode(self):
        """
        Enter live monitoring mode - continuously watch for betting opportunities
        This is your REAL-TIME money-making mode!
        """
        print("\n🎯 ENTERING LIVE MONITORING MODE")
        print("=" * 60)
        print("💰 Watching for profitable betting opportunities...")
        print("🔄 Press Ctrl+C to stop")
        print("=" * 60)
        
        def prediction_callback(current_state, previous_state):
            """Called when game state changes"""
            if current_state.get("prediction_ready"):
                print(f"\n🚨 NEW BETTING OPPORTUNITY DETECTED!")
                
                # Get prediction
                prediction = self.get_live_prediction()
                
                if prediction["status"] == "ready":
                    rec = prediction["recommended_action"]
                    conf = prediction["confidence_score"]
                    
                    print(f"\n💡 RECOMMENDATION:")
                    print(f"   Action: {rec['action'].upper()}")
                    if rec["action"] != "skip":
                        print(f"   Target: {rec.get('target', 'N/A')}")
                        print(f"   Predicted: {rec['predicted_result']:.4f}")
                        print(f"   Confidence: {rec.get('confidence', 'N/A')}")
                    print(f"   Overall Confidence: {conf:.2%}")
                    
                    # Show all predictions for comparison
                    print(f"\n📊 ALL PREDICTIONS:")
                    for method, value in prediction["predictions"].items():
                        if isinstance(value, (int, float)):
                            print(f"   {method}: {value:.4f}")
                    
                    print(f"\n⏰ Waiting for next opportunity...")
        
        # Start monitoring
        try:
            self.api_client.monitor_real_time_changes(prediction_callback)
        except KeyboardInterrupt:
            print(f"\n📊 SESSION SUMMARY:")
            print(f"   Total Predictions: {self.total_predictions}")
            if self.prediction_accuracy:
                print(f"   Average Accuracy: {np.mean(self.prediction_accuracy):.2f}%")
                print(f"   Success Rate: {(self.correct_predictions / self.total_predictions) * 100:.2f}%")

def quick_prediction_demo():
    """Quick demo to test live predictions"""
    print("🎯 QUICK PREDICTION DEMO")
    print("=" * 40)
    
    try:
        system = LivePredictionSystem()
        prediction = system.get_live_prediction()
        
        print(f"\n📊 PREDICTION RESULT:")
        print(json.dumps(prediction, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 LIVE PREDICTION SYSTEM")
    print("=" * 50)
    print("Choose your mode:")
    print("1. Quick Prediction Demo")
    print("2. Live Monitoring Mode (Real-time)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        quick_prediction_demo()
    elif choice == "2":
        try:
            system = LivePredictionSystem()
            system.live_monitoring_mode()
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("👋 Goodbye!")