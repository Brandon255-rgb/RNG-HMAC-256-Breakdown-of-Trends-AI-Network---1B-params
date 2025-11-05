#!/usr/bin/env python3
"""
ULTIMATE STAKE PREDICTOR
Combines hash analysis, massive data analysis, and AI pattern detection
for maximum prediction accuracy on Stake dice games
"""

import sys
import os
import numpy as np
import torch
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque
import threading
from datetime import datetime

# Import all our components
from realtime_stake_predictor import StakeRealTimePredictor
from massive_stake_analyzer import StakeMassiveDataGenerator, StakePatternAI, StakeLivePredictor
from stake_dice_predictor import StakeDiceConverter, V8RandomnessPredictor

class UltimateStakePredictor:
    """
    Ultimate prediction system combining:
    1. Hash-based prediction (deterministic)
    2. AI pattern detection (learned patterns)  
    3. V8 randomness prediction (mathematical)
    4. Real-time adaptation (screen input)
    """
    
    def __init__(self):
        print("🚀 INITIALIZING ULTIMATE STAKE PREDICTOR")
        print("=" * 60)
        
        # Initialize all components
        self.hash_converter = StakeDiceConverter()
        self.v8_predictor = V8RandomnessPredictor()
        self.ai_predictor = StakePatternAI()
        self.realtime_predictor = StakeRealTimePredictor()
        
        # Prediction ensemble weights
        self.weights = {
            'hash': 0.3,      # Hash-based predictions
            'ai': 0.4,        # AI learned patterns  
            'v8': 0.2,        # V8 mathematical prediction
            'pattern': 0.1    # Real-time pattern analysis
        }
        
        # Performance tracking
        self.predictions_made = []
        self.accuracy_history = []
        
        print("✅ All prediction systems loaded!")
        print(f"🧠 AI Model: {'✅ Ready' if self.realtime_predictor.live_predictor.model_loaded else '⚠️  Needs Training'}")
        print(f"🔢 V8 Predictor: {'✅ Ready' if hasattr(self.v8_predictor, 'state') else '⚠️  Needs State Recovery'}")
        
    def predict_with_hash_seeds(self, 
                               client_seed: str, 
                               server_seed: str, 
                               start_nonce: int, 
                               count: int = 10) -> List[Dict]:
        """Predict using known seeds (most accurate when seeds are known)"""
        print(f"\n🎯 HASH-BASED PREDICTIONS (Most Accurate)")
        print(f"Client Seed: {client_seed}")
        print(f"Server Seed: {server_seed}")
        print(f"Starting Nonce: {start_nonce}")
        
        predictions = []
        for i in range(count):
            nonce = start_nonce + i
            dice_result = self.hash_converter.hash_to_dice(client_seed, server_seed, nonce, "dice")
            
            prediction = {
                'method': 'hash',
                'nonce': nonce,
                'predicted_value': dice_result,
                'confidence': 1.0,  # Perfect confidence with known seeds
                'sequence': i + 1
            }
            predictions.append(prediction)
            
            print(f"  Nonce {nonce}: {dice_result:.2f}")
            
        return predictions
    
    def predict_with_ai(self, recent_results: List[float], count: int = 5) -> List[Dict]:
        """Predict using AI model"""
        if not self.realtime_predictor.live_predictor.model_loaded:
            print("⚠️  AI model not loaded")
            return []
            
        print(f"\n🤖 AI PREDICTIONS")
        ai_predictions = self.realtime_predictor.live_predictor.predict_next_numbers(recent_results, count)
        
        for pred in ai_predictions:
            pred['method'] = 'ai'
            
        return ai_predictions
    
    def predict_with_v8(self, count: int = 5) -> List[Dict]:
        """Predict using V8 randomness recovery"""
        if not hasattr(self.v8_predictor, 'state') or self.v8_predictor.state is None:
            print("⚠️  V8 state not recovered")
            return []
            
        print(f"\n🔧 V8 MATHEMATICAL PREDICTIONS")
        v8_predictions = self.v8_predictor.predict_next(count)
        
        predictions = []
        for i, pred in enumerate(v8_predictions):
            dice_result = pred * 100  # Convert to dice range
            predictions.append({
                'method': 'v8',
                'predicted_value': dice_result,
                'confidence': 0.8,
                'sequence': i + 1,
                'raw_random': pred
            })
            print(f"  Prediction {i+1}: {dice_result:.2f}")
            
        return predictions
    
    def predict_with_patterns(self, recent_results: List[float], count: int = 5) -> List[Dict]:
        """Predict using pattern analysis"""
        print(f"\n📈 PATTERN-BASED PREDICTIONS")
        pattern_predictions = self.realtime_predictor.pattern_based_predictions(count)
        
        predictions = []
        for i, pred in enumerate(pattern_predictions):
            predictions.append({
                'method': 'pattern',
                'predicted_value': pred['value'],
                'confidence': 0.6,  # Lower confidence for patterns
                'sequence': i + 1,
                'pattern_method': pred['method'],
                'variance': pred['variance']
            })
            print(f"  Pattern {i+1}: {pred['value']:.2f} ({pred['method']})")
            
        return predictions
    
    def ensemble_prediction(self, 
                          recent_results: List[float] = None,
                          client_seed: str = None,
                          server_seed: str = None,
                          current_nonce: int = None,
                          count: int = 5) -> List[Dict]:
        """
        Ultimate ensemble prediction combining all methods
        """
        print(f"\n🌟 ULTIMATE ENSEMBLE PREDICTION")
        print("=" * 50)
        
        all_predictions = {}
        
        # 1. Hash-based predictions (if seeds known)
        if client_seed and server_seed and current_nonce:
            hash_preds = self.predict_with_hash_seeds(client_seed, server_seed, current_nonce + 1, count)
            all_predictions['hash'] = hash_preds
        
        # 2. AI predictions (if recent data available)
        if recent_results and len(recent_results) >= 10:
            ai_preds = self.predict_with_ai(recent_results, count)
            if ai_preds:
                all_predictions['ai'] = ai_preds
        
        # 3. V8 predictions (if state recovered)
        v8_preds = self.predict_with_v8(count)
        if v8_preds:
            all_predictions['v8'] = v8_preds
            
        # 4. Pattern predictions (if recent data available)
        if recent_results and len(recent_results) >= 5:
            # Add recent results to realtime predictor
            for result in recent_results[-20:]:  # Last 20 results
                self.realtime_predictor.observation_history.append(result)
            
            pattern_preds = self.predict_with_patterns(recent_results, count)
            all_predictions['pattern'] = pattern_preds
        
        # Combine all predictions
        ensemble_results = []
        
        for seq in range(count):
            weighted_sum = 0
            total_weight = 0
            confidence_sum = 0
            methods_used = []
            
            # Collect predictions for this sequence
            seq_predictions = {}
            for method, predictions in all_predictions.items():
                if seq < len(predictions):
                    pred = predictions[seq]
                    seq_predictions[method] = pred
                    methods_used.append(method)
            
            if not seq_predictions:
                continue  # Skip if no predictions available
                
            # Calculate weighted average
            for method, pred in seq_predictions.items():
                weight = self.weights.get(method, 0.1)
                value = pred['predicted_value']
                confidence = pred.get('confidence', 0.5)
                
                weighted_sum += value * weight * confidence
                total_weight += weight * confidence
                confidence_sum += confidence
            
            if total_weight > 0:
                final_prediction = weighted_sum / total_weight
                avg_confidence = confidence_sum / len(seq_predictions)
                
                # Determine confidence level
                if avg_confidence > 0.8 and len(methods_used) >= 3:
                    conf_level = "🟢 VERY HIGH"
                elif avg_confidence > 0.6 and len(methods_used) >= 2:
                    conf_level = "🟢 HIGH"
                elif avg_confidence > 0.4:
                    conf_level = "🟡 MEDIUM"
                else:
                    conf_level = "🔴 LOW"
                
                ensemble_result = {
                    'sequence': seq + 1,
                    'predicted_value': final_prediction,
                    'confidence': avg_confidence,
                    'confidence_level': conf_level,
                    'methods_used': methods_used,
                    'individual_predictions': seq_predictions,
                    'ensemble_weight': total_weight
                }
                
                ensemble_results.append(ensemble_result)
                
                # Display result
                methods_str = "+".join(methods_used)
                print(f"🎯 #{seq+1}: {final_prediction:.1f} - {conf_level} [{methods_str}]")
        
        # Generate final recommendation
        self.generate_ultimate_strategy(ensemble_results)
        
        return ensemble_results
    
    def generate_ultimate_strategy(self, predictions: List[Dict]):
        """Generate ultimate betting strategy"""
        if not predictions:
            print("❌ No predictions available for strategy")
            return
            
        print(f"\n💎 ULTIMATE BETTING STRATEGY")
        print("=" * 40)
        
        # High confidence predictions
        high_conf = [p for p in predictions if p['confidence'] > 0.7]
        medium_conf = [p for p in predictions if 0.5 <= p['confidence'] <= 0.7]
        
        if high_conf:
            print(f"🔥 PRIORITY BETS ({len(high_conf)} high confidence):")
            for pred in high_conf[:3]:  # Top 3
                value = pred['predicted_value']
                methods = "+".join(pred['methods_used'])
                print(f"   Round {pred['sequence']}: {value:.1f} [{methods}]")
                
                # Specific betting advice
                if value < 20:
                    print(f"      💡 Strong Under 25 bet")
                elif value > 80:
                    print(f"      💡 Strong Over 75 bet")
                elif 45 <= value <= 55:
                    print(f"      💡 Around 50 range bet ({value-5:.0f}-{value+5:.0f})")
                else:
                    print(f"      💡 Range bet {value-8:.0f}-{value+8:.0f}")
                    
        if medium_conf:
            print(f"\n⚡ SECONDARY BETS ({len(medium_conf)} medium confidence):")
            for pred in medium_conf[:2]:  # Top 2
                value = pred['predicted_value']
                print(f"   Round {pred['sequence']}: {value:.1f} (backup option)")
                
        # Overall strategy
        if high_conf:
            avg_high = np.mean([p['predicted_value'] for p in high_conf])
            print(f"\n🎰 OVERALL STRATEGY:")
            print(f"   Primary focus: {avg_high:.1f} range")
            print(f"   Confidence: {np.mean([p['confidence'] for p in high_conf]):.1%}")
            print(f"   Risk level: {'LOW' if len(high_conf) >= 3 else 'MEDIUM'}")
        else:
            print(f"\n⚠️  WAIT RECOMMENDATION: No high-confidence predictions")
            print(f"   Consider gathering more data before betting")
    
    def train_system(self, total_rolls: int = 100000):
        """Train all AI components"""
        print(f"🏋️ TRAINING ULTIMATE SYSTEM")
        print("=" * 40)
        
        # Generate massive training dataset
        generator = StakeMassiveDataGenerator()
        dataset_file = generator.generate_massive_dataset(total_rolls=total_rolls)
        
        # Train AI components
        print("🧠 Training AI pattern detection...")
        self.ai_predictor.train_on_massive_data(dataset_file, epochs=30)
        
        # Reload realtime predictor with trained model
        self.realtime_predictor = StakeRealTimePredictor()
        
        print("✅ System training complete!")
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print(f"\n🎮 INTERACTIVE ULTIMATE PREDICTOR")
        print("=" * 50)
        print("Commands:")
        print("  'observe [value]' - Add observed result")
        print("  'predict' - Get ultimate predictions")
        print("  'seeds [client] [server] [nonce]' - Predict with known seeds")
        print("  'train [count]' - Train system with data")
        print("  'analyze' - Analyze current data")
        print("  'quit' - Exit")
        
        recent_observations = []
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                
                if not cmd:
                    continue
                    
                if cmd[0] == 'quit':
                    break
                    
                elif cmd[0] == 'observe':
                    if len(cmd) > 1:
                        try:
                            value = float(cmd[1])
                            if 0 <= value <= 100:
                                recent_observations.append(value)
                                print(f"✅ Added observation: {value} (Total: {len(recent_observations)})")
                            else:
                                print("❌ Value must be 0-100")
                        except ValueError:
                            print("❌ Invalid number")
                    else:
                        print("Usage: observe [value]")
                        
                elif cmd[0] == 'predict':
                    count = 5
                    if len(cmd) > 1:
                        try:
                            count = int(cmd[1])
                        except:
                            pass
                    
                    self.ensemble_prediction(
                        recent_results=recent_observations,
                        count=count
                    )
                    
                elif cmd[0] == 'seeds':
                    if len(cmd) >= 4:
                        client_seed = cmd[1]
                        server_seed = cmd[2]
                        try:
                            nonce = int(cmd[3])
                            count = int(cmd[4]) if len(cmd) > 4 else 5
                            
                            self.ensemble_prediction(
                                recent_results=recent_observations,
                                client_seed=client_seed,
                                server_seed=server_seed,
                                current_nonce=nonce,
                                count=count
                            )
                        except ValueError:
                            print("❌ Invalid nonce number")
                    else:
                        print("Usage: seeds [client] [server] [nonce] [count?]")
                        
                elif cmd[0] == 'train':
                    count = 100000
                    if len(cmd) > 1:
                        try:
                            count = int(cmd[1])
                        except:
                            pass
                    self.train_system(count)
                    
                elif cmd[0] == 'analyze':
                    if recent_observations:
                        print(f"\n📊 CURRENT DATA ANALYSIS:")
                        print(f"  Observations: {len(recent_observations)}")
                        print(f"  Range: {min(recent_observations):.1f} - {max(recent_observations):.1f}")
                        print(f"  Average: {np.mean(recent_observations):.1f}")
                        print(f"  Last 5: {recent_observations[-5:]}")
                        
                        if len(recent_observations) >= 10:
                            # Quick pattern check
                            autocorr = np.corrcoef(recent_observations[:-1], recent_observations[1:])[0, 1]
                            trend = np.mean(recent_observations[-5:]) - np.mean(recent_observations[-10:-5]) if len(recent_observations) >= 10 else 0
                            print(f"  Autocorr: {autocorr:.3f}")
                            print(f"  Recent trend: {trend:+.1f}")
                    else:
                        print("❌ No observations to analyze")
                        
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main interface"""
    print("🌟 ULTIMATE STAKE PREDICTOR v2.0")
    print("=" * 60)
    print("🚀 Maximum Accuracy Ensemble System")
    print("🔥 Hash + AI + V8 + Pattern Analysis")
    print("💎 Real-time Adaptation & Training")
    print()
    
    predictor = UltimateStakePredictor()
    
    print("\nOptions:")
    print("1. Interactive mode (RECOMMENDED)")
    print("2. Train system with massive data")
    print("3. Quick prediction test")
    print("4. Demo with sample seeds")
    
    try:
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            predictor.interactive_mode()
            
        elif choice == '2':
            rolls = int(input("Training data size (default 100000): ") or "100000")
            predictor.train_system(rolls)
            predictor.interactive_mode()
            
        elif choice == '3':
            # Quick test
            print("Quick test - enter 5 recent results:")
            test_results = []
            for i in range(5):
                result = input(f"Result {i+1}: ")
                try:
                    test_results.append(float(result))
                except:
                    test_results.append(50.0)
            
            predictor.ensemble_prediction(recent_results=test_results)
            
        elif choice == '4':
            # Demo with sample seeds
            print("🎮 Demo with sample Stake seeds:")
            predictor.ensemble_prediction(
                recent_results=[45.2, 67.8, 23.1, 89.4, 12.7, 55.9, 78.3, 34.6, 91.2, 18.5],
                client_seed="demo_client",
                server_seed="demo_server",
                current_nonce=1000,
                count=10
            )
            
        else:
            print("Starting interactive mode...")
            predictor.interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()