#!/usr/bin/env python3
"""
Stake Dice Game Predictor
Combines pattern analysis with Stake dice mechanics for prediction
"""

import sys
import os
import time
import requests
import json
import struct
import hashlib
import hmac
import random
import statistics
from typing import List, Dict, Any

# Try to import Z3 for advanced prediction, fall back to pattern analysis
try:
    import z3
    Z3_AVAILABLE = True
    print("🔧 Z3 solver available for advanced prediction")
except ImportError:
    Z3_AVAILABLE = False
    print("⚠️  Z3 not available, using pattern analysis")

class V8RandomnessPredictor:
    """Advanced randomness predictor using multiple techniques"""
    
    def __init__(self):
        self.known_values = []
        self.state = None
        self.pattern_model = None
        
    def xorshift128plus(self, s0, s1):
        """XorShift128+ algorithm"""
        s1 = s1 ^ (s1 << 23)
        s1 = s1 ^ (s0 ^ (s1 >> 17) ^ (s0 >> 26))
        return s1, (s0 + s1) & 0xffffffffffffffff
    
    def ieee754_to_double(self, bits):
        """Convert IEEE 754 bits to double"""
        return struct.unpack('>d', struct.pack('>Q', bits))[0]
    
    def double_to_random(self, double_val):
        """Convert double to [0,1) random value"""
        return (double_val - 1.0)
    
    def recover_state_z3(self, observed_values):
        """Recover state using Z3 solver"""
        if not Z3_AVAILABLE:
            return False
            
        if len(observed_values) < 2:
            print("Need at least 2 observed values to recover state")
            return False
            
        try:
            # Convert observed values back to internal representation
            internal_values = []
            for val in observed_values:
                # Reverse the random() function
                double_val = val + 1.0
                bits = struct.unpack('>Q', struct.pack('>d', double_val))[0]
                internal_values.append(bits)
            
            # Create Z3 solver
            solver = z3.Solver()
            
            # Define state variables
            s0 = z3.BitVec('s0', 64)
            s1 = z3.BitVec('s1', 64)
            
            # Add constraints for each observed value
            state0, state1 = s0, s1
            for i, target in enumerate(internal_values[:4]):  # Limit to prevent timeout
                state1_new = state1 ^ (state1 << 23)
                state1_new = state1_new ^ (state0 ^ (state1_new >> 17) ^ (state0 >> 26))
                output = (state0 + state1_new) & 0xffffffffffffffff
                
                # Add constraint
                solver.add(output == target)
                
                # Update state for next iteration
                state0, state1 = state1, state1_new
            
            # Check if satisfiable
            if solver.check() == z3.sat:
                model = solver.model()
                self.state = (model[s0].as_long(), model[s1].as_long())
                print(f"✅ Z3 recovered state: {self.state}")
                return True
            else:
                print("❌ Z3 could not recover state from given values")
                return False
                
        except Exception as e:
            print(f"❌ Z3 solver error: {e}")
            return False
    
    def recover_state_pattern(self, observed_values):
        """Recover patterns using statistical analysis"""
        if len(observed_values) < 5:
            print("Need at least 5 values for pattern analysis")
            return False
        
        try:
            self.known_values = observed_values.copy()
            
            # Analyze differences between consecutive values
            diffs = [observed_values[i+1] - observed_values[i] for i in range(len(observed_values)-1)]
            second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
            
            # Look for periodic patterns
            periods = []
            for period in range(2, min(10, len(observed_values)//2)):
                is_periodic = True
                for i in range(period, len(observed_values)-period):
                    if abs(observed_values[i] - observed_values[i-period]) > 0.001:
                        is_periodic = False
                        break
                if is_periodic:
                    periods.append(period)
            
            # Store pattern information
            self.pattern_model = {
                'values': observed_values,
                'diffs': diffs,
                'second_diffs': second_diffs,
                'periods': periods,
                'mean': statistics.mean(observed_values),
                'stdev': statistics.stdev(observed_values) if len(observed_values) > 1 else 0.1,
                'trend': (observed_values[-1] - observed_values[0]) / len(observed_values)
            }
            
            print(f"✅ Pattern analysis complete: {len(periods)} periods found")
            print(f"   Mean: {self.pattern_model['mean']:.4f}, StdDev: {self.pattern_model['stdev']:.4f}")
            print(f"   Trend: {self.pattern_model['trend']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pattern analysis error: {e}")
            return False
    
    def recover_state(self, observed_values):
        """Attempt to recover state using available methods"""
        # Try Z3 first, fall back to pattern analysis
        if Z3_AVAILABLE and self.recover_state_z3(observed_values):
            return True
        
        return self.recover_state_pattern(observed_values)
    
    def predict_next_z3(self, count=1):
        """Predict using Z3-recovered state"""
        if not self.state:
            return []
        
        predictions = []
        s0, s1 = self.state
        
        for _ in range(count):
            s1_new = s1 ^ (s1 << 23)
            s1_new = s1_new ^ (s0 ^ (s1_new >> 17) ^ (s0 >> 26))
            output = (s0 + s1_new) & 0xffffffffffffffff
            
            # Convert to double
            double_val = self.ieee754_to_double(output | (0x3FF << 52))
            random_val = self.double_to_random(double_val)
            
            predictions.append(max(0.0, min(1.0, random_val)))  # Clamp to [0,1]
            
            # Update state
            s0, s1 = s1, s1_new
        
        # Update internal state
        self.state = (s0, s1)
        return predictions
    
    def predict_next_pattern(self, count=1):
        """Predict using pattern analysis"""
        if not self.pattern_model:
            return []
        
        predictions = []
        model = self.pattern_model
        last_values = model['values'][-5:]  # Use last 5 values for prediction
        
        for i in range(count):
            # Use different prediction strategies
            if model['periods']:
                # Use periodic pattern
                period = model['periods'][0]
                base_idx = len(model['values']) - period + (i % period)
                if base_idx < len(model['values']):
                    pred = model['values'][base_idx]
                else:
                    pred = model['mean']
            else:
                # Use trend and noise
                trend_component = model['trend'] * (len(model['values']) + i)
                noise_component = random.gauss(0, model['stdev'] * 0.1)
                pred = model['mean'] + trend_component + noise_component
            
            # Add some randomness based on differences
            if model['diffs']:
                last_diff = model['diffs'][-1]
                pred += last_diff * 0.1 + random.gauss(0, model['stdev'] * 0.05)
            
            # Clamp to valid range
            pred = max(0.0, min(1.0, pred))
            predictions.append(pred)
            
            # Update model with prediction for next iteration
            model['values'].append(pred)
            if len(model['values']) > 100:  # Limit memory
                model['values'] = model['values'][-50:]
        
        return predictions
    
    def predict_next(self, count=1):
        """Predict next random values using available method"""
        if self.state and Z3_AVAILABLE:
            return self.predict_next_z3(count)
        elif self.pattern_model:
            return self.predict_next_pattern(count)
        else:
            print("❌ No prediction model available")
            return []

class StakeDiceConverter:
    """Convert hash/random values to Stake dice outcomes"""
    
    @staticmethod
    def hash_to_dice(client_seed, server_seed, nonce, game_type="dice"):
        """Convert seeds to dice outcome using Stake's algorithm"""
        # Create HMAC-SHA512
        message = f"{client_seed}-{nonce}"
        hash_value = hmac.new(
            server_seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        # Convert to random number
        hash_int = int(hash_value[:8], 16)  # Use first 8 hex chars
        random_val = hash_int / 0x100000000  # Convert to [0,1)
        
        if game_type == "dice":
            # Dice game: multiply by 100 for percentage
            return random_val * 100
        elif game_type == "limbo":
            # Limbo game calculation
            return max(1.0, 1.0 / (1.0 - random_val))
        
        return random_val
    
    @staticmethod
    def random_to_dice(random_val, game_type="dice", multiplier=2.0):
        """Convert raw random value to dice outcome"""
        if game_type == "dice":
            dice_result = random_val * 100
            return {
                'result': dice_result,
                'win': dice_result >= (100 / multiplier),
                'multiplier': multiplier,
                'prediction': f"{dice_result:.2f}"
            }
        elif game_type == "limbo":
            limbo_result = max(1.0, 1.0 / (1.0 - random_val)) if random_val < 1.0 else 1000000
            return {
                'result': limbo_result,
                'win': limbo_result >= multiplier,
                'multiplier': multiplier,
                'prediction': f"{limbo_result:.2f}x"
            }
        
        return {'result': random_val, 'prediction': str(random_val)}

class StakeDicePredictor:
    """Main prediction system combining V8 predictor with Stake mechanics"""
    
    def __init__(self):
        self.v8_predictor = V8RandomnessPredictor()
        self.dice_converter = StakeDiceConverter()
        self.prediction_history = []
        
    def train_from_observed_results(self, observed_results):
        """Train predictor from observed Stake dice results"""
        print(f"Training from {len(observed_results)} observed results...")
        
        # Convert dice results back to random values
        random_values = []
        for result in observed_results:
            if isinstance(result, dict):
                if 'result' in result:
                    random_val = result['result'] / 100.0  # Dice percentage to random
                else:
                    random_val = result.get('random', 0.5)
            else:
                random_val = float(result) / 100.0
            
            random_values.append(random_val)
        
        # Attempt to recover V8 state
        success = self.v8_predictor.recover_state(random_values)
        if success:
            print("✅ Successfully trained predictor!")
            return True
        else:
            print("❌ Failed to recover randomness pattern")
            return False
    
    def predict_next_dice(self, count=5, game_type="dice", multiplier=2.0):
        """Predict next dice rolls"""
        print(f"\n🎲 Predicting next {count} {game_type} results...")
        
        # Get predictions from V8 predictor
        random_predictions = self.v8_predictor.predict_next(count)
        
        if not random_predictions:
            print("❌ No predictions available - train first!")
            return []
        
        predictions = []
        for i, random_val in enumerate(random_predictions):
            dice_result = self.dice_converter.random_to_dice(
                random_val, game_type, multiplier
            )
            dice_result['sequence'] = i + 1
            predictions.append(dice_result)
            
            # Display prediction
            status = "🟢 WIN" if dice_result['win'] else "🔴 LOSE"
            print(f"  Roll {i+1}: {dice_result['prediction']} - {status}")
        
        self.prediction_history.extend(predictions)
        return predictions
    
    def simulate_stake_session(self, server_seed="default_server", client_seed="player123", start_nonce=1, count=10):
        """Simulate a Stake session with known seeds"""
        print(f"\n🎰 Simulating Stake session...")
        print(f"Server Seed: {server_seed}")
        print(f"Client Seed: {client_seed}")
        print(f"Starting Nonce: {start_nonce}")
        
        results = []
        for nonce in range(start_nonce, start_nonce + count):
            dice_result = self.dice_converter.hash_to_dice(
                client_seed, server_seed, nonce, "dice"
            )
            
            result = {
                'nonce': nonce,
                'result': dice_result,
                'random': dice_result / 100.0,
                'hash_based': True
            }
            results.append(result)
            
            print(f"  Nonce {nonce}: {dice_result:.2f}%")
        
        return results
    
    def analyze_patterns(self):
        """Analyze patterns in prediction history"""
        if len(self.prediction_history) < 5:
            print("Need more predictions to analyze patterns")
            return
        
        results = [p['result'] for p in self.prediction_history]
        wins = [p for p in self.prediction_history if p['win']]
        
        print(f"\n📊 Pattern Analysis:")
        print(f"  Total Predictions: {len(results)}")
        print(f"  Wins: {len(wins)} ({len(wins)/len(results)*100:.1f}%)")
        print(f"  Average Result: {sum(results)/len(results):.2f}")
        print(f"  Min/Max: {min(results):.2f} / {max(results):.2f}")
        
        # Look for streaks
        win_streak = 0
        max_win_streak = 0
        for p in self.prediction_history:
            if p['win']:
                win_streak += 1
                max_win_streak = max(max_win_streak, win_streak)
            else:
                win_streak = 0
        
        print(f"  Max Win Streak: {max_win_streak}")

def main():
    """Main demonstration"""
    print("🎲 Stake Dice Predictor v1.0")
    print("=" * 50)
    
    predictor = StakeDicePredictor()
    
    while True:
        print("\nOptions:")
        print("1. Simulate Stake session (known seeds)")
        print("2. Train from observed results")
        print("3. Predict next dice rolls")
        print("4. Predict Limbo game")
        print("5. Analyze patterns")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                # Simulate session
                server_seed = input("Server seed (or Enter for default): ").strip() or "default_server"
                client_seed = input("Client seed (or Enter for default): ").strip() or "player123"
                count = int(input("Number of rolls (default 10): ") or "10")
                
                results = predictor.simulate_stake_session(server_seed, client_seed, 1, count)
                
                # Auto-train from simulation
                if input("\nTrain predictor from these results? (y/n): ").lower() == 'y':
                    predictor.train_from_observed_results(results)
            
            elif choice == '2':
                # Manual training
                print("\nEnter observed dice results (0-100), one per line.")
                print("Type 'done' when finished:")
                
                observed = []
                while True:
                    value = input("Result: ").strip()
                    if value.lower() == 'done':
                        break
                    try:
                        observed.append(float(value))
                    except ValueError:
                        print("Invalid number, try again")
                
                if len(observed) >= 2:
                    predictor.train_from_observed_results(observed)
                else:
                    print("Need at least 2 results to train")
            
            elif choice == '3':
                # Predict dice
                count = int(input("Number of predictions (default 5): ") or "5")
                multiplier = float(input("Target multiplier (default 2.0): ") or "2.0")
                
                predictions = predictor.predict_next_dice(count, "dice", multiplier)
                
                if predictions:
                    print(f"\n💰 Betting Strategy:")
                    win_predictions = [p for p in predictions if p['win']]
                    if win_predictions:
                        print(f"  Recommended bets: Rolls {[p['sequence'] for p in win_predictions]}")
                        print(f"  Expected win rate: {len(win_predictions)/len(predictions)*100:.1f}%")
                    else:
                        print("  No winning predictions - consider waiting or lower multiplier")
            
            elif choice == '4':
                # Predict Limbo
                count = int(input("Number of predictions (default 5): ") or "5")
                multiplier = float(input("Target multiplier (default 2.0): ") or "2.0")
                
                predictions = predictor.predict_next_dice(count, "limbo", multiplier)
                
                if predictions:
                    print(f"\n🚀 Limbo Strategy:")
                    win_predictions = [p for p in predictions if p['win']]
                    if win_predictions:
                        print(f"  Safe rounds: {[p['sequence'] for p in win_predictions]}")
                        expected_mult = sum(p['result'] for p in win_predictions) / len(win_predictions)
                        print(f"  Average multiplier: {expected_mult:.2f}x")
            
            elif choice == '5':
                # Analyze patterns
                predictor.analyze_patterns()
            
            elif choice == '6':
                print("👋 Goodbye!")
                break
            
            else:
                print("Invalid choice, try again")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()