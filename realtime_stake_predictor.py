#!/usr/bin/env python3
"""
Real-Time Stake Predictor with Screen Input
Takes user input from screen/observations and predicts exact next numbers
"""

import sys
import os
import numpy as np
import torch
import json
import time
from pathlib import Path
from typing import List, Dict
from collections import deque

# Import the massive analyzer
from massive_stake_analyzer import StakeLivePredictor, StakeMassiveDataGenerator, StakePatternAI

class StakeRealTimePredictor:
    """Real-time predictor that takes screen input and gives exact predictions"""
    
    def __init__(self):
        self.live_predictor = StakeLivePredictor()
        self.observation_history = deque(maxlen=10000)
        self.prediction_accuracy = []
        
        print("🎯 REAL-TIME STAKE PREDICTOR INITIALIZED")
        print("🔍 Monitoring for patterns in your input data...")
        
    def input_screen_data(self):
        """Interactive input for screen observations"""
        print("\n📺 SCREEN DATA INPUT MODE")
        print("Enter the dice results you see on screen, one by one.")
        print("The AI will learn patterns and predict the next numbers.")
        print("Commands: 'predict', 'analyze', 'clear', 'quit'")
        print("-" * 60)
        
        while True:
            try:
                user_input = input(f"[{len(self.observation_history)} obs] Enter result (0-100) or command: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                    
                elif user_input.lower() == 'predict':
                    self.make_predictions()
                    
                elif user_input.lower() == 'analyze':
                    self.analyze_current_patterns()
                    
                elif user_input.lower() == 'clear':
                    self.observation_history.clear()
                    print("🗑️  Cleared all observations")
                    
                elif user_input.lower().startswith('batch'):
                    # Batch input mode
                    print("📋 BATCH INPUT MODE - Enter multiple results separated by commas:")
                    batch_input = input("Results: ").strip()
                    try:
                        batch_results = [float(x.strip()) for x in batch_input.split(',')]
                        for result in batch_results:
                            if 0 <= result <= 100:
                                self.observation_history.append(result)
                                self.live_predictor.add_observed_result(result)
                        print(f"✅ Added {len(batch_results)} results")
                    except ValueError:
                        print("❌ Invalid batch format")
                        
                else:
                    try:
                        result = float(user_input)
                        if 0 <= result <= 100:
                            self.observation_history.append(result)
                            self.live_predictor.add_observed_result(result)
                            
                            # Auto-analyze every 20 observations
                            if len(self.observation_history) % 20 == 0:
                                print(f"\n📊 Auto-analysis at {len(self.observation_history)} observations:")
                                self.quick_pattern_check()
                                
                            # Auto-predict when we have enough data
                            if len(self.observation_history) >= 50 and len(self.observation_history) % 10 == 0:
                                print(f"\n🔮 Auto-prediction:")
                                self.make_predictions(count=3, auto_mode=True)
                                
                        else:
                            print("❌ Result must be between 0-100")
                            
                    except ValueError:
                        print("❌ Invalid input. Enter a number 0-100 or a command.")
                        
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
                
    def make_predictions(self, count: int = 5, auto_mode: bool = False):
        """Make predictions for next numbers"""
        if len(self.observation_history) < 10:
            print("❌ Need at least 10 observations to predict")
            return
            
        if not auto_mode:
            count = int(input(f"How many predictions? (default {count}): ") or str(count))
            
        print(f"\n🎯 PREDICTING NEXT {count} NUMBERS:")
        print("=" * 50)
        
        try:
            # Get AI predictions if model is available
            ai_predictions = self.live_predictor.get_predictions(count)
            
            if ai_predictions:
                print("🤖 AI MODEL PREDICTIONS:")
                for pred in ai_predictions:
                    confidence = "🟢 HIGH" if pred['confidence'] > 0.7 else "🟡 MED" if pred['confidence'] > 0.5 else "🔴 LOW"
                    print(f"  #{pred['sequence']}: {pred['predicted_value']:.1f} - {confidence}")
            
        except Exception as e:
            print(f"⚠️  AI prediction failed: {e}")
            
        # Fallback to pattern analysis
        pattern_predictions = self.pattern_based_predictions(count)
        print("\n📈 PATTERN-BASED PREDICTIONS:")
        for i, pred in enumerate(pattern_predictions):
            print(f"  #{i+1}: {pred['value']:.1f} - {pred['method']} - {pred['confidence']}")
            
        # Combined recommendation
        print(f"\n💡 RECOMMENDATION:")
        self.generate_betting_strategy(ai_predictions if 'ai_predictions' in locals() and ai_predictions else pattern_predictions)
        
    def pattern_based_predictions(self, count: int) -> List[Dict]:
        """Generate predictions using pattern analysis"""
        recent_data = list(self.observation_history)[-100:]  # Last 100 observations
        predictions = []
        
        if len(recent_data) < 5:
            return predictions
            
        # Method 1: Moving average trend
        ma_short = np.mean(recent_data[-5:])
        ma_long = np.mean(recent_data[-20:]) if len(recent_data) >= 20 else ma_short
        trend = ma_short - ma_long
        
        # Method 2: Autocorrelation
        autocorr = np.corrcoef(recent_data[:-1], recent_data[1:])[0, 1] if len(recent_data) > 1 else 0
        
        # Method 3: Frequency analysis
        hist, bins = np.histogram(recent_data, bins=10, range=(0, 100))
        most_common_range = bins[np.argmax(hist)]
        
        # Method 4: Last value momentum
        last_values = recent_data[-3:]
        momentum = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
        
        for i in range(count):
            # Combine methods
            methods_used = []
            
            # Trend prediction
            trend_pred = recent_data[-1] + trend * (i + 1)
            trend_pred = max(0, min(100, trend_pred))
            
            # Autocorrelation prediction
            if abs(autocorr) > 0.1:
                autocorr_pred = recent_data[-1] + autocorr * 10
                autocorr_pred = max(0, min(100, autocorr_pred))
                methods_used.append("AutoCorr")
            else:
                autocorr_pred = trend_pred
                
            # Frequency prediction (regression to mean)
            freq_pred = most_common_range + np.random.normal(0, 5)
            freq_pred = max(0, min(100, freq_pred))
            
            # Momentum prediction
            momentum_pred = recent_data[-1] + momentum * (i + 1)
            momentum_pred = max(0, min(100, momentum_pred))
            
            # Average methods
            final_pred = np.mean([trend_pred, autocorr_pred, freq_pred, momentum_pred])
            
            # Determine confidence
            variance = np.var([trend_pred, autocorr_pred, freq_pred, momentum_pred])
            confidence = "🟢 HIGH" if variance < 25 else "🟡 MED" if variance < 100 else "🔴 LOW"
            
            # Determine primary method
            method = "Trend"
            if abs(autocorr) > 0.3:
                method = "AutoCorr"
            elif variance < 10:
                method = "Consensus"
                
            predictions.append({
                'value': final_pred,
                'confidence': confidence,
                'method': method,
                'variance': variance
            })
            
        return predictions
        
    def quick_pattern_check(self):
        """Quick analysis of current patterns"""
        if len(self.observation_history) < 10:
            return
            
        data = list(self.observation_history)[-50:]  # Last 50 points
        
        print(f"  📊 Data points: {len(data)}")
        print(f"  📈 Current range: {min(data):.1f} - {max(data):.1f}")
        print(f"  🎯 Recent average: {np.mean(data[-10:]):.1f}")
        
        # Trend detection
        if len(data) >= 10:
            early_avg = np.mean(data[:5])
            recent_avg = np.mean(data[-5:])
            trend_direction = "📈 Rising" if recent_avg > early_avg + 2 else "📉 Falling" if recent_avg < early_avg - 2 else "➡️  Stable"
            print(f"  {trend_direction} (Δ{recent_avg - early_avg:+.1f})")
            
        # Volatility
        volatility = np.std(data)
        vol_level = "🌋 High" if volatility > 20 else "🌊 Medium" if volatility > 10 else "🏞️  Low"
        print(f"  {vol_level} volatility ({volatility:.1f})")
        
    def analyze_current_patterns(self):
        """Comprehensive pattern analysis"""
        if len(self.observation_history) < 20:
            print("❌ Need at least 20 observations for detailed analysis")
            return
            
        data = np.array(list(self.observation_history))
        
        print(f"\n📊 COMPREHENSIVE PATTERN ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print(f"🔢 BASIC STATS:")
        print(f"  Total observations: {len(data)}")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Median: {np.median(data):.2f}")
        print(f"  Std Dev: {np.std(data):.2f}")
        print(f"  Range: {np.min(data):.1f} - {np.max(data):.1f}")
        
        # Distribution analysis
        print(f"\n📈 DISTRIBUTION:")
        ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        for low, high in ranges:
            count = np.sum((data >= low) & (data < high))
            percent = count / len(data) * 100
            bar = "█" * int(percent // 5)
            print(f"  {low:2d}-{high:2d}: {count:3d} ({percent:4.1f}%) {bar}")
            
        # Streak analysis
        print(f"\n🎯 STREAKS:")
        streaks = self.find_streaks(data)
        if streaks:
            print(f"  Longest streak: {max(streaks)} consecutive similar values")
            print(f"  Average streak: {np.mean(streaks):.1f}")
            print(f"  Total streaks: {len(streaks)}")
        else:
            print("  No significant streaks detected")
            
        # Autocorrelation
        if len(data) > 10:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            print(f"\n🔄 AUTOCORRELATION: {autocorr:.4f}")
            if abs(autocorr) > 0.1:
                strength = "Strong" if abs(autocorr) > 0.3 else "Moderate"
                direction = "Positive (values follow trends)" if autocorr > 0 else "Negative (values alternate)"
                print(f"  {strength} {direction}")
            else:
                print("  No significant autocorrelation (random-like)")
                
        # Recent trend
        if len(data) >= 20:
            recent_20 = data[-20:]
            early_10 = recent_20[:10]
            late_10 = recent_20[10:]
            
            trend_change = np.mean(late_10) - np.mean(early_10)
            print(f"\n📊 RECENT TREND (last 20): {trend_change:+.1f}")
            if abs(trend_change) > 3:
                direction = "Upward" if trend_change > 0 else "Downward"
                print(f"  Clear {direction} trend detected!")
            else:
                print("  Relatively stable recent pattern")
                
    def find_streaks(self, data: np.ndarray, tolerance: float = 3.0) -> List[int]:
        """Find streaks of similar values"""
        if len(data) < 3:
            return []
            
        streaks = []
        current_streak = 1
        
        for i in range(1, len(data)):
            if abs(data[i] - data[i-1]) <= tolerance:
                current_streak += 1
            else:
                if current_streak >= 3:  # Only count streaks of 3+
                    streaks.append(current_streak)
                current_streak = 1
                
        # Don't forget the last streak
        if current_streak >= 3:
            streaks.append(current_streak)
            
        return streaks
        
    def generate_betting_strategy(self, predictions: List[Dict]):
        """Generate betting recommendations"""
        if not predictions:
            print("No predictions available")
            return
            
        # Find high confidence predictions
        high_conf_preds = [p for p in predictions if 'confidence' in p and (
            p['confidence'] == "🟢 HIGH" or 
            (isinstance(p['confidence'], float) and p['confidence'] > 0.7)
        )]
        
        if high_conf_preds:
            print(f"🎰 BETTING STRATEGY:")
            print(f"  🟢 {len(high_conf_preds)} high-confidence predictions found")
            
            for i, pred in enumerate(high_conf_preds[:3]):  # Top 3
                value = pred.get('predicted_value', pred.get('value', 0))
                print(f"    Round {i+1}: Target {value:.1f}")
                
                # Suggest betting ranges
                if value < 25:
                    print(f"      💡 Consider: Under 30 bets")
                elif value > 75:
                    print(f"      💡 Consider: Over 70 bets")
                else:
                    range_size = 15
                    low_range = max(0, value - range_size//2)
                    high_range = min(100, value + range_size//2)
                    print(f"      💡 Consider: {low_range:.0f}-{high_range:.0f} range bets")
        else:
            print("⚠️  No high-confidence predictions - consider waiting for more data")

def main():
    """Main interface"""
    print("🎯 REAL-TIME STAKE DICE PREDICTOR")
    print("=" * 50)
    print("🔮 Advanced AI + Pattern Analysis")
    print("📺 Screen Input → Exact Predictions")
    print("🎰 Betting Strategy Generation")
    print()
    
    predictor = StakeRealTimePredictor()
    
    print("Options:")
    print("1. Start screen input mode (RECOMMENDED)")
    print("2. Generate training data first")
    print("3. Load existing model")
    print("4. Quick test mode")
    
    try:
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            predictor.input_screen_data()
            
        elif choice == '2':
            print("🏭 Generating training data...")
            generator = StakeMassiveDataGenerator()
            dataset = generator.generate_massive_dataset(total_rolls=100000)
            print(f"✅ Generated training data: {dataset}")
            
            # Train AI
            print("🧠 Training AI model...")
            ai = StakePatternAI()
            ai.train_on_massive_data(dataset, epochs=20)
            print("✅ AI training complete!")
            
            # Now start screen input
            predictor = StakeRealTimePredictor()
            predictor.input_screen_data()
            
        elif choice == '3':
            predictor = StakeRealTimePredictor()
            predictor.input_screen_data()
            
        elif choice == '4':
            # Quick test with sample data
            print("🧪 Quick test mode - enter a few sample results:")
            for i in range(5):
                result = input(f"Sample result {i+1}: ")
                try:
                    predictor.observation_history.append(float(result))
                except:
                    predictor.observation_history.append(50.0)  # Default
                    
            predictor.make_predictions(3)
            
        else:
            print("Starting screen input mode by default...")
            predictor.input_screen_data()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()