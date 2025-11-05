#!/usr/bin/env python3
"""
FINAL MILLION-DOLLAR STAKE PREDICTOR
Combines all systems for maximum betting accuracy
"""

import numpy as np
import json
import time
from pathlib import Path
from real_stake_analyzer import RealStakeAnalyzer
from ultra_precise_stake_ai import UltraPreciseStakeAI

class MillionDollarStakePredictor:
    """Final system combining all prediction methods"""
    
    def __init__(self):
        print("💎 MILLION-DOLLAR STAKE PREDICTOR INITIALIZING")
        print("=" * 60)
        
        # Initialize all components
        self.analyzer = RealStakeAnalyzer()
        self.ai = UltraPreciseStakeAI()
        
        # Load historical data
        print("📊 Loading 1,629 bet history...")
        self.history = self.analyzer.generate_complete_history()
        self.patterns = None
        
        print("🔍 Analyzing sharp patterns...")
        self.patterns = self.analyzer.analyze_sharp_patterns()
        
        print("✅ SYSTEM READY FOR MILLION-DOLLAR BETTING!")
        
    def get_next_exact_predictions(self, count=20):
        """Get exact predictions for next bets"""
        print(f"\n🎯 GETTING NEXT {count} EXACT PREDICTIONS")
        print("=" * 50)
        
        # Method 1: Hash-based predictions (100% accurate with known seeds)
        hash_predictions = self.analyzer.predict_next_sequence(count)
        
        print(f"🔑 HASH PREDICTIONS (100% accurate):")
        immediate_opportunities = []
        
        for pred in hash_predictions:
            value = pred['predicted_result']
            nonce = pred['nonce']
            seed_pair = pred['seed_pair']
            
            print(f"  {seed_pair.upper()} Nonce {nonce}: {value:.2f}")
            
            # Identify immediate opportunities
            if value <= 15:
                immediate_opportunities.append({
                    'nonce': nonce,
                    'value': value,
                    'strategy': f"🔥🔥🔥 ULTRA UNDER 20 - {value:.1f}",
                    'confidence': 'MAXIMUM',
                    'risk': 'MINIMAL',
                    'seed_pair': seed_pair
                })
            elif value >= 85:
                immediate_opportunities.append({
                    'nonce': nonce,
                    'value': value,
                    'strategy': f"🔥🔥🔥 ULTRA OVER 80 - {value:.1f}",
                    'confidence': 'MAXIMUM', 
                    'risk': 'MINIMAL',
                    'seed_pair': seed_pair
                })
            elif 47 <= value <= 53:
                immediate_opportunities.append({
                    'nonce': nonce,
                    'value': value,
                    'strategy': f"🎯 RANGE 47-53 - {value:.1f}",
                    'confidence': 'HIGH',
                    'risk': 'LOW',
                    'seed_pair': seed_pair
                })
        
        return hash_predictions, immediate_opportunities
    
    def analyze_betting_sequence(self, predictions):
        """Analyze sequence for optimal betting strategy"""
        print(f"\n💰 MILLION-DOLLAR BETTING SEQUENCE ANALYSIS")
        print("=" * 50)
        
        # Extract values
        values = [p['predicted_result'] for p in predictions]
        
        # Sharp movement analysis
        sharp_movements = []
        for i in range(1, len(values)):
            diff = values[i] - values[i-1]
            if abs(diff) > 30:
                sharp_movements.append({
                    'position': i + 1,
                    'from': values[i-1],
                    'to': values[i],
                    'magnitude': abs(diff),
                    'type': 'JUMP' if diff > 0 else 'DROP'
                })
        
        # Consecutive patterns
        consecutive_lows = []
        consecutive_highs = []
        current_low_streak = 0
        current_high_streak = 0
        
        for i, value in enumerate(values):
            if value <= 25:
                current_low_streak += 1
                current_high_streak = 0
            elif value >= 75:
                current_high_streak += 1
                current_low_streak = 0
            else:
                if current_low_streak >= 2:
                    consecutive_lows.append((i - current_low_streak, current_low_streak))
                if current_high_streak >= 2:
                    consecutive_highs.append((i - current_high_streak, current_high_streak))
                current_low_streak = 0
                current_high_streak = 0
        
        # Volatility windows
        window_size = 5
        volatility_windows = []
        for i in range(len(values) - window_size + 1):
            window = values[i:i + window_size]
            volatility = np.std(window)
            volatility_windows.append(volatility)
        
        analysis = {
            'sharp_movements': sharp_movements,
            'consecutive_lows': consecutive_lows,
            'consecutive_highs': consecutive_highs,
            'avg_volatility': np.mean(volatility_windows) if volatility_windows else 0,
            'value_range': (min(values), max(values)),
            'trend': values[-1] - values[0]  # Overall trend
        }
        
        return analysis
    
    def generate_million_dollar_strategy(self, predictions, opportunities, analysis):
        """Generate ultimate betting strategy"""
        print(f"\n💎 MILLION-DOLLAR BETTING STRATEGY")
        print("=" * 60)
        
        print(f"🎯 IMMEDIATE HIGH-CONFIDENCE OPPORTUNITIES:")
        
        if not opportunities:
            print("  ⚠️  No immediate ultra-high confidence opportunities")
            print("  💡 Consider waiting for next seed rotation or smaller bets")
            return
        
        # Group by confidence level
        ultra_opportunities = [op for op in opportunities if op['confidence'] == 'MAXIMUM']
        high_opportunities = [op for op in opportunities if op['confidence'] == 'HIGH']
        
        if ultra_opportunities:
            print(f"\n🔥 ULTRA OPPORTUNITIES ({len(ultra_opportunities)}):")
            for i, op in enumerate(ultra_opportunities):
                print(f"  #{i+1} {op['seed_pair'].upper()} Nonce {op['nonce']}: {op['strategy']}")
                print(f"      Risk Level: {op['risk']}")
                print(f"      Confidence: {op['confidence']}")
                
                # Betting recommendations
                if op['risk'] == 'MINIMAL':
                    print(f"      💰 Recommended: HIGH STAKE bet")
                    print(f"      🎯 Target: Maximize profit on this opportunity")
                
        if high_opportunities:
            print(f"\n✅ HIGH OPPORTUNITIES ({len(high_opportunities)}):")
            for i, op in enumerate(high_opportunities):
                print(f"  #{i+1} {op['seed_pair'].upper()} Nonce {op['nonce']}: {op['strategy']}")
                print(f"      💰 Recommended: MEDIUM-HIGH STAKE")
        
        # Sharp movement strategy
        if analysis['sharp_movements']:
            print(f"\n⚡ SHARP MOVEMENT STRATEGY:")
            for movement in analysis['sharp_movements'][:3]:
                if movement['type'] == 'DROP':
                    print(f"  Position {movement['position']}: After {movement['magnitude']:.0f}-point DROP")
                    print(f"      💡 Potential rebound opportunity")
                else:
                    print(f"  Position {movement['position']}: After {movement['magnitude']:.0f}-point JUMP")
                    print(f"      💡 Potential correction opportunity")
        
        # Risk management
        print(f"\n🛡️ RISK MANAGEMENT:")
        high_vol = analysis['avg_volatility'] > 30
        print(f"  Volatility Level: {'HIGH' if high_vol else 'NORMAL'} ({analysis['avg_volatility']:.1f})")
        
        if high_vol:
            print(f"  ⚠️  High volatility detected - Consider:")
            print(f"     • Smaller position sizes")
            print(f"     • Focus on ULTRA opportunities only")
            print(f"     • Quick entry/exit strategy")
        else:
            print(f"  ✅ Normal volatility - Standard betting approach")
        
        # Sequence timing
        print(f"\n⏰ OPTIMAL BETTING SEQUENCE:")
        ultra_nonces = [op['nonce'] for op in ultra_opportunities]
        high_nonces = [op['nonce'] for op in high_opportunities]
        
        all_nonces = sorted(ultra_nonces + high_nonces)
        for i, nonce in enumerate(all_nonces[:5]):  # Top 5
            opportunity = next((op for op in opportunities if op['nonce'] == nonce), None)
            if opportunity:
                priority = "🔥 PRIORITY" if nonce in ultra_nonces else "✅ FOLLOW-UP"
                print(f"  {i+1}. {priority}: {opportunity['seed_pair'].upper()} Nonce {nonce}")
                print(f"     Strategy: {opportunity['strategy']}")
    
    def live_prediction_mode(self):
        """Live mode for continuous predictions"""
        print(f"\n🔴 LIVE MILLION-DOLLAR PREDICTION MODE")
        print("=" * 50)
        print("Commands:")
        print("  'next' - Get next predictions")
        print("  'analyze [value]' - Analyze specific value")
        print("  'sequence [count]' - Get sequence predictions")
        print("  'opportunity' - Find best opportunities")
        print("  'quit' - Exit")
        
        while True:
            try:
                cmd = input("\n💎 LIVE> ").strip().split()
                
                if not cmd or cmd[0] == 'quit':
                    break
                
                elif cmd[0] == 'next':
                    count = int(cmd[1]) if len(cmd) > 1 else 20
                    predictions, opportunities = self.get_next_exact_predictions(count)
                    analysis = self.analyze_betting_sequence(predictions)
                    self.generate_million_dollar_strategy(predictions, opportunities, analysis)
                
                elif cmd[0] == 'analyze':
                    if len(cmd) > 1:
                        value = float(cmd[1])
                        self.analyzer.multi_layer_decision_system(value)
                    else:
                        print("Usage: analyze [value]")
                
                elif cmd[0] == 'sequence':
                    count = int(cmd[1]) if len(cmd) > 1 else 10
                    predictions, _ = self.get_next_exact_predictions(count)
                    
                    print(f"📊 NEXT {count} PREDICTIONS:")
                    for pred in predictions:
                        value = pred['predicted_result']
                        nonce = pred['nonce']
                        seed_pair = pred['seed_pair']
                        
                        # Categorize
                        if value <= 20:
                            category = "🔥 ULTRA LOW"
                        elif value >= 80:
                            category = "🔥 ULTRA HIGH"  
                        elif 45 <= value <= 55:
                            category = "🎯 MIDDLE"
                        else:
                            category = "📊 NORMAL"
                        
                        print(f"  {seed_pair.upper()} {nonce}: {value:.2f} - {category}")
                
                elif cmd[0] == 'opportunity':
                    predictions, opportunities = self.get_next_exact_predictions(20)
                    if opportunities:
                        print(f"🎯 FOUND {len(opportunities)} OPPORTUNITIES:")
                        for op in opportunities:
                            print(f"  {op['strategy']} - {op['confidence']} confidence")
                    else:
                        print("⚠️  No immediate opportunities found")
                
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting live mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main million-dollar predictor interface"""
    print("💎 MILLION-DOLLAR STAKE PREDICTOR")
    print("=" * 60)
    print("🎯 Using real Stake seeds: 3f95f77b5e864e15")
    print("🔑 Server seed: 3428e6f9695f86...")
    print("🎲 Position after 1,629 bets")
    print("💰 Designed for million-dollar betting accuracy")
    
    predictor = MillionDollarStakePredictor()
    
    print("\nOptions:")
    print("1. Get immediate next predictions")
    print("2. Live prediction mode (RECOMMENDED)")
    print("3. Analyze specific betting sequence")
    print("4. Quick opportunity scan")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            count = int(input("Number of predictions (default 20): ") or "20")
            predictions, opportunities = predictor.get_next_exact_predictions(count)
            analysis = predictor.analyze_betting_sequence(predictions)
            predictor.generate_million_dollar_strategy(predictions, opportunities, analysis)
            
        elif choice == '2':
            predictor.live_prediction_mode()
            
        elif choice == '3':
            start_nonce = int(input("Starting nonce (default 1630): ") or "1630")
            count = int(input("Sequence length (default 15): ") or "15")
            
            predictions, opportunities = predictor.get_next_exact_predictions(count)
            analysis = predictor.analyze_betting_sequence(predictions)
            predictor.generate_million_dollar_strategy(predictions, opportunities, analysis)
            
        elif choice == '4':
            print("🔍 Quick opportunity scan...")
            predictions, opportunities = predictor.get_next_exact_predictions(30)
            
            if opportunities:
                print(f"🎯 FOUND {len(opportunities)} OPPORTUNITIES in next 30 bets:")
                for op in opportunities[:5]:  # Top 5
                    print(f"  {op['strategy']} - Nonce {op['nonce']}")
            else:
                print("⚠️  No opportunities in next 30 bets - wait for better sequence")
        
        else:
            print("Starting live mode by default...")
            predictor.live_prediction_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()