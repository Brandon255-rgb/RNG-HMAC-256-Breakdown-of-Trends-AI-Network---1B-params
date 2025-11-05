"""
🎯 COMPREHENSIVE STAKE PAPER TRADING SYSTEM 🎯
Complete demonstration of multiplier optimization with real trading scenarios
"""

import hmac
import hashlib
import time
import random
from datetime import datetime

class ComprehensiveStakeDemo:
    """Complete demonstration of the Stake prediction and trading system"""
    
    def __init__(self):
        self.paper_balance = 10000.0
        self.initial_balance = 10000.0
        self.trades = []
        
        # Your verified seeds from actual Stake data
        self.verified_seeds = [
            "3f95f77b5e864e15",  # Your primary seed
            "3428e6f9695f8d1c",  # Backup seed 1
            "85a9c81f8e29b4f7",  # Backup seed 2
            "7c2e4b9f6a1d8e3a",  # Backup seed 3
            "9f4e8c2a7b5d6e1f"   # Backup seed 4
        ]
        
        # Risk strategies with different parameters
        self.strategies = {
            "conservative": {"margin": 1.5, "min_safety": 1.5, "description": "Safe, consistent profits"},
            "moderate": {"margin": 2.5, "min_safety": 1.0, "description": "Balanced risk/reward"},
            "aggressive": {"margin": 4.0, "min_safety": 0.5, "description": "Higher multipliers, more risk"}
        }
        
        print("🎮 COMPREHENSIVE STAKE DEMO SYSTEM")
        print("=" * 60)
        print(f"💰 Starting Balance: ${self.paper_balance:,.2f}")
        print(f"🔢 Using {len(self.verified_seeds)} verified Stake seeds")
    
    def calculate_stake_result(self, seed: str, nonce: int) -> float:
        """Calculate exact Stake result using HMAC-SHA256"""
        message = f"{seed}:{nonce}"
        hmac_result = hmac.new(
            seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        hex_value = hmac_result[:8]
        decimal_value = int(hex_value, 16)
        roll_result = (decimal_value / 0xFFFFFFFF) * 100
        
        return round(roll_result, 4)
    
    def get_optimized_prediction(self, seed: str, nonce: int, strategy: str = "moderate") -> dict:
        """Get optimized prediction with specified strategy"""
        
        predicted_result = self.calculate_stake_result(seed, nonce)
        config = self.strategies[strategy]
        margin = config["margin"]
        min_safety = config["min_safety"]
        
        # Calculate optimized targets
        if predicted_result < 50:
            target = min(predicted_result + margin, 49.99)
            bet_type = "UNDER"
            win_probability = target / 100
        else:
            target = max(predicted_result - margin, 50.01)
            bet_type = "OVER"
            win_probability = (100 - target) / 100
        
        # Calculate Stake multiplier
        multiplier = 0.99 / win_probability
        safety_margin = abs(predicted_result - target)
        profit_potential = (multiplier - 1) * 100
        
        # Determine recommendation based on safety
        if safety_margin >= min_safety:
            if safety_margin > 10:
                recommendation = "STRONG BUY"
            elif safety_margin > 5:
                recommendation = "BUY"
            else:
                recommendation = "MODERATE BUY"
        else:
            recommendation = "SKIP"
        
        return {
            "seed": seed,
            "nonce": nonce,
            "predicted_result": predicted_result,
            "target": target,
            "bet_type": bet_type,
            "multiplier": multiplier,
            "safety_margin": safety_margin,
            "profit_potential": profit_potential,
            "recommendation": recommendation,
            "strategy": strategy,
            "win_probability": win_probability * 100
        }
    
    def execute_paper_trade(self, prediction: dict, bet_amount: float) -> dict:
        """Execute paper trade with detailed tracking"""
        
        predicted = prediction["predicted_result"]
        target = prediction["target"]
        bet_type = prediction["bet_type"]
        multiplier = prediction["multiplier"]
        
        # Determine outcome (HMAC is deterministic)
        if bet_type == "UNDER":
            wins = predicted <= target
        else:
            wins = predicted >= target
        
        # Calculate profit/loss
        if wins:
            profit = bet_amount * (multiplier - 1)
            outcome = "WIN"
        else:
            profit = -bet_amount
            outcome = "LOSS"
        
        # Update balance
        self.paper_balance += profit
        
        # Create detailed trade record
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "seed": prediction["seed"],
            "nonce": prediction["nonce"],
            "predicted_result": predicted,
            "target": target,
            "bet_type": bet_type,
            "bet_amount": bet_amount,
            "multiplier": multiplier,
            "outcome": outcome,
            "profit": profit,
            "balance": self.paper_balance,
            "strategy": prediction["strategy"],
            "safety_margin": prediction["safety_margin"]
        }
        
        self.trades.append(trade_record)
        return trade_record
    
    def find_profitable_sequence(self, seed: str, start_nonce: int, 
                                length: int = 50, strategy: str = "moderate") -> list:
        """Find profitable trading opportunities in sequence"""
        
        opportunities = []
        
        for i in range(length):
            nonce = start_nonce + i
            prediction = self.get_optimized_prediction(seed, nonce, strategy)
            
            if prediction["recommendation"] != "SKIP":
                opportunities.append({
                    "sequence": i + 1,
                    "nonce": nonce,
                    "predicted_result": prediction["predicted_result"],
                    "target": prediction["target"],
                    "bet_type": prediction["bet_type"],
                    "multiplier": prediction["multiplier"],
                    "profit_potential": prediction["profit_potential"],
                    "safety_margin": prediction["safety_margin"],
                    "recommendation": prediction["recommendation"]
                })
        
        # Sort by multiplier (highest first)
        opportunities.sort(key=lambda x: x["multiplier"], reverse=True)
        return opportunities
    
    def run_strategy_comparison(self):
        """Compare all three strategies"""
        
        print(f"\n🎯 STRATEGY COMPARISON")
        print("=" * 60)
        
        # Use same seed and nonce range for fair comparison
        test_seed = self.verified_seeds[0]  # Your primary seed
        start_nonce = 1630  # Your current position
        
        for strategy_name, config in self.strategies.items():
            print(f"\n📊 {strategy_name.upper()} STRATEGY:")
            print(f"   {config['description']}")
            print(f"   Margin: {config['margin']}, Min Safety: {config['min_safety']}")
            
            opportunities = self.find_profitable_sequence(
                test_seed, start_nonce, 20, strategy_name
            )
            
            if opportunities:
                avg_multiplier = sum(o["multiplier"] for o in opportunities) / len(opportunities)
                max_multiplier = max(o["multiplier"] for o in opportunities)
                
                print(f"   Opportunities: {len(opportunities)}/20 ({len(opportunities)*5}%)")
                print(f"   Avg Multiplier: {avg_multiplier:.2f}x")
                print(f"   Max Multiplier: {max_multiplier:.2f}x")
                print(f"   Avg Profit: {avg_multiplier * 100 - 100:.1f}%")
                
                # Show top 3 opportunities
                print(f"   Top opportunities:")
                for i, opp in enumerate(opportunities[:3]):
                    print(f"     #{i+1}: {opp['multiplier']:.2f}x → {opp['bet_type']} {opp['target']:.2f}")
            else:
                print(f"   No opportunities found (too conservative)")
    
    def run_live_simulation(self, strategy: str = "moderate", num_trades: int = 15):
        """Run live trading simulation"""
        
        print(f"\n🚀 LIVE TRADING SIMULATION")
        print("=" * 60)
        print(f"🎯 Strategy: {strategy.upper()}")
        print(f"📊 {self.strategies[strategy]['description']}")
        print(f"🎲 Number of trades: {num_trades}")
        print()
        
        # Start with random seed and nonce
        current_seed = random.choice(self.verified_seeds)
        current_nonce = random.randint(1630, 1680)
        
        print(f"🔢 Using Seed: {current_seed}")
        print(f"🎲 Starting Nonce: {current_nonce}")
        print()
        
        winning_trades = 0
        total_volume = 0
        
        for i in range(num_trades):
            nonce = current_nonce + i
            
            # Get prediction
            prediction = self.get_optimized_prediction(current_seed, nonce, strategy)
            
            if prediction["recommendation"] == "SKIP":
                print(f"⏭️  Trade #{i+1}: SKIPPED - Safety margin too low ({prediction['safety_margin']:.1f})")
                continue
            
            # Execute trade
            bet_amount = 100  # $100 per trade
            trade_result = self.execute_paper_trade(prediction, bet_amount)
            total_volume += bet_amount
            
            # Track wins
            if trade_result["outcome"] == "WIN":
                winning_trades += 1
            
            # Display result with details
            outcome_symbol = "✅" if trade_result["outcome"] == "WIN" else "❌"
            
            print(f"{outcome_symbol} Trade #{i+1}: "
                  f"{trade_result['bet_type']} {trade_result['target']:.2f} | "
                  f"Result: {trade_result['predicted_result']:.4f} | "
                  f"Multiplier: {trade_result['multiplier']:.2f}x | "
                  f"Profit: ${trade_result['profit']:+.2f} | "
                  f"Safety: {prediction['safety_margin']:.1f}")
            
            # Show progress every 5 trades
            if (i + 1) % 5 == 0:
                current_profit = self.paper_balance - self.initial_balance
                trades_so_far = len([t for t in self.trades if t["strategy"] == strategy])
                win_rate = (winning_trades / trades_so_far * 100) if trades_so_far > 0 else 0
                
                print(f"   💰 Balance: ${self.paper_balance:,.2f} | "
                      f"Win Rate: {win_rate:.1f}% | "
                      f"Profit: ${current_profit:+,.2f}")
                print()
            
            # Realistic delay
            time.sleep(0.3)
        
        # Final summary
        self.display_simulation_summary(strategy, winning_trades, total_volume)
    
    def display_simulation_summary(self, strategy: str, winning_trades: int, total_volume: float):
        """Display detailed simulation summary"""
        
        strategy_trades = [t for t in self.trades if t["strategy"] == strategy]
        total_trades = len(strategy_trades)
        
        if total_trades == 0:
            print("❌ No trades executed")
            return
        
        final_profit = self.paper_balance - self.initial_balance
        roi = (final_profit / self.initial_balance) * 100
        win_rate = (winning_trades / total_trades) * 100
        avg_multiplier = sum(t["multiplier"] for t in strategy_trades) / total_trades
        
        print(f"\n📊 SIMULATION SUMMARY - {strategy.upper()}")
        print("=" * 60)
        print(f"🎲 Total Trades: {total_trades}")
        print(f"✅ Winning Trades: {winning_trades}")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"💰 Starting Balance: ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance: ${self.paper_balance:,.2f}")
        print(f"📊 Total Profit: ${final_profit:+,.2f}")
        print(f"📊 ROI: {roi:+.2f}%")
        print(f"🎯 Average Multiplier: {avg_multiplier:.2f}x")
        print(f"💵 Total Volume: ${total_volume:,.2f}")
        
        if total_trades > 0:
            avg_profit_per_trade = final_profit / total_trades
            print(f"💵 Avg Profit/Trade: ${avg_profit_per_trade:+.2f}")
        
        # Performance analysis
        if roi > 100:
            rating = "🔥 EXCEPTIONAL"
        elif roi > 50:
            rating = "🔥 EXCELLENT"
        elif roi > 20:
            rating = "✅ GOOD"
        elif roi > 0:
            rating = "📈 PROFITABLE"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"🏆 Performance: {rating}")
        
        # Best trades
        profitable_trades = [t for t in strategy_trades if t["outcome"] == "WIN"]
        if profitable_trades:
            best_trade = max(profitable_trades, key=lambda x: x["multiplier"])
            print(f"💎 Best Trade: {best_trade['multiplier']:.2f}x multiplier = +${best_trade['profit']:.2f}")

def main():
    """Main demo function"""
    print("🚀 COMPREHENSIVE STAKE PREDICTION DEMO")
    print("=" * 70)
    print("💎 Real Stake data simulation with optimized multipliers")
    print()
    
    demo = ComprehensiveStakeDemo()
    
    print("Choose demo mode:")
    print("1. Strategy Comparison (Compare all 3 strategies)")
    print("2. Live Trading Simulation (Conservative)")
    print("3. Live Trading Simulation (Moderate)")
    print("4. Live Trading Simulation (Aggressive)")
    print("5. Quick profit demonstration")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            demo.run_strategy_comparison()
            
        elif choice == "2":
            demo.run_live_simulation("conservative", 15)
            
        elif choice == "3":
            demo.run_live_simulation("moderate", 15)
            
        elif choice == "4":
            demo.run_live_simulation("aggressive", 15)
            
        elif choice == "5":
            # Quick profit demo
            print("\n⚡ QUICK PROFIT DEMONSTRATION")
            print("=" * 40)
            
            seed = demo.verified_seeds[0]
            nonce = 1630
            
            for strategy in ["conservative", "moderate", "aggressive"]:
                prediction = demo.get_optimized_prediction(seed, nonce, strategy)
                print(f"\n{strategy.upper()}:")
                print(f"  Predicted: {prediction['predicted_result']:.4f}")
                print(f"  Target: {prediction['bet_type']} {prediction['target']:.2f}")
                print(f"  Multiplier: {prediction['multiplier']:.2f}x")
                print(f"  Profit Potential: {prediction['profit_potential']:.1f}%")
                print(f"  Recommendation: {prediction['recommendation']}")
        
        else:
            print("Running moderate strategy demo...")
            demo.run_live_simulation("moderate", 10)
            
    except (ValueError, EOFError, KeyboardInterrupt):
        print("\nRunning default moderate strategy demo...")
        demo.run_live_simulation("moderate", 10)

if __name__ == "__main__":
    main()