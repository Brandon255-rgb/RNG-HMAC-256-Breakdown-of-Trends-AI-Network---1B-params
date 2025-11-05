"""
🎯 ADVANCED MULTIPLIER MAXIMIZER 🎯
Optimizes bet targets based on prediction confidence for maximum profit

This system adjusts betting targets to get higher multipliers while maintaining safety.
If we predict 21 with 95% confidence, we bet under 24 for better payout!
"""

import hmac
import hashlib
import json
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import statistics

class MultiplierMaximizer:
    """
    Advanced system that maximizes multipliers based on prediction confidence
    The key to turning predictions into maximum profit!
    """
    
    def __init__(self):
        print("🚀 MULTIPLIER MAXIMIZER INITIALIZING...")
        print("=" * 60)
        
        # Stake payout calculation constants
        self.house_edge = 0.01  # 1% house edge
        self.base_multiplier = 0.99  # 99% base
        
        # Risk tolerance levels
        self.risk_levels = {
            "conservative": {"margin": 2.0, "min_confidence": 0.95},
            "moderate": {"margin": 3.0, "min_confidence": 0.90},
            "aggressive": {"margin": 5.0, "min_confidence": 0.85}
        }
        
        # Paper trading tracking
        self.paper_balance = 10000.0  # Start with $10k virtual money
        self.paper_trades = []
        self.strategy_stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "total_profit": 0.0,
            "best_multiplier": 0.0,
            "average_multiplier": 0.0
        }
        
        print("✅ System ready for multiplier optimization!")
    
    def calculate_optimal_target(self, predicted_value: float, confidence: float, 
                               risk_level: str = "moderate") -> Dict[str, Any]:
        """
        Calculate optimal betting target to maximize multiplier
        This is the core function that maximizes your profits!
        """
        
        if risk_level not in self.risk_levels:
            risk_level = "moderate"
        
        config = self.risk_levels[risk_level]
        margin = config["margin"]
        min_conf = config["min_confidence"]
        
        # Only proceed if confidence is high enough
        if confidence < min_conf:
            return {
                "recommendation": "SKIP",
                "reason": f"Confidence {confidence:.1%} below threshold {min_conf:.1%}",
                "predicted_value": predicted_value,
                "confidence": confidence
            }
        
        # Calculate optimal targets
        if predicted_value < 50:
            # Betting UNDER - can add margin above predicted value
            optimal_target = min(predicted_value + margin, 49.99)
            bet_type = "UNDER"
            win_probability = optimal_target / 100
            
        else:
            # Betting OVER - can subtract margin below predicted value  
            optimal_target = max(predicted_value - margin, 50.01)
            bet_type = "OVER"
            win_probability = (100 - optimal_target) / 100
        
        # Calculate multiplier (Stake's formula)
        multiplier = self.base_multiplier / win_probability
        
        # Calculate expected value
        expected_value = (multiplier * confidence) - 1
        
        # Safety check - ensure we maintain edge
        safety_margin = abs(predicted_value - optimal_target)
        
        return {
            "recommendation": f"BET {bet_type} {optimal_target:.2f}",
            "predicted_value": predicted_value,
            "optimal_target": optimal_target,
            "bet_type": bet_type,
            "multiplier": multiplier,
            "win_probability": win_probability,
            "confidence": confidence,
            "expected_value": expected_value,
            "safety_margin": safety_margin,
            "risk_level": risk_level,
            "profit_potential": (multiplier - 1) * 100  # Percentage profit
        }
    
    def calculate_stake_result(self, seed: str, nonce: int) -> float:
        """Calculate exact Stake result"""
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
    
    def analyze_multiplier_opportunities(self, seed: str, start_nonce: int, 
                                       count: int = 50, risk_level: str = "moderate") -> List[Dict]:
        """
        Analyze upcoming opportunities for maximum multiplier gains
        """
        opportunities = []
        
        for i in range(count):
            nonce = start_nonce + i
            predicted_result = self.calculate_stake_result(seed, nonce)
            
            # High confidence since HMAC is deterministic
            confidence = 0.999  # 99.9% confidence
            
            # Get optimal target
            analysis = self.calculate_optimal_target(predicted_result, confidence, risk_level)
            
            if analysis["recommendation"] != "SKIP":
                opportunities.append({
                    "nonce": nonce,
                    "sequence": i + 1,
                    "predicted_result": predicted_result,
                    "optimal_target": analysis["optimal_target"],
                    "multiplier": analysis["multiplier"],
                    "profit_potential": analysis["profit_potential"],
                    "expected_value": analysis["expected_value"],
                    "bet_type": analysis["bet_type"],
                    "safety_margin": analysis["safety_margin"]
                })
        
        # Sort by profit potential
        opportunities.sort(key=lambda x: x["profit_potential"], reverse=True)
        
        return opportunities
    
    def simulate_paper_trade(self, prediction: Dict, bet_amount: float) -> Dict[str, Any]:
        """
        Simulate a trade with paper money for testing
        """
        predicted_result = prediction["predicted_result"]
        optimal_target = prediction["optimal_target"]
        bet_type = prediction["bet_type"]
        multiplier = prediction["multiplier"]
        
        # Determine if trade would win
        if bet_type == "UNDER":
            wins = predicted_result <= optimal_target
        else:  # OVER
            wins = predicted_result >= optimal_target
        
        # Calculate profit/loss
        if wins:
            profit = bet_amount * (multiplier - 1)
            self.paper_balance += profit
            outcome = "WIN"
        else:
            profit = -bet_amount
            self.paper_balance += profit
            outcome = "LOSS"
        
        # Record trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "nonce": prediction.get("nonce", 0),
            "predicted_result": predicted_result,
            "target": optimal_target,
            "bet_type": bet_type,
            "bet_amount": bet_amount,
            "multiplier": multiplier,
            "outcome": outcome,
            "profit": profit,
            "balance": self.paper_balance
        }
        
        self.paper_trades.append(trade_record)
        
        # Update stats
        self.strategy_stats["total_trades"] += 1
        if wins:
            self.strategy_stats["winning_trades"] += 1
        self.strategy_stats["total_profit"] += profit
        self.strategy_stats["best_multiplier"] = max(self.strategy_stats["best_multiplier"], multiplier)
        
        # Calculate average multiplier
        multipliers = [t["multiplier"] for t in self.paper_trades]
        self.strategy_stats["average_multiplier"] = statistics.mean(multipliers)
        
        return trade_record
    
    def run_paper_trading_session(self, seed: str, start_nonce: int, 
                                 bet_amount: float = 100, num_trades: int = 50,
                                 risk_level: str = "moderate") -> Dict[str, Any]:
        """
        Run complete paper trading session to test strategy
        """
        print(f"\n🎮 PAPER TRADING SESSION STARTING")
        print("=" * 50)
        print(f"💰 Starting Balance: ${self.paper_balance:.2f}")
        print(f"🎯 Risk Level: {risk_level.upper()}")
        print(f"💵 Bet Amount: ${bet_amount:.2f}")
        print(f"🎲 Number of Trades: {num_trades}")
        print(f"🔢 Starting Nonce: {start_nonce}")
        
        # Get opportunities
        opportunities = self.analyze_multiplier_opportunities(
            seed, start_nonce, num_trades, risk_level
        )
        
        # Execute trades
        successful_trades = []
        
        for i, opp in enumerate(opportunities[:num_trades]):
            trade_result = self.simulate_paper_trade(opp, bet_amount)
            
            if trade_result["outcome"] == "WIN":
                successful_trades.append(trade_result)
                print(f"✅ Trade #{i+1}: {trade_result['bet_type']} {trade_result['target']:.2f} "
                      f"→ {trade_result['multiplier']:.2f}x = +${trade_result['profit']:.2f}")
            else:
                print(f"❌ Trade #{i+1}: {trade_result['bet_type']} {trade_result['target']:.2f} "
                      f"→ LOSS = ${trade_result['profit']:.2f}")
        
        # Calculate session results
        session_results = {
            "starting_balance": 10000.0,
            "ending_balance": self.paper_balance,
            "total_profit": self.paper_balance - 10000.0,
            "roi_percentage": ((self.paper_balance - 10000.0) / 10000.0) * 100,
            "total_trades": len(opportunities),
            "winning_trades": len(successful_trades),
            "win_rate": len(successful_trades) / len(opportunities) * 100 if opportunities else 0,
            "average_multiplier": self.strategy_stats["average_multiplier"],
            "best_multiplier": self.strategy_stats["best_multiplier"],
            "total_volume": len(opportunities) * bet_amount
        }
        
        return session_results
    
    def display_session_results(self, results: Dict[str, Any]):
        """Display paper trading results"""
        print(f"\n📊 PAPER TRADING RESULTS")
        print("=" * 50)
        print(f"💰 Starting Balance: ${results['starting_balance']:,.2f}")
        print(f"💰 Ending Balance: ${results['ending_balance']:,.2f}")
        print(f"📈 Total Profit: ${results['total_profit']:,.2f}")
        print(f"📊 ROI: {results['roi_percentage']:+.2f}%")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"🎲 Total Trades: {results['total_trades']}")
        print(f"✅ Winning Trades: {results['winning_trades']}")
        print(f"🔥 Best Multiplier: {results['best_multiplier']:.2f}x")
        print(f"📊 Avg Multiplier: {results['average_multiplier']:.2f}x")
        print(f"💵 Total Volume: ${results['total_volume']:,.2f}")
        
        # Performance rating
        if results['roi_percentage'] > 50:
            rating = "🔥 EXCELLENT"
        elif results['roi_percentage'] > 20:
            rating = "✅ GOOD"
        elif results['roi_percentage'] > 0:
            rating = "📈 PROFITABLE"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"🏆 Performance Rating: {rating}")
    
    def optimize_risk_level(self, seed: str, start_nonce: int, bet_amount: float = 100) -> Dict[str, Any]:
        """
        Test all risk levels to find optimal strategy
        """
        print(f"\n🎯 OPTIMIZING RISK LEVEL")
        print("=" * 50)
        
        results = {}
        
        for risk_level in ["conservative", "moderate", "aggressive"]:
            # Reset balance for each test
            self.paper_balance = 10000.0
            self.paper_trades = []
            self.strategy_stats = {
                "total_trades": 0,
                "winning_trades": 0,
                "total_profit": 0.0,
                "best_multiplier": 0.0,
                "average_multiplier": 0.0
            }
            
            print(f"\n🧪 Testing {risk_level.upper()} strategy...")
            
            session_results = self.run_paper_trading_session(
                seed, start_nonce, bet_amount, 30, risk_level
            )
            
            results[risk_level] = session_results
            
            print(f"   ROI: {session_results['roi_percentage']:+.2f}% | "
                  f"Win Rate: {session_results['win_rate']:.1f}% | "
                  f"Avg Multiplier: {session_results['average_multiplier']:.2f}x")
        
        # Find best strategy
        best_strategy = max(results.keys(), key=lambda x: results[x]['roi_percentage'])
        
        print(f"\n🏆 OPTIMAL STRATEGY: {best_strategy.upper()}")
        print(f"   Best ROI: {results[best_strategy]['roi_percentage']:+.2f}%")
        print(f"   Win Rate: {results[best_strategy]['win_rate']:.1f}%")
        print(f"   Avg Multiplier: {results[best_strategy]['average_multiplier']:.2f}x")
        
        return {
            "best_strategy": best_strategy,
            "all_results": results,
            "recommendation": f"Use {best_strategy} risk level for maximum profit"
        }

def interactive_multiplier_demo():
    """Interactive demo of the multiplier maximizer"""
    print("🚀 MULTIPLIER MAXIMIZER DEMO")
    print("=" * 60)
    
    maximizer = MultiplierMaximizer()
    
    # Example prediction
    seed = "3f95f77b5e864e15"  # Your verified seed
    nonce = 1630
    
    print("🔮 Example: If we predict next roll will be 21.50...")
    print()
    
    predicted_value = 21.50
    confidence = 0.999  # 99.9% confidence with HMAC
    
    for risk_level in ["conservative", "moderate", "aggressive"]:
        analysis = maximizer.calculate_optimal_target(predicted_value, confidence, risk_level)
        
        print(f"📊 {risk_level.upper()} Strategy:")
        print(f"   Target: {analysis['optimal_target']:.2f}")
        print(f"   Multiplier: {analysis['multiplier']:.2f}x")
        print(f"   Profit Potential: {analysis['profit_potential']:.1f}%")
        print(f"   Safety Margin: {analysis['safety_margin']:.1f}")
        print()
    
    print("💡 Instead of betting under 21.50, we can bet under 24.50 for higher multiplier!")
    print("   This increases payout while maintaining safety margin.")

def run_full_optimization():
    """Run complete optimization analysis"""
    maximizer = MultiplierMaximizer()
    
    # Your verified seed and starting position
    seed = "3f95f77b5e864e15"
    start_nonce = 1630
    
    # Run optimization
    optimization_results = maximizer.optimize_risk_level(seed, start_nonce, 100)
    
    print(f"\n🎯 OPTIMIZATION COMPLETE!")
    print(f"Best Strategy: {optimization_results['best_strategy'].upper()}")
    print(f"Recommendation: {optimization_results['recommendation']}")
    
    return optimization_results

if __name__ == "__main__":
    print("🎯 ADVANCED MULTIPLIER MAXIMIZER")
    print("=" * 60)
    print("💎 Maximize profits with optimized bet targets!")
    print()
    print("Choose mode:")
    print("1. Interactive Demo")
    print("2. Full Paper Trading Optimization")
    print("3. Quick Example")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        interactive_multiplier_demo()
    elif choice == "2":
        run_full_optimization()
    elif choice == "3":
        maximizer = MultiplierMaximizer()
        result = maximizer.calculate_optimal_target(21.50, 0.999, "moderate")
        print(f"\n⚡ QUICK EXAMPLE:")
        print(f"Predicted: 21.50 → Bet {result['recommendation']}")
        print(f"Multiplier: {result['multiplier']:.2f}x")
        print(f"Profit: {result['profit_potential']:.1f}%")
    else:
        print("👋 Goodbye!")