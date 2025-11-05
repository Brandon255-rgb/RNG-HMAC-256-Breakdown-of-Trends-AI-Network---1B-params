#!/usr/bin/env python3
"""
REAL STAKE DASHBOARD WITH LIVE API
=================================
Real-time web dashboard with actual Stake API integration
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import hashlib
import hmac
import time
import threading
from datetime import datetime
import os
from dotenv import load_dotenv
from real_stake_api import RealStakeAPI, AdvancedPredictorSystem
import requests

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

class RealStakePredictorBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.stake_api = RealStakeAPI(api_key)
        self.predictor = AdvancedPredictorSystem()
        
        self.is_running = False
        self.demo_mode = True
        self.demo_balance = 1000.0
        self.real_balance = 0.0
        self.strategy = 'conservative'
        self.bet_amount = 1.0
        
        self.current_seeds = {}
        self.betting_history = []
        self.total_profit = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        self.api_connected = False
        self.last_api_check = 0
        
        # Test API connection on startup
        self.test_api_connection()
        
    def test_api_connection(self):
        """Test and establish API connection"""
        try:
            print("🔍 Testing API connection...")
            self.api_connected = self.stake_api.test_connection()
            
            if self.api_connected:
                self.real_balance = self.stake_api.get_balance()
                seeds = self.stake_api.get_current_seeds()
                if seeds:
                    self.current_seeds = seeds
                    print(f"✅ API Connected! Balance: ${self.real_balance:.2f}")
                    return True
            
            print("❌ API connection failed")
            return False
            
        except Exception as e:
            print(f"❌ API connection error: {e}")
            self.api_connected = False
            return False
    
    def refresh_seeds(self):
        """Get fresh seeds from live API"""
        if not self.api_connected:
            if not self.test_api_connection():
                return False
        
        try:
            seeds = self.stake_api.get_current_seeds()
            if seeds:
                self.current_seeds = seeds
                print(f"🌱 Seeds refreshed: Nonce {seeds['nonce']}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error refreshing seeds: {e}")
            return False
    
    def generate_stake_result(self, client_seed, server_seed_hash, nonce):
        """Generate Stake dice result using HMAC-SHA512"""
        try:
            message = f"{client_seed}:{nonce}"
            signature = hmac.new(
                server_seed_hash.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            # Convert first 8 hex chars to number and scale to 0-99.99
            seed = int(signature[:8], 16)
            result = (seed / 0xFFFFFFFF) * 100
            return round(result, 4)
        except Exception as e:
            print(f"Error generating result: {e}")
            return 0.0
    
    def calculate_multiplier(self, target, is_over=True):
        """Calculate multiplier for given target"""
        if is_over:
            win_chance = (100 - target) / 100
        else:
            win_chance = target / 100
        return (0.99 / win_chance) if win_chance > 0 else 0
    
    def optimize_bet(self, predicted_result, strategy='conservative'):
        """Optimize bet for maximum multiplier with safety"""
        params = self.strategy_params[strategy]
        margin = params['margin']
        max_bet_pct = params['max_bet_pct']
        
        # Determine direction and calculate target
        if predicted_result < 50:
            # Bet UNDER with safety margin
            target = min(99.0, predicted_result + margin)
            direction = 'under'
        else:
            # Bet OVER with safety margin  
            target = max(1.0, predicted_result - margin)
            direction = 'over'
        
        # Calculate multiplier
        multiplier = self.calculate_multiplier(target, direction == 'over')
        
        # Calculate bet amount (percentage of balance, max $100)
        bet_amount = min(100.0, self.balance * max_bet_pct)
        bet_amount = max(1.0, bet_amount)  # Minimum $1
        
        potential_profit = bet_amount * (multiplier - 1)
        
        return {
            'direction': direction,
            'target': round(target, 2),
            'multiplier': round(multiplier, 2),
            'bet_amount': round(bet_amount, 2),
            'potential_profit': round(potential_profit, 2),
            'predicted_result': predicted_result,
            'should_bet': multiplier > 1.1 and bet_amount <= self.balance,
            'win_chance': round((100 - target) if direction == 'over' else target, 2),
            'strategy': strategy
        }
    
    def place_demo_bet(self, bet_data):
        """Place a demo bet and return result"""
        nonce = self.current_seeds['nonce']
        
        # Get actual result using Stake's HMAC
        actual_result = self.generate_stake_result(
            self.current_seeds['client'],
            self.current_seeds['server'], 
            nonce
        )
        
        # Check if bet won
        if bet_data['direction'] == 'under':
            won = actual_result < bet_data['target']
        else:
            won = actual_result > bet_data['target']
        
        # Update stats
        self.total_bets += 1
        if won:
            self.winning_bets += 1
            self.balance += bet_data['potential_profit']
            profit = bet_data['potential_profit']
        else:
            self.balance -= bet_data['bet_amount']
            profit = -bet_data['bet_amount']
        
        self.current_profit = self.balance - self.starting_balance
        
        # Create bet record
        bet_record = {
            'id': self.total_bets,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'nonce': nonce,
            'predicted': bet_data['predicted_result'],
            'actual': actual_result,
            'direction': bet_data['direction'],
            'target': bet_data['target'],
            'multiplier': bet_data['multiplier'],
            'bet_amount': bet_data['bet_amount'],
            'won': won,
            'profit': round(profit, 2),
            'balance': round(self.balance, 2),
            'strategy': self.current_strategy,
            'win_chance': bet_data['win_chance']
        }
        
        self.bet_history.append(bet_record)
        if len(self.bet_history) > 100:  # Keep last 100 bets
            self.bet_history.pop(0)
        
        # Update profit history for chart
        self.profit_history.append({
            'bet_num': self.total_bets,
            'profit': self.current_profit,
            'balance': self.balance
        })
        if len(self.profit_history) > 50:
            self.profit_history.pop(0)
        
        # Increment nonce for next bet
        self.current_seeds['nonce'] += 1
        
        return bet_record
    
    def get_stats(self):
        """Get current bot statistics"""
        win_rate = (self.winning_bets / self.total_bets * 100) if self.total_bets > 0 else 0
        roi = (self.current_profit / self.starting_balance * 100) if self.starting_balance > 0 else 0
        
        avg_profit = self.current_profit / self.total_bets if self.total_bets > 0 else 0
        
        return {
            'balance': round(self.balance, 2),
            'starting_balance': self.starting_balance,
            'current_profit': round(self.current_profit, 2),
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'losing_bets': self.total_bets - self.winning_bets,
            'win_rate': round(win_rate, 1),
            'roi': round(roi, 1),
            'avg_profit_per_bet': round(avg_profit, 2),
            'current_strategy': self.current_strategy,
            'is_running': self.is_running,
            'current_nonce': self.current_seeds['nonce'],
            'bet_speed': self.bet_speed,
            'api_connected': bool(self.api_key)
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.balance = 1000.0
        self.starting_balance = 1000.0
        self.total_bets = 0
        self.winning_bets = 0
        self.current_profit = 0.0
        self.bet_history = []
        self.profit_history = []
        self.current_seeds['nonce'] = 1629

# Global bot instance
bot = StakePredictorBot()

@app.route('/')
def dashboard():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current bot statistics"""
    return jsonify(bot.get_stats())

@app.route('/api/history')
def get_history():
    """Get recent betting history"""
    return jsonify(bot.bet_history[-20:])

@app.route('/api/profit_history')
def get_profit_history():
    """Get profit history for chart"""
    return jsonify(bot.profit_history)

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the betting bot"""
    if not bot.is_running:
        bot.is_running = True
        threading.Thread(target=run_bot_loop, daemon=True).start()
        socketio.emit('bot_status', {'status': 'started', 'message': 'Bot started successfully!'})
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST']) 
def stop_bot():
    """Stop the betting bot"""
    bot.is_running = False
    socketio.emit('bot_status', {'status': 'stopped', 'message': 'Bot stopped!'})
    return jsonify({'status': 'stopped'})

@app.route('/api/strategy', methods=['POST'])
def change_strategy():
    """Change betting strategy"""
    data = request.get_json()
    strategy = data.get('strategy')
    if strategy in bot.strategies:
        bot.current_strategy = strategy
        socketio.emit('strategy_changed', {'strategy': strategy})
        return jsonify({'status': 'changed', 'strategy': strategy})
    return jsonify({'status': 'invalid_strategy'})

@app.route('/api/speed', methods=['POST'])
def change_speed():
    """Change betting speed"""
    data = request.get_json()
    speed = float(data.get('speed', 2.0))
    bot.bet_speed = max(0.5, min(10.0, speed))  # Limit between 0.5-10 seconds
    return jsonify({'status': 'changed', 'speed': bot.bet_speed})

@app.route('/api/reset', methods=['POST'])
def reset_bot():
    """Reset bot statistics"""
    was_running = bot.is_running
    bot.is_running = False
    time.sleep(0.5)  # Give time for loop to stop
    
    bot.reset_stats()
    
    socketio.emit('bot_reset', {'message': 'Bot reset successfully!'})
    return jsonify({'status': 'reset'})

@app.route('/api/manual_bet', methods=['POST'])
def manual_bet():
    """Place a manual bet"""
    if bot.is_running:
        return jsonify({'status': 'error', 'message': 'Stop bot first'})
    
    data = request.get_json()
    
    # Get prediction for current nonce
    predicted_result = bot.generate_stake_result(
        bot.current_seeds['client'],
        bot.current_seeds['server'],
        bot.current_seeds['nonce']
    )
    
    # Optimize bet
    bet_data = bot.optimize_bet(predicted_result, bot.current_strategy)
    
    if bet_data['should_bet'] and bot.balance >= bet_data['bet_amount']:
        bet_record = bot.place_demo_bet(bet_data)
        socketio.emit('bet_placed', bet_record)
        socketio.emit('stats_update', bot.get_stats())
        return jsonify({'status': 'placed', 'bet': bet_record})
    else:
        return jsonify({'status': 'skipped', 'reason': 'Insufficient balance or poor odds'})

def run_bot_loop():
    """Main bot loop that places bets automatically"""
    print("🤖 Bot loop started!")
    
    while bot.is_running:
        try:
            # Get next prediction using HMAC
            predicted_result = bot.generate_stake_result(
                bot.current_seeds['client'],
                bot.current_seeds['server'],
                bot.current_seeds['nonce']
            )
            
            # Optimize bet based on current strategy
            bet_data = bot.optimize_bet(predicted_result, bot.current_strategy)
            
            # Check if we should place this bet
            if bet_data['should_bet'] and bot.balance >= bet_data['bet_amount']:
                # Place the bet
                bet_record = bot.place_demo_bet(bet_data)
                
                # Emit updates to dashboard
                socketio.emit('bet_placed', bet_record)
                socketio.emit('stats_update', bot.get_stats())
                socketio.emit('profit_update', bot.profit_history[-1] if bot.profit_history else None)
                
                # Console logging
                result_emoji = "✅" if bet_record['won'] else "❌"
                print(f"🎰 Bet #{bot.total_bets}: {bet_record['direction'].upper()} {bet_record['target']} "
                      f"| Predicted: {bet_record['predicted']:.2f} | Actual: {bet_record['actual']:.4f} "
                      f"| {result_emoji} {'WON' if bet_record['won'] else 'LOST'} "
                      f"| Profit: ${bet_record['profit']:+.2f} | Balance: ${bet_record['balance']:.2f}")
                
                # Stop if balance is too low
                if bot.balance < 10:
                    print("💸 Balance too low, stopping bot...")
                    bot.is_running = False
                    socketio.emit('bot_status', {'status': 'stopped', 'message': 'Bot stopped - insufficient balance'})
            else:
                # Skip this bet
                print(f"⏭️  Skipping bet (nonce {bot.current_seeds['nonce']}) - poor odds or insufficient balance")
                bot.current_seeds['nonce'] += 1
            
            # Wait before next bet
            time.sleep(bot.bet_speed)
            
        except Exception as e:
            print(f"❌ Bot error: {e}")
            socketio.emit('bot_error', {'error': str(e)})
            time.sleep(5)
    
    print("🛑 Bot loop stopped!")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("🔌 Client connected to dashboard")
    emit('stats_update', bot.get_stats())
    emit('history_update', bot.bet_history[-20:])
    emit('profit_history_update', bot.profit_history)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("🔌 Client disconnected from dashboard")

if __name__ == '__main__':
    # Create templates and static directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 STAKE PREDICTOR DASHBOARD STARTING...")
    print("=" * 50)
    print(f"💰 API Key Found: {'✅' if bot.api_key else '❌'}")
    if bot.api_key:
        print(f"    Key: {bot.api_key[:20]}...{bot.api_key[-10:]}")
    print(f"🌱 Client Seed: {bot.current_seeds['client']}")
    print(f"🌱 Server Seed: {bot.current_seeds['server'][:20]}...")
    print(f"🔢 Starting Nonce: {bot.current_seeds['nonce']}")
    print("🌐 Dashboard URL: http://localhost:5000")
    print("=" * 50)
    
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")