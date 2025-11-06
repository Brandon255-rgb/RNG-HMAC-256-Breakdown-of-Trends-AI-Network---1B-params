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
    
    def get_next_prediction(self, history_length=10):
        """Get prediction using all our advanced methods"""
        if not self.current_seeds:
            if not self.refresh_seeds():
                return None
        
        # Get recent betting history for pattern analysis
        recent_results = []
        if len(self.betting_history) > 0:
            recent_results = [bet['result'] for bet in self.betting_history[-history_length:]]
        
        # Use advanced predictor system
        next_nonce = self.current_seeds['nonce'] + 1
        prediction = self.predictor.predict_next_result(
            self.current_seeds['client_seed'],
            self.current_seeds['server_seed_hash'],
            next_nonce,
            recent_results
        )
        
        return prediction
    
    def optimize_bet(self, prediction, confidence):
        """Optimize bet based on prediction and strategy"""
        if not prediction:
            return None
        
        predicted_value = prediction['final_prediction']
        
        strategies = {
            'ultra_safe': {'margin': 4.0, 'max_bet_ratio': 0.02},
            'conservative': {'margin': 2.5, 'max_bet_ratio': 0.05},
            'moderate': {'margin': 1.5, 'max_bet_ratio': 0.10},
            'aggressive': {'margin': 0.8, 'max_bet_ratio': 0.20}
        }
        
        strategy_config = strategies.get(self.strategy, strategies['conservative'])
        margin = strategy_config['margin']
        max_bet_ratio = strategy_config['max_bet_ratio']
        
        # Determine bet direction and target
        if predicted_value < 50 - margin:
            condition = 'under'
            target = min(predicted_value + margin, 49.5)
        elif predicted_value > 50 + margin:
            condition = 'over'
            target = max(predicted_value - margin, 50.5)
        else:
            # Too close to 50, skip this bet
            return None
        
        # Calculate optimal bet amount
        current_balance = self.demo_balance if self.demo_mode else self.real_balance
        max_bet = current_balance * max_bet_ratio
        
        # Adjust bet amount based on confidence
        confidence_multiplier = min(confidence / 100.0, 1.0)
        optimal_bet = min(self.bet_amount * confidence_multiplier, max_bet)
        
        # Calculate potential payout
        if condition == 'under':
            multiplier = 99.0 / target
        else:  # over
            multiplier = 99.0 / (100 - target)
        
        return {
            'amount': round(optimal_bet, 2),
            'target': round(target, 2),
            'condition': condition,
            'multiplier': round(multiplier, 2),
            'predicted_result': predicted_value,
            'confidence': confidence,
            'margin': margin
        }
    
    def place_bet(self, bet_params):
        """Place bet using real Stake API"""
        if not bet_params:
            return None
        
        try:
            # Refresh seeds before betting
            if not self.refresh_seeds():
                print("❌ Failed to refresh seeds before betting")
                return None
            
            # Place bet through API
            result = self.stake_api.place_dice_bet(
                amount=bet_params['amount'],
                target=bet_params['target'],
                condition=bet_params['condition'],
                demo_mode=self.demo_mode
            )
            
            if result:
                # Update balances
                if self.demo_mode:
                    if result['won']:
                        self.demo_balance += result['payout'] - bet_params['amount']
                    else:
                        self.demo_balance -= bet_params['amount']
                else:
                    self.real_balance = self.stake_api.get_balance()
                
                # Update statistics
                profit = result['payout'] - bet_params['amount']
                self.total_profit += profit
                
                if result['won']:
                    self.win_streak += 1
                    self.loss_streak = 0
                else:
                    self.loss_streak += 1
                    self.win_streak = 0
                
                # Add to history
                bet_record = {
                    'timestamp': datetime.now().isoformat(),
                    'amount': bet_params['amount'],
                    'target': bet_params['target'],
                    'condition': bet_params['condition'],
                    'predicted': bet_params['predicted_result'],
                    'result': result['result'],
                    'won': result['won'],
                    'payout': result['payout'],
                    'profit': profit,
                    'balance': self.demo_balance if self.demo_mode else self.real_balance,
                    'multiplier': result['payoutMultiplier'],
                    'confidence': bet_params['confidence'],
                    'nonce': result.get('nonce', 0),
                    'demo': self.demo_mode
                }
                
                self.betting_history.append(bet_record)
                
                # Update nonce for next prediction
                if 'nonce' in result:
                    self.current_seeds['nonce'] = result['nonce']
                
                return bet_record
            
            return None
            
        except Exception as e:
            print(f"❌ Error placing bet: {e}")
            return None
    
    def run_bot_cycle(self):
        """Run one complete bot cycle"""
        try:
            # Get prediction
            prediction = self.get_next_prediction()
            if not prediction:
                return False
            
            # Optimize bet
            bet_params = self.optimize_bet(prediction, prediction['confidence'])
            if not bet_params:
                return False
            
            # Place bet
            bet_result = self.place_bet(bet_params)
            if bet_result:
                # Emit real-time update
                socketio.emit('bet_placed', bet_result)
                socketio.emit('stats_update', {
                    'balance': self.demo_balance if self.demo_mode else self.real_balance,
                    'total_profit': self.total_profit,
                    'win_streak': self.win_streak,
                    'loss_streak': self.loss_streak,
                    'total_bets': len(self.betting_history),
                    'win_rate': sum(1 for bet in self.betting_history if bet['won']) / len(self.betting_history) * 100 if self.betting_history else 0,
                    'api_connected': self.api_connected,
                    'current_seeds': self.current_seeds
                })
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Bot cycle error: {e}")
            return False
    
    def get_stats(self):
        """Get current bot statistics"""
        win_rate = 0
        if len(self.betting_history) > 0:
            win_rate = sum(1 for bet in self.betting_history if bet['won']) / len(self.betting_history) * 100
        
        return {
            'is_running': self.is_running,
            'demo_mode': self.demo_mode,
            'balance': self.demo_balance if self.demo_mode else self.real_balance,
            'strategy': self.strategy,
            'bet_amount': self.bet_amount,
            'total_profit': self.total_profit,
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'total_bets': len(self.betting_history),
            'win_rate': win_rate,
            'api_connected': self.api_connected,
            'current_seeds': self.current_seeds
        }

# Initialize bot with API key
api_key = os.getenv('STAKE_API_KEY')
if not api_key:
    print("❌ STAKE_API_KEY not found in .env file!")
    bot = None
else:
    print(f"✅ API Key found: {api_key[:10]}...")
    bot = RealStakePredictorBot(api_key)

# Flask routes
@app.route('/')
def dashboard():
    """Serve the main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current bot statistics"""
    if bot:
        return jsonify(bot.get_stats())
    return jsonify({'error': 'Bot not initialized'})

@app.route('/api/history')
def get_history():
    """Get recent betting history"""
    if bot:
        return jsonify(bot.betting_history[-50:])  # Return last 50 bets
    return jsonify([])

@app.route('/api/seeds/current')
def get_current_seeds():
    """Get current Stake seeds"""
    if bot and bot.api_connected:
        seeds = bot.stake_api.get_current_seeds()
        return jsonify(seeds if seeds else {'error': 'Failed to get seeds'})
    return jsonify({'error': 'API not connected'})

@app.route('/api/seeds/refresh', methods=['POST'])
def refresh_seeds():
    """Refresh seeds from Stake API"""
    if bot:
        success = bot.refresh_seeds()
        return jsonify({
            'success': success,
            'seeds': bot.current_seeds if success else None
        })
    return jsonify({'success': False, 'error': 'Bot not initialized'})

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    if bot:
        emit('stats_update', bot.get_stats())

@socketio.on('start_bot')
def handle_start_bot(data):
    """Start the bot"""
    if not bot:
        emit('error', {'message': 'Bot not initialized'})
        return
    
    if bot.is_running:
        emit('error', {'message': 'Bot is already running'})
        return
    
    # Update bot settings
    bot.strategy = data.get('strategy', 'conservative')
    bot.bet_amount = float(data.get('bet_amount', 1.0))
    bot.demo_mode = data.get('demo_mode', True)
    
    bot.is_running = True
    
    # Start bot in background thread
    def run_bot():
        while bot.is_running:
            try:
                success = bot.run_bot_cycle()
                if not success:
                    time.sleep(1)  # Short pause on failure
                else:
                    time.sleep(3)  # Normal pause between bets
            except Exception as e:
                print(f"Bot error: {e}")
                time.sleep(5)
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    emit('bot_started', bot.get_stats())
    print(f"🤖 Bot started - Strategy: {bot.strategy}, Demo: {bot.demo_mode}")

@socketio.on('stop_bot')
def handle_stop_bot():
    """Stop the bot"""
    if bot:
        bot.is_running = False
        emit('bot_stopped', bot.get_stats())
        print("🛑 Bot stopped")

@socketio.on('update_settings')
def handle_update_settings(data):
    """Update bot settings"""
    if bot:
        bot.strategy = data.get('strategy', bot.strategy)
        bot.bet_amount = float(data.get('bet_amount', bot.bet_amount))
        bot.demo_mode = data.get('demo_mode', bot.demo_mode)
        
        emit('settings_updated', bot.get_stats())
        print(f"⚙️ Settings updated - Strategy: {bot.strategy}, Amount: ${bot.bet_amount}")

@socketio.on('test_api')
def handle_test_api():
    """Test API connection"""
    if bot:
        success = bot.test_api_connection()
        emit('api_test_result', {
            'success': success,
            'connected': bot.api_connected,
            'balance': bot.real_balance if success else 0,
            'seeds': bot.current_seeds if success else {}
        })

@socketio.on('place_manual_bet')
def handle_manual_bet(data):
    """Place a manual bet"""
    if not bot or not bot.api_connected:
        emit('error', {'message': 'Bot not connected to API'})
        return
    
    try:
        bet_params = {
            'amount': float(data['amount']),
            'target': float(data['target']),
            'condition': data['condition'],
            'predicted_result': float(data.get('predicted', 50)),
            'confidence': float(data.get('confidence', 50))
        }
        
        result = bot.place_bet(bet_params)
        if result:
            emit('bet_placed', result)
            emit('stats_update', bot.get_stats())
        else:
            emit('error', {'message': 'Failed to place bet'})
            
    except Exception as e:
        emit('error', {'message': f'Error placing manual bet: {e}'})

if __name__ == '__main__':
    print("🚀 STAKE PREDICTOR DASHBOARD STARTING...")
    if api_key:
        print(f"✅ API Key Found: {api_key[:10]}...")
    else:
        print("❌ No API Key Found!")
    
    print("🌐 Dashboard URL: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)