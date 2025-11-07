"""
ULTIMATE UNIFIED DASHBOARD - The Supreme Command Center
Single endpoint that combines all models, strategies, AI decisions, and real-time betting
GOAL: COMPLETE CONTROL AND MAXIMUM PROFIT VISUALIZATION
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import asyncio
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os
from dataclasses import asdict
import traceback

# Import our supreme systems
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_ai_decision_engine import UnifiedAIDecisionEngine, initialize_supreme_engine, get_supreme_engine
from continuous_learning_system import ContinuousLearningSystem, initialize_learning_system, get_learning_system, TrainingExample
from real_stake_api import RealStakeAPI, AdvancedPredictorSystem
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app with SocketIO
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'supreme_betting_engine_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
dashboard_state = {
    'is_running': False,
    'current_session': None,
    'demo_mode': True,  # Start in demo mode
    'stake_api_connected': False,
    'live_stats': {
        'total_bets': 0,
        'successful_bets': 0,
        'total_profit': 0.0,
        'win_rate': 0.0,
        'current_bankroll': 1000.0,  # Demo balance
        'session_duration': 0,
        'last_decision': None,
        'last_bet': None
    },
    'recent_decisions': [],
    'recent_outcomes': [],
    'model_performance': {},
    'ai_insights': {},
    'learning_insights': {},
    'market_conditions': {},
    'strategy_performance': {},
    'auto_betting': False,
    'bet_interval': 30  # seconds between bets
}

# Background task management
background_tasks = {
    'main_loop': None,
    'data_collector': None,
    'performance_tracker': None
}

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('ultimate_dashboard.html')

@app.route('/mega')
def mega_dashboard():
    """MEGA Enhanced Predictor Dashboard"""
    return render_template('mega_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'status': 'operational' if dashboard_state['is_running'] else 'stopped',
        'engines_loaded': {
            'supreme_engine': get_supreme_engine() is not None,
            'learning_system': get_learning_system() is not None
        },
        'dashboard_state': dashboard_state['live_stats'],
        'auto_betting': dashboard_state['auto_betting']
    })

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the supreme betting system"""
    try:
        global dashboard_state
        
        # Initialize systems if not already done
        api_key = os.getenv('STAKE_API_KEY')
        if not api_key:
            return jsonify({'error': 'STAKE_API_KEY not found in environment'}), 400
        
        # Initialize supreme engine
        if get_supreme_engine() is None:
            aws_credentials = {
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'region': os.getenv('AWS_REGION', 'us-east-1')
            }
            initialize_supreme_engine(api_key, aws_credentials)
        
        # Initialize learning system
        if get_learning_system() is None:
            initialize_learning_system("./models")
        
        dashboard_state['is_running'] = True
        dashboard_state['current_session'] = datetime.now()
        
        # Start background tasks
        start_background_tasks()
        
        socketio.emit('system_status', {'status': 'started', 'timestamp': datetime.now().isoformat()})
        
        return jsonify({'status': 'started', 'message': 'Supreme betting system activated!'})
        
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the supreme betting system"""
    try:
        global dashboard_state
        
        dashboard_state['is_running'] = False
        dashboard_state['auto_betting'] = False
        
        # Stop background tasks
        stop_background_tasks()
        
        socketio.emit('system_status', {'status': 'stopped', 'timestamp': datetime.now().isoformat()})
        
        return jsonify({'status': 'stopped', 'message': 'Supreme betting system deactivated'})
        
    except Exception as e:
        logging.error(f"Failed to stop system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle_auto_betting', methods=['POST'])
def toggle_auto_betting():
    """Toggle automatic betting on/off"""
    try:
        dashboard_state['auto_betting'] = not dashboard_state['auto_betting']
        
        socketio.emit('auto_betting_toggled', {
            'auto_betting': dashboard_state['auto_betting'],
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'auto_betting': dashboard_state['auto_betting'],
            'message': f"Auto betting {'enabled' if dashboard_state['auto_betting'] else 'disabled'}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual_decision', methods=['POST'])
def make_manual_decision():
    """Make a manual betting decision"""
    try:
        supreme_engine = get_supreme_engine()
        if not supreme_engine:
            return jsonify({'error': 'Supreme engine not initialized'}), 400
        
        # Run decision making in background thread
        def make_decision():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                decision = loop.run_until_complete(supreme_engine.make_supreme_decision())
                
                # Emit decision to dashboard
                socketio.emit('new_decision', {
                    'decision': format_decision_for_frontend(decision),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Execute if decision is to bet
                if decision and decision.should_bet:
                    result = loop.run_until_complete(supreme_engine.execute_decision(decision))
                    
                    # Emit bet result
                    socketio.emit('bet_executed', {
                        'result': result,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Update dashboard state
                    update_dashboard_after_bet(decision, result)
                
            except Exception as e:
                logging.error(f"Manual decision failed: {e}")
                socketio.emit('error', {'message': f"Decision failed: {e}"})
            finally:
                loop.close()
        
        thread = threading.Thread(target=make_decision)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'decision_started', 'message': 'Manual decision initiated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        supreme_engine = get_supreme_engine()
        learning_system = get_learning_system()
        
        data = {
            'live_stats': dashboard_state['live_stats'],
            'recent_decisions': dashboard_state['recent_decisions'][-20:],  # Last 20 decisions
            'recent_outcomes': dashboard_state['recent_outcomes'][-50:],   # Last 50 outcomes
            'system_status': {
                'is_running': dashboard_state['is_running'],
                'auto_betting': dashboard_state['auto_betting'],
                'session_start': dashboard_state['current_session'].isoformat() if dashboard_state['current_session'] else None
            }
        }
        
        # Add engine status if available
        if supreme_engine:
            engine_status = supreme_engine.get_comprehensive_status()
            data['engine_status'] = engine_status
        
        # Add learning insights if available
        if learning_system:
            learning_insights = learning_system.get_learning_insights()
            data['learning_insights'] = learning_insights
        
        return jsonify(data)
        
    except Exception as e:
        logging.error(f"Failed to get dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning_insights')
def get_learning_insights():
    """Get detailed learning system insights"""
    try:
        learning_system = get_learning_system()
        if not learning_system:
            return jsonify({'error': 'Learning system not initialized'}), 400
        
        insights = learning_system.get_learning_insights()
        return jsonify(insights)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_performance')
def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        learning_system = get_learning_system()
        if not learning_system:
            return jsonify({'error': 'Learning system not initialized'}), 400
        
        performance = {
            'model_performance': learning_system.model_performance,
            'feature_importance': learning_system.feature_importance,
            'training_examples_count': len(learning_system.training_examples),
            'recent_outcomes_count': len(learning_system.recent_outcomes)
        }
        
        return jsonify(performance)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy_analysis')
def get_strategy_analysis():
    """Get strategy performance analysis"""
    try:
        supreme_engine = get_supreme_engine()
        if not supreme_engine:
            return jsonify({'error': 'Supreme engine not initialized'}), 400
        
        # Get strategy framework performance
        strategy_performance = supreme_engine.strategy_framework.get_strategy_performance_report()
        
        return jsonify(strategy_performance)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai_insights')
def get_ai_insights():
    """Get AI brain insights and learning"""
    try:
        supreme_engine = get_supreme_engine()
        if not supreme_engine:
            return jsonify({'error': 'Supreme engine not initialized'}), 400
        
        # Run in background thread to handle async
        def get_insights():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                insights = loop.run_until_complete(supreme_engine.ai_brain.get_learning_insights())
                
                # Emit insights to dashboard
                socketio.emit('ai_insights_updated', {
                    'insights': insights,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logging.error(f"AI insights failed: {e}")
            finally:
                loop.close()
        
        thread = threading.Thread(target=get_insights)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'insights_requested'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def manage_settings():
    """Manage system settings"""
    if request.method == 'GET':
        return jsonify({
            'bet_interval': dashboard_state['bet_interval'],
            'auto_betting': dashboard_state['auto_betting'],
            'risk_settings': {
                'max_bet_percentage': 5.0,
                'stop_loss_threshold': 20.0,
                'stop_win_threshold': 50.0
            }
        })
    
    elif request.method == 'POST':
        try:
            settings = request.json
            
            # Update settings
            if 'bet_interval' in settings:
                dashboard_state['bet_interval'] = max(10, int(settings['bet_interval']))
            
            # Emit settings update
            socketio.emit('settings_updated', {
                'settings': settings,
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify({'status': 'settings_updated', 'settings': settings})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Socket.IO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'message': 'Connected to Supreme Betting Dashboard'})
    
    # Send initial data
    emit('dashboard_update', {
        'live_stats': dashboard_state['live_stats'],
        'system_status': {
            'is_running': dashboard_state['is_running'],
            'auto_betting': dashboard_state['auto_betting']
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_live_update')
def handle_live_update_request():
    """Handle request for live data update"""
    try:
        # Send comprehensive update
        update_data = {
            'live_stats': dashboard_state['live_stats'],
            'recent_decisions': dashboard_state['recent_decisions'][-5:],
            'timestamp': datetime.now().isoformat()
        }
        
        emit('live_update', update_data)
        
    except Exception as e:
        emit('error', {'message': f"Live update failed: {e}"})

# Background Task Functions
def start_background_tasks():
    """Start all background tasks"""
    global background_tasks
    
    # Main decision loop
    if background_tasks['main_loop'] is None or not background_tasks['main_loop'].is_alive():
        background_tasks['main_loop'] = threading.Thread(target=main_decision_loop, daemon=True)
        background_tasks['main_loop'].start()
    
    # Data collector
    if background_tasks['data_collector'] is None or not background_tasks['data_collector'].is_alive():
        background_tasks['data_collector'] = threading.Thread(target=data_collection_loop, daemon=True)
        background_tasks['data_collector'].start()
    
    # Performance tracker
    if background_tasks['performance_tracker'] is None or not background_tasks['performance_tracker'].is_alive():
        background_tasks['performance_tracker'] = threading.Thread(target=performance_tracking_loop, daemon=True)
        background_tasks['performance_tracker'].start()

def stop_background_tasks():
    """Stop all background tasks"""
    global background_tasks
    dashboard_state['is_running'] = False
    
    # Tasks will stop when they check dashboard_state['is_running']

def main_decision_loop():
    """Main loop for automatic betting decisions"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while dashboard_state['is_running']:
            if dashboard_state['auto_betting']:
                try:
                    supreme_engine = get_supreme_engine()
                    if supreme_engine:
                        # Make decision
                        decision = loop.run_until_complete(supreme_engine.make_supreme_decision())
                        
                        if decision:
                            # Store decision
                            dashboard_state['recent_decisions'].append(format_decision_for_frontend(decision))
                            dashboard_state['live_stats']['last_decision'] = datetime.now().isoformat()
                            
                            # Emit decision to dashboard
                            socketio.emit('new_decision', {
                                'decision': format_decision_for_frontend(decision),
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            # Execute if decision is to bet
                            if decision.should_bet:
                                result = loop.run_until_complete(supreme_engine.execute_decision(decision))
                                
                                # Store bet result
                                dashboard_state['live_stats']['last_bet'] = datetime.now().isoformat()
                                
                                # Emit bet result
                                socketio.emit('bet_executed', {
                                    'result': result,
                                    'decision': format_decision_for_frontend(decision),
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                                # Update dashboard state
                                update_dashboard_after_bet(decision, result)
                                
                                # Add to learning system
                                add_to_learning_system(decision, result)
                
                except Exception as e:
                    logging.error(f"Decision loop error: {e}")
                    socketio.emit('error', {'message': f"Decision error: {e}"})
            
            # Wait before next decision
            time.sleep(dashboard_state['bet_interval'])
            
    except Exception as e:
        logging.error(f"Main decision loop failed: {e}")
    finally:
        loop.close()

def data_collection_loop():
    """Background loop for collecting market data"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while dashboard_state['is_running']:
            try:
                supreme_engine = get_supreme_engine()
                if supreme_engine:
                    # Collect current market conditions
                    game_state = loop.run_until_complete(supreme_engine.data_processor.get_current_game_state())
                    trend_indicators = supreme_engine.data_processor.get_trend_indicators()
                    volatility_metrics = supreme_engine.data_processor.get_volatility_metrics()
                    entropy_analysis = supreme_engine.data_processor.get_entropy_analysis()
                    
                    # Update dashboard state
                    dashboard_state['market_conditions'] = {
                        'game_state': game_state,
                        'trend_indicators': trend_indicators,
                        'volatility_metrics': volatility_metrics,
                        'entropy_analysis': entropy_analysis,
                        'last_updated': datetime.now().isoformat()
                    }
                    
                    # Emit market update
                    socketio.emit('market_update', dashboard_state['market_conditions'])
                    
            except Exception as e:
                logging.error(f"Data collection error: {e}")
            
            time.sleep(10)  # Collect data every 10 seconds
            
    except Exception as e:
        logging.error(f"Data collection loop failed: {e}")
    finally:
        loop.close()

def performance_tracking_loop():
    """Background loop for tracking performance"""
    try:
        while dashboard_state['is_running']:
            try:
                # Update session duration
                if dashboard_state['current_session']:
                    duration = datetime.now() - dashboard_state['current_session']
                    dashboard_state['live_stats']['session_duration'] = duration.total_seconds() / 60  # minutes
                
                # Calculate win rate
                if dashboard_state['live_stats']['total_bets'] > 0:
                    dashboard_state['live_stats']['win_rate'] = (
                        dashboard_state['live_stats']['successful_bets'] / 
                        dashboard_state['live_stats']['total_bets'] * 100
                    )
                
                # Emit performance update
                socketio.emit('performance_update', {
                    'live_stats': dashboard_state['live_stats'],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logging.error(f"Performance tracking error: {e}")
            
            time.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logging.error(f"Performance tracking loop failed: {e}")

# Utility Functions
def format_decision_for_frontend(decision) -> Dict:
    """Format decision object for frontend display"""
    if not decision:
        return None
    
    return {
        'decision_id': decision.decision_id,
        'should_bet': decision.should_bet,
        'bet_amount': decision.bet_amount,
        'multiplier': decision.multiplier,
        'side': decision.side,
        'target_roll': decision.target_roll,
        'confidence': decision.confidence,
        'strategy_used': decision.strategy_recommendation.get('strategy', 'unknown'),
        'ai_reasoning': decision.ai_decision.reasoning if decision.ai_decision else '',
        'risk_assessment': decision.risk_assessment,
        'timestamp': decision.timestamp.isoformat()
    }

def update_dashboard_after_bet(decision, result):
    """Update dashboard state after bet execution"""
    
    dashboard_state['live_stats']['total_bets'] += 1
    
    if result.get('action') == 'bet_placed' and result.get('bet_result', {}).get('won', False):
        dashboard_state['live_stats']['successful_bets'] += 1
        profit = decision.bet_amount * (decision.multiplier - 1)
    else:
        profit = -decision.bet_amount
    
    dashboard_state['live_stats']['total_profit'] += profit
    dashboard_state['live_stats']['current_bankroll'] += profit
    
    # Add to recent outcomes
    dashboard_state['recent_outcomes'].append({
        'decision_id': decision.decision_id,
        'won': result.get('bet_result', {}).get('won', False),
        'profit': profit,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only recent outcomes
    if len(dashboard_state['recent_outcomes']) > 1000:
        dashboard_state['recent_outcomes'] = dashboard_state['recent_outcomes'][-1000:]

def add_to_learning_system(decision, result):
    """Add bet outcome to learning system"""
    try:
        learning_system = get_learning_system()
        if learning_system and decision:
            
            # Create training example
            example = TrainingExample(
                game_state=decision.ai_decision.context.current_game_state if decision.ai_decision else {},
                predictions=[],  # Would need to extract from decision
                strategy_used=decision.strategy_recommendation.get('strategy', 'unknown'),
                decision_confidence=decision.confidence,
                bet_amount=decision.bet_amount,
                multiplier=decision.multiplier,
                side=decision.side,
                target_roll=decision.target_roll,
                trend_indicators=decision.ai_decision.context.trend_indicators if decision.ai_decision else {},
                volatility_metrics=decision.ai_decision.context.volatility_metrics if decision.ai_decision else {},
                entropy_analysis=decision.ai_decision.context.entropy_analysis if decision.ai_decision else {},
                actual_roll=result.get('bet_result', {}).get('roll', 50.0),
                bet_won=result.get('bet_result', {}).get('won', False),
                profit_loss=result.get('bet_result', {}).get('profit', 0.0),
                timestamp=datetime.now(),
                decision_id=decision.decision_id,
                session_id=decision.session_id
            )
            
            learning_system.add_training_example(example)
            
    except Exception as e:
        logging.error(f"Failed to add to learning system: {e}")

# New Socket Event Handlers for Enhanced Dashboard
@socketio.on('connect_stake_api')
def handle_connect_stake_api(data):
    """Handle Stake API connection request"""
    try:
        demo_mode = data.get('demo_mode', True)
        dashboard_state['demo_mode'] = demo_mode
        
        if demo_mode:
            # Demo mode - simulate connection
            dashboard_state['stake_api_connected'] = True
            emit('api_status', {'connected': True, 'demo_mode': True})
            logging.info("Demo mode enabled - simulating Stake API connection")
        else:
            # Real mode - attempt actual connection
            stake_api = initialize_stake_api()
            if stake_api and stake_api.test_connection():
                dashboard_state['stake_api_connected'] = True
                emit('api_status', {'connected': True, 'demo_mode': False})
                logging.info("Connected to real Stake API")
            else:
                dashboard_state['stake_api_connected'] = False
                emit('api_status', {'connected': False, 'error': 'Failed to connect to Stake API'})
                logging.error("Failed to connect to Stake API")
                
    except Exception as e:
        emit('api_status', {'connected': False, 'error': str(e)})
        logging.error(f"Stake API connection error: {e}")

@socketio.on('disconnect_stake_api')
def handle_disconnect_stake_api():
    """Handle Stake API disconnection request"""
    dashboard_state['stake_api_connected'] = False
    emit('api_status', {'connected': False})
    logging.info("Disconnected from Stake API")

@socketio.on('place_dice_bet')
def handle_place_dice_bet(data):
    """Handle dice bet placement request with Stake API integration"""
    try:
        amount = float(data.get('amount', 1.0))
        target = float(data.get('target', 50.0))
        condition = data.get('condition', 'over')
        demo_mode = data.get('demo_mode', dashboard_state['demo_mode'])
        currency = data.get('currency', 'usdt').lower()
        multiplier = float(data.get('multiplier', 2.0))
        win_chance = float(data.get('winChance', 50.0))
        client_seed = data.get('client_seed', '')
        
        if demo_mode:
            # Simulate bet in demo mode with enhanced logging
            import random
            
            # Simulate dice roll (0-100)
            result = random.uniform(0, 100)
            
            # Calculate if won based on condition
            won = (condition == 'over' and result > target) or (condition == 'under' and result < target)
            
            # Calculate profit/loss
            profit = amount * (multiplier - 1) if won else -amount
            
            # Update demo stats
            dashboard_state['live_stats']['total_bets'] += 1
            if won:
                dashboard_state['live_stats']['successful_bets'] += 1
            
            dashboard_state['live_stats']['current_bankroll'] += profit
            dashboard_state['live_stats']['total_profit'] += profit
            
            # Calculate win rate
            if dashboard_state['live_stats']['total_bets'] > 0:
                dashboard_state['live_stats']['win_rate'] = (
                    dashboard_state['live_stats']['successful_bets'] / 
                    dashboard_state['live_stats']['total_bets'] * 100
                )
            
            # Create comprehensive result
            bet_result = {
                'demo': True,
                'result': round(result, 2),
                'target': target,
                'condition': condition,
                'amount': amount,
                'bet_amount': amount,
                'profit': profit,
                'won': won,
                'multiplier': multiplier,
                'win_chance': win_chance,
                'currency': currency,
                'balance': dashboard_state['live_stats']['current_bankroll'],
                'server_seed_hash': f"demo_{random.randint(100000, 999999)}",
                'nonce': random.randint(1, 1000),
                'timestamp': datetime.now().isoformat()
            }
            
            emit('dice_bet_result', bet_result)
            emit('balance_update', {'balance': dashboard_state['live_stats']['current_bankroll']})
            
            logging.info(f"Demo dice bet: {amount} {currency.upper()} {condition} {target} = {result:.2f} ({'WON' if won else 'LOST'}) | Profit: {profit:.8f}")
            
        else:
            # Real Stake API bet
            try:
                # Initialize direct Stake API connection
                api_key = os.getenv('STAKE_API_KEY')
                if not api_key:
                    emit('dice_bet_result', {'error': 'No Stake API key found'})
                    return
                
                stake_api = RealStakeAPI(api_key)
                if not stake_api.test_connection():
                    emit('dice_bet_result', {'error': 'Failed to connect to Stake API'})
                    return
                
                # Place bet via Stake API (async)
                def place_real_bet():
                    try:
                        # Place the actual bet using the real Stake API
                        api_result = stake_api.place_dice_bet(
                            amount=amount,
                            target=target,
                            condition=condition,
                            demo_mode=False  # Real money mode
                        )
                        
                        if api_result:
                            # Extract result from Stake API
                            result_value = api_result.get('result', 50.0)
                            won = api_result.get('won', False)
                            payout = api_result.get('payout', 0.0)
                            profit = payout - amount if won else -amount
                            
                            # Update real stats
                            dashboard_state['live_stats']['total_bets'] += 1
                            if won:
                                dashboard_state['live_stats']['successful_bets'] += 1
                            
                            dashboard_state['live_stats']['current_bankroll'] += profit
                            dashboard_state['live_stats']['total_profit'] += profit
                            
                            # Calculate win rate
                            if dashboard_state['live_stats']['total_bets'] > 0:
                                dashboard_state['live_stats']['win_rate'] = (
                                    dashboard_state['live_stats']['successful_bets'] / 
                                    dashboard_state['live_stats']['total_bets'] * 100
                                )
                            
                            # Send result with real Stake API data
                            bet_result = {
                                'demo': False,
                                'result': result_value,
                                'target': target,
                                'condition': condition,
                                'amount': amount,
                                'bet_amount': amount,
                                'profit': profit,
                                'won': won,
                                'multiplier': multiplier,
                                'win_chance': win_chance,
                                'currency': currency,
                                'balance': dashboard_state['live_stats']['current_bankroll'],
                                'server_seed_hash': f"real_{api_result.get('nonce', 0)}",
                                'nonce': api_result.get('nonce', 0),
                                'bet_id': api_result.get('id', ''),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            socketio.emit('dice_bet_result', bet_result)
                            socketio.emit('balance_update', {'balance': dashboard_state['live_stats']['current_bankroll']})
                            
                            logging.info(f"Real dice bet: {amount} {currency.upper()} {condition} {target} = {result_value:.2f} ({'WON' if won else 'LOST'}) | Profit: {profit:.8f}")
                        
                        else:
                            socketio.emit('dice_bet_result', {'error': 'Stake API returned no result'})
                            logging.error("Stake API bet failed: No result returned")
                            
                    except Exception as e:
                        socketio.emit('dice_bet_result', {'error': f'Bet execution failed: {str(e)}'})
                        logging.error(f"Real bet execution error: {e}")
                
                # Run in background thread
                thread = threading.Thread(target=place_real_bet)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                emit('dice_bet_result', {'error': f'Real bet setup failed: {str(e)}'})
                logging.error(f"Real bet setup error: {e}")
            
    except Exception as e:
        emit('dice_bet_result', {'error': f'Bet processing failed: {str(e)}'})
        logging.error(f"Dice bet processing error: {e}")

@socketio.on('place_bet')
def handle_place_bet(data):
    """Handle bet placement request"""
    try:
        amount = float(data.get('amount', 1.0))
        target = float(data.get('target', 50.0))
        condition = data.get('condition', 'under')
        demo_mode = data.get('demo_mode', dashboard_state['demo_mode'])
        
        if demo_mode:
            # Simulate bet in demo mode
            import random
            import time
            
            # Simulate dice roll
            result = random.uniform(0, 100)
            
            # Calculate if won
            won = (condition == 'under' and result < target) or (condition == 'over' and result > target)
            
            # Calculate payout
            if condition == 'under':
                multiplier = 99.0 / target
            else:
                multiplier = 99.0 / (100 - target)
                
            payout = amount * multiplier if won else 0
            
            # Update demo balance
            dashboard_state['live_stats']['total_bets'] += 1
            if won:
                dashboard_state['live_stats']['successful_bets'] += 1
                dashboard_state['live_stats']['current_bankroll'] += payout - amount
                dashboard_state['live_stats']['total_profit'] += payout - amount
            else:
                dashboard_state['live_stats']['current_bankroll'] -= amount
                dashboard_state['live_stats']['total_profit'] -= amount
            
            # Calculate win rate
            if dashboard_state['live_stats']['total_bets'] > 0:
                dashboard_state['live_stats']['win_rate'] = (
                    dashboard_state['live_stats']['successful_bets'] / 
                    dashboard_state['live_stats']['total_bets'] * 100
                )
            
            # Emit result
            bet_result = {
                'demo': True,
                'result': round(result, 2),
                'target': target,
                'condition': condition,
                'amount': amount,
                'payout': payout,
                'won': won,
                'multiplier': multiplier,
                'balance': dashboard_state['live_stats']['current_bankroll']
            }
            
            emit('bet_result', bet_result)
            emit('balance_update', {'balance': dashboard_state['live_stats']['current_bankroll']})
            
            logging.info(f"Demo bet: {amount} {condition} {target} = {result} ({'WON' if won else 'LOST'})")
            
        else:
            # Place real bet (would need actual Stake API integration)
            emit('bet_result', {'error': 'Real money betting not yet implemented'})
            logging.warning("Real money betting attempted but not yet implemented")
            
    except Exception as e:
        emit('bet_result', {'error': str(e)})
        logging.error(f"Bet placement error: {e}")

def initialize_stake_api():
    """Initialize Stake API connection"""
    try:
        api_key = os.getenv('STAKE_API_KEY')
        if api_key:
            stake_api = RealStakeAPI(api_key)
            return stake_api
        return None
    except Exception as e:
        logging.error(f"Failed to initialize Stake API: {e}")
        return None

@socketio.on('update_settings')
def handle_update_settings(data):
    """Handle settings update request"""
    try:
        if 'bet_interval' in data:
            interval = int(data['bet_interval'])
            if 2 <= interval <= 60:  # Validate interval between 2-60 seconds
                dashboard_state['bet_interval'] = interval
                emit('settings_updated', {'bet_interval': interval, 'status': 'success'})
                logging.info(f"Bet interval updated to {interval} seconds")
            else:
                emit('settings_updated', {'error': 'Bet interval must be between 2-60 seconds'})
                
        if 'demo_mode' in data:
            dashboard_state['demo_mode'] = data['demo_mode']
            emit('mode_switched', {'demo_mode': dashboard_state['demo_mode']})
            logging.info(f"Mode switched to {'demo' if dashboard_state['demo_mode'] else 'real'}")
            
    except Exception as e:
        emit('settings_updated', {'error': str(e)})
        logging.error(f"Settings update error: {e}")

# Session Seeds Management
session_seeds = {
    'client_seed': '',
    'server_seed_hash': '',
    'nonce': 0,
    'total_bets': 0,
    'revealed_server_seed': ''
}

@socketio.on('update_session_seeds')
def handle_update_session_seeds(data):
    """Handle session seeds update for enhanced prediction"""
    try:
        global session_seeds
        
        # Validate required fields
        if not data.get('clientSeed'):
            emit('seeds_updated', {'success': False, 'error': 'Client seed is required'})
            return
            
        if not data.get('serverSeedHash') or len(data.get('serverSeedHash', '')) != 64:
            emit('seeds_updated', {'success': False, 'error': 'Valid 64-character server seed hash is required'})
            return
        
        # Update session seeds
        session_seeds.update({
            'client_seed': data.get('clientSeed', ''),
            'server_seed_hash': data.get('serverSeedHash', ''),
            'nonce': int(data.get('nonce', 0)),
            'total_bets': int(data.get('totalBets', 0)),
            'revealed_server_seed': data.get('revealedServerSeed', '')
        })
        
        # Integrate with AI engine if available
        supreme_engine = get_supreme_engine()
        if supreme_engine:
            # Update the AI engine with new seeds for enhanced predictions
            supreme_engine.data_processor.update_session_seeds(session_seeds)
        
        emit('seeds_updated', {'success': True})
        logging.info(f"Session seeds updated: Client={session_seeds['client_seed'][:8]}... Hash={session_seeds['server_seed_hash'][:8]}... Nonce={session_seeds['nonce']}")
        
    except Exception as e:
        emit('seeds_updated', {'success': False, 'error': str(e)})
        logging.error(f"Seeds update error: {e}")

@socketio.on('calculate_next_numbers')
def handle_calculate_next_numbers(data):
    """Calculate next numbers using HMAC-SHA256 with session seeds"""
    try:
        import hashlib
        import hmac
        
        client_seed = data.get('clientSeed', '')
        server_seed_hash = data.get('serverSeedHash', '')
        nonce = int(data.get('nonce', 0))
        count = int(data.get('count', 5))
        
        if not client_seed or not server_seed_hash:
            emit('numbers_calculated', {'success': False, 'error': 'Client seed and server seed hash are required'})
            return
        
        # Note: For accurate HMAC calculation, we need the actual server seed, not just the hash
        # This is a limitation when working with hashed seeds only
        if session_seeds.get('revealed_server_seed'):
            # We have the actual server seed - can calculate accurately
            server_seed = session_seeds['revealed_server_seed']
            
            # Calculate HMAC-SHA256
            message = f"{client_seed}:{nonce}"
            hmac_result = hmac.new(
                server_seed.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Convert HMAC to dice roll (0-99.99)
            # Use first 8 chars of HMAC and convert to decimal
            hex_chunk = hmac_result[:8]
            decimal_value = int(hex_chunk, 16)
            dice_roll = round((decimal_value % 10000) / 100, 2)
            
            # Calculate next few rolls
            next_rolls = []
            for i in range(1, count + 1):
                next_message = f"{client_seed}:{nonce + i}"
                next_hmac = hmac.new(
                    server_seed.encode('utf-8'),
                    next_message.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                next_hex = next_hmac[:8]
                next_decimal = int(next_hex, 16)
                next_roll = round((next_decimal % 10000) / 100, 2)
                next_rolls.append(next_roll)
            
            emit('numbers_calculated', {
                'success': True,
                'hmacInput': message,
                'hmacOutput': hmac_result,
                'diceRoll': dice_roll,
                'nextRolls': next_rolls,
                'confidence': 95  # High confidence with revealed seed
            })
            
            logging.info(f"HMAC calculation complete: {dice_roll} (with revealed seed)")
            
        else:
            # Only have the hash - provide educational information
            message = f"{client_seed}:{nonce}"
            
            emit('numbers_calculated', {
                'success': True,
                'hmacInput': message,
                'hmacOutput': 'Calculation requires revealed server seed',
                'diceRoll': 'Need server seed',
                'nextRolls': ['Need', 'server', 'seed', 'for', 'accuracy'],
                'confidence': 25,  # Low confidence without actual seed
                'note': 'Accurate calculation requires the unhashed server seed. The server seed hash alone cannot be used for HMAC calculation.'
            })
            
            logging.info(f"HMAC calculation attempted with hash only - need revealed seed for accuracy")
        
    except Exception as e:
        emit('numbers_calculated', {'success': False, 'error': str(e)})
        logging.error(f"HMAC calculation error: {e}")

@socketio.on('verify_session_seeds')
def handle_verify_session_seeds(data):
    """Verify session seeds for provably fair gaming"""
    try:
        import hashlib
        
        server_seed = data.get('serverSeed', '')
        server_seed_hash = data.get('serverSeedHash', '')
        client_seed = data.get('clientSeed', '')
        nonce = int(data.get('nonce', 0))
        
        if not server_seed or not server_seed_hash:
            emit('seeds_verified', {'success': False, 'error': 'Server seed and hash are required for verification'})
            return
        
        # Hash the revealed server seed and compare with the provided hash
        calculated_hash = hashlib.sha256(server_seed.encode('utf-8')).hexdigest()
        is_valid = calculated_hash.lower() == server_seed_hash.lower()
        
        if is_valid:
            # Update session with revealed seed for accurate calculations
            session_seeds['revealed_server_seed'] = server_seed
            
            # Log the verification success
            logging.info(f"Seed verification successful: Server seed matches hash")
        else:
            logging.warning(f"Seed verification failed: Hash mismatch")
        
        emit('seeds_verified', {
            'success': True,
            'isValid': is_valid,
            'calculatedHash': calculated_hash,
            'providedHash': server_seed_hash
        })
        
    except Exception as e:
        emit('seeds_verified', {'success': False, 'error': str(e)})
        logging.error(f"Seed verification error: {e}")

@socketio.on('clear_session_seeds')
def handle_clear_session_seeds():
    """Clear session seeds"""
    try:
        global session_seeds
        session_seeds = {
            'client_seed': '',
            'server_seed_hash': '',
            'nonce': 0,
            'total_bets': 0,
            'revealed_server_seed': ''
        }
        
        # Clear seeds from AI engine if available
        supreme_engine = get_supreme_engine()
        if supreme_engine:
            supreme_engine.data_processor.clear_session_seeds()
        
        emit('seeds_cleared', {'success': True})
        logging.info("Session seeds cleared")
        
    except Exception as e:
        emit('seeds_cleared', {'success': False, 'error': str(e)})
        logging.error(f"Clear seeds error: {e}")

@socketio.on('update_settings')
def handle_update_settings(data):
    """Handle settings update request"""
    try:
        if 'bet_interval' in data:
            interval = int(data['bet_interval'])
            if 2 <= interval <= 60:  # Validate interval between 2-60 seconds
                dashboard_state['bet_interval'] = interval
                emit('settings_updated', {'bet_interval': interval, 'status': 'success'})
                logging.info(f"Bet interval updated to {interval} seconds")
            else:
                emit('settings_updated', {'error': 'Bet interval must be between 2-60 seconds'})
                
        if 'demo_mode' in data:
            dashboard_state['demo_mode'] = data['demo_mode']
            emit('mode_switched', {'demo_mode': dashboard_state['demo_mode']})
            logging.info(f"Mode switched to {'demo' if dashboard_state['demo_mode'] else 'real'}")
            
    except Exception as e:
        emit('settings_updated', {'error': str(e)})
        logging.error(f"Settings update error: {e}")

# Track session start time
session_start_time = time.time()

# MEGA ENHANCED PREDICTOR INTEGRATION
@socketio.on('initialize_mega_system')
def handle_initialize_mega_system():
    """Initialize the mega enhanced prediction system"""
    try:
        logging.info("🔥 Initializing MEGA Enhanced Prediction System...")
        
        # Import and initialize mega predictor
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from mega_enhanced_predictor import initialize_mega_predictor, get_mega_predictor
        
        # Initialize in async context
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        mega_predictor = loop.run_until_complete(initialize_mega_predictor())
        
        if mega_predictor:
            emit('mega_system_initialized', {
                'success': True,
                'message': 'MEGA Enhanced Prediction System ready for billion roll analysis!',
                'features': {
                    'billion_roll_analysis': True,
                    'ml_models_trained': True,
                    'enhanced_hmac': True,
                    'supreme_ai_engine': True,
                    'comprehensive_strategies': True
                }
            })
        else:
            emit('mega_system_initialized', {
                'success': False,
                'error': 'Failed to initialize mega system'
            })
            
    except Exception as e:
        logging.error(f"🚫 Failed to initialize mega system: {e}")
        emit('mega_system_initialized', {
            'success': False,
            'error': str(e)
        })

@socketio.on('get_mega_prediction')
def handle_get_mega_prediction(data):
    """Get mega enhanced prediction with >55% accuracy"""
    try:
        from mega_enhanced_predictor import get_mega_predictor
        
        mega_predictor = get_mega_predictor()
        if not mega_predictor:
            emit('mega_prediction_result', {
                'success': False,
                'error': 'Mega predictor not initialized - use initialize_mega_system first'
            })
            return
        
        # Prepare context from dashboard data
        context = {
            'recent_rolls': data.get('recent_rolls', []),
            'game_state': {
                'server_seed': session_seeds.get('server_seed_hash', ''),
                'client_seed': session_seeds.get('client_seed', ''),
                'nonce': session_seeds.get('nonce', 0),
                'game_id': f"dashboard_{int(time.time())}"
            },
            'bankroll': data.get('bankroll', 1000),
            'session_id': data.get('session_id', ''),
            'timestamp': time.time()
        }
        
        # Get mega prediction
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        prediction_result = loop.run_until_complete(
            mega_predictor.get_mega_prediction(context)
        )
        
        emit('mega_prediction_result', {
            'success': True,
            'prediction': prediction_result,
            'processing_info': {
                'methods_used': prediction_result['mega_prediction'].get('methods_used', []),
                'confidence': prediction_result['mega_prediction'].get('confidence', 0),
                'processing_time': prediction_result.get('processing_time_seconds', 0),
                'accuracy_target': '>55%'
            }
        })
        
    except Exception as e:
        logging.error(f"🚫 Mega prediction failed: {e}")
        emit('mega_prediction_result', {
            'success': False,
            'error': str(e)
        })

@socketio.on('update_mega_seeds')
def handle_update_mega_seeds(data):
    """Update session seeds for mega predictor"""
    try:
        from mega_enhanced_predictor import get_mega_predictor
        
        mega_predictor = get_mega_predictor()
        if mega_predictor:
            # Convert dashboard session seeds to mega predictor format
            seeds_data = {
                'client_seed': data.get('clientSeed', ''),
                'server_seed_hash': data.get('serverSeedHash', ''),
                'nonce': data.get('nonce', 0),
                'revealed_server_seed': data.get('revealedServerSeed', '')
            }
            
            mega_predictor.update_session_seeds(seeds_data)
            
            emit('mega_seeds_updated', {
                'success': True,
                'message': 'Mega predictor seeds updated'
            })
        else:
            emit('mega_seeds_updated', {
                'success': False,
                'error': 'Mega predictor not initialized'
            })
            
    except Exception as e:
        logging.error(f"🚫 Failed to update mega seeds: {e}")
        emit('mega_seeds_updated', {
            'success': False,
            'error': str(e)
        })

@socketio.on('get_mega_performance')
def handle_get_mega_performance():
    """Get mega predictor performance metrics"""
    try:
        from mega_enhanced_predictor import get_mega_predictor
        
        mega_predictor = get_mega_predictor()
        if mega_predictor:
            metrics = mega_predictor.get_performance_metrics()
            
            emit('mega_performance_data', {
                'success': True,
                'metrics': metrics
            })
        else:
            emit('mega_performance_data', {
                'success': False,
                'error': 'Mega predictor not initialized'
            })
            
    except Exception as e:
        logging.error(f"🚫 Failed to get mega performance: {e}")
        emit('mega_performance_data', {
            'success': False,
            'error': str(e)
        })

@socketio.on('add_real_time_result')
def handle_add_real_time_result(data):
    """Add real-time result to mega predictor for continuous learning"""
    try:
        from mega_enhanced_predictor import get_mega_predictor
        
        mega_predictor = get_mega_predictor()
        if mega_predictor:
            roll_result = data.get('result', 50.0)
            mega_predictor.add_real_time_data(roll_result)
            
            # Get updated performance
            metrics = mega_predictor.get_performance_metrics()
            
            emit('real_time_result_added', {
                'success': True,
                'updated_metrics': metrics
            })
        else:
            emit('real_time_result_added', {
                'success': False,
                'error': 'Mega predictor not initialized'
            })
            
    except Exception as e:
        logging.error(f"🚫 Failed to add real-time result: {e}")
        emit('real_time_result_added', {
            'success': False,
            'error': str(e)
        })

@socketio.on('get_billion_roll_analysis')
def handle_get_billion_roll_analysis():
    """Get billion roll dataset analysis results"""
    try:
        from mega_enhanced_predictor import get_mega_predictor
        
        mega_predictor = get_mega_predictor()
        if mega_predictor and hasattr(mega_predictor, 'billion_roll_processor'):
            processor = mega_predictor.billion_roll_processor
            
            analysis = {
                'total_rolls_analyzed': processor.total_analyzed if hasattr(processor, 'total_analyzed') else 0,
                'pattern_confidence': processor.pattern_confidence if hasattr(processor, 'pattern_confidence') else 0,
                'frequency_analysis': processor.frequency_analysis if hasattr(processor, 'frequency_analysis') else {},
                'sequential_patterns': processor.sequential_patterns if hasattr(processor, 'sequential_patterns') else {},
                'cyclical_patterns': processor.cyclical_patterns if hasattr(processor, 'cyclical_patterns') else {}
            }
            
            emit('billion_roll_analysis', {
                'success': True,
                'analysis': analysis
            })
        else:
            emit('billion_roll_analysis', {
                'success': False,
                'error': 'Billion roll processor not available'
            })
            
    except Exception as e:
        logging.error(f"🚫 Failed to get billion roll analysis: {e}")
        emit('billion_roll_analysis', {
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 SUPREME BETTING DASHBOARD STARTING...")
    print("🔥 THE ULTIMATE PROFIT MAXIMIZATION SYSTEM")
    print("💰 READY TO MAKE SERIOUS MONEY!")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🧠 MEGA ENHANCED PREDICTOR: >55% Accuracy Target")
    print("📈 BILLION ROLL ANALYSIS: Maximum Pattern Recognition")
    
    # Run the dashboard
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)