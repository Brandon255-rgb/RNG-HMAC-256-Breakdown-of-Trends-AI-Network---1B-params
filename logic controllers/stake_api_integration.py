#!/usr/bin/env python3
"""
Stake API Integration for Real-Time Prediction System
Provides live game state, current nonce tracking, and betting history
"""

import os
import time
import requests
import json
import hashlib
import hmac
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
from pathlib import Path

# Environment variable setup
from dotenv import load_dotenv
load_dotenv()

@dataclass
class StakeBetResult:
    """Single bet result from Stake API"""
    id: str
    amount: float
    payout: float
    result: float
    win: bool
    timestamp: str
    nonce: Optional[int] = None
    
@dataclass
class StakeGameState:
    """Current game state information"""
    current_nonce: int
    client_seed: str
    server_seed: str
    server_seed_hashed: str
    total_bets: int
    recent_results: List[float]
    is_seed_active: bool

class StakeAPIClient:
    """
    Enhanced Stake API client for real-time game tracking
    """
    
    def __init__(self, api_key: str = None, cf_clearance: str = None, 
                 cf_bm: str = None, cfuvid: str = None):
        """Initialize with API credentials"""
        
        # Try to load from environment if not provided
        self.api_key = api_key or os.getenv('STAKE_API_KEY')
        self.cf_clearance = cf_clearance or os.getenv('STAKE_CF_CLEARANCE')
        self.cf_bm = cf_bm or os.getenv('STAKE_CF_BM')
        self.cfuvid = cfuvid or os.getenv('STAKE_CFUVID')
        
        if not all([self.api_key, self.cf_clearance, self.cf_bm, self.cfuvid]):
            print("⚠️  Missing API credentials. Some features may not work.")
            print("Set environment variables: STAKE_API_KEY, STAKE_CF_CLEARANCE, STAKE_CF_BM, STAKE_CFUVID")
            
        self.session = requests.Session()
        self.api_url = 'https://stake.com/_api/graphql'
        
        # Setup headers
        self.headers = {
            'content-type': 'application/json',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'x-access-token': self.api_key,
            'cookie': f'cf_clearance={self.cf_clearance}; __cf_bm={self.cf_bm}; _cfuvid={self.cfuvid}'
        }
        self.session.headers.update(self.headers)
        
        # Internal tracking
        self.current_game_state = None
        self.bet_history = []
        
        print(f"🔗 Stake API Client initialized")
        if self.api_key:
            print(f"   API Key: {self.api_key[:10]}...")
            self.test_connection()
        
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            balance_data = self.get_user_balances()
            if balance_data:
                print("✅ API connection successful!")
                return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def get_user_balances(self) -> List[Dict]:
        """Get user balance information"""
        query = '''
        query UserBalances {
            user {
                id
                balances {
                    available {
                        amount
                        currency
                    }
                    vault {
                        amount
                        currency
                    }
                }
            }
        }
        '''
        
        try:
            response = self.session.post(self.api_url, json={'query': query})
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ Balance query error: {data['errors']}")
                return []
                
            balances = data.get('data', {}).get('user', {}).get('balances', [])
            print(f"💰 Account balances retrieved: {len(balances)} currencies")
            
            return balances
            
        except Exception as e:
            print(f"❌ Error getting balances: {e}")
            return []
    
    def get_current_seeds(self) -> Optional[StakeGameState]:
        """Get current client/server seed information"""
        query = '''
        query CurrentSeeds {
            user {
                id
                activeClientSeed {
                    seed
                }
                activeServerSeed {
                    seedHash
                    nonce
                }
            }
        }
        '''
        
        try:
            response = self.session.post(self.api_url, json={'query': query})
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ Seeds query error: {data['errors']}")
                return None
                
            user_data = data.get('data', {}).get('user', {})
            client_seed_data = user_data.get('activeClientSeed', {})
            server_seed_data = user_data.get('activeServerSeed', {})
            
            if not client_seed_data or not server_seed_data:
                print("⚠️  Could not retrieve seed information")
                return None
                
            game_state = StakeGameState(
                current_nonce=server_seed_data.get('nonce', 0),
                client_seed=client_seed_data.get('seed', ''),
                server_seed='',  # Server seed is only revealed after rotation
                server_seed_hashed=server_seed_data.get('seedHash', ''),
                total_bets=server_seed_data.get('nonce', 0),
                recent_results=[],
                is_seed_active=True
            )
            
            print(f"🎲 Current game state retrieved:")
            print(f"   Client Seed: {game_state.client_seed}")
            print(f"   Server Hash: {game_state.server_seed_hashed[:20]}...")
            print(f"   Current Nonce: {game_state.current_nonce}")
            
            self.current_game_state = game_state
            return game_state
            
        except Exception as e:
            print(f"❌ Error getting seeds: {e}")
            return None
    
    def get_bet_history(self, limit: int = 50, game_type: str = "dice") -> List[StakeBetResult]:
        """Get recent betting history"""
        query = '''
        query BetHistory($limit: Int!, $offset: Int!) {
            user {
                bets(limit: $limit, offset: 0) {
                    data {
                        id
                        amount
                        payout
                        payoutMultiplier
                        updatedAt
                        currency
                        game
                        state
                    }
                }
            }
        }
        '''
        
        variables = {
            'limit': limit,
            'offset': 0
        }
        
        try:
            response = self.session.post(self.api_url, json={
                'query': query,
                'variables': variables
            })
            data = response.json()
            
            if 'errors' in data:
                print(f"❌ History query error: {data['errors']}")
                return []
                
            bets_data = data.get('data', {}).get('user', {}).get('bets', {}).get('data', [])
            
            bet_results = []
            for bet in bets_data:
                if bet.get('game', '').lower() == game_type.lower():
                    # Extract dice result from state
                    state = bet.get('state', {})
                    result = state.get('result', 0.0) if isinstance(state, dict) else 0.0
                    
                    bet_result = StakeBetResult(
                        id=bet.get('id', ''),
                        amount=bet.get('amount', 0.0),
                        payout=bet.get('payout', 0.0),
                        result=result,
                        win=bet.get('payoutMultiplier', 0) > 1,
                        timestamp=bet.get('updatedAt', ''),
                        nonce=None  # Will be calculated
                    )
                    bet_results.append(bet_result)
            
            print(f"📊 Retrieved {len(bet_results)} {game_type} bets")
            self.bet_history = bet_results
            return bet_results
            
        except Exception as e:
            print(f"❌ Error getting bet history: {e}")
            return []
    
    def calculate_nonce_from_timestamp(self, bet_results: List[StakeBetResult], 
                                     current_nonce: int) -> List[StakeBetResult]:
        """Calculate approximate nonce values for historical bets"""
        if not bet_results:
            return bet_results
            
        # Sort by timestamp (newest first)
        sorted_bets = sorted(bet_results, key=lambda x: x.timestamp, reverse=True)
        
        # Assign nonces working backwards from current
        nonce = current_nonce
        for bet in sorted_bets:
            bet.nonce = nonce
            nonce -= 1
            
        print(f"🔢 Calculated nonces: {current_nonce - len(sorted_bets) + 1} to {current_nonce}")
        return sorted_bets
    
    def predict_next_hash_results(self, client_seed: str, server_seed_hash: str, 
                                 start_nonce: int, count: int = 10) -> List[Dict]:
        """
        Predict next results using known hash algorithm
        Note: This requires the actual server seed, not just the hash
        """
        print(f"🔮 Predicting next {count} hash-based results...")
        print(f"⚠️  Note: Requires actual server seed for accuracy")
        
        # For now, return placeholder predictions
        # In a real implementation, you'd need the revealed server seed
        predictions = []
        for i in range(count):
            nonce = start_nonce + i + 1
            # Placeholder calculation - would use actual HMAC-SHA512
            predicted_result = ((nonce * 17) % 10000) / 100  # Fake calculation
            
            predictions.append({
                'nonce': nonce,
                'predicted_result': predicted_result,
                'method': 'hash_prediction',
                'confidence': 0.5  # Low confidence without actual server seed
            })
            
        return predictions
    
    def get_real_time_environment(self) -> Dict:
        """Get complete real-time environment state"""
        print(f"🌐 Gathering real-time environment data...")
        
        # Get current seeds
        game_state = self.get_current_seeds()
        if not game_state:
            return {'error': 'Could not retrieve game state'}
        
        # Get recent betting history
        bet_history = self.get_bet_history(limit=100)
        
        # Calculate nonces for historical bets
        if bet_history and game_state:
            bet_history = self.calculate_nonce_from_timestamp(bet_history, game_state.current_nonce)
        
        # Extract recent results
        recent_results = [bet.result for bet in bet_history[:20]]  # Last 20 results
        game_state.recent_results = recent_results
        
        # Get balance information
        balances = self.get_user_balances()
        
        environment = {
            'game_state': game_state,
            'bet_history': bet_history,
            'recent_results': recent_results,
            'balances': balances,
            'timestamp': datetime.now().isoformat(),
            'total_historical_bets': len(bet_history),
            'api_connected': True
        }
        
        print(f"✅ Environment captured:")
        print(f"   Current Nonce: {game_state.current_nonce}")
        print(f"   Recent Results: {len(recent_results)} values")
        print(f"   Historical Bets: {len(bet_history)} bets")
        
        return environment
    
    def monitor_live_changes(self, callback_function=None, interval: int = 10):
        """
        Monitor for live changes in game state
        Calls callback_function when new bets are detected
        """
        print(f"👁️  Starting live monitoring (checking every {interval}s)")
        
        last_nonce = 0
        if self.current_game_state:
            last_nonce = self.current_game_state.current_nonce
        
        while True:
            try:
                # Check current state
                current_state = self.get_current_seeds()
                if current_state and current_state.current_nonce > last_nonce:
                    print(f"🎯 New bet detected! Nonce: {last_nonce} → {current_state.current_nonce}")
                    
                    # Get latest bet
                    new_bets = self.get_bet_history(limit=5)
                    if new_bets and callback_function:
                        callback_function(new_bets[0])  # Call with newest bet
                    
                    last_nonce = current_state.current_nonce
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("🛑 Live monitoring stopped")
                break
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                time.sleep(interval)

class StakeEnvironmentTracker:
    """
    High-level tracker that combines API data with prediction models
    """
    
    def __init__(self, api_client: StakeAPIClient):
        self.api_client = api_client
        self.current_environment = None
        self.prediction_queue = queue.Queue()
        
    def sync_environment(self) -> Dict:
        """Synchronize current environment state"""
        print(f"🔄 Synchronizing environment...")
        
        environment = self.api_client.get_real_time_environment()
        self.current_environment = environment
        
        return environment
    
    def get_current_position(self) -> Dict:
        """Get exact current position in betting sequence"""
        if not self.current_environment:
            self.sync_environment()
        
        game_state = self.current_environment.get('game_state')
        if not game_state:
            return {'error': 'No game state available'}
        
        position_info = {
            'current_nonce': game_state.current_nonce,
            'next_nonce': game_state.current_nonce + 1,
            'client_seed': game_state.client_seed,
            'server_seed_hash': game_state.server_seed_hashed,
            'total_bets_made': game_state.total_bets,
            'seed_position': f"Bet {game_state.current_nonce} of current seed pair"
        }
        
        print(f"📍 Current Position:")
        print(f"   Next Nonce: {position_info['next_nonce']}")
        print(f"   Client Seed: {position_info['client_seed']}")
        print(f"   Total Bets: {position_info['total_bets_made']}")
        
        return position_info
    
    def predict_next_with_context(self, count: int = 5) -> List[Dict]:
        """Generate predictions with full environmental context"""
        # Get current position
        position = self.get_current_position()
        if 'error' in position:
            return [{'error': position['error']}]
        
        # Generate predictions using available methods
        predictions = []
        
        # Method 1: Hash-based prediction (if we have server seed)
        if 'server_seed' in position and position.get('server_seed'):
            hash_predictions = self.api_client.predict_next_hash_results(
                position['client_seed'],
                position['server_seed'],
                position['current_nonce'],
                count
            )
            predictions.extend(hash_predictions)
        
        # Method 2: Pattern analysis from recent results
        recent_results = self.current_environment.get('recent_results', [])
        if len(recent_results) >= 10:
            pattern_predictions = self._analyze_recent_patterns(recent_results, count)
            predictions.extend(pattern_predictions)
        
        # Method 3: AI model predictions (if available)
        # This would integrate with your existing AI models
        
        print(f"🎯 Generated {len(predictions)} contextual predictions")
        return predictions
    
    def _analyze_recent_patterns(self, recent_results: List[float], count: int) -> List[Dict]:
        """Analyze patterns in recent results"""
        if len(recent_results) < 5:
            return []
        
        import numpy as np
        
        # Simple pattern analysis
        predictions = []
        mean_result = np.mean(recent_results)
        trend = np.mean(recent_results[-5:]) - np.mean(recent_results[-10:]) if len(recent_results) >= 10 else 0
        
        for i in range(count):
            # Trend-based prediction
            predicted_value = mean_result + (trend * (i + 1))
            predicted_value = max(0, min(100, predicted_value))  # Clamp to valid range
            
            predictions.append({
                'sequence': i + 1,
                'predicted_result': predicted_value,
                'method': 'pattern_analysis',
                'confidence': 0.6,
                'based_on': f'{len(recent_results)} recent results'
            })
        
        return predictions

def main():
    """Demo the API integration"""
    print("🚀 STAKE API INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize API client
    api_client = StakeAPIClient()
    
    if not api_client.api_key:
        print("⚠️  No API key provided. Demo mode only.")
        print("\nTo use with real API:")
        print("1. Set environment variables in .env file:")
        print("   STAKE_API_KEY=your_api_key")
        print("   STAKE_CF_CLEARANCE=your_clearance")
        print("   STAKE_CF_BM=your_cf_bm")
        print("   STAKE_CFUVID=your_cfuvid")
        print("\n2. Get these values from browser developer tools")
        return
    
    # Initialize environment tracker
    tracker = StakeEnvironmentTracker(api_client)
    
    while True:
        print("\n🎮 Options:")
        print("1. Sync current environment")
        print("2. Get current position")
        print("3. Get betting history")
        print("4. Predict next results")
        print("5. Start live monitoring")
        print("6. Exit")
        
        try:
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == '1':
                environment = tracker.sync_environment()
                if 'error' not in environment:
                    game_state = environment['game_state']
                    print(f"\n✅ Environment synced!")
                    print(f"   Client Seed: {game_state.client_seed}")
                    print(f"   Current Nonce: {game_state.current_nonce}")
                    print(f"   Recent Results: {len(game_state.recent_results)}")
                    
            elif choice == '2':
                position = tracker.get_current_position()
                if 'error' not in position:
                    print(f"\n📍 Current Position Details:")
                    for key, value in position.items():
                        print(f"   {key}: {value}")
                        
            elif choice == '3':
                history = api_client.get_bet_history(limit=20)
                if history:
                    print(f"\n📊 Recent Betting History:")
                    for i, bet in enumerate(history[:10]):
                        win_status = "🟢 WIN" if bet.win else "🔴 LOSS"
                        print(f"   {i+1}. Result: {bet.result:.2f} - {win_status}")
                        
            elif choice == '4':
                predictions = tracker.predict_next_with_context(count=5)
                if predictions and 'error' not in predictions[0]:
                    print(f"\n🔮 Next Predictions:")
                    for pred in predictions:
                        method = pred.get('method', 'unknown')
                        result = pred.get('predicted_result', 0)
                        conf = pred.get('confidence', 0)
                        print(f"   {method}: {result:.2f} (confidence: {conf:.1%})")
                        
            elif choice == '5':
                def on_new_bet(bet_result):
                    print(f"🎲 NEW BET: {bet_result.result:.2f} - {'WIN' if bet_result.win else 'LOSS'}")
                    
                print("Starting live monitoring... Press Ctrl+C to stop")
                api_client.monitor_live_changes(callback_function=on_new_bet, interval=5)
                
            elif choice == '6':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()