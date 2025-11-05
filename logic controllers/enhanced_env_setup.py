"""
Enhanced Environment Setup for Stake API Integration
Handles API credentials, session management, and real-time iteration tracking
"""

import os
import json
import requests
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

@dataclass
class StakeConfig:
    """Configuration class for Stake API integration"""
    api_key: str
    base_url: str = "https://stake.com/_api/graphql"
    ws_url: str = "wss://ws.stake.com/socket.io/"
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    timeout: int = 30
    max_retries: int = 3

class EnhancedStakeAPI:
    """Enhanced Stake API client with real-time tracking capabilities"""
    
    def __init__(self, config: StakeConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {config.api_key}',
            'X-API-Key': config.api_key
        })
        
        # Real-time tracking variables
        self.current_seed = None
        self.current_nonce = 0
        self.total_bets_placed = 0
        self.session_start_time = datetime.now(timezone.utc)
        self.bet_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def authenticate(self) -> bool:
        """Verify API authentication and get user profile"""
        query = """
        query UserProfile {
            user {
                id
                name
                balance {
                    available
                    vault
                    currency
                }
                preferences {
                    currency
                    language
                }
            }
        }
        """
        
        try:
            response = self._make_request(query)
            if response and 'data' in response and response['data']['user']:
                user_data = response['data']['user']
                self.logger.info(f"✅ Authenticated as: {user_data['name']}")
                self.logger.info(f"💰 Balance: {user_data['balance']['available']} {user_data['balance']['currency']}")
                return True
            else:
                self.logger.error("❌ Authentication failed")
                return False
        except Exception as e:
            self.logger.error(f"❌ Authentication error: {e}")
            return False
    
    def get_current_game_state(self) -> Dict[str, Any]:
        """Get current game state and active seeds"""
        query = """
        query GameState {
            user {
                activeSeeds {
                    id
                    seed
                    nonce
                    game
                    createdAt
                }
                recentBets: bets(limit: 10) {
                    id
                    game
                    nonce
                    seed
                    amount
                    payout
                    createdAt
                }
            }
        }
        """
        
        try:
            response = self._make_request(query)
            if response and 'data' in response:
                return response['data']
            return {}
        except Exception as e:
            self.logger.error(f"❌ Error getting game state: {e}")
            return {}
    
    def detect_current_iteration(self) -> Dict[str, Any]:
        """
        Detect exact current iteration position for predictions
        This is the KEY function you need for knowing where you are!
        """
        game_state = self.get_current_game_state()
        
        if not game_state or 'user' not in game_state:
            return {"error": "Could not fetch game state"}
        
        user_data = game_state['user']
        
        # Get active seeds (current game sessions)
        active_seeds = user_data.get('activeSeeds', [])
        recent_bets = user_data.get('recentBets', [])
        
        iteration_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_sessions": len(active_seeds),
            "seeds": {},
            "current_positions": {},
            "next_nonces": {},
            "prediction_ready": False
        }
        
        # Process each active seed
        for seed_data in active_seeds:
            seed_hash = seed_data['seed']
            current_nonce = seed_data['nonce']
            game_type = seed_data['game']
            
            iteration_info["seeds"][game_type] = {
                "seed": seed_hash,
                "current_nonce": current_nonce,
                "next_nonce": current_nonce + 1,
                "session_age": seed_data['createdAt']
            }
            
            iteration_info["current_positions"][game_type] = current_nonce
            iteration_info["next_nonces"][game_type] = current_nonce + 1
        
        # If we have dice game active, we're ready for predictions!
        if 'dice' in iteration_info["seeds"]:
            iteration_info["prediction_ready"] = True
            iteration_info["dice_seed"] = iteration_info["seeds"]["dice"]["seed"]
            iteration_info["dice_next_nonce"] = iteration_info["next_nonces"]["dice"]
            
            self.logger.info(f"🎲 DICE GAME DETECTED!")
            self.logger.info(f"🔢 Current Seed: {iteration_info['dice_seed']}")
            self.logger.info(f"🎯 Next Nonce: {iteration_info['dice_next_nonce']}")
            self.logger.info(f"✅ Ready for prediction!")
        
        return iteration_info
    
    def monitor_real_time_changes(self, callback_function=None) -> None:
        """
        Monitor real-time changes in betting state
        Calls callback function when new bets are detected
        """
        self.logger.info("🔄 Starting real-time monitoring...")
        
        last_known_state = {}
        
        while True:
            try:
                current_state = self.detect_current_iteration()
                
                # Check for changes
                if current_state != last_known_state:
                    self.logger.info("🔔 State change detected!")
                    
                    if callback_function:
                        callback_function(current_state, last_known_state)
                    
                    # Log important changes
                    if current_state.get("prediction_ready"):
                        seed = current_state.get("dice_seed", "Unknown")
                        nonce = current_state.get("dice_next_nonce", 0)
                        self.logger.info(f"🎲 NEW PREDICTION OPPORTUNITY: Seed={seed[:16]}... Nonce={nonce}")
                    
                    last_known_state = current_state.copy()
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(5)  # Wait longer on error
    
    def get_betting_history(self, limit: int = 100) -> List[Dict]:
        """Get detailed betting history for analysis"""
        query = f"""
        query BettingHistory {{
            user {{
                bets(limit: {limit}) {{
                    id
                    game
                    nonce
                    seed
                    amount
                    payout
                    multiplier
                    profit
                    createdAt
                    outcome
                }}
            }}
        }}
        """
        
        try:
            response = self._make_request(query)
            if response and 'data' in response and response['data']['user']:
                return response['data']['user']['bets']
            return []
        except Exception as e:
            self.logger.error(f"❌ Error getting betting history: {e}")
            return []
    
    def _make_request(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Make GraphQL request with error handling"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self.config.base_url,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.warning(f"⚠️ Request failed with status {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None

def setup_enhanced_environment():
    """Setup enhanced environment with all necessary configurations"""
    
    # Verify API key
    api_key = os.getenv('STAKE_API_KEY')
    if not api_key:
        print("❌ ERROR: STAKE_API_KEY not found in environment!")
        print("📝 Please add your API key to the .env file")
        return None
    
    print(f"✅ API Key loaded: {api_key[:16]}...{api_key[-8:]}")
    
    # Create configuration
    config = StakeConfig(api_key=api_key)
    
    # Initialize API client
    api_client = EnhancedStakeAPI(config)
    
    # Test authentication
    if not api_client.authenticate():
        print("❌ Failed to authenticate with Stake API")
        return None
    
    print("🚀 Enhanced environment setup complete!")
    return api_client

def demo_iteration_detection():
    """Demo function to show iteration detection in action"""
    print("\n🎯 ITERATION DETECTION DEMO")
    print("=" * 50)
    
    api_client = setup_enhanced_environment()
    if not api_client:
        return
    
    # Get current state
    current_state = api_client.detect_current_iteration()
    
    print("\n📊 CURRENT GAME STATE:")
    print(f"⏰ Timestamp: {current_state.get('timestamp', 'Unknown')}")
    print(f"🎮 Active Sessions: {current_state.get('active_sessions', 0)}")
    print(f"🎲 Prediction Ready: {current_state.get('prediction_ready', False)}")
    
    if current_state.get("prediction_ready"):
        print(f"\n🎯 DICE GAME ACTIVE:")
        print(f"🔢 Seed: {current_state.get('dice_seed', 'Unknown')}")
        print(f"🎲 Next Nonce: {current_state.get('dice_next_nonce', 'Unknown')}")
        print(f"\n✅ YOU CAN NOW MAKE PREDICTIONS!")
    else:
        print(f"\n⚠️ No active dice games detected")
        print(f"💡 Start a dice game on Stake to begin predictions")
    
    # Show recent betting history
    history = api_client.get_betting_history(limit=5)
    if history:
        print(f"\n📈 RECENT BETS:")
        for bet in history[:3]:
            print(f"  🎲 {bet['game']}: Nonce {bet['nonce']}, Profit: {bet.get('profit', 0)}")

if __name__ == "__main__":
    # Run demo
    demo_iteration_detection()
    
    print(f"\n🔄 Want to monitor real-time? Uncomment the line below:")
    print(f"# api_client.monitor_real_time_changes()")