#!/usr/bin/env python3
"""
CLOUDSCRAPER STAKE API ACCESS
============================
Using CloudScraper to bypass Cloudflare and access Stake API for real betting
"""

import cloudscraper
import json
import time
import os
from dotenv import load_dotenv

class StakeAPIAccess:
    def __init__(self, api_key):
        self.api_key = api_key
        self.scraper = None
        self.setup_scraper()
        
    def setup_scraper(self):
        """Setup CloudScraper with optimal settings"""
        print("🔥 Setting up CloudScraper for Stake...")
        
        # Create scraper with browser simulation
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            },
            delay=10,  # Delay between requests
            debug=False
        )
        
        # Set realistic headers
        self.scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'identity',  # Don't compress response
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Origin': 'https://stake.com',
            'Referer': 'https://stake.com/',
            'Authorization': f'Bearer {self.api_key}',
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        })
        
        print("✅ CloudScraper configured")
    
    def test_api_access(self):
        """Test API access with CloudScraper"""
        print("\n🔍 Testing Stake API access...")
        
        try:
            # First, visit main page to establish session
            print("   Visiting main page...")
            main_response = self.scraper.get('https://stake.com/', timeout=30)
            print(f"   Main page: {main_response.status_code}")
            
            if main_response.status_code != 200:
                print(f"   ❌ Failed to load main page: {main_response.status_code}")
                return False
            
            # Wait a bit
            time.sleep(3)
            
            # Now try GraphQL API
            print("   Testing GraphQL API...")
            
            query = {
                "query": """
                query {
                    user {
                        id
                        name
                        balances {
                            available {
                                amount
                                currency
                            }
                        }
                        activeClientSeed {
                            seed
                        }
                        activeServerSeed {
                            seedHash
                            nonce
                        }
                    }
                }
                """
            }
            
            api_response = self.scraper.post(
                'https://stake.com/_api/graphql',
                json=query,
                timeout=30
            )
            
            print(f"   API Response: {api_response.status_code}")
            
            if api_response.status_code == 200:
                try:
                    data = api_response.json()
                    if 'data' in data and data['data'] and 'user' in data['data']:
                        user = data['data']['user']
                        print(f"   ✅ SUCCESS! Connected as: {user['name']}")
                        
                        # Print balance info
                        for balance in user['balances']:
                            amount = balance['available']['amount']
                            currency = balance['available']['currency']
                            print(f"   💰 {currency}: {amount}")
                        
                        # Print seed info
                        if 'activeClientSeed' in user and user['activeClientSeed']:
                            print(f"   🌱 Client Seed: {user['activeClientSeed']['seed']}")
                        if 'activeServerSeed' in user and user['activeServerSeed']:
                            print(f"   🌱 Server Hash: {user['activeServerSeed']['seedHash'][:20]}...")
                            print(f"   🔢 Nonce: {user['activeServerSeed']['nonce']}")
                        
                        return True
                    else:
                        print(f"   ❌ Invalid API response structure: {data}")
                        return False
                        
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response: {api_response.text[:200]}")
                    return False
            else:
                print(f"   ❌ API request failed: {api_response.status_code}")
                print(f"   Response: {api_response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def place_test_bet(self, demo=True):
        """Place a test bet using the API"""
        print(f"\n🎰 Placing {'DEMO' if demo else 'REAL'} test bet...")
        
        try:
            # Dice bet mutation
            mutation = """
            mutation DiceBet($amount: Float!, $target: Float!, $condition: CasinoGameDiceConditionEnum!, $currency: CurrencyEnum!) {
                diceBet(
                    amount: $amount
                    target: $target
                    condition: $condition
                    currency: $currency
                ) {
                    id
                    user {
                        id
                        balances {
                            available {
                                amount
                                currency
                            }
                        }
                    }
                    bet {
                        ... on CasinoGameDice {
                            result
                            target
                            condition
                            payout
                            payoutMultiplier
                        }
                    }
                    currency
                    amount
                    payout
                    createdAt
                }
            }
            """
            
            variables = {
                "amount": 0.01 if not demo else 1.0,  # Small amount for real, $1 for demo
                "target": 50.0,
                "condition": "UNDER",
                "currency": "USD"
            }
            
            bet_response = self.scraper.post(
                'https://stake.com/_api/graphql',
                json={"query": mutation, "variables": variables},
                timeout=30
            )
            
            print(f"   Bet Response: {bet_response.status_code}")
            
            if bet_response.status_code == 200:
                try:
                    bet_data = bet_response.json()
                    
                    if 'data' in bet_data and 'diceBet' in bet_data['data']:
                        bet = bet_data['data']['diceBet']['bet']
                        
                        print(f"   ✅ BET PLACED SUCCESSFULLY!")
                        print(f"   🎲 Result: {bet['result']}")
                        print(f"   🎯 Target: {bet['condition']} {bet['target']}")
                        print(f"   💰 Amount: ${variables['amount']}")
                        print(f"   💸 Payout: ${bet_data['data']['diceBet']['payout']}")
                        print(f"   📈 Multiplier: {bet['payoutMultiplier']}x")
                        
                        # Check if won
                        if bet['condition'] == 'UNDER' and bet['result'] < bet['target']:
                            print(f"   🎉 WON!")
                        elif bet['condition'] == 'OVER' and bet['result'] > bet['target']:
                            print(f"   🎉 WON!")
                        else:
                            print(f"   😞 LOST")
                        
                        # Print updated balance
                        user = bet_data['data']['diceBet']['user']
                        for balance in user['balances']:
                            if balance['available']['currency'] == variables['currency']:
                                print(f"   💳 New Balance: {balance['available']['amount']} {balance['available']['currency']}")
                        
                        return True
                    else:
                        print(f"   ❌ Bet failed: {bet_data}")
                        return False
                        
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON response: {bet_response.text[:200]}")
                    return False
            else:
                print(f"   ❌ Bet request failed: {bet_response.status_code}")
                print(f"   Response: {bet_response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Bet error: {e}")
            return False
    
    def run_full_test(self):
        """Run complete API test"""
        print("🚀 RUNNING COMPLETE STAKE API TEST")
        print("=" * 50)
        
        # Test API access
        if not self.test_api_access():
            print("\n💀 API ACCESS FAILED!")
            return False
        
        print("\n🎯 API ACCESS SUCCESSFUL!")
        
        # Test demo betting
        print("\n" + "=" * 30)
        demo_success = self.place_test_bet(demo=True)
        
        if demo_success:
            print("\n✅ DEMO BETTING WORKS!")
            
            # Ask about real betting
            print("\n" + "=" * 30)
            print("💰 REAL BETTING TEST")
            print("⚠️  WARNING: This will use real money!")
            
            choice = input("Do you want to test REAL betting? (y/N): ").lower()
            
            if choice == 'y':
                real_success = self.place_test_bet(demo=False)
                if real_success:
                    print("\n🎉 REAL BETTING WORKS!")
                    print("🚀 FULL API ACCESS ACHIEVED!")
                    return True
                else:
                    print("\n❌ Real betting failed")
                    return False
            else:
                print("\n✅ Demo test complete. Real betting skipped.")
                return True
        else:
            print("\n❌ Demo betting failed")
            return False

def main():
    """Main test function"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ NO API KEY FOUND!")
        print("Make sure STAKE_API_KEY is set in your .env file")
        return
    
    print(f"🔑 Using API Key: {api_key[:20]}...")
    
    stake_api = StakeAPIAccess(api_key)
    success = stake_api.run_full_test()
    
    if success:
        print("\n🎊 SUCCESS! STAKE API IS WORKING!")
        print("🔥 You can now use real/demo betting with your API key!")
    else:
        print("\n💀 FAILED TO ACCESS STAKE API")
        print("💡 Try using a VPN or different network")

if __name__ == "__main__":
    main()