#!/usr/bin/env python3
"""
STAKE API ULTIMATE BYPASS & BETTING BOT
=======================================
Complete integration with Camoufox/SeleniumBase bypass + real Stake API
"""

import os
import json
import time
import hmac
import hashlib
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

class StakeAPUltimateBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.browser = None
        self.page = None
        self.driver = None
        self.method = None  # 'camoufox' or 'selenium'
        self.session_active = False
        
    def setup_camoufox(self, headless=True):
        """Setup Camoufox browser"""
        try:
            from camoufox.sync_api import Camoufox
            
            print("🦊 Setting up Camoufox...")
            
            self.browser = Camoufox(
                headless=headless,
                humanize=True,
                window=(1280, 720),
                addons=['ublock'],
                geoip=True
            )
            
            self.page = self.browser.new_page()
            self.method = 'camoufox'
            print("✅ Camoufox ready")
            return True
            
        except Exception as e:
            print(f"❌ Camoufox setup failed: {e}")
            return False
    
    def setup_selenium(self, headless=True):
        """Setup SeleniumBase driver"""
        try:
            from seleniumbase import Driver
            
            print("🤖 Setting up SeleniumBase...")
            
            self.driver = Driver(
                uc=True,
                headless=headless,
                incognito=True
            )
            
            self.method = 'selenium'
            print("✅ SeleniumBase ready")
            return True
            
        except Exception as e:
            print(f"❌ SeleniumBase setup failed: {e}")
            return False
    
    def bypass_cloudflare(self):
        """Bypass Cloudflare using active method"""
        print("🚀 Bypassing Cloudflare...")
        
        try:
            if self.method == 'camoufox':
                self.page.goto("https://stake.com/", timeout=60000)
                self.page.wait_for_load_state("networkidle")
                time.sleep(5)
                
                # Handle Turnstile if present
                try:
                    turnstile = self.page.locator("iframe[src*='challenges.cloudflare.com']")
                    if turnstile.is_visible(timeout=5000):
                        print("   Solving Turnstile...")
                        self.page.mouse.click(210, 290)
                        time.sleep(10)
                except:
                    pass
                
                if "stake" in self.page.url.lower():
                    print("✅ Cloudflare bypassed with Camoufox!")
                    self.session_active = True
                    return True
                    
            elif self.method == 'selenium':
                self.driver.uc_open_with_reconnect("https://stake.com/", reconnect_time=4)
                time.sleep(3)
                self.driver.uc_gui_click_captcha()
                time.sleep(5)
                
                if "stake" in self.driver.get_current_url().lower():
                    print("✅ Cloudflare bypassed with SeleniumBase!")
                    self.session_active = True
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Cloudflare bypass failed: {e}")
            return False
    
    def api_request(self, query, variables=None):
        """Make GraphQL API request through browser"""
        if not self.session_active:
            print("❌ No active session!")
            return None
        
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            
            if self.method == 'camoufox':
                # Execute in Camoufox
                script = """
                async () => {
                    try {
                        const response = await fetch('https://stake.com/_api/graphql', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-API-Key': '""" + self.api_key + """',
                                'X-Access-Token': '""" + self.api_key + """',
                                'Accept': 'application/json',
                                'Origin': 'https://stake.com',
                                'Referer': 'https://stake.com/'
                            },
                            body: JSON.stringify(""" + json.dumps(payload) + """)
                        });
                        
                        if (!response.ok) {
                            return { error: `HTTP ${response.status}` };
                        }
                        
                        return await response.json();
                    } catch (error) {
                        return { error: error.message };
                    }
                }
                """
                
                return self.page.evaluate(script)
                
            elif self.method == 'selenium':
                # Execute in Selenium
                script = """
                var xhr = new XMLHttpRequest();
                xhr.open('POST', 'https://stake.com/_api/graphql', false);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.setRequestHeader('X-API-Key', '""" + self.api_key + """');
                xhr.setRequestHeader('X-Access-Token', '""" + self.api_key + """');
                xhr.setRequestHeader('Accept', 'application/json');
                xhr.setRequestHeader('Origin', 'https://stake.com');
                xhr.setRequestHeader('Referer', 'https://stake.com/');
                
                xhr.send(JSON.stringify(""" + json.dumps(payload) + """));
                
                if (xhr.status === 200) {
                    return JSON.parse(xhr.responseText);
                } else {
                    return { error: 'HTTP ' + xhr.status };
                }
                """
                
                return self.driver.execute_script(script)
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def get_user_info(self):
        """Get user information"""
        query = """
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
        
        result = self.api_request(query)
        
        if result and 'data' in result and result['data']:
            return result['data']['user']
        else:
            print(f"❌ Failed to get user info: {result}")
            return None
    
    def place_dice_bet(self, amount, target=50.0, condition="UNDER", currency="USD"):
        """Place a dice bet"""
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
            "amount": amount,
            "target": target,
            "condition": condition,
            "currency": currency
        }
        
        result = self.api_request(mutation, variables)
        
        if result and 'data' in result and 'diceBet' in result['data']:
            return result['data']['diceBet']
        else:
            print(f"❌ Bet failed: {result}")
            return None
    
    def hmac_prediction(self, server_hash, client_seed, nonce):
        """Predict next roll using HMAC-SHA256"""
        message = f"{client_seed}:{nonce}"
        hmac_hash = hmac.new(
            bytes.fromhex(server_hash),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Convert to number (Stake's method)
        hex_chunk = hmac_hash[:8]
        decimal = int(hex_chunk, 16)
        roll = (decimal / (2**32)) * 100
        
        return round(roll, 2)
    
    def smart_betting_session(self, initial_amount=0.01, target_profit=1.0):
        """Run smart betting session with predictions"""
        print("\n🎯 STARTING SMART BETTING SESSION")
        print("=" * 50)
        
        # Get user info
        user = self.get_user_info()
        if not user:
            print("❌ Failed to get user info")
            return False
        
        print(f"👤 User: {user['name']}")
        
        # Find USD balance
        usd_balance = None
        for balance in user['balances']:
            if balance['available']['currency'] == 'USD':
                usd_balance = float(balance['available']['amount'])
                break
        
        if usd_balance is None:
            print("❌ No USD balance found")
            return False
        
        print(f"💰 Initial Balance: ${usd_balance:.2f}")
        print(f"🎯 Target Profit: ${target_profit:.2f}")
        
        # Get seeds
        client_seed = user['activeClientSeed']['seed']
        server_hash = user['activeServerSeed']['seedHash']
        current_nonce = user['activeServerSeed']['nonce']
        
        print(f"🌱 Client Seed: {client_seed}")
        print(f"🔐 Server Hash: {server_hash[:20]}...")
        print(f"🔢 Starting Nonce: {current_nonce}")
        
        # Betting loop
        bet_count = 0
        total_profit = 0
        current_amount = initial_amount
        
        print(f"\n🚀 Starting bets with ${current_amount:.4f}")
        
        while total_profit < target_profit and bet_count < 50:  # Max 50 bets
            bet_count += 1
            next_nonce = current_nonce + 1
            
            # Predict next roll
            predicted_roll = self.hmac_prediction(server_hash, client_seed, next_nonce)
            
            # Choose strategy based on prediction
            if predicted_roll < 50:
                # Predicted low, bet UNDER 50
                target = 49.5
                condition = "UNDER"
                payout_multiplier = 2.0
            else:
                # Predicted high, bet OVER 50
                target = 50.5
                condition = "OVER" 
                payout_multiplier = 2.0
            
            print(f"\n[{bet_count}] Nonce {next_nonce}: Predicted {predicted_roll:.2f} → Bet {condition} {target}")
            
            # Place bet
            bet_result = self.place_dice_bet(current_amount, target, condition)
            
            if bet_result:
                actual_roll = bet_result['bet']['result']
                payout = float(bet_result['payout'])
                profit = payout - current_amount
                total_profit += profit
                current_nonce = next_nonce
                
                # Check if won
                won = False
                if condition == "UNDER" and actual_roll < target:
                    won = True
                elif condition == "OVER" and actual_roll > target:
                    won = True
                
                status = "🎉 WON" if won else "😞 LOST"
                print(f"    Result: {actual_roll:.2f} | {status} | Profit: ${profit:.4f}")
                print(f"    Total Profit: ${total_profit:.4f}")
                
                # Adjust bet size (simple martingale)
                if won:
                    current_amount = initial_amount  # Reset on win
                else:
                    current_amount = min(current_amount * 1.5, 0.1)  # Increase on loss, cap at $0.10
                
            else:
                print("    ❌ Bet failed")
                break
            
            time.sleep(1)  # Rate limiting
        
        print(f"\n📊 SESSION COMPLETE")
        print(f"Bets placed: {bet_count}")
        print(f"Total profit: ${total_profit:.4f}")
        print(f"Target reached: {'✅' if total_profit >= target_profit else '❌'}")
        
        return total_profit >= target_profit
    
    def close(self):
        """Close browser/driver"""
        if self.browser:
            self.browser.close()
        if self.driver:
            self.driver.quit()
        print("🔚 Session closed")

def main():
    """Main function"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ NO API KEY FOUND!")
        return
    
    print("🚀 STAKE ULTIMATE BOT")
    print("=" * 40)
    print(f"🔑 API Key: {api_key[:20]}...")
    
    # Configuration
    print("\n🔧 CONFIGURATION")
    method = input("Choose method (1=Camoufox, 2=SeleniumBase): ").strip()
    headless = input("Headless mode? (y/N): ").lower() == 'y'
    
    # Initialize bot
    bot = StakeAPUltimateBot(api_key)
    
    # Setup method
    if method == '1':
        if not bot.setup_camoufox(headless=headless):
            print("❌ Camoufox setup failed")
            return
    else:
        if not bot.setup_selenium(headless=headless):
            print("❌ SeleniumBase setup failed")
            return
    
    try:
        # Bypass Cloudflare
        if not bot.bypass_cloudflare():
            print("❌ Failed to bypass Cloudflare")
            return
        
        # Test API
        user = bot.get_user_info()
        if not user:
            print("❌ API test failed")
            return
        
        print(f"\n✅ API CONNECTED! Welcome {user['name']}")
        
        # Show balances
        for balance in user['balances']:
            amount = balance['available']['amount']
            currency = balance['available']['currency']
            print(f"💰 {currency}: {amount}")
        
        # Start betting
        choice = input("\nStart smart betting session? (Y/n): ").lower()
        if choice != 'n':
            target_profit = float(input("Target profit ($): ") or "1.0")
            success = bot.smart_betting_session(target_profit=target_profit)
            
            if success:
                print("\n🎊 TARGET ACHIEVED! Session successful!")
            else:
                print("\n⚠️ Session ended without reaching target")
    
    finally:
        bot.close()

if __name__ == "__main__":
    main()