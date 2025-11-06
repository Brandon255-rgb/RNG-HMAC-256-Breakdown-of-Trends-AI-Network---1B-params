#!/usr/bin/env python3
"""
ADVANCED CLOUDFLARE BYPASS FOR STAKE API
========================================
Using Camoufox and SeleniumBase - The most effective Cloudflare bypass solutions
"""

import os
import json
import time
import asyncio
from dotenv import load_dotenv
from datetime import datetime

class CamoufoxStakeBypass:
    def __init__(self, api_key):
        self.api_key = api_key
        self.browser = None
        self.page = None
        
    def setup_browser(self, headless=False):
        """Setup Camoufox browser with optimal settings"""
        from camoufox.sync_api import Camoufox
        
        print("🦊 Setting up Camoufox anti-detect browser...")
        
        self.browser = Camoufox(
            headless=headless,
            humanize=True,  # Realistic mouse movements and typing
            window=(1280, 720),  # Standard window size
            addons=['ublock'],  # Ad blocker for faster loading
            geoip=True,  # Realistic geolocation
            screen=1920,  # Screen resolution
            locale='en-US',  # Locale setting
            timezone='America/New_York',  # Timezone
            webgl_vendor='Intel Inc.',  # WebGL vendor
            webgl_renderer='Intel Iris OpenGL Engine'  # WebGL renderer
        )
        
        self.page = self.browser.new_page()
        
        # Set realistic headers
        self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        })
        
        print("✅ Camoufox browser configured")
        
    def bypass_cloudflare(self):
        """Bypass Cloudflare and establish session"""
        from playwright.sync_api import TimeoutError
        
        print("\n🚀 Bypassing Cloudflare with Camoufox...")
        
        try:
            # Visit Stake main page first
            print("   Visiting Stake.com...")
            self.page.goto("https://stake.com/", wait_until="domcontentloaded", timeout=60000)
            
            # Wait for page to fully load
            self.page.wait_for_load_state("domcontentloaded")
            self.page.wait_for_load_state("networkidle")
            
            print("   Waiting for Cloudflare challenge...")
            time.sleep(5)  # Let any Cloudflare challenge load
            
            # Check if Cloudflare challenge appears
            try:
                # Look for Cloudflare elements
                cf_challenge = self.page.locator("input[type='checkbox'][name='cf-turnstile-response']")
                if cf_challenge.is_visible(timeout=5000):
                    print("   🎯 Cloudflare Turnstile detected, solving...")
                    
                    # Get the challenge iframe
                    iframe = self.page.locator("iframe[src*='challenges.cloudflare.com']")
                    if iframe.is_visible():
                        # Click the checkbox
                        bbox = cf_challenge.bounding_box()
                        if bbox:
                            # Humanized click at center of checkbox
                            x = bbox['x'] + bbox['width'] / 2
                            y = bbox['y'] + bbox['height'] / 2
                            self.page.mouse.click(x, y)
                            print("   ✅ Clicked Turnstile checkbox")
                            
                            # Wait for challenge to complete
                            try:
                                self.page.wait_for_timeout(10000)
                                print("   ✅ Challenge completed!")
                            except TimeoutError:
                                print("   ⚠️  Challenge timeout, but continuing...")
                    
            except Exception as e:
                print(f"   ℹ️  No visible Cloudflare challenge: {e}")
            
            # Wait for final page load
            self.page.wait_for_timeout(3000)
            
            # Check if we successfully loaded Stake
            if "stake" in self.page.url.lower():
                print("   ✅ Successfully bypassed Cloudflare!")
                return True
            else:
                print("   ❌ Failed to reach Stake")
                return False
                
        except Exception as e:
            print(f"   ❌ Cloudflare bypass failed: {e}")
            return False
    
    def test_api_access(self):
        """Test GraphQL API access through browser"""
        print("\n🔍 Testing Stake API through browser...")
        
        try:
            # Execute GraphQL query in browser context
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
            
            # Execute fetch request in browser
            script = f"""
            async () => {{
                try {{
                    const response = await fetch('https://stake.com/_api/graphql', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer {self.api_key}',
                            'X-API-Key': '{self.api_key}',
                            'Origin': 'https://stake.com',
                            'Referer': 'https://stake.com/'
                        }},
                        body: JSON.stringify({{
                            query: `{query}`
                        }})
                    }});
                    
                    if (!response.ok) {{{{
                        return {{{{ error: `HTTP ${{{{response.status}}}}: ${{{{response.statusText}}}}` }}}};
                    }}}}
                    
                    const data = await response.json();
                    return data;
                }} catch (error) {{
                    return {{ error: error.message }};
                }}
            }}
            """
            
            result = self.page.evaluate(script)
            
            if 'error' in result:
                print(f"   ❌ API Error: {result['error']}")
                return False
            
            if 'data' in result and result['data'] and 'user' in result['data']:
                user = result['data']['user']
                print(f"   ✅ API SUCCESS! Connected as: {user['name']}")
                
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
                print(f"   ❌ Invalid API response: {result}")
                return False
                
        except Exception as e:
            print(f"   ❌ API test failed: {e}")
            return False
    
    def place_bet_through_browser(self, amount=0.01, demo=True):
        """Place bet through browser context"""
        print(f"\n🎰 Placing {'DEMO' if demo else 'REAL'} bet through browser...")
        
        try:
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
                "amount": amount if not demo else 1.0,
                "target": 50.0,
                "condition": "UNDER",
                "currency": "USD"
            }
            
            script = f"""
            async () => {{
                try {{
                    const response = await fetch('https://stake.com/_api/graphql', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer {self.api_key}',
                            'X-API-Key': '{self.api_key}',
                            'Origin': 'https://stake.com',
                            'Referer': 'https://stake.com/'
                        }},
                        body: JSON.stringify({{
                            query: `{mutation}`,
                            variables: {json.dumps(variables)}
                        }})
                    }});
                    
                    if (!response.ok) {{
                        return {{ error: `HTTP ${{response.status}}: ${{response.statusText}}` }};
                    }
                    
                    const data = await response.json();
                    return data;
                }} catch (error) {{
                    return {{ error: error.message }};
                }}
            }}
            """
            
            result = self.page.evaluate(script)
            
            if 'error' in result:
                print(f"   ❌ Bet Error: {result['error']}")
                return False
            
            if 'data' in result and 'diceBet' in result['data']:
                bet = result['data']['diceBet']['bet']
                bet_data = result['data']['diceBet']
                
                print(f"   ✅ BET PLACED SUCCESSFULLY!")
                print(f"   🎲 Result: {bet['result']}")
                print(f"   🎯 Target: {bet['condition']} {bet['target']}")
                print(f"   💰 Amount: ${variables['amount']}")
                print(f"   💸 Payout: ${bet_data['payout']}")
                print(f"   📈 Multiplier: {bet['payoutMultiplier']}x")
                
                # Check if won
                if bet['condition'] == 'UNDER' and bet['result'] < bet['target']:
                    print(f"   🎉 WON!")
                elif bet['condition'] == 'OVER' and bet['result'] > bet['target']:
                    print(f"   🎉 WON!")
                else:
                    print(f"   😞 LOST")
                
                return True
            else:
                print(f"   ❌ Invalid bet response: {result}")
                return False
                
        except Exception as e:
            print(f"   ❌ Bet failed: {e}")
            return False
    
    def close(self):
        """Close browser"""
        if self.browser:
            self.browser.close()
            print("🦊 Camoufox browser closed")

class SeleniumBaseStakeBypass:
    def __init__(self, api_key):
        self.api_key = api_key
        self.driver = None
        
    def setup_driver(self, headless=False):
        """Setup SeleniumBase undetected Chrome driver"""
        from seleniumbase import Driver
        
        print("🤖 Setting up SeleniumBase undetected driver...")
        
        # Launch in undetected-chromedriver mode
        self.driver = Driver(
            uc=True,  # Undetected Chrome mode
            headless=headless,
            incognito=True,  # Private browsing
            no_sandbox=True,  # Bypass sandbox
            disable_dev_shm=True,  # Use /dev/shm
            disable_gpu=False,  # Enable GPU
            disable_blink_features="AutomationControlled",  # Disable automation detection
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        
        print("✅ SeleniumBase driver configured")
        
    def bypass_cloudflare(self):
        """Bypass Cloudflare using SeleniumBase"""
        from seleniumbase.common.exceptions import TextNotVisibleException
        
        print("\n🚀 Bypassing Cloudflare with SeleniumBase...")
        
        try:
            # Visit the target page with reconnect capability
            url = "https://stake.com/"
            print("   Connecting to Stake with UC mode...")
            
            # Use uc_open_with_reconnect for better Cloudflare handling
            self.driver.uc_open_with_reconnect(url, reconnect_time=4)
            
            print("   Waiting for page load...")
            time.sleep(3)
            
            # Handle Turnstile CAPTCHA if it appears
            print("   Checking for Cloudflare challenge...")
            self.driver.uc_gui_click_captcha()  # Automatically handles Turnstile
            
            # Wait a bit more
            time.sleep(5)
            
            # Check if we successfully loaded Stake
            current_url = self.driver.get_current_url()
            if "stake" in current_url.lower():
                print("   ✅ Successfully bypassed Cloudflare!")
                return True
            else:
                print(f"   ❌ Failed to reach Stake. Current URL: {current_url}")
                return False
                
        except Exception as e:
            print(f"   ❌ Cloudflare bypass failed: {e}")
            return False
    
    def test_api_access(self):
        """Test API access through Selenium"""
        print("\n🔍 Testing Stake API through Selenium...")
        
        try:
            # Execute GraphQL query via JavaScript
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
            
            script = f"""
            var xhr = new XMLHttpRequest();
            xhr.open('POST', 'https://stake.com/_api/graphql', false);
            xhr.setRequestHeader('Content-Type', 'application/json');
            xhr.setRequestHeader('Authorization', 'Bearer {self.api_key}');
            xhr.setRequestHeader('X-API-Key', '{self.api_key}');
            xhr.setRequestHeader('Origin', 'https://stake.com');
            xhr.setRequestHeader('Referer', 'https://stake.com/');
            
            xhr.send(JSON.stringify({{
                query: `{query}`
            }}));
            
            if (xhr.status === 200) {{
                return JSON.parse(xhr.responseText);
            }} else {{
                return {{ error: 'HTTP ' + xhr.status + ': ' + xhr.statusText }};
            }}
            """
            
            result = self.driver.execute_script(script)
            
            if isinstance(result, dict) and 'error' in result:
                print(f"   ❌ API Error: {result['error']}")
                return False
            
            if isinstance(result, dict) and 'data' in result and result['data'] and 'user' in result['data']:
                user = result['data']['user']
                print(f"   ✅ API SUCCESS! Connected as: {user['name']}")
                
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
                print(f"   ❌ Invalid API response: {result}")
                return False
                
        except Exception as e:
            print(f"   ❌ API test failed: {e}")
            return False
    
    def close(self):
        """Close driver"""
        if self.driver:
            self.driver.quit()
            print("🤖 SeleniumBase driver closed")

class AdvancedStakeBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.camoufox = None
        self.selenium = None
        
    def test_camoufox_method(self, headless=False):
        """Test Camoufox bypass method"""
        print("\n" + "="*60)
        print("🦊 TESTING CAMOUFOX METHOD")
        print("="*60)
        
        try:
            self.camoufox = CamoufoxStakeBypass(self.api_key)
            self.camoufox.setup_browser(headless=headless)
            
            # Bypass Cloudflare
            if not self.camoufox.bypass_cloudflare():
                print("❌ Camoufox failed to bypass Cloudflare")
                return False
            
            # Test API access
            if not self.camoufox.test_api_access():
                print("❌ Camoufox failed API test")
                return False
            
            # Test betting
            if not self.camoufox.place_bet_through_browser(demo=True):
                print("❌ Camoufox failed betting test")
                return False
            
            print("✅ CAMOUFOX METHOD SUCCESSFUL!")
            return True
            
        except Exception as e:
            print(f"❌ Camoufox method failed: {e}")
            return False
        finally:
            if self.camoufox:
                self.camoufox.close()
    
    def test_selenium_method(self, headless=False):
        """Test SeleniumBase bypass method"""
        print("\n" + "="*60)
        print("🤖 TESTING SELENIUMBASE METHOD")
        print("="*60)
        
        try:
            self.selenium = SeleniumBaseStakeBypass(self.api_key)
            self.selenium.setup_driver(headless=headless)
            
            # Bypass Cloudflare
            if not self.selenium.bypass_cloudflare():
                print("❌ SeleniumBase failed to bypass Cloudflare")
                return False
            
            # Test API access
            if not self.selenium.test_api_access():
                print("❌ SeleniumBase failed API test")
                return False
            
            print("✅ SELENIUMBASE METHOD SUCCESSFUL!")
            return True
            
        except Exception as e:
            print(f"❌ SeleniumBase method failed: {e}")
            return False
        finally:
            if self.selenium:
                self.selenium.close()
    
    def run_complete_test(self, headless=False):
        """Run complete test of both methods"""
        print("🚀 ADVANCED CLOUDFLARE BYPASS TEST")
        print("="*80)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔑 API Key: {self.api_key[:20]}...")
        print(f"👁️  Headless Mode: {headless}")
        print("="*80)
        
        methods_tested = 0
        methods_succeeded = 0
        
        # Test Camoufox
        methods_tested += 1
        if self.test_camoufox_method(headless=headless):
            methods_succeeded += 1
        
        time.sleep(3)  # Brief pause between methods
        
        # Test SeleniumBase
        methods_tested += 1
        if self.test_selenium_method(headless=headless):
            methods_succeeded += 1
        
        # Final results
        print("\n" + "="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        print(f"🔍 Methods Tested: {methods_tested}")
        print(f"✅ Methods Succeeded: {methods_succeeded}")
        print(f"📈 Success Rate: {(methods_succeeded/methods_tested)*100:.1f}%")
        
        if methods_succeeded > 0:
            print("\n🎊 SUCCESS! Advanced bypass methods work!")
            print("🔥 You can now use real/demo betting with Stake API!")
            print("💡 Recommended: Use Camoufox for best stealth, SeleniumBase for reliability")
        else:
            print("\n💀 All methods failed. Try:")
            print("   1. Different network/VPN")
            print("   2. Different time of day")
            print("   3. Check if API key is valid")
        
        return methods_succeeded > 0

def main():
    """Main test function"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ NO API KEY FOUND!")
        print("Make sure STAKE_API_KEY is set in your .env file")
        return
    
    bot = AdvancedStakeBot(api_key)
    
    # Ask user for headless mode
    print("🔧 CONFIGURATION")
    print("="*40)
    headless_choice = input("Run in headless mode? (y/N): ").lower()
    headless = headless_choice == 'y'
    
    print("\n⚠️  This will test both Camoufox and SeleniumBase methods")
    print("💻 Browser windows may open during testing")
    choice = input("Continue? (Y/n): ").lower()
    
    if choice != 'n':
        success = bot.run_complete_test(headless=headless)
        
        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Integrate working method into main bot")
            print("2. Set up automated betting")
            print("3. Configure risk management")
            print("4. Deploy production system")
        
        return success
    else:
        print("Test cancelled by user")
        return False

if __name__ == "__main__":
    main()