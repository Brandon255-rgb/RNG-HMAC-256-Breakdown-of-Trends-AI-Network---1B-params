#!/usr/bin/env python3
"""
ADVANCED STAKE API BYPASS
========================
Multiple advanced techniques to bypass Cloudflare and use Stake API with real betting
"""

import requests
import json
import time
import random
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import cloudscraper
from requests_html import HTMLSession
import os
from dotenv import load_dotenv

class AdvancedStakeAPIBypass:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = None
        self.cookies = {}
        self.headers = {}
        
    def method_1_cloudscraper(self):
        """Method 1: CloudScraper - designed specifically for Cloudflare"""
        print("🔥 METHOD 1: CloudScraper Bypass")
        try:
            # Install cloudscraper if not installed
            try:
                import cloudscraper
            except ImportError:
                print("Installing cloudscraper...")
                os.system("pip install cloudscraper")
                import cloudscraper
            
            scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'desktop': True
                }
            )
            
            # Add API key
            scraper.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            })
            
            # Test GraphQL query
            query = {"query": "query { user { id name balances { available { amount currency } } } }"}
            
            response = scraper.post('https://stake.com/_api/graphql', json=query, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data'] and 'user' in data['data']:
                    print("   ✅ CloudScraper SUCCESS!")
                    self.session = scraper
                    return True
            
            print(f"   Response: {response.text[:200]}...")
            return False
            
        except Exception as e:
            print(f"   ❌ CloudScraper Error: {e}")
            return False
    
    def method_2_selenium_undetected(self):
        """Method 2: Undetected Chrome with Selenium"""
        print("🔥 METHOD 2: Undetected Chrome Selenium")
        try:
            # Install undetected-chromedriver if not installed
            try:
                import undetected_chromedriver as uc
            except ImportError:
                print("Installing undetected-chromedriver...")
                os.system("pip install undetected-chromedriver")
                import undetected_chromedriver as uc
            
            options = uc.ChromeOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            driver = uc.Chrome(options=options)
            
            print("   Opening Stake.com...")
            driver.get('https://stake.com/')
            
            # Wait for page load
            time.sleep(5)
            
            # Check if we bypassed Cloudflare
            if "Just a moment" not in driver.page_source:
                print("   ✅ Bypassed Cloudflare!")
                
                # Get cookies
                cookies = driver.get_cookies()
                cookie_dict = {cookie['name']: cookie['value'] for cookie in cookies}
                
                # Create requests session with cookies
                session = requests.Session()
                session.cookies.update(cookie_dict)
                session.headers.update({
                    'User-Agent': driver.execute_script("return navigator.userAgent;"),
                    'Authorization': f'Bearer {self.api_key}',
                    'X-API-Key': self.api_key,
                    'Content-Type': 'application/json',
                    'Referer': 'https://stake.com/'
                })
                
                driver.quit()
                
                # Test API call
                query = {"query": "query { user { id name } }"}
                response = session.post('https://stake.com/_api/graphql', json=query, timeout=30)
                
                print(f"   API Test: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and data['data']:
                        print("   ✅ Selenium SUCCESS!")
                        self.session = session
                        return True
                
                return False
            else:
                print("   ❌ Still blocked by Cloudflare")
                driver.quit()
                return False
                
        except Exception as e:
            print(f"   ❌ Selenium Error: {e}")
            try:
                driver.quit()
            except:
                pass
            return False
    
    def method_3_requests_html(self):
        """Method 3: Requests-HTML with JavaScript rendering"""
        print("🔥 METHOD 3: Requests-HTML with JS")
        try:
            # Install requests-html if not installed
            try:
                from requests_html import HTMLSession
            except ImportError:
                print("Installing requests-html...")
                os.system("pip install requests-html")
                from requests_html import HTMLSession
            
            session = HTMLSession()
            
            # Set headers
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            })
            
            print("   Getting main page...")
            response = session.get('https://stake.com/')
            
            # Render JavaScript
            response.html.render(timeout=20, wait=5)
            
            if response.status_code == 200 and "Just a moment" not in response.html.html:
                print("   ✅ Bypassed Cloudflare with JS rendering!")
                
                # Add API headers
                session.headers.update({
                    'Authorization': f'Bearer {self.api_key}',
                    'X-API-Key': self.api_key,
                    'Content-Type': 'application/json'
                })
                
                # Test API
                query = {"query": "query { user { id } }"}
                api_response = session.post('https://stake.com/_api/graphql', json=query, timeout=30)
                
                print(f"   API Test: {api_response.status_code}")
                
                if api_response.status_code == 200:
                    data = api_response.json()
                    if 'data' in data:
                        print("   ✅ Requests-HTML SUCCESS!")
                        self.session = session
                        return True
                
                return False
            else:
                print("   ❌ Still blocked by Cloudflare")
                return False
                
        except Exception as e:
            print(f"   ❌ Requests-HTML Error: {e}")
            return False
    
    def method_4_proxy_rotation(self):
        """Method 4: Proxy rotation to bypass IP blocking"""
        print("🔥 METHOD 4: Proxy Rotation")
        try:
            # Free proxy list (you can add paid proxies here)
            proxies_list = [
                {'http': 'http://proxy1.example.com:8080', 'https': 'https://proxy1.example.com:8080'},
                {'http': 'http://proxy2.example.com:8080', 'https': 'https://proxy2.example.com:8080'},
                # Add more proxies here
            ]
            
            for i, proxy in enumerate(proxies_list):
                try:
                    print(f"   Testing proxy {i+1}...")
                    
                    session = requests.Session()
                    session.proxies.update(proxy)
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json'
                    })
                    
                    # Test connection
                    response = session.get('https://stake.com/', timeout=10)
                    
                    if response.status_code == 200 and "Just a moment" not in response.text:
                        print(f"   ✅ Proxy {i+1} works!")
                        
                        # Test API
                        query = {"query": "query { user { id } }"}
                        api_response = session.post('https://stake.com/_api/graphql', json=query, timeout=15)
                        
                        if api_response.status_code == 200:
                            print("   ✅ Proxy method SUCCESS!")
                            self.session = session
                            return True
                    
                except Exception as e:
                    print(f"   Proxy {i+1} failed: {e}")
                    continue
            
            print("   ❌ No working proxies found")
            return False
            
        except Exception as e:
            print(f"   ❌ Proxy Error: {e}")
            return False
    
    def method_5_direct_api_endpoints(self):
        """Method 5: Try different API endpoints that might not be protected"""
        print("🔥 METHOD 5: Alternative API Endpoints")
        try:
            # Different possible endpoints
            endpoints = [
                'https://stake.com/_api/graphql',
                'https://api.stake.com/v1/graphql',
                'https://backend.stake.com/graphql',
                'https://stake.com/api/graphql',
                'https://www.stake.com/_api/graphql',
                'https://mobile-api.stake.com/graphql'
            ]
            
            session = requests.Session()
            
            # Try different user agents
            user_agents = [
                'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15',  # Mobile
                'Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/88.0',  # Android
                'StakeApp/1.0 (iOS)',  # App user agent
                'StakeApp/1.0 (Android)'
            ]
            
            for endpoint in endpoints:
                for ua in user_agents:
                    try:
                        session.headers.update({
                            'User-Agent': ua,
                            'Authorization': f'Bearer {self.api_key}',
                            'X-API-Key': self.api_key,
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        })
                        
                        query = {"query": "query { user { id } }"}
                        response = session.post(endpoint, json=query, timeout=10)
                        
                        print(f"   {endpoint} with {ua[:20]}...: {response.status_code}")
                        
                        if response.status_code == 200:
                            data = response.json()
                            if 'data' in data and data['data']:
                                print("   ✅ Alternative endpoint SUCCESS!")
                                self.session = session
                                return True
                    
                    except Exception as e:
                        continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ Alternative endpoints Error: {e}")
            return False
    
    def get_working_api_session(self):
        """Try all methods to get a working API session"""
        print("🚀 BYPASSING CLOUDFLARE FOR REAL STAKE API ACCESS")
        print("=" * 60)
        
        methods = [
            self.method_1_cloudscraper,
            self.method_5_direct_api_endpoints,
            self.method_2_selenium_undetected,
            self.method_3_requests_html,
            self.method_4_proxy_rotation
        ]
        
        for i, method in enumerate(methods, 1):
            print(f"\n[{i}/5] Trying {method.__name__.replace('method_', '').replace('_', ' ').title()}...")
            
            if method():
                print("\n🎉 SUCCESS! API ACCESS ACHIEVED!")
                return self.session
            
            print(f"   Method {i} failed, trying next...")
            time.sleep(2)
        
        print("\n💀 ALL METHODS FAILED!")
        print("💡 SOLUTIONS:")
        print("   1. Use a VPN to change your IP")
        print("   2. Try from a different network")
        print("   3. Wait 24 hours for IP cooldown")
        print("   4. Contact Stake support about API access")
        
        return None
    
    def test_real_betting(self):
        """Test real betting functionality once we have access"""
        if not self.session:
            print("❌ No working session available")
            return False
        
        try:
            print("\n🎰 TESTING REAL BETTING FUNCTIONALITY")
            print("=" * 40)
            
            # Get user info
            user_query = {"query": "query { user { id name balances { available { amount currency } } } }"}
            response = self.session.post('https://stake.com/_api/graphql', json=user_query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                user = data['data']['user']
                print(f"✅ User: {user['name']}")
                
                for balance in user['balances']:
                    amount = balance['available']['amount']
                    currency = balance['available']['currency']
                    print(f"💰 Balance: {amount} {currency}")
                
                # Test demo bet
                print("\n🎮 Testing Demo Dice Bet...")
                bet_mutation = """
                mutation DiceBet($amount: Float!, $target: Float!, $condition: CasinoGameDiceConditionEnum!, $currency: CurrencyEnum!) {
                    diceBet(
                        amount: $amount
                        target: $target
                        condition: $condition
                        currency: $currency
                    ) {
                        id
                        amount
                        payout
                        bet {
                            ... on CasinoGameDice {
                                result
                                target
                                condition
                            }
                        }
                    }
                }
                """
                
                variables = {
                    "amount": 0.01,  # Small test bet
                    "target": 50.0,
                    "condition": "UNDER",
                    "currency": "USD"
                }
                
                bet_response = self.session.post(
                    'https://stake.com/_api/graphql',
                    json={"query": bet_mutation, "variables": variables},
                    timeout=30
                )
                
                print(f"Bet Response: {bet_response.status_code}")
                
                if bet_response.status_code == 200:
                    bet_data = bet_response.json()
                    print(f"✅ BET PLACED SUCCESSFULLY!")
                    print(f"   Bet ID: {bet_data['data']['diceBet']['id']}")
                    print(f"   Result: {bet_data['data']['diceBet']['bet']['result']}")
                    print(f"   Payout: {bet_data['data']['diceBet']['payout']}")
                    return True
                else:
                    print(f"❌ Bet failed: {bet_response.text}")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Betting test error: {e}")
            return False

def main():
    """Main function to test all bypass methods"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ NO API KEY FOUND IN .env FILE!")
        return
    
    print(f"🔑 Using API Key: {api_key[:20]}...")
    
    bypass = AdvancedStakeAPIBypass(api_key)
    session = bypass.get_working_api_session()
    
    if session:
        print("\n🎯 TESTING REAL BETTING...")
        bypass.test_real_betting()
    else:
        print("\n💀 FAILED TO BYPASS CLOUDFLARE")

if __name__ == "__main__":
    main()