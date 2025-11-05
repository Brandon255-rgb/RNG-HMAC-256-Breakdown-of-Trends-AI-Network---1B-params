#!/usr/bin/env python3
"""
CLOUDFLARE BYPASS FOR STAKE API
==============================
Advanced techniques to bypass Cloudflare protection and access Stake API
"""

import requests
import time
import json
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
import certifi

class CloudflareBypass:
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.setup_session()
        
    def setup_session(self):
        """Setup session with Cloudflare bypass techniques"""
        
        # Real browser headers that Cloudflare expects
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Origin': 'https://stake.com',
            'Referer': 'https://stake.com/',
            'X-Requested-With': 'XMLHttpRequest',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # SSL/TLS settings
        self.session.verify = certifi.where()
        
    def get_csrf_token(self):
        """Get CSRF token from main page"""
        try:
            print("🔒 Getting CSRF token...")
            response = self.session.get('https://stake.com/', timeout=15)
            
            if response.status_code == 200:
                # Extract CSRF token from HTML or cookies
                for cookie in response.cookies:
                    if 'csrf' in cookie.name.lower():
                        print(f"✅ CSRF token found: {cookie.name}")
                        return cookie.value
                
                # Look for token in HTML
                html = response.text
                if 'csrf-token' in html:
                    import re
                    match = re.search(r'csrf-token["\s]*content="([^"]+)"', html)
                    if match:
                        token = match.group(1)
                        print(f"✅ CSRF token extracted: {token[:20]}...")
                        return token
            
            print("⚠️ No CSRF token found, proceeding without...")
            return None
            
        except Exception as e:
            print(f"❌ Error getting CSRF token: {e}")
            return None
    
    def test_cloudflare_bypass(self):
        """Test various Cloudflare bypass techniques"""
        print("🔥 TESTING CLOUDFLARE BYPASS TECHNIQUES")
        print("=" * 50)
        
        # Method 1: Direct API with proper headers
        print("\n1. Testing direct API access...")
        success = self.test_direct_api()
        if success:
            return True
        
        # Method 2: Browser simulation with cookies
        print("\n2. Testing browser simulation...")
        success = self.test_browser_simulation()
        if success:
            return True
        
        # Method 3: Session establishment
        print("\n3. Testing session establishment...")
        success = self.test_session_establishment()
        if success:
            return True
        
        # Method 4: Alternative endpoints
        print("\n4. Testing alternative endpoints...")
        success = self.test_alternative_endpoints()
        if success:
            return True
        
        print("\n❌ All bypass methods failed!")
        return False
    
    def test_direct_api(self):
        """Test direct API access with proper authentication"""
        try:
            # Add API key to headers
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Try GraphQL endpoint
            query = {
                "query": "query { user { id name } }"
            }
            
            response = self.session.post(
                'https://stake.com/_api/graphql',
                json=query,
                headers=headers,
                timeout=15
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    print("   ✅ Direct API access successful!")
                    return True
            
            print(f"   Response: {response.text[:200]}...")
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_browser_simulation(self):
        """Simulate real browser behavior"""
        try:
            print("   Simulating browser visit...")
            
            # Step 1: Visit main page
            response = self.session.get('https://stake.com/', timeout=15)
            print(f"   Main page: {response.status_code}")
            
            if response.status_code != 200:
                return False
            
            # Wait like a real user
            time.sleep(random.uniform(2, 4))
            
            # Step 2: Get CSRF token
            csrf_token = self.get_csrf_token()
            
            # Step 3: Add auth headers
            if csrf_token:
                self.session.headers['X-CSRF-Token'] = csrf_token
            
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Step 4: Try API call
            query = {"query": "query { user { id name } }"}
            
            response = self.session.post(
                'https://stake.com/_api/graphql',
                json=query,
                timeout=15
            )
            
            print(f"   API call: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    print("   ✅ Browser simulation successful!")
                    return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_session_establishment(self):
        """Establish proper session with cookies"""
        try:
            print("   Establishing session...")
            
            # Clear existing cookies
            self.session.cookies.clear()
            
            # Visit different pages to build session
            pages = [
                'https://stake.com/',
                'https://stake.com/casino',
                'https://stake.com/casino/games/dice'
            ]
            
            for page in pages:
                response = self.session.get(page, timeout=10)
                print(f"   {page}: {response.status_code}")
                time.sleep(random.uniform(1, 2))
                
                if response.status_code != 200:
                    continue
            
            # Now try API with session
            self.session.headers['Authorization'] = f'Bearer {self.api_key}'
            
            query = {"query": "query { user { id } }"}
            response = self.session.post(
                'https://stake.com/_api/graphql',
                json=query,
                timeout=15
            )
            
            print(f"   Session API: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    print("   ✅ Session establishment successful!")
                    return True
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def test_alternative_endpoints(self):
        """Test alternative API endpoints"""
        try:
            print("   Testing alternative endpoints...")
            
            # Alternative endpoints to try
            endpoints = [
                'https://stake.com/api/v2/user',
                'https://stake.com/_api/user',
                'https://api.stake.com/graphql',
                'https://stake.com/graphql'
            ]
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'X-API-Key': self.api_key
            }
            
            for endpoint in endpoints:
                try:
                    if 'graphql' in endpoint:
                        query = {"query": "query { user { id } }"}
                        response = self.session.post(endpoint, json=query, headers=headers, timeout=10)
                    else:
                        response = self.session.get(endpoint, headers=headers, timeout=10)
                    
                    print(f"   {endpoint}: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data and ('data' in data or 'user' in data):
                            print(f"   ✅ Alternative endpoint successful: {endpoint}")
                            return True
                
                except Exception as e:
                    print(f"   {endpoint}: Error - {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def get_working_session(self):
        """Get a working session that bypasses Cloudflare"""
        if self.test_cloudflare_bypass():
            print("\n✅ Cloudflare bypass successful!")
            print("🔧 Session configured and ready for API calls")
            return self.session
        else:
            print("\n❌ Unable to bypass Cloudflare")
            print("💡 Try using a different API key or VPN")
            return None

def test_bypass():
    """Test the Cloudflare bypass"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ No API key found!")
        return False
    
    print(f"🔑 Testing with API key: {api_key[:20]}...")
    
    bypass = CloudflareBypass(api_key)
    working_session = bypass.get_working_session()
    
    if working_session:
        print("\n🎉 SUCCESS! Cloudflare bypass working!")
        return True
    else:
        print("\n💀 FAILED! Cloudflare is blocking all attempts")
        return False

if __name__ == "__main__":
    test_bypass()