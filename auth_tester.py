"""
Robust Stake API Authentication System
Handles multiple authentication methods and session management
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class StakeAuthenticator:
    """Handle different Stake authentication methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.api_key = os.getenv('STAKE_API_KEY')
        self.base_url = "https://stake.com/_api/graphql"
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Origin': 'https://stake.com',
            'Referer': 'https://stake.com/'
        })
    
    def test_api_key_auth(self) -> bool:
        """Test API key authentication"""
        if not self.api_key:
            print("❌ No API key found in environment")
            return False
        
        # Method 1: Authorization header
        headers = {'Authorization': f'Bearer {self.api_key}'}
        if self._test_auth(headers, "Bearer token"):
            return True
        
        # Method 2: X-API-Key header
        headers = {'X-API-Key': self.api_key}
        if self._test_auth(headers, "X-API-Key"):
            return True
        
        # Method 3: API key in query
        query_with_key = """
        query TestAuth($apiKey: String!) {
            user(apiKey: $apiKey) {
                id
                name
            }
        }
        """
        if self._test_graphql_auth(query_with_key, {"apiKey": self.api_key}, "Query parameter"):
            return True
        
        return False
    
    def test_session_auth(self) -> bool:
        """Test session-based authentication (requires manual login)"""
        print("🔄 Testing session authentication...")
        
        # Simple query without auth
        query = """
        query {
            user {
                id
                name
            }
        }
        """
        
        response = self._make_request(query)
        if response and 'data' in response and response['data'].get('user'):
            print("✅ Session authentication successful")
            return True
        
        print("❌ Session authentication failed")
        return False
    
    def get_public_data(self) -> Dict[str, Any]:
        """Get public data that doesn't require authentication"""
        query = """
        query {
            info {
                currencies {
                    name
                    symbol
                }
            }
        }
        """
        
        response = self._make_request(query)
        if response:
            print("✅ Public API access working")
            return response
        
        print("❌ Public API access failed")
        return {}
    
    def _test_auth(self, headers: Dict[str, str], method_name: str) -> bool:
        """Test authentication with specific headers"""
        print(f"🔄 Testing {method_name} authentication...")
        
        # Update session headers
        self.session.headers.update(headers)
        
        query = """
        query {
            user {
                id
                name
                balance {
                    available
                    currency
                }
            }
        }
        """
        
        response = self._make_request(query)
        
        if response and 'data' in response and response['data'].get('user'):
            user = response['data']['user']
            print(f"✅ {method_name} authentication successful!")
            print(f"👤 User: {user.get('name', 'Unknown')}")
            if user.get('balance'):
                print(f"💰 Balance: {user['balance'].get('available', 0)} {user['balance'].get('currency', 'USD')}")
            return True
        
        print(f"❌ {method_name} authentication failed")
        return False
    
    def _test_graphql_auth(self, query: str, variables: Dict, method_name: str) -> bool:
        """Test GraphQL authentication with variables"""
        print(f"🔄 Testing {method_name} authentication...")
        
        payload = {
            "query": query,
            "variables": variables
        }
        
        try:
            response = self.session.post(self.base_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data'].get('user'):
                    print(f"✅ {method_name} authentication successful!")
                    return True
            
        except Exception as e:
            print(f"❌ {method_name} error: {e}")
        
        return False
    
    def _make_request(self, query: str, variables: Dict = None) -> Optional[Dict]:
        """Make GraphQL request"""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        try:
            response = self.session.post(self.base_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Request failed: {response.status_code}")
                if response.text:
                    print(f"Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
        
        return None

def comprehensive_auth_test():
    """Comprehensive authentication testing"""
    print("🔐 STAKE API AUTHENTICATION TEST")
    print("=" * 50)
    
    auth = StakeAuthenticator()
    
    print(f"🔑 API Key: {auth.api_key[:16] if auth.api_key else 'None'}...{auth.api_key[-8:] if auth.api_key else ''}")
    print()
    
    # Test different authentication methods
    auth_methods = [
        ("API Key Authentication", auth.test_api_key_auth),
        ("Session Authentication", auth.test_session_auth),
        ("Public Data Access", lambda: bool(auth.get_public_data()))
    ]
    
    successful_methods = []
    
    for method_name, test_func in auth_methods:
        print(f"\n📋 Testing: {method_name}")
        print("-" * 30)
        
        try:
            if test_func():
                successful_methods.append(method_name)
                print(f"✅ {method_name} - SUCCESS")
            else:
                print(f"❌ {method_name} - FAILED")
        except Exception as e:
            print(f"❌ {method_name} - ERROR: {e}")
    
    # Summary
    print(f"\n📊 AUTHENTICATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successful methods: {len(successful_methods)}")
    
    for method in successful_methods:
        print(f"   ✓ {method}")
    
    if successful_methods:
        print(f"\n🎯 RECOMMENDATION:")
        if "API Key Authentication" in successful_methods:
            print("   Use API Key for automated trading")
        elif "Session Authentication" in successful_methods:
            print("   Use session for manual trading monitoring")
        else:
            print("   Use public data for analysis only")
    else:
        print(f"\n⚠️  NO AUTHENTICATION METHODS WORKING")
        print("   Please check:")
        print("   1. API key is valid and active")
        print("   2. Account has API access enabled")
        print("   3. Network connection is stable")
    
    return successful_methods

def quick_test():
    """Quick test to verify basic connectivity"""
    print("⚡ QUICK CONNECTIVITY TEST")
    print("=" * 30)
    
    auth = StakeAuthenticator()
    
    # Test basic connectivity
    print("🌐 Testing basic connectivity...")
    public_data = auth.get_public_data()
    
    if public_data:
        print("✅ Connection to Stake API successful")
        
        # Show available currencies
        if 'data' in public_data and public_data['data'].get('info', {}).get('currencies'):
            currencies = public_data['data']['info']['currencies']
            print(f"💱 Available currencies: {len(currencies)}")
            for curr in currencies[:3]:  # Show first 3
                print(f"   • {curr['name']} ({curr['symbol']})")
            if len(currencies) > 3:
                print(f"   ... and {len(currencies) - 3} more")
        
        return True
    else:
        print("❌ Cannot connect to Stake API")
        return False

if __name__ == "__main__":
    print("🚀 STAKE API AUTHENTICATION TESTER")
    print("=" * 50)
    print("Choose test mode:")
    print("1. Quick connectivity test")
    print("2. Comprehensive authentication test")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        comprehensive_auth_test()
    else:
        print("👋 Goodbye!")