#!/usr/bin/env python3
"""
STAKE SESSION LOGIN TEST
=======================
Test if we need to login first before using API
"""

import cloudscraper
import json
import os
import time
from dotenv import load_dotenv

def test_stake_login():
    """Test Stake login and session"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    print("🔐 TESTING STAKE SESSION LOGIN")
    print(f"🔑 API Key: {api_key[:20]}...")
    
    # Create session
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    
    # First, visit the main page to get session
    print("\n1. Visiting main page...")
    main_response = scraper.get('https://stake.com/', timeout=30)
    print(f"   Status: {main_response.status_code}")
    
    # Get cookies
    print(f"   Cookies: {len(scraper.cookies)} cookies set")
    for cookie in scraper.cookies:
        print(f"      {cookie.name}: {cookie.value[:20]}...")
    
    # Try without auth headers first
    print("\n2. Testing API without auth...")
    query = {"query": "query { user { id name } }"}
    
    response = scraper.post('https://stake.com/_api/graphql', json=query, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   Response: {data}")
        except:
            print(f"   Raw: {response.text[:200]}")
    
    # Now try with API key in different formats
    print("\n3. Testing with API key in Authorization header...")
    scraper.headers.update({
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'https://stake.com',
        'Referer': 'https://stake.com/'
    })
    
    response = scraper.post('https://stake.com/_api/graphql', json=query, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   Response: {data}")
        except:
            print(f"   Raw: {response.text[:200]}")
    
    # Try with X-API-Key header
    print("\n4. Testing with X-API-Key header...")
    scraper.headers.update({
        'X-API-Key': api_key,
        'X-Access-Token': api_key
    })
    
    response = scraper.post('https://stake.com/_api/graphql', json=query, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   Response: {data}")
        except:
            print(f"   Raw: {response.text[:200]}")
    
    # Try with different query that might not need auth
    print("\n5. Testing public query...")
    public_query = {
        "query": """
        query {
            info {
                currencies {
                    name
                    type
                }
            }
        }
        """
    }
    
    response = scraper.post('https://stake.com/_api/graphql', json=public_query, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)[:500]}...")
        except:
            print(f"   Raw: {response.text[:200]}")
    
    # Check if we need to login through web interface first
    print("\n6. Looking for login endpoints...")
    
    # Try to find login forms
    if 'login' in main_response.text.lower():
        print("   Login form detected on main page")
    
    # Check common login endpoints
    login_endpoints = [
        '/auth/login',
        '/login',
        '/api/auth/login',
        '/_api/auth/login'
    ]
    
    for endpoint in login_endpoints:
        try:
            resp = scraper.get(f'https://stake.com{endpoint}', timeout=10)
            if resp.status_code == 200:
                print(f"   Found endpoint: {endpoint}")
        except:
            pass

if __name__ == "__main__":
    test_stake_login()