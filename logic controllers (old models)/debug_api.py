#!/usr/bin/env python3
"""
DEBUG STAKE API RESPONSE
========================
Debug what we're actually getting from the API
"""

import cloudscraper
import json
import os
from dotenv import load_dotenv

def debug_api_response():
    """Debug the API response"""
    load_dotenv()
    api_key = os.getenv('STAKE_API_KEY')
    
    if not api_key:
        print("❌ No API key found")
        return
    
    print("🔍 DEBUGGING STAKE API RESPONSE")
    print(f"🔑 API Key: {api_key[:20]}...")
    
    # Setup scraper
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    
    scraper.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'identity',  # No compression
        'Connection': 'keep-alive',
        'Origin': 'https://stake.com',
        'Referer': 'https://stake.com/',
        'Authorization': f'Bearer {api_key}',
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    })
    
    # Visit main page first
    print("\n1. Loading main page...")
    main_response = scraper.get('https://stake.com/', timeout=30)
    print(f"   Status: {main_response.status_code}")
    
    # Try API
    print("\n2. Testing API...")
    query = {
        "query": """
        query {
            user {
                id
                name
            }
        }
        """
    }
    
    api_response = scraper.post('https://stake.com/_api/graphql', json=query, timeout=30)
    print(f"   Status: {api_response.status_code}")
    print(f"   Headers: {dict(api_response.headers)}")
    
    # Check content type
    content_type = api_response.headers.get('content-type', 'unknown')
    print(f"   Content-Type: {content_type}")
    
    # Raw content
    raw_content = api_response.content
    print(f"   Raw Content Length: {len(raw_content)} bytes")
    print(f"   First 200 chars: {raw_content[:200]}")
    
    # Try to decode as text
    try:
        text_content = api_response.text
        print(f"   Text Content: {text_content[:500]}")
    except Exception as e:
        print(f"   ❌ Text decode error: {e}")
    
    # Try to parse as JSON
    try:
        json_content = api_response.json()
        print(f"   JSON Content: {json_content}")
    except Exception as e:
        print(f"   ❌ JSON parse error: {e}")
    
    # Check if it's empty
    if not raw_content:
        print("   ⚠️  Response is completely empty!")
    
    # Check if it's HTML (error page)
    if b'<html' in raw_content[:100].lower():
        print("   ⚠️  Response appears to be HTML (error page)")
        print(f"   HTML snippet: {raw_content[:500].decode('utf-8', errors='ignore')}")

if __name__ == "__main__":
    debug_api_response()