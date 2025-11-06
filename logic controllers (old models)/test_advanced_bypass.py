#!/usr/bin/env python3
"""
ADVANCED CLOUDFLARE BYPASS TEST
==============================
Test both Camoufox and SeleniumBase bypass methods
"""

import os
import time
from dotenv import load_dotenv

def test_camoufox_bypass():
    """Test Camoufox bypass method"""
    print("🦊 Testing Camoufox...")
    
    try:
        from camoufox.sync_api import Camoufox
        from playwright.sync_api import TimeoutError
        
        with Camoufox(
            headless=False,
            humanize=True,
            window=(1280, 720)
        ) as browser:
            page = browser.new_page()
            
            print("   Visiting Stake.com...")
            page.goto("https://stake.com/", timeout=60000)
            
            # Wait for page load
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_load_state("networkidle")
            time.sleep(5)
            
            # Look for Turnstile
            try:
                turnstile = page.locator("iframe[src*='challenges.cloudflare.com']")
                if turnstile.is_visible(timeout=5000):
                    print("   🎯 Turnstile detected, clicking...")
                    # Click at turnstile position
                    page.mouse.click(210, 290)
                    time.sleep(10)
                else:
                    print("   ✅ No Turnstile challenge")
            except:
                print("   ✅ No Turnstile detected")
            
            # Check success
            if "stake" in page.url.lower():
                print("   ✅ Camoufox bypass successful!")
                return True
            else:
                print("   ❌ Camoufox bypass failed")
                return False
                
    except Exception as e:
        print(f"   ❌ Camoufox error: {e}")
        return False

def test_seleniumbase_bypass():
    """Test SeleniumBase bypass method"""
    print("🤖 Testing SeleniumBase...")
    
    try:
        from seleniumbase import Driver
        
        # Launch undetected Chrome
        driver = Driver(uc=True, headless=False)
        
        print("   Connecting to Stake...")
        driver.uc_open_with_reconnect("https://stake.com/", reconnect_time=4)
        
        time.sleep(3)
        
        print("   Handling Turnstile...")
        driver.uc_gui_click_captcha()
        
        time.sleep(5)
        
        # Check success
        current_url = driver.get_current_url()
        if "stake" in current_url.lower():
            print("   ✅ SeleniumBase bypass successful!")
            driver.quit()
            return True
        else:
            print(f"   ❌ SeleniumBase bypass failed. URL: {current_url}")
            driver.quit()
            return False
            
    except Exception as e:
        print(f"   ❌ SeleniumBase error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 TESTING ADVANCED CLOUDFLARE BYPASS METHODS")
    print("=" * 50)
    
    methods_tested = 0
    methods_succeeded = 0
    
    # Test Camoufox
    print("\n[1/2] Testing Camoufox method...")
    methods_tested += 1
    if test_camoufox_bypass():
        methods_succeeded += 1
    
    time.sleep(3)
    
    # Test SeleniumBase
    print("\n[2/2] Testing SeleniumBase method...")
    methods_tested += 1
    if test_seleniumbase_bypass():
        methods_succeeded += 1
    
    # Results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"Methods tested: {methods_tested}")
    print(f"Methods succeeded: {methods_succeeded}")
    print(f"Success rate: {(methods_succeeded/methods_tested)*100:.1f}%")
    
    if methods_succeeded > 0:
        print("\n✅ SUCCESS! Advanced bypass methods work!")
        print("🔥 Ready for Stake API integration!")
    else:
        print("\n❌ All methods failed")
        print("💡 Try different network or VPN")
    
    return methods_succeeded > 0

if __name__ == "__main__":
    main()