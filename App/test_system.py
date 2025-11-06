#!/usr/bin/env python3
"""
QUICK SYSTEM TEST - Validate all components work
===============================================
This script tests the core functionality before launching
"""

import os
import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test Oracle
        from massive_pretrain_oracle import OracleCore
        print("   ✅ OracleCore")
        
        # Test Bedrock Bot
        from supreme_bedrock_bot import SupremeBedrockBot, BettingDecision
        print("   ✅ SupremeBedrockBot")
        
        # Test other components
        import torch
        print("   ✅ PyTorch")
        
        import flask
        print("   ✅ Flask")
        
        import boto3
        print("   ✅ Boto3")
        
        from dotenv import load_dotenv
        print("   ✅ python-dotenv")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_oracle_creation():
    """Test Oracle model creation"""
    print("\n🧪 Testing Oracle creation...")
    
    try:
        from massive_pretrain_oracle import OracleCore
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        oracle = OracleCore().to(device)
        
        # Test forward pass
        test_input = torch.randn(1, 10).to(device)
        with torch.no_grad():
            output = oracle(test_input)
        
        print(f"   ✅ Oracle created on {device}")
        print(f"   ✅ Test forward pass: {output.item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Oracle test failed: {e}")
        traceback.print_exc()
        return False

def test_bedrock_bot():
    """Test Bedrock bot creation"""
    print("\n🧪 Testing Bedrock Bot...")
    
    try:
        from supreme_bedrock_bot import SupremeBedrockBot, MarketConditions
        
        # Create bot (will fail gracefully without AWS creds)
        bot = SupremeBedrockBot()
        
        # Test market conditions
        conditions = MarketConditions(
            recent_volatility=10.0,
            streak_length=2,
            pattern_strength=0.7,
            anomaly_detected=False,
            session_performance=0.6
        )
        
        print("   ✅ Bedrock Bot created")
        print("   ✅ MarketConditions tested")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Bedrock test failed: {e}")
        traceback.print_exc()
        return False

def test_environment():
    """Test environment setup"""
    print("\n🧪 Testing environment...")
    
    # Check .env file
    if os.path.exists('.env'):
        print("   ✅ .env file found")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for key variables (don't print values)
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY') 
        stake_key = os.getenv('STAKE_API_KEY')
        
        print(f"   {'✅' if aws_key else '❌'} AWS_ACCESS_KEY_ID")
        print(f"   {'✅' if aws_secret else '❌'} AWS_SECRET_ACCESS_KEY")
        print(f"   {'✅' if stake_key else '❌'} STAKE_API_KEY")
        
        return bool(aws_key and aws_secret)
        
    else:
        print("   ❌ .env file not found")
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("\n🧪 Testing Flask app...")
    
    try:
        from main import app
        print("   ✅ Flask app imported")
        
        # Test app configuration
        assert app.config['SECRET_KEY'] == 'supreme_oracle_main_2025'
        print("   ✅ App configuration")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Flask test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀" + "=" * 50 + "🚀")
    print("    SUPREME ORACLE SYSTEM TEST")
    print("🚀" + "=" * 50 + "🚀")
    
    tests = [
        ("Component Imports", test_imports),
        ("Oracle Model", test_oracle_creation),
        ("Bedrock Bot", test_bedrock_bot),
        ("Environment", test_environment),
        ("Flask App", test_flask_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * (len(test_name) + 4))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED - System ready for launch!")
        return True
    else:
        print("⚠️  Some tests failed - fix issues before launching")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)