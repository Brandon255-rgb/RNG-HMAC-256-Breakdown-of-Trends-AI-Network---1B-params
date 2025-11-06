#!/usr/bin/env python3
"""
Test MEGA Enhanced Predictor Initialization
==========================================
"""

import os
import sys
import asyncio
import logging

# Add logic controllers to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'logic controllers'))

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_mega_predictor():
    """Test the mega enhanced predictor"""
    try:
        print("Testing MEGA Enhanced Predictor...")
        
        # Import and initialize
        from mega_enhanced_predictor import initialize_mega_predictor, get_mega_predictor
        
        print("Initializing mega predictor...")
        mega_predictor = await initialize_mega_predictor()
        
        if mega_predictor:
            print("SUCCESS: MEGA Enhanced Predictor initialized!")
            
            # Test a prediction
            test_context = {
                'recent_rolls': [45.23, 67.89, 34.56],
                'game_state': {
                    'server_seed': 'test_seed_hash',
                    'client_seed': 'test_client',
                    'nonce': 100,
                    'game_id': 'test_001'
                },
                'bankroll': 1000,
                'session_id': 'test',
                'timestamp': 1234567890
            }
            
            print("Getting test prediction...")
            result = await mega_predictor.get_mega_prediction(test_context)
            
            if result:
                print(f"Prediction successful!")
                print(f"Confidence: {result['mega_prediction'].get('confidence', 0):.2f}%")
                print(f"Methods: {result['mega_prediction'].get('methods_used', [])}")
                print(f"Processing time: {result.get('processing_time_seconds', 0):.3f}s")
                return True
            else:
                print("FAILED: No prediction result")
                return False
        else:
            print("FAILED: Could not initialize mega predictor")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("MEGA ENHANCED PREDICTOR TEST")
    print("=" * 60)
    
    success = asyncio.run(test_mega_predictor())
    
    if success:
        print("=" * 60)
        print("TEST PASSED: System ready for deployment!")
        print("=" * 60)
    else:
        print("=" * 60)
        print("TEST FAILED: Check error messages above")
        print("=" * 60)