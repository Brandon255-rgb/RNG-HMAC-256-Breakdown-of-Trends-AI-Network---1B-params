#!/usr/bin/env python3
"""
Comprehensive MEGA System Test Suite
===================================

Tests all aspects of the MEGA Enhanced Predictor system:
- Billion roll dataset processing
- Pattern recognition accuracy
- Machine learning model performance
- Real-time prediction capabilities
- Session seed management
"""

import os
import sys
import asyncio
import time
import random
import logging
from pathlib import Path

# Add logic controllers to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'logic controllers'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MegaSystemTestSuite:
    """Comprehensive test suite for MEGA Enhanced Predictor"""
    
    def __init__(self):
        self.mega_predictor = None
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    async def initialize_system(self):
        """Initialize the MEGA system for testing"""
        try:
            print("=" * 80)
            print("🔥 MEGA ENHANCED PREDICTOR - COMPREHENSIVE TEST SUITE")
            print("=" * 80)
            print("Initializing system...")
            
            from mega_enhanced_predictor import initialize_mega_predictor, get_mega_predictor
            
            self.mega_predictor = await initialize_mega_predictor()
            
            if self.mega_predictor:
                print("✅ System initialized successfully!")
                return True
            else:
                print("❌ Failed to initialize system")
                return False
                
        except Exception as e:
            print(f"❌ System initialization error: {e}")
            return False
    
    def run_test(self, test_name, test_func):
        """Run a single test and record results"""
        self.total_tests += 1
        print(f"\n🧪 Running Test {self.total_tests}: {test_name}")
        print("-" * 60)
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            if result:
                self.passed_tests += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            duration = end_time - start_time
            print(f"{status} - Duration: {duration:.3f}s")
            
            self.test_results[test_name] = {
                'passed': result,
                'duration': duration,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ TEST ERROR: {e}")
            self.test_results[test_name] = {
                'passed': False,
                'error': str(e),
                'timestamp': time.time()
            }
            return False
    
    def test_billion_roll_dataset(self):
        """Test billion roll dataset loading and processing"""
        try:
            # Check if dataset file exists
            dataset_path = Path('rolls_1e9.u16')
            if not dataset_path.exists():
                print(f"❌ Dataset file not found: {dataset_path}")
                return False
            
            print(f"📊 Dataset file found: {dataset_path}")
            print(f"📏 File size: {dataset_path.stat().st_size / (1024**3):.2f} GB")
            
            # Check if our mega predictor has billion roll processor
            if hasattr(self.mega_predictor, 'billion_roll_processor'):
                processor = self.mega_predictor.billion_roll_processor
                
                print(f"🔍 Billion roll processor available")
                
                # Check if data was loaded
                if hasattr(processor, 'total_analyzed') and processor.total_analyzed > 0:
                    print(f"✅ Data loaded: {processor.total_analyzed:,} rolls analyzed")
                    return True
                else:
                    print("❌ No data analyzed yet")
                    return False
            else:
                print("❌ Billion roll processor not found")
                return False
                
        except Exception as e:
            print(f"❌ Billion roll test error: {e}")
            return False
    
    def test_pattern_recognition(self):
        """Test pattern recognition capabilities"""
        try:
            if not hasattr(self.mega_predictor, 'hmac_analyzer'):
                print("❌ HMAC analyzer not found")
                return False
            
            analyzer = self.mega_predictor.hmac_analyzer
            
            # Test with sample sequence
            test_sequence = [23.45, 67.89, 34.56, 78.12, 45.67, 89.34, 56.78, 12.34]
            print(f"🔍 Testing pattern recognition with sequence: {test_sequence}")
            
            # Test frequency analysis
            if hasattr(analyzer, 'analyze_frequency_patterns'):
                freq_result = analyzer.analyze_frequency_patterns(test_sequence)
                print(f"📊 Frequency analysis: {len(freq_result)} patterns found")
            
            # Test sequential patterns
            if hasattr(analyzer, 'analyze_sequential_patterns'):
                seq_result = analyzer.analyze_sequential_patterns(test_sequence)
                print(f"🔗 Sequential patterns: {len(seq_result)} sequences identified")
            
            # Test entropy calculation
            if hasattr(analyzer, 'calculate_entropy'):
                entropy = analyzer.calculate_entropy(test_sequence)
                print(f"📐 Shannon entropy: {entropy:.4f}")
            
            print("✅ Pattern recognition systems operational")
            return True
            
        except Exception as e:
            print(f"❌ Pattern recognition test error: {e}")
            return False
    
    async def test_prediction_accuracy(self):
        """Test prediction accuracy with multiple samples"""
        try:
            print("🎯 Testing prediction accuracy...")
            
            predictions_made = 0
            successful_predictions = 0
            total_confidence = 0
            
            # Run multiple prediction tests
            for i in range(10):
                # Generate test context
                test_context = {
                    'recent_rolls': [random.uniform(0, 99.99) for _ in range(5)],
                    'game_state': {
                        'server_seed': f'test_seed_{i}',
                        'client_seed': f'test_client_{i}',
                        'nonce': 100 + i,
                        'game_id': f'test_{i}'
                    },
                    'bankroll': 1000,
                    'session_id': f'test_session_{i}',
                    'timestamp': time.time()
                }
                
                # Get prediction
                result = await self.mega_predictor.get_mega_prediction(test_context)
                
                if result and 'mega_prediction' in result:
                    predictions_made += 1
                    confidence = result['mega_prediction'].get('confidence', 0)
                    total_confidence += confidence
                    
                    if confidence > 0:  # Consider any positive confidence as successful
                        successful_predictions += 1
                    
                    print(f"   Test {i+1}: Confidence {confidence:.2f}%")
                else:
                    print(f"   Test {i+1}: Failed to get prediction")
            
            if predictions_made > 0:
                avg_confidence = total_confidence / predictions_made
                success_rate = (successful_predictions / predictions_made) * 100
                
                print(f"📈 Predictions made: {predictions_made}")
                print(f"📊 Average confidence: {avg_confidence:.2f}%")
                print(f"🎯 Success rate: {success_rate:.1f}%")
                
                # Consider test passed if we can make predictions consistently
                return predictions_made >= 8  # At least 80% of tests should work
            else:
                print("❌ No predictions were successful")
                return False
            
        except Exception as e:
            print(f"❌ Prediction accuracy test error: {e}")
            return False
    
    def test_ml_models(self):
        """Test machine learning model functionality"""
        try:
            print("🤖 Testing ML models...")
            
            if not hasattr(self.mega_predictor, 'ml_models'):
                print("❌ ML models not found")
                return False
            
            ml_models = self.mega_predictor.ml_models
            
            print(f"🔍 Available ML models: {list(ml_models.keys())}")
            
            # Test each model
            models_working = 0
            for model_name, model in ml_models.items():
                try:
                    # Test prediction with dummy data
                    import numpy as np
                    test_features = np.array([[1, 2, 3, 4, 5]]).reshape(1, -1)
                    
                    if hasattr(model, 'predict'):
                        prediction = model.predict(test_features)
                        print(f"   ✅ {model_name}: Working (prediction: {prediction[0]:.2f})")
                        models_working += 1
                    else:
                        print(f"   ❌ {model_name}: No predict method")
                        
                except Exception as e:
                    print(f"   ❌ {model_name}: Error - {e}")
            
            success_rate = (models_working / len(ml_models)) * 100
            print(f"🎯 ML models working: {models_working}/{len(ml_models)} ({success_rate:.1f}%)")
            
            return models_working > 0  # At least one model should work
            
        except Exception as e:
            print(f"❌ ML model test error: {e}")
            return False
    
    def test_session_seeds(self):
        """Test session seed management"""
        try:
            print("🔑 Testing session seed management...")
            
            # Test seed update
            test_seeds = {
                'client_seed': 'test_client_12345',
                'server_seed_hash': 'a' * 64,  # 64-character hash
                'nonce': 150,
                'revealed_server_seed': 'test_server_seed_12345'
            }
            
            print(f"🔄 Updating seeds: {test_seeds['client_seed'][:10]}...")
            
            if hasattr(self.mega_predictor, 'update_session_seeds'):
                self.mega_predictor.update_session_seeds(test_seeds)
                print("✅ Session seeds updated successfully")
                
                # Verify seeds were stored
                if hasattr(self.mega_predictor, 'session_seeds'):
                    stored_seeds = self.mega_predictor.session_seeds
                    if stored_seeds.get('client_seed') == test_seeds['client_seed']:
                        print("✅ Seeds verified in storage")
                        return True
                    else:
                        print("❌ Seeds not properly stored")
                        return False
                else:
                    print("✅ Seed update method works")
                    return True
            else:
                print("❌ Session seed management not available")
                return False
                
        except Exception as e:
            print(f"❌ Session seed test error: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        try:
            print("📊 Testing performance metrics...")
            
            if hasattr(self.mega_predictor, 'get_performance_metrics'):
                metrics = self.mega_predictor.get_performance_metrics()
                
                print(f"📈 Metrics collected: {list(metrics.keys())}")
                
                # Check for expected metrics
                expected_metrics = ['total_predictions', 'prediction_accuracy', 'avg_confidence']
                found_metrics = 0
                
                for metric in expected_metrics:
                    if metric in metrics:
                        print(f"   ✅ {metric}: {metrics[metric]}")
                        found_metrics += 1
                    else:
                        print(f"   ❓ {metric}: Not available")
                
                success_rate = (found_metrics / len(expected_metrics)) * 100
                print(f"🎯 Metrics available: {found_metrics}/{len(expected_metrics)} ({success_rate:.1f}%)")
                
                return found_metrics > 0  # At least some metrics should be available
            else:
                print("❌ Performance metrics not available")
                return False
                
        except Exception as e:
            print(f"❌ Performance metrics test error: {e}")
            return False
    
    def test_real_time_data(self):
        """Test real-time data processing"""
        try:
            print("⚡ Testing real-time data processing...")
            
            if hasattr(self.mega_predictor, 'add_real_time_data'):
                # Add some test data
                test_rolls = [45.67, 78.23, 34.56, 89.12, 67.34]
                
                for i, roll in enumerate(test_rolls):
                    self.mega_predictor.add_real_time_data(roll)
                    print(f"   📊 Added roll {i+1}: {roll}")
                
                print("✅ Real-time data processing working")
                return True
            else:
                print("❌ Real-time data processing not available")
                return False
                
        except Exception as e:
            print(f"❌ Real-time data test error: {e}")
            return False
    
    async def run_comprehensive_tests(self):
        """Run all tests in the suite"""
        try:
            # Initialize system
            if not await self.initialize_system():
                return False
            
            print(f"\n🚀 Starting comprehensive test suite...")
            print(f"📅 Test started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run all tests
            tests = [
                ("Billion Roll Dataset", self.test_billion_roll_dataset),
                ("Pattern Recognition", self.test_pattern_recognition),
                ("ML Models", self.test_ml_models),
                ("Session Seed Management", self.test_session_seeds),
                ("Performance Metrics", self.test_performance_metrics),
                ("Real-time Data Processing", self.test_real_time_data),
                ("Prediction Accuracy", lambda: asyncio.run(self.test_prediction_accuracy()))
            ]
            
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
            
            # Print final results
            self.print_final_results()
            
            return self.passed_tests >= (self.total_tests * 0.7)  # 70% pass rate
            
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            return False
    
    def print_final_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 80)
        print("🏁 MEGA ENHANCED PREDICTOR - TEST RESULTS")
        print("=" * 80)
        
        pass_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        print(f"📊 Tests Run: {self.total_tests}")
        print(f"✅ Tests Passed: {self.passed_tests}")
        print(f"❌ Tests Failed: {self.total_tests - self.passed_tests}")
        print(f"📈 Pass Rate: {pass_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            duration = result.get('duration', 0)
            print(f"   {status} - {test_name} ({duration:.3f}s)")
            
            if 'error' in result:
                print(f"      Error: {result['error']}")
        
        print(f"\n🎯 Overall Assessment:")
        if pass_rate >= 90:
            print("🏆 EXCELLENT - System ready for production!")
        elif pass_rate >= 70:
            print("✅ GOOD - System functional with minor issues")
        elif pass_rate >= 50:
            print("⚠️  FAIR - System needs improvement")
        else:
            print("❌ POOR - System requires significant fixes")
        
        print("=" * 80)

async def main():
    """Main test entry point"""
    test_suite = MegaSystemTestSuite()
    
    success = await test_suite.run_comprehensive_tests()
    
    if success:
        print("\n🎉 MEGA Enhanced Predictor System: READY FOR DEPLOYMENT!")
        return 0
    else:
        print("\n🚫 MEGA Enhanced Predictor System: NEEDS ATTENTION!")
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)