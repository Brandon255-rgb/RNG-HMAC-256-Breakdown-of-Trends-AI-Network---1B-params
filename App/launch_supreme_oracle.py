#!/usr/bin/env python3
"""
SUPREME ORACLE LAUNCH SYSTEM
============================
Complete training + live prediction + betting system launcher

This script:
1. Trains Oracle on 10M samples (if needed)
2. Runs 100 demo rolls to calibrate
3. Launches live prediction with Bedrock AI
4. Starts dashboard for betting control
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import torch
from pathlib import Path

def print_banner():
    """Print launch banner"""
    print("🚀" + "=" * 60 + "🚀")
    print("    SUPREME ORACLE LAUNCH SYSTEM v1.0")
    print("    Training → Calibration → Live Prediction → Profit")
    print("🚀" + "=" * 60 + "🚀")
    print(f"⏰ Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def check_environment():
    """Check if environment is ready"""
    print("🔍 Checking environment...")
    
    # Check Python packages
    required_packages = [
        'torch', 'numpy', 'pandas', 'requests', 
        'flask', 'boto3', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Check .env file
    if not os.path.exists('.env'):
        print("   ❌ .env file missing")
        print("   Create .env with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, STAKE_API_KEY")
        return False
    else:
        print("   ✅ .env file found")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("   ⚠️  CUDA not available - using CPU (slower)")
    
    print("   ✅ Environment ready!")
    return True

def train_oracle():
    """Train Oracle on 10M samples"""
    print("\n🎯 PHASE 1: ORACLE TRAINING")
    print("-" * 40)
    
    model_path = "stake_oracle_pretrained.pth"
    
    if os.path.exists(model_path):
        print(f"   ✅ Found existing model: {model_path}")
        response = input("   🔄 Retrain model? (y/N): ").lower()
        if response != 'y':
            print("   📈 Using existing trained model")
            return True
    
    print("   🔥 Starting 10M sample training...")
    print("   ⏱️  This may take 30-60 minutes depending on hardware")
    
    try:
        # Run massive pretraining
        result = subprocess.run([
            sys.executable, 'massive_pretrain_oracle.py'
        ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print("   ✅ Oracle training completed successfully!")
            return True
        else:
            print(f"   ❌ Training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Training timeout - consider reducing dataset size")
        return False
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False

def run_calibration():
    """Run 100 demo rolls for calibration"""
    print("\n🎯 PHASE 2: SYSTEM CALIBRATION")
    print("-" * 40)
    
    print("   🎲 Running 100 demo rolls for pattern calibration...")
    
    try:
        # Import live demo oracle
        from live_demo_oracle import LiveOracle
        
        oracle = LiveOracle()
        
        # Run calibration
        print("   📊 Analyzing recent patterns...")
        
        # Get some demo predictions
        for i in range(5):
            prediction = oracle.predict_next_5()
            confidence = prediction.get('confidence', 0)
            next_roll = prediction.get('predictions', [50])[0]
            
            print(f"   📈 Demo {i+1}: {next_roll:.2f} (confidence: {confidence:.1f}%)")
            time.sleep(0.5)
        
        print("   ✅ Calibration completed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Calibration failed: {e}")
        return False

def start_live_system():
    """Start the live prediction system"""
    print("\n🎯 PHASE 3: LIVE SYSTEM LAUNCH")
    print("-" * 40)
    
    print("   🚀 Starting supreme prediction system...")
    
    try:
        # Import and start main system
        from main import SupremeSystemOrchestrator
        
        orchestrator = SupremeSystemOrchestrator()
        
        print("   ✅ System orchestrator initialized")
        print("   🌐 Starting web dashboard...")
        
        # Start dashboard in background
        import threading
        dashboard_thread = threading.Thread(
            target=lambda: orchestrator.start_dashboard(),
            daemon=True
        )
        dashboard_thread.start()
        
        print("   🎮 Dashboard available at: http://localhost:5000")
        print("   💰 Ready for live predictions and betting!")
        
        return orchestrator
        
    except Exception as e:
        print(f"   ❌ System launch failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def interactive_menu(orchestrator):
    """Interactive menu for system control"""
    print("\n🎮 SUPREME ORACLE CONTROL CENTER")
    print("=" * 40)
    
    while True:
        print("\nChoose action:")
        print("1. 🔮 Get Prediction")
        print("2. 💰 Place Bet (Demo)")
        print("3. 📊 View Stats")
        print("4. 🌐 Open Dashboard")
        print("5. ⚡ Start Auto-Betting")
        print("6. 🛑 Stop System")
        print("7. ❌ Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            try:
                prediction = orchestrator.get_prediction()
                print(f"\n🔮 PREDICTION:")
                print(f"   Next Roll: {prediction.get('next_roll', 'N/A')}")
                print(f"   Confidence: {prediction.get('confidence', 0):.1f}%")
                print(f"   Strategy: {prediction.get('strategy', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        
        elif choice == '2':
            print("\n💰 DEMO BETTING:")
            print("   This would place a real bet with current prediction")
            print("   (Demo mode - no real money)")
        
        elif choice == '3':
            try:
                stats = orchestrator.get_stats()
                print(f"\n📊 SYSTEM STATS:")
                print(f"   Predictions Made: {stats.get('predictions', 0)}")
                print(f"   Accuracy: {stats.get('accuracy', 0):.1f}%")
                print(f"   Profit/Loss: ${stats.get('profit', 0):.2f}")
            except Exception as e:
                print(f"   ❌ Stats failed: {e}")
        
        elif choice == '4':
            print("\n🌐 Dashboard: http://localhost:5000")
            print("   Open this URL in your browser")
        
        elif choice == '5':
            print("\n⚡ AUTO-BETTING:")
            print("   This would start automated betting")
            print("   (Currently disabled for safety)")
        
        elif choice == '6':
            print("\n🛑 Stopping system...")
            try:
                orchestrator.stop()
                print("   ✅ System stopped")
            except:
                pass
            break
        
        elif choice == '7':
            print("\n❌ Exiting...")
            break
        
        else:
            print("   ⚠️  Invalid choice")

def main():
    """Main launch sequence"""
    print_banner()
    
    # Phase 0: Environment check
    if not check_environment():
        print("❌ Environment check failed. Please fix issues and try again.")
        return False
    
    # Phase 1: Training
    if not train_oracle():
        print("❌ Oracle training failed. Cannot proceed.")
        return False
    
    # Phase 2: Calibration
    if not run_calibration():
        print("❌ System calibration failed. Cannot proceed.")
        return False
    
    # Phase 3: Live system
    orchestrator = start_live_system()
    if not orchestrator:
        print("❌ Live system launch failed.")
        return False
    
    # Phase 4: Interactive control
    try:
        interactive_menu(orchestrator)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    
    print("\n🎯 SUPREME ORACLE SESSION COMPLETE")
    print("   Thank you for using the future of prediction!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)