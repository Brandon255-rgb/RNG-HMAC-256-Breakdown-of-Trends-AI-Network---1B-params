#!/usr/bin/env python3
"""
🚀 SUPREME AI BETTING DASHBOARD LAUNCHER
Launch script for the ultimate profit maximization system
"""

import os
import sys
import subprocess

def main():
    print("🚀 SUPREME AI BETTING DASHBOARD")
    print("=" * 50)
    print("💰 THE ULTIMATE PROFIT MAXIMIZATION SYSTEM")
    print("🧠 Powered by Amazon Bedrock AI + 22 Neural Strategies")
    print("=" * 50)
    
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the dashboard in logic controllers
    dashboard_path = os.path.join(current_dir, 'logic controllers', 'ultimate_supreme_dashboard.py')
    
    # Check if dashboard exists
    if not os.path.exists(dashboard_path):
        print("❌ ERROR: Dashboard file not found!")
        print(f"Looking for: {dashboard_path}")
        return 1
    
    # Change to the logic controllers directory
    os.chdir(os.path.join(current_dir, 'logic controllers'))
    
    print(f"📁 Working directory: {os.getcwd()}")
    print("🔥 Starting Supreme AI Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("=" * 50)
    
    try:
        # Launch the dashboard
        subprocess.run([sys.executable, 'ultimate_supreme_dashboard.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard shutdown requested by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed to start: {e}")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())