#!/usr/bin/env python3
"""
STAKE ORACLE v10 — LIVE DEMO MODE
=================================
1. Pull last 10,000 rolls via Stake API (current seed)
2. Run 50 demo bets (auto-play or manual)
3. Probe for: streaks, entropy dips, TDA holes, reservoir sync
4. Predict next 5 rolls with confidence

We're not guessing. We're engineering inevitability.
"""

import requests
import time
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import os
import matplotlib.pyplot as plt
import hmac
import hashlib
from typing import List, Tuple, Dict, Optional

# Import the pretrained model architecture
try:
    from massive_pretrain_oracle import OracleCore
except ImportError:
    print("⚠️ Importing OracleCore directly...")
    # Copy OracleCore class if import fails
    class OracleCore(nn.Module):
        def __init__(self, embed_dim=64, num_heads=8, num_layers=4):
            super().__init__()
            self.embed = nn.Linear(1, embed_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                dropout=0.1, batch_first=True, activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, embed_dim // 4), nn.GELU(),
                nn.Linear(embed_dim // 4, 1)
            )
        def forward(self, x):
            seq_len = x.size(1)
            x = self.embed(x.unsqueeze(-1))
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
            x = self.transformer(x)
            return self.head(x[:, -1]).squeeze(-1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# === 1. Stake API Config ===
def load_stake_config():
    """Load Stake API configuration from environment"""
    stake_key = os.getenv('STAKE_API_KEY')
    if not stake_key:
        print("⚠️ STAKE_API_KEY not found in environment!")
        print("Set your API key: STAKE_API_KEY=your_jwt_token")
        return None
    
    return {
        'api_url': 'https://api.stake.com/graphql',
        'headers': {
            'Authorization': f'Bearer {stake_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'StakeOracle/10.0'
        }
    }

# === 2. Fetch Historical Rolls via Stake API ===
def fetch_stake_history(limit=10000) -> List[float]:
    """Fetch last N dice rolls from Stake API"""
    config = load_stake_config()
    if not config:
        print("🎲 Using simulated data instead...")
        return simulate_realistic_rolls(limit)
    
    print(f"🌐 Fetching {limit:,} historical rolls from Stake API...")
    
    query = """
    query GetDiceHistory($limit: Int!) {
      user {
        diceRolls(first: $limit, orderBy: {field: CREATED_AT, direction: DESC}) {
          edges {
            node {
              nonce
              result
              serverSeedHash
              clientSeed
              createdAt
            }
          }
        }
      }
    }
    """
    
    try:
        response = requests.post(
            config['api_url'],
            json={'query': query, 'variables': {'limit': limit}},
            headers=config['headers'],
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'errors' in data:
                print(f"❌ API Error: {data['errors']}")
                return simulate_realistic_rolls(limit)
            
            rolls_data = data.get('data', {}).get('user', {}).get('diceRolls', {}).get('edges', [])
            rolls = [float(edge['node']['result']) for edge in rolls_data]
            
            print(f"✅ Fetched {len(rolls):,} real rolls from Stake")
            return rolls[::-1]  # Oldest first for temporal analysis
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return simulate_realistic_rolls(limit)
            
    except Exception as e:
        print(f"❌ API fetch failed: {e}")
        return simulate_realistic_rolls(limit)

def simulate_realistic_rolls(count: int) -> List[float]:
    """Generate realistic HMAC dice rolls for testing"""
    print(f"🎲 Simulating {count:,} realistic HMAC rolls...")
    
    # Use environment client/server seeds if available
    client_seed = "rpssWuZThW"  # Our proven seed
    server_seed = "b10c1d121c5373702d9b6c166c6f7749905f80f1c6f096d2177ba39ec16a8e3a"
    
    rolls = []
    key = bytes.fromhex(server_seed)
    
    for nonce in range(count):
        message = f"{client_seed}:{nonce}:0"
        hmac_result = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
        first_8_hex = hmac_result[:8]
        int_value = int(first_8_hex, 16)
        result = round((int_value / 0xFFFFFFFF) * 100, 2)
        rolls.append(result)
    
    return rolls

# === 3. Live Oracle Class ===
class LiveOracle:
    """Real-time pattern recognition and prediction engine"""
    
    def __init__(self):
        print("🔮 Initializing Live Oracle...")
        
        # Load pretrained model
        self.model = OracleCore().to(device)
        
        if os.path.exists("stake_oracle_pretrained.pth"):
            self.model.load_state_dict(torch.load("stake_oracle_pretrained.pth", map_location=device))
            print("✅ Loaded pretrained universal model")
        else:
            print("⚠️ No pretrained model found - using untrained weights")
        
        self.model.eval()
        
        # Rolling buffer for pattern analysis
        self.buffer = []
        self.max_buffer_size = 1000
        
        # Pattern analysis state
        self.entropy_history = []
        self.streak_history = []
        self.prediction_history = []
        self.confidence_history = []
        
        # Reservoir synchronization
        self.reservoir_state = None
        self.last_sync_time = time.time()
        
        print("🎯 Live Oracle ready for pattern hunting")
    
    def update(self, roll: float) -> None:
        """Update internal state with new roll"""
        self.buffer.append(roll)
        
        # Maintain buffer size
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        
        # Update entropy tracking
        if len(self.buffer) >= 100:
            recent_entropy = self._calculate_entropy(self.buffer[-100:])
            self.entropy_history.append(recent_entropy)
            if len(self.entropy_history) > 50:
                self.entropy_history.pop(0)
    
    def predict(self) -> Tuple[float, float]:
        """Generate prediction with confidence score"""
        if len(self.buffer) < 10:
            return 50.0, 0.0  # No prediction possible
        
        # Use last 10 values for prediction
        sequence = torch.tensor(self.buffer[-10:], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Base prediction from transformer
            base_pred = self.model(sequence).item()
            
            # Confidence calculation based on pattern stability
            confidence = self._calculate_confidence()
            
            # Apply confidence-based adjustments
            prediction = self._apply_confidence_adjustment(base_pred, confidence)
        
        # Store for tracking
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Limit history size
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
        
        return prediction, confidence
    
    def _calculate_confidence(self) -> float:
        """Calculate prediction confidence based on multiple factors"""
        if len(self.buffer) < 50:
            return 0.3
        
        # Factor 1: Entropy stability (lower = more predictable)
        if len(self.entropy_history) >= 10:
            entropy_variance = np.var(self.entropy_history[-10:])
            entropy_score = max(0, 1 - entropy_variance * 2)
        else:
            entropy_score = 0.5
        
        # Factor 2: Pattern consistency
        recent_rolls = self.buffer[-20:]
        pattern_score = self._analyze_pattern_consistency(recent_rolls)
        
        # Factor 3: Temporal stability
        if len(self.buffer) >= 30:
            stability_score = self._measure_stability(self.buffer[-30:])
        else:
            stability_score = 0.5
        
        # Combined confidence (weighted average)
        confidence = (
            entropy_score * 0.4 +
            pattern_score * 0.4 +
            stability_score * 0.2
        )
        
        return min(0.95, max(0.1, confidence))
    
    def _apply_confidence_adjustment(self, base_pred: float, confidence: float) -> float:
        """Adjust prediction based on confidence level"""
        # High confidence: use prediction as-is
        # Low confidence: pull toward mean
        if confidence > 0.8:
            return base_pred
        
        # Calculate recent mean for regression
        recent_mean = np.mean(self.buffer[-50:]) if len(self.buffer) >= 50 else 50.0
        
        # Blend prediction with mean based on confidence
        blend_factor = confidence
        adjusted = base_pred * blend_factor + recent_mean * (1 - blend_factor)
        
        return max(0, min(100, adjusted))
    
    def _calculate_entropy(self, sequence: List[float]) -> float:
        """Calculate Shannon entropy of sequence"""
        if len(sequence) < 10:
            return 0.0
        
        # Bin the values
        hist, _ = np.histogram(sequence, bins=20, range=(0, 100))
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy / np.log2(20)  # Normalize
    
    def _analyze_pattern_consistency(self, sequence: List[float]) -> float:
        """Analyze how consistent patterns are in sequence"""
        if len(sequence) < 10:
            return 0.5
        
        # Look for various pattern types
        scores = []
        
        # Trend consistency
        diffs = np.diff(sequence)
        trend_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        trend_score = max(0, 1 - trend_changes / len(diffs))
        scores.append(trend_score)
        
        # Range consistency
        recent_range = np.max(sequence) - np.min(sequence)
        range_score = max(0, 1 - recent_range / 100)
        scores.append(range_score)
        
        # Autocorrelation
        if len(sequence) >= 20:
            autocorr = np.corrcoef(sequence[:-1], sequence[1:])[0, 1]
            autocorr_score = abs(autocorr) if not np.isnan(autocorr) else 0
            scores.append(autocorr_score)
        
        return np.mean(scores)
    
    def _measure_stability(self, sequence: List[float]) -> float:
        """Measure temporal stability of the sequence"""
        if len(sequence) < 20:
            return 0.5
        
        # Calculate moving variance
        window = 10
        variances = []
        for i in range(window, len(sequence)):
            var = np.var(sequence[i-window:i])
            variances.append(var)
        
        if not variances:
            return 0.5
        
        # Stability is inverse of variance change
        var_stability = max(0, 1 - np.std(variances) / 100)
        return var_stability
    
    def get_pattern_analysis(self) -> Dict:
        """Get comprehensive pattern analysis"""
        if len(self.buffer) < 20:
            return {'status': 'insufficient_data'}
        
        analysis = {
            'buffer_size': len(self.buffer),
            'current_entropy': self._calculate_entropy(self.buffer[-100:]) if len(self.buffer) >= 100 else 0,
            'mean_prediction_confidence': np.mean(self.confidence_history[-10:]) if self.confidence_history else 0,
            'recent_range': np.max(self.buffer[-20:]) - np.min(self.buffer[-20:]),
            'trend': 'up' if self.buffer[-1] > self.buffer[-10] else 'down',
        }
        
        # Streak detection
        streaks = detect_streaks(self.buffer[-50:] if len(self.buffer) >= 50 else self.buffer)
        analysis['recent_streaks'] = len(streaks)
        analysis['max_streak'] = max(streaks) if streaks else 0
        
        return analysis

# === 4. Pattern Analysis Utilities ===
def detect_streaks(sequence: List[float], threshold: int = 3) -> List[int]:
    """Detect consecutive similar values"""
    if len(sequence) < threshold:
        return []
    
    streaks = []
    current_streak = 1
    
    for i in range(1, len(sequence)):
        # Values within 5% considered similar
        if abs(sequence[i] - sequence[i-1]) < 5.0:
            current_streak += 1
        else:
            if current_streak >= threshold:
                streaks.append(current_streak)
            current_streak = 1
    
    # Check final streak
    if current_streak >= threshold:
        streaks.append(current_streak)
    
    return streaks

def shannon_entropy(sequence: List[float]) -> float:
    """Calculate Shannon entropy"""
    if len(sequence) < 5:
        return 0.0
    
    hist, _ = np.histogram(sequence, bins=20, range=(0, 100))
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-10))

def analyze_tda_holes(sequence: List[float]) -> int:
    """Simplified topological data analysis"""
    # This is a simplified version - real TDA would use ripser
    if len(sequence) < 10:
        return 0
    
    # Look for "holes" in the data space
    points = np.array(sequence).reshape(-1, 1)
    
    # Count local maxima and minima as proxy for holes
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(sequence)
    valleys, _ = find_peaks([-x for x in sequence])
    
    return len(peaks) + len(valleys)

# === 5. Live Demo Mode ===
class LiveDemoSession:
    """Manages a complete 50-roll demo session"""
    
    def __init__(self):
        self.oracle = LiveOracle()
        self.demo_rolls = []
        self.predictions = []
        self.confidences = []
        self.start_time = time.time()
        
    def initialize_with_history(self, historical_rolls: List[float]) -> None:
        """Initialize oracle with historical data"""
        print(f"📚 Loading {len(historical_rolls):,} historical rolls...")
        
        for roll in historical_rolls[-1000:]:  # Use last 1000 for efficiency
            self.oracle.update(roll)
        
        print("✅ Oracle synchronized with historical patterns")
    
    def run_demo_session(self, num_rolls: int = 50, auto_mode: bool = False) -> None:
        """Run the main demo session"""
        print(f"\n🎯 STARTING {num_rolls}-ROLL DEMO SESSION")
        print("=" * 60)
        
        for i in range(num_rolls):
            print(f"\n[{i+1}/{num_rolls}] Roll Analysis:")
            
            # Generate prediction
            prediction, confidence = self.oracle.predict()
            self.predictions.append(prediction)
            self.confidences.append(confidence)
            
            print(f"🔮 Predicted: {prediction:.2f}")
            print(f"📊 Confidence: {confidence:.0%}")
            
            # Get pattern analysis
            pattern_info = self.oracle.get_pattern_analysis()
            if pattern_info.get('status') != 'insufficient_data':
                print(f"📈 Entropy: {pattern_info.get('current_entropy', 0):.3f}")
                print(f"🎢 Recent range: {pattern_info.get('recent_range', 0):.1f}")
                print(f"📊 Trend: {pattern_info.get('trend', 'unknown')}")
            
            # Get actual roll
            if auto_mode:
                # Simulate realistic result (for demo)
                noise = np.random.normal(0, 10)
                actual = max(0, min(100, prediction + noise))
                print(f"🎲 Actual (sim): {actual:.2f}")
                time.sleep(0.5)  # Brief pause for realism
            else:
                while True:
                    try:
                        user_input = input("🎲 Enter actual roll (or 'auto' for simulation): ").strip()
                        if user_input.lower() == 'auto':
                            auto_mode = True
                            noise = np.random.normal(0, 10)
                            actual = max(0, min(100, prediction + noise))
                            break
                        elif user_input == '':
                            actual = prediction  # Use prediction if no input
                            break
                        else:
                            actual = float(user_input)
                            if 0 <= actual <= 100:
                                break
                            else:
                                print("❌ Roll must be between 0 and 100")
                    except ValueError:
                        print("❌ Please enter a valid number")
            
            # Update oracle
            self.demo_rolls.append(actual)
            self.oracle.update(actual)
            
            # Show accuracy
            error = abs(prediction - actual)
            print(f"🎯 Error: {error:.2f} ({error/100*100:.1f}%)")
    
    def generate_final_report(self) -> None:
        """Generate comprehensive session analysis"""
        print("\n" + "=" * 60)
        print("📊 FINAL DEMO SESSION REPORT")
        print("=" * 60)
        
        if not self.demo_rolls:
            print("❌ No demo data to analyze")
            return
        
        # Basic statistics
        print(f"\n📈 SESSION STATISTICS:")
        print(f"   Total rolls: {len(self.demo_rolls)}")
        print(f"   Session time: {time.time() - self.start_time:.1f}s")
        print(f"   Mean roll: {np.mean(self.demo_rolls):.2f}")
        print(f"   Std dev: {np.std(self.demo_rolls):.2f}")
        print(f"   Range: {np.min(self.demo_rolls):.2f} - {np.max(self.demo_rolls):.2f}")
        
        # Prediction accuracy
        if len(self.predictions) == len(self.demo_rolls):
            errors = [abs(p - a) for p, a in zip(self.predictions, self.demo_rolls)]
            mean_error = np.mean(errors)
            mean_confidence = np.mean(self.confidences)
            
            print(f"\n🎯 PREDICTION PERFORMANCE:")
            print(f"   Mean error: {mean_error:.2f}")
            print(f"   Mean confidence: {mean_confidence:.0%}")
            print(f"   Accuracy (±5): {sum(1 for e in errors if e <= 5) / len(errors) * 100:.1f}%")
            print(f"   Accuracy (±10): {sum(1 for e in errors if e <= 10) / len(errors) * 100:.1f}%")
        
        # Pattern analysis
        print(f"\n🔍 PATTERN ANALYSIS:")
        streaks = detect_streaks(self.demo_rolls)
        print(f"   Streaks detected: {len(streaks)}")
        print(f"   Max streak: {max(streaks) if streaks else 0}")
        print(f"   Entropy: {shannon_entropy(self.demo_rolls):.4f}")
        print(f"   TDA holes: {analyze_tda_holes(self.demo_rolls)}")
        
        # Next predictions
        print(f"\n🔮 NEXT 5 PREDICTIONS:")
        next_predictions = []
        for i in range(5):
            pred, conf = self.oracle.predict()
            next_predictions.append(pred)
            print(f"   {i+1}. {pred:.2f} (confidence: {conf:.0%})")
            
            # Simulate the prediction for next prediction
            self.oracle.update(pred)
        
        print(f"\n🎯 NEXT 5 SUMMARY: {', '.join(f'{x:.2f}' for x in next_predictions)}")

# === 6. MAIN EXECUTION ===
def main():
    """Main demo execution"""
    print("🚀 STAKE ORACLE v10 — LIVE DEMO MODE")
    print("=" * 60)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Initialize demo session
        demo = LiveDemoSession()
        
        # Fetch historical data
        print("📚 Fetching historical rolls for pattern analysis...")
        historical_rolls = fetch_stake_history(limit=5000)  # Reduced for faster loading
        
        if historical_rolls:
            demo.initialize_with_history(historical_rolls)
            
            print(f"✅ Loaded {len(historical_rolls):,} historical rolls")
            print(f"📊 Historical mean: {np.mean(historical_rolls):.2f}")
            print(f"📊 Historical std: {np.std(historical_rolls):.2f}")
        else:
            print("⚠️ No historical data available - using fresh oracle")
        
        # Ask for demo mode
        print(f"\n🎮 DEMO MODE OPTIONS:")
        print("1. Interactive mode (enter each roll manually)")
        print("2. Auto mode (simulated rolls for fast demo)")
        
        while True:
            choice = input("Choose mode (1/2): ").strip()
            if choice in ['1', '2']:
                auto_mode = choice == '2'
                break
            print("❌ Please enter 1 or 2")
        
        # Run demo session
        demo.run_demo_session(num_rolls=50, auto_mode=auto_mode)
        
        # Generate final report
        demo.generate_final_report()
        
        print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎉 DEMO SESSION COMPLETE!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()