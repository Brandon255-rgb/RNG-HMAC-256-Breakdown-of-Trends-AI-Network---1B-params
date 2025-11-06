#!/usr/bin/env python3
"""
STAKE ORACLE v10 — SUPPORT UTILITIES
====================================
Advanced pattern analysis utilities for chaos engineering:
- Streak detection with configurable thresholds
- Shannon entropy calculation for randomness analysis  
- Topological Data Analysis (TDA) hole detection
- Reservoir synchronization and state management
- Confidence scoring and stability metrics

We're not guessing. We're engineering inevitability.
"""

import numpy as np
import scipy.signal
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# === STREAK DETECTION ===
def detect_streaks(sequence: List[float], threshold: int = 3, similarity_pct: float = 5.0) -> List[Dict]:
    """
    Detect consecutive similar values in sequence
    
    Args:
        sequence: List of values to analyze
        threshold: Minimum streak length to detect
        similarity_pct: Percentage threshold for considering values similar
    
    Returns:
        List of streak dictionaries with start, end, length, and values
    """
    if len(sequence) < threshold:
        return []
    
    streaks = []
    current_streak = {
        'start': 0,
        'length': 1,
        'values': [sequence[0]],
        'mean': sequence[0]
    }
    
    for i in range(1, len(sequence)):
        # Check if current value is similar to streak mean
        if abs(sequence[i] - current_streak['mean']) <= similarity_pct:
            # Extend current streak
            current_streak['length'] += 1
            current_streak['values'].append(sequence[i])
            current_streak['mean'] = np.mean(current_streak['values'])
        else:
            # End current streak if it meets threshold
            if current_streak['length'] >= threshold:
                current_streak['end'] = i - 1
                current_streak['variance'] = np.var(current_streak['values'])
                streaks.append(current_streak.copy())
            
            # Start new streak
            current_streak = {
                'start': i,
                'length': 1,
                'values': [sequence[i]],
                'mean': sequence[i]
            }
    
    # Check final streak
    if current_streak['length'] >= threshold:
        current_streak['end'] = len(sequence) - 1
        current_streak['variance'] = np.var(current_streak['values'])
        streaks.append(current_streak)
    
    return streaks

def analyze_streak_patterns(streaks: List[Dict]) -> Dict:
    """Analyze patterns in detected streaks"""
    if not streaks:
        return {'total_streaks': 0}
    
    lengths = [s['length'] for s in streaks]
    means = [s['mean'] for s in streaks]
    variances = [s['variance'] for s in streaks]
    
    return {
        'total_streaks': len(streaks),
        'avg_length': np.mean(lengths),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'length_std': np.std(lengths),
        'avg_value': np.mean(means),
        'value_range': max(means) - min(means),
        'avg_variance': np.mean(variances),
        'streak_density': len(streaks) / sum(lengths) if sum(lengths) > 0 else 0
    }

# === ENTROPY ANALYSIS ===
def shannon_entropy(sequence: List[float], bins: int = 20, normalize: bool = True) -> float:
    """
    Calculate Shannon entropy of sequence
    
    Args:
        sequence: Input sequence
        bins: Number of bins for histogram
        normalize: Whether to normalize by log(bins)
    
    Returns:
        Shannon entropy value
    """
    if len(sequence) < 2:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(sequence, bins=bins, range=(0, 100))
    
    # Convert to probabilities
    hist = hist / hist.sum()
    
    # Calculate entropy
    ent = -np.sum(hist * np.log2(hist + 1e-12))
    
    # Normalize if requested
    if normalize:
        ent = ent / np.log2(bins)
    
    return ent

def conditional_entropy(sequence: List[float], lag: int = 1, bins: int = 10) -> float:
    """Calculate conditional entropy H(X_t | X_{t-lag})"""
    if len(sequence) <= lag:
        return 0.0
    
    # Create lagged pairs
    current = sequence[lag:]
    lagged = sequence[:-lag]
    
    # Discretize values
    current_binned = np.digitize(current, np.linspace(0, 100, bins+1)) - 1
    lagged_binned = np.digitize(lagged, np.linspace(0, 100, bins+1)) - 1
    
    # Calculate joint and marginal entropies
    joint_hist = np.histogram2d(current_binned, lagged_binned, bins=bins)[0]
    joint_probs = joint_hist / joint_hist.sum()
    
    marginal_lagged = np.sum(joint_probs, axis=0)
    
    # Conditional entropy
    cond_ent = 0.0
    for j in range(bins):
        if marginal_lagged[j] > 0:
            conditional_probs = joint_probs[:, j] / marginal_lagged[j]
            cond_ent += marginal_lagged[j] * entropy(conditional_probs + 1e-12, base=2)
    
    return cond_ent

def mutual_information(x: List[float], y: List[float], bins: int = 10) -> float:
    """Calculate mutual information between two sequences"""
    if len(x) != len(y) or len(x) < 10:
        return 0.0
    
    # Discretize
    x_binned = np.digitize(x, np.linspace(0, 100, bins+1)) - 1
    y_binned = np.digitize(y, np.linspace(0, 100, bins+1)) - 1
    
    # Calculate entropies
    h_x = entropy(np.bincount(x_binned) / len(x_binned), base=2)
    h_y = entropy(np.bincount(y_binned) / len(y_binned), base=2)
    
    # Joint entropy
    joint_hist = np.histogram2d(x_binned, y_binned, bins=bins)[0]
    joint_probs = joint_hist / joint_hist.sum()
    h_xy = entropy(joint_probs.flatten() + 1e-12, base=2)
    
    return h_x + h_y - h_xy

# === TOPOLOGICAL DATA ANALYSIS ===
def analyze_tda_holes(sequence: List[float], window_size: int = 50) -> Dict:
    """
    Simplified Topological Data Analysis for detecting holes in data structure
    
    Args:
        sequence: Input sequence
        window_size: Size of sliding window for local analysis
    
    Returns:
        Dictionary with TDA metrics
    """
    if len(sequence) < window_size:
        return {'holes': 0, 'persistence': 0}
    
    results = {
        'total_holes': 0,
        'avg_persistence': 0,
        'hole_density': 0,
        'local_analyses': []
    }
    
    # Sliding window analysis
    for i in range(0, len(sequence) - window_size + 1, window_size // 2):
        window = sequence[i:i + window_size]
        local_result = _analyze_local_topology(window)
        results['local_analyses'].append(local_result)
        results['total_holes'] += local_result['holes']
    
    if results['local_analyses']:
        avg_persistence = np.mean([r['persistence'] for r in results['local_analyses']])
        results['avg_persistence'] = avg_persistence
        results['hole_density'] = results['total_holes'] / len(results['local_analyses'])
    
    return results

def _analyze_local_topology(window: List[float]) -> Dict:
    """Analyze local topological features in a window"""
    # Convert to 2D embedding using delay coordinates
    if len(window) < 10:
        return {'holes': 0, 'persistence': 0}
    
    # Find local maxima and minima (0-dimensional features)
    peaks, peak_props = scipy.signal.find_peaks(window, height=np.mean(window))
    valleys, valley_props = scipy.signal.find_peaks([-x for x in window], height=-np.mean(window))
    
    # Estimate 1-dimensional holes (cycles)
    # Look for enclosed regions between peaks and valleys
    holes = 0
    persistence = 0
    
    if len(peaks) > 1 and len(valleys) > 1:
        # Sort all extrema
        all_extrema = [(i, window[i], 'peak') for i in peaks] + [(i, -window[i], 'valley') for i in valleys]
        all_extrema.sort()
        
        # Look for alternating peak-valley patterns (potential holes)
        for i in range(len(all_extrema) - 2):
            curr_type = all_extrema[i][2]
            next_type = all_extrema[i + 1][2]
            
            if curr_type != next_type:  # Alternating pattern
                # Calculate persistence (strength of feature)
                height_diff = abs(all_extrema[i][1] - all_extrema[i + 1][1])
                if height_diff > np.std(window):  # Significant feature
                    holes += 1
                    persistence += height_diff
    
    return {
        'holes': holes,
        'persistence': persistence / max(1, holes),
        'peaks': len(peaks),
        'valleys': len(valleys),
        'extrema_ratio': (len(peaks) + len(valleys)) / len(window)
    }

# === RESERVOIR SYNCHRONIZATION ===
class ReservoirState:
    """Manages reservoir computing state for pattern synchronization"""
    
    def __init__(self, size: int = 100, spectral_radius: float = 0.95, sparsity: float = 0.1):
        self.size = size
        self.state = np.zeros(size)
        
        # Generate reservoir weight matrix
        self.W = np.random.randn(size, size) * sparsity
        self.W[np.random.rand(size, size) > sparsity] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        self.W = self.W * (spectral_radius / np.max(np.abs(eigenvalues)))
        
        # Input weights
        self.W_in = np.random.randn(size, 1) * 0.1
        
        # Leak rate
        self.alpha = 0.1
        
        # Synchronization metrics
        self.sync_history = deque(maxlen=50)
        self.last_sync_score = 0.0
    
    def update(self, input_value: float) -> np.ndarray:
        """Update reservoir state with new input"""
        # Reservoir equation: x(t+1) = (1-α)x(t) + α*tanh(W*x(t) + W_in*u(t))
        input_vector = np.array([[input_value]])
        
        new_state = (1 - self.alpha) * self.state + \
                   self.alpha * np.tanh(self.W @ self.state.reshape(-1, 1) + 
                                       self.W_in @ input_vector).flatten()
        
        # Calculate synchronization score (how much state changed)
        sync_score = np.linalg.norm(new_state - self.state)
        self.sync_history.append(sync_score)
        self.last_sync_score = sync_score
        
        self.state = new_state
        return self.state
    
    def get_sync_metrics(self) -> Dict:
        """Get reservoir synchronization metrics"""
        if len(self.sync_history) < 10:
            return {'status': 'insufficient_data'}
        
        sync_array = np.array(self.sync_history)
        
        return {
            'current_sync': self.last_sync_score,
            'avg_sync': np.mean(sync_array),
            'sync_variance': np.var(sync_array),
            'sync_trend': 'increasing' if sync_array[-5:].mean() > sync_array[-10:-5].mean() else 'decreasing',
            'stability': 1.0 / (1.0 + np.var(sync_array)),  # Higher = more stable
            'reservoir_norm': np.linalg.norm(self.state)
        }

# === CONFIDENCE SCORING ===
def calculate_prediction_confidence(
    sequence: List[float],
    prediction: float,
    lookback: int = 50
) -> Tuple[float, Dict]:
    """
    Calculate confidence score for a prediction based on multiple factors
    
    Args:
        sequence: Historical sequence
        prediction: The prediction to score
        lookback: Number of recent values to consider
    
    Returns:
        Tuple of (confidence_score, detailed_metrics)
    """
    if len(sequence) < 10:
        return 0.1, {'status': 'insufficient_data'}
    
    recent = sequence[-lookback:] if len(sequence) >= lookback else sequence
    
    # Factor 1: Entropy-based confidence (lower entropy = higher confidence)
    ent = shannon_entropy(recent)
    entropy_confidence = max(0, 1 - ent)
    
    # Factor 2: Stability-based confidence
    if len(recent) >= 20:
        windows = [recent[i:i+10] for i in range(len(recent)-10)]
        window_means = [np.mean(w) for w in windows]
        stability = 1.0 / (1.0 + np.var(window_means))
    else:
        stability = 0.5
    
    # Factor 3: Pattern consistency
    streaks = detect_streaks(recent, threshold=3)
    pattern_score = len(streaks) / max(1, len(recent) // 10)  # Normalize by expected streaks
    pattern_confidence = min(1.0, pattern_score)
    
    # Factor 4: Prediction reasonableness (how well it fits recent distribution)
    recent_mean = np.mean(recent)
    recent_std = np.std(recent)
    z_score = abs(prediction - recent_mean) / max(recent_std, 1.0)
    reasonableness = max(0, 1 - z_score / 3)  # Within 3 sigma is reasonable
    
    # Weighted combination
    weights = [0.3, 0.25, 0.25, 0.2]
    confidence = (
        entropy_confidence * weights[0] +
        stability * weights[1] +
        pattern_confidence * weights[2] +
        reasonableness * weights[3]
    )
    
    # Detailed metrics
    metrics = {
        'entropy': ent,
        'entropy_confidence': entropy_confidence,
        'stability': stability,
        'pattern_score': pattern_score,
        'pattern_confidence': pattern_confidence,
        'reasonableness': reasonableness,
        'z_score': z_score,
        'recent_mean': recent_mean,
        'recent_std': recent_std,
        'final_confidence': confidence
    }
    
    return min(0.95, max(0.05, confidence)), metrics

# === STABILITY METRICS ===
def measure_sequence_stability(sequence: List[float], window_sizes: List[int] = [10, 20, 50]) -> Dict:
    """Measure stability across multiple time scales"""
    if len(sequence) < max(window_sizes):
        return {'status': 'insufficient_data'}
    
    stability_metrics = {}
    
    for window_size in window_sizes:
        if len(sequence) >= window_size * 2:
            # Calculate moving statistics
            windows = [sequence[i:i+window_size] for i in range(len(sequence)-window_size+1)]
            
            moving_means = [np.mean(w) for w in windows]
            moving_stds = [np.std(w) for w in windows]
            moving_entropies = [shannon_entropy(w) for w in windows]
            
            stability_metrics[f'window_{window_size}'] = {
                'mean_stability': 1.0 / (1.0 + np.var(moving_means)),
                'std_stability': 1.0 / (1.0 + np.var(moving_stds)),
                'entropy_stability': 1.0 / (1.0 + np.var(moving_entropies)),
                'mean_trend': np.polyfit(range(len(moving_means)), moving_means, 1)[0],
                'overall_stability': np.mean([
                    1.0 / (1.0 + np.var(moving_means)),
                    1.0 / (1.0 + np.var(moving_stds)),
                    1.0 / (1.0 + np.var(moving_entropies))
                ])
            }
    
    # Overall stability score
    if stability_metrics:
        overall_scores = [metrics['overall_stability'] for metrics in stability_metrics.values()]
        stability_metrics['combined_stability'] = np.mean(overall_scores)
    
    return stability_metrics

# === VISUALIZATION UTILITIES ===
def plot_pattern_analysis(sequence: List[float], save_path: Optional[str] = None) -> None:
    """Create comprehensive pattern analysis visualization"""
    if len(sequence) < 20:
        print("❌ Insufficient data for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stake Oracle Pattern Analysis', fontsize=16)
    
    # 1. Time series
    axes[0, 0].plot(sequence, alpha=0.7, color='blue')
    axes[0, 0].set_title('Sequence Time Series')
    axes[0, 0].set_xlabel('Roll Number')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution
    axes[0, 1].hist(sequence, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Value Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Moving entropy
    if len(sequence) >= 50:
        window_size = min(20, len(sequence) // 5)
        entropies = [shannon_entropy(sequence[i:i+window_size]) 
                    for i in range(len(sequence)-window_size+1)]
        axes[0, 2].plot(entropies, color='red', alpha=0.7)
        axes[0, 2].set_title(f'Moving Entropy (window={window_size})')
        axes[0, 2].set_xlabel('Position')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Autocorrelation
    if len(sequence) >= 30:
        max_lag = min(20, len(sequence) // 3)
        autocorr = [np.corrcoef(sequence[:-lag], sequence[lag:])[0, 1] 
                   for lag in range(1, max_lag) if lag < len(sequence)]
        axes[1, 0].plot(range(1, len(autocorr)+1), autocorr, 'o-', color='purple')
        axes[1, 0].set_title('Autocorrelation')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. Streaks visualization
    streaks = detect_streaks(sequence)
    if streaks:
        streak_starts = [s['start'] for s in streaks]
        streak_lengths = [s['length'] for s in streaks]
        axes[1, 1].scatter(streak_starts, streak_lengths, alpha=0.7, color='orange')
        axes[1, 1].set_title('Detected Streaks')
        axes[1, 1].set_xlabel('Start Position')
        axes[1, 1].set_ylabel('Streak Length')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Stability metrics
    stability = measure_sequence_stability(sequence)
    if 'combined_stability' in stability:
        windows = [int(k.split('_')[1]) for k in stability.keys() if k.startswith('window_')]
        stabilities = [stability[f'window_{w}']['overall_stability'] for w in windows]
        axes[1, 2].plot(windows, stabilities, 'o-', color='brown')
        axes[1, 2].set_title('Stability vs Window Size')
        axes[1, 2].set_xlabel('Window Size')
        axes[1, 2].set_ylabel('Stability Score')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pattern analysis saved to {save_path}")
    else:
        plt.show()

# === COMPREHENSIVE ANALYSIS FUNCTION ===
def comprehensive_pattern_analysis(sequence: List[float]) -> Dict:
    """
    Perform complete pattern analysis on sequence
    
    Args:
        sequence: Input sequence to analyze
    
    Returns:
        Dictionary with all analysis results
    """
    if len(sequence) < 10:
        return {'status': 'insufficient_data', 'message': 'Need at least 10 data points'}
    
    print("🔍 Running comprehensive pattern analysis...")
    
    analysis = {
        'basic_stats': {
            'length': len(sequence),
            'mean': np.mean(sequence),
            'std': np.std(sequence),
            'min': np.min(sequence),
            'max': np.max(sequence),
            'range': np.max(sequence) - np.min(sequence)
        }
    }
    
    # Streak analysis
    streaks = detect_streaks(sequence)
    analysis['streaks'] = analyze_streak_patterns(streaks)
    
    # Entropy analysis
    analysis['entropy'] = {
        'shannon': shannon_entropy(sequence),
        'conditional_1': conditional_entropy(sequence, lag=1) if len(sequence) > 1 else 0,
        'conditional_5': conditional_entropy(sequence, lag=5) if len(sequence) > 5 else 0,
    }
    
    # TDA analysis
    analysis['topology'] = analyze_tda_holes(sequence)
    
    # Stability analysis
    analysis['stability'] = measure_sequence_stability(sequence)
    
    # Confidence for mean prediction
    mean_pred = np.mean(sequence)
    conf, conf_metrics = calculate_prediction_confidence(sequence, mean_pred)
    analysis['confidence'] = {
        'for_mean_prediction': conf,
        'metrics': conf_metrics
    }
    
    # Summary score
    scores = []
    if 'combined_stability' in analysis['stability']:
        scores.append(analysis['stability']['combined_stability'])
    scores.append(1 - analysis['entropy']['shannon'])  # Lower entropy = higher score
    scores.append(conf)
    
    analysis['overall_score'] = np.mean(scores) if scores else 0.5
    analysis['status'] = 'complete'
    
    print("✅ Comprehensive analysis complete")
    return analysis

if __name__ == "__main__":
    # Demo the utilities
    print("🎯 STAKE ORACLE v10 — SUPPORT UTILITIES DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    sample_sequence = []
    
    # Add some patterns for testing
    for i in range(200):
        if i < 50:
            # Random walk
            value = 50 + np.cumsum(np.random.randn(1) * 2)[0]
        elif i < 100:
            # Sine wave with noise
            value = 50 + 20 * np.sin(i * 0.2) + np.random.randn() * 5
        elif i < 150:
            # Streaky behavior
            if i % 10 < 5:
                value = 30 + np.random.randn() * 3
            else:
                value = 70 + np.random.randn() * 3
        else:
            # Random
            value = np.random.uniform(0, 100)
        
        sample_sequence.append(max(0, min(100, value)))
    
    # Run comprehensive analysis
    results = comprehensive_pattern_analysis(sample_sequence)
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"   Overall Score: {results['overall_score']:.3f}")
    print(f"   Streaks Found: {results['streaks']['total_streaks']}")
    print(f"   Shannon Entropy: {results['entropy']['shannon']:.3f}")
    print(f"   TDA Holes: {results['topology']['total_holes']}")
    
    if 'combined_stability' in results['stability']:
        print(f"   Combined Stability: {results['stability']['combined_stability']:.3f}")
    
    print(f"   Confidence Score: {results['confidence']['for_mean_prediction']:.3f}")
    
    print("\n🎉 Support utilities demo complete!")