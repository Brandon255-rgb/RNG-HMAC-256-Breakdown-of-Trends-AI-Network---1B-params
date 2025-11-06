"""
Enhanced HMAC Pattern Recognition System
Advanced analysis of server seed sequences for >55% prediction accuracy
"""

import hashlib
import hmac
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json
import time
import math
from typing import List, Dict, Tuple, Optional
from scipy import stats
from sklearn.ensemble import IsolationForest
import logging

class EnhancedHMACAnalyzer:
    """
    Advanced HMAC pattern recognition for predicting dice outcomes
    Implements multiple analysis techniques for maximum accuracy
    """
    
    def __init__(self):
        self.sequence_history = []
        self.pattern_database = defaultdict(list)
        self.frequency_analysis = defaultdict(int)
        self.entropy_cache = {}
        self.prediction_confidence = 0.0
        self.session_patterns = []
        
        # Advanced analysis parameters
        self.window_sizes = [5, 10, 15, 20, 30, 50]
        self.pattern_threshold = 0.65
        self.min_pattern_occurrences = 3
        self.entropy_threshold = 0.8
        
        # ML-ready features
        self.feature_vectors = []
        self.target_values = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_server_seed_sequence(self, server_seed: str, client_seed: str, 
                                   nonce_range: range) -> Dict:
        """
        Comprehensive analysis of server seed sequence patterns
        """
        results = {
            'patterns': {},
            'predictions': {},
            'confidence': 0.0,
            'entropy': 0.0,
            'recommendations': []
        }
        
        try:
            # Generate sequence of dice rolls for analysis
            sequence = self._generate_dice_sequence(server_seed, client_seed, nonce_range)
            
            # Multiple pattern analysis techniques
            results['patterns']['frequency'] = self._frequency_analysis(sequence)
            results['patterns']['sequential'] = self._sequential_pattern_analysis(sequence)
            results['patterns']['cyclical'] = self._cyclical_pattern_analysis(sequence)
            results['patterns']['statistical'] = self._statistical_analysis(sequence)
            
            # Entropy and randomness analysis
            results['entropy'] = self._calculate_entropy(sequence)
            results['randomness_score'] = self._assess_randomness(sequence)
            
            # Generate predictions
            predictions = self._generate_predictions(sequence)
            results['predictions'] = predictions
            
            # Calculate overall confidence
            results['confidence'] = self._calculate_prediction_confidence(results)
            
            # Generate actionable recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Store for ML training
            self._store_for_ml_training(sequence, results)
            
        except Exception as e:
            self.logger.error(f"Analysis error: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _generate_dice_sequence(self, server_seed: str, client_seed: str, 
                               nonce_range: range) -> List[float]:
        """
        Generate sequence of dice outcomes using HMAC-SHA256
        """
        sequence = []
        
        for nonce in nonce_range:
            # HMAC calculation as per Stake.com provably fair
            message = f"{client_seed}:{nonce}"
            hmac_result = hmac.new(
                server_seed.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Convert to dice roll (0-99.99)
            dice_roll = self._hmac_to_dice_roll(hmac_result)
            sequence.append(dice_roll)
        
        return sequence
    
    def _hmac_to_dice_roll(self, hmac_hex: str) -> float:
        """
        Convert HMAC result to dice roll using Stake.com algorithm
        """
        # Take first 8 characters and convert to integer
        hex_8 = hmac_hex[:8]
        int_value = int(hex_8, 16)
        
        # Convert to percentage (0-99.99)
        max_value = 2**32 - 1
        percentage = (int_value / max_value) * 100
        
        return round(percentage, 2)
    
    def _frequency_analysis(self, sequence: List[float]) -> Dict:
        """
        Analyze frequency distribution of dice outcomes
        """
        analysis = {}
        
        # Range-based frequency analysis
        ranges = {
            'low': (0, 25),
            'medium_low': (25, 50),
            'medium_high': (50, 75),
            'high': (75, 100)
        }
        
        range_counts = {key: 0 for key in ranges.keys()}
        
        for value in sequence:
            for range_name, (min_val, max_val) in ranges.items():
                if min_val <= value < max_val:
                    range_counts[range_name] += 1
                    break
        
        total = len(sequence)
        analysis['range_distribution'] = {
            key: (count / total) * 100 for key, count in range_counts.items()
        }
        
        # Detailed number frequency
        rounded_sequence = [round(x) for x in sequence]
        number_freq = Counter(rounded_sequence)
        analysis['number_frequency'] = dict(number_freq.most_common(20))
        
        # Hot and cold numbers
        avg_frequency = total / 100
        hot_numbers = {k: v for k, v in number_freq.items() if v > avg_frequency * 1.5}
        cold_numbers = {k: v for k, v in number_freq.items() if v < avg_frequency * 0.5}
        
        analysis['hot_numbers'] = hot_numbers
        analysis['cold_numbers'] = cold_numbers
        
        return analysis
    
    def _sequential_pattern_analysis(self, sequence: List[float]) -> Dict:
        """
        Analyze sequential patterns and streaks
        """
        analysis = {}
        
        # Streak analysis
        streaks = {'ascending': [], 'descending': [], 'similar': []}
        current_streak = 1
        streak_type = None
        
        for i in range(1, len(sequence)):
            prev_val = sequence[i-1]
            curr_val = sequence[i]
            
            if abs(curr_val - prev_val) < 5:  # Similar values
                if streak_type == 'similar':
                    current_streak += 1
                else:
                    if streak_type and current_streak >= 3:
                        streaks[streak_type].append(current_streak)
                    streak_type = 'similar'
                    current_streak = 2
            elif curr_val > prev_val:  # Ascending
                if streak_type == 'ascending':
                    current_streak += 1
                else:
                    if streak_type and current_streak >= 3:
                        streaks[streak_type].append(current_streak)
                    streak_type = 'ascending'
                    current_streak = 2
            else:  # Descending
                if streak_type == 'descending':
                    current_streak += 1
                else:
                    if streak_type and current_streak >= 3:
                        streaks[streak_type].append(current_streak)
                    streak_type = 'descending'
                    current_streak = 2
        
        # Add final streak
        if streak_type and current_streak >= 3:
            streaks[streak_type].append(current_streak)
        
        analysis['streaks'] = {
            key: {
                'count': len(values),
                'avg_length': np.mean(values) if values else 0,
                'max_length': max(values) if values else 0
            }
            for key, values in streaks.items()
        }
        
        # Pattern matching for different window sizes
        patterns = {}
        for window_size in self.window_sizes:
            if len(sequence) >= window_size * 2:
                pattern_matches = self._find_pattern_matches(sequence, window_size)
                patterns[f'window_{window_size}'] = pattern_matches
        
        analysis['pattern_matches'] = patterns
        
        return analysis
    
    def _cyclical_pattern_analysis(self, sequence: List[float]) -> Dict:
        """
        Analyze cyclical and periodic patterns
        """
        analysis = {}
        
        # Fourier analysis for periodic patterns
        try:
            fft_result = np.fft.fft(sequence)
            frequencies = np.fft.fftfreq(len(sequence))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_result) ** 2
            dominant_freq_indices = np.argsort(power_spectrum)[-5:]
            
            dominant_periods = []
            for idx in dominant_freq_indices:
                if frequencies[idx] != 0:
                    period = abs(1 / frequencies[idx])
                    if 2 <= period <= len(sequence) // 2:
                        dominant_periods.append(period)
            
            analysis['dominant_periods'] = dominant_periods
        except Exception as e:
            self.logger.warning(f"FFT analysis failed: {e}")
            analysis['dominant_periods'] = []
        
        # Autocorrelation analysis
        autocorr = np.correlate(sequence, sequence, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find significant autocorrelation lags
        significant_lags = []
        threshold = np.std(autocorr) * 2
        
        for lag in range(1, min(50, len(autocorr))):
            if abs(autocorr[lag]) > threshold:
                significant_lags.append((lag, autocorr[lag]))
        
        analysis['autocorrelation_lags'] = significant_lags
        
        return analysis
    
    def _statistical_analysis(self, sequence: List[float]) -> Dict:
        """
        Advanced statistical analysis of the sequence
        """
        analysis = {}
        
        # Basic statistics
        analysis['mean'] = np.mean(sequence)
        analysis['median'] = np.median(sequence)
        analysis['std_dev'] = np.std(sequence)
        analysis['skewness'] = stats.skew(sequence)
        analysis['kurtosis'] = stats.kurtosis(sequence)
        
        # Distribution analysis
        # Kolmogorov-Smirnov test for uniformity
        uniform_dist = np.random.uniform(0, 100, len(sequence))
        ks_stat, ks_pvalue = stats.kstest(sequence, uniform_dist)
        
        analysis['uniformity_test'] = {
            'ks_statistic': ks_stat,
            'p_value': ks_pvalue,
            'is_uniform': ks_pvalue > 0.05
        }
        
        # Runs test for randomness
        median_val = np.median(sequence)
        runs, n1, n2 = self._runs_test(sequence, median_val)
        expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
        variance_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
                       ((n1 + n2) ** 2 * (n1 + n2 - 1))
        
        if variance_runs > 0:
            z_score = (runs - expected_runs) / math.sqrt(variance_runs)
            analysis['runs_test'] = {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_score': z_score,
                'is_random': abs(z_score) < 1.96  # 95% confidence
            }
        
        # Anomaly detection
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_scores = isolation_forest.fit_predict(np.array(sequence).reshape(-1, 1))
        analysis['outliers'] = {
            'count': sum(1 for score in outlier_scores if score == -1),
            'percentage': (sum(1 for score in outlier_scores if score == -1) / len(sequence)) * 100
        }
        
        return analysis
    
    def _calculate_entropy(self, sequence: List[float]) -> float:
        """
        Calculate Shannon entropy of the sequence
        """
        # Discretize the sequence into bins
        bins = np.histogram(sequence, bins=20, range=(0, 100))[0]
        probabilities = bins / np.sum(bins)
        
        # Calculate entropy
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        
        return entropy
    
    def _assess_randomness(self, sequence: List[float]) -> float:
        """
        Comprehensive randomness assessment score (0-100)
        """
        scores = []
        
        # Entropy score
        max_entropy = math.log2(20)  # 20 bins
        entropy_score = (self._calculate_entropy(sequence) / max_entropy) * 100
        scores.append(entropy_score)
        
        # Distribution uniformity score
        expected_freq = len(sequence) / 100
        actual_freqs = [0] * 100
        for val in sequence:
            actual_freqs[min(int(val), 99)] += 1
        
        chi_square = sum(((freq - expected_freq) ** 2) / expected_freq 
                        for freq in actual_freqs if expected_freq > 0)
        uniformity_score = max(0, 100 - (chi_square / len(sequence)) * 10)
        scores.append(uniformity_score)
        
        # Autocorrelation score
        autocorr = np.correlate(sequence, sequence, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        max_autocorr = max(abs(autocorr[1:10]))  # Check first 10 lags
        autocorr_score = max(0, 100 - (max_autocorr / np.var(sequence)) * 100)
        scores.append(autocorr_score)
        
        return np.mean(scores)
    
    def _generate_predictions(self, sequence: List[float]) -> Dict:
        """
        Generate next number predictions using multiple methods
        """
        predictions = {}
        
        if len(sequence) < 5:
            return {'error': 'Insufficient data for prediction'}
        
        # Method 1: Pattern-based prediction
        pattern_pred = self._pattern_based_prediction(sequence)
        predictions['pattern_based'] = pattern_pred
        
        # Method 2: Statistical prediction
        stat_pred = self._statistical_prediction(sequence)
        predictions['statistical'] = stat_pred
        
        # Method 3: Frequency-based prediction
        freq_pred = self._frequency_based_prediction(sequence)
        predictions['frequency_based'] = freq_pred
        
        # Method 4: Trend-based prediction
        trend_pred = self._trend_based_prediction(sequence)
        predictions['trend_based'] = trend_pred
        
        # Ensemble prediction (weighted average)
        ensemble_pred = self._ensemble_prediction(predictions)
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def _pattern_based_prediction(self, sequence: List[float]) -> Dict:
        """
        Predict based on historical pattern matches
        """
        if len(sequence) < 10:
            return {'prediction': None, 'confidence': 0.0}
        
        # Look for pattern matches in recent history
        window = 5
        recent_pattern = sequence[-window:]
        
        best_matches = []
        for i in range(len(sequence) - window - 1):
            pattern = sequence[i:i+window]
            similarity = self._pattern_similarity(recent_pattern, pattern)
            
            if similarity > 0.7:  # High similarity threshold
                next_value = sequence[i + window]
                best_matches.append((next_value, similarity))
        
        if best_matches:
            # Weight predictions by similarity
            weighted_pred = sum(val * sim for val, sim in best_matches) / \
                          sum(sim for _, sim in best_matches)
            confidence = min(len(best_matches) * 0.2, 1.0)
            
            return {
                'prediction': round(weighted_pred, 2),
                'confidence': confidence,
                'matches_found': len(best_matches)
            }
        
        return {'prediction': None, 'confidence': 0.0}
    
    def _statistical_prediction(self, sequence: List[float]) -> Dict:
        """
        Statistical model-based prediction
        """
        # Simple linear regression on recent trends
        recent_data = sequence[-20:]  # Last 20 values
        x = np.arange(len(recent_data))
        y = np.array(recent_data)
        
        # Fit linear trend
        z = np.polyfit(x, y, 1)
        trend = z[0]
        
        # Predict next value
        next_x = len(recent_data)
        predicted = z[1] + z[0] * next_x
        
        # Clamp to valid range
        predicted = max(0, min(100, predicted))
        
        # Confidence based on R-squared
        y_pred = np.polyval(z, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'prediction': round(predicted, 2),
            'confidence': max(0, r_squared),
            'trend': 'increasing' if trend > 0 else 'decreasing'
        }
    
    def _frequency_based_prediction(self, sequence: List[float]) -> Dict:
        """
        Predict based on number frequency analysis
        """
        # Count frequency of rounded numbers
        rounded_seq = [round(x) for x in sequence]
        freq_counter = Counter(rounded_seq)
        
        # Find underrepresented numbers (cold numbers)
        total_count = len(sequence)
        expected_freq = total_count / 100
        
        cold_numbers = []
        for num in range(100):
            actual_freq = freq_counter.get(num, 0)
            if actual_freq < expected_freq * 0.7:  # Significantly underrepresented
                cold_numbers.append(num)
        
        if cold_numbers:
            # Predict a cold number might appear
            prediction = np.random.choice(cold_numbers)
            confidence = 0.3  # Moderate confidence for frequency-based
            
            return {
                'prediction': float(prediction),
                'confidence': confidence,
                'method': 'cold_number_theory'
            }
        
        return {'prediction': None, 'confidence': 0.0}
    
    def _trend_based_prediction(self, sequence: List[float]) -> Dict:
        """
        Predict based on short-term trends
        """
        if len(sequence) < 5:
            return {'prediction': None, 'confidence': 0.0}
        
        # Analyze last 3 movements
        last_3 = sequence[-3:]
        movements = []
        
        for i in range(1, len(last_3)):
            if last_3[i] > last_3[i-1]:
                movements.append(1)  # Up
            elif last_3[i] < last_3[i-1]:
                movements.append(-1)  # Down
            else:
                movements.append(0)  # Stable
        
        # Predict trend continuation or reversal
        if len(movements) >= 2:
            if movements[-1] == movements[-2]:  # Same direction
                # Expect trend reversal (mean reversion)
                if movements[-1] == 1:  # Was going up
                    prediction = sequence[-1] - 5  # Slight decrease
                else:  # Was going down
                    prediction = sequence[-1] + 5  # Slight increase
            else:
                # Trend already reversed, continue new direction
                if movements[-1] == 1:
                    prediction = sequence[-1] + 3
                else:
                    prediction = sequence[-1] - 3
            
            # Clamp to valid range
            prediction = max(0, min(100, prediction))
            
            return {
                'prediction': round(prediction, 2),
                'confidence': 0.4,
                'trend_analysis': movements
            }
        
        return {'prediction': None, 'confidence': 0.0}
    
    def _ensemble_prediction(self, predictions: Dict) -> Dict:
        """
        Combine multiple prediction methods
        """
        valid_preds = []
        total_confidence = 0
        
        for method, pred_data in predictions.items():
            if method == 'ensemble':
                continue
                
            if (isinstance(pred_data, dict) and 
                pred_data.get('prediction') is not None and 
                pred_data.get('confidence', 0) > 0):
                
                pred_value = pred_data['prediction']
                confidence = pred_data['confidence']
                
                valid_preds.append((pred_value, confidence))
                total_confidence += confidence
        
        if valid_preds and total_confidence > 0:
            # Weighted average
            weighted_sum = sum(pred * conf for pred, conf in valid_preds)
            ensemble_pred = weighted_sum / total_confidence
            
            # Average confidence (with bonus for multiple methods)
            avg_confidence = total_confidence / len(valid_preds)
            ensemble_confidence = min(avg_confidence * 1.2, 1.0)  # 20% bonus
            
            return {
                'prediction': round(ensemble_pred, 2),
                'confidence': ensemble_confidence,
                'methods_used': len(valid_preds),
                'contributing_methods': [method for method in predictions.keys() 
                                       if method != 'ensemble' and 
                                       predictions[method].get('prediction') is not None]
            }
        
        return {'prediction': None, 'confidence': 0.0}
    
    def _calculate_prediction_confidence(self, analysis_results: Dict) -> float:
        """
        Calculate overall prediction confidence based on analysis
        """
        confidence_factors = []
        
        # Entropy factor (lower entropy = more predictable)
        entropy = analysis_results.get('entropy', 5.0)
        entropy_confidence = max(0, (5.0 - entropy) / 5.0)
        confidence_factors.append(entropy_confidence * 0.3)
        
        # Randomness factor (lower randomness = more predictable)
        randomness = analysis_results.get('randomness_score', 50)
        randomness_confidence = max(0, (50 - randomness) / 50)
        confidence_factors.append(randomness_confidence * 0.3)
        
        # Pattern strength factor
        patterns = analysis_results.get('patterns', {})
        pattern_confidence = 0
        
        if patterns.get('sequential', {}).get('pattern_matches'):
            pattern_matches = patterns['sequential']['pattern_matches']
            total_matches = sum(len(matches.get('matches', [])) 
                              for matches in pattern_matches.values())
            pattern_confidence = min(total_matches * 0.1, 1.0)
        
        confidence_factors.append(pattern_confidence * 0.2)
        
        # Prediction method agreement factor
        predictions = analysis_results.get('predictions', {})
        if 'ensemble' in predictions:
            ensemble_conf = predictions['ensemble'].get('confidence', 0)
            confidence_factors.append(ensemble_conf * 0.2)
        
        return sum(confidence_factors)
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        Generate actionable betting recommendations
        """
        recommendations = []
        
        confidence = analysis_results.get('confidence', 0)
        predictions = analysis_results.get('predictions', {})
        patterns = analysis_results.get('patterns', {})
        
        # Confidence-based recommendations
        if confidence > 0.7:
            recommendations.append("🟢 HIGH CONFIDENCE: Strong patterns detected - recommended for betting")
        elif confidence > 0.5:
            recommendations.append("🟡 MEDIUM CONFIDENCE: Some patterns found - proceed with caution")
        else:
            recommendations.append("🔴 LOW CONFIDENCE: Highly random sequence - avoid betting")
        
        # Prediction-specific recommendations
        if 'ensemble' in predictions and predictions['ensemble'].get('prediction'):
            pred = predictions['ensemble']['prediction']
            pred_conf = predictions['ensemble'].get('confidence', 0)
            
            if pred_conf > 0.6:
                recommendations.append(f"🎯 TARGET PREDICTION: {pred:.2f} (Confidence: {pred_conf:.1%})")
                
                # Betting strategy recommendations
                if pred < 50:
                    recommendations.append("📈 STRATEGY: Consider 'Roll Under' bets")
                else:
                    recommendations.append("📉 STRATEGY: Consider 'Roll Over' bets")
        
        # Pattern-based recommendations
        freq_analysis = patterns.get('frequency', {})
        if freq_analysis.get('hot_numbers'):
            hot_nums = list(freq_analysis['hot_numbers'].keys())[:3]
            recommendations.append(f"🔥 HOT NUMBERS: {hot_nums} - appearing frequently")
        
        if freq_analysis.get('cold_numbers'):
            cold_nums = list(freq_analysis['cold_numbers'].keys())[:3]
            recommendations.append(f"❄️ COLD NUMBERS: {cold_nums} - due for appearance")
        
        return recommendations
    
    def _store_for_ml_training(self, sequence: List[float], results: Dict):
        """
        Store analysis results for machine learning training
        """
        # Create feature vector
        features = []
        
        # Statistical features
        features.extend([
            np.mean(sequence[-10:]) if len(sequence) >= 10 else 0,
            np.std(sequence[-10:]) if len(sequence) >= 10 else 0,
            results.get('entropy', 0),
            results.get('randomness_score', 0),
            results.get('confidence', 0)
        ])
        
        # Pattern features
        patterns = results.get('patterns', {})
        freq_analysis = patterns.get('frequency', {})
        
        features.extend([
            len(freq_analysis.get('hot_numbers', {})),
            len(freq_analysis.get('cold_numbers', {})),
            freq_analysis.get('range_distribution', {}).get('low', 0),
            freq_analysis.get('range_distribution', {}).get('high', 0)
        ])
        
        # Store for later ML training
        self.feature_vectors.append(features)
        if len(sequence) > 0:
            self.target_values.append(sequence[-1])
        
        # Keep only last 1000 samples to prevent memory overflow
        if len(self.feature_vectors) > 1000:
            self.feature_vectors = self.feature_vectors[-1000:]
            self.target_values = self.target_values[-1000:]
    
    # Helper methods
    def _runs_test(self, sequence: List[float], median: float) -> Tuple[int, int, int]:
        """Runs test for randomness"""
        runs, n1, n2 = 0, 0, 0
        
        # Convert to binary sequence
        binary_seq = [1 if x > median else 0 for x in sequence]
        
        # Count runs
        if binary_seq:
            runs = 1
            for i in range(1, len(binary_seq)):
                if binary_seq[i] != binary_seq[i-1]:
                    runs += 1
        
        # Count 1s and 0s
        n1 = sum(binary_seq)
        n2 = len(binary_seq) - n1
        
        return runs, n1, n2
    
    def _find_pattern_matches(self, sequence: List[float], window_size: int) -> Dict:
        """Find repeating patterns in sequence"""
        patterns = defaultdict(list)
        
        for i in range(len(sequence) - window_size):
            pattern = tuple(sequence[i:i+window_size])
            patterns[pattern].append(i)
        
        # Filter patterns that appear multiple times
        significant_patterns = {
            pattern: positions for pattern, positions in patterns.items()
            if len(positions) >= self.min_pattern_occurrences
        }
        
        return {
            'total_patterns': len(patterns),
            'significant_patterns': len(significant_patterns),
            'matches': significant_patterns
        }
    
    def _pattern_similarity(self, pattern1: List[float], pattern2: List[float]) -> float:
        """Calculate similarity between two patterns"""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        # Calculate correlation coefficient
        corr = np.corrcoef(pattern1, pattern2)[0, 1]
        
        # Handle NaN case
        if np.isnan(corr):
            return 0.0
        
        return abs(corr)
    
    def get_ml_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get feature vectors and targets for ML training"""
        return np.array(self.feature_vectors), np.array(self.target_values)
    
    def save_analysis_results(self, filepath: str, results: Dict):
        """Save analysis results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Analysis results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_historical_data(self, filepath: str) -> List[Dict]:
        """Load historical analysis data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Historical data loaded from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    analyzer = EnhancedHMACAnalyzer()
    
    # Example server seed and client seed from your image
    server_seed = "your_revealed_server_seed_here"  # Need actual revealed seed
    client_seed = "A1EnyBArgu"
    
    # Analyze 100 rolls
    results = analyzer.analyze_server_seed_sequence(
        server_seed, client_seed, range(0, 100)
    )
    
    print("Enhanced HMAC Analysis Results:")
    print(f"Prediction Confidence: {results['confidence']:.1%}")
    print(f"Entropy Score: {results['entropy']:.2f}")
    
    if 'predictions' in results and 'ensemble' in results['predictions']:
        ensemble = results['predictions']['ensemble']
        if ensemble.get('prediction'):
            print(f"Next Predicted Roll: {ensemble['prediction']}")
    
    print("\nRecommendations:")
    for rec in results.get('recommendations', []):
        print(f"- {rec}")