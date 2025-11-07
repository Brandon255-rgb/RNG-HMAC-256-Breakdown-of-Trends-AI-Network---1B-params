#!/usr/bin/env python3
"""
Oracle Support Utilities
========================
Support functions for Oracle AI pattern analysis and prediction
"""

import numpy as np
import hashlib
import hmac
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
import statistics
import math
from datetime import datetime

def detect_streaks(data: List[float], threshold: float = 50.0) -> List[int]:
    """
    Detect streaks in a series of dice roll results
    
    Args:
        data: List of roll results (0-100)
        threshold: Threshold for over/under classification (default 50.0)
    
    Returns:
        List of streak lengths (positive for over streaks, negative for under)
    """
    if not data or len(data) < 2:
        return []
    
    streaks = []
    current_streak = 1
    current_type = "over" if data[0] > threshold else "under"
    
    for i in range(1, len(data)):
        roll_type = "over" if data[i] > threshold else "under"
        
        if roll_type == current_type:
            current_streak += 1
        else:
            # End of streak
            streak_value = current_streak if current_type == "over" else -current_streak
            streaks.append(streak_value)
            
            # Start new streak
            current_streak = 1
            current_type = roll_type
    
    # Add final streak
    streak_value = current_streak if current_type == "over" else -current_streak
    streaks.append(streak_value)
    
    return streaks

def shannon_entropy(data: List[float], bins: int = 10) -> float:
    """
    Calculate Shannon entropy of data distribution
    
    Args:
        data: List of values to analyze
        bins: Number of bins for histogram (default 10)
    
    Returns:
        Shannon entropy value
    """
    if not data:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(data, bins=bins, range=(0, 100))
    
    # Calculate probabilities
    total = len(data)
    probabilities = hist / total
    
    # Remove zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    # Normalize to 0-1 range
    max_entropy = np.log2(bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy

def calculate_bias_score(data: List[float], threshold: float = 50.0) -> float:
    """
    Calculate bias score indicating over/under tendency
    
    Args:
        data: List of roll results (0-100)
        threshold: Threshold for bias calculation (default 50.0)
    
    Returns:
        Bias score (-1 to 1, negative = under bias, positive = over bias)
    """
    if not data:
        return 0.0
    
    over_count = sum(1 for x in data if x > threshold)
    total_count = len(data)
    
    if total_count == 0:
        return 0.0
    
    over_percentage = over_count / total_count
    expected_percentage = 0.5  # 50% for fair dice
    
    # Calculate bias (-1 to 1)
    bias = (over_percentage - expected_percentage) * 2
    
    return bias

def detect_patterns(data: List[float], window_size: int = 10) -> Dict[str, Any]:
    """
    Detect various patterns in dice roll data
    
    Args:
        data: List of roll results
        window_size: Size of analysis window
    
    Returns:
        Dictionary containing pattern analysis results
    """
    if len(data) < window_size:
        return {"patterns_found": False, "reason": "Insufficient data"}
    
    patterns = {
        "patterns_found": True,
        "streak_analysis": {},
        "frequency_analysis": {},
        "trend_analysis": {},
        "volatility_analysis": {}
    }
    
    # Streak analysis
    streaks = detect_streaks(data)
    if streaks:
        patterns["streak_analysis"] = {
            "longest_over_streak": max([s for s in streaks if s > 0], default=0),
            "longest_under_streak": abs(min([s for s in streaks if s < 0], default=0)),
            "total_streaks": len(streaks),
            "average_streak_length": statistics.mean([abs(s) for s in streaks])
        }
    
    # Frequency analysis
    recent_data = data[-window_size:]
    patterns["frequency_analysis"] = {
        "over_frequency": len([x for x in recent_data if x > 50]) / len(recent_data),
        "under_frequency": len([x for x in recent_data if x <= 50]) / len(recent_data),
        "average_roll": statistics.mean(recent_data),
        "median_roll": statistics.median(recent_data),
        "std_deviation": statistics.stdev(recent_data) if len(recent_data) > 1 else 0
    }
    
    # Trend analysis (simple linear trend)
    if len(data) >= 5:
        x = np.arange(len(data))
        y = np.array(data)
        trend_slope = np.polyfit(x, y, 1)[0]
        
        patterns["trend_analysis"] = {
            "trend_direction": "increasing" if trend_slope > 0 else "decreasing",
            "trend_strength": abs(trend_slope),
            "is_trending": abs(trend_slope) > 0.1
        }
    
    # Volatility analysis
    if len(data) >= 3:
        rolling_changes = [abs(data[i] - data[i-1]) for i in range(1, len(data))]
        patterns["volatility_analysis"] = {
            "average_change": statistics.mean(rolling_changes),
            "volatility_score": statistics.stdev(rolling_changes) if len(rolling_changes) > 1 else 0,
            "max_change": max(rolling_changes),
            "is_volatile": statistics.stdev(rolling_changes) > 10 if len(rolling_changes) > 1 else False
        }
    
    return patterns

def hmac_to_dice_result(server_seed: str, client_seed: str, nonce: int) -> float:
    """
    Convert HMAC-SHA256 result to dice roll (0-99.99)
    Implements provably fair algorithm used by many gambling sites
    
    Args:
        server_seed: Server seed string
        client_seed: Client seed string
        nonce: Bet nonce/counter
    
    Returns:
        Dice roll result (0-99.99)
    """
    # Create HMAC message
    message = f"{client_seed}:{nonce}"
    
    # Calculate HMAC-SHA256
    hmac_result = hmac.new(
        server_seed.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Convert first 8 hex characters to integer
    hex_substring = hmac_result[:8]
    hex_value = int(hex_substring, 16)
    
    # Convert to 0-99.99 range
    dice_result = (hex_value / (16**8)) * 100
    
    return round(dice_result, 2)

def verify_provably_fair(server_seed: str, server_seed_hash: str, client_seed: str, nonce: int, result: float) -> bool:
    """
    Verify a bet result is provably fair
    
    Args:
        server_seed: Revealed server seed
        server_seed_hash: Original server seed hash
        client_seed: Client seed used
        nonce: Bet nonce
        result: Claimed result
    
    Returns:
        True if result is valid, False otherwise
    """
    # Verify server seed hash
    calculated_hash = hashlib.sha256(server_seed.encode('utf-8')).hexdigest()
    if calculated_hash != server_seed_hash:
        return False
    
    # Calculate expected result
    expected_result = hmac_to_dice_result(server_seed, client_seed, nonce)
    
    # Allow small floating point differences
    return abs(expected_result - result) < 0.01

def analyze_seed_randomness(server_seed_hash: str, client_seed: str, nonce_range: int = 100) -> Dict[str, Any]:
    """
    Analyze randomness of seed combination (using hash only)
    
    Args:
        server_seed_hash: Server seed hash
        client_seed: Client seed
        nonce_range: Number of nonces to analyze
    
    Returns:
        Dictionary with randomness analysis
    """
    analysis = {
        "entropy_score": 0.0,
        "distribution_score": 0.0,
        "pattern_score": 0.0,
        "overall_score": 0.0,
        "recommendations": []
    }
    
    try:
        # Generate sequence using hash (for analysis only)
        sequence = []
        for nonce in range(nonce_range):
            # Create pseudo-random value from hash + nonce
            combined = f"{server_seed_hash}:{client_seed}:{nonce}"
            hash_val = hashlib.sha256(combined.encode()).hexdigest()
            # Convert to 0-100 range
            value = (int(hash_val[:8], 16) / (16**8)) * 100
            sequence.append(value)
        
        # Calculate entropy
        analysis["entropy_score"] = shannon_entropy(sequence)
        
        # Check distribution
        bins = [0] * 10
        for val in sequence:
            bin_idx = min(int(val / 10), 9)
            bins[bin_idx] += 1
        
        # Distribution uniformity score
        expected_per_bin = len(sequence) / 10
        distribution_variance = statistics.variance(bins)
        analysis["distribution_score"] = max(0, 1 - (distribution_variance / (expected_per_bin ** 2)))
        
        # Pattern analysis
        patterns = detect_patterns(sequence)
        if patterns["patterns_found"]:
            streak_score = 1 / (1 + patterns["streak_analysis"].get("average_streak_length", 1))
            analysis["pattern_score"] = streak_score
        else:
            analysis["pattern_score"] = 0.5
        
        # Overall score
        analysis["overall_score"] = (
            analysis["entropy_score"] * 0.4 +
            analysis["distribution_score"] * 0.3 +
            analysis["pattern_score"] * 0.3
        )
        
        # Recommendations
        if analysis["entropy_score"] < 0.7:
            analysis["recommendations"].append("Low entropy detected - consider changing seeds")
        
        if analysis["distribution_score"] < 0.8:
            analysis["recommendations"].append("Uneven distribution - seeds may be predictable")
        
        if analysis["overall_score"] > 0.8:
            analysis["recommendations"].append("Good randomness quality")
        
    except Exception as e:
        analysis["error"] = str(e)
        analysis["recommendations"].append("Analysis failed - check seed format")
    
    return analysis

def calculate_confidence_score(data: List[float], patterns: Dict[str, Any]) -> float:
    """
    Calculate confidence score for predictions based on data and patterns
    
    Args:
        data: Historical roll data
        patterns: Pattern analysis results
    
    Returns:
        Confidence score (0-100)
    """
    if not data or not patterns.get("patterns_found"):
        return 0.0
    
    base_confidence = 30.0  # Base confidence level
    
    # Adjust for data quantity
    data_factor = min(len(data) / 100, 1.0) * 20  # Up to 20 points for data quantity
    
    # Adjust for pattern strength
    pattern_factor = 0.0
    if "streak_analysis" in patterns:
        avg_streak = patterns["streak_analysis"].get("average_streak_length", 1)
        if avg_streak > 3:
            pattern_factor += 15  # Strong streaking pattern
    
    if "trend_analysis" in patterns:
        if patterns["trend_analysis"].get("is_trending"):
            pattern_factor += 10  # Trending pattern
    
    # Adjust for volatility (lower volatility = higher confidence)
    volatility_factor = 0.0
    if "volatility_analysis" in patterns:
        if not patterns["volatility_analysis"].get("is_volatile", True):
            volatility_factor += 15  # Low volatility
    
    # Calculate final confidence
    confidence = base_confidence + data_factor + pattern_factor + volatility_factor
    
    return min(confidence, 95.0)  # Cap at 95%

def generate_prediction(data: List[float], patterns: Dict[str, Any], current_nonce: int = 0) -> Dict[str, Any]:
    """
    Generate prediction for next dice roll based on patterns
    
    Args:
        data: Historical roll data
        patterns: Pattern analysis results
        current_nonce: Current nonce value
    
    Returns:
        Dictionary with prediction details
    """
    prediction = {
        "predicted_value": 50.0,
        "confidence": 0.0,
        "reasoning": "No data available",
        "bias_direction": "neutral",
        "suggested_bet": "none"
    }
    
    if not data or len(data) < 5:
        return prediction
    
    try:
        # Calculate basic statistics
        recent_avg = statistics.mean(data[-10:]) if len(data) >= 10 else statistics.mean(data)
        overall_avg = statistics.mean(data)
        bias_score = calculate_bias_score(data)
        
        # Base prediction on recent trend
        if len(data) >= 3:
            trend = np.polyfit(range(len(data[-10:])), data[-10:], 1)[0] if len(data) >= 10 else 0
            predicted_value = recent_avg + trend
        else:
            predicted_value = recent_avg
        
        # Adjust for bias
        if abs(bias_score) > 0.1:
            if bias_score > 0:  # Over bias
                predicted_value = max(predicted_value, 55)
                prediction["bias_direction"] = "over"
            else:  # Under bias
                predicted_value = min(predicted_value, 45)
                prediction["bias_direction"] = "under"
        
        # Ensure realistic range
        predicted_value = max(5, min(95, predicted_value))
        
        # Calculate confidence
        confidence = calculate_confidence_score(data, patterns)
        
        # Generate reasoning
        reasoning_parts = []
        if abs(bias_score) > 0.15:
            reasoning_parts.append(f"Strong {prediction['bias_direction']} bias detected")
        
        if patterns.get("trend_analysis", {}).get("is_trending"):
            direction = patterns["trend_analysis"]["trend_direction"]
            reasoning_parts.append(f"Following {direction} trend")
        
        if not reasoning_parts:
            reasoning_parts.append("Based on statistical analysis")
        
        reasoning = "; ".join(reasoning_parts)
        
        # Suggest bet strategy
        if confidence > 70 and abs(predicted_value - 50) > 5:
            if predicted_value > 55:
                prediction["suggested_bet"] = "over 55"
            elif predicted_value < 45:
                prediction["suggested_bet"] = "under 45"
            else:
                prediction["suggested_bet"] = "low confidence bet"
        else:
            prediction["suggested_bet"] = "hold"
        
        prediction.update({
            "predicted_value": round(predicted_value, 2),
            "confidence": round(confidence, 1),
            "reasoning": reasoning
        })
        
    except Exception as e:
        prediction["reasoning"] = f"Prediction error: {str(e)}"
    
    return prediction

def format_analysis_summary(data: List[float], patterns: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    """
    Format a readable analysis summary
    
    Args:
        data: Roll data
        patterns: Pattern analysis
        prediction: Prediction results
    
    Returns:
        Formatted summary string
    """
    if not data:
        return "No data available for analysis"
    
    summary_parts = []
    
    # Data overview
    summary_parts.append(f"Analyzed {len(data)} rolls")
    summary_parts.append(f"Average: {statistics.mean(data):.2f}")
    
    # Bias information
    bias_score = calculate_bias_score(data)
    if abs(bias_score) > 0.1:
        direction = "over" if bias_score > 0 else "under"
        strength = "strong" if abs(bias_score) > 0.2 else "moderate"
        summary_parts.append(f"{strength} {direction} bias ({bias_score:.2f})")
    
    # Pattern information
    if patterns.get("patterns_found"):
        if "streak_analysis" in patterns:
            avg_streak = patterns["streak_analysis"].get("average_streak_length", 0)
            summary_parts.append(f"Avg streak: {avg_streak:.1f}")
    
    # Prediction
    summary_parts.append(f"Prediction: {prediction['predicted_value']:.2f}")
    summary_parts.append(f"Confidence: {prediction['confidence']:.1f}%")
    
    return " | ".join(summary_parts)

# Betting Strategy Utilities

def calculate_optimal_bet_size(bankroll: float, confidence: float, edge: float = 0.01) -> float:
    """
    Calculate optimal bet size using Kelly Criterion
    
    Args:
        bankroll: Current bankroll
        confidence: Confidence in prediction (0-1)
        edge: Expected edge (default 1%)
    
    Returns:
        Recommended bet size
    """
    if confidence <= 0.5 or edge <= 0:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # Where b = odds-1, p = confidence, q = 1-confidence
    
    # Assume we're betting with slight edge
    win_probability = confidence
    lose_probability = 1 - confidence
    
    if win_probability <= lose_probability:
        return 0.0
    
    # Conservative Kelly fraction (25% of full Kelly)
    kelly_fraction = ((win_probability * (1 + edge)) - lose_probability) / (1 + edge)
    conservative_kelly = kelly_fraction * 0.25
    
    # Never bet more than 5% of bankroll
    max_bet_percentage = 0.05
    
    bet_percentage = min(conservative_kelly, max_bet_percentage)
    bet_size = bankroll * bet_percentage
    
    return max(bet_size, 0.0)

def assess_risk_level(confidence: float, volatility: float, streak_length: int = 0) -> str:
    """
    Assess risk level for betting decision
    
    Args:
        confidence: Confidence score (0-100)
        volatility: Volatility score (0-100)
        streak_length: Current streak length
    
    Returns:
        Risk level string
    """
    risk_score = 0
    
    # Confidence factor (higher confidence = lower risk)
    if confidence >= 80:
        risk_score += 1
    elif confidence >= 60:
        risk_score += 2
    else:
        risk_score += 4
    
    # Volatility factor
    if volatility >= 30:
        risk_score += 2
    elif volatility >= 15:
        risk_score += 1
    
    # Streak factor (longer streaks = higher risk of reversal)
    if abs(streak_length) >= 5:
        risk_score += 2
    elif abs(streak_length) >= 3:
        risk_score += 1
    
    if risk_score <= 2:
        return "Low"
    elif risk_score <= 4:
        return "Medium"
    elif risk_score <= 6:
        return "High"
    else:
        return "Very High"

# Export main functions
__all__ = [
    'detect_streaks',
    'shannon_entropy',
    'calculate_bias_score',
    'detect_patterns',
    'hmac_to_dice_result',
    'verify_provably_fair',
    'analyze_seed_randomness',
    'calculate_confidence_score',
    'generate_prediction',
    'format_analysis_summary',
    'calculate_optimal_bet_size',
    'assess_risk_level'
]