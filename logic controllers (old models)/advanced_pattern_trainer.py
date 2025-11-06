#!/usr/bin/env python3
"""
ADVANCED PATTERN TRAINING SYSTEM
===============================

Trains sophisticated pattern recognition models on billion roll dataset for:
- Gap size analysis and prediction
- Frequency correlations and timing
- Sequential order patterns
- Out-of-order appearance detection
- Multi-dimensional pattern relationships
- Hash seed correlation analysis

TRAINS ONCE - SAVES KNOWLEDGE - INSTANT PREDICTIONS
"""

import os
import sys
import numpy as np
import pickle
import time
import hashlib
import hmac
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPatternTrainer:
    """
    Advanced pattern training system for comprehensive number prediction
    """
    
    def __init__(self, dataset_path='rolls_1e9.u16'):
        self.dataset_path = dataset_path
        self.model_save_path = 'trained_models'
        self.pattern_save_path = 'pattern_analysis'
        
        # Pattern analysis structures
        self.gap_patterns = defaultdict(list)
        self.frequency_analysis = defaultdict(int)
        self.sequence_patterns = defaultdict(list)
        self.correlation_matrix = {}
        self.hash_seed_patterns = defaultdict(list)
        self.order_analysis = defaultdict(int)
        self.timing_patterns = defaultdict(list)
        
        # Machine learning models
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Training parameters
        self.chunk_size = 1000000  # Process 1M rolls at a time
        self.pattern_window = 50   # Look at 50-roll patterns
        self.gap_analysis_depth = 20  # Analyze gaps up to 20 positions
        
        # Create directories
        Path(self.model_save_path).mkdir(exist_ok=True)
        Path(self.pattern_save_path).mkdir(exist_ok=True)
        
    def load_billion_roll_dataset(self):
        """Load and process billion roll dataset efficiently"""
        try:
            logger.info("Loading billion roll dataset for advanced pattern training...")
            
            if not os.path.exists(self.dataset_path):
                logger.error(f"Dataset not found: {self.dataset_path}")
                return None
                
            # Load data in chunks to avoid memory issues
            file_size = os.path.getsize(self.dataset_path)
            total_rolls = file_size // 2  # 2 bytes per u16
            
            logger.info(f"Dataset size: {file_size / (1024**3):.2f} GB")
            logger.info(f"Expected rolls: {total_rolls:,}")
            
            all_rolls = []
            
            with open(self.dataset_path, 'rb') as f:
                while True:
                    # Read chunk
                    chunk_data = f.read(self.chunk_size * 2)
                    if not chunk_data:
                        break
                    
                    # Convert to numpy array
                    chunk_rolls = np.frombuffer(chunk_data, dtype=np.uint16)
                    
                    # Convert to dice roll format (0-99.99)
                    chunk_rolls = (chunk_rolls % 10000) / 100.0
                    
                    all_rolls.extend(chunk_rolls)
                    
                    if len(all_rolls) % 10000000 == 0:  # Log every 10M rolls
                        logger.info(f"Loaded {len(all_rolls):,} rolls...")
            
            logger.info(f"Successfully loaded {len(all_rolls):,} rolls")
            return np.array(all_rolls)
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def analyze_gap_patterns(self, rolls):
        """Analyze gap sizes between similar numbers"""
        logger.info("Analyzing gap patterns...")
        
        # Create number buckets (0-9, 10-19, etc.)
        buckets = {}
        for i, roll in enumerate(rolls):
            bucket = int(roll // 10)  # 0-9 buckets
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(i)
        
        # Analyze gaps within each bucket
        for bucket, positions in buckets.items():
            gaps = []
            for i in range(1, len(positions)):
                gap = positions[i] - positions[i-1]
                gaps.append(gap)
            
            if gaps:
                self.gap_patterns[bucket] = {
                    'gaps': gaps,
                    'avg_gap': np.mean(gaps),
                    'std_gap': np.std(gaps),
                    'min_gap': min(gaps),
                    'max_gap': max(gaps),
                    'gap_distribution': Counter(gaps)
                }
        
        logger.info(f"Analyzed gap patterns for {len(buckets)} number buckets")
    
    def analyze_frequency_correlations(self, rolls):
        """Analyze frequency patterns and correlations"""
        logger.info("Analyzing frequency correlations...")
        
        # Basic frequency analysis
        for roll in rolls:
            bucket = int(roll)  # Integer part
            decimal = int((roll - bucket) * 100)  # Decimal part
            
            self.frequency_analysis[f"int_{bucket}"] += 1
            self.frequency_analysis[f"dec_{decimal}"] += 1
        
        # Correlation analysis - how often numbers appear together
        for i in range(len(rolls) - self.pattern_window):
            window = rolls[i:i + self.pattern_window]
            
            for j in range(len(window) - 1):
                current = int(window[j])
                next_num = int(window[j + 1])
                
                key = f"{current}->{next_num}"
                if key not in self.correlation_matrix:
                    self.correlation_matrix[key] = 0
                self.correlation_matrix[key] += 1
        
        logger.info(f"Analyzed correlations for {len(self.correlation_matrix)} number pairs")
    
    def analyze_sequence_patterns(self, rolls):
        """Analyze sequential patterns and order"""
        logger.info("Analyzing sequence patterns...")
        
        # Analyze sequences of different lengths
        for seq_len in [3, 5, 7, 10]:
            for i in range(len(rolls) - seq_len):
                sequence = tuple(int(rolls[i + j]) for j in range(seq_len))
                
                # In-order sequences
                if list(sequence) == sorted(sequence):
                    self.order_analysis[f"ascending_{seq_len}"] += 1
                elif list(sequence) == sorted(sequence, reverse=True):
                    self.order_analysis[f"descending_{seq_len}"] += 1
                else:
                    self.order_analysis[f"mixed_{seq_len}"] += 1
                
                # Store sequence for pattern analysis
                self.sequence_patterns[seq_len].append(sequence)
        
        logger.info(f"Analyzed sequence patterns for various lengths")
    
    def analyze_hash_seed_correlations(self, rolls):
        """Analyze correlations with hash seed patterns"""
        logger.info("Analyzing hash seed correlations...")
        
        # Simulate hash seed analysis
        test_seeds = [
            'example_server_seed',
            'stake_server_seed_123',
            'provably_fair_seed',
            'random_seed_test'
        ]
        
        for seed in test_seeds:
            seed_patterns = []
            
            # Generate expected patterns based on HMAC
            for nonce in range(min(1000, len(rolls))):
                client_seed = f"client_seed_{nonce}"
                
                # Calculate HMAC
                message = f"{client_seed}:{nonce}"
                hmac_result = hmac.new(
                    seed.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                # Convert to expected roll
                hex_chunk = hmac_result[:8]
                decimal_value = int(hex_chunk, 16)
                expected_roll = (decimal_value % 10000) / 100.0
                
                seed_patterns.append(expected_roll)
            
            # Compare with actual rolls
            if len(seed_patterns) <= len(rolls):
                correlation = np.corrcoef(seed_patterns, rolls[:len(seed_patterns)])[0, 1]
                self.hash_seed_patterns[seed] = {
                    'correlation': correlation if not np.isnan(correlation) else 0,
                    'patterns': seed_patterns[:100]  # Store first 100 for analysis
                }
        
        logger.info(f"Analyzed hash seed correlations for {len(test_seeds)} seeds")
    
    def analyze_timing_patterns(self, rolls):
        """Analyze timing and appearance patterns"""
        logger.info("Analyzing timing patterns...")
        
        # Analyze when certain numbers appear relative to sequence position
        for i, roll in enumerate(rolls[:100000]):  # First 100k for timing analysis
            bucket = int(roll // 10)
            position_in_sequence = i % 100  # Position within 100-roll cycles
            
            self.timing_patterns[bucket].append(position_in_sequence)
        
        # Calculate timing statistics
        for bucket, positions in self.timing_patterns.items():
            if positions:
                self.timing_patterns[bucket] = {
                    'avg_position': np.mean(positions),
                    'std_position': np.std(positions),
                    'position_distribution': Counter(positions),
                    'early_appearances': sum(1 for p in positions if p < 25),
                    'late_appearances': sum(1 for p in positions if p > 75)
                }
        
        logger.info(f"Analyzed timing patterns for number buckets")
    
    def extract_advanced_features(self, rolls, start_idx, window_size=50):
        """Extract advanced features for machine learning"""
        
        if start_idx + window_size >= len(rolls):
            return None
        
        window = rolls[start_idx:start_idx + window_size]
        
        features = []
        
        # Basic statistical features
        features.extend([
            np.mean(window),
            np.std(window),
            np.min(window),
            np.max(window),
            np.median(window)
        ])
        
        # Gap analysis features
        gaps = [window[i+1] - window[i] for i in range(len(window)-1)]
        features.extend([
            np.mean(gaps),
            np.std(gaps),
            max(gaps),
            min(gaps)
        ])
        
        # Frequency features
        unique_counts = Counter([int(r) for r in window])
        features.extend([
            len(unique_counts),  # Number of unique values
            max(unique_counts.values()),  # Most frequent count
            min(unique_counts.values())   # Least frequent count
        ])
        
        # Sequence order features
        ascending_count = sum(1 for i in range(len(window)-1) if window[i] <= window[i+1])
        features.extend([
            ascending_count / (len(window)-1),  # Proportion ascending
            (len(window)-1 - ascending_count) / (len(window)-1)  # Proportion descending
        ])
        
        # Pattern recognition features
        repeated_patterns = 0
        for i in range(len(window) - 2):
            for j in range(i + 3, len(window) - 1):
                if window[i:i+2] == window[j:j+2]:
                    repeated_patterns += 1
        features.append(repeated_patterns)
        
        # Hash correlation features (simplified)
        hash_simulation = sum(int(r) * (i+1) for i, r in enumerate(window)) % 10000
        features.extend([
            hash_simulation / 10000,
            (hash_simulation % 100) / 100
        ])
        
        return np.array(features)
    
    def train_prediction_models(self, rolls):
        """Train multiple machine learning models for prediction"""
        logger.info("Training advanced prediction models...")
        
        # Prepare training data
        features = []
        targets = []
        
        logger.info("Extracting features from billion roll dataset...")
        
        # Extract features and targets
        for i in range(0, len(rolls) - self.pattern_window - 1, 100):  # Every 100th for speed
            feature_vector = self.extract_advanced_features(rolls, i, self.pattern_window)
            
            if feature_vector is not None:
                target = rolls[i + self.pattern_window]  # Predict next roll
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(features) % 1000 == 0 and len(features) > 0:
                logger.info(f"Extracted {len(features)} feature vectors...")
                
            if len(features) >= 50000:  # Limit for training speed
                break
        
        if not features:
            logger.error("No features extracted for training")
            return False
        
        X = np.array(features)
        y = np.array(targets)
        
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        # Define models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        # Train each model
        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")
            
            start_time = time.time()
            
            if model_name == 'neural_network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            training_time = time.time() - start_time
            
            # Store model
            self.models[model_name] = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = model.feature_importances_
            
            logger.info(f"  {model_name} - R²: {r2:.6f}, MSE: {mse:.6f}, Time: {training_time:.2f}s")
        
        logger.info("Advanced model training completed!")
        return True
    
    def save_trained_models(self):
        """Save all trained models and patterns to disk"""
        logger.info("Saving trained models and patterns...")
        
        # Save ML models
        for model_name, model in self.models.items():
            model_file = os.path.join(self.model_save_path, f'{model_name}.joblib')
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = os.path.join(self.model_save_path, f'{scaler_name}.joblib')
            joblib.dump(scaler, scaler_file)
        
        # Save pattern analysis
        patterns_to_save = {
            'gap_patterns': dict(self.gap_patterns),
            'frequency_analysis': dict(self.frequency_analysis),
            'sequence_patterns': {k: list(v)[:1000] for k, v in self.sequence_patterns.items()},  # Limit size
            'correlation_matrix': dict(self.correlation_matrix),
            'hash_seed_patterns': dict(self.hash_seed_patterns),
            'order_analysis': dict(self.order_analysis),
            'timing_patterns': dict(self.timing_patterns),
            'feature_importance': dict(self.feature_importance)
        }
        
        patterns_file = os.path.join(self.pattern_save_path, 'advanced_patterns.json')
        with open(patterns_file, 'w') as f:
            json.dump(patterns_to_save, f, indent=2, default=str)
        
        # Create training metadata
        metadata = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_path': self.dataset_path,
            'models_trained': list(self.models.keys()),
            'pattern_window': self.pattern_window,
            'gap_analysis_depth': self.gap_analysis_depth,
            'chunk_size': self.chunk_size,
            'training_completed': True
        }
        
        metadata_file = os.path.join(self.model_save_path, 'training_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All models and patterns saved successfully!")
    
    def run_complete_training(self):
        """Run the complete advanced pattern training process"""
        try:
            logger.info("=" * 80)
            logger.info("ADVANCED PATTERN TRAINING SYSTEM - STARTING")
            logger.info("=" * 80)
            
            start_time = time.time()
            
            # Step 1: Load dataset
            logger.info("Step 1/7: Loading billion roll dataset...")
            rolls = self.load_billion_roll_dataset()
            
            if rolls is None:
                logger.error("Failed to load dataset")
                return False
            
            logger.info(f"Dataset loaded: {len(rolls):,} rolls")
            
            # Step 2: Analyze gap patterns
            logger.info("Step 2/7: Analyzing gap patterns...")
            self.analyze_gap_patterns(rolls)
            
            # Step 3: Analyze frequency correlations
            logger.info("Step 3/7: Analyzing frequency correlations...")
            self.analyze_frequency_correlations(rolls)
            
            # Step 4: Analyze sequence patterns
            logger.info("Step 4/7: Analyzing sequence patterns...")
            self.analyze_sequence_patterns(rolls)
            
            # Step 5: Analyze hash seed correlations
            logger.info("Step 5/7: Analyzing hash seed correlations...")
            self.analyze_hash_seed_correlations(rolls)
            
            # Step 6: Analyze timing patterns
            logger.info("Step 6/7: Analyzing timing patterns...")
            self.analyze_timing_patterns(rolls)
            
            # Step 7: Train prediction models
            logger.info("Step 7/7: Training machine learning models...")
            success = self.train_prediction_models(rolls)
            
            if not success:
                logger.error("Model training failed")
                return False
            
            # Save everything
            self.save_trained_models()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info("ADVANCED PATTERN TRAINING COMPLETED!")
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Models trained: {len(self.models)}")
            logger.info(f"Patterns analyzed: Gap, Frequency, Sequence, Hash, Timing")
            logger.info("Models saved and ready for instant predictions!")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

def main():
    """Main training entry point"""
    trainer = AdvancedPatternTrainer()
    
    # Check if models already exist
    metadata_file = os.path.join(trainer.model_save_path, 'training_metadata.json')
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('training_completed', False):
            print("=" * 80)
            print("MODELS ALREADY TRAINED!")
            print("=" * 80)
            print(f"Training date: {metadata.get('training_date', 'Unknown')}")
            print(f"Models available: {', '.join(metadata.get('models_trained', []))}")
            print("Use the dashboard for instant predictions!")
            print("To retrain, delete the 'trained_models' folder.")
            print("=" * 80)
            return True
    
    # Run training
    success = trainer.run_complete_training()
    
    if success:
        print("\n🎉 TRAINING COMPLETE! Models ready for dashboard use!")
        return True
    else:
        print("\n❌ TRAINING FAILED! Check logs for details.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)