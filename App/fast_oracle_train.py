#!/usr/bin/env python3
"""
ORACLE FAST TRAINING - 50,000 SAMPLES
=====================================

Modified training script that uses only 50,000 samples from the billion roll dataset
for faster training and testing. We can expand this later.

Environment variables are already configured:
- STAKE_API_KEY
- AWS credentials for Bedrock
- All production settings ready
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import time
from datetime import datetime
from oracle_train import SupremePatternOracle
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastOracle(SupremePatternOracle):
    """Fast training version of the Oracle using 50k samples"""
    
    def __init__(self):
        super().__init__()
        self.sample_size = 50000  # Use 50k instead of 1B
        
    def load_sample_rolls(self) -> np.ndarray:
        """Load 50,000 samples from the billion roll dataset"""
        logger.info(f"📊 Loading {self.sample_size:,} samples from 'rolls_1e9.u16'...")
        
        try:
            # Load full dataset
            full_rolls = np.fromfile('rolls_1e9.u16', dtype=np.uint16)
            full_rolls = (full_rolls % 10000) / 100.0  # Convert to 0-99.99 range
            
            logger.info(f"✅ Full dataset loaded: {len(full_rolls):,} rolls")
            
            # Sample 50k from different parts of the dataset for diversity
            sample_indices = np.linspace(0, len(full_rolls) - 1, self.sample_size, dtype=int)
            sample_rolls = full_rolls[sample_indices]
            
            logger.info(f"✅ Sampled {len(sample_rolls):,} rolls")
            logger.info(f"📈 Sample Range: {sample_rolls.min():.2f} - {sample_rolls.max():.2f}")
            logger.info(f"📊 Sample Mean: {sample_rolls.mean():.2f}, Std: {sample_rolls.std():.2f}")
            
            return sample_rolls
            
        except FileNotFoundError:
            logger.error("❌ 'rolls_1e9.u16' not found! Please ensure the billion roll dataset exists.")
            raise
    
    def fast_train_oracle(self):
        """Fast training on 50k samples"""
        logger.info("🔥 FAST TRAINING SUPREME PATTERN ORACLE (50K SAMPLES)...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load sample data
        rolls = self.load_sample_rolls()
        
        # Step 1: Feature Extraction (simplified)
        logger.info("🔍 Step 1: Feature Extraction (Fast Mode)")
        
        window_size = 50
        step_size = 5  # Smaller step for more samples from the 50k
        
        X, y = [], []
        for i in range(0, len(rolls) - window_size, step_size):
            window = rolls[i:i + window_size]
            target = rolls[i + window_size]
            
            # Extract basic features
            features = [
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.median(window),
                len([x for x in window if x > 50]) / len(window),  # % over 50
                window[-1],  # last value
                window[-1] - window[-2] if len(window) > 1 else 0,  # trend
            ]
            
            X.append(features)
            y.append(target)
            
            if len(X) % 1000 == 0:
                logger.info(f"   Processed: {len(X):,} samples")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Extracted {len(X):,} feature vectors")
        logger.info(f"📊 Feature shape: {X.shape}")
        
        # Step 2: Train Simple Neural Network
        logger.info("🧠 Step 2: Training Neural Network")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(X.shape[1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 50  # Fast training
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test_tensor)
                    test_loss = criterion(test_outputs, y_test_tensor)
                    logger.info(f"   Epoch {epoch}: Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'fast_oracle.pth')
        logger.info("💾 Model saved as 'fast_oracle.pth'")
        
        # Step 3: Test predictions
        logger.info("🎯 Step 3: Testing Predictions")
        
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            
            # Calculate accuracy within tolerance
            tolerance = 5.0  # ±5 points
            accurate = torch.abs(test_predictions - y_test_tensor) <= tolerance
            accuracy = accurate.float().mean().item()
            
            logger.info(f"✅ Prediction Accuracy (±{tolerance}): {accuracy*100:.1f}%")
        
        # Step 4: Generate sample predictions
        logger.info("🔮 Step 4: Sample Predictions")
        
        # Use last few windows for prediction
        for i in range(5):
            idx = -(i+1) * 10  # Take samples from end
            if abs(idx) < len(X_test):
                features = X_test[idx:idx+1]
                actual = y_test[idx]
                
                prediction = model(torch.FloatTensor(features)).item()
                logger.info(f"   Prediction {i+1}: {prediction:.2f} | Actual: {actual:.2f} | Diff: {abs(prediction-actual):.2f}")
        
        # Training summary
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("🎉 FAST TRAINING COMPLETE!")
        logger.info(f"⏱️ Training Time: {training_time:.1f} seconds")
        logger.info(f"📊 Samples Used: {self.sample_size:,}")
        logger.info(f"🎯 Final Accuracy: {accuracy*100:.1f}%")
        logger.info(f"💾 Model Saved: fast_oracle.pth")
        logger.info("=" * 80)
        
        return {
            'training_time': training_time,
            'accuracy': accuracy,
            'samples_used': self.sample_size,
            'model_path': 'fast_oracle.pth'
        }

def main():
    """Run fast Oracle training"""
    print("🔥 FAST ORACLE TRAINING - 50,000 SAMPLES")
    print("=" * 60)
    print("Environment variables loaded from .env file")
    print("Production system ready for faster training cycle")
    print()
    
    # Initialize fast oracle
    oracle = FastOracle()
    
    # Run fast training
    results = oracle.fast_train_oracle()
    
    print("\n🚀 FAST TRAINING RESULTS:")
    print(f"   Training Time: {results['training_time']:.1f} seconds")
    print(f"   Accuracy: {results['accuracy']*100:.1f}%")
    print(f"   Samples Used: {results['samples_used']:,}")
    print(f"   Model File: {results['model_path']}")
    
    print("\n✅ Fast Oracle training complete!")
    print("🔄 Ready to expand to more samples when needed")

if __name__ == "__main__":
    main()