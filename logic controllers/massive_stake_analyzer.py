#!/usr/bin/env python3
"""
Massive Stake Data Analyzer with AI Pattern Detection
Generates 1,000,000+ rolls, analyzes patterns with neural networks
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import threading
import queue
from datetime import datetime

# Import our enhanced AI model
sys.path.append(os.path.dirname(__file__))
from u16_seq_model import SuperChargedSeqTransformer, HardAttentionBlock

class StakeMassiveDataGenerator:
    """Generate massive amounts of Stake dice data for pattern analysis"""
    
    def __init__(self, output_dir: str = "stake_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_stake_hash(self, client_seed: str, server_seed: str, nonce: int) -> str:
        """Generate Stake hash using HMAC-SHA512"""
        message = f"{client_seed}-{nonce}"
        return hmac.new(
            server_seed.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
    
    def hash_to_dice_result(self, hash_value: str) -> float:
        """Convert hash to dice result (0-100)"""
        # Use first 8 hex chars for deterministic result
        hash_int = int(hash_value[:8], 16)
        random_val = hash_int / 0x100000000
        return random_val * 100
    
    def generate_massive_dataset(self, 
                                client_seed: str = "player_seed",
                                server_seeds: List[str] = None,
                                total_rolls: int = 1000000,
                                chunk_size: int = 10000) -> str:
        """Generate massive dataset with multiple server seeds"""
        
        if server_seeds is None:
            server_seeds = [
                "server_seed_1", "server_seed_2", "server_seed_3", 
                "server_seed_4", "server_seed_5"
            ]
        
        print(f"🎲 Generating {total_rolls:,} Stake dice rolls...")
        print(f"📊 Using {len(server_seeds)} different server seeds")
        
        dataset_file = self.output_dir / f"massive_stake_data_{total_rolls}.npz"
        
        all_results = []
        all_hashes = []
        all_metadata = []
        
        rolls_per_seed = total_rolls // len(server_seeds)
        
        for seed_idx, server_seed in enumerate(server_seeds):
            print(f"  🔄 Processing server seed {seed_idx + 1}/{len(server_seeds)}: {server_seed}")
            
            for chunk_start in range(0, rolls_per_seed, chunk_size):
                chunk_end = min(chunk_start + chunk_size, rolls_per_seed)
                chunk_results = []
                chunk_hashes = []
                chunk_meta = []
                
                for nonce in range(chunk_start + 1, chunk_end + 1):
                    # Generate hash and result
                    hash_val = self.generate_stake_hash(client_seed, server_seed, nonce)
                    dice_result = self.hash_to_dice_result(hash_val)
                    
                    chunk_results.append(dice_result)
                    chunk_hashes.append(hash_val[:16])  # Store first 16 chars
                    chunk_meta.append({
                        'nonce': nonce,
                        'server_seed_idx': seed_idx,
                        'timestamp': chunk_start + nonce
                    })
                
                all_results.extend(chunk_results)
                all_hashes.extend(chunk_hashes)
                all_metadata.extend(chunk_meta)
                
                # Progress update
                total_processed = seed_idx * rolls_per_seed + chunk_end
                if total_processed % 50000 == 0:
                    print(f"    📈 Progress: {total_processed:,}/{total_rolls:,} ({total_processed/total_rolls*100:.1f}%)")
        
        # Save dataset
        print(f"💾 Saving dataset to {dataset_file}")
        np.savez_compressed(
            dataset_file,
            results=np.array(all_results, dtype=np.float32),
            hashes=all_hashes,
            metadata=all_metadata,
            client_seed=client_seed,
            server_seeds=server_seeds,
            total_rolls=len(all_results)
        )
        
        print(f"✅ Generated {len(all_results):,} rolls successfully!")
        return str(dataset_file)

class StakePatternAI:
    """AI system for detecting patterns in massive Stake data"""
    
    def __init__(self, model_dir: str = "stake_ai_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 AI running on: {self.device}")
        
        self.model = None
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        
    def prepare_sequences(self, data: np.ndarray, seq_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training"""
        # Normalize data
        self.scaler_mean = np.mean(data)
        self.scaler_std = np.std(data)
        normalized_data = (data - self.scaler_mean) / self.scaler_std
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(normalized_data) - seq_length):
            seq = normalized_data[i:i + seq_length]
            target = normalized_data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        
        return (
            torch.FloatTensor(sequences).unsqueeze(-1),  # Add feature dimension
            torch.FloatTensor(targets)
        )
    
    def create_enhanced_model(self, seq_length: int = 128, d_model: int = 256) -> SuperChargedSeqTransformer:
        """Create enhanced transformer model for pattern detection"""
        model = SuperChargedSeqTransformer(
            vocab_size=1,  # Continuous values
            d_model=d_model,
            nhead=8,
            num_layers=6,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            max_seq_length=seq_length,
            use_fourier_attention=True,
            use_memory_tokens=True,
            memory_size=64,
            patch_size=8
        )
        return model.to(self.device)
    
    def train_on_massive_data(self, dataset_file: str, epochs: int = 50, batch_size: int = 64):
        """Train AI model on massive dataset"""
        print(f"🚀 Training AI on massive dataset: {dataset_file}")
        
        # Load data
        data = np.load(dataset_file)
        results = data['results']
        print(f"📊 Training on {len(results):,} data points")
        
        # Prepare sequences
        seq_length = 128
        X, y = self.prepare_sequences(results, seq_length)
        print(f"🔄 Created {len(X):,} training sequences")
        
        # Create model
        self.model = self.create_enhanced_model(seq_length)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x.long())  # Model expects integer input
                
                # Convert output to continuous prediction
                predictions = output.squeeze(-1)  # Remove last dimension if needed
                if predictions.dim() > 1:
                    predictions = predictions[:, -1]  # Take last timestep
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
            
            avg_loss = total_loss / num_batches
            scheduler.step()
            
            print(f"🎯 Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(f"best_stake_model_epoch_{epoch+1}")
                print(f"  💾 Saved new best model!")
        
        print(f"✅ Training completed! Best loss: {best_loss:.6f}")
    
    def predict_next_numbers(self, recent_results: List[float], count: int = 10) -> List[Dict]:
        """Predict next numbers using trained AI model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        
        # Prepare input sequence
        input_seq = np.array(recent_results[-128:])  # Use last 128 results
        
        # Pad if necessary
        if len(input_seq) < 128:
            padding = np.zeros(128 - len(input_seq))
            input_seq = np.concatenate([padding, input_seq])
        
        # Normalize
        normalized_seq = (input_seq - self.scaler_mean) / self.scaler_std
        
        current_seq = torch.FloatTensor(normalized_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for i in range(count):
                # Get prediction
                output = self.model(current_seq.long())
                
                # Extract prediction
                if output.dim() > 2:
                    pred = output[0, -1, 0]  # Last timestep, first feature
                else:
                    pred = output[0, -1]
                
                # Denormalize
                pred_denorm = (pred.cpu().item() * self.scaler_std) + self.scaler_mean
                
                # Clamp to valid range
                pred_denorm = max(0.0, min(100.0, pred_denorm))
                
                predictions.append({
                    'sequence': i + 1,
                    'predicted_value': pred_denorm,
                    'confidence': torch.sigmoid(pred).cpu().item(),  # Convert to confidence
                    'raw_output': pred.cpu().item()
                })
                
                # Update sequence for next prediction
                new_normalized = (pred_denorm - self.scaler_mean) / self.scaler_std
                new_point = torch.FloatTensor([[[[new_normalized]]]]).to(self.device)
                current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)
        
        return predictions
    
    def save_model(self, name: str):
        """Save trained model"""
        model_path = self.model_dir / f"{name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'model_config': {
                'vocab_size': 1,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'max_seq_length': 128
            }
        }, model_path)
        print(f"💾 Model saved to {model_path}")
    
    def load_model(self, name: str):
        """Load trained model"""
        model_path = self.model_dir / f"{name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        config = checkpoint['model_config']
        self.model = SuperChargedSeqTransformer(**config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        
        print(f"✅ Model loaded from {model_path}")

class StakeLivePredictor:
    """Real-time prediction system for Stake dice"""
    
    def __init__(self, model_name: str = "best_stake_model"):
        self.ai_system = StakePatternAI()
        
        # Try to load trained model
        try:
            self.ai_system.load_model(model_name)
            self.model_loaded = True
            print("✅ AI model loaded successfully!")
        except FileNotFoundError:
            print("⚠️  No trained model found. Need to train first!")
            self.model_loaded = False
        
        self.recent_results = deque(maxlen=1000)  # Store recent results
        self.prediction_history = []
    
    def add_observed_result(self, result: float):
        """Add newly observed result"""
        self.recent_results.append(result)
        print(f"📊 Added result: {result:.2f} (Total observed: {len(self.recent_results)})")
    
    def get_predictions(self, count: int = 5) -> List[Dict]:
        """Get next predictions"""
        if not self.model_loaded:
            print("❌ No model loaded!")
            return []
        
        if len(self.recent_results) < 10:
            print("❌ Need at least 10 observed results!")
            return []
        
        print(f"🔮 Generating {count} predictions...")
        predictions = self.ai_system.predict_next_numbers(list(self.recent_results), count)
        
        # Display predictions
        print("\n🎯 PREDICTIONS:")
        for pred in predictions:
            confidence_level = "🟢 HIGH" if pred['confidence'] > 0.7 else "🟡 MED" if pred['confidence'] > 0.5 else "🔴 LOW"
            print(f"  #{pred['sequence']}: {pred['predicted_value']:.2f} - {confidence_level} ({pred['confidence']:.3f})")
        
        self.prediction_history.extend(predictions)
        return predictions
    
    def analyze_accuracy(self, actual_results: List[float]) -> Dict:
        """Analyze prediction accuracy"""
        if not self.prediction_history:
            return {"error": "No predictions to analyze"}
        
        recent_predictions = self.prediction_history[-len(actual_results):]
        
        if len(recent_predictions) != len(actual_results):
            return {"error": "Mismatch between predictions and actual results"}
        
        errors = []
        for pred, actual in zip(recent_predictions, actual_results):
            error = abs(pred['predicted_value'] - actual)
            errors.append(error)
        
        return {
            "mean_absolute_error": np.mean(errors),
            "max_error": np.max(errors),
            "min_error": np.min(errors),
            "accuracy_within_1": sum(1 for e in errors if e <= 1.0) / len(errors),
            "accuracy_within_5": sum(1 for e in errors if e <= 5.0) / len(errors),
            "accuracy_within_10": sum(1 for e in errors if e <= 10.0) / len(errors)
        }

def main():
    """Main interface for massive Stake analysis"""
    print("🎲 MASSIVE STAKE ANALYZER WITH AI PATTERN DETECTION")
    print("=" * 60)
    
    # Initialize systems
    data_generator = StakeMassiveDataGenerator()
    ai_system = StakePatternAI()
    live_predictor = StakeLivePredictor()
    
    while True:
        print("\n🎯 OPTIONS:")
        print("1. Generate massive dataset (1M+ rolls)")
        print("2. Train AI on massive data")
        print("3. Live prediction mode")
        print("4. Batch analyze existing data")
        print("5. Load pre-trained model")
        print("6. Test prediction accuracy")
        print("7. Exit")
        
        try:
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                # Generate massive dataset
                total_rolls = int(input("Total rolls to generate (default 1000000): ") or "1000000")
                client_seed = input("Client seed (default 'player_seed'): ") or "player_seed"
                
                server_seeds = []
                print("Enter server seeds (press Enter on empty line to finish):")
                while True:
                    seed = input(f"Server seed {len(server_seeds) + 1}: ").strip()
                    if not seed:
                        break
                    server_seeds.append(seed)
                
                if not server_seeds:
                    server_seeds = ["server_1", "server_2", "server_3", "server_4", "server_5"]
                
                dataset_file = data_generator.generate_massive_dataset(
                    client_seed=client_seed,
                    server_seeds=server_seeds,
                    total_rolls=total_rolls
                )
                print(f"✅ Dataset saved: {dataset_file}")
            
            elif choice == '2':
                # Train AI
                dataset_files = list(Path("stake_data").glob("massive_stake_data_*.npz"))
                if not dataset_files:
                    print("❌ No datasets found! Generate data first.")
                    continue
                
                print("\nAvailable datasets:")
                for i, file in enumerate(dataset_files):
                    print(f"  {i + 1}. {file.name}")
                
                file_idx = int(input("Select dataset (number): ")) - 1
                dataset_file = str(dataset_files[file_idx])
                
                epochs = int(input("Training epochs (default 50): ") or "50")
                ai_system.train_on_massive_data(dataset_file, epochs=epochs)
                
                # Update live predictor
                live_predictor = StakeLivePredictor()
            
            elif choice == '3':
                # Live prediction mode
                print("\n🔴 LIVE PREDICTION MODE")
                print("Enter observed results one by one, type 'predict' to get predictions")
                print("Type 'quit' to exit this mode")
                
                while True:
                    user_input = input("Result (0-100) or 'predict' or 'quit': ").strip().lower()
                    
                    if user_input == 'quit':
                        break
                    elif user_input == 'predict':
                        count = int(input("Number of predictions (default 5): ") or "5")
                        predictions = live_predictor.get_predictions(count)
                    else:
                        try:
                            result = float(user_input)
                            if 0 <= result <= 100:
                                live_predictor.add_observed_result(result)
                            else:
                                print("❌ Result must be between 0-100")
                        except ValueError:
                            print("❌ Invalid input")
            
            elif choice == '4':
                # Batch analysis
                dataset_files = list(Path("stake_data").glob("massive_stake_data_*.npz"))
                if not dataset_files:
                    print("❌ No datasets found!")
                    continue
                
                print("\nAvailable datasets:")
                for i, file in enumerate(dataset_files):
                    print(f"  {i + 1}. {file.name}")
                
                file_idx = int(input("Select dataset: ")) - 1
                data = np.load(dataset_files[file_idx])
                results = data['results']
                
                print(f"\n📊 DATASET ANALYSIS:")
                print(f"  Total rolls: {len(results):,}")
                print(f"  Mean: {np.mean(results):.2f}")
                print(f"  Std: {np.std(results):.2f}")
                print(f"  Min/Max: {np.min(results):.2f} / {np.max(results):.2f}")
                
                # Quick pattern analysis
                print("\n🔍 QUICK PATTERN ANALYSIS:")
                
                # Autocorrelation
                autocorr = np.corrcoef(results[:-1], results[1:])[0, 1]
                print(f"  Autocorrelation: {autocorr:.6f}")
                
                # Frequency analysis
                bins = np.histogram(results, bins=20)[0]
                entropy = -np.sum((bins / len(results)) * np.log2((bins / len(results)) + 1e-10))
                print(f"  Entropy: {entropy:.3f}")
                
                # Streak analysis
                streaks = []
                current_streak = 1
                for i in range(1, len(results)):
                    if abs(results[i] - results[i-1]) < 5:  # Similar values
                        current_streak += 1
                    else:
                        if current_streak > 1:
                            streaks.append(current_streak)
                        current_streak = 1
                
                if streaks:
                    print(f"  Max streak: {max(streaks)}")
                    print(f"  Avg streak: {np.mean(streaks):.1f}")
            
            elif choice == '5':
                # Load model
                model_files = list(Path("stake_ai_models").glob("*.pth"))
                if not model_files:
                    print("❌ No trained models found!")
                    continue
                
                print("\nAvailable models:")
                for i, file in enumerate(model_files):
                    print(f"  {i + 1}. {file.stem}")
                
                model_idx = int(input("Select model: ")) - 1
                model_name = model_files[model_idx].stem
                
                live_predictor = StakeLivePredictor(model_name)
            
            elif choice == '6':
                # Test accuracy
                print("\n🎯 ACCURACY TESTING")
                test_data = []
                
                print("Enter test results (type 'done' when finished):")
                while True:
                    result = input("Test result: ").strip()
                    if result.lower() == 'done':
                        break
                    try:
                        test_data.append(float(result))
                    except ValueError:
                        print("Invalid number")
                
                if len(test_data) > 0:
                    accuracy = live_predictor.analyze_accuracy(test_data)
                    if "error" not in accuracy:
                        print(f"\n📈 ACCURACY RESULTS:")
                        print(f"  Mean Error: {accuracy['mean_absolute_error']:.2f}")
                        print(f"  Within 1 point: {accuracy['accuracy_within_1']*100:.1f}%")
                        print(f"  Within 5 points: {accuracy['accuracy_within_5']*100:.1f}%")
                        print(f"  Within 10 points: {accuracy['accuracy_within_10']*100:.1f}%")
                    else:
                        print(f"❌ {accuracy['error']}")
            
            elif choice == '7':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

if __name__ == "__main__":
    main()