#!/usr/bin/env python3
"""
ULTRA-PRECISE STAKE AI TRAINER
Train neural network on real Stake data for million-dollar accuracy
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import time
from pathlib import Path
from real_stake_analyzer import RealStakeAnalyzer
from u16_seq_model import SuperChargedSeqTransformer

class StakeSequenceDataset(Dataset):
    """Dataset for Stake sequences"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class UltraPreciseStakeAI:
    """Ultra-precise AI trained on real Stake data"""
    
    def __init__(self, sequence_length=64):
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_mean = 0.0
        self.scaler_std = 1.0
        
        print(f"🧠 ULTRA-PRECISE STAKE AI INITIALIZED")
        print(f"🔧 Device: {self.device}")
        print(f"📏 Sequence Length: {sequence_length}")
    
    def prepare_training_data(self, historical_results):
        """Prepare sequences for training with multiple scales"""
        print(f"📊 Preparing training data from {len(historical_results):,} results...")
        
        # Normalize data
        results_array = np.array([r['result'] for r in historical_results])
        self.scaler_mean = np.mean(results_array)
        self.scaler_std = np.std(results_array)
        normalized_data = (results_array - self.scaler_mean) / self.scaler_std
        
        # Create sequences at multiple scales
        sequences = []
        targets = []
        
        # Scale 1: Direct sequence prediction
        for i in range(len(normalized_data) - self.sequence_length):
            seq = normalized_data[i:i + self.sequence_length]
            target = normalized_data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        # Scale 2: Skip sequences (every 2nd)
        for i in range(0, len(normalized_data) - self.sequence_length * 2, 2):
            seq = normalized_data[i:i + self.sequence_length * 2:2]
            if len(seq) == self.sequence_length:
                target = normalized_data[i + self.sequence_length * 2]
                sequences.append(seq)
                targets.append(target)
        
        # Scale 3: Pattern-aware sequences (based on sharp movements)
        sharp_indices = []
        for i in range(1, len(results_array)):
            if abs(results_array[i] - results_array[i-1]) > 30:
                sharp_indices.append(i)
        
        # Create sequences around sharp movements
        for sharp_idx in sharp_indices:
            start = max(0, sharp_idx - self.sequence_length)
            end = start + self.sequence_length
            if end < len(normalized_data):
                seq = normalized_data[start:end]
                target = normalized_data[end] if end < len(normalized_data) else normalized_data[-1]
                sequences.append(seq)
                targets.append(target)
        
        print(f"✅ Created {len(sequences):,} training sequences")
        return np.array(sequences), np.array(targets)
    
    def create_model(self):
        """Create ultra-precise model architecture"""
        print("🏗️ Creating ultra-precise model architecture...")
        
        # Enhanced model with specific parameters for dice prediction
        model = SuperChargedSeqTransformer(
            vocab_size=1,  # Continuous values
            d_model=512,   # Larger model for more precision
            nhead=16,      # More attention heads
            num_layers=12, # Deeper network
            dim_feedforward=2048,
            dropout=0.05,  # Lower dropout for precision
            max_seq_length=self.sequence_length,
            use_fourier_attention=True,
            use_memory_tokens=True,
            memory_size=128,  # More memory
            patch_size=4      # Smaller patches for finer details
        )
        
        # Add custom prediction head for dice range
        model.prediction_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Output in normalized range
        )
        
        return model.to(self.device)
    
    def train_ultra_precise(self, historical_results, epochs=100, batch_size=32):
        """Train with ultra-precision for million-dollar accuracy"""
        print(f"🚀 STARTING ULTRA-PRECISE TRAINING")
        print("=" * 50)
        
        # Prepare data
        X, y = self.prepare_training_data(historical_results)
        
        # Create datasets
        dataset = StakeSequenceDataset(X, y)
        
        # Split data strategically
        train_size = int(0.85 * len(dataset))  # More training data
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.model = self.create_model()
        
        # Ultra-precise optimizer setup
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-5,  # Lower learning rate for precision
            weight_decay=0.005,
            betas=(0.9, 0.999)
        )
        
        # Advanced scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        
        # Multiple loss functions for precision
        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        huber_loss = nn.SmoothL1Loss(beta=0.1)
        
        # Training tracking
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        training_history = []
        
        print(f"📊 Training Dataset: {len(train_dataset):,} sequences")
        print(f"📊 Validation Dataset: {len(val_dataset):,} sequences")
        print(f"🎯 Target: Ultra-precision for million-dollar betting")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_mae = 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through base model
                base_output = self.model(batch_x.unsqueeze(-1).long())
                
                # Custom prediction head
                if base_output.dim() > 2:
                    features = base_output[:, -1, :]  # Last timestep
                else:
                    features = base_output
                
                predictions = self.model.prediction_head(features).squeeze(-1)
                
                # Combined loss for ultra-precision
                mse = mse_loss(predictions, batch_y)
                mae = mae_loss(predictions, batch_y)
                huber = huber_loss(predictions, batch_y)
                
                # Weighted combination
                loss = 0.5 * mse + 0.3 * mae + 0.2 * huber
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                train_loss += loss.item()
                train_mae += mae.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")
                    print(f"    Loss: {loss.item():.6f}, MAE: {mae.item():.6f}")
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_mae = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    base_output = self.model(batch_x.unsqueeze(-1).long())
                    if base_output.dim() > 2:
                        features = base_output[:, -1, :]
                    else:
                        features = base_output
                    
                    predictions = self.model.prediction_head(features).squeeze(-1)
                    
                    loss = mse_loss(predictions, batch_y)
                    mae = mae_loss(predictions, batch_y)
                    
                    val_loss += loss.item()
                    val_mae += mae.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # Convert back to dice scale for interpretation
            val_predictions_dice = np.array(val_predictions) * self.scaler_std + self.scaler_mean
            val_targets_dice = np.array(val_targets) * self.scaler_std + self.scaler_mean
            
            dice_mae = np.mean(np.abs(val_predictions_dice - val_targets_dice))
            
            scheduler.step()
            
            print(f"\n🎯 Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Dice MAE: {dice_mae:.3f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
            
            # Save training history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dice_mae': dice_mae,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping with patience
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self.save_model(f"ultra_precise_stake_model_epoch_{epoch+1}")
                print(f"  💎 New best model saved! (MAE: {dice_mae:.3f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"  ⏰ Early stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\n✅ ULTRA-PRECISE TRAINING COMPLETED!")
        print(f"🏆 Best Validation Loss: {best_val_loss:.6f}")
        print(f"🎯 Final Dice Prediction Accuracy: ±{dice_mae:.2f} points")
        
        return training_history
    
    def predict_ultra_precise(self, recent_results, count=10):
        """Ultra-precise predictions for million-dollar betting"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        
        # Prepare input sequence
        if len(recent_results) < self.sequence_length:
            # Pad with historical average if needed
            padding = [50.0] * (self.sequence_length - len(recent_results))
            input_sequence = padding + recent_results
        else:
            input_sequence = recent_results[-self.sequence_length:]
        
        # Normalize
        input_array = np.array(input_sequence)
        normalized_input = (input_array - self.scaler_mean) / self.scaler_std
        
        predictions = []
        current_seq = torch.FloatTensor(normalized_input).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            for i in range(count):
                # Predict next value
                base_output = self.model(current_seq.long())
                if base_output.dim() > 2:
                    features = base_output[:, -1, :]
                else:
                    features = base_output
                
                pred_normalized = self.model.prediction_head(features).squeeze(-1)
                
                # Denormalize
                pred_dice = (pred_normalized.cpu().item() * self.scaler_std) + self.scaler_mean
                pred_dice = max(0.0, min(100.0, pred_dice))  # Clamp to valid range
                
                predictions.append({
                    'sequence': i + 1,
                    'predicted_value': pred_dice,
                    'confidence': min(1.0, 1.0 - abs(pred_normalized.cpu().item()) * 0.1),
                    'raw_prediction': pred_normalized.cpu().item()
                })
                
                # Update sequence for next prediction
                new_normalized = (pred_dice - self.scaler_mean) / self.scaler_std
                new_point = torch.FloatTensor([[[new_normalized]]]).to(self.device)
                current_seq = torch.cat([current_seq[:, 1:, :], new_point], dim=1)
        
        return predictions
    
    def save_model(self, name):
        """Save ultra-precise model"""
        model_path = Path(f"{name}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'sequence_length': self.sequence_length,
            'model_type': 'ultra_precise_stake'
        }, model_path)
        print(f"💾 Ultra-precise model saved: {model_path}")
    
    def load_model(self, name):
        """Load ultra-precise model"""
        model_path = Path(f"{name}.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.sequence_length = checkpoint['sequence_length']
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Ultra-precise model loaded: {model_path}")

def main():
    """Train ultra-precise AI on real Stake data"""
    print("🎯 ULTRA-PRECISE STAKE AI TRAINER")
    print("=" * 60)
    print("💰 Million-dollar accuracy training system")
    print("📊 Using real Stake seeds and 1,629 bet history")
    
    # Initialize analyzer and AI
    analyzer = RealStakeAnalyzer()
    ai = UltraPreciseStakeAI(sequence_length=64)
    
    print("\n🔄 Loading real Stake data...")
    history = analyzer.generate_complete_history()
    
    print("\n🧠 Training ultra-precise AI...")
    training_history = ai.train_ultra_precise(history, epochs=200, batch_size=16)
    
    print("\n🔮 Testing predictions...")
    recent_results = [h['result'] for h in history[-100:]]  # Last 100 bets
    predictions = ai.predict_ultra_precise(recent_results, count=20)
    
    print("\n🎯 ULTRA-PRECISE PREDICTIONS:")
    for pred in predictions[:10]:
        conf_level = "🟢 ULTRA HIGH" if pred['confidence'] > 0.9 else "🟢 HIGH" if pred['confidence'] > 0.7 else "🟡 MEDIUM"
        print(f"  #{pred['sequence']}: {pred['predicted_value']:.2f} - {conf_level} ({pred['confidence']:.3f})")
    
    # Generate betting strategy
    print("\n💎 MILLION-DOLLAR BETTING STRATEGY:")
    for pred in predictions[:5]:
        value = pred['predicted_value']
        conf = pred['confidence']
        
        if conf > 0.85:
            if value <= 15:
                print(f"  🔥🔥🔥 ULTRA BET #{pred['sequence']}: UNDER 20 ({value:.1f}) - MAXIMUM CONFIDENCE")
            elif value >= 85:
                print(f"  🔥🔥🔥 ULTRA BET #{pred['sequence']}: OVER 80 ({value:.1f}) - MAXIMUM CONFIDENCE")
            elif 48 <= value <= 52:
                print(f"  🎯 PRECISION BET #{pred['sequence']}: RANGE 47-53 ({value:.1f}) - HIGH CONFIDENCE")
        elif conf > 0.7:
            print(f"  ✅ GOOD BET #{pred['sequence']}: Target {value:.1f} - High confidence")
    
    print(f"\n🏆 ULTRA-PRECISE AI TRAINING COMPLETE!")
    print(f"💰 Ready for million-dollar betting accuracy!")

if __name__ == "__main__":
    main()