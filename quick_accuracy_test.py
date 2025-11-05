#!/usr/bin/env python3
"""
Quick Model Comparison Script
Compare accuracy between different model architectures
"""

import torch
import numpy as np
from u16_seq_model import *
import time

def quick_accuracy_test(u16_path, num_samples=10000):
    """Quick test to compare model accuracies"""
    
    print("🔬 QUICK ACCURACY COMPARISON")
    print("=" * 50)
    
    # Load test data
    rolls = memmap_rolls(u16_path)
    test_data = rolls[:num_samples]
    
    # Create simple test dataset
    window = 64
    vocab = 1024
    
    sequences = []
    targets = []
    
    for i in range(len(test_data) - window - 4):
        seq = (test_data[i:i+window] * vocab // 10001).astype(np.int64)
        tgt = (test_data[i+window:i+window+4] * vocab // 10001).astype(np.int64)
        sequences.append(seq)
        targets.append(tgt)
    
    if len(sequences) < 100:
        print("❌ Not enough data for testing")
        return
    
    # Convert to tensors
    X = torch.from_numpy(np.array(sequences[:1000]))  # Use 1000 samples
    y = torch.from_numpy(np.array(targets[:1000]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.to(device), y.to(device)
    
    print(f"📊 Test Data: {len(X)} sequences")
    print(f"🖥️  Device: {device}")
    print()
    
    # Test models
    models_to_test = [
        {
            'name': 'Original SeqTransformer',
            'class': SeqTransformer,
            'kwargs': {'vocab_size': vocab, 'd_model': 256, 'nhead': 8, 'nlayers': 4}
        },
        {
            'name': 'Supercharged Model',
            'class': SuperChargedSeqTransformer,
            'kwargs': {'vocab_size': vocab, 'd_model': 256, 'nhead': 8, 'nlayers': 4, 'patch_len': 8}
        }
    ]
    
    results = {}
    
    for model_info in models_to_test:
        print(f"🧠 Testing: {model_info['name']}")
        
        try:
            # Initialize model
            model = model_info['class'](**model_info['kwargs']).to(device)
            model.eval()
            
            # Random prediction (untrained)
            accuracies = []
            inference_times = []
            
            with torch.no_grad():
                for i in range(0, len(X), 50):  # Test in small batches
                    batch_X = X[i:i+50]
                    batch_y = y[i:i+50]
                    
                    start_time = time.time()
                    
                    # Different forward pass for different models
                    if 'SuperCharged' in model_info['name']:
                        outputs, _ = model(batch_X)
                        predictions = outputs[0]  # First head
                    else:
                        outputs = model(batch_X)
                        if isinstance(outputs, list):
                            predictions = outputs[0]
                        else:
                            predictions = outputs
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    # Compute accuracy for first prediction
                    pred_tokens = predictions.argmax(dim=-1)
                    accuracy = (pred_tokens == batch_y[:, 0]).float().mean().item()
                    accuracies.append(accuracy)
            
            avg_accuracy = np.mean(accuracies)
            avg_inference_time = np.mean(inference_times)
            
            # Model info
            params = sum(p.numel() for p in model.parameters())
            
            results[model_info['name']] = {
                'accuracy': avg_accuracy,
                'inference_time': avg_inference_time,
                'parameters': params
            }
            
            print(f"   ✓ Accuracy: {avg_accuracy:.2%}")
            print(f"   ⏱️  Inference: {avg_inference_time*1000:.2f}ms/batch")
            print(f"   📊 Parameters: {params:,}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    # Summary
    print("📋 COMPARISON SUMMARY")
    print("=" * 50)
    
    if len(results) >= 2:
        original_acc = results.get('Original SeqTransformer', {}).get('accuracy', 0)
        supercharged_acc = results.get('Supercharged Model', {}).get('accuracy', 0)
        
        if original_acc > 0:
            improvement = ((supercharged_acc - original_acc) / original_acc) * 100
            print(f"🚀 Accuracy Improvement: {improvement:+.1f}%")
        
        print(f"📊 Original Model: {original_acc:.2%}")
        print(f"🔥 Supercharged: {supercharged_acc:.2%}")
    
    print("\n💡 Note: These are untrained model comparisons.")
    print("   Train with 'supercharged' command for full benefits!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick model accuracy comparison")
    parser.add_argument("--u16_path", required=True, help="Path to data file")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to test")
    
    args = parser.parse_args()
    quick_accuracy_test(args.u16_path, args.samples)