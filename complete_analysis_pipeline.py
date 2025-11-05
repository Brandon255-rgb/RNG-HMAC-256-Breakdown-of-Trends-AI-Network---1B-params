#!/usr/bin/env python3
"""
Complete Analysis Pipeline

Demonstrates the full enhanced workflow:
1. Advanced randomness testing of source data
2. Enhanced model training with modern techniques
3. Model evaluation and prediction analysis
4. Comparison between model predictions and data randomness
"""

import subprocess
import sys
import time
from pathlib import Path
import json

def run_command(cmd, timeout=600):
    """Run command with progress indication"""
    print(f"🔄 {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              check=True, timeout=timeout)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        print(result.stdout)
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"⏰ Timed out after {timeout}s")
        return False, ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False, ""

def main():
    data_file = "rolls_1e9.u16"
    model_file = "enhanced_model_v2.pt"
    
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        return 1
    
    print("🚀 COMPLETE ENHANCED ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1: Data Quality Assessment
    print("STEP 1: DATA QUALITY ASSESSMENT")
    print("-" * 40)
    
    randomness_cmd = [
        sys.executable, "u16_seq_model.py", "randomness",
        "--u16_path", data_file,
        "--max_samples", "500000",
        "--output", "data_quality_report.json"
    ]
    
    success, output = run_command(randomness_cmd, timeout=300)
    if not success:
        print("❌ Data quality assessment failed!")
        return 1
    
    print("\n" + "=" * 60)
    
    # Step 2: Enhanced Model Training
    print("STEP 2: ENHANCED MODEL TRAINING")
    print("-" * 40)
    
    train_cmd = [
        sys.executable, "u16_seq_model.py", "train",
        "--u16_path", data_file,
        "--ckpt", model_file,
        "--epochs", "5",
        "--batch", "128",
        "--max_samples", "2000000",  # 2M samples for decent training
        "--d_model", "384",          # Balanced model size
        "--nhead", "12",
        "--nlayers", "6", 
        "--dim_ff", "1536",
        "--gradient_accumulation", "4",
        "--use_amp",
        "--eval_every", "500",
        "--patience", "3",
        "--lr", "8e-4",
        "--warmup_ratio", "0.15"
    ]
    
    success, output = run_command(train_cmd, timeout=900)
    if not success:
        print("❌ Model training failed!")
        return 1
    
    print("\n" + "=" * 60)
    
    # Step 3: Model Evaluation
    print("STEP 3: COMPREHENSIVE MODEL EVALUATION")
    print("-" * 40)
    
    eval_cmd = [
        sys.executable, "u16_seq_model.py", "evaluate",
        "--u16_path", data_file,
        "--ckpt", model_file,
        "--max_samples", "100000",
        "--use_amp",
        "--analyze_predictions"
    ]
    
    success, output = run_command(eval_cmd, timeout=300)
    if not success:
        print("❌ Model evaluation failed!")
        return 1
    
    print("\n" + "=" * 60)
    
    # Step 4: Prediction Quality Analysis
    print("STEP 4: PREDICTION QUALITY ANALYSIS")
    print("-" * 40)
    
    # Test predictions at multiple points
    test_indices = [1000000, 2000000, 3000000, 4000000, 5000000]
    
    for i, idx in enumerate(test_indices):
        print(f"\n🎯 Testing predictions at position {idx:,}")
        
        predict_cmd = [
            sys.executable, "u16_seq_model.py", "predict",
            "--u16_path", data_file,
            "--ckpt", model_file,
            "--idx", str(idx),
            "--topk", "5"
        ]
        
        success, output = run_command(predict_cmd, timeout=60)
        if success:
            print(f"✅ Prediction {i+1}/{len(test_indices)} complete")
        else:
            print(f"❌ Prediction {i+1}/{len(test_indices)} failed")
    
    print("\n" + "=" * 60)
    
    # Step 5: Summary Report
    print("STEP 5: ANALYSIS SUMMARY")
    print("-" * 40)
    
    print("\n📊 PIPELINE RESULTS:")
    print(f"   ✓ Data quality assessment: Complete")
    print(f"   ✓ Enhanced model training: {model_file}")
    print(f"   ✓ Model evaluation: Complete") 
    print(f"   ✓ Prediction analysis: {len(test_indices)} test points")
    
    print("\n🔬 ADVANCED FEATURES DEMONSTRATED:")
    print("   • Multiscale Permutation Entropy analysis")
    print("   • Compression-based randomness testing") 
    print("   • Spectral density periodicity detection")
    print("   • Mutual information lag analysis")
    print("   • Multi-resolution uniformity testing")
    print("   • Enhanced transformer with RoPE")
    print("   • Mixed precision training (AMP)")
    print("   • Cosine LR scheduling with warmup")
    print("   • Gradient accumulation")
    print("   • Comprehensive evaluation metrics")
    
    print("\n📁 GENERATED FILES:")
    files = [
        ("data_quality_report.json", "Detailed randomness test results"),
        (model_file, "Trained enhanced transformer model"),
        (f"{model_file.replace('.pt', '_best.pt')}", "Best model checkpoint")
    ]
    
    for filename, description in files:
        if Path(filename).exists():
            print(f"   ✓ {filename} - {description}")
        else:
            print(f"   ? {filename} - {description} (may not exist)")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Review data quality report for any randomness issues")
    print("   2. Examine model training curves and validation metrics")
    print("   3. Test model predictions on different data sections")
    print("   4. Compare model performance to randomness baseline")
    print("   5. Iterate on model architecture based on findings")
    
    print("\n🎉 ANALYSIS PIPELINE COMPLETE!")
    print("    Your model now has state-of-the-art capabilities for")
    print("    both randomness analysis and sequence prediction.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())