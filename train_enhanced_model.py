#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced model
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and print its output"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    # Configuration
    data_file = "rolls_1e9.u16"
    model_file = "enhanced_model.pt"
    
    if not Path(data_file).exists():
        print(f"Error: Data file {data_file} not found!")
        print("Please run generator.py first to create the data.")
        return 1
    
    print("Enhanced Model Training Example")
    print("=" * 40)
    
    # Training with enhanced features
    print("\n1. Training enhanced model...")
    train_cmd = [
        sys.executable, "u16_seq_model.py", "train",
        "--u16_path", data_file,
        "--ckpt", model_file,
        "--epochs", "3",
        "--batch", "128",  # Smaller batch for demo
        "--max_samples", "1000000",  # 1M samples for quick training
        "--d_model", "256",  # Smaller model for demo
        "--nhead", "8",
        "--nlayers", "6",
        "--dim_ff", "1024",
        "--gradient_accumulation", "2",
        "--use_amp",  # Enable mixed precision
        "--eval_every", "500",
        "--patience", "3",
        "--lr", "1e-3",
        "--warmup_ratio", "0.1"
    ]
    
    if not run_command(train_cmd):
        print("Training failed!")
        return 1
    
    print("\n2. Evaluating model...")
    eval_cmd = [
        sys.executable, "u16_seq_model.py", "evaluate",
        "--u16_path", data_file,
        "--ckpt", model_file,
        "--max_samples", "50000",
        "--use_amp",
        "--analyze_predictions"
    ]
    
    if not run_command(eval_cmd):
        print("Evaluation failed!")
        return 1
    
    print("\n3. Making predictions...")
    predict_cmd = [
        sys.executable, "u16_seq_model.py", "predict",
        "--u16_path", data_file,
        "--ckpt", model_file,
        "--idx", "1000000",
        "--topk", "5"
    ]
    
    if not run_command(predict_cmd):
        print("Prediction failed!")
        return 1
    
    print("\n4. Running analysis...")
    analyse_cmd = [
        sys.executable, "u16_seq_model.py", "analyse",
        "--u16_path", data_file,
        "--target", "5000",
        "--band_pct", "1.0",
        "--max_samples", "100000",
        "--out", "analysis_results.txt"
    ]
    
    if not run_command(analyse_cmd):
        print("Analysis failed!")
        return 1
    
    print("\nAll steps completed successfully!")
    print(f"Enhanced model saved as: {model_file}")
    print("Check the output files for detailed results.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())