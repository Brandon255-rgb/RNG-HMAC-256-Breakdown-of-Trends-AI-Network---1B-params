#!/usr/bin/env python3
"""
Advanced Randomness Testing Demo

Demonstrates the new randomness testing capabilities with various
state-of-the-art techniques for detecting deviations from IID.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, timeout=300):
    """Run a command with timeout"""
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              check=True, timeout=timeout)
        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")
        print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
        return True
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    data_file = "rolls_1e9.u16"
    
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        print("Please run generator.py first to create the data.")
        return 1
    
    print("🔬 Advanced Randomness Testing Demo")
    print("=" * 50)
    print()
    print("This demo runs state-of-the-art randomness tests including:")
    print("• Multiscale Permutation Entropy")
    print("• Compression Race (GZIP vs LZMA)")
    print("• Spectral Density Analysis")
    print("• Mutual Information at Multiple Lags")
    print("• Approximate & Sample Entropy")
    print("• Multi-Resolution Binning Analysis")
    print("• Runs Tests with FDR Correction")
    print()
    
    # Quick test on a smaller sample
    print("1️⃣  Running QUICK randomness analysis (100K samples)...")
    quick_cmd = [
        sys.executable, "u16_seq_model.py", "randomness",
        "--u16_path", data_file,
        "--max_samples", "100000",
        "--quick",
        "--output", "quick_randomness_report.json"
    ]
    
    if not run_command(quick_cmd, timeout=120):
        print("Quick analysis failed!")
        return 1
    
    print("\n" + "="*50)
    
    # Comprehensive test on larger sample
    print("2️⃣  Running COMPREHENSIVE randomness analysis (1M samples)...")
    comp_cmd = [
        sys.executable, "u16_seq_model.py", "randomness",
        "--u16_path", data_file,
        "--max_samples", "1000000",
        "--start", "0",
        "--output", "comprehensive_randomness_report.json"
    ]
    
    if not run_command(comp_cmd, timeout=600):
        print("Comprehensive analysis failed!")
        return 1
    
    print("\n" + "="*50)
    
    # Test different sections of the data
    print("3️⃣  Testing different data sections for consistency...")
    
    sections = [
        (0, "beginning"),
        (500_000_000, "middle"),
        (900_000_000, "end")
    ]
    
    for start_pos, section_name in sections:
        print(f"\n🔍 Analyzing {section_name} section (starting at {start_pos:,})...")
        
        section_cmd = [
            sys.executable, "u16_seq_model.py", "randomness",
            "--u16_path", data_file,
            "--max_samples", "100000",
            "--start", str(start_pos),
            "--quick",
            "--output", f"randomness_{section_name}.json"
        ]
        
        if not run_command(section_cmd, timeout=120):
            print(f"❌ Failed to analyze {section_name} section")
        else:
            print(f"✅ {section_name.capitalize()} section analysis complete")
    
    print("\n" + "="*70)
    print("🎯 RANDOMNESS TESTING COMPLETE!")
    print("="*70)
    print()
    print("📊 Generated reports:")
    print("   • quick_randomness_report.json - Fast analysis results")
    print("   • comprehensive_randomness_report.json - Full analysis")
    print("   • randomness_beginning.json - First section analysis")
    print("   • randomness_middle.json - Middle section analysis") 
    print("   • randomness_end.json - Final section analysis")
    print()
    print("🔬 Analysis techniques used:")
    print("   ✓ Multiscale Permutation Entropy (MPE)")
    print("   ✓ Compression-based entropy estimation")
    print("   ✓ Spectral analysis for hidden periodicities")
    print("   ✓ Mutual information lag analysis")
    print("   ✓ Approximate & Sample entropy")
    print("   ✓ Multi-resolution uniformity testing")
    print("   ✓ Runs tests with FDR correction")
    print()
    print("💡 Interpretation guide:")
    print("   • ✅ = Test passed (data appears random)")
    print("   • ⚠️  = Potential structure detected")
    print("   • ❌ = Strong evidence of non-randomness")
    print()
    print("🔍 Next steps:")
    print("   1. Review the console output for immediate findings")
    print("   2. Examine JSON files for detailed numerical results")
    print("   3. Compare results across different data sections")
    print("   4. If issues found, investigate with targeted analysis")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())