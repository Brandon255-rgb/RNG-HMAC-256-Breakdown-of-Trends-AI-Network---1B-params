# Advanced Randomness Testing Suite

## Overview
Your model now includes a comprehensive suite of state-of-the-art randomness testing techniques, implementing many of the methods you requested for detecting subtle deviations from IID (Independent and Identically Distributed) data.

## Implemented Techniques

### 🔄 **Multiscale Permutation Entropy (MPE)**
- **What**: Analyzes complexity at different time scales by computing permutation entropy on coarse-grained data
- **Detects**: Scale-dependent patterns, long-range correlations
- **Theory**: True IID data should maintain high complexity across all scales
- **Implementation**: Tests scales 1-20 with order-3 patterns

### 🗜️ **Compression Race**
- **What**: Compares GZIP vs LZMA compression on rolling windows
- **Detects**: Compressible structure, algorithmic complexity
- **Theory**: Random data should compress poorly and consistently
- **Implementation**: 50K sample windows, tracks entropy rates

### 🌊 **Spectral Density Analysis**
- **What**: Analyzes power spectra of indicator sequences at multiple thresholds
- **Detects**: Hidden periodicities, spectral structure
- **Theory**: IID data should have flat (white noise) spectrum
- **Implementation**: FFT analysis with Kolmogorov-Smirnov tests

### 🔗 **Mutual Information at Lags**
- **What**: Measures information sharing between values separated by various lags
- **Detects**: Temporal dependencies, memory effects
- **Theory**: IID data should have MI ≈ 0 at all lags > 0
- **Implementation**: Lag analysis up to 50 steps with bias correction

### 📈 **Approximate & Sample Entropy**
- **What**: Measures regularity and predictability in time series
- **Detects**: Deterministic patterns, reduced complexity
- **Theory**: Random data should have high entropy values
- **Implementation**: Both ApEn and SampEn with adaptive parameters

### 🔍 **Multi-Resolution Binning Analysis**
- **What**: Tests uniformity across different binning resolutions (256, 1024, 4096, 8192)
- **Detects**: Scale-dependent artifacts, quantization effects
- **Theory**: Uniform distribution should be preserved across scales
- **Implementation**: Chi-square and KS tests with KL divergence

### 🏃 **Runs Tests with FDR Correction**
- **What**: Tests run length distributions at multiple thresholds
- **Detects**: Clustering, anti-clustering patterns
- **Theory**: Random binary sequences should follow expected run distributions
- **Implementation**: Benjamini-Hochberg FDR correction across thresholds

## Command Usage

### Quick Analysis (Fast)
```bash
python u16_seq_model.py randomness --u16_path rolls_1e9.u16 --quick --max_samples 100000
```

### Comprehensive Analysis
```bash
python u16_seq_model.py randomness --u16_path rolls_1e9.u16 --max_samples 1000000 --output results.json
```

### Section Comparison
```bash
# Test different parts of your data
python u16_seq_model.py randomness --u16_path rolls_1e9.u16 --start 0 --max_samples 100000
python u16_seq_model.py randomness --u16_path rolls_1e9.u16 --start 500000000 --max_samples 100000
```

## Interpretation Guide

### 🎯 **Overall Assessment**
The suite provides an overall randomness score based on:
- Scale complexity maintenance (MPE)
- Compression resistance
- Spectral whiteness
- Temporal independence (MI)
- Multi-scale uniformity
- Run distribution conformity

### ⚠️ **Warning Signs**
- **Low complexity at large scales**: Indicates long-range structure
- **Significant periodicity**: Hidden cycles or patterns
- **High mutual information**: Temporal dependencies
- **Non-uniform distributions**: Biased generators or artifacts
- **Abnormal run patterns**: Clustering or regularity

### ✅ **Good Randomness Indicators**
- High permutation entropy across all scales (> 0.95)
- Poor compression ratios (> 7:1 for both algorithms)
- Flat power spectra (p-values > 0.05)
- Low mutual information (< 0.01) at all lags
- Uniform distributions across all bin resolutions
- Expected run distributions

## Scientific Rigor

### **Statistical Corrections**
- False Discovery Rate (FDR) correction for multiple testing
- Bias-corrected mutual information estimators
- Adaptive parameter selection based on data characteristics

### **Null Hypothesis Testing**
Each test compares against theoretical expectations for IID data:
- Permutation entropy should approach maximum (log₂(m!))
- Compression should be minimal and uniform
- Spectral density should be flat
- Mutual information should be near zero
- Distributions should be uniform
- Runs should follow expected patterns

### **Robustness**
- Multiple complementary approaches reduce false positives
- Scale-invariant testing prevents missing scale-specific artifacts
- Phase-space analysis captures different aspects of randomness

## Performance

### **Timing Estimates**
- Quick mode (100K samples): ~30-60 seconds
- Comprehensive (1M samples): ~5-10 minutes
- Memory usage: ~100-500MB depending on sample size

### **Scalability**
- Automatically samples large datasets
- Optimized algorithms for billion-scale data
- Configurable precision vs speed trade-offs

## Advanced Features

### **JSON Output**
Detailed results saved in structured format for:
- Automated analysis pipelines
- Statistical comparison across datasets
- Longitudinal monitoring of generator quality

### **Comparative Analysis**
Test different:
- Data sections (temporal drift detection)
- Generator seeds (consistency verification)
- Parameter variations (sensitivity analysis)

This implementation provides research-grade randomness testing suitable for cryptographic, scientific, and Monte Carlo applications where data quality is critical.