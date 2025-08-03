# DFT-LRP for CNN1D Time Series Classification

This implementation provides comprehensive Layer-wise Relevance Propagation (LRP) and Discrete Fourier Transform LRP (DFT-LRP) analysis for the CNN1D_Wide model on vibration time series data.

## Overview

The implementation includes:

1. **Standard LRP** - Layer-wise relevance propagation in the time domain
2. **DFT-LRP** - Relevance propagation from time to frequency domain using DFT
3. **STDFT-LRP** - Short-time DFT-LRP for time-frequency analysis
4. **Comprehensive Visualizations** - Time, frequency, and time-frequency domain plots

Based on the work from [jvielhaben/DFT-LRP](https://github.com/jvielhaben/DFT-LRP).

## Files

### Core Implementation
- `explain_cnn1d.py` - Main script to run analysis on trained models
- `dft_lrp_module.py` - Core DFT-LRP implementation with clean interface
- `lrp_visualization.py` - Visualization functions for all domains
- `test_dft_lrp.py` - Test script to validate functionality

### Existing Utilities (from jvielhaben/DFT-LRP)
- `utils/dft_lrp.py` - Original DFT-LRP implementation
- `utils/dft_utils.py` - DFT utility functions  
- `utils/lrp_utils.py` - LRP utilities using Zennit

## Usage

### Basic Analysis

```bash
# Analyze with synthetic data
python explain_cnn1d.py --use_synthetic --mode single --sample_idx 0

# Analyze real data (if available)
python explain_cnn1d.py --data_path ./data/path --mode single --sample_idx 0

# Batch analysis of multiple samples
python explain_cnn1d.py --use_synthetic --mode batch --max_samples 5

# Interactive analysis
python explain_cnn1d.py --use_synthetic --mode interactive
```

### Advanced Options

```bash
# Use trained model
python explain_cnn1d.py --model_path ./trained_model.pth --use_synthetic

# Specify target class for explanation
python explain_cnn1d.py --use_synthetic --target_class 1  # 1=Bad, 0=Good

# Adjust memory usage
python explain_cnn1d.py --use_synthetic --precision 16 --batch_size 4

# Custom output directory
python explain_cnn1d.py --use_synthetic --output_dir ./my_analysis
```

## Command Line Options

- `--model_path`: Path to trained CNN1D model (default: `./cnn1d_model.pth`)
- `--data_path`: Path to data directory 
- `--use_synthetic`: Use synthetic data for demonstration
- `--mode`: Analysis mode (`single`, `batch`, `interactive`)
- `--sample_idx`: Sample index to analyze (single mode)
- `--target_class`: Target class for explanation (0=Good, 1=Bad)
- `--output_dir`: Output directory for results
- `--max_samples`: Maximum samples for batch mode
- `--precision`: Numerical precision (16 or 32)
- `--batch_size`: Batch size for memory management

## Programmatic Usage

```python
from dft_lrp_module import DFTLRPExplainer, load_trained_model, create_sample_data
from lrp_visualization import create_analysis_report

# Load model and data
model = load_trained_model('./model.pth')
data = create_sample_data(batch_size=1, signal_length=2000, n_channels=3)

# Initialize explainer
explainer = DFTLRPExplainer(model, signal_length=2000)

# Comprehensive analysis
results = explainer.analyze_sample(
    data=data,
    include_frequency=True,
    include_time_frequency=True
)

# Generate visualizations
create_analysis_report(results, output_dir='./analysis_output')
```

## Features

### Analysis Domains

1. **Time Domain**
   - Standard LRP relevance scores
   - Direct interpretation of important time points
   - Channel-wise analysis for X, Y, Z axes

2. **Frequency Domain** 
   - DFT-LRP relevance propagation
   - Identifies important frequency components
   - Magnitude and phase analysis

3. **Time-Frequency Domain**
   - Short-time DFT-LRP (memory permitting)
   - Spectrograms with relevance overlays
   - Temporal evolution of frequency importance

### Visualizations

- **Summary plots** - Comprehensive overview of all analyses
- **Time domain plots** - Signal with relevance heatmap overlay
- **Frequency domain plots** - Magnitude/phase spectra with relevance
- **Time-frequency plots** - Spectrograms with relevance
- **Channel comparison** - Side-by-side analysis of all channels

### Memory Management

- Batch processing for large datasets
- Configurable precision (16/32-bit)
- Automatic cleanup and garbage collection
- CPU/GPU device handling

## Model Compatibility

The implementation works with:
- **CNN1D_Wide** - Primary target model with wide kernels
- 3-channel vibration data (X, Y, Z axes)
- Signal length of 2000 samples (configurable)
- Binary classification (Good/Bad)

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install zennit
pip install numpy matplotlib seaborn
pip install h5py scikit-learn
```

## Output Structure

```
analysis_output/
├── summary_sample_0.png                    # Comprehensive overview
├── time_domain_sample_0.png               # Time domain analysis
├── frequency_domain_sample_0.png          # Frequency domain analysis
├── time_frequency_sample_0.png            # Time-frequency analysis
├── channel_comparison_time_sample_0.png   # Channel comparison (time)
├── channel_comparison_frequency_sample_0.png  # Channel comparison (freq)
└── channel_comparison_time_frequency_sample_0.png  # Channel comparison (TF)
```

## Troubleshooting

### Memory Issues
- Reduce `--precision` to 16
- Decrease `--batch_size`
- Use `--mode single` instead of batch
- Disable time-frequency analysis for large signals

### Model Loading Issues
- Ensure model architecture matches saved weights
- Check signal length compatibility
- Verify model is in eval mode

### CUDA Issues
- Implementation falls back to CPU automatically
- Use `--precision 16` for GPU memory constraints

## Technical Details

### DFT-LRP Algorithm
1. Compute standard LRP relevance in time domain
2. Apply DFT transformation to both signal and relevance
3. Propagate relevance through DFT using element-wise operations
4. Transform back to interpretable frequency domain

### Short-Time DFT-LRP
1. Apply windowed DFT to signal segments
2. Propagate relevance through each window
3. Create time-frequency relevance map
4. Visualize as spectrogram with relevance overlay

### Implementation Notes
- Uses existing jvielhaben/DFT-LRP core algorithms
- Adds memory management and batch processing
- Provides clean interface for CNN1D models
- Includes comprehensive error handling

## Citation

If you use this implementation, please cite the original DFT-LRP work:

```bibtex
@article{vielhaben2023dft,
  title={Discrete Fourier Transform LRP for Deep Learning},
  author={Vielhaben, J and others},
  journal={...},
  year={2023}
}
```