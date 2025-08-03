#!/usr/bin/env python3
"""
Example usage of DFT-LRP for CNN1D time series classification.

This script demonstrates how to use the DFT-LRP implementation
to explain CNN1D_Wide model predictions on vibration data.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dft_lrp_module import DFTLRPExplainer, create_sample_data
from lrp_visualization import plot_comparison_summary, plot_time_domain_relevance
from Classification.cnn1D_model import CNN1D_Wide


def example_basic_analysis():
    """Basic example of using DFT-LRP for a single sample."""
    print("="*60)
    print("Basic DFT-LRP Analysis Example")
    print("="*60)
    
    # 1. Create a model (normally you would load a trained model)
    print("1. Creating CNN1D_Wide model...")
    model = CNN1D_Wide()
    model.eval()
    
    # 2. Create sample data (normally you would load real vibration data)
    print("2. Creating sample vibration data...")
    data = create_sample_data(
        batch_size=1, 
        signal_length=2000, 
        n_channels=3,
        noise_level=0.1
    )
    print(f"   Data shape: {data.shape}")
    
    # 3. Initialize the DFT-LRP explainer
    print("3. Initializing DFT-LRP explainer...")
    explainer = DFTLRPExplainer(
        model=model,
        signal_length=2000,
        precision=32,
        use_cuda=False  # Use CPU for this example
    )
    
    # 4. Perform time domain analysis
    print("4. Computing time domain relevance...")
    time_results = explainer.explain_time_domain(data)
    
    print("   Results:")
    print(f"   - Predicted class: {time_results['prediction'][0]} ({'Bad' if time_results['prediction'][0] == 1 else 'Good'})")
    print(f"   - Confidence: {np.max(time_results['probabilities'][0]):.3f}")
    print(f"   - Relevance shape: {time_results['relevance'].shape}")
    
    # 5. Perform frequency domain analysis
    print("5. Computing frequency domain relevance...")
    try:
        freq_results = explainer.explain_frequency_domain(data)
        print("   ✓ Frequency domain analysis completed")
        print(f"   - Signal freq shape: {freq_results['signal_freq'].shape}")
        print(f"   - Relevance freq shape: {freq_results['relevance_freq'].shape}")
    except Exception as e:
        print(f"   ✗ Frequency domain analysis failed: {e}")
        freq_results = None
    
    # 6. Create visualization
    print("6. Creating visualization...")
    
    # Plot time domain results
    fig1 = plot_time_domain_relevance(
        data=time_results['input_data'],
        relevance=time_results['relevance'],
        title="Time Domain Relevance Analysis",
        sample_idx=0
    )
    plt.savefig('example_time_domain.png', dpi=300, bbox_inches='tight')
    print("   ✓ Time domain plot saved as 'example_time_domain.png'")
    plt.show()
    plt.close(fig1)
    
    # If frequency analysis worked, create comparison plot
    if freq_results:
        results = {
            'time': time_results,
            'frequency': freq_results
        }
        
        fig2 = plot_comparison_summary(results, sample_idx=0)
        plt.savefig('example_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✓ Comparison plot saved as 'example_comparison.png'")
        plt.show()
        plt.close(fig2)
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


def example_batch_analysis():
    """Example of analyzing multiple samples."""
    print("\n" + "="*60)
    print("Batch Analysis Example")
    print("="*60)
    
    # Create model and data
    model = CNN1D_Wide()
    model.eval()
    
    # Create multiple samples
    data = create_sample_data(batch_size=3, signal_length=2000, n_channels=3)
    print(f"Created batch data with shape: {data.shape}")
    
    # Initialize explainer
    explainer = DFTLRPExplainer(model, signal_length=2000, use_cuda=False)
    
    # Analyze each sample
    for i in range(data.shape[0]):
        print(f"\nAnalyzing sample {i+1}/3...")
        
        sample_data = data[i:i+1]  # Keep batch dimension
        results = explainer.explain_time_domain(sample_data)
        
        pred_class = results['prediction'][0]
        confidence = np.max(results['probabilities'][0])
        
        print(f"  Sample {i}: Class {pred_class} ({'Bad' if pred_class == 1 else 'Good'}) "
              f"with confidence {confidence:.3f}")
    
    print("\nBatch analysis completed!")


def example_channel_analysis():
    """Example of analyzing individual channels (X, Y, Z axes)."""
    print("\n" + "="*60)
    print("Channel Analysis Example")
    print("="*60)
    
    model = CNN1D_Wide()
    model.eval()
    
    # Create data with distinct patterns in each channel
    data = torch.zeros(1, 3, 2000)
    t = torch.linspace(0, 10, 2000)
    
    # Channel 0 (X): Low frequency oscillation
    data[0, 0, :] = torch.sin(2 * np.pi * 2 * t) + 0.1 * torch.randn(2000)
    
    # Channel 1 (Y): Medium frequency oscillation  
    data[0, 1, :] = torch.sin(2 * np.pi * 5 * t) + 0.1 * torch.randn(2000)
    
    # Channel 2 (Z): High frequency oscillation
    data[0, 2, :] = torch.sin(2 * np.pi * 10 * t) + 0.1 * torch.randn(2000)
    
    print("Created data with different frequency content per channel:")
    print("  - Channel 0 (X): 2 Hz oscillation")
    print("  - Channel 1 (Y): 5 Hz oscillation") 
    print("  - Channel 2 (Z): 10 Hz oscillation")
    
    # Analyze
    explainer = DFTLRPExplainer(model, signal_length=2000, use_cuda=False)
    results = explainer.explain_time_domain(data)
    
    # Show relevance statistics per channel
    relevance = results['relevance'][0]  # Remove batch dimension
    
    print("\nRelevance statistics per channel:")
    channel_names = ['X', 'Y', 'Z']
    
    for i, name in enumerate(channel_names):
        rel_mean = np.mean(np.abs(relevance[i]))
        rel_max = np.max(np.abs(relevance[i]))
        rel_std = np.std(relevance[i])
        
        print(f"  {name}: Mean=|{rel_mean:.4f}|, Max=|{rel_max:.4f}|, Std={rel_std:.4f}")
    
    # Plot individual channels
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    time_axis = np.arange(2000)
    
    for i, name in enumerate(channel_names):
        # Original signal
        axes[i, 0].plot(time_axis, data[0, i].numpy(), 'k-', linewidth=1)
        axes[i, 0].set_title(f'Channel {name} - Original Signal')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Relevance
        axes[i, 1].plot(time_axis, relevance[i], 'r-', linewidth=1)
        axes[i, 1].set_title(f'Channel {name} - Relevance')
        axes[i, 1].set_ylabel('Relevance')
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_xlabel('Time')
    axes[2, 1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('example_channel_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Channel analysis plot saved as 'example_channel_analysis.png'")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    try:
        # Run all examples
        example_basic_analysis()
        example_batch_analysis()
        example_channel_analysis()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("Check the generated PNG files for visualizations.")
        print("="*80)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()