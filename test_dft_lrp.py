#!/usr/bin/env python3
"""
Test script for DFT-LRP implementation
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality of the DFT-LRP implementation."""
    print("Testing DFT-LRP implementation...")
    
    try:
        # Import modules
        from dft_lrp_module import DFTLRPExplainer, create_sample_data
        from Classification.cnn1D_model import CNN1D_Wide
        print("✓ Imports successful")
        
        # Create model
        model = CNN1D_Wide()
        model.eval()
        print("✓ Model creation successful")
        
        # Create sample data
        data = create_sample_data(batch_size=2, signal_length=2000, n_channels=3)
        print(f"✓ Sample data created with shape: {data.shape}")
        
        # Initialize explainer
        explainer = DFTLRPExplainer(
            model=model,
            signal_length=2000,
            precision=32,
            use_cuda=False  # Use CPU for testing
        )
        print("✓ DFT-LRP explainer initialized")
        
        # Test time domain analysis
        print("\nTesting time domain analysis...")
        time_results = explainer.explain_time_domain(data[0:1])  # Single sample
        print(f"✓ Time domain analysis successful")
        print(f"  - Relevance shape: {time_results['relevance'].shape}")
        print(f"  - Prediction: {time_results['prediction']}")
        print(f"  - Probabilities: {time_results['probabilities']}")
        
        # Test visualization import
        from lrp_visualization import plot_time_domain_relevance, plot_comparison_summary
        print("✓ Visualization imports successful")
        
        print("\n" + "="*50)
        print("All basic tests passed successfully!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)