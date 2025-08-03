#!/usr/bin/env python3
"""
CNN1D Model Explainability Analysis using DFT-LRP

This script provides a comprehensive analysis of CNN1D_Wide model predictions
using Layer-wise Relevance Propagation (LRP) and Discrete Fourier Transform LRP (DFT-LRP).

The implementation supports:
1. Standard LRP in time domain
2. DFT-LRP for frequency domain relevance
3. Short-time DFT-LRP for time-frequency analysis
4. Comprehensive visualizations

Based on the work from jvielhaben/DFT-LRP repository.

Usage:
    python explain_cnn1d.py [--model_path MODEL_PATH] [--data_path DATA_PATH] 
                           [--sample_idx SAMPLE_IDX] [--output_dir OUTPUT_DIR]
                           [--batch_size BATCH_SIZE] [--use_synthetic]
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import gc

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from dft_lrp_module import DFTLRPExplainer, load_trained_model, create_sample_data
from lrp_visualization import (
    plot_comparison_summary, 
    plot_time_domain_relevance,
    plot_frequency_domain_relevance, 
    plot_time_frequency_relevance,
    create_analysis_report
)

# Import model and data utilities
from Classification.cnn1D_model import CNN1D_Wide
from utils.dataloader import stratified_group_split
import h5py

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def load_real_data(data_directory: str, n_samples: int = 5) -> tuple:
    """
    Load real vibration data from the dataset.
    
    Args:
        data_directory: Path to the data directory
        n_samples: Number of samples to load
        
    Returns:
        Tuple of (data, labels, file_paths)
    """
    try:
        print(f"Loading real data from: {data_directory}")
        
        # Try to use the existing data loader
        train_loader, val_loader, test_loader, _ = stratified_group_split(data_directory)
        
        # Get a few samples from the test set
        data_samples = []
        labels_samples = []
        count = 0
        
        for data_batch, label_batch in test_loader:
            if count >= n_samples:
                break
                
            batch_size = data_batch.shape[0]
            for i in range(min(batch_size, n_samples - count)):
                data_samples.append(data_batch[i].numpy())
                labels_samples.append(label_batch[i].numpy())
                count += 1
                
                if count >= n_samples:
                    break
        
        if len(data_samples) > 0:
            data = np.stack(data_samples, axis=0)
            labels = np.array(labels_samples)
            print(f"Loaded {len(data_samples)} real samples with shape: {data.shape}")
            return data, labels, None
        else:
            print("No real data found, falling back to synthetic data")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Falling back to synthetic data")
        return None, None, None


def analyze_single_sample(explainer: DFTLRPExplainer,
                         data: np.ndarray,
                         sample_idx: int = 0,
                         target_class: int = None,
                         output_dir: str = "./analysis_output") -> dict:
    """
    Perform comprehensive analysis on a single sample.
    
    Args:
        explainer: DFTLRPExplainer instance
        data: Input data array
        sample_idx: Index of sample to analyze
        target_class: Target class for explanation
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Sample {sample_idx}")
    print(f"{'='*60}")
    
    # Extract single sample if batch provided
    if len(data.shape) == 3 and data.shape[0] > 1:
        sample_data = data[sample_idx:sample_idx+1]
    else:
        sample_data = data
    
    print(f"Sample shape: {sample_data.shape}")
    
    # Perform comprehensive analysis
    try:
        results = explainer.analyze_sample(
            data=sample_data,
            target_class=target_class,
            include_frequency=True,
            include_time_frequency=True
        )
        
        # Print analysis summary
        print("\nAnalysis Summary:")
        print("-" * 40)
        
        if 'time' in results and results['time'] is not None:
            pred = results['time']['prediction'][0] if hasattr(results['time']['prediction'], '__len__') else results['time']['prediction']
            probs = results['time']['probabilities'][0] if len(results['time']['probabilities'].shape) > 1 else results['time']['probabilities']
            class_names = ['Good', 'Bad']
            
            print(f"Predicted Class: {class_names[pred]} (class {pred})")
            print(f"Confidence: {probs[pred]:.3f}")
            print(f"Class Probabilities: Good={probs[0]:.3f}, Bad={probs[1]:.3f}")
            
            if target_class is not None:
                print(f"Target Class: {class_names[target_class]} (class {target_class})")
                if target_class == pred:
                    print("✓ Prediction matches target")
                else:
                    print("✗ Prediction differs from target")
        
        # Generate detailed visualizations
        print(f"\nGenerating visualizations...")
        saved_plots = create_analysis_report(
            results=results,
            sample_idx=0,  # We're analyzing a single sample, so index is 0
            output_dir=output_dir,
            show_plots=False  # Don't show plots in batch mode
        )
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


def batch_analysis(explainer: DFTLRPExplainer,
                  data: np.ndarray,
                  labels: np.ndarray = None,
                  output_dir: str = "./batch_analysis",
                  max_samples: int = 5) -> dict:
    """
    Perform analysis on multiple samples.
    
    Args:
        explainer: DFTLRPExplainer instance
        data: Input data array
        labels: Ground truth labels (optional)
        output_dir: Output directory for results
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary containing batch analysis results
    """
    print(f"\n{'='*60}")
    print(f"Batch Analysis of {min(len(data), max_samples)} Samples")
    print(f"{'='*60}")
    
    batch_results = {}
    n_samples = min(len(data), max_samples)
    
    for i in range(n_samples):
        sample_output_dir = os.path.join(output_dir, f"sample_{i}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        target_class = labels[i] if labels is not None else None
        
        print(f"\nProcessing sample {i+1}/{n_samples}...")
        
        try:
            results = analyze_single_sample(
                explainer=explainer,
                data=data,
                sample_idx=i,
                target_class=target_class,
                output_dir=sample_output_dir
            )
            
            batch_results[f'sample_{i}'] = results
            
            # Force cleanup between samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
            continue
    
    print(f"\nBatch analysis completed!")
    print(f"Results saved to: {output_dir}")
    
    return batch_results


def interactive_analysis(explainer: DFTLRPExplainer,
                        data: np.ndarray,
                        labels: np.ndarray = None):
    """
    Run interactive analysis mode with user input.
    
    Args:
        explainer: DFTLRPExplainer instance
        data: Input data array
        labels: Ground truth labels (optional)
    """
    print(f"\n{'='*60}")
    print("Interactive Analysis Mode")
    print(f"{'='*60}")
    print(f"Available samples: 0 to {len(data)-1}")
    
    while True:
        try:
            user_input = input("\nEnter sample index to analyze (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            sample_idx = int(user_input)
            
            if sample_idx < 0 or sample_idx >= len(data):
                print(f"Invalid sample index. Must be between 0 and {len(data)-1}")
                continue
            
            target_class = labels[sample_idx] if labels is not None else None
            output_dir = f"./interactive_analysis/sample_{sample_idx}"
            
            results = analyze_single_sample(
                explainer=explainer,
                data=data,
                sample_idx=sample_idx,
                target_class=target_class,
                output_dir=output_dir
            )
            
            # Show summary plot
            if results:
                fig = plot_comparison_summary(results, sample_idx=0)
                plt.show()
                plt.close(fig)
        
        except ValueError:
            print("Please enter a valid integer or 'quit'")
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error during analysis: {e}")


def main():
    """Main function to run the CNN1D explanation analysis."""
    parser = argparse.ArgumentParser(
        description="CNN1D Model Explainability Analysis using DFT-LRP"
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='./cnn1d_model.pth',
        help='Path to the trained CNN1D model file'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='./data/final/new_selection/normalized_windowed_downsampled_data_lessBAD',
        help='Path to the data directory'
    )
    
    parser.add_argument(
        '--sample_idx', 
        type=int, 
        default=0,
        help='Index of sample to analyze (for single sample mode)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./dft_lrp_analysis',
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
        help='Batch size for processing (for memory management)'
    )
    
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=5,
        help='Maximum number of samples to analyze in batch mode'
    )
    
    parser.add_argument(
        '--use_synthetic', 
        action='store_true',
        help='Use synthetic data instead of real data'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['single', 'batch', 'interactive'],
        default='single',
        help='Analysis mode: single sample, batch processing, or interactive'
    )
    
    parser.add_argument(
        '--precision', 
        type=int, 
        choices=[16, 32],
        default=32,
        help='Numerical precision for DFT-LRP (16 or 32)'
    )
    
    parser.add_argument(
        '--target_class', 
        type=int, 
        choices=[0, 1],
        default=None,
        help='Target class for explanation (0=Good, 1=Bad). If None, uses predicted class.'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("CNN1D Model Explainability Analysis using DFT-LRP")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    try:
        model = load_trained_model(args.model_path, device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model for demonstration...")
        model = CNN1D_Wide()
        model.to(device)
        model.eval()
    
    # Load data
    data, labels = None, None
    
    if not args.use_synthetic:
        data, labels, _ = load_real_data(args.data_path, args.max_samples)
    
    if data is None:
        print("\nGenerating synthetic data for demonstration...")
        data = create_sample_data(
            batch_size=args.max_samples, 
            signal_length=2000, 
            n_channels=3,
            noise_level=0.1
        ).numpy()
        labels = np.random.choice([0, 1], size=args.max_samples)  # Random labels
        print(f"✓ Generated synthetic data with shape: {data.shape}")
    
    # Initialize explainer
    print(f"\nInitializing DFT-LRP explainer...")
    try:
        explainer = DFTLRPExplainer(
            model=model,
            signal_length=data.shape[-1],
            device=device,
            precision=args.precision
        )
        print("✓ DFT-LRP explainer initialized successfully")
    except Exception as e:
        print(f"Error initializing explainer: {e}")
        return
    
    # Run analysis based on mode
    try:
        if args.mode == 'single':
            results = analyze_single_sample(
                explainer=explainer,
                data=data,
                sample_idx=args.sample_idx,
                target_class=args.target_class,
                output_dir=args.output_dir
            )
            
            # Show summary plot
            if results:
                print("\nDisplaying summary plot...")
                fig = plot_comparison_summary(results, sample_idx=0)
                plt.show()
                plt.close(fig)
        
        elif args.mode == 'batch':
            batch_results = batch_analysis(
                explainer=explainer,
                data=data,
                labels=labels,
                output_dir=args.output_dir,
                max_samples=args.max_samples
            )
        
        elif args.mode == 'interactive':
            interactive_analysis(
                explainer=explainer,
                data=data,
                labels=labels
            )
        
        print(f"\n{'='*60}")
        print("Analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            del explainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


if __name__ == "__main__":
    main()