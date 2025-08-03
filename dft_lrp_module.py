"""
DFT-LRP Module for Time Series Classification with CNN1D_Wide

This module provides a clean interface to the DFT-LRP functionality for explaining
CNN1D_Wide model predictions on vibration data. It builds on the existing 
implementations in utils/dft_lrp.py and utils/lrp_utils.py.

Based on the work from jvielhaben/DFT-LRP repository.
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from typing import Tuple, Optional, Union, Dict
import warnings

# Import existing utilities
from utils.dft_lrp import DFTLRP
from utils.lrp_utils import zennit_relevance
from Classification.cnn1D_model import CNN1D_Wide


class DFTLRPExplainer:
    """
    Main class for explaining CNN1D_Wide predictions using both standard LRP and DFT-LRP.
    
    This class provides a unified interface for:
    1. Standard LRP in time domain
    2. DFT-LRP for frequency domain relevance
    3. Short-time DFT-LRP for time-frequency analysis
    
    Attributes:
        model: The CNN1D_Wide model to explain
        signal_length: Length of input signals (default: 2000)
        device: PyTorch device (CPU/CUDA)
        precision: Numerical precision (32 or 16)
        dft_lrp: DFTLRP instance for frequency domain analysis
    """
    
    def __init__(self, 
                 model: CNN1D_Wide,
                 signal_length: int = 2000,
                 device: Optional[torch.device] = None,
                 precision: int = 32,
                 use_cuda: bool = None):
        """
        Initialize the DFT-LRP explainer.
        
        Args:
            model: Trained CNN1D_Wide model
            signal_length: Length of input time series (default 2000)
            device: PyTorch device, if None will auto-detect
            precision: Numerical precision, 32 or 16 (default 32)
            use_cuda: Whether to use CUDA, if None will auto-detect
        """
        self.model = model
        self.signal_length = signal_length
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if use_cuda is None:
            self.use_cuda = self.device.type == 'cuda'
        else:
            self.use_cuda = use_cuda
            
        self.precision = precision
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize DFT-LRP with memory-efficient settings
        self._init_dft_lrp()
        
        print(f"DFTLRPExplainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Signal length: {self.signal_length}")
        print(f"  Precision: {self.precision}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_dft_lrp(self):
        """Initialize DFT-LRP instances with memory management."""
        try:
            # Standard DFT-LRP
            self.dft_lrp = DFTLRP(
                signal_length=self.signal_length,
                precision=self.precision,
                cuda=self.use_cuda,
                leverage_symmetry=True,  # Use symmetry for real signals
                create_inverse=True,
                create_transpose_inverse=True,
                create_forward=True,
                create_dft=True,
                create_stdft=False  # Don't create STDFT by default to save memory
            )
            
            # Short-time DFT-LRP (create on demand)
            self.stdft_lrp = None
            
        except Exception as e:
            print(f"Warning: Could not initialize DFT-LRP: {e}")
            print("DFT-LRP analysis will not be available.")
            self.dft_lrp = None
    
    def _init_stdft_lrp(self, window_width: int = 256, window_shift: int = 128, window_shape: str = "halfsine"):
        """Initialize Short-Time DFT-LRP on demand."""
        if self.stdft_lrp is None:
            try:
                self.stdft_lrp = DFTLRP(
                    signal_length=self.signal_length,
                    precision=self.precision,
                    cuda=self.use_cuda,
                    leverage_symmetry=True,
                    window_width=window_width,
                    window_shift=window_shift,
                    window_shape=window_shape,
                    create_inverse=True,
                    create_transpose_inverse=True,
                    create_forward=True,
                    create_dft=False,  # Don't create standard DFT
                    create_stdft=True
                )
            except Exception as e:
                print(f"Warning: Could not initialize STDFT-LRP: {e}")
                self.stdft_lrp = None
    
    def explain_time_domain(self, 
                           data: Union[torch.Tensor, np.ndarray],
                           target_class: Optional[int] = None,
                           attribution_method: str = "lrp",
                           lrp_rule: str = "EpsilonPlus") -> Dict:
        """
        Generate relevance explanations in the time domain using standard LRP.
        
        Args:
            data: Input data of shape (batch_size, channels, time) or (channels, time)
            target_class: Target class to explain (if None, uses predicted class)
            attribution_method: Attribution method ("lrp", "gxi", "sensitivity", "ig")
            lrp_rule: LRP rule to use when attribution_method="lrp"
            
        Returns:
            Dictionary containing:
                - relevance: Relevance scores in time domain
                - prediction: Model prediction
                - target_class: Target class used for explanation
                - probabilities: Class probabilities
        """
        # Prepare input data
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        if len(data.shape) == 2:  # Add batch dimension
            data = data.unsqueeze(0)
            
        data = data.to(self.device)
        batch_size = data.shape[0]
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        # Determine target class
        if target_class is None:
            target_class = predictions.cpu().numpy()
        elif isinstance(target_class, int):
            target_class = np.array([target_class] * batch_size)
        
        # Compute relevance using Zennit
        try:
            relevance = zennit_relevance(
                input=data,
                model=self.model,
                target=target_class,
                attribution_method=attribution_method,
                zennit_choice=lrp_rule,
                rel_is_model_out=True,
                cuda=self.use_cuda
            )
        except Exception as e:
            print(f"Error computing relevance: {e}")
            relevance = np.zeros_like(data.cpu().numpy())
        
        return {
            'relevance': relevance,
            'prediction': predictions.cpu().numpy(),
            'target_class': target_class,
            'probabilities': probabilities.cpu().numpy(),
            'input_data': data.cpu().numpy()
        }
    
    def explain_frequency_domain(self,
                                data: Union[torch.Tensor, np.ndarray],
                                target_class: Optional[int] = None,
                                attribution_method: str = "lrp",
                                lrp_rule: str = "EpsilonPlus",
                                batch_size: int = 8,
                                epsilon: float = 1e-6) -> Dict:
        """
        Generate relevance explanations in the frequency domain using DFT-LRP.
        
        Args:
            data: Input data of shape (batch_size, channels, time) or (channels, time)
            target_class: Target class to explain (if None, uses predicted class)
            attribution_method: Attribution method for time domain relevance
            lrp_rule: LRP rule to use
            batch_size: Batch size for processing to manage memory
            epsilon: Epsilon value for DFT-LRP stability
            
        Returns:
            Dictionary containing:
                - relevance_time: Relevance scores in time domain
                - relevance_freq: Relevance scores in frequency domain  
                - signal_freq: Signal in frequency domain
                - frequencies: Frequency values
                - prediction: Model prediction
                - target_class: Target class used
                - probabilities: Class probabilities
        """
        if self.dft_lrp is None:
            raise RuntimeError("DFT-LRP not available. Check initialization.")
        
        # Get time domain relevance first
        time_result = self.explain_time_domain(
            data=data, 
            target_class=target_class,
            attribution_method=attribution_method,
            lrp_rule=lrp_rule
        )
        
        relevance_time = time_result['relevance']
        input_data = time_result['input_data']
        
        # Process each sample in batches to manage memory
        n_samples = relevance_time.shape[0]
        n_channels = relevance_time.shape[1]
        
        # Frequency domain results
        signal_freq_all = []
        relevance_freq_all = []
        
        try:
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch_relevance = relevance_time[i:end_idx]
                batch_signal = input_data[i:end_idx]
                
                # Apply DFT-LRP to each axis separately for better memory management
                batch_signal_freq = []
                batch_relevance_freq = []
                
                for axis in range(n_channels):
                    axis_signal = batch_signal[:, axis:axis+1, :]
                    axis_relevance = batch_relevance[:, axis:axis+1, :]
                    
                    # Apply DFT-LRP
                    signal_freq, relevance_freq = self.dft_lrp.dft_lrp(
                        relevance=axis_relevance,
                        signal=axis_signal,
                        epsilon=epsilon,
                        real=False  # Return complex values
                    )
                    
                    batch_signal_freq.append(signal_freq)
                    batch_relevance_freq.append(relevance_freq)
                
                # Combine axes
                batch_signal_freq = np.concatenate(batch_signal_freq, axis=1)
                batch_relevance_freq = np.concatenate(batch_relevance_freq, axis=1)
                
                signal_freq_all.append(batch_signal_freq)
                relevance_freq_all.append(batch_relevance_freq)
                
                # Force cleanup
                gc.collect()
                if self.use_cuda:
                    torch.cuda.empty_cache()
        
            # Combine all batches
            signal_freq = np.concatenate(signal_freq_all, axis=0)
            relevance_freq = np.concatenate(relevance_freq_all, axis=0)
            
        except Exception as e:
            print(f"Error in DFT-LRP computation: {e}")
            # Fallback: create empty frequency domain results
            freq_length = self.signal_length // 2 + 1
            signal_freq = np.zeros((n_samples, n_channels, freq_length), dtype=np.complex64)
            relevance_freq = np.zeros((n_samples, n_channels, freq_length), dtype=np.complex64)
        
        # Create frequency values
        frequencies = np.fft.rfftfreq(self.signal_length, d=1.0)  # Assuming unit sampling rate
        
        return {
            'relevance_time': relevance_time,
            'relevance_freq': relevance_freq,
            'signal_freq': signal_freq,
            'frequencies': frequencies,
            'prediction': time_result['prediction'],
            'target_class': time_result['target_class'],
            'probabilities': time_result['probabilities'],
            'input_data': input_data
        }
    
    def explain_time_frequency_domain(self,
                                     data: Union[torch.Tensor, np.ndarray],
                                     target_class: Optional[int] = None,
                                     attribution_method: str = "lrp",
                                     lrp_rule: str = "EpsilonPlus",
                                     window_width: int = 256,
                                     window_shift: int = 128,
                                     window_shape: str = "halfsine",
                                     epsilon: float = 1e-6) -> Dict:
        """
        Generate relevance explanations in the time-frequency domain using Short-Time DFT-LRP.
        
        Args:
            data: Input data of shape (batch_size, channels, time) or (channels, time)
            target_class: Target class to explain
            attribution_method: Attribution method for time domain relevance
            lrp_rule: LRP rule to use
            window_width: STFT window width
            window_shift: STFT window shift (hop size)
            window_shape: Window shape ("halfsine" or "rectangle")
            epsilon: Epsilon value for DFT-LRP stability
            
        Returns:
            Dictionary containing time-frequency relevance analysis
        """
        # Initialize STDFT-LRP if needed
        self._init_stdft_lrp(window_width, window_shift, window_shape)
        
        if self.stdft_lrp is None:
            raise RuntimeError("Short-Time DFT-LRP not available.")
        
        # Get time domain relevance first
        time_result = self.explain_time_domain(
            data=data,
            target_class=target_class,
            attribution_method=attribution_method,
            lrp_rule=lrp_rule
        )
        
        relevance_time = time_result['relevance']
        input_data = time_result['input_data']
        
        try:
            # Apply Short-Time DFT-LRP
            signal_stft, relevance_stft = self.stdft_lrp.dft_lrp(
                relevance=relevance_time,
                signal=input_data,
                short_time=True,
                epsilon=epsilon,
                real=False
            )
            
            # Calculate time and frequency axes for STFT
            n_windows = signal_stft.shape[-1] if len(signal_stft.shape) > 2 else 1
            time_axis = np.arange(n_windows) * window_shift
            freq_axis = np.fft.rfftfreq(window_width, d=1.0)
            
        except Exception as e:
            print(f"Error in Short-Time DFT-LRP: {e}")
            # Create fallback results
            n_samples, n_channels = relevance_time.shape[:2]
            n_windows = (self.signal_length - window_width) // window_shift + 1
            freq_bins = window_width // 2 + 1
            
            signal_stft = np.zeros((n_samples, n_channels, n_windows, freq_bins), dtype=np.complex64)
            relevance_stft = np.zeros((n_samples, n_channels, n_windows, freq_bins), dtype=np.complex64)
            time_axis = np.arange(n_windows) * window_shift
            freq_axis = np.fft.rfftfreq(window_width, d=1.0)
        
        return {
            'relevance_time': relevance_time,
            'relevance_stft': relevance_stft,
            'signal_stft': signal_stft,
            'time_axis': time_axis,
            'freq_axis': freq_axis,
            'window_width': window_width,
            'window_shift': window_shift,
            'prediction': time_result['prediction'],
            'target_class': time_result['target_class'],
            'probabilities': time_result['probabilities'],
            'input_data': input_data
        }
    
    def analyze_sample(self,
                      data: Union[torch.Tensor, np.ndarray],
                      target_class: Optional[int] = None,
                      include_frequency: bool = True,
                      include_time_frequency: bool = False,  # Changed default to False
                      **kwargs) -> Dict:
        """
        Comprehensive analysis of a single sample or batch using all available methods.
        
        Args:
            data: Input data
            target_class: Target class to explain
            include_frequency: Whether to include frequency domain analysis
            include_time_frequency: Whether to include time-frequency analysis
            **kwargs: Additional arguments passed to analysis methods
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Time domain analysis (always performed)
        print("Computing time domain relevance...")
        results['time'] = self.explain_time_domain(data, target_class, **kwargs)
        
        # Frequency domain analysis
        if include_frequency and self.dft_lrp is not None:
            print("Computing frequency domain relevance...")
            try:
                results['frequency'] = self.explain_frequency_domain(data, target_class, **kwargs)
            except Exception as e:
                print(f"Frequency domain analysis failed: {e}")
                results['frequency'] = None
        
        # Time-frequency domain analysis
        if include_time_frequency:
            print("Computing time-frequency domain relevance...")
            try:
                results['time_frequency'] = self.explain_time_frequency_domain(data, target_class, **kwargs)
            except Exception as e:
                print(f"Time-frequency domain analysis failed: {e}")
                results['time_frequency'] = None
        
        return results
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'dft_lrp') and self.dft_lrp is not None:
                del self.dft_lrp
            if hasattr(self, 'stdft_lrp') and self.stdft_lrp is not None:
                del self.stdft_lrp
            
            if self.use_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def load_trained_model(model_path: str, device: Optional[torch.device] = None) -> CNN1D_Wide:
    """
    Load a trained CNN1D_Wide model from file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Loaded CNN1D_Wide model in eval mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model instance
    model = CNN1D_Wide()
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def create_sample_data(batch_size: int = 1, 
                      signal_length: int = 2000, 
                      n_channels: int = 3,
                      noise_level: float = 0.1) -> torch.Tensor:
    """
    Create synthetic vibration-like data for testing.
    
    Args:
        batch_size: Number of samples
        signal_length: Length of each signal
        n_channels: Number of channels (X, Y, Z axes)
        noise_level: Level of random noise to add
        
    Returns:
        Synthetic data tensor of shape (batch_size, n_channels, signal_length)
    """
    # Create time axis
    t = torch.linspace(0, 10, signal_length)
    
    # Create synthetic signals with different frequencies for each channel
    data = torch.zeros(batch_size, n_channels, signal_length)
    
    for i in range(batch_size):
        for ch in range(n_channels):
            # Different frequency components for each channel
            freq1 = 2 + ch * 0.5  # Base frequency
            freq2 = 10 + ch * 2   # Higher frequency component
            
            # Create signal with multiple frequency components
            signal = (torch.sin(2 * np.pi * freq1 * t) + 
                     0.5 * torch.sin(2 * np.pi * freq2 * t) +
                     0.3 * torch.sin(2 * np.pi * freq2 * 2 * t))
            
            # Add some random noise
            noise = noise_level * torch.randn_like(signal)
            data[i, ch, :] = signal + noise
    
    return data