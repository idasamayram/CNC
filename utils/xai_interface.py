"""
XAI Interface - Unified access to all XAI methods for vibration data models

This module provides a simple interface to all implemented XAI methods:
- LRP (Layer-wise Relevance Propagation)
- DFT-LRP (Discrete Fourier Transform with LRP)
- FFT-LRP (Fast Fourier Transform with LRP)
- Gradient-based methods
- SmoothGrad
- Occlusion-based methods

Each method can be accessed through a unified API that handles device management,
visualization, and result formatting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.xai_implementation import compute_lrp_relevance, compute_dft_lrp_relevance, compute_fft_lrp_relevance
from utils.baseline_xai import (
    gradient_relevance, grad_times_input_relevance, smoothgrad_relevance,
    occlusion_signal_relevance, occlusion_simpler_relevance, summarize_attributions
)

class XAIExplainer:
    """Unified interface for XAI methods on vibration data models."""
    
    def __init__(self, model, device=None):
        """
        Initialize XAI explainer with a model.
        
        Args:
            model: PyTorch model to explain
            device: Device to use (defaults to CUDA if available)
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def explain(self, sample, method="lrp", label=None, **kwargs):
        """
        Generate explanation for a sample using specified method.
        
        Args:
            sample: Input sample (3, time_steps) as numpy array or tensor
            method: XAI method to use ("lrp", "dft_lrp", "fft_lrp", "gradient", 
                    "grad_x_input", "smoothgrad", "occlusion", "occlusion_window")
            label: Target label (uses model prediction if None)
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with explanation results and metadata
        """
        # Ensure sample is properly formatted
        if isinstance(sample, np.ndarray):
            sample = torch.tensor(sample, dtype=torch.float32)
        sample = sample.to(self.device)
        
        # Apply selected XAI method
        result = {}
        
        if method == "lrp":
            relevance, input_signal, pred_label = compute_lrp_relevance(
                self.model, sample, label, device=self.device
            )
            result = {
                "relevance": relevance,
                "input_signal": input_signal,
                "predicted_label": pred_label,
                "method": "LRP"
            }
            
        elif method == "dft_lrp":
            relevance_time, relevance_freq, signal_freq, input_signal, freqs, pred_label = compute_dft_lrp_relevance(
                self.model, sample, label, device=self.device, **kwargs
            )
            result = {
                "time_relevance": relevance_time,
                "freq_relevance": relevance_freq,
                "signal_freq": signal_freq,
                "input_signal": input_signal,
                "frequencies": freqs,
                "predicted_label": pred_label,
                "method": "DFT-LRP"
            }
            
        elif method == "fft_lrp":
            (relevance_time, relevance_freq, signal_freq, 
             relevance_timefreq, signal_timefreq, input_signal, 
             freqs, pred_label) = compute_fft_lrp_relevance(
                self.model, sample, label, device=self.device, **kwargs
            )
            result = {
                "time_relevance": relevance_time,
                "freq_relevance": relevance_freq,
                "signal_freq": signal_freq,
                "timefreq_relevance": relevance_timefreq,
                "timefreq_signal": signal_timefreq,
                "input_signal": input_signal,
                "frequencies": freqs,
                "predicted_label": pred_label,
                "method": "FFT-LRP"
            }
            
        elif method == "gradient":
            relevance, pred_label = gradient_relevance(self.model, sample, label)
            result = {
                "relevance": relevance.cpu().numpy(),
                "predicted_label": pred_label,
                "method": "Gradient"
            }
            
        elif method == "grad_x_input":
            relevance, pred_label = grad_times_input_relevance(self.model, sample, label)
            result = {
                "relevance": relevance.cpu().numpy(),
                "predicted_label": pred_label,
                "method": "Gradient × Input"
            }
            
        elif method == "smoothgrad":
            relevance, pred_label = smoothgrad_relevance(
                self.model, sample, num_samples=kwargs.get('num_samples', 40), 
                noise_level=kwargs.get('noise_level', 1), target=label
            )
            result = {
                "relevance": relevance.cpu().numpy(),
                "predicted_label": pred_label,
                "method": "SmoothGrad"
            }
            
        elif method == "occlusion":
            relevance, pred_label = occlusion_signal_relevance(
                self.model, sample, target=label, 
                occlusion_type=kwargs.get('occlusion_type', 'zero')
            )
            result = {
                "relevance": relevance.cpu().numpy(),
                "predicted_label": pred_label,
                "method": f"Occlusion ({kwargs.get('occlusion_type', 'zero')})"
            }
            
        elif method == "occlusion_window":
            relevance, pred_label = occlusion_simpler_relevance(
                self.model, sample, target=label,
                occlusion_type=kwargs.get('occlusion_type', 'zero'),
                window_size=kwargs.get('window_size', 40)
            )
            result = {
                "relevance": relevance.cpu().numpy(),
                "predicted_label": pred_label,
                "method": f"Window Occlusion ({kwargs.get('occlusion_type', 'zero')})"
            }
        
        else:
            raise ValueError(f"Unknown XAI method: {method}")
        
        # Add summary statistics
        if "relevance" in result:
            result["summary"] = summarize_attributions(result["relevance"])
        
        return result
    
    def visualize(self, result, plot_type="time", figsize=(12, 8), title=None):
        """
        Visualize XAI results.
        
        Args:
            result: Result dictionary from explain() method
            plot_type: Type of plot ("time", "frequency", "time_frequency", "summary")
            figsize: Figure size as tuple
            title: Optional title override
            
        Returns:
            Matplotlib figure object
        """
        method = result.get("method", "")
        
        if plot_type == "time":
            # Time-domain visualization
            fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            input_signal = result.get("input_signal")
            relevance = result.get("relevance", result.get("time_relevance"))
            
            if input_signal is not None and relevance is not None:
                for i, ax in enumerate(axs):
                    # Plot signal
                    ax.plot(input_signal[i], color='blue', alpha=0.5, label='Signal')
                    
                    # Plot relevance as filled area
                    ax.fill_between(
                        range(len(relevance[i])), 
                        0, 
                        relevance[i], 
                        where=(relevance[i] > 0),
                        color='green', alpha=0.5, label='Positive Relevance'
                    )
                    ax.fill_between(
                        range(len(relevance[i])), 
                        0, 
                        relevance[i], 
                        where=(relevance[i] < 0),
                        color='red', alpha=0.5, label='Negative Relevance'
                    )
                    
                    ax.set_ylabel(f'Axis {i}')
                    if i == 0:
                        ax.legend()
                
                axs[-1].set_xlabel('Time steps')
                plt_title = title or f"{method} Explanation (Time Domain)"
                fig.suptitle(plt_title)
            
        elif plot_type == "frequency":
            # Frequency-domain visualization
            fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            freq_relevance = result.get("freq_relevance")
            signal_freq = result.get("signal_freq")
            freqs = result.get("frequencies")
            
            if freq_relevance is not None and signal_freq is not None and freqs is not None:
                for i, ax in enumerate(axs):
                    # Plot magnitude spectrum
                    ax.plot(freqs, np.abs(signal_freq[i]), color='blue', alpha=0.5, label='Magnitude Spectrum')
                    
                    # Plot frequency relevance
                    if np.iscomplexobj(freq_relevance[i]):
                        relevance_plot = np.abs(freq_relevance[i])
                    else:
                        relevance_plot = freq_relevance[i]
                        
                    ax.plot(freqs, relevance_plot, color='red', alpha=0.7, label='Frequency Relevance')
                    
                    ax.set_ylabel(f'Axis {i}')
                    if i == 0:
                        ax.legend()
                
                axs[-1].set_xlabel('Frequency (Hz)')
                plt_title = title or f"{method} Explanation (Frequency Domain)"
                fig.suptitle(plt_title)
                
        elif plot_type == "time_frequency":
            # Time-frequency domain visualization (for FFT-LRP)
            timefreq_relevance = result.get("timefreq_relevance")
            
            if timefreq_relevance is not None:
                fig, axs = plt.subplots(3, 1, figsize=figsize)
                
                for i, ax in enumerate(axs):
                    # Display time-frequency relevance as a heatmap
                    if np.iscomplexobj(timefreq_relevance[i]):
                        rel_tf = np.abs(timefreq_relevance[i])
                    else:
                        rel_tf = timefreq_relevance[i]
                        
                    im = ax.imshow(
                        rel_tf, 
                        aspect='auto', 
                        origin='lower',
                        cmap='viridis'
                    )
                    
                    ax.set_ylabel(f'Frequency (Axis {i})')
                    
                axs[-1].set_xlabel('Time frames')
                plt_title = title or f"{method} Explanation (Time-Frequency Domain)"
                fig.suptitle(plt_title)
                fig.colorbar(im, ax=axs)
                
        elif plot_type == "summary":
            # Summary visualization
            summary = result.get("summary")
            
            if summary:
                fig, axs = plt.subplots(3, 1, figsize=figsize)
                
                for i, ax in enumerate(axs):
                    axis_summary = summary.get(f"Axis {i}")
                    if axis_summary:
                        metrics = [
                            "Total Positive Relevance", 
                            "Total Negative Relevance", 
                            "Total Relevance (Pos + Neg)"
                        ]
                        values = [axis_summary.get(metric, 0) for metric in metrics]
                        
                        ax.bar(metrics, values, color=['green', 'red', 'blue'])
                        ax.set_ylabel(f'Relevance (Axis {i})')
                        ax.tick_params(axis='x', rotation=45)
                
                plt_title = title or f"{method} Explanation (Summary)"
                fig.suptitle(plt_title)
                fig.tight_layout()
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return fig

# Example usage
if __name__ == "__main__":
    # Load a pre-trained model
    # model = torch.load("../cnn1d_freq_model.ckpt")
    
    # Create explainer
    # explainer = XAIExplainer(model)
    
    # Generate and visualize explanations
    # sample = load_sample_function(...) 
    # result = explainer.explain(sample, method="dft_lrp")
    # explainer.visualize(result, plot_type="time").show()
    # explainer.visualize(result, plot_type="frequency").show()
    pass
