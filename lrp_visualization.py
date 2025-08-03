"""
LRP Visualization Module for Time Series Classification

This module provides comprehensive visualization functions for displaying 
relevance scores from DFT-LRP analysis in time, frequency, and time-frequency domains.

Based on the work from jvielhaben/DFT-LRP repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings


# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")


def plot_time_domain_relevance(data: np.ndarray,
                              relevance: np.ndarray,
                              time_axis: Optional[np.ndarray] = None,
                              channel_names: Optional[List[str]] = None,
                              title: str = "Time Domain Relevance",
                              figsize: Tuple[int, int] = (14, 8),
                              sample_idx: int = 0,
                              save_path: Optional[str] = None,
                              show_colorbar: bool = True) -> plt.Figure:
    """
    Visualize relevance scores in the time domain.
    
    Args:
        data: Input data of shape (batch_size, channels, time)
        relevance: Relevance scores of same shape as data
        time_axis: Time axis values (if None, use indices)
        channel_names: Names for each channel (default: X, Y, Z)
        title: Plot title
        figsize: Figure size
        sample_idx: Which sample to plot from batch
        save_path: Path to save figure (optional)
        show_colorbar: Whether to show colorbar for relevance
        
    Returns:
        matplotlib Figure object
    """
    if len(data.shape) == 2:  # Single sample
        data = data[np.newaxis, ...]
        relevance = relevance[np.newaxis, ...]
    
    n_samples, n_channels, signal_length = data.shape
    
    if sample_idx >= n_samples:
        raise ValueError(f"sample_idx {sample_idx} >= n_samples {n_samples}")
    
    if time_axis is None:
        time_axis = np.arange(signal_length)
    
    if channel_names is None:
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
    
    # Create subplot layout
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Get sample data
    sample_data = data[sample_idx]
    sample_relevance = relevance[sample_idx]
    
    # Normalize relevance for better visualization
    vmax = np.max(np.abs(sample_relevance))
    vmin = -vmax
    
    for ch in range(n_channels):
        ax = axes[ch]
        
        # Plot original signal
        ax.plot(time_axis, sample_data[ch], 'k-', alpha=0.7, linewidth=1, label='Signal')
        
        # Create relevance heatmap
        relevance_2d = sample_relevance[ch][np.newaxis, :]
        im = ax.imshow(relevance_2d, aspect='auto', cmap='RdBu_r', 
                      vmin=vmin, vmax=vmax, alpha=0.8,
                      extent=[time_axis[0], time_axis[-1], 
                             np.min(sample_data[ch]), np.max(sample_data[ch])])
        
        # Styling
        ax.set_ylabel(f'{channel_names[ch]}\nAmplitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add colorbar for the last subplot
        if ch == n_channels - 1 and show_colorbar:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label('Relevance Score', fontsize=10)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_frequency_domain_relevance(signal_freq: np.ndarray,
                                   relevance_freq: np.ndarray,
                                   frequencies: np.ndarray,
                                   channel_names: Optional[List[str]] = None,
                                   title: str = "Frequency Domain Relevance",
                                   figsize: Tuple[int, int] = (14, 8),
                                   sample_idx: int = 0,
                                   plot_type: str = "magnitude",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize relevance scores in the frequency domain.
    
    Args:
        signal_freq: Signal in frequency domain (complex)
        relevance_freq: Relevance in frequency domain (complex)
        frequencies: Frequency axis values
        channel_names: Names for each channel
        title: Plot title
        figsize: Figure size
        sample_idx: Which sample to plot from batch
        plot_type: Type of plot ("magnitude", "phase", "both")
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if len(signal_freq.shape) == 2:  # Single sample
        signal_freq = signal_freq[np.newaxis, ...]
        relevance_freq = relevance_freq[np.newaxis, ...]
    
    n_samples, n_channels, freq_length = signal_freq.shape
    
    if sample_idx >= n_samples:
        raise ValueError(f"sample_idx {sample_idx} >= n_samples {n_samples}")
    
    if channel_names is None:
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
    
    # Get sample data
    sample_signal = signal_freq[sample_idx]
    sample_relevance = relevance_freq[sample_idx]
    
    # Create subplot layout based on plot type
    if plot_type == "both":
        fig, axes = plt.subplots(n_channels, 2, figsize=(figsize[0], figsize[1]), sharex=True)
        if n_channels == 1:
            axes = axes[np.newaxis, :]
    else:
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
        if n_channels == 1:
            axes = [axes]
    
    for ch in range(n_channels):
        if plot_type == "both":
            ax_mag, ax_phase = axes[ch]
        else:
            ax_mag = axes[ch]
        
        # Magnitude plot
        signal_mag = np.abs(sample_signal[ch])
        relevance_mag = np.abs(sample_relevance[ch])
        
        ax_mag.semilogy(frequencies, signal_mag, 'k-', alpha=0.7, linewidth=1, label='Signal Magnitude')
        ax_mag.semilogy(frequencies, relevance_mag, 'r-', linewidth=2, label='Relevance Magnitude')
        
        ax_mag.set_ylabel(f'{channel_names[ch]}\nMagnitude', fontsize=10)
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend()
        ax_mag.set_title(f'Magnitude Spectrum - {channel_names[ch]}' if plot_type != "both" else 'Magnitude')
        
        # Phase plot (if requested)
        if plot_type == "both":
            signal_phase = np.angle(sample_signal[ch])
            relevance_phase = np.angle(sample_relevance[ch])
            
            ax_phase.plot(frequencies, signal_phase, 'k-', alpha=0.7, linewidth=1, label='Signal Phase')
            ax_phase.plot(frequencies, relevance_phase, 'r-', linewidth=2, label='Relevance Phase')
            
            ax_phase.set_ylabel('Phase (rad)', fontsize=10)
            ax_phase.grid(True, alpha=0.3)
            ax_phase.legend()
            ax_phase.set_title('Phase Spectrum')
        
        elif plot_type == "phase":
            signal_phase = np.angle(sample_signal[ch])
            relevance_phase = np.angle(sample_relevance[ch])
            
            ax_mag.plot(frequencies, signal_phase, 'k-', alpha=0.7, linewidth=1, label='Signal Phase')
            ax_mag.plot(frequencies, relevance_phase, 'r-', linewidth=2, label='Relevance Phase')
            
            ax_mag.set_ylabel(f'{channel_names[ch]}\nPhase (rad)', fontsize=10)
            ax_mag.set_title(f'Phase Spectrum - {channel_names[ch]}')
    
    # Set x-label
    if plot_type == "both":
        axes[-1, 0].set_xlabel('Frequency (Hz)', fontsize=12)
        axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
    else:
        axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_time_frequency_relevance(relevance_stft: np.ndarray,
                                 time_axis: np.ndarray,
                                 freq_axis: np.ndarray,
                                 channel_names: Optional[List[str]] = None,
                                 title: str = "Time-Frequency Relevance",
                                 figsize: Tuple[int, int] = (14, 10),
                                 sample_idx: int = 0,
                                 save_path: Optional[str] = None,
                                 cmap: str = 'RdBu_r') -> plt.Figure:
    """
    Visualize relevance scores in the time-frequency domain.
    
    Args:
        relevance_stft: Relevance STFT of shape (batch_size, channels, time_windows, freq_bins)
        time_axis: Time axis for STFT windows
        freq_axis: Frequency axis for STFT
        channel_names: Names for each channel
        title: Plot title
        figsize: Figure size
        sample_idx: Which sample to plot from batch
        save_path: Path to save figure
        cmap: Colormap for relevance visualization
        
    Returns:
        matplotlib Figure object
    """
    if len(relevance_stft.shape) == 3:  # Single sample
        relevance_stft = relevance_stft[np.newaxis, ...]
    
    n_samples, n_channels = relevance_stft.shape[:2]
    
    if sample_idx >= n_samples:
        raise ValueError(f"sample_idx {sample_idx} >= n_samples {n_samples}")
    
    if channel_names is None:
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
    
    # Get sample data
    sample_relevance = relevance_stft[sample_idx]
    
    # Create subplots
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    # Normalize relevance across all channels for consistent color scale
    vmax = np.max(np.abs(sample_relevance))
    vmin = -vmax
    
    for ch in range(n_channels):
        ax = axes[ch]
        
        # Get magnitude of complex relevance
        relevance_mag = np.abs(sample_relevance[ch])
        
        # Create time-frequency plot
        im = ax.imshow(relevance_mag.T, aspect='auto', origin='lower',
                      cmap=cmap, vmin=0, vmax=vmax,
                      extent=[time_axis[0], time_axis[-1], 
                             freq_axis[0], freq_axis[-1]])
        
        ax.set_ylabel(f'{channel_names[ch]}\nFrequency (Hz)', fontsize=10)
        ax.set_title(f'Time-Frequency Relevance - {channel_names[ch]}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Relevance Magnitude', fontsize=9)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_summary(results: Dict,
                           sample_idx: int = 0,
                           figsize: Tuple[int, int] = (16, 12),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive comparison plot showing all analysis results.
    
    Args:
        results: Dictionary containing analysis results from DFTLRPExplainer.analyze_sample()
        sample_idx: Which sample to plot from batch
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Determine available analyses
    has_time = 'time' in results and results['time'] is not None
    has_freq = 'frequency' in results and results['frequency'] is not None
    has_tf = 'time_frequency' in results and results['time_frequency'] is not None
    
    if not has_time:
        raise ValueError("Time domain results required for comparison plot")
    
    # Calculate subplot layout
    n_rows = 1 + (1 if has_freq else 0) + (1 if has_tf else 0)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 3, hspace=0.3, wspace=0.3)
    
    # Get data
    time_data = results['time']['input_data']
    time_relevance = results['time']['relevance']
    prediction = results['time']['prediction']
    probabilities = results['time']['probabilities']
    
    if len(time_data.shape) == 2:
        time_data = time_data[np.newaxis, ...]
        time_relevance = time_relevance[np.newaxis, ...]
    
    n_channels = time_data.shape[1]
    signal_length = time_data.shape[2]
    time_axis = np.arange(signal_length)
    
    channel_names = [f'{"XYZ"[i] if i < 3 else f"Ch{i}"}' for i in range(n_channels)]
    
    # Row 1: Time domain
    row_idx = 0
    for ch in range(min(n_channels, 3)):  # Limit to 3 channels for space
        ax = fig.add_subplot(gs[row_idx, ch])
        
        # Plot signal
        ax.plot(time_axis, time_data[sample_idx, ch], 'k-', alpha=0.7, linewidth=1)
        
        # Overlay relevance as colored background
        relevance_ch = time_relevance[sample_idx, ch]
        vmax = np.max(np.abs(relevance_ch))
        
        # Create relevance heatmap
        relevance_2d = relevance_ch[np.newaxis, :]
        ax.imshow(relevance_2d, aspect='auto', cmap='RdBu_r', alpha=0.6,
                 vmin=-vmax, vmax=vmax,
                 extent=[0, signal_length, np.min(time_data[sample_idx, ch]), 
                        np.max(time_data[sample_idx, ch])])
        
        ax.set_title(f'Time Domain - {channel_names[ch]}', fontsize=10)
        ax.set_xlabel('Time')
        if ch == 0:
            ax.set_ylabel('Amplitude')
    
    # Add prediction info
    if n_channels >= 3:
        ax = fig.add_subplot(gs[row_idx, 2])
    else:
        ax = fig.add_subplot(gs[row_idx, -1])
    
    # Prediction summary
    pred_class = prediction[sample_idx] if hasattr(prediction, '__len__') else prediction
    pred_prob = probabilities[sample_idx] if len(probabilities.shape) > 1 else probabilities
    
    classes = ['Good', 'Bad']
    colors = ['green', 'red']
    bars = ax.bar(classes, pred_prob, color=colors, alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    ax.set_title(f'Prediction: {classes[pred_class]}\nConfidence: {pred_prob[pred_class]:.3f}')
    
    # Add probability values on bars
    for bar, prob in zip(bars, pred_prob):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    # Row 2: Frequency domain (if available)
    if has_freq:
        row_idx += 1
        freq_data = results['frequency']
        signal_freq = freq_data['signal_freq']
        relevance_freq = freq_data['relevance_freq']
        frequencies = freq_data['frequencies']
        
        for ch in range(min(n_channels, 3)):
            ax = fig.add_subplot(gs[row_idx, ch])
            
            signal_mag = np.abs(signal_freq[sample_idx, ch])
            relevance_mag = np.abs(relevance_freq[sample_idx, ch])
            
            ax.semilogy(frequencies, signal_mag, 'k-', alpha=0.7, linewidth=1, label='Signal')
            ax.semilogy(frequencies, relevance_mag, 'r-', linewidth=2, label='Relevance')
            
            ax.set_title(f'Frequency Domain - {channel_names[ch]}', fontsize=10)
            ax.set_xlabel('Frequency (Hz)')
            if ch == 0:
                ax.set_ylabel('Magnitude')
                ax.legend()
    
    # Row 3: Time-frequency domain (if available)
    if has_tf:
        row_idx += 1
        tf_data = results['time_frequency']
        relevance_stft = tf_data['relevance_stft']
        time_axis_tf = tf_data['time_axis']
        freq_axis_tf = tf_data['freq_axis']
        
        for ch in range(min(n_channels, 3)):
            ax = fig.add_subplot(gs[row_idx, ch])
            
            relevance_mag = np.abs(relevance_stft[sample_idx, ch])
            
            im = ax.imshow(relevance_mag.T, aspect='auto', origin='lower',
                          cmap='hot', 
                          extent=[time_axis_tf[0], time_axis_tf[-1],
                                 freq_axis_tf[0], freq_axis_tf[-1]])
            
            ax.set_title(f'Time-Frequency - {channel_names[ch]}', fontsize=10)
            ax.set_xlabel('Time')
            if ch == 0:
                ax.set_ylabel('Frequency (Hz)')
            
            # Add small colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Relevance', fontsize=8)
    
    fig.suptitle('DFT-LRP Analysis Summary', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_channel_comparison(results: Dict,
                           domain: str = "time",
                           sample_idx: int = 0,
                           figsize: Tuple[int, int] = (14, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare relevance scores across all channels in a specific domain.
    
    Args:
        results: Analysis results dictionary
        domain: Domain to plot ("time", "frequency", or "time_frequency")
        sample_idx: Which sample to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if domain not in results or results[domain] is None:
        raise ValueError(f"Domain '{domain}' not available in results")
    
    data = results[domain]
    
    if domain == "time":
        input_data = data['input_data']
        relevance = data['relevance']
        
        if len(input_data.shape) == 2:
            input_data = input_data[np.newaxis, ...]
            relevance = relevance[np.newaxis, ...]
        
        n_channels = input_data.shape[1]
        signal_length = input_data.shape[2]
        
        fig, axes = plt.subplots(n_channels, 2, figsize=figsize, sharex=True)
        if n_channels == 1:
            axes = axes[np.newaxis, :]
        
        time_axis = np.arange(signal_length)
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
        
        for ch in range(n_channels):
            # Signal plot
            axes[ch, 0].plot(time_axis, input_data[sample_idx, ch], 'k-', linewidth=1)
            axes[ch, 0].set_title(f'Original Signal - {channel_names[ch]}')
            axes[ch, 0].set_ylabel('Amplitude')
            axes[ch, 0].grid(True, alpha=0.3)
            
            # Relevance plot
            axes[ch, 1].plot(time_axis, relevance[sample_idx, ch], 'r-', linewidth=1)
            axes[ch, 1].set_title(f'Relevance - {channel_names[ch]}')
            axes[ch, 1].set_ylabel('Relevance')
            axes[ch, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Time')
        axes[-1, 1].set_xlabel('Time')
        
    elif domain == "frequency":
        signal_freq = data['signal_freq']
        relevance_freq = data['relevance_freq']
        frequencies = data['frequencies']
        
        if len(signal_freq.shape) == 2:
            signal_freq = signal_freq[np.newaxis, ...]
            relevance_freq = relevance_freq[np.newaxis, ...]
        
        n_channels = signal_freq.shape[1]
        
        fig, axes = plt.subplots(n_channels, 2, figsize=figsize, sharex=True)
        if n_channels == 1:
            axes = axes[np.newaxis, :]
        
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
        
        for ch in range(n_channels):
            # Signal magnitude
            signal_mag = np.abs(signal_freq[sample_idx, ch])
            axes[ch, 0].semilogy(frequencies, signal_mag, 'k-', linewidth=1)
            axes[ch, 0].set_title(f'Signal Magnitude - {channel_names[ch]}')
            axes[ch, 0].set_ylabel('Magnitude')
            axes[ch, 0].grid(True, alpha=0.3)
            
            # Relevance magnitude
            relevance_mag = np.abs(relevance_freq[sample_idx, ch])
            axes[ch, 1].semilogy(frequencies, relevance_mag, 'r-', linewidth=1)
            axes[ch, 1].set_title(f'Relevance Magnitude - {channel_names[ch]}')
            axes[ch, 1].set_ylabel('Relevance Magnitude')
            axes[ch, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Frequency (Hz)')
        axes[-1, 1].set_xlabel('Frequency (Hz)')
    
    else:  # time_frequency
        relevance_stft = data['relevance_stft']
        time_axis = data['time_axis']
        freq_axis = data['freq_axis']
        
        if len(relevance_stft.shape) == 3:
            relevance_stft = relevance_stft[np.newaxis, ...]
        
        n_channels = relevance_stft.shape[1]
        
        fig, axes = plt.subplots(1, n_channels, figsize=figsize, sharex=True, sharey=True)
        if n_channels == 1:
            axes = [axes]
        
        channel_names = [f'Channel {i} ({"XYZ"[i] if i < 3 else str(i)})' for i in range(n_channels)]
        
        vmax = np.max(np.abs(relevance_stft[sample_idx]))
        
        for ch in range(n_channels):
            relevance_mag = np.abs(relevance_stft[sample_idx, ch])
            
            im = axes[ch].imshow(relevance_mag.T, aspect='auto', origin='lower',
                               cmap='hot', vmin=0, vmax=vmax,
                               extent=[time_axis[0], time_axis[-1],
                                      freq_axis[0], freq_axis[-1]])
            
            axes[ch].set_title(f'{channel_names[ch]}')
            axes[ch].set_xlabel('Time')
            
            if ch == 0:
                axes[ch].set_ylabel('Frequency (Hz)')
            
            # Add colorbar to last subplot
            if ch == n_channels - 1:
                cbar = plt.colorbar(im, ax=axes[ch])
                cbar.set_label('Relevance Magnitude')
    
    fig.suptitle(f'{domain.replace("_", "-").title()} Domain - Channel Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_analysis_report(results: Dict,
                          sample_idx: int = 0,
                          output_dir: str = "./dft_lrp_analysis",
                          show_plots: bool = True) -> Dict[str, str]:
    """
    Generate a complete analysis report with multiple visualizations.
    
    Args:
        results: Analysis results from DFTLRPExplainer
        sample_idx: Which sample to analyze
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    try:
        # 1. Comprehensive summary
        if show_plots:
            fig1 = plot_comparison_summary(results, sample_idx=sample_idx)
            plt.show()
        
        save_path1 = os.path.join(output_dir, f"summary_sample_{sample_idx}.png")
        fig1 = plot_comparison_summary(results, sample_idx=sample_idx, save_path=save_path1)
        saved_plots['summary'] = save_path1
        plt.close(fig1)
        
        # 2. Time domain detailed view
        if 'time' in results and results['time'] is not None:
            time_data = results['time']
            
            if show_plots:
                fig2 = plot_time_domain_relevance(
                    time_data['input_data'], 
                    time_data['relevance'],
                    sample_idx=sample_idx
                )
                plt.show()
            
            save_path2 = os.path.join(output_dir, f"time_domain_sample_{sample_idx}.png")
            fig2 = plot_time_domain_relevance(
                time_data['input_data'], 
                time_data['relevance'],
                sample_idx=sample_idx,
                save_path=save_path2
            )
            saved_plots['time_domain'] = save_path2
            plt.close(fig2)
        
        # 3. Frequency domain detailed view
        if 'frequency' in results and results['frequency'] is not None:
            freq_data = results['frequency']
            
            if show_plots:
                fig3 = plot_frequency_domain_relevance(
                    freq_data['signal_freq'],
                    freq_data['relevance_freq'],
                    freq_data['frequencies'],
                    sample_idx=sample_idx,
                    plot_type="both"
                )
                plt.show()
            
            save_path3 = os.path.join(output_dir, f"frequency_domain_sample_{sample_idx}.png")
            fig3 = plot_frequency_domain_relevance(
                freq_data['signal_freq'],
                freq_data['relevance_freq'],
                freq_data['frequencies'],
                sample_idx=sample_idx,
                plot_type="both",
                save_path=save_path3
            )
            saved_plots['frequency_domain'] = save_path3
            plt.close(fig3)
        
        # 4. Time-frequency detailed view
        if 'time_frequency' in results and results['time_frequency'] is not None:
            tf_data = results['time_frequency']
            
            if show_plots:
                fig4 = plot_time_frequency_relevance(
                    tf_data['relevance_stft'],
                    tf_data['time_axis'],
                    tf_data['freq_axis'],
                    sample_idx=sample_idx
                )
                plt.show()
            
            save_path4 = os.path.join(output_dir, f"time_frequency_sample_{sample_idx}.png")
            fig4 = plot_time_frequency_relevance(
                tf_data['relevance_stft'],
                tf_data['time_axis'],
                tf_data['freq_axis'],
                sample_idx=sample_idx,
                save_path=save_path4
            )
            saved_plots['time_frequency'] = save_path4
            plt.close(fig4)
        
        # 5. Channel comparison plots
        for domain in ['time', 'frequency', 'time_frequency']:
            if domain in results and results[domain] is not None:
                if show_plots:
                    fig5 = plot_channel_comparison(results, domain=domain, sample_idx=sample_idx)
                    plt.show()
                
                save_path5 = os.path.join(output_dir, f"channel_comparison_{domain}_sample_{sample_idx}.png")
                fig5 = plot_channel_comparison(results, domain=domain, sample_idx=sample_idx, save_path=save_path5)
                saved_plots[f'channel_comparison_{domain}'] = save_path5
                plt.close(fig5)
        
        print(f"Analysis report generated successfully!")
        print(f"Plots saved to: {output_dir}")
        print(f"Generated {len(saved_plots)} plots:")
        for name, path in saved_plots.items():
            print(f"  - {name}: {path}")
        
    except Exception as e:
        print(f"Error generating analysis report: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_plots