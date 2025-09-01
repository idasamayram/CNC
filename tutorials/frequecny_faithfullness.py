
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from utils.baseline_xai import load_model

# Import your XAI methods
from utils.xai_implementation import compute_lrp_relevance, compute_basic_dft_lrp
from utils.baseline_xai import grad_times_input_relevance, smoothgrad_relevance, occlusion_simpler_relevance
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from datetime import datetime
import pandas as pd

# Import your model and data loading utilities
from Classification.cnn1D_model import CNN1D_Wide
from torch.utils.data import DataLoader
from utils.baseline_xai import load_model


# Import the frequency window flipping functions defined above
# ... (include the frequency_window_flipping_single, frequency_window_flipping_batch, etc. functions)

# Import the wrapper functions defined above
# ... (include the dft_lrp_freq_wrapper, vanilla_gradient_freq_wrapper, etc. functions)


def frequency_window_flipping_single(model, sample, attribution_method, n_steps=20,
                                     window_size=10, most_relevant_first=True,
                                     device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on frequency domain relevances using EnhancedDFTLRP.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        device: Device to run computations on

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute attributions - expect a function that returns (relevance_time, relevance_freq, signal_freq, ...)
    attribution_results = attribution_method(model, sample)

    # Extract frequency domain relevance and signal
    if isinstance(attribution_results, tuple) and len(attribution_results) >= 3:
        # Unpack the results0 - these functions typically return:
        # (relevance_time, relevance_freq, signal_freq, input_signal, freqs, target)
        _, relevance_freq, signal_freq, input_signal, freqs, target_class = attribution_results
    else:
        raise ValueError("Attribution method must return frequency domain relevance and signal")

    # Determine frequency length
    freq_length = relevance_freq.shape[1]

    # Calculate number of windows in frequency domain
    n_freq_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_freq_windows += 1

    # Calculate window importance by averaging relevance within each window
    window_importance = np.zeros((n_channels, n_freq_windows))

    for channel in range(n_channels):
        for window_idx in range(n_freq_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute attribution within the window
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        original_score = original_prob[target_class].item()

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_freq_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Create EnhancedDFTLRP instance for inverse DFT
    dft_handler = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=True,
        precision=32,
        cuda=(device == "cuda"),
        create_stdft=False,  # We only need DFT, not STDFT
        create_inverse=True,  # We need inverse DFT
        create_transpose_inverse=False,  # Not needed for our purpose
        create_forward=False  # Not needed for our purpose
    )

    # Iteratively flip frequency domain windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Create a copy of the original frequency signal
        flipped_freq_signal = signal_freq.copy()

        # Directly modify frequency components based on window importance sorting
        for i in range(n_windows_to_flip):
            # Convert flat index to channel, window indices
            flat_idx = sorted_indices[i]
            channel_idx = flat_idx // n_freq_windows
            window_idx = flat_idx % n_freq_windows

            # Get window boundaries
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Set the frequency components to zero (flipping)
            flipped_freq_signal[channel_idx, start_idx:end_idx] = 0.0

        # Convert back to time domain using inverse DFT via EnhancedDFTLRP
        try:
            flipped_time_signal = np.zeros_like(input_signal)

            for channel in range(n_channels):
                # Use reshape_signal to prepare for inverse_fourier_layer
                channel_freq = flipped_freq_signal[channel:channel + 1]

                # Apply inverse Fourier transform using EnhancedDFTLRP
                with torch.no_grad():
                    # Convert to tensor
                    channel_freq_tensor = torch.tensor(
                        np.concatenate([
                            np.real(channel_freq),
                            np.imag(channel_freq)
                        ], axis=-1),
                        dtype=torch.float32
                    ).to(device)

                    # Apply inverse fourier transform
                    channel_time_tensor = dft_handler.inverse_fourier_layer(channel_freq_tensor)

                    # Get back to CPU and numpy
                    flipped_time_signal[channel] = channel_time_tensor.cpu().numpy()

        except Exception as e:
            print(f"Error in inverse DFT: {e}")
            # Fallback to numpy's inverse FFT if EnhancedDFTLRP fails
            for channel in range(n_channels):
                if freq_length < time_steps:  # We used rfft with leverage_symmetry
                    flipped_time_signal[channel] = np.fft.irfft(flipped_freq_signal[channel], n=time_steps)
                else:
                    flipped_time_signal[channel] = np.fft.ifft(flipped_freq_signal[channel]).real

        # Get model prediction on the modified signal
        flipped_tensor = torch.tensor(flipped_time_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Cleanup
    del dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts


def visualize_frequency_window_flipping_progression(model, sample, attribution_method,
                                                    n_steps=5, window_size=10, most_relevant_first=True,
                                                    save_path=None,
                                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of frequency domain window flipping with proper plotting.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        n_steps: Number of steps to visualize
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        save_path: Path to save the visualization image (if None, just displays)
        device: Device to run computations on

    Returns:
        fig: Matplotlib figure object
    """
    from utils.dft_lrp import EnhancedDFTLRP
    import matplotlib.pyplot as plt

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute attributions
    attribution_results = attribution_method(model, sample)

    # Extract necessary components
    if isinstance(attribution_results, tuple) and len(attribution_results) >= 6:
        # Unpack the results0
        _, relevance_freq, signal_freq, input_signal, freqs, target_class = attribution_results
    else:
        raise ValueError("Attribution method must return all required components")

    # Determine frequency length
    freq_length = relevance_freq.shape[1]

    # Calculate number of windows in frequency domain
    n_freq_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_freq_windows += 1

    # Calculate window importance by averaging relevance within each window
    window_importance = np.zeros((n_channels, n_freq_windows))

    for channel in range(n_channels):
        for window_idx in range(n_freq_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute attribution within the window
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.item()
        original_score = original_prob[target_class].item()

    # Calculate windows to flip per step
    total_windows = n_channels * n_freq_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Create EnhancedDFTLRP instance for inverse DFT
    dft_handler = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=True,
        precision=32,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=True,
        create_transpose_inverse=False,
        create_forward=False
    )

    # Set up the plot - 2 rows per channel (time and freq domain) x (n_steps + 1) columns
    fig, axes = plt.subplots(2 * n_channels, n_steps + 1, figsize=(20, 4 * n_channels))
    fig.suptitle(
        f'Frequency Domain Window Flipping Progression\n(Flipping {"Most" if most_relevant_first else "Least"} Important First)',
        fontsize=18)

    # Plot the original signals
    for channel in range(n_channels):
        # Time domain - original signal
        row_idx = channel * 2
        if n_channels > 1:
            axes[row_idx, 0].plot(input_signal[channel], 'b-')
            axes[row_idx, 0].set_title(f'Original Time (Score: {original_score:.3f})')
            axes[row_idx, 0].set_ylabel(f'Channel {channel}')
        else:
            axes[0].plot(input_signal[channel], 'b-')
            axes[0].set_title(f'Original Time (Score: {original_score:.3f})')
            axes[0].set_ylabel(f'Channel {channel}')

        # Frequency domain - original signal magnitude
        row_idx = channel * 2 + 1
        magnitude = np.abs(signal_freq[channel])
        if n_channels > 1:
            axes[row_idx, 0].semilogy(freqs[:freq_length], magnitude, 'b-')
            axes[row_idx, 0].set_title(f'Original Freq')
            axes[row_idx, 0].set_ylabel(f'Channel {channel}\nMagnitude (log)')
            axes[row_idx, 0].set_xlim([0, freqs[min(freq_length // 4, len(freqs) - 1)]])  # Focus on lower frequencies
        else:
            axes[1].semilogy(freqs[:freq_length], magnitude, 'b-')
            axes[1].set_title(f'Original Freq')
            axes[1].set_ylabel(f'Channel {channel}\nMagnitude (log)')
            axes[1].set_xlim([0, freqs[min(freq_length // 4, len(freqs) - 1)]])

    # Iteratively flip windows and visualize
    flipped_signals = []  # Store flipped signals for visualization
    flipped_freq_signals = []  # Store flipped frequency signals
    scores = []  # Store prediction scores

    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Create a modified frequency domain signal
        flipped_freq_signal = signal_freq.copy()

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_freq_windows
        window_indices = windows_to_flip % n_freq_windows

        # Set flipped windows to zero
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Set frequency components to zero
            flipped_freq_signal[channel_idx, start_idx:end_idx] = 0.0

        # Convert back to time domain
        try:
            flipped_time_signal = np.zeros_like(input_signal)

            for channel in range(n_channels):
                # Use reshape_signal to prepare for inverse_fourier_layer
                channel_freq = flipped_freq_signal[channel:channel + 1]

                # Apply inverse Fourier transform using EnhancedDFTLRP
                with torch.no_grad():
                    # Convert to tensor format needed by EnhancedDFTLRP
                    channel_freq_tensor = torch.tensor(
                        np.concatenate([
                            np.real(channel_freq),
                            np.imag(channel_freq)
                        ], axis=-1),
                        dtype=torch.float32
                    ).to(device)

                    # Apply inverse fourier transform
                    channel_time_tensor = dft_handler.inverse_fourier_layer(channel_freq_tensor)

                    # Get back to CPU and numpy
                    flipped_time_signal[channel] = channel_time_tensor.cpu().numpy()

        except Exception as e:
            print(f"Error in inverse DFT: {e}")
            # Fallback to numpy's inverse FFT
            for channel in range(n_channels):
                if freq_length < time_steps:  # We used rfft with leverage_symmetry
                    flipped_time_signal[channel] = np.fft.irfft(flipped_freq_signal[channel], n=time_steps)
                else:
                    flipped_time_signal[channel] = np.fft.ifft(flipped_freq_signal[channel]).real

        # Get model output
        flipped_tensor = torch.tensor(flipped_time_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Store for visualization
        flipped_signals.append(flipped_time_signal)
        flipped_freq_signals.append(flipped_freq_signal)
        scores.append(score)

        # Plot the flipped signals
        for channel in range(n_channels):
            # Time domain - flipped signal
            row_idx = channel * 2
            if n_channels > 1:
                axes[row_idx, step].plot(flipped_time_signal[channel], 'r-')
                axes[row_idx, step].set_title(f'{n_windows_to_flip / total_windows * 100:.1f}% (Score: {score:.3f})')
            else:
                axes[0, step].plot(flipped_time_signal[channel], 'r-')
                axes[0, step].set_title(f'{n_windows_to_flip / total_windows * 100:.1f}% (Score: {score:.3f})')

            # Frequency domain - flipped signal magnitude
            row_idx = channel * 2 + 1
            magnitude = np.abs(flipped_freq_signal[channel])
            if n_channels > 1:
                axes[row_idx, step].semilogy(freqs[:freq_length], magnitude, 'r-')
                axes[row_idx, step].set_title(f'Flipped Freq (Ch {channel})')
                axes[row_idx, step].set_xlim([0, freqs[min(freq_length // 4, len(freqs) - 1)]])

                # Highlight flipped windows
                for i in range(len(windows_to_flip)):
                    if channel_indices[i] == channel:
                        window_idx = window_indices[i]
                        start_idx = window_idx * window_size
                        end_idx = min((window_idx + 1) * window_size, freq_length)
                        axes[row_idx, step].axvspan(
                            freqs[start_idx],
                            freqs[min(end_idx - 1, len(freqs) - 1)],  # Ensure index is valid
                            alpha=0.2, color='gray'
                        )
            else:
                axes[1, step].semilogy(freqs[:freq_length], magnitude, 'r-')
                axes[1, step].set_title(f'Flipped Freq')
                axes[1, step].set_xlim([0, freqs[min(freq_length // 4, len(freqs) - 1)]])

                # Highlight flipped windows
                for i in range(len(windows_to_flip)):
                    if channel_indices[i] == channel:
                        window_idx = window_indices[i]
                        start_idx = window_idx * window_size
                        end_idx = min((window_idx + 1) * window_size, freq_length)
                        axes[1, step].axvspan(
                            freqs[start_idx],
                            freqs[min(end_idx - 1, len(freqs) - 1)],
                            alpha=0.2, color='gray'
                        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure if a path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()

    # Cleanup
    del dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fig


# Example of how to call the visualization function
def test_frequency_visualization(model_path, data_dir):
    """Test frequency domain window flipping visualization"""
    from utils.baseline_xai import load_model, load_sample_data

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Load a sample
    samples, labels, _ = load_sample_data(data_dir, num_samples=1)
    sample = samples[0].to(device)
    label = labels[0]

    print(f"Visualizing frequency window flipping for sample with label {label}")

    # Define wrapper for DFT-LRP
    def dft_lrp_wrapper(model, sample):
        from utils.xai_implementation import compute_basic_dft_lrp
        return compute_basic_dft_lrp(
            model=model,
            sample=sample,
            label=label,
            device=device,
            signal_length=sample.shape[1],
            leverage_symmetry=True,
            precision=32,
            sampling_rate=400
        )

    # Create visualization
    fig = visualize_frequency_window_flipping_progression(
        model=model,
        sample=sample,
        attribution_method=dft_lrp_wrapper,
        n_steps=5,
        window_size=5,
        most_relevant_first=True,
        save_path="freq_window_flipping.png",
        device=device
    )

    return fig


def frequency_window_flipping_batch(model, test_loader, attribution_methods, n_steps=10,
                                    window_size=10, most_relevant_first=True,
                                    reference_value=None, max_samples=None,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform frequency domain window flipping analysis on a batch of time series samples.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        device: Device to run computations on

    Returns:
        results0: Dictionary with results0 for each method
    """
    # Initialize results0 storage
    results = {method_name: [] for method_name in attribution_methods}

    # Keep track of the current sample count
    sample_count = 0

    # Process each batch in the test loader
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
        # Process each sample in the batch
        for i in range(len(data)):
            sample = data[i].to(device)
            target = targets[i].to(device)

            # Increment sample counter
            sample_count += 1

            # Print progress every 10 samples
            if sample_count % 10 == 0:
                print(f"Processing sample {sample_count}")

            # Check if we've reached the maximum number of samples
            if max_samples is not None and sample_count > max_samples:
                break

            # Process each attribution method
            for method_name, attribution_func in attribution_methods.items():
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                try:
                    # Compute scores for this sample using the current method
                    scores, flipped_pcts = frequency_window_flipping_single(
                        model, sample, lambda m, s: attribution_func(m, s, target),
                        n_steps, window_size, most_relevant_first, reference_value, device
                    )

                    # Store results0
                    results[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": scores,
                        "flipped_pcts": flipped_pcts,
                        "auc": np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
                    })

                except Exception as e:
                    print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                    # Store empty result to maintain sample count consistency
                    results[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": None,
                        "flipped_pcts": None,
                        "auc": float('nan')
                    })

        # Check if we've reached the maximum number of samples
        if max_samples is not None and sample_count >= max_samples:
            print(f"Reached maximum number of samples ({max_samples})")
            break

    return results


def save_frequency_flipping_visualization(model, sample, attribution_method, output_path,
                                          n_steps=5, window_size=10, most_relevant_first=True,
                                          device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run frequency domain window flipping and save the visualization of the progression.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        output_path: Path to save visualization images
        n_steps: Number of steps to visualize
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        device: Device to run computations on
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Run frequency flipping and get visualization data
    scores, flipped_pcts, viz_data = frequency_window_flipping_single(  # Changed to use the correct function name
        model, sample, attribution_method, n_steps, window_size,
        most_relevant_first, None, device
    )

    # Extract data
    original_signals = viz_data['original']
    flipped_signals = viz_data['flipped']
    freqs = viz_data['freqs']

    # Get number of channels
    n_channels = original_signals['time'].shape[0]
    time_steps = original_signals['time'].shape[1]
    freq_length = original_signals['freq'].shape[1]

    # Save summary plot
    plt.figure(figsize=(10, 6))
    plt.plot(flipped_pcts, scores, 'o-', linewidth=2)
    plt.title(f"Frequency Window Flipping ({'Most' if most_relevant_first else 'Least'} important first)")
    plt.xlabel("Percentage of Windows Flipped (%)")
    plt.ylabel("Prediction Score")
    plt.grid(True)
    plt.savefig(f"{output_path}/summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save initial state (original signals)
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4 * n_channels))

    # If there's only one channel, make axes 2D
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    for ch in range(n_channels):
        # Plot time domain
        axes[ch, 0].plot(original_signals['time'][ch])
        axes[ch, 0].set_title(f"Channel {ch} - Original Time Domain")
        axes[ch, 0].set_xlabel("Time Steps")
        axes[ch, 0].set_ylabel("Amplitude")

        # Plot frequency domain (magnitude)
        freq_magnitude = np.abs(original_signals['freq'][ch])
        axes[ch, 1].semilogy(freqs[:freq_length], freq_magnitude)
        axes[ch, 1].set_title(f"Channel {ch} - Original Frequency Domain")
        axes[ch, 1].set_xlabel("Frequency (Hz)")
        axes[ch, 1].set_ylabel("Magnitude (log scale)")
        axes[ch, 1].set_xlim([0, freqs[freq_length // 4]])  # Focus on lower frequencies

    plt.tight_layout()
    plt.savefig(f"{output_path}/step0_original.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save each step of flipping
    for step, step_data in enumerate(flipped_signals):
        fig, axes = plt.subplots(n_channels, 2, figsize=(14, 4 * n_channels))

        # If there's only one channel, make axes 2D
        if n_channels == 1:
            axes = axes.reshape(1, -1)

        for ch in range(n_channels):
            # Plot time domain
            axes[ch, 0].plot(original_signals['time'][ch], 'b-', alpha=0.5, label="Original")
            axes[ch, 0].plot(step_data['time'][ch], 'r-', label="After Flipping")
            axes[ch, 0].set_title(f"Channel {ch} - Time Domain - {step_data['flipped_pct']:.1f}% Flipped")
            axes[ch, 0].set_xlabel("Time Steps")
            axes[ch, 0].set_ylabel("Amplitude")
            axes[ch, 0].legend()

            # Plot frequency domain (magnitude)
            freq_magnitude_orig = np.abs(original_signals['freq'][ch])
            freq_magnitude_flipped = np.abs(step_data['freq'][ch])

            axes[ch, 1].semilogy(freqs[:freq_length], freq_magnitude_orig, 'b-', alpha=0.5, label="Original")
            axes[ch, 1].semilogy(freqs[:freq_length], freq_magnitude_flipped, 'r-', label="After Flipping")

            # Highlight flipped regions
            for i in range(freq_length):
                if step_data['mask'][ch, i]:
                    axes[ch, 1].axvspan(freqs[i], freqs[i], color='gray', alpha=0.3)

            axes[ch, 1].set_title(f"Channel {ch} - Frequency Domain - Score: {scores[step + 1]:.4f}")
            axes[ch, 1].set_xlabel("Frequency (Hz)")
            axes[ch, 1].set_ylabel("Magnitude (log scale)")
            axes[ch, 1].set_xlim([0, freqs[freq_length // 4]])  # Focus on lower frequencies
            axes[ch, 1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_path}/step{step + 1}_flipped{step_data['flipped_pct']:.1f}pct.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    # Create a summary file with scores
    with open(f"{output_path}/flipping_results.txt", "w") as f:
        f.write(f"Frequency Window Flipping Results\n")
        f.write(f"{'Most' if most_relevant_first else 'Least'} important windows first\n")
        f.write(f"Window size: {window_size}\n\n")
        f.write(f"Percentage Flipped (%)  |  Prediction Score\n")
        f.write(f"----------------------------------------\n")
        for pct, score in zip(flipped_pcts, scores):
            f.write(f"{pct:20.2f}  |  {score:.6f}\n")

        # Calculate AUC
        auc = np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
        f.write(f"\nArea Under the Curve (AUC): {auc:.6f}\n")

    print(f"Visualization saved to {output_path}")
    return scores, flipped_pcts


def visualize_single_sample_flipping(model, sample, attribution_method, label=None,
                                     output_path=None, n_steps=5, window_size=10,
                                     device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate detailed visualizations of frequency window flipping for a single sample.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        attribution_method: Function that generates attributions
        label: Optional target label
        output_path: Path to save visualizations (None = display only)
        n_steps: Number of flipping steps to visualize
        window_size: Size of frequency windows
        device: Device to use

    Returns:
        dict: Dictionary with both most-first and least-first results0
    """
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime

    # If no output path is specified, create one with timestamp
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./single_sample_flipping_{timestamp}"

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Create wrapper function to handle label
    def wrapped_attribution(model, sample):
        if label is not None:
            return attribution_method(model, sample, label)
        else:
            return attribution_method(model, sample)

    # Get original prediction
    with torch.no_grad():
        orig_output = model(sample.unsqueeze(0))
        orig_prob = torch.softmax(orig_output, dim=1)[0]
        pred_class = torch.argmax(orig_prob).item()
        orig_score = orig_prob[pred_class].item()

    # Print information
    print(f"Sample prediction: Class {pred_class}, Score: {orig_score:.4f}")
    print(f"Running frequency window flipping with {n_steps} steps, window size: {window_size}")

    # Run both most important first and least important first
    print("Flipping most important windows first...")
    scores_most, pcts_most = save_frequency_flipping_visualization(
        model=model,
        sample=sample,
        attribution_method=wrapped_attribution,
        output_path=f"{output_path}/most_important",
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        device=device
    )

    print("Flipping least important windows first...")
    scores_least, pcts_least = save_frequency_flipping_visualization(
        model=model,
        sample=sample,
        attribution_method=wrapped_attribution,
        output_path=f"{output_path}/least_important",
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        device=device
    )

    # Calculate AUCs
    auc_most = np.trapz(scores_most, pcts_most) / pcts_most[-1]
    auc_least = np.trapz(scores_least, pcts_least) / pcts_least[-1]
    faithfulness_ratio = auc_least / auc_most if auc_most > 0 else float('nan')

    # Create summary plot comparing both approaches
    plt.figure(figsize=(12, 8))
    plt.plot(pcts_most, scores_most, 'ro-', linewidth=2, markersize=8,
             label=f"Most Important First (AUC: {auc_most:.4f})")
    plt.plot(pcts_least, scores_least, 'bo-', linewidth=2, markersize=8,
             label=f"Least Important First (AUC: {auc_least:.4f})")
    plt.title(f"Frequency Window Flipping Comparison\nFaithfulness Ratio: {faithfulness_ratio:.4f}", fontsize=16)
    plt.xlabel("Percentage of Windows Flipped (%)", fontsize=14)
    plt.ylabel("Prediction Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_path}/comparison_plot.png", dpi=300, bbox_inches='tight')

    # Save summary metrics
    with open(f"{output_path}/summary_metrics.txt", "w") as f:
        f.write(f"Frequency Window Flipping - Summary Metrics\n")
        f.write(f"========================================\n\n")
        f.write(f"Sample Prediction: Class {pred_class}, Score: {orig_score:.6f}\n")
        f.write(f"Window Size: {window_size}\n")
        f.write(f"Number of Steps: {n_steps}\n\n")
        f.write(f"AUC (Most Important First): {auc_most:.6f}\n")
        f.write(f"AUC (Least Important First): {auc_least:.6f}\n")
        f.write(f"Faithfulness Ratio (Least/Most): {faithfulness_ratio:.6f}\n")

    print(f"\nResults saved to {output_path}")
    print(f"AUC (Most Important First): {auc_most:.6f}")
    print(f"AUC (Least Important First): {auc_least:.6f}")
    print(f"Faithfulness Ratio: {faithfulness_ratio:.6f}")

    return {
        "most_first": {"scores": scores_most, "pcts": pcts_most, "auc": auc_most},
        "least_first": {"scores": scores_least, "pcts": pcts_least, "auc": auc_least},
        "faithfulness_ratio": faithfulness_ratio
    }
# frequency_window_flipping_evaluation.py
def test_frequency_flipping():
    """Test frequency domain window flipping with visualization."""
    import torch
    import os
    from datetime import datetime
    from utils.baseline_xai import load_model, load_sample_data
    from utils.xai_implementation import compute_basic_dft_lrp

    # Configuration
    model_path = "../cnn1d_model_wide_new.ckpt"  # Update with your model path
    data_dir = "../data/final/new_selection/normalized_windowed_downsampled_data_lessBAD"  # Update with your data path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Load a sample
    samples, labels, _ = load_sample_data(data_dir, num_samples=1)
    sample = samples[0].to(device)
    label = labels[0]

    print(f"Testing frequency window flipping for sample with label {label}")

    # Create wrapper function for DFT-LRP
    def dft_lrp_wrapper(model, sample):
        return compute_basic_dft_lrp(
            model=model,
            sample=sample,
            label=label,
            device=device,
            signal_length=sample.shape[1],
            leverage_symmetry=True,
            precision=32,
            sampling_rate=400
        )

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./results/freq_flipping_vis_{timestamp}"

    # Run visualization
    scores_most, pcts_most = save_frequency_flipping_visualization(
        model=model,
        sample=sample,
        attribution_method=dft_lrp_wrapper,
        output_path=f"{output_path}/most_important",
        n_steps=5,
        window_size=10,
        most_relevant_first=True,
        device=device
    )

    # Also try least important first
    scores_least, pcts_least = save_frequency_flipping_visualization(
        model=model,
        sample=sample,
        attribution_method=dft_lrp_wrapper,
        output_path=f"{output_path}/least_important",
        n_steps=5,
        window_size=10,
        most_relevant_first=False,
        device=device
    )

    print(f"Most important first - AUC: {np.trapz(scores_most, pcts_most) / pcts_most[-1]:.6f}")
    print(f"Least important first - AUC: {np.trapz(scores_least, pcts_least) / pcts_least[-1]:.6f}")
    print(
        f"Faithfulness ratio: {(np.trapz(scores_least, pcts_least) / pcts_least[-1]) / (np.trapz(scores_most, pcts_most) / pcts_most[-1]):.6f}")

    return output_path


def dft_lrp_freq_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for DFT-LRP attribution focusing on frequency domain relevances.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        label: Optional target label
        device: Computing device

    Returns:
        All components from compute_basic_dft_lrp
    """
    from utils.xai_implementation import compute_basic_dft_lrp

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute DFT-LRP relevances
    return compute_basic_dft_lrp(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
    )

def vanilla_gradient_freq_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for vanilla gradient in frequency domain.
    """
    from utils.xai_implementation import compute_dft_vanilla_gradient

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_vanilla_gradient(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
    )

def grad_input_freq_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for gradient×input in frequency domain.
    """
    from utils.xai_implementation import compute_dft_gradient_input

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_gradient_input(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
    )

def smoothgrad_freq_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for SmoothGrad in frequency domain.
    """
    from utils.xai_implementation import compute_dft_smoothgrad

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_smoothgrad(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        num_samples=40,
        noise_level=1.0
    )

def occlusion_freq_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for occlusion method in frequency domain.
    """
    from utils.xai_implementation import compute_dft_occlusion

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_occlusion(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        occlusion_type="zero",
        window_size=40
    )


def analyze_single_sample_from_dataset():
    """Analyze a single sample from your test dataset with frequency window flipping."""
    import torch
    import os
    from datetime import datetime
    from utils.baseline_xai import load_model

    # Configuration
    model_path = "../cnn1d_model_test.ckpt"  # Update with your model path
    data_dir = "../data/final/new_strategy/balanced_data"  # Update with your data path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Load test data
    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    # Get a sample
    for data, targets in test_loader:
        sample = data[0].to(device)  # First sample in batch
        target = targets[0].item()  # First target in batch
        break

    print(f"Analyzing sample with target class {target}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./single_sample_analysis_{timestamp}"
    os.makedirs(output_path, exist_ok=True)

    # Run analysis with DFT-LRP
    dft_lrp_results = visualize_single_sample_flipping(
        model=model,
        sample=sample,
        attribution_method=dft_lrp_freq_wrapper,
        label=target,
        output_path=f"{output_path}/dft_lrp",
        n_steps=5,
        window_size=5,
        device=device
    )

    # Run analysis with Gradient×Input for comparison
    grad_input_results = visualize_single_sample_flipping(
        model=model,
        sample=sample,
        attribution_method=grad_input_freq_wrapper,
        label=target,
        output_path=f"{output_path}/grad_input",
        n_steps=5,
        window_size=5,
        device=device
    )

    # Compare the methods
    print("\nComparison of Frequency Domain XAI Methods:")
    print(f"DFT-LRP Faithfulness Ratio: {dft_lrp_results['faithfulness_ratio']:.4f}")
    print(f"Grad×Input Faithfulness Ratio: {grad_input_results['faithfulness_ratio']:.4f}")

    # Create a comparison plot of the two methods (most important first)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(
        dft_lrp_results["most_first"]["pcts"],
        dft_lrp_results["most_first"]["scores"],
        'ro-', linewidth=2, markersize=8,
        label=f"DFT-LRP (AUC: {dft_lrp_results['most_first']['auc']:.4f})"
    )
    plt.plot(
        grad_input_results["most_first"]["pcts"],
        grad_input_results["most_first"]["scores"],
        'bo-', linewidth=2, markersize=8,
        label=f"Grad×Input (AUC: {grad_input_results['most_first']['auc']:.4f})"
    )
    plt.title("Frequency Window Flipping - Method Comparison\n(Most Important First)", fontsize=16)
    plt.xlabel("Percentage of Windows Flipped (%)", fontsize=14)
    plt.ylabel("Prediction Score", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig(f"{output_path}/method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from datetime import datetime
import pandas as pd

# Import your model and data loading utilities
from Classification.cnn1D_model import CNN1D_Wide
from torch.utils.data import DataLoader
from utils.baseline_xai import load_model


# Import the frequency window flipping functions defined above
# ... (include the frequency_window_flipping_single, frequency_window_flipping_batch, etc. functions)

# Import the wrapper functions defined above
# ... (include the dft_lrp_freq_wrapper, vanilla_gradient_freq_wrapper, etc. functions)

def aggregate_results(results):
    """
    Aggregate window flipping results0 across all samples.

    Args:
        results: Dictionary with results0 for each method and sample

    Returns:
        agg_results: Dictionary with aggregated results0
    """
    agg_results = {}

    for method_name, samples in results.items():
        # Initialize storage for this method
        method_results = {
            "avg_scores": [],
            "flipped_pcts": None,
            "auc_values": [],
            "mean_auc": None,
            "std_auc": None
        }

        # Collect all AUC values
        valid_samples = [s for s in samples if s["scores"] is not None]

        if not valid_samples:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = method_results
            continue

        # Get flipped percentages from first sample (should be the same for all)
        method_results["flipped_pcts"] = valid_samples[0]["flipped_pcts"]

        # Initialize scores array with correct dimensions
        n_steps = len(valid_samples[0]["scores"])
        all_scores = np.zeros((len(valid_samples), n_steps))

        # Collect scores and AUC values
        for i, sample in enumerate(valid_samples):
            all_scores[i] = sample["scores"]
            method_results["auc_values"].append(sample["auc"])

        # Calculate average scores at each flipping step
        method_results["avg_scores"] = np.mean(all_scores, axis=0)

        # Calculate mean and std of AUC values
        method_results["mean_auc"] = np.mean(method_results["auc_values"])
        method_results["std_auc"] = np.std(method_results["auc_values"])

        # Store results0 for this method
        agg_results[method_name] = method_results

    return agg_results


def plot_aggregate_results(agg_results, most_relevant_first=True):
    """
    Plot aggregated window flipping results0.

    Args:
        agg_results: Dictionary with aggregated results0 for each method
        most_relevant_first: Whether most relevant windows were flipped first
    """
    plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Aggregate Frequency Domain Window Flipping Results\n(Flipping {flip_order} windows first)', fontsize=16)
    plt.xlabel('Percentage of Frequency Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def plot_auc_distribution(agg_results, most_relevant_first=True):
    """
    Plot distribution of AUC values for each method.

    Args:
        agg_results: Dictionary with aggregated results0 for each method
        most_relevant_first: Whether most relevant windows were flipped first
    """
    plt.figure(figsize=(14, 6))

    # Count number of methods with valid data
    valid_methods = [m for m, r in agg_results.items() if len(r["auc_values"]) > 0]
    n_methods = len(valid_methods)

    if n_methods == 0:
        print("No valid methods to plot")
        return

    # Create boxplots
    boxes = []
    labels = []

    for method_name, results in agg_results.items():
        if len(results["auc_values"]) == 0:
            continue

        boxes.append(results["auc_values"])
        labels.append(method_name)

    plt.boxplot(boxes, labels=labels)

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Frequency Domain AUC Distribution\n(Flipping {flip_order} windows first)', fontsize=16)
    plt.ylabel('AUC Value', fontsize=14)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)

    # Add text with mean and std
    for i, method_name in enumerate(labels):
        mean_auc = agg_results[method_name]["mean_auc"]
        std_auc = agg_results[method_name]["std_auc"]
        plt.text(i + 1, plt.ylim()[0] + 0.05, f"μ={mean_auc:.4f}\nσ={std_auc:.4f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def save_results_to_csv(agg_results, results, filename_prefix):
    """
    Save results0 to CSV files.

    Args:
        agg_results: Dictionary with aggregated results0
        results: Dictionary with individual sample results0
        filename_prefix: Prefix for filenames
    """
    # Save aggregated results0
    agg_data = []
    for method_name, method_results in agg_results.items():
        agg_data.append({
            "Method": method_name,
            "Mean AUC": method_results["mean_auc"],
            "Std AUC": method_results["std_auc"],
            "Num Samples": len(method_results["auc_values"])
        })

    agg_df = pd.DataFrame(agg_data)
    agg_df.to_csv(f"{filename_prefix}_aggregate.csv", index=False)

    # Save individual sample results0
    samples_data = []
    for method_name, samples in results.items():
        for sample in samples:
            if sample["scores"] is not None:
                samples_data.append({
                    "Method": method_name,
                    "Sample Index": sample["sample_idx"],
                    "AUC": sample["auc"]
                })

    samples_df = pd.DataFrame(samples_data)
    samples_df.to_csv(f"{filename_prefix}_samples.csv", index=False)


def run_frequency_flipping_evaluation(model, test_loader, attribution_methods,
                                      n_steps=10, window_size=5,
                                      most_relevant_first=True, max_samples=None,
                                      output_dir="./results0",
                                      device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency domain window flipping evaluation on test set and save results0.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        max_samples: Maximum number of samples to process (None = all)
        output_dir: Directory to save results0
        device: Device to run computations on

    Returns:
        results0: Dictionary with individual sample results0
        agg_results: Dictionary with aggregated results0
    """
    import os
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/freq_window_flipping_{flip_order}_{timestamp}"

    print(f"Starting frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")

    # Run window flipping on all samples
    results = frequency_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=None,
        max_samples=max_samples,
        device=device
    )

    print("Computing aggregated results0...")

    # Aggregate results0
    agg_results = aggregate_results(results)

    print("Plotting results0...")

    # Plot and save aggregated results0
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')

    # Plot and save AUC distributions
    dist_fig = plot_auc_distribution(agg_results, most_relevant_first)
    dist_fig.savefig(f"{filename_prefix}_auc_distribution.png", dpi=300, bbox_inches='tight')

    # Save numerical results0 to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nFrequency Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


def main():
    """Main function to run the evaluation."""
    # Configuration
    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"  # Path to your trained model
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./frequency_results"
    n_steps = 10
    window_size = 5  # Smaller window size for frequency domain
    max_samples = 5  # Set to None to use all test samples

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Set up attribution methods with frequency domain attributions
    attribution_methods = {
        "DFT-LRP": dft_lrp_freq_wrapper,
        "Vanilla-Gradient-Freq": vanilla_gradient_freq_wrapper,
        "Gradient-Input-Freq": grad_input_freq_wrapper,
        "SmoothGrad-Freq": smoothgrad_freq_wrapper,
        "Occlusion-Freq": occlusion_freq_wrapper
    }
    #load test sata
    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    print(f"Loaded test data with {len(test_loader.dataset)} samples")

    test_frequency_visualization(model_path, data_dir)

    # Run window flipping evaluation - most important first
    print("\n===== Evaluating with most important frequency windows flipped first =====")
    results_most_first, agg_results_most_first = run_frequency_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        max_samples=max_samples,
        output_dir=output_dir,
        device=device
    )

    # Run window flipping evaluation - least important first
    print("\n===== Evaluating with least important frequency windows flipped first =====")
    results_least_first, agg_results_least_first = run_frequency_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        max_samples=max_samples,
        output_dir=output_dir,
        device=device
    )

    # Compute and print faithfulness ratios
    print("\n===== Frequency Domain Faithfulness Ratios =====")
    print("(Ratio of AUC when flipping least important windows first vs. most important first)")
    print("Higher ratio indicates better explanation faithfulness")
    print("-" * 60)
    print(f"{'Method':<20} {'AUC Ratio':<12} {'Most First':<12} {'Least First':<12}")
    print("-" * 60)

    for method_name in attribution_methods.keys():
        most_auc = agg_results_most_first[method_name]["mean_auc"]
        least_auc = agg_results_least_first[method_name]["mean_auc"]

        if np.isnan(most_auc) or np.isnan(least_auc) or most_auc == 0:
            ratio = float('nan')
        else:
            ratio = least_auc / most_auc

        print(f"{method_name:<20} {ratio:<12.4f} {most_auc:<12.4f} {least_auc:<12.4f}")


if __name__ == "__main__":
    main()


