import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd


def time_frequency_window_flipping(model, sample, attribution_method, n_steps=20,
                                   window_size_time=5, window_size_freq=5, most_relevant_first=True,
                                   reference_value=None,
                                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on time-frequency domain relevances with improved memory management.
    """
    import gc
    import torch
    import numpy as np
    from scipy import signal as sp_signal

    # Ensure sample is on the correct device
    sample = sample.to(device)

    try:
        # Compute attributions
        attribution_results = attribution_method(model, sample)

        # Extract time-frequency domain relevance
        if isinstance(attribution_results, tuple) and len(attribution_results) >= 5:
            _, _, _, relevance_timefreq, signal_timefreq, input_signal, _, target_class = attribution_results
        else:
            raise ValueError("Attribution method must return time-frequency domain relevance")

        # Free memory right after extraction
        for item in attribution_results:
            if isinstance(item, torch.Tensor):
                del item

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Get dimensions of time-frequency representation
        n_channels, freq_bins, time_frames = relevance_timefreq.shape

        # Get original signal length (time_steps)
        time_steps = input_signal.shape[1]

        # Calculate number of windows in time and frequency dimensions
        n_time_windows = time_frames // window_size_time
        if time_frames % window_size_time > 0:
            n_time_windows += 1

        n_freq_windows = freq_bins // window_size_freq
        if freq_bins % window_size_freq > 0:
            n_freq_windows += 1

        # Calculate window importance by averaging relevance within each window
        window_importance = np.zeros((n_channels, n_freq_windows, n_time_windows))

        for channel in range(n_channels):
            for freq_window in range(n_freq_windows):
                for time_window in range(n_time_windows):
                    freq_start = freq_window * window_size_freq
                    freq_end = min((freq_window + 1) * window_size_freq, freq_bins)

                    time_start = time_window * window_size_time
                    time_end = min((time_window + 1) * window_size_time, time_frames)

                    # Average absolute attribution within the time-frequency window
                    window_importance[channel, freq_window, time_window] = np.mean(
                        np.abs(relevance_timefreq[channel, freq_start:freq_end, time_start:time_end])
                    )

        # Free memory after importance calculation
        del relevance_timefreq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

            # Free memory
            del original_output, original_prob

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Track model outputs
        scores = [original_score]
        flipped_pcts = [0.0]

        # Calculate windows to flip per step
        total_windows = n_channels * n_freq_windows * n_time_windows
        windows_per_step = max(1, total_windows // n_steps)

        # Create reference signal in time-frequency domain (zeros)
        ref_signal_tf = np.zeros_like(signal_timefreq) if reference_value is None else np.full_like(signal_timefreq,
                                                                                                    reference_value)

        # Iteratively flip time-frequency windows
        for step in range(1, n_steps + 1):
            # Calculate how many windows to flip at this step
            n_windows_to_flip = min(step * windows_per_step, total_windows)

            # Create a copy of the original time-frequency signal
            flipped_tf_signal = signal_timefreq.copy()

            # Directly modify time-frequency components based on window importance sorting
            for i in range(n_windows_to_flip):
                # Convert flat index to 3D indices
                flat_idx = sorted_indices[i]
                time_window = flat_idx % n_time_windows
                freq_window = (flat_idx // n_time_windows) % n_freq_windows
                channel_idx = flat_idx // (n_freq_windows * n_time_windows)

                # Get window boundaries
                freq_start = freq_window * window_size_freq
                freq_end = min((freq_window + 1) * window_size_freq, freq_bins)

                time_start = time_window * window_size_time
                time_end = min((time_window + 1) * window_size_time, time_frames)

                # Replace the time-frequency components with reference values
                flipped_tf_signal[channel_idx, freq_start:freq_end, time_start:time_end] = \
                    ref_signal_tf[channel_idx, freq_start:freq_end, time_start:time_end]

            # Convert modified time-frequency representation back to time domain
            flipped_time_signal = np.zeros_like(input_signal)

            try:
                # For ISTFT, we need to estimate reasonable STFT parameters based on our data
                # Estimate window size from signal dimensions
                estimated_window_size = freq_bins * 2  # A reasonable guess
                estimated_overlap = estimated_window_size // 2  # 50% overlap is common

                for channel in range(n_channels):
                    # Try using scipy's ISTFT with estimated parameters
                    _, reconstructed = sp_signal.istft(
                        flipped_tf_signal[channel],
                        fs=400,  # Sampling rate
                        window='hann',  # Common window function
                        nperseg=estimated_window_size,
                        noverlap=estimated_overlap,
                        input_onesided=True,  # Assume we're using real signals
                        boundary=True
                    )

                    # Ensure correct length
                    if len(reconstructed) != time_steps:
                        if len(reconstructed) > time_steps:
                            # Truncate if too long
                            flipped_time_signal[channel] = reconstructed[:time_steps]
                        else:
                            # Pad if too short
                            print(
                                f"Reconstructed signal length ({len(reconstructed)}) shorter than expected ({time_steps})")
                            flipped_time_signal[channel] = np.pad(
                                reconstructed,
                                (0, time_steps - len(reconstructed)),
                                mode='constant'
                            )
                    else:
                        flipped_time_signal[channel] = reconstructed

            except Exception as e:
                print(f"Error in inverse STFT: {e}")
                # Fallback: use bandstop filtering approach
                flipped_time_signal = input_signal.copy()

                for i in range(n_windows_to_flip):
                    # Convert flat index to 3D indices
                    flat_idx = sorted_indices[i]
                    time_window = flat_idx % n_time_windows
                    freq_window = (flat_idx // n_time_windows) % n_freq_windows
                    channel_idx = flat_idx // (n_freq_windows * n_time_windows)

                    # Map time window to time points in the signal
                    t_start = int(time_window * window_size_time * time_steps / time_frames)
                    t_end = int(min((time_window + 1) * window_size_time, time_frames) * time_steps / time_frames)

                    # Ensure we have a valid range
                    if t_end <= t_start:
                        t_end = min(t_start + 1, time_steps)

                    # Just zero out this time region as simplest fallback
                    flipped_time_signal[channel_idx, t_start:t_end] = 0

            # Convert to tensor for model input
            flipped_tensor = torch.tensor(flipped_time_signal, dtype=torch.float32, device=device).unsqueeze(0)

            # Get model prediction
            with torch.no_grad():
                output = model(flipped_tensor)
                prob = torch.softmax(output, dim=1)[0]
                score = prob[target_class].item()

                # Free memory
                del output, prob

            # Clean up
            del flipped_tensor, flipped_time_signal
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Track results0
            scores.append(score)
            flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

        # Final cleanup
        del signal_timefreq, ref_signal_tf, window_importance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return scores, flipped_pcts

    except Exception as e:
        print(f"Error in time_frequency_window_flipping: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
def time_frequency_window_flipping_batch(model, test_loader, attribution_methods, n_steps=10,
                                         window_size_time=5, window_size_freq=5, most_relevant_first=True,
                                         reference_value=None, max_samples=None,
                                         device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform time-frequency window flipping analysis on a batch of time series samples
    with improved memory management.
    """
    import gc
    import torch

    # Initialize results0 storage
    results = {method_name: [] for method_name in attribution_methods}

    # Keep track of the current sample count
    sample_count = 0

    # Process one method at a time to reduce memory pressure
    for method_name, attribution_func in attribution_methods.items():
        print(f"\nProcessing method: {method_name}")

        # Reset sample counter for this method
        method_sample_count = 0

        # Process each batch in the test loader
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc=f"Processing {method_name}")):
            # Aggressively clean up memory before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Process each sample in the batch
            for i in range(len(data)):
                # Increment sample counter
                sample_count += 1
                method_sample_count += 1

                # Print progress
                if method_sample_count % 5 == 0:
                    print(f"  Sample {method_sample_count}/{max_samples or 'all'} for {method_name}")

                # Check if we've reached the maximum number of samples
                if max_samples is not None and method_sample_count > max_samples:
                    break

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                try:
                    # Load sample to device
                    sample = data[i].to(device)
                    target = targets[i].to(device)

                    # Compute scores for this sample using the current method
                    scores, flipped_pcts = time_frequency_window_flipping(
                        model, sample, lambda m, s: attribution_func(m, s, target),
                        n_steps, window_size_time, window_size_freq,
                        most_relevant_first, reference_value, device
                    )

                    # Store results0
                    results[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": scores,
                        "flipped_pcts": flipped_pcts,
                        "auc": np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
                    })

                    # Explicitly delete temporary variables
                    del sample, target, scores, flipped_pcts

                except Exception as e:
                    print(f"Error processing sample {method_sample_count} with method {method_name}: {str(e)}")
                    # Store empty result to maintain sample count consistency
                    results[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": None,
                        "flipped_pcts": None,
                        "auc": float('nan')
                    })

                # Force garbage collection after each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Check if we've reached the maximum number of samples for this method
            if max_samples is not None and method_sample_count >= max_samples:
                print(f"Reached maximum number of samples ({max_samples}) for method {method_name}")
                break

    return results

def visualize_time_frequency_window_flipping(model, sample, attribution_method,
                                             n_steps=5, window_size_time=5, window_size_freq=5,
                                             most_relevant_first=True, reference_value=None,
                                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of time-frequency window flipping.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that returns time-frequency attributions
        n_steps: Number of steps to visualize
        window_size_time: Size of windows in time dimension
        window_size_freq: Size of windows in frequency dimension
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        device: Device to run computations on
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute attributions
    attribution_results = attribution_method(model, sample)

    # Extract time-frequency domain relevance
    if isinstance(attribution_results, tuple) and len(attribution_results) >= 7:
        # The compute_enhanced_dft_lrp function returns:
        # (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
        #  input_signal, freqs, target)
        _, _, _, relevance_timefreq, signal_timefreq, input_signal, freqs, target_class = attribution_results
    else:
        raise ValueError("Attribution method must return time-frequency domain relevance")

    # Get dimensions of time-frequency representation
    n_channels, freq_bins, time_frames = relevance_timefreq.shape

    # Calculate number of windows in time and frequency dimensions
    n_time_windows = time_frames // window_size_time
    if time_frames % window_size_time > 0:
        n_time_windows += 1

    n_freq_windows = freq_bins // window_size_freq
    if freq_bins % window_size_freq > 0:
        n_freq_windows += 1

    # Calculate window importance by averaging relevance within each window
    window_importance = np.zeros((n_channels, n_freq_windows, n_time_windows))

    for channel in range(n_channels):
        for freq_window in range(n_freq_windows):
            for time_window in range(n_time_windows):
                freq_start = freq_window * window_size_freq
                freq_end = min((freq_window + 1) * window_size_freq, freq_bins)

                time_start = time_window * window_size_time
                time_end = min((time_window + 1) * window_size_time, time_frames)

                # Average absolute attribution within the time-frequency window
                window_importance[channel, freq_window, time_window] = np.mean(
                    np.abs(relevance_timefreq[channel, freq_start:freq_end, time_start:time_end])
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

    # Create reference signal in time-frequency domain (zeros)
    ref_signal_tf = np.zeros_like(signal_timefreq) if reference_value is None else np.full_like(signal_timefreq,
                                                                                                reference_value)

    # Calculate windows to flip per step
    total_windows = n_channels * n_freq_windows * n_time_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Set up the plot - each row is a channel, and we plot time domain signal, TF spectrogram, and flipped versions
    fig, axes = plt.subplots(n_channels, n_steps + 1, figsize=(5 * (n_steps + 1), 3 * n_channels))
    fig.suptitle(
        f'Time-Frequency Window Flipping Progression ({"Most" if most_relevant_first else "Least"} Important First)',
        fontsize=16)

    # If we only have one channel, make sure axes is 2D
    if n_channels == 1:
        axes = np.expand_dims(axes, axis=0)

    # Plot the original signal and TF representation
    for channel in range(n_channels):
        # Plot original time-frequency representation
        im = axes[channel, 0].imshow(
            np.abs(signal_timefreq[channel]),
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, time_steps, 0, freq_bins]
        )
        axes[channel, 0].set_title(f'Original (Score: {original_score:.3f})')
        axes[channel, 0].set_xlabel('Time')
        axes[channel, 0].set_ylabel('Frequency')
        fig.colorbar(im, ax=axes[channel, 0], fraction=0.046, pad=0.04)

    # Create an EnhancedDFTLRP instance for inverse STDFT or use scipy.signal as fallback
    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=time_steps,
            leverage_symmetry=True,  # Assuming real signals
            precision=32,
            cuda=False,  # Use CPU for better stability
            window_shift=time_frames // 2,  # Adjust based on your actual params
            window_width=freq_bins * 2,
            window_shape="rectangle",
            create_dft=False,
            create_forward=False,
            create_inverse=False,
            create_transpose_inverse=False,
            create_stdft=True
        )
    except:
        dftlrp = None
        print("Warning: EnhancedDFTLRP initialization failed, will use scipy.signal as fallback")

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Create a copy of the original time-frequency signal
        flipped_tf_signal = signal_timefreq.copy()

        # Record which windows were flipped for visualization
        flipped_windows = []

        # Directly modify time-frequency components based on window importance sorting
        for i in range(n_windows_to_flip):
            # Convert flat index to channel, freq window, time window indices
            flat_idx = sorted_indices[i]

            # Calculate 3D indices
            time_window = flat_idx % n_time_windows
            freq_window = (flat_idx // n_time_windows) % n_freq_windows
            channel_idx = flat_idx // (n_freq_windows * n_time_windows)

            # Get window boundaries
            freq_start = freq_window * window_size_freq
            freq_end = min((freq_window + 1) * window_size_freq, freq_bins)

            time_start = time_window * window_size_time
            time_end = min((time_window + 1) * window_size_time, time_frames)

            # Replace the time-frequency components with reference values (zeros)
            flipped_tf_signal[channel_idx, freq_start:freq_end, time_start:time_end] = \
                ref_signal_tf[channel_idx, freq_start:freq_end, time_start:time_end]

            # Record this window for visualization
            flipped_windows.append((channel_idx, freq_start, freq_end, time_start, time_end))

        # Convert modified time-frequency representation back to time domain
        try:
            # Use scipy.signal for inverse STFT
            from scipy import signal as sp_signal

            flipped_time_signal = np.zeros_like(input_signal)

            for channel in range(n_channels):
                # Reconstruct the time domain signal using inverse STFT
                _, reconstructed_signal = sp_signal.istft(
                    flipped_tf_signal[channel],
                    fs=400,  # Sampling rate (assumed 400 Hz)
                    nperseg=window_size_freq * 2,
                    noverlap=window_size_freq,
                    boundary=None
                )

                # Make sure the reconstructed signal has the right length
                if len(reconstructed_signal) >= time_steps:
                    flipped_time_signal[channel] = reconstructed_signal[:time_steps]
                else:
                    # Pad if necessary
                    flipped_time_signal[channel, :len(reconstructed_signal)] = reconstructed_signal
                    print(
                        f"Warning: Reconstructed signal length ({len(reconstructed_signal)}) is shorter than expected ({time_steps})")

        except Exception as e:
            print(f"Error in inverse STFT: {e}")
            print("Falling back to using original time domain signal with masked regions")

            # Fallback: create a mask for the time domain from the time-frequency windows
            flipped_time_signal = input_signal.copy()

            # For each flipped window, estimate which time points it affects
            for channel_idx, freq_start, freq_end, time_start, time_end in flipped_windows:
                # Simple approach: map time frames to time points
                t_start = int(time_start * time_steps / time_frames)
                t_end = min(int(time_end * time_steps / time_frames), time_steps)

                # Zero out these time points
                if channel_idx < flipped_time_signal.shape[0] and t_end > t_start:
                    flipped_time_signal[channel_idx, t_start:t_end] = 0

        # Get model prediction
        flipped_tensor = torch.tensor(flipped_time_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Plot the flipped time-frequency representation
        for channel in range(n_channels):
            # Plot flipped time-frequency representation
            im = axes[channel, step].imshow(
                np.abs(flipped_tf_signal[channel]),
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, time_steps, 0, freq_bins]
            )
            axes[channel, step].set_title(
                f'{n_windows_to_flip / total_windows * 100:.1f}% Flipped\n(Score: {score:.3f})')
            axes[channel, step].set_xlabel('Time')

            # Highlight flipped regions
            for ch_idx, f_start, f_end, t_start, t_end in flipped_windows:
                if ch_idx == channel:
                    # Map TF coordinates to plot coordinates
                    t_start_plot = t_start * time_steps / time_frames
                    t_end_plot = t_end * time_steps / time_frames

                    # Draw rectangle around flipped region
                    rect = plt.Rectangle(
                        (t_start_plot, f_start),
                        t_end_plot - t_start_plot,
                        f_end - f_start,
                        linewidth=1,
                        edgecolor='r',
                        facecolor='none'
                    )
                    axes[channel, step].add_patch(rect)

            fig.colorbar(im, ax=axes[channel, step], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Clean up
    if dftlrp is not None:
        del dftlrp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def aggregate_results(results):
    """Aggregate window flipping results0 across all samples."""
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
    """Plot aggregated window flipping results0."""
    plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Aggregate Time-Frequency Window Flipping Results\n(Flipping {flip_order} windows first)', fontsize=16)
    plt.xlabel('Percentage of Time-Frequency Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def run_time_frequency_flipping_evaluation(model, test_loader, attribution_methods,
                                           n_steps=10, window_size_time=5, window_size_freq=5,
                                           most_relevant_first=True, max_samples=None,
                                           output_dir="./results0",
                                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """Run complete time-frequency window flipping evaluation with memory management."""
    import os
    import gc
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/tf_window_flipping_{flip_order}_{timestamp}"

    print(f"Starting time-frequency window flipping evaluation with {len(attribution_methods)} methods")
    print(
        f"Settings: n_steps={n_steps}, window_size_time={window_size_time}, window_size_freq={window_size_freq}, most_relevant_first={most_relevant_first}")

    # Clear memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Run window flipping on all samples
    results = time_frequency_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size_time=window_size_time,
        window_size_freq=window_size_freq,
        most_relevant_first=most_relevant_first,
        reference_value=None,
        max_samples=max_samples,
        device=device
    )

    print("Computing aggregated results0...")

    # Clear memory before aggregation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Aggregate results0
    agg_results = aggregate_results(results)

    print("Plotting results0...")

    # Plot and save aggregated results0
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)  # Close figure to free memory

    # Save numerical results0 to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nTime-Frequency Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results
def save_results_to_csv(agg_results, results, filename_prefix):
    """Save results0 to CSV files."""
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


# time_frequency_faithfulness_evaluation.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
from datetime import datetime

# Import your model and data loading utilities
from Classification.cnn1D_model import CNN1D_Wide
from torch.utils.data import DataLoader
from utils.baseline_xai import load_model


# Import attribution method wrappers
def tf_lrp_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Wrapper for LRP time-frequency attribution"""
    from utils.xai_implementation import compute_enhanced_dft_lrp

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute LRP relevance with time-frequency domain
    return compute_enhanced_dft_lrp(
        model=model,
        sample=sample,
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shape="rectangle"
    )


def tf_gradient_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Wrapper for Gradient time-frequency attribution"""
    from utils.xai_implementation import compute_enhanced_dft_for_xai_method

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="gradient",
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shape="rectangle"
    )


def tf_grad_input_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Wrapper for Gradient*Input time-frequency attribution"""
    from utils.xai_implementation import compute_enhanced_dft_for_xai_method

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="gradient_input",
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shape="rectangle"
    )


def tf_smoothgrad_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Wrapper for SmoothGrad time-frequency attribution"""
    from utils.xai_implementation import compute_enhanced_dft_for_xai_method

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="smoothgrad",
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shape="rectangle",
        num_samples=40,
        noise_level=1.0
    )


def tf_occlusion_wrapper(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Wrapper for Occlusion time-frequency attribution"""
    from utils.xai_implementation import compute_enhanced_dft_for_xai_method

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="occlusion",
        label=label,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shape="rectangle",
        occlusion_type="zero",
        window_size=40
    )


def main():
    # Add to the beginning of your script to configure PyTorch's memory management
    # Limit PyTorch's maximum memory usage (adjust percentage as needed)
    if torch.cuda.is_available():
        # Limit to using 80% of available GPU memory
        torch.cuda.set_per_process_memory_fraction(0.8)
        # Use more aggressive memory cleanup
        torch.cuda.empty_cache()
        print(
            f"Limited PyTorch to use 80% of available GPU memory: {torch.cuda.get_device_properties(0).total_memory * 0.8 / 1e9:.2f} GB")


    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt.ckpt"  # Path to your trained model
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results"
    n_steps = 10
    window_size_time = 10
    window_size_freq = 10
    max_samples = 5  # Reduce sample count to avoid memory issues

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Set up attribution methods for time-frequency domain
    attribution_methods = {
        "TF-LRP": tf_lrp_wrapper,
        "TF-GradInput": tf_grad_input_wrapper,
        "TF-SmoothGrad": tf_smoothgrad_wrapper,
        "TF-Occlusion": tf_occlusion_wrapper
    }

    # Load test data
    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    print(f"Loaded test data with {len(test_loader.dataset)} samples")

    # Memory management function
    def clear_memory():
        print("Clearing memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Run time-frequency window flipping evaluation - most important first
    print("\n===== Evaluating with most important time-frequency windows flipped first =====")
    clear_memory()

    results_most_first, agg_results_most_first = run_time_frequency_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size_time=window_size_time,
        window_size_freq=window_size_freq,
        most_relevant_first=True,
        max_samples=max_samples,
        output_dir=output_dir,
        device=device
    )

    # Clear memory before next evaluation
    clear_memory()

    # Run time-frequency window flipping evaluation - least important first
    print("\n===== Evaluating with least important time-frequency windows flipped first =====")
    results_least_first, agg_results_least_first = run_time_frequency_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size_time=window_size_time,
        window_size_freq=window_size_freq,
        most_relevant_first=False,
        max_samples=max_samples,
        output_dir=output_dir,
        device=device
    )

    # Final memory cleanup
    clear_memory()

    # Compute and print faithfulness ratios
    print("\n===== Time-Frequency Faithfulness Ratios =====")
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