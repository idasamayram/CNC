import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd
import os
import random
import h5py
from scipy import stats
from scipy import signal as sp_signal
from pathlib import Path
from datetime import datetime
from utils.dft_lrp import EnhancedDFTLRP
from utils.xai_implementation import compute_enhanced_dft_for_xai_method
# Import your model loading utility
from utils.baseline_xai import load_model, occlusion_signal_relevance


def timefreq_guided_time_window_flipping_single(model, sample, attribution_method, target_class=None,
                                                n_steps=5, window_size=10, most_relevant_first=True,
                                                device="cuda" if torch.cuda.is_available() else "cpu",
                                                leverage_symmetry=True, sampling_rate=400,
                                                window_width=64, window_shift=16,
                                                window_shape="rectangle"):
    """
    Memory-efficient time-frequency guided window flipping:
    1. Compute time-frequency attributions to identify important regions
    2. Map important time-frequency regions to the corresponding time domain segments
    3. Directly flip those time domain segments with class-specific reference values
    4. Measure impact on model predictions
    """
    # clean torch cache and gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()
            print(f"No target class provided, using predicted class: {target_class}")

        original_score = original_prob[target_class].item()
        print(f"Original score for class {target_class}: {original_score:.4f}")

    # Use class-specific reference values
    if target_class == 0:  # Normal/Good class
        # Use mild noise as reference
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 0.5)

        print(f"Using mild_noise reference for normal class sample")
    else:  # Faulty/Bad class
        # Use zeros as reference
        reference_value = torch.zeros_like(sample)
        print(f"Using zero reference for faulty class sample")

    # Create reference value as numpy array for later use
    reference_np = reference_value.detach().cpu().numpy()

    # Get time-frequency domain attributions - only calculate once for efficiency
    try:
        print(f"Computing time-frequency attributions...")
        # Clear memory before heavy computation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _, _, _, relevance_timefreq, signal_timefreq, input_signal, freqs, _ = attribution_method(
            model=model,
            sample=sample,
            target=target_class
        )

        print(f"Time-frequency attribution shape: {relevance_timefreq.shape}")
        print(f"Time-frequency signal shape: {signal_timefreq.shape}")

    except Exception as e:
        print(f"Error computing time-frequency attributions: {e}")
        raise

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get dimensions of time-frequency representation
    n_channels, n_freq_bins, n_time_frames = relevance_timefreq.shape

    # Create mapping from time-frequency windows to time domain segments
    tf_to_time_mapping = []

    for frame_idx in range(n_time_frames):
        # Calculate the corresponding time domain segment for this time-frequency frame
        time_start = frame_idx * window_shift
        time_end = min(time_start + window_width, time_steps)

        # Store the mapping
        tf_to_time_mapping.append((time_start, time_end))

    # Calculate importance of each time-frequency window
    # We'll use entire time-frequency frames (all frequencies for a given time frame)
    window_importance = np.zeros((n_channels, n_time_frames))

    for channel in range(n_channels):
        for frame_idx in range(n_time_frames):
            # Average absolute attribution across all frequency bins for this time frame
            window_importance[channel, frame_idx] = np.mean(
                np.abs(relevance_timefreq[channel, :, frame_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Map sorted indices to (channel, frame) coordinates
    indices = np.unravel_index(sorted_indices, (n_channels, n_time_frames))
    channel_indices = indices[0]
    frame_indices = indices[1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Create equal steps for visualization
    total_windows = n_channels * n_time_frames
    windows_per_step = total_windows // n_steps

    # Create a mask to track which time segments have been flipped
    time_mask = np.zeros((n_channels, time_steps), dtype=bool)

    # Iteratively flip time domain segments corresponding to important time-frequency windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get flipped sample
        flipped_time = input_signal.copy()

        # Create mask of flipped time segments for this step
        new_time_mask = time_mask.copy()

        # For each window to flip
        for i in range(n_windows_to_flip):
            channel_idx = channel_indices[i]
            frame_idx = frame_indices[i]

            # Get corresponding time domain segment
            time_start, time_end = tf_to_time_mapping[frame_idx]

            # Only flip if not already flipped
            if not np.all(time_mask[channel_idx, time_start:time_end]):
                # Flip the time domain segment
                flipped_time[channel_idx, time_start:time_end] = reference_np[channel_idx, time_start:time_end]

                # Mark as flipped in the mask
                new_time_mask[channel_idx, time_start:time_end] = True

        # Update the mask
        time_mask = new_time_mask

        # Calculate the percentage of total time points flipped
        flipped_pct = np.sum(time_mask) / (n_channels * time_steps) * 100.0

        # Get model output for flipped sample
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(flipped_pct)
        print(f"Step {step}: Score after flipping {flipped_pct:.1f}% of time points: {score:.4f}")

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores, flipped_pcts, time_mask


def timefreq_guided_time_window_flipping_batch(model, samples, attribution_methods, n_steps=5,
                                               window_size=10, most_relevant_first=True,
                                               max_samples=None, leverage_symmetry=True,
                                               sampling_rate=400, window_width=64, window_shift=16,
                                               window_shape="rectangle",
                                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform time-frequency guided time domain window flipping on a batch of samples.
    """
    # clean cache and gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Initialize results storage
    results = {method_name: [] for method_name in attribution_methods}

    # Keep track of the current sample count
    sample_count = 0

    # Process each sample
    for sample_idx, (sample_data, target) in enumerate(tqdm(samples, desc="Processing samples")):
        # Skip if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break

        # Move sample to device
        sample = sample_data.to(device)
        target_class = target

        # Increment sample counter
        sample_count += 1

        # Print progress
        if sample_count % 5 == 0:
            print(f"Processing sample {sample_count}/{len(samples) if max_samples is None else max_samples}")

        # Process each attribution method
        for method_name, attribution_func in attribution_methods.items():
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                print(f"Processing sample {sample_count}, method: {method_name}")
                scores, flipped_pcts, _ = timefreq_guided_time_window_flipping_single(
                    model=model,
                    sample=sample,
                    attribution_method=attribution_func,
                    target_class=target_class,
                    n_steps=n_steps,
                    window_size=window_size,
                    most_relevant_first=most_relevant_first,
                    device=device,
                    leverage_symmetry=leverage_symmetry,
                    sampling_rate=sampling_rate,
                    window_width=window_width,
                    window_shift=window_shift,
                    window_shape=window_shape
                )

                # Calculate AUC
                auc = np.trapz(scores, flipped_pcts) / flipped_pcts[-1]

                # Store results
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": scores,
                    "flipped_pcts": flipped_pcts,
                    "auc": auc
                })

            except Exception as e:
                print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Store empty result
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results


def visualize_timefreq_guided_flipping_sample(model, sample, target, attribution_method,
                                              n_steps=5, window_size=10, most_relevant_first=True,
                                              save_path=None, leverage_symmetry=True, sampling_rate=400,
                                              window_width=64, window_shift=16, window_shape="rectangle",
                                              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of time-frequency guided time domain window flipping for a sample.
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Run flipping process and get the masks for visualization
    scores, flipped_pcts, final_mask = timefreq_guided_time_window_flipping_single(
        model=model,
        sample=sample,
        attribution_method=attribution_method,
        target_class=target,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        device=device,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape
    )

    # Get time-frequency representation for visualization (using scipy for memory efficiency)
    input_signal = sample.cpu().numpy()

    # Compute STFT for visualization
    spectrograms = []
    for c in range(n_channels):
        f, t, Zxx = sp_signal.stft(
            input_signal[c],
            fs=sampling_rate,
            window='hann',
            nperseg=window_width,
            noverlap=window_width - window_shift
        )
        spectrograms.append(np.abs(Zxx))

    # Use class-specific reference values
    if target == 0:  # Normal class
        # Use mild noise as reference
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 0.5)
        reference_type = "mild_noise"
    else:  # Faulty class
        # Use zeros as reference
        reference_value = torch.zeros_like(sample)
        reference_type = "zero"

    reference_np = reference_value.cpu().numpy()

    # Create a copy of original signal
    original_signal = input_signal.copy()

    # Calculate intermediate masks for visualization
    time_masks = []
    step_masks = []
    flipped_signals = []

    # Always include original (unflipped)
    time_masks.append(np.zeros_like(input_signal, dtype=bool))
    flipped_signals.append(original_signal.copy())

    # Create equal steps for visualization
    total_steps = min(n_steps, len(flipped_pcts) - 1)  # Exclude the initial point
    indices_to_show = [int(i * (len(flipped_pcts) - 1) / total_steps) for i in range(1, total_steps + 1)]

    # Get time-frequency attributions for visualization
    try:
        _, _, _, relevance_timefreq, _, _, _, _ = attribution_method(
            model=model,
            sample=sample,
            target=target
        )
    except:
        # If attribution fails, use dummy data
        n_time_frames = (time_steps - window_width) // window_shift + 1
        n_freq_bins = window_width // 2 + 1 if leverage_symmetry else window_width
        relevance_timefreq = np.zeros((n_channels, n_freq_bins, n_time_frames))

    # Create mapping from time-frequency frames to time domain segments
    tf_to_time_mapping = []
    for frame_idx in range(relevance_timefreq.shape[2]):
        time_start = frame_idx * window_shift
        time_end = min(time_start + window_width, time_steps)
        tf_to_time_mapping.append((time_start, time_end))

    # Calculate importance of each time-frequency window
    window_importance = np.zeros((n_channels, relevance_timefreq.shape[2]))
    for channel in range(n_channels):
        for frame_idx in range(relevance_timefreq.shape[2]):
            window_importance[channel, frame_idx] = np.mean(
                np.abs(relevance_timefreq[channel, :, frame_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Map sorted indices to (channel, frame) coordinates
    indices = np.unravel_index(sorted_indices, (n_channels, relevance_timefreq.shape[2]))
    channel_indices = indices[0]
    frame_indices = indices[1]

    # Create masks for visualization steps
    curr_mask = np.zeros((n_channels, time_steps), dtype=bool)

    total_windows = n_channels * relevance_timefreq.shape[2]
    windows_per_step = total_windows // total_steps

    for step in range(1, total_steps + 1):
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Create new mask for this step
        new_mask = curr_mask.copy()

        # Get flipped signal for this step
        flipped_signal = original_signal.copy()

        # Apply flips for this step
        for i in range(n_windows_to_flip):
            channel_idx = channel_indices[i]
            frame_idx = frame_indices[i]

            time_start, time_end = tf_to_time_mapping[frame_idx]
            new_mask[channel_idx, time_start:time_end] = True

            # Apply reference value
            flipped_signal[channel_idx, time_start:time_end] = reference_np[channel_idx, time_start:time_end]

        # Store mask and flipped signal
        time_masks.append(new_mask.copy())
        flipped_signals.append(flipped_signal.copy())

        # Update current mask
        curr_mask = new_mask

    # Create visualization
    plt.close('all')

    # Choose one channel to visualize
    channel_to_show = 0

    # Create figure with rows for time domain signal and spectrogram
    n_steps_to_show = min(5, len(time_masks))
    fig, axes = plt.subplots(2, n_steps_to_show, figsize=(4 * n_steps_to_show, 8), squeeze=False)

    # Choose steps to show (first, intermediate, last)
    if n_steps_to_show <= 3:
        steps_to_show = list(range(len(time_masks)))
    else:
        steps_to_show = [0]  # Always include original
        step_spacing = (len(time_masks) - 1) // (n_steps_to_show - 1)
        steps_to_show.extend([1 + i * step_spacing for i in range(n_steps_to_show - 1)])

    # Plot each step
    for col, step_idx in enumerate(steps_to_show):
        # Time domain plot
        ax_time = axes[0, col]
        ax_time.plot(flipped_signals[step_idx][channel_to_show])

        # Highlight flipped regions
        if step_idx > 0:
            for i in range(time_steps):
                if time_masks[step_idx][channel_to_show, i]:
                    ax_time.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.5)

        if step_idx == 0:
            title = "Original"
            score = scores[0]
        else:
            title = f"{flipped_pcts[step_idx]:.1f}% Flipped"
            score = scores[step_idx]

        ax_time.set_title(f"{title}\nScore: {score:.3f}")
        ax_time.set_ylabel("Amplitude")
        ax_time.set_xlabel("Time")

        # Compute spectrogram for this step
        f, t, Zxx = sp_signal.stft(
            flipped_signals[step_idx][channel_to_show],
            fs=sampling_rate,
            window='hann',
            nperseg=window_width,
            noverlap=window_width - window_shift
        )

        # Plot spectrogram
        ax_spec = axes[1, col]
        im = ax_spec.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        ax_spec.set_ylabel('Frequency [Hz]')
        ax_spec.set_xlabel('Time [s]')

        # Add colorbar to the last spectrogram
        if col == n_steps_to_show - 1:
            fig.colorbar(im, ax=ax_spec, label='Magnitude')

    # Add overall title
    plt.suptitle(f'Time-Frequency Guided Flipping - {"Most" if most_relevant_first else "Least"} Important First\n' +
                 f'Sample Class: {"Normal" if target == 0 else "Faulty"}, Reference: {reference_type}',
                 fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_both_classes_timefreq_guided(model, samples, attribution_method, n_steps=5, window_size=10,
                                           most_relevant_first=True, output_dir="./results",
                                           window_width=64, window_shift=16, window_shape="rectangle",
                                           leverage_symmetry=True, sampling_rate=400,
                                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize time-frequency guided window flipping for one sample of each class.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find one sample of each class
    normal_sample = None
    faulty_sample = None

    for sample_data, target in samples:
        if target == 0 and normal_sample is None:
            normal_sample = (sample_data, target)
            print("Found normal sample")
        elif target == 1 and faulty_sample is None:
            faulty_sample = (sample_data, target)
            print("Found faulty sample")

        if normal_sample is not None and faulty_sample is not None:
            break

    # Visualize normal sample
    if normal_sample is not None:
        sample_data, target = normal_sample
        print("\nVisualizing normal class sample (class 0)")

        try:
            plt.close('all')
            fig = visualize_timefreq_guided_flipping_sample(
                model=model,
                sample=sample_data,
                target=target,
                attribution_method=attribution_method,
                n_steps=n_steps,
                window_size=window_size,
                most_relevant_first=most_relevant_first,
                save_path=f"{output_dir}/tf_guided_flipping_normal.png",
                window_width=window_width,
                window_shift=window_shift,
                window_shape=window_shape,
                leverage_symmetry=leverage_symmetry,
                sampling_rate=sampling_rate,
                device=device
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error visualizing normal sample: {e}")
            import traceback
            traceback.print_exc()

    # Visualize faulty sample
    if faulty_sample is not None:
        sample_data, target = faulty_sample
        print("\nVisualizing faulty class sample (class 1)")

        try:
            plt.close('all')
            fig = visualize_timefreq_guided_flipping_sample(
                model=model,
                sample=sample_data,
                target=target,
                attribution_method=attribution_method,
                n_steps=n_steps,
                window_size=window_size,
                most_relevant_first=most_relevant_first,
                save_path=f"{output_dir}/tf_guided_flipping_faulty.png",
                window_width=window_width,
                window_shift=window_shift,
                window_shape=window_shape,
                leverage_symmetry=leverage_symmetry,
                sampling_rate=sampling_rate,
                device=device
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error visualizing faulty sample: {e}")
            import traceback
            traceback.print_exc()


def extract_important_timefreq_features(model, sample, attribution_method, target_class=None,
                                        n_windows=10, window_size=10, sampling_rate=400,
                                        window_width=64, window_shift=16, window_shape="rectangle",
                                        leverage_symmetry=True,
                                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extract features from important time-frequency regions and their corresponding time domain segments.
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

    # Get time-frequency attributions
    try:
        _, _, _, relevance_timefreq, signal_timefreq, input_signal, freqs, _ = attribution_method(
            model=model,
            sample=sample,
            target=target_class
        )

        print(f"Time-frequency attribution shape: {relevance_timefreq.shape}")
    except Exception as e:
        print(f"Error computing time-frequency attributions: {e}")
        return []

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create mapping from time-frequency frames to time domain segments
    tf_to_time_mapping = []
    for frame_idx in range(relevance_timefreq.shape[2]):
        time_start = frame_idx * window_shift
        time_end = min(time_start + window_width, time_steps)
        tf_to_time_mapping.append((time_start, time_end))

    # Calculate importance of each time-frequency window
    window_importance = np.zeros((n_channels, relevance_timefreq.shape[2]))
    for channel in range(n_channels):
        for frame_idx in range(relevance_timefreq.shape[2]):
            window_importance[channel, frame_idx] = np.mean(
                np.abs(relevance_timefreq[channel, :, frame_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)[::-1]  # Descending order

    # Take top n_windows
    top_indices = sorted_indices[:min(n_windows, len(sorted_indices))]

    # Map to channel, frame coordinates
    indices = np.unravel_index(top_indices, (n_channels, relevance_timefreq.shape[2]))
    channel_indices = indices[0]
    frame_indices = indices[1]

    # Extract features from important windows
    important_windows = []

    for i in range(len(top_indices)):
        channel_idx = channel_indices[i]
        frame_idx = frame_indices[i]

        # Get corresponding time domain segment
        time_start, time_end = tf_to_time_mapping[frame_idx]
        time_data = input_signal[channel_idx, time_start:time_end]

        # Get time-frequency data
        tf_data = signal_timefreq[channel_idx, :, frame_idx]
        relevance_data = relevance_timefreq[channel_idx, :, frame_idx]

        # Create window info
        window_info = {
            'channel': channel_idx,
            'frame_idx': frame_idx,
            'time_start': time_start,
            'time_end': time_end,
            'relevance': flat_importance[top_indices[i]]
        }

        # Extract time domain features
        window_info['avg_amplitude'] = np.mean(np.abs(time_data))
        window_info['max_amplitude'] = np.max(np.abs(time_data))
        window_info['std_amplitude'] = np.std(time_data)

        if len(time_data) > 1:
            window_info['zero_crossing_rate'] = np.sum(np.abs(np.diff(np.signbit(time_data)))) / (len(time_data) - 1)
            window_info['energy'] = np.sum(time_data ** 2)

            # Peak to average ratio
            avg_amplitude = np.mean(np.abs(time_data))
            if avg_amplitude > 0:
                window_info['peak_to_avg'] = np.max(np.abs(time_data)) / avg_amplitude

        # Extract frequency domain features
        magnitude = np.abs(tf_data)

        window_info['avg_magnitude'] = np.mean(magnitude)
        window_info['max_magnitude'] = np.max(magnitude)
        window_info['std_magnitude'] = np.std(magnitude)

        if np.sum(magnitude) > 0:
            # Spectral centroid
            freq_indices = np.arange(len(tf_data))
            freq_values = np.array([freqs[i] if i < len(freqs) else 0 for i in freq_indices])
            window_info['spectral_centroid'] = np.sum(freq_values * magnitude) / np.sum(magnitude)

            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            window_info['peak_freq_hz'] = freqs[peak_idx] if peak_idx < len(freqs) else 0

            # Spectral bandwidth (spread around centroid)
            centroid = window_info['spectral_centroid']
            window_info['spectral_bandwidth'] = np.sqrt(
                np.sum(((freq_values - centroid) ** 2) * magnitude) / np.sum(magnitude)
            )

        # Add time-frequency domain specific features
        # Time persistence measures how consistent the signal is across time
        magnitude_sums = np.sum(magnitude)
        if magnitude_sums > 0:
            window_info['power_density'] = magnitude_sums / (freqs[-1] if len(freqs) > 0 else 1)

        # Add to list of important windows
        important_windows.append(window_info)

    return important_windows


def collect_important_timefreq_features(model, samples, attribution_method,
                                        n_samples_per_class=10, n_windows=10, window_size=10,
                                        sampling_rate=400, window_width=64, window_shift=16,
                                        window_shape="rectangle", leverage_symmetry=True,
                                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Collect features from important time-frequency windows across multiple samples.
    """
    all_windows = []

    # Separate samples by class
    class_0_samples = [(s, t) for s, t in samples if t == 0]
    class_1_samples = [(s, t) for s, t in samples if t == 1]

    # Process class 0 samples
    n_class_0 = min(n_samples_per_class, len(class_0_samples))
    for i in range(n_class_0):
        sample, target = class_0_samples[i]

        try:
            print(f"Processing normal sample {i + 1}/{n_class_0}")
            windows = extract_important_timefreq_features(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                sampling_rate=sampling_rate,
                window_width=window_width,
                window_shift=window_shift,
                window_shape=window_shape,
                leverage_symmetry=leverage_symmetry,
                device=device
            )

            # Add sample metadata
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 0
                window['class_name'] = 'normal'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing normal sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Process class 1 samples
    n_class_1 = min(n_samples_per_class, len(class_1_samples))
    for i in range(n_class_1):
        sample, target = class_1_samples[i]

        try:
            print(f"Processing faulty sample {i + 1}/{n_class_1}")
            windows = extract_important_timefreq_features(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                sampling_rate=sampling_rate,
                window_width=window_width,
                window_shift=window_shift,
                window_shape=window_shape,
                leverage_symmetry=leverage_symmetry,
                device=device
            )

            # Add sample metadata
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 1
                window['class_name'] = 'faulty'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing faulty sample {i}: {e}")
            import traceback
            traceback.print_exc()

    return all_windows


def analyze_timefreq_features(windows, output_dir="./results/timefreq_features"):
    """
    Analyze features from important time-frequency windows.
    """
    import pandas as pd
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    features_to_keep = [
        'class', 'class_name', 'channel', 'sample_idx', 'relevance',
        'avg_amplitude', 'max_amplitude', 'std_amplitude', 'zero_crossing_rate',
        'energy', 'peak_to_avg', 'avg_magnitude', 'max_magnitude', 'std_magnitude',
        'spectral_centroid', 'peak_freq_hz', 'spectral_bandwidth', 'power_density'
    ]

    # Filter out windows with missing features
    valid_windows = []
    for window in windows:
        if all(k in window for k in ['relevance', 'spectral_centroid']):
            valid_windows.append(window)

    if not valid_windows:
        print("No valid windows found for analysis")
        return None

    # Create DataFrame with available features
    df = pd.DataFrame([{k: v for k, v in w.items() if k in features_to_keep} for w in valid_windows])

    # Calculate class statistics for each feature
    features = [col for col in df.columns if col not in ['class', 'class_name', 'channel', 'sample_idx']]

    class_stats = []
    for feature in features:
        if feature not in df.columns:
            continue

        normal_values = df[df['class'] == 0][feature].dropna()
        faulty_values = df[df['class'] == 1][feature].dropna()

        if len(normal_values) > 0 and len(faulty_values) > 0:
            try:
                t_stat, p_value = stats.ttest_ind(normal_values, faulty_values, equal_var=False)

                class_stats.append({
                    'feature': feature,
                    't_statistic': abs(t_stat),
                    'p_value': p_value,
                    'normal_mean': normal_values.mean(),
                    'faulty_mean': faulty_values.mean(),
                    'normal_std': normal_values.std(),
                    'faulty_std': faulty_values.std(),
                    'difference_pct': ((faulty_values.mean() - normal_values.mean()) /
                                       normal_values.mean() * 100) if normal_values.mean() != 0 else 0
                })
            except Exception as e:
                print(f"Error calculating statistics for {feature}: {e}")

    # Sort by statistical significance
    class_stats.sort(key=lambda x: x['p_value'])
    class_stats_df = pd.DataFrame(class_stats)

    # Save results
    if len(class_stats_df) > 0:
        class_stats_df.to_csv(f"{output_dir}/timefreq_feature_stats.csv", index=False)
        df.to_csv(f"{output_dir}/timefreq_features_data.csv", index=False)

        # Create visualizations
        plt.close('all')

        # Plot top 5 most discriminative features
        top_features = class_stats_df.head(min(5, len(class_stats_df)))['feature'].tolist()

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i + 1)

            sns.boxplot(x='class_name', y=feature, data=df)
            plt.title(f"{feature}\np={class_stats_df[class_stats_df['feature'] == feature]['p_value'].values[0]:.4f}")

            if i >= 2:
                plt.xlabel('Class')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_discriminative_features.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create correlation matrix
        plt.figure(figsize=(12, 10))

        # Filter numeric features
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]

        if numeric_features:
            corr = df[numeric_features].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Matrix')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()

        # Create channel distribution plot
        plt.figure(figsize=(10, 6))

        normal_channel_counts = df[df['class'] == 0]['channel'].value_counts().sort_index()
        faulty_channel_counts = df[df['class'] == 1]['channel'].value_counts().sort_index()

        # Ensure all channels are represented
        all_channels = sorted(set(df['channel'].unique()))
        x = np.arange(len(all_channels))
        width = 0.35

        normal_counts = [normal_channel_counts.get(c, 0) for c in all_channels]
        faulty_counts = [faulty_channel_counts.get(c, 0) for c in all_channels]

        plt.bar(x - width / 2, normal_counts, width, label='Normal')
        plt.bar(x + width / 2, faulty_counts, width, label='Faulty')

        plt.xlabel('Channel')
        plt.ylabel('Count')
        plt.title('Channel Distribution of Important Time-Frequency Windows')
        plt.xticks(x, [f'Channel {c}' for c in all_channels])
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/channel_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    return class_stats_df


def aggregate_results(results):
    """
    Aggregate window flipping results across all samples.
    This function is the same as in your existing code, kept for compatibility.
    """
    agg_results = {}

    for method_name, samples in results.items():
        # Initialize storage for this method
        method_results = {
            "avg_scores": [],
            "flipped_pcts": None,
            "auc_values": [],
            "mean_auc": None,
            "std_auc": None,
            "class_specific": {}  # Add class-specific metrics
        }

        # Filter out samples with errors
        valid_samples = [s for s in samples if s["scores"] is not None]

        if not valid_samples:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = method_results
            continue

        # Extract AUC values
        auc_values = [s["auc"] for s in valid_samples if not np.isnan(s["auc"])]

        # Calculate mean and std of AUC
        if len(auc_values) > 0:
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
        else:
            mean_auc = None
            std_auc = None

        # Interpolate scores to a common x-axis for averaging
        try:
            # Find common x-axis (flipped percentages)
            common_x = np.linspace(0, 100, 101)  # 0% to 100% in 1% increments

            # Interpolate each sample to common x-axis
            interpolated_scores = []

            for sample in valid_samples:
                flipped_pcts = sample["flipped_pcts"]
                scores = sample["scores"]

                # Check if we have enough points for interpolation
                if len(flipped_pcts) > 1 and len(scores) > 1:
                    # Ensure flipped_pcts is strictly increasing
                    if np.all(np.diff(flipped_pcts) > 0):
                        # Interpolate to common x-axis
                        from scipy.interpolate import interp1d
                        f = interp1d(flipped_pcts, scores, bounds_error=False, fill_value="extrapolate")
                        interpolated_scores.append(f(common_x))

            # Average interpolated scores
            if len(interpolated_scores) > 0:
                avg_scores = np.mean(interpolated_scores, axis=0)
            else:
                avg_scores = None

            # Calculate class-specific metrics
            class_specific = {}

            # Check for class 0 (normal) samples
            class_0_samples = [s for s in valid_samples if s["target"] == 0]
            if class_0_samples:
                class_0_aucs = [s["auc"] for s in class_0_samples if not np.isnan(s["auc"])]
                class_specific["class_0_metrics"] = {
                    "mean_auc": np.mean(class_0_aucs),
                    "std_auc": np.std(class_0_aucs),
                    "count": len(class_0_samples)
                }

                # Interpolate class 0 scores
                class_0_scores = []
                for sample in class_0_samples:
                    if len(sample["flipped_pcts"]) > 1 and len(sample["scores"]) > 1:
                        if np.all(np.diff(sample["flipped_pcts"]) > 0):
                            f = interp1d(sample["flipped_pcts"], sample["scores"], bounds_error=False,
                                         fill_value="extrapolate")
                            class_0_scores.append(f(common_x))

                if class_0_scores:
                    class_specific["class_0"] = {"avg_scores": np.mean(class_0_scores, axis=0)}
                    class_specific["class_0"]["auc"] = np.trapz(class_specific["class_0"]["avg_scores"],
                                                                common_x) / 100.0

            # Check for class 1 (faulty) samples
            class_1_samples = [s for s in valid_samples if s["target"] == 1]
            if class_1_samples:
                class_1_aucs = [s["auc"] for s in class_1_samples if not np.isnan(s["auc"])]
                class_specific["class_1_metrics"] = {
                    "mean_auc": np.mean(class_1_aucs),
                    "std_auc": np.std(class_1_aucs),
                    "count": len(class_1_samples)
                }

                # Interpolate class 1 scores
                class_1_scores = []
                for sample in class_1_samples:
                    if len(sample["flipped_pcts"]) > 1 and len(sample["scores"]) > 1:
                        if np.all(np.diff(sample["flipped_pcts"]) > 0):
                            f = interp1d(sample["flipped_pcts"], sample["scores"], bounds_error=False,
                                         fill_value="extrapolate")
                            class_1_scores.append(f(common_x))

                if class_1_scores:
                    class_specific["class_1"] = {"avg_scores": np.mean(class_1_scores, axis=0)}
                    class_specific["class_1"]["auc"] = np.trapz(class_specific["class_1"]["avg_scores"],
                                                                common_x) / 100.0

            # Store results
            method_results["avg_scores"] = avg_scores
            method_results["flipped_pcts"] = common_x
            method_results["auc_values"] = auc_values
            method_results["mean_auc"] = mean_auc
            method_results["std_auc"] = std_auc
            method_results["class_specific"] = class_specific

        except Exception as e:
            print(f"Error aggregating results for {method_name}: {e}")
            method_results["avg_scores"] = None
            method_results["flipped_pcts"] = None
            method_results["auc_values"] = auc_values
            method_results["mean_auc"] = mean_auc
            method_results["std_auc"] = std_auc
            method_results["class_specific"] = {}

        # Store results for this method
        agg_results[method_name] = method_results

    return agg_results


def plot_aggregate_results(agg_results, most_relevant_first=True):
    """
    Plot aggregated window flipping results.
    """
    # Close any existing figures
    plt.close('all')

    # Create new figure
    fig = plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(
        f'Aggregate Time-Frequency Guided Window Flipping Results\n(Flipping {flip_order} windows first, class-specific reference)',
        fontsize=16)
    plt.xlabel('Percentage of Time Points Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    return fig


def plot_class_specific_results(agg_results, most_relevant_first=True):
    """
    Plot class-specific window flipping results.
    """
    # Close any existing figures
    plt.close('all')

    # Set up the figure - one plot per method
    method_count = len(agg_results)
    fig, axes = plt.subplots(1, method_count, figsize=(6 * method_count, 6), squeeze=False)

    for i, (method_name, results) in enumerate(agg_results.items()):
        ax = axes[0, i]

        # Check if we have class-specific results
        if "class_specific" in results and results["class_specific"]:
            class_specific = results["class_specific"]

            # Plot for class 0 if available
            if "class_0" in class_specific and "avg_scores" in class_specific["class_0"]:
                ax.plot(results["flipped_pcts"], class_specific["class_0"]["avg_scores"],
                        label=f"Normal (AUC: {class_specific['class_0']['auc']:.4f})",
                        color='blue')

            # Plot for class 1 if available
            if "class_1" in class_specific and "avg_scores" in class_specific["class_1"]:
                ax.plot(results["flipped_pcts"], class_specific["class_1"]["avg_scores"],
                        label=f"Faulty (AUC: {class_specific['class_1']['auc']:.4f})",
                        color='red')

        # If no class-specific data, just plot the overall result if available
        elif results["avg_scores"] is not None and len(results["avg_scores"]) > 0:
            mean_auc = results.get("mean_auc")
            if mean_auc is not None:
                ax.plot(results["flipped_pcts"], results["avg_scores"],
                        label=f"Overall (AUC: {mean_auc:.4f})")
            else:
                ax.plot(results["flipped_pcts"], results["avg_scores"],
                        label="Overall (AUC: N/A)")

        ax.set_title(f'{method_name}')
        ax.set_xlabel('Percentage of Time Points Flipped (%)')
        ax.set_ylabel('Prediction Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.legend()

    flip_order = "most important" if most_relevant_first else "least important"
    fig.suptitle(
        f'Class-Specific Time-Frequency Guided Window Flipping Results\n(Flipping {flip_order} windows first)',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    return fig


def save_results_to_csv(agg_results, results, filename_prefix):
    """
    Save results to CSV files.
    """
    # Save aggregated results
    agg_data = []
    for method_name, method_results in agg_results.items():
        # Basic method metrics
        method_data = {
            "Method": method_name,
            "Mean_AUC": method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan'),
            "Std_AUC": method_results["std_auc"] if method_results["std_auc"] is not None else float('nan'),
            "Num_Samples": len(method_results["auc_values"])
        }

        # Add class-specific metrics if available
        if "class_specific" in method_results:
            class_specific = method_results["class_specific"]

            # Add class-0 metrics
            if "class_0_metrics" in class_specific:
                method_data["Class0_AUC"] = class_specific["class_0_metrics"]["mean_auc"]
                method_data["Class0_Count"] = class_specific["class_0_metrics"]["count"]

            # Add class-1 metrics
            if "class_1_metrics" in class_specific:
                method_data["Class1_AUC"] = class_specific["class_1_metrics"]["mean_auc"]
                method_data["Class1_Count"] = class_specific["class_1_metrics"]["count"]

        agg_data.append(method_data)

    # Save to CSV
    pd.DataFrame(agg_data).to_csv(f"{filename_prefix}_aggregate.csv", index=False)

    # Save detailed sample results
    samples_data = []
    for method_name, samples in results.items():
        for sample in samples:
            if sample["scores"] is None:
                continue

            # Basic sample data
            sample_data = {
                "Method": method_name,
                "Sample_Index": sample["sample_idx"],
                "True_Class": sample["target"],
                "AUC": sample["auc"]
            }

            samples_data.append(sample_data)

    if samples_data:
        pd.DataFrame(samples_data).to_csv(f"{filename_prefix}_samples.csv", index=False)


def run_timefreq_guided_flipping_evaluation(model, samples, attribution_methods,
                                            n_steps=5, window_size=10, most_relevant_first=True,
                                            max_samples=None, window_width=64, window_shift=16,
                                            window_shape="rectangle", leverage_symmetry=True,
                                            sampling_rate=400, output_dir="./results",
                                            device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete time-frequency guided window flipping evaluation.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/tf_guided_flipping_{flip_order}_{timestamp}"

    print(f"Starting time-frequency guided window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"STFT parameters: window_width={window_width}, window_shift={window_shift}, window_shape={window_shape}")

    # Run window flipping on samples
    results = timefreq_guided_time_window_flipping_batch(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        max_samples=max_samples,
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    print("Computing aggregated results...")

    # Aggregate results
    agg_results = aggregate_results(results)

    print("Creating visualizations...")

    # Plot and save aggregated results
    plt.close('all')
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first)
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Save numerical results
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nTime-Frequency Guided Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


def stdft_lrp_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-LRP attribution method.
    """
    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Call compute_enhanced_dft_for_xai_method with appropriate parameters
    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="lrp",
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=64,  # Smaller window for reduced memory
        window_shift=16,  # Smaller shift for reduced memory
        window_shape="rectangle"
    )


def stdft_gradient_input_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-GradInput attribution method.
    """
    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Call compute_enhanced_dft_for_xai_method with appropriate parameters
    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="gradient_input",
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=64,
        window_shift=16,
        window_shape="rectangle"
    )


def stdft_smoothgrad_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-SmoothGrad attribution method.
    """
    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Call compute_enhanced_dft_for_xai_method with appropriate parameters
    return compute_enhanced_dft_for_xai_method(
        model=model,
        sample=sample,
        xai_method="smoothgrad",
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=64,
        window_shift=16,
        window_shape="rectangle",
        num_samples=20,
        noise_level=0.2
    )


def stdft_occlusion_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-Occlusion attribution method.
    """
    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Call compute_enhanced_dft_for_xai_method with appropriate parameters
    try:
        return compute_enhanced_dft_for_xai_method(
            model=model,
            sample=sample,
            xai_method="occlusion",
            label=target,
            device=device,
            signal_length=sample.shape[1],
            leverage_symmetry=True,
            precision=32,
            sampling_rate=400,
            window_width=64,
            window_shift=16,
            window_shape="rectangle",
            occlusion_type="zero",
            window_size=40
        )
    except Exception as e:
        print(f"Error in STDFT-Occlusion: {e}")
        # Create empty placeholder results with correct shapes
        n_channels, time_steps = sample.shape
        freq_length = time_steps // 2 + 1  # For leverage_symmetry=True
        n_frames = (time_steps - 64) // 16 + 1  # Based on window_width and window_shift

        # Empty results with correct shapes
        relevance_time = np.zeros((n_channels, time_steps), dtype=np.float32)
        relevance_freq = np.zeros((n_channels, freq_length), dtype=np.float32)
        signal_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        relevance_timefreq = np.zeros((n_channels, freq_length, n_frames), dtype=np.float32)
        signal_timefreq = np.zeros((n_channels, freq_length, n_frames), dtype=np.complex128)
        freqs = np.fft.rfftfreq(time_steps, d=1.0 / 400)

        return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
                sample.cpu().numpy(), freqs, target)


def load_balanced_samples(data_dir, n_samples_per_class=10):
    """
    Load balanced samples from good and bad subfolders.

    Args:
        data_dir: Path to data directory
        n_samples_per_class: Number of samples to load per class

    Returns:
        samples: List of (sample, target) tuples
    """
    data_dir = Path(data_dir)
    samples = []

    # Load samples from good subfolder
    good_dir = data_dir / "good"
    good_files = list(good_dir.glob("*.h5"))

    if len(good_files) < n_samples_per_class:
        print(f"Warning: Only {len(good_files)} good samples available, requested {n_samples_per_class}")

    # Randomly sample
    selected_good = random.sample(good_files, min(n_samples_per_class, len(good_files)))

    # Load good samples
    for file_path in selected_good:
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3) for downsampled

        # Transpose to (3, 2000) for CNN
        data = np.transpose(data, (1, 0))
        samples.append((torch.tensor(data, dtype=torch.float32), 0))  # 0 = good class

    # Load samples from bad subfolder
    bad_dir = data_dir / "bad"
    bad_files = list(bad_dir.glob("*.h5"))

    if len(bad_files) < n_samples_per_class:
        print(f"Warning: Only {len(bad_files)} bad samples available, requested {n_samples_per_class}")

    # Randomly sample
    selected_bad = random.sample(bad_files, min(n_samples_per_class, len(bad_files)))

    # Load bad samples
    for file_path in selected_bad:
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3) for downsampled

        # Transpose to (3, 2000) for CNN
        data = np.transpose(data, (1, 0))
        samples.append((torch.tensor(data, dtype=torch.float32), 1))  # 1 = bad class

    print(f"Loaded {len(selected_good)} good samples and {len(selected_bad)} bad samples")
    return samples


def main():
    """
    Main function to run time-frequency guided window flipping evaluation.
    Also includes feature extraction from important time-frequency windows.
    """
    # Clean up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results/timefreq_domain"
    n_samples_per_class = 5  # Reduced for memory efficiency
    n_steps = 5  # Reduced steps for efficiency
    window_size = 10
    sampling_rate = 400

    # STFT parameters - reduced for memory efficiency
    window_width = 64  # Smaller window
    window_shift = 16  # Smaller shift
    window_shape = "rectangle"
    leverage_symmetry = True

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Set up time-frequency attribution methods dictionary
    timefreq_attribution_methods = {
        "STFT-LRP": stdft_lrp_wrapper,
        "STFT-GradInput": stdft_gradient_input_wrapper,
        "STFT-SmoothGrad": stdft_smoothgrad_wrapper,
        "STFT-Occlusion": stdft_occlusion_wrapper
    }

    # Load balanced samples
    samples = load_balanced_samples(data_dir, n_samples_per_class)

    # Visualize time-frequency guided window flipping for samples of both classes
    print("\n===== Visualizing Time-Frequency Guided Window Flipping for Both Classes =====")
    visualize_both_classes_timefreq_guided(
        model=model,
        samples=samples,
        attribution_method=stdft_lrp_wrapper,
        n_steps=5,
        window_size=window_size,
        most_relevant_first=True,
        output_dir=output_dir,
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    # Evaluate with most important windows flipped first
    print("\n===== Evaluating with most important time-frequency guided windows flipped first =====")
    results_most_first, agg_results_most_first = run_timefreq_guided_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=timefreq_attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        max_samples=len(samples),
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        output_dir=output_dir,
        device=device
    )

    # Evaluate with least important windows flipped first
    print("\n===== Evaluating with least important time-frequency guided windows flipped first =====")
    results_least_first, agg_results_least_first = run_timefreq_guided_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=timefreq_attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        max_samples=len(samples),
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        output_dir=output_dir,
        device=device
    )

    # Calculate and print faithfulness ratios
    print("\n===== Time-Frequency Guided Window Flipping Faithfulness Ratios =====")
    print("(Ratio of AUC when flipping least important windows first vs. most important first)")
    print("Higher ratio indicates better explanation faithfulness")
    print("-" * 70)
    print(f"{'Method':<20} {'AUC Ratio':<12} {'Most First':<12} {'Least First':<12}")
    print("-" * 70)

    for method_name in timefreq_attribution_methods.keys():
        most_auc = agg_results_most_first[method_name]["mean_auc"]
        least_auc = agg_results_least_first[method_name]["mean_auc"]

        if np.isnan(most_auc) or np.isnan(least_auc) or most_auc == 0:
            ratio = float('nan')
        else:
            ratio = least_auc / most_auc

        print(f"{method_name:<20} {ratio:<12.4f} {most_auc:<12.4f} {least_auc:<12.4f}")

    # Extract important time-frequency features using STFT-LRP
    print("\n===== Extracting Important Time-Frequency Features =====")
    timefreq_features = collect_important_timefreq_features(
        model=model,
        samples=samples,
        attribution_method=stdft_lrp_wrapper,
        n_samples_per_class=min(10, n_samples_per_class),  # Use fewer samples for feature extraction
        n_windows=10,  # Extract 10 top windows per sample
        window_size=window_size,
        sampling_rate=sampling_rate,
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape,
        leverage_symmetry=leverage_symmetry,
        device=device
    )

    # Analyze important time-frequency features
    print("\n===== Analyzing Important Time-Frequency Features =====")
    feature_stats = analyze_timefreq_features(timefreq_features, output_dir=f"{output_dir}/features")

    # Print top discriminative features
    if feature_stats is not None and not feature_stats.empty:
        print("\n===== Top Discriminative Time-Frequency Features =====")
        top_features = feature_stats.head(5)
        print(top_features[["feature", "p_value", "normal_mean", "faulty_mean", "difference_pct"]])

    print("\nTime-Frequency guided window flipping evaluation complete!")


if __name__ == "__main__":
    main()