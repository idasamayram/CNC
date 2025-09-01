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
from pathlib import Path
from datetime import datetime
from utils.dft_lrp import EnhancedDFTLRP
from utils.xai_implementation import compute_enhanced_dft_for_xai_method
# Import your model loading utility
from utils.baseline_xai import load_model, occlusion_signal_relevance


def time_frequency_window_flipping_single(model, sample, attribution_method, target_class=None,
                                          n_steps=20, window_size=40, most_relevant_first=True,
                                          reference_value="complete_zero",
                                          device="cuda" if torch.cuda.is_available() else "cpu",
                                          leverage_symmetry=True, sampling_rate=400,
                                          window_width=128, window_shift=None,
                                          window_shape="rectangle"):
    """
    Perform window flipping analysis in the time-frequency domain on a single time series sample.
    This function:
    1. Transforms the signal to time-frequency domain
    2. Computes attributions in time-frequency domain
    3. Identifies important windows and flips them
    4. Measures the impact on model prediction

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions in time-frequency domain
        target_class: Target class for explanation (default: model's prediction)
        n_steps: Number of steps to divide the flipping process
        window_size: Size of windows to flip in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows ("complete_zero", "mild_noise")
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows (default: window_width // 2)
        window_shape: Shape of the window ("rectangle", "hann", "hamming", etc.)

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Set default window shift if not provided
    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap by default

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()
            print(f"No target class provided, using predicted class: {target_class}")

        original_score = original_prob[target_class].item()
        print(f"Original score for class {target_class}: {original_score:.4f}")

    # Get time-frequency domain attributions
    # Note: For memory efficiency, we're storing the results directly
    try:
        print(f"Computing time-frequency attributions...")
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

    # Calculate window importance
    # We'll use 2D windows in the time-frequency space
    n_freq_windows = n_freq_bins // window_size + (1 if n_freq_bins % window_size > 0 else 0)
    n_time_windows = n_time_frames // window_size + (1 if n_time_frames % window_size > 0 else 0)

    print(f"Time-frequency space divided into {n_freq_windows}x{n_time_windows} windows per channel")

    # Initialize window importance array
    window_importance = np.zeros((n_channels, n_freq_windows, n_time_windows))

    # Calculate importance of each window based on average absolute attribution
    for channel in range(n_channels):
        for freq_window in range(n_freq_windows):
            freq_start = freq_window * window_size
            freq_end = min((freq_window + 1) * window_size, n_freq_bins)

            for time_window in range(n_time_windows):
                time_start = time_window * window_size
                time_end = min((time_window + 1) * window_size, n_time_frames)

                # Average absolute attribution within this time-frequency window
                window_importance[channel, freq_window, time_window] = np.mean(
                    np.abs(relevance_timefreq[channel, freq_start:freq_end, time_start:time_end])
                )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Prepare for window flipping
    total_windows = n_channels * n_freq_windows * n_time_windows

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Create exponential progression for more gradual steps
    flip_percentages = np.linspace(0, 1, n_steps + 1) ** 1.5
    windows_per_step = [int(total_windows * pct) for pct in flip_percentages[1:]]

    # Set reference value for time-frequency domain based on selected option
    if reference_value == "complete_zero":
        # Zero out frequency components completely
        reference_tf = np.zeros_like(signal_timefreq)
    elif reference_value == "mild_noise":
        # Use mild noise in time-frequency domain
        reference_tf = np.zeros_like(signal_timefreq)
        for c in range(n_channels):
            magnitude = np.abs(signal_timefreq[c]).std() * 0.5  # Use 0.5 as noise scaling factor
            phase = np.random.uniform(-np.pi, np.pi, (n_freq_bins, n_time_frames))
            reference_tf[c] = magnitude * np.exp(1j * phase)
    else:
        # Default to complete zero for any unrecognized reference value
        reference_tf = np.zeros_like(signal_timefreq)
        print(f"Warning: Unrecognized reference value '{reference_value}'. Using complete_zero instead.")

    # Setup EnhancedDFTLRP for inverse transform
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        window_shift=window_shift,
        window_width=window_width,
        window_shape=window_shape,
        create_forward=False,  # We only need inverse transform
        create_inverse=True,
        create_stdft=True,
        create_transpose_inverse=False
    )

    # Iteratively flip windows
    for step, n_windows_to_flip in enumerate(windows_per_step, 1):
        # Create a copy of the original time-frequency representation
        flipped_tf = signal_timefreq.copy()

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, freq_window, time_window indices
        # The formula unravels the flattened index back to 3D coordinates
        indices = np.unravel_index(windows_to_flip, (n_channels, n_freq_windows, n_time_windows))
        channel_indices = indices[0]
        freq_window_indices = indices[1]
        time_window_indices = indices[2]

        # Apply reference values to time-frequency domain
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            freq_window_idx = freq_window_indices[i]
            time_window_idx = time_window_indices[i]

            # Calculate window boundaries
            freq_start = freq_window_idx * window_size
            freq_end = min((freq_window_idx + 1) * window_size, n_freq_bins)
            time_start = time_window_idx * window_size
            time_end = min((time_window_idx + 1) * window_size, n_time_frames)

            # Apply reference value to this time-frequency window
            flipped_tf[channel_idx, freq_start:freq_end, time_start:time_end] = reference_tf[channel_idx,
                                                                                freq_start:freq_end,
                                                                                time_start:time_end]

        # Transform back to time domain
        flipped_time = np.zeros((n_channels, time_steps))

        for c in range(n_channels):
            try:
                # Convert to tensor for inverse transform
                # Reshape to format expected by EnhancedDFTLRP's inverse transform
                if dftlrp.has_st_inverse_fourier_layer:
                    # Format time-frequency data for dftlrp
                    tf_data = flipped_tf[c].reshape(-1)  # Flatten to 1D
                    tf_tensor = torch.tensor(tf_data, dtype=torch.complex64).to(device)

                    # Apply inverse transform
                    time_tensor = dftlrp.st_inverse_fourier_layer(tf_tensor.unsqueeze(0).real)
                    flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                else:
                    print("Warning: STDFT inverse layer not available, using alternative method")
                    # Alternative: Use standard ISTFT from scipy/librosa
                    from scipy import signal as sp_signal
                    f, t, Zxx = sp_signal.stft(input_signal[c], fs=sampling_rate,
                                               window='hann', nperseg=window_width,
                                               noverlap=window_width - window_shift)

                    # Reconstruct the flipped version in scipy's format
                    Zxx_flipped = Zxx.copy()
                    for freq_idx in range(Zxx.shape[0]):
                        for time_idx in range(Zxx.shape[1]):
                            # Map to our frequency and time bins
                            freq_bin = int(freq_idx * n_freq_bins / Zxx.shape[0])
                            time_bin = int(time_idx * n_time_frames / Zxx.shape[1])

                            if freq_bin < n_freq_bins and time_bin < n_time_frames:
                                Zxx_flipped[freq_idx, time_idx] = flipped_tf[c, freq_bin, time_bin]

                    # Inverse STFT
                    _, flipped_time[c] = sp_signal.istft(Zxx_flipped, fs=sampling_rate,
                                                         window='hann', nperseg=window_width,
                                                         noverlap=window_width - window_shift)

                    # Ensure correct length
                    if len(flipped_time[c]) > time_steps:
                        flipped_time[c] = flipped_time[c][:time_steps]
                    elif len(flipped_time[c]) < time_steps:
                        pad = np.zeros(time_steps - len(flipped_time[c]))
                        flipped_time[c] = np.concatenate([flipped_time[c], pad])

            except Exception as e:
                print(f"Error in inverse transform for channel {c}: {e}")
                # Fall back to original signal if inverse transform fails
                flipped_time[c] = input_signal[c]

        # Get model output for flipped sample
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        print(f"Step {step}: Score after flipping {n_windows_to_flip} windows: {score:.4f} ({flipped_pcts[-1]:.1f}%)")

    # Clean up
    del dftlrp, flipped_tf, reference_tf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts


def time_frequency_window_flipping_single_class_specific(model, sample, attribution_method, target_class=None,
                                                         n_steps=20, window_size=40, most_relevant_first=True,
                                                         device="cuda" if torch.cuda.is_available() else "cpu",
                                                         leverage_symmetry=True, sampling_rate=400,
                                                         window_width=128, window_shift=None,
                                                         window_shape="rectangle"):
    """
    Perform window flipping with class-specific reference values in time-frequency domain.
    Uses mild_noise for normal class (0) and complete_zero for faulty class (1).

    Args:
        Same as time_frequency_window_flipping_single, but automatically selects reference value
        based on target class.

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # Set class-specific reference value based on target class
    if target_class == 0:  # Normal/Good class
        reference_value = "mild_noise"
        print(f"Using mild_noise reference for normal class sample")
    else:  # Faulty/Bad class (class 1)
        reference_value = "complete_zero"
        print(f"Using complete_zero reference for faulty class sample")

    # Call the regular window flipping function with selected reference
    return time_frequency_window_flipping_single(
        model=model,
        sample=sample,
        attribution_method=attribution_method,
        target_class=target_class,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
        device=device,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        window_width=window_width,
        window_shift=window_shift,
        window_shape=window_shape
    )


def time_frequency_window_flipping_batch(model, samples, attribution_methods, n_steps=10,
                                         window_size=40, most_relevant_first=True,
                                         reference_value="complete_zero", max_samples=None,
                                         leverage_symmetry=True, sampling_rate=400,
                                         window_width=128, window_shift=None,
                                         window_shape="rectangle",
                                         device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis in time-frequency domain on a batch of time series samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of windows to flip in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        device: Device to run computations on

    Returns:
        results: Dictionary with results for each method and sample
    """
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

        # Print progress every 10 samples
        if sample_count % 10 == 0:
            print(f"Processing sample {sample_count}/{len(samples) if max_samples is None else max_samples}")

        # Process each attribution method
        for method_name, attribution_func in attribution_methods.items():
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                # Compute scores for this sample using the current method
                print(f"Processing sample {sample_count}, method: {method_name}")
                scores, flipped_pcts = time_frequency_window_flipping_single(
                    model=model,
                    sample=sample,
                    attribution_method=attribution_func,
                    target_class=target_class,
                    n_steps=n_steps,
                    window_size=window_size,
                    most_relevant_first=most_relevant_first,
                    reference_value=reference_value,
                    device=device,
                    leverage_symmetry=leverage_symmetry,
                    sampling_rate=sampling_rate,
                    window_width=window_width,
                    window_shift=window_shift,
                    window_shape=window_shape
                )

                # Calculate AUC for this sample
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
                # Store empty result to maintain sample count consistency
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results


def time_frequency_window_flipping_batch_class_specific(model, samples, attribution_methods, n_steps=10,
                                                        window_size=40, most_relevant_first=True,
                                                        max_samples=None, leverage_symmetry=True,
                                                        sampling_rate=400, window_width=128,
                                                        window_shift=None, window_shape="rectangle",
                                                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis with class-specific reference values.
    Uses mild_noise for normal class (0) and complete_zero for faulty class (1).

    Args:
        Same as time_frequency_window_flipping_batch, without reference_value parameter

    Returns:
        results: Dictionary with results for each method and sample
    """
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

        # Print progress every 10 samples
        if sample_count % 10 == 0:
            print(f"Processing sample {sample_count}/{len(samples) if max_samples is None else max_samples}")

        # Process each attribution method
        for method_name, attribution_func in attribution_methods.items():
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                # Compute scores with class-specific reference values
                print(f"Processing sample {sample_count}, method: {method_name}")
                scores, flipped_pcts = time_frequency_window_flipping_single_class_specific(
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

                # Calculate AUC for this sample
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
                # Store empty result to maintain sample count consistency
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results


def aggregate_results(results):
    """
    Aggregate window flipping results across all samples.

    Args:
        results: Dictionary with results for each method and sample

    Returns:
        agg_results: Dictionary with aggregated results
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

        # Collect all valid samples
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
            class_0_samples = [s for s in valid_samples if s["target"] == 0]
            class_1_samples = [s for s in valid_samples if s["target"] == 1]

            # Process class 0 (normal)
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

            # Process class 1 (faulty)
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
            method_results["flipped_pcts"] = common_x
            method_results["avg_scores"] = avg_scores
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


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot aggregated window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first
        reference_value: Value used to replace flipped windows

    Returns:
        fig: Figure object
    """
    # Close any existing figures first to avoid white plots
    plt.close('all')

    # Create new figure
    fig = plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(
        f'Aggregate Time-Frequency Window Flipping Results\n(Flipping {flip_order} windows first, \nReference Value Method: {reference_value})',
        fontsize=16)
    plt.xlabel('Percentage of Time-Frequency Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    return fig


def plot_class_specific_results(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot class-specific window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first
        reference_value: Value used to replace flipped windows

    Returns:
        fig: Figure object
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
        ax.set_xlabel('Percentage of Windows Flipped (%)')
        ax.set_ylabel('Prediction Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.legend()

    flip_order = "most important" if most_relevant_first else "least important"
    fig.suptitle(
        f'Class-Specific Time-Frequency Window Flipping Results\n(Flipping {flip_order} windows first)',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    return fig


def plot_auc_by_class(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot AUC values separated by class.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first
        reference_value: Value used to replace flipped windows

    Returns:
        fig: Figure object
    """
    # Close any existing figures
    plt.close('all')

    # Prepare data
    methods = []
    class0_aucs = []
    class0_counts = []
    class1_aucs = []
    class1_counts = []

    for method_name, results in agg_results.items():
        if "class_specific" in results:
            class_specific = results["class_specific"]

            if "class_0_metrics" in class_specific:
                methods.append(method_name)
                class0_aucs.append(class_specific["class_0_metrics"]["mean_auc"])
                class0_counts.append(class_specific["class_0_metrics"]["count"])

                if "class_1_metrics" in class_specific:
                    class1_aucs.append(class_specific["class_1_metrics"]["mean_auc"])
                    class1_counts.append(class_specific["class_1_metrics"]["count"])
                else:
                    class1_aucs.append(0)
                    class1_counts.append(0)

    if not methods:
        print("No class-specific AUC data available")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.35

    rects1 = ax.bar(x - width / 2, class0_aucs, width, label=f'Normal (n={class0_counts[0]})', color='lightgreen')
    rects2 = ax.bar(x + width / 2, class1_aucs, width, label=f'Faulty (n={class1_counts[0]})', color='salmon')

    ax.set_ylabel('Mean AUC')
    flip_order = "most important" if most_relevant_first else "least important"
    ax.set_title(f'AUC by Class for Different Attribution Methods\n'
                 f'(Flipping {flip_order} time-frequency windows first, \nReference Value Method: {reference_value})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Add text labels
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()

    return fig


def extract_important_timefreq_windows(model, sample, attribution_method, target_class=None,
                                       n_windows=10, window_size=10, sampling_rate=400,
                                       window_width=128, window_shift=None, window_shape="rectangle",
                                       leverage_symmetry=True,
                                       device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extract the most important windows from time-frequency domain based on attribution scores.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions in time-frequency domain
        target_class: Target class for explanation (default: model's prediction)
        n_windows: Number of top windows to extract
        window_size: Size of windows in time-frequency domain
        sampling_rate: Sampling rate of the signal in Hz
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        leverage_symmetry: Whether to use symmetry in STFT
        device: Device to run computations on

    Returns:
        important_windows: List of dictionaries containing window info
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Set default window shift if not provided
    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap by default

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

    # Get time-frequency domain attributions
    # This function should return all the necessary time-frequency representations
    try:
        _, _, _, relevance_timefreq, signal_timefreq, input_signal, freqs, _ = attribution_method(
            model=model,
            sample=sample,
            target=target_class
        )

        print(f"Relevance time-frequency shape: {relevance_timefreq.shape}")
    except Exception as e:
        print(f"Error computing time-frequency attributions: {e}")
        return []

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get dimensions of time-frequency representation
    n_channels, n_freq_bins, n_time_frames = relevance_timefreq.shape

    # Calculate window importance
    # We'll use 2D windows in the time-frequency space
    n_freq_windows = n_freq_bins // window_size + (1 if n_freq_bins % window_size > 0 else 0)
    n_time_windows = n_time_frames // window_size + (1 if n_time_frames % window_size > 0 else 0)

    print(f"Time-frequency space divided into {n_freq_windows}x{n_time_windows} windows per channel")

    # Initialize window importance array and store window info
    window_importance = np.zeros((n_channels, n_freq_windows, n_time_windows))
    window_info_map = {}  # To store window information keyed by (channel, freq_window, time_window)

    # Calculate importance of each window and extract window info
    for channel in range(n_channels):
        for freq_window in range(n_freq_windows):
            freq_start = freq_window * window_size
            freq_end = min((freq_window + 1) * window_size, n_freq_bins)

            # Calculate the actual frequency range for this window
            freq_range_start = freqs[freq_start] if freq_start < len(freqs) else 0
            freq_range_end = freqs[freq_end - 1] if freq_end - 1 < len(freqs) else sampling_rate / 2

            for time_window in range(n_time_windows):
                time_start = time_window * window_size
                time_end = min((time_window + 1) * window_size, n_time_frames)

                # Extract window data
                window_data = signal_timefreq[channel, freq_start:freq_end, time_start:time_end]
                relevance_data = relevance_timefreq[channel, freq_start:freq_end, time_start:time_end]

                # Calculate importance based on average absolute attribution
                importance = np.mean(np.abs(relevance_data))
                window_importance[channel, freq_window, time_window] = importance

                # Store window information
                window_info = {
                    'channel': channel,
                    'freq_window_idx': freq_window,
                    'time_window_idx': time_window,
                    'freq_start': freq_start,
                    'freq_end': freq_end,
                    'time_start': time_start,
                    'time_end': time_end,
                    'freq_hz_start': freq_range_start,
                    'freq_hz_end': freq_range_end,
                    'tf_data': window_data.copy(),
                    'relevance_data': relevance_data.copy(),
                    'relevance': importance
                }

                # Add spectral statistics
                magnitudes = np.abs(window_data)
                phases = np.angle(window_data)

                window_info['avg_magnitude'] = np.mean(magnitudes)
                window_info['max_magnitude'] = np.max(magnitudes)
                window_info['std_magnitude'] = np.std(magnitudes)

                if np.sum(magnitudes) > 0:
                    # Compute frequency centroid within this window
                    freq_indices = np.arange(freq_start, freq_end)
                    freq_values = np.array([freqs[i] if i < len(freqs) else 0 for i in freq_indices])
                    window_info['spectral_centroid'] = np.sum(freq_values * np.mean(magnitudes, axis=1)) / np.sum(
                        np.mean(magnitudes, axis=1))

                # Add time persistence (measure of how consistent the signal is across time)
                if window_data.shape[1] > 1:
                    time_variance = np.var(np.mean(magnitudes, axis=0))
                    max_possible_variance = np.square(np.mean(magnitudes))
                    window_info['time_persistence'] = 1.0 - min(
                        time_variance / max_possible_variance if max_possible_variance > 0 else 0, 1.0)
                else:
                    window_info['time_persistence'] = 1.0

                # Add window to map
                key = (channel, freq_window, time_window)
                window_info_map[key] = window_info

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)[::-1]  # Descending order

    # Take top n_windows
    top_indices = sorted_indices[:min(n_windows, len(sorted_indices))]

    # Convert flat indices to 3D coordinates
    top_coords = np.unravel_index(top_indices, (n_channels, n_freq_windows, n_time_windows))

    # Extract the most important windows
    important_windows = []
    for i in range(len(top_indices)):
        channel = top_coords[0][i]
        freq_window = top_coords[1][i]
        time_window = top_coords[2][i]

        key = (channel, freq_window, time_window)
        if key in window_info_map:
            important_windows.append(window_info_map[key])

    return important_windows


def collect_important_timefreq_windows(model, samples, attribution_method,
                                       n_samples_per_class=10, n_windows=10, window_size=10,
                                       sampling_rate=400, window_width=128, window_shift=None,
                                       window_shape="rectangle", leverage_symmetry=True,
                                       device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Collect important time-frequency windows from multiple samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions in time-frequency domain
        n_samples_per_class: Number of samples to process per class
        n_windows: Number of top windows to extract per sample
        window_size: Size of windows in time-frequency domain
        sampling_rate: Sampling rate of the signal in Hz
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        leverage_symmetry: Whether to use symmetry in STFT
        device: Device to run computations on

    Returns:
        all_windows: List of dictionaries containing window info
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
            # Extract important windows for this sample
            print(f"Processing normal sample {i + 1}/{n_class_0}")
            windows = extract_important_timefreq_windows(
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

            # Add sample metadata to each window
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 0
                window['class_name'] = 'normal'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing normal sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Process class 1 samples
    n_class_1 = min(n_samples_per_class, len(class_1_samples))
    for i in range(n_class_1):
        sample, target = class_1_samples[i]

        try:
            # Extract important windows for this sample
            print(f"Processing faulty sample {i + 1}/{n_class_1}")
            windows = extract_important_timefreq_windows(
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

            # Add sample metadata to each window
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 1
                window['class_name'] = 'faulty'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing faulty sample {i}: {str(e)}")
            import traceback
            traceback.print_exc()

    return all_windows


def visualize_timefreq_windows(windows, output_dir="./results/timefreq_windows"):
    """
    Visualize the important time-frequency windows.

    Args:
        windows: List of window dictionaries from extract_important_timefreq_windows
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group windows by class
    normal_windows = [w for w in windows if w.get('class') == 0]
    faulty_windows = [w for w in windows if w.get('class') == 1]

    # Create a scatter plot of frequency vs time persistence, colored by relevance
    plt.close('all')
    plt.figure(figsize=(12, 10))

    # Plot normal windows
    plt.subplot(2, 1, 1)
    if normal_windows:
        relevances = [w['relevance'] for w in normal_windows if 'relevance' in w]
        centroids = [w.get('spectral_centroid', 0) for w in normal_windows]
        persistence = [w.get('time_persistence', 0) for w in normal_windows]

        plt.scatter(centroids, persistence, c=relevances, cmap='viridis',
                    alpha=0.7, s=100, edgecolors='w')
        plt.colorbar(label='Relevance')
    plt.title('Important Time-Frequency Windows - Normal Samples')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Time Persistence')
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1)

    # Plot faulty windows
    plt.subplot(2, 1, 2)
    if faulty_windows:
        relevances = [w['relevance'] for w in faulty_windows if 'relevance' in w]
        centroids = [w.get('spectral_centroid', 0) for w in faulty_windows]
        persistence = [w.get('time_persistence', 0) for w in faulty_windows]

        plt.scatter(centroids, persistence, c=relevances, cmap='plasma',
                    alpha=0.7, s=100, edgecolors='w')
        plt.colorbar(label='Relevance')
    plt.title('Important Time-Frequency Windows - Faulty Samples')
    plt.xlabel('Spectral Centroid (Hz)')
    plt.ylabel('Time Persistence')
    plt.grid(alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/timefreq_windows_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create histograms of spectral centroids
    plt.figure(figsize=(12, 6))

    normal_centroids = [w.get('spectral_centroid', 0) for w in normal_windows]
    faulty_centroids = [w.get('spectral_centroid', 0) for w in faulty_windows]

    plt.hist(normal_centroids, bins=20, alpha=0.5, label='Normal', density=True)
    plt.hist(faulty_centroids, bins=20, alpha=0.5, label='Faulty', density=True)

    plt.title('Distribution of Spectral Centroids in Important Windows')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(left=0)

    plt.savefig(f"{output_dir}/spectral_centroid_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create channel distribution plot
    plt.figure(figsize=(10, 6))

    # Count windows by channel for each class
    channels = sorted(set(w['channel'] for w in windows if 'channel' in w))

    normal_counts = [len([w for w in normal_windows if w.get('channel') == c]) for c in channels]
    faulty_counts = [len([w for w in faulty_windows if w.get('channel') == c]) for c in channels]

    x = np.arange(len(channels))
    width = 0.35

    plt.bar(x - width / 2, normal_counts, width, label='Normal')
    plt.bar(x + width / 2, faulty_counts, width, label='Faulty')

    plt.xlabel('Channel')
    plt.ylabel('Count')
    plt.title('Channel Distribution of Important Time-Frequency Windows')
    plt.xticks(x, [f'Channel {c}' for c in channels])
    plt.legend()

    plt.savefig(f"{output_dir}/channel_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize example time-frequency windows
    try:
        # Select a few example windows from each class
        n_examples = min(3, len(normal_windows), len(faulty_windows))

        if n_examples > 0:
            # Sort windows by relevance
            normal_sorted = sorted(normal_windows, key=lambda w: w.get('relevance', 0), reverse=True)
            faulty_sorted = sorted(faulty_windows, key=lambda w: w.get('relevance', 0), reverse=True)

            # Display top windows
            plt.figure(figsize=(15, 10))

            for i in range(n_examples):
                # Normal window
                if i < len(normal_sorted):
                    plt.subplot(2, n_examples, i + 1)
                    window = normal_sorted[i]
                    if 'tf_data' in window and window['tf_data'] is not None:
                        plt.imshow(np.abs(window['tf_data']), aspect='auto', origin='lower',
                                   interpolation='none', cmap='viridis')
                        plt.title(f"Normal #{i + 1}\nRelevance: {window.get('relevance', 0):.4f}\n"
                                  f"Centroid: {window.get('spectral_centroid', 0):.1f} Hz")
                        plt.colorbar(label='Magnitude')

                # Faulty window
                if i < len(faulty_sorted):
                    plt.subplot(2, n_examples, n_examples + i + 1)
                    window = faulty_sorted[i]
                    if 'tf_data' in window and window['tf_data'] is not None:
                        plt.imshow(np.abs(window['tf_data']), aspect='auto', origin='lower',
                                   interpolation='none', cmap='plasma')
                        plt.title(f"Faulty #{i + 1}\nRelevance: {window.get('relevance', 0):.4f}\n"
                                  f"Centroid: {window.get('spectral_centroid', 0):.1f} Hz")
                        plt.colorbar(label='Magnitude')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/example_windows.png", dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error visualizing example windows: {e}")

    return


def analyze_timefreq_windows(windows, output_dir="./results/timefreq_windows"):
    """
    Analyze the important time-frequency windows and identify distinguishing features.

    Args:
        windows: List of window dictionaries
        output_dir: Directory to save results

    Returns:
        stats_df: DataFrame with feature statistics
    """
    import pandas as pd
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Filter windows with NaN values or missing keys
    valid_windows = []
    for window in windows:
        if not all(k in window for k in ['relevance', 'spectral_centroid', 'time_persistence']):
            continue
        if any(np.isnan(window.get(k, 0)) for k in ['relevance', 'spectral_centroid', 'time_persistence']):
            continue
        valid_windows.append(window)

    if not valid_windows:
        print("No valid windows found for analysis")
        return None

    # Convert to DataFrame for easier analysis
    features_to_keep = [
        'class', 'class_name', 'channel', 'sample_idx', 'relevance',
        'spectral_centroid', 'time_persistence', 'avg_magnitude', 'max_magnitude',
        'std_magnitude', 'freq_hz_start', 'freq_hz_end'
    ]

    df = pd.DataFrame([{k: v for k, v in w.items() if k in features_to_keep} for w in valid_windows])

    # Add derived features
    if len(df) > 0:
        # Frequency bandwidth
        if 'freq_hz_start' in df.columns and 'freq_hz_end' in df.columns:
            df['frequency_bandwidth'] = df['freq_hz_end'] - df['freq_hz_start']

        # Power density (power per Hz)
        if 'avg_magnitude' in df.columns and 'frequency_bandwidth' in df.columns:
            df['power_density'] = df['avg_magnitude'] / df['frequency_bandwidth'].replace(0, np.nan)

        # Calculate center frequency as the average of start and end
        if 'freq_hz_start' in df.columns and 'freq_hz_end' in df.columns:
            df['center_frequency'] = (df['freq_hz_start'] + df['freq_hz_end']) / 2

        # Add temporal variance (inverse of persistence)
        if 'time_persistence' in df.columns:
            df['temporal_variance'] = 1.0 - df['time_persistence']

    # Analyze class differences
    features = [
        'relevance', 'spectral_centroid', 'center_frequency', 'time_persistence',
        'avg_magnitude', 'max_magnitude', 'std_magnitude', 'frequency_bandwidth',
        'power_density', 'temporal_variance'
    ]

    # Filter out features not in the DataFrame
    features = [f for f in features if f in df.columns]

    class_stats = []
    for feature in features:
        if feature not in df.columns:
            continue

        normal_values = df[df['class'] == 0][feature].dropna()
        faulty_values = df[df['class'] == 1][feature].dropna()

        # Calculate separation statistics
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
        df.to_csv(f"{output_dir}/timefreq_windows_data.csv", index=False)

        # Create summary visualizations
        plt.close('all')
        plt.figure(figsize=(12, 8))

        # Plot top 5 most discriminative features (or fewer if less available)
        top_features = class_stats_df.head(min(5, len(class_stats_df)))['feature'].tolist()

        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i + 1)

            sns.boxplot(x='class_name', y=feature, data=df)
            plt.title(f"{feature}\np={class_stats_df[class_stats_df['feature'] == feature]['p_value'].values[0]:.4f}")

            if i >= 2:
                plt.xlabel('Class')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_discriminative_features.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create feature correlation matrix
        plt.close('all')
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

    return class_stats_df


def save_results_to_csv(agg_results, results, filename_prefix):
    """
    Save results to CSV files.

    Args:
        agg_results: Dictionary with aggregated results
        results: Dictionary with individual sample results
        filename_prefix: Prefix for filenames
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


def visualize_timefreq_window_flipping_sample(model, sample, target, attribution_method,
                                              n_steps=5, window_size=10, most_relevant_first=True,
                                              reference_value="complete_zero", save_path=None,
                                              leverage_symmetry=True, sampling_rate=400,
                                              window_width=128, window_shift=None,
                                              window_shape="rectangle",
                                              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of time-frequency domain window flipping for a single time series sample.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        target: Target label for the sample
        attribution_method: Function that generates attributions in time-frequency domain
        n_steps: Number of steps to visualize
        window_size: Size of windows in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        save_path: Optional path to save the figure
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        device: Device to run computations on

    Returns:
        fig: Matplotlib figure
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Set default window shift if not provided
    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap by default

    # Get original prediction and time-frequency representations
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = target
        original_score = original_prob[target_class].item()

    # Get time-frequency domain attributions
    try:
        _, _, _, relevance_timefreq, signal_timefreq, input_signal, freqs, _ = attribution_method(
            model=model,
            sample=sample,
            target=target_class
        )

        print(f"Time-frequency attribution shape: {relevance_timefreq.shape}")
    except Exception as e:
        print(f"Error computing time-frequency attributions: {e}")
        return None

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get dimensions of time-frequency representation
    n_channels, n_freq_bins, n_time_frames = relevance_timefreq.shape

    # Set reference value for time-frequency domain
    if reference_value == "complete_zero":
        # Zero out frequency components completely
        reference_tf = np.zeros_like(signal_timefreq)
    elif reference_value == "mild_noise":
        # Use mild noise in time-frequency domain
        reference_tf = np.zeros_like(signal_timefreq)
        for c in range(n_channels):
            magnitude = np.abs(signal_timefreq[c]).std() * 0.5
            phase = np.random.uniform(-np.pi, np.pi, (n_freq_bins, n_time_frames))
            reference_tf[c] = magnitude * np.exp(1j * phase)
    else:
        # Default to complete zero
        reference_tf = np.zeros_like(signal_timefreq)

    # Calculate window importance
    n_freq_windows = n_freq_bins // window_size + (1 if n_freq_bins % window_size > 0 else 0)
    n_time_windows = n_time_frames // window_size + (1 if n_time_frames % window_size > 0 else 0)

    window_importance = np.zeros((n_channels, n_freq_windows, n_time_windows))

    for channel in range(n_channels):
        for freq_window in range(n_freq_windows):
            freq_start = freq_window * window_size
            freq_end = min((freq_window + 1) * window_size, n_freq_bins)

            for time_window in range(n_time_windows):
                time_start = time_window * window_size
                time_end = min((time_window + 1) * window_size, n_time_frames)

                # Average absolute attribution within this window
                window_importance[channel, freq_window, time_window] = np.mean(
                    np.abs(relevance_timefreq[channel, freq_start:freq_end, time_start:time_end])
                )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Create DFTLRP object for inverse transform
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        window_shift=window_shift,
        window_width=window_width,
        window_shape=window_shape,
        create_forward=False,
        create_inverse=True,
        create_stdft=True
    )

    # Prepare for step-by-step visualization
    total_windows = n_channels * n_freq_windows * n_time_windows

    # Create exponential progression for more gradual steps
    flip_percentages = np.linspace(0, 1, n_steps + 1) ** 1.5
    windows_per_step = [int(total_windows * pct) for pct in flip_percentages[1:]]

    # Store results for each step
    original_tf = signal_timefreq.copy()
    flipped_tfs = [original_tf.copy()]  # Start with original
    flipped_times = [input_signal.copy()]
    scores = [original_score]
    flipped_pcts = [0.0]

    # Create masks to visualize flipped regions
    masks = [np.zeros((n_channels, n_freq_bins, n_time_frames), dtype=bool)]

    # Iteratively flip windows for each step
    for step, n_windows_to_flip in enumerate(windows_per_step, 1):
        # Create a copy of the original time-frequency representation
        flipped_tf = original_tf.copy()
        flipped_mask = np.zeros((n_channels, n_freq_bins, n_time_frames), dtype=bool)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, freq_window, time_window indices
        indices = np.unravel_index(windows_to_flip, (n_channels, n_freq_windows, n_time_windows))
        channel_indices = indices[0]
        freq_window_indices = indices[1]
        time_window_indices = indices[2]

        # Apply reference values to time-frequency domain
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            freq_window_idx = freq_window_indices[i]
            time_window_idx = time_window_indices[i]

            # Calculate window boundaries
            freq_start = freq_window_idx * window_size
            freq_end = min((freq_window_idx + 1) * window_size, n_freq_bins)
            time_start = time_window_idx * window_size
            time_end = min((time_window_idx + 1) * window_size, n_time_frames)

            # Apply reference value to this time-frequency window
            flipped_tf[channel_idx, freq_start:freq_end, time_start:time_end] = reference_tf[channel_idx,
                                                                                freq_start:freq_end,
                                                                                time_start:time_end]
            flipped_mask[channel_idx, freq_start:freq_end, time_start:time_end] = True

        # Transform back to time domain
        flipped_time = np.zeros((n_channels, time_steps))

        for c in range(n_channels):
            try:
                if dftlrp.has_st_inverse_fourier_layer:
                    # Format time-frequency data for dftlrp
                    tf_data = flipped_tf[c].reshape(-1)  # Flatten to 1D
                    tf_tensor = torch.tensor(tf_data, dtype=torch.complex64).to(device)

                    # Apply inverse transform
                    time_tensor = dftlrp.st_inverse_fourier_layer(tf_tensor.unsqueeze(0).real)
                    flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                else:
                    # Alternative: Use standard ISTFT from scipy
                    from scipy import signal as sp_signal
                    _, flipped_time[c] = sp_signal.istft(flipped_tf[c], fs=sampling_rate,
                                                         window='hann', nperseg=window_width,
                                                         noverlap=window_width - window_shift)

                    # Ensure correct length
                    if len(flipped_time[c]) > time_steps:
                        flipped_time[c] = flipped_time[c][:time_steps]
                    elif len(flipped_time[c]) < time_steps:
                        pad = np.zeros(time_steps - len(flipped_time[c]))
                        flipped_time[c] = np.concatenate([flipped_time[c], pad])

            except Exception as e:
                print(f"Error in inverse transform for channel {c}: {e}")
                # Fall back to original signal if inverse transform fails
                flipped_time[c] = input_signal[c]

        # Get model output for flipped sample
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Store results for this step
        flipped_tfs.append(flipped_tf.copy())
        flipped_times.append(flipped_time.copy())
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        masks.append(flipped_mask.copy())

    # Clean up
    del dftlrp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Create visualization
    plt.close('all')

    # For simplicity, we'll show just one channel (first channel)
    channel_to_show = 0

    # Create a figure with multiple rows: time domain signals and spectrograms
    fig = plt.figure(figsize=(15, 10))
    n_cols = min(3, len(flipped_tfs))  # Show at most 3 steps side by side

    # Select steps to visualize if we have more than 3
    step_indices = [0]  # Always include original
    if len(flipped_tfs) > 3:
        # Add intermediate and final step
        middle_idx = len(flipped_tfs) // 2
        step_indices.extend([middle_idx, len(flipped_tfs) - 1])
    else:
        # Add all remaining steps
        step_indices.extend(list(range(1, len(flipped_tfs))))

    # Plot time domain signals (top row)
    for i, step_idx in enumerate(step_indices):
        ax_time = plt.subplot(2, n_cols, i + 1)
        ax_time.plot(flipped_times[step_idx][channel_to_show])

        if step_idx == 0:
            title = "Original"
        else:
            title = f"{flipped_pcts[step_idx]:.1f}% Flipped"

        # Add score to title
        title += f"\nScore: {scores[step_idx]:.3f}"
        ax_time.set_title(title)

        ax_time.set_ylabel("Amplitude")
        ax_time.set_xlabel("Time")

    # Plot spectrograms (bottom row)
    for i, step_idx in enumerate(step_indices):
        ax_spec = plt.subplot(2, n_cols, n_cols + i + 1)

        # Plot magnitude spectrogram
        im = ax_spec.imshow(np.log1p(np.abs(flipped_tfs[step_idx][channel_to_show])),
                            aspect='auto', origin='lower', cmap='viridis',
                            interpolation='none')

        # Highlight flipped regions
        if step_idx > 0:
            # Create mask overlay for flipped regions
            mask_rgb = np.zeros((*masks[step_idx][channel_to_show].shape, 4))
            mask_rgb[masks[step_idx][channel_to_show], 3] = 0.3  # Alpha for transparent overlay
            mask_rgb[masks[step_idx][channel_to_show], 0] = 1.0  # Red component

            ax_spec.imshow(mask_rgb, aspect='auto', origin='lower', interpolation='none')

        ax_spec.set_xlabel("Time Frame")
        ax_spec.set_ylabel("Frequency Bin")

        # Add colorbar for the last plot
        if i == n_cols - 1:
            plt.colorbar(im, ax=ax_spec, label="Log Magnitude")

    # Add overall title
    plt.suptitle(
        f'Time-Frequency Window Flipping Progression - {"Most" if most_relevant_first else "Least"} Important First\n' +
        f'Sample Class: {"Normal" if target == 0 else "Faulty"}, Reference: {reference_value}',
        fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_both_classes_timefreq(model, samples, attribution_method, n_steps=5, window_size=10,
                                    most_relevant_first=True, output_dir="./results",
                                    window_width=128, window_shift=None, window_shape="rectangle",
                                    leverage_symmetry=True, sampling_rate=400,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize time-frequency window flipping for one normal sample and one faulty sample.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions in time-frequency domain
        n_steps: Number of steps to visualize
        window_size: Size of windows in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        output_dir: Directory to save visualizations
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        device: Device to run computations on
    """
    # Create output directory if it doesn't exist
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

    # Process normal sample
    if normal_sample is not None:
        sample_data, target = normal_sample

        try:
            print("\nVisualizing normal class sample (class 0)")
            print("Using mild_noise as reference value")

            # Visualize with mild_noise reference
            plt.close('all')
            fig = visualize_timefreq_window_flipping_sample(
                model=model,
                sample=sample_data,
                target=target,
                attribution_method=attribution_method,
                n_steps=n_steps,
                window_size=window_size,
                most_relevant_first=most_relevant_first,
                reference_value="mild_noise",
                save_path=f"{output_dir}/timefreq_flipping_normal.png",
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
    else:
        print("No normal sample found!")

    # Process faulty sample
    if faulty_sample is not None:
        sample_data, target = faulty_sample

        try:
            print("\nVisualizing faulty class sample (class 1)")
            print("Using complete_zero as reference value")

            # Visualize with complete_zero reference
            plt.close('all')
            fig = visualize_timefreq_window_flipping_sample(
                model=model,
                sample=sample_data,
                target=target,
                attribution_method=attribution_method,
                n_steps=n_steps,
                window_size=window_size,
                most_relevant_first=most_relevant_first,
                reference_value="complete_zero",
                save_path=f"{output_dir}/timefreq_flipping_faulty.png",
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
    else:
        print("No faulty sample found!")


def run_timefreq_window_flipping_evaluation(model, samples, attribution_methods,
                                            n_steps=10, window_size=10, most_relevant_first=True,
                                            reference_value="complete_zero", max_samples=None,
                                            window_width=128, window_shift=None, window_shape="rectangle",
                                            leverage_symmetry=True, sampling_rate=400,
                                            output_dir="./results",
                                            device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete time-frequency window flipping evaluation on test set and save results.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of windows to flip in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        output_dir: Directory to save results
        device: Device to run computations on

    Returns:
        results: Dictionary with individual sample results
        agg_results: Dictionary with aggregated results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/timefreq_window_flipping_{flip_order}_{reference_value}_{timestamp}"

    print(f"Starting time-frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}")
    print(f"STFT parameters: window_width={window_width}, window_shift={window_shift}, window_shape={window_shape}")

    # Run window flipping on all samples
    results = time_frequency_window_flipping_batch(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
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
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first, reference_value)
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Plot and save AUC by class
    plt.close('all')
    auc_fig = plot_auc_by_class(agg_results, most_relevant_first, reference_value)
    if auc_fig:
        auc_fig.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
        plt.close(auc_fig)

    # Save numerical results to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nTime-Frequency Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


def run_class_specific_timefreq_window_flipping_evaluation(model, samples, attribution_methods,
                                                           n_steps=10, window_size=10, most_relevant_first=True,
                                                           max_samples=None,
                                                           window_width=128, window_shift=None,
                                                           window_shape="rectangle",
                                                           leverage_symmetry=True, sampling_rate=400,
                                                           output_dir="./results",
                                                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete time-frequency window flipping evaluation with class-specific reference values.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of windows to flip in time-frequency domain
        most_relevant_first: If True, flip most relevant windows first
        max_samples: Maximum number of samples to process (None = all)
        window_width: Width of the STFT window
        window_shift: Shift between adjacent windows
        window_shape: Shape of the window function
        leverage_symmetry: Whether to use symmetry in STFT
        sampling_rate: Sampling rate of the signal in Hz
        output_dir: Directory to save results
        device: Device to run computations on

    Returns:
        results: Dictionary with individual sample results
        agg_results: Dictionary with aggregated results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/timefreq_window_flipping_class_specific_{flip_order}_{timestamp}"

    print(
        f"Starting class-specific time-frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference values: mild_noise for normal class, complete_zero for faulty class")
    print(f"STFT parameters: window_width={window_width}, window_shift={window_shift}, window_shape={window_shape}")

    # Run window flipping on all samples with class-specific references
    results = time_frequency_window_flipping_batch_class_specific(
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
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value="class_specific")
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first, reference_value="class_specific")
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Plot and save AUC by class
    plt.close('all')
    auc_fig = plot_auc_by_class(agg_results, most_relevant_first, reference_value="class_specific")
    if auc_fig:
        auc_fig.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
        plt.close(auc_fig)

    # Save numerical results to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nClass-Specific Time-Frequency Domain Window Flipping Evaluation Summary:")
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

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        target: Optional target class
        device: Device to run on

    Returns:
        The full tuple from compute_enhanced_dft_for_xai_method for time-frequency domain flipping
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
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shift=64,
        window_shape="rectangle"
    )


def stdft_gradient_input_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-GradInput attribution method.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        target: Optional target class
        device: Device to run on

    Returns:
        The full tuple from compute_enhanced_dft_for_xai_method for time-frequency domain flipping
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
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shift=64,
        window_shape="rectangle"
    )


def stdft_smoothgrad_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-SmoothGrad attribution method.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        target: Optional target class
        device: Device to run on

    Returns:
        The full tuple from compute_enhanced_dft_for_xai_method for time-frequency domain flipping
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
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        window_width=128,
        window_shift=64,
        window_shape="rectangle",
        num_samples=20,
        noise_level=0.2
    )


def stdft_occlusion_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for STDFT-Occlusion attribution method.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        target: Optional target class
        device: Device to run on

    Returns:
        The full tuple from compute_enhanced_dft_for_xai_method for time-frequency domain flipping
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
            signal_length=2000,
            leverage_symmetry=True,
            precision=32,
            sampling_rate=400,
            window_width=128,
            window_shift=64,
            window_shape="rectangle",
            occlusion_type="zero",
            window_size=40
        )
    except Exception as e:
        print(f"Error in STDFT-Occlusion: {e}")
        # Create empty placeholder results with correct shapes
        n_channels, time_steps = sample.shape
        freq_length = time_steps // 2 + 1  # For leverage_symmetry=True
        n_frames = (time_steps - 128) // 64 + 1  # Based on window_width and window_shift

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
    Main function to run time-frequency domain window flipping evaluation.
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
    n_samples_per_class = 5  # Reduced for time-frequency analysis since it's more computationally intensive
    n_steps = 10
    window_size = 10
    sampling_rate = 400

    # STFT parameters
    window_width = 128
    window_shift = 64  # 50% overlap
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

    # Visualize time-frequency window flipping for samples of both classes
    print("\n===== Visualizing Time-Frequency Domain Window Flipping for Both Classes =====")
    visualize_both_classes_timefreq(
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
    print("\n===== Evaluating with most important time-frequency windows flipped first =====")
    results_most_first, agg_results_most_first = run_class_specific_timefreq_window_flipping_evaluation(
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
    print("\n===== Evaluating with least important time-frequency windows flipped first =====")
    results_least_first, agg_results_least_first = run_class_specific_timefreq_window_flipping_evaluation(
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
    print("\n===== Time-Frequency Domain Faithfulness Ratios =====")
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

    # Extract important time-frequency windows using STFT-LRP
    print("\n===== Extracting Important Time-Frequency Windows =====")
    timefreq_windows = collect_important_timefreq_windows(
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

    # Visualize and analyze important time-frequency windows
    print("\n===== Analyzing Important Time-Frequency Windows =====")
    visualize_timefreq_windows(timefreq_windows, output_dir=f"{output_dir}/windows")
    timefreq_window_stats = analyze_timefreq_windows(timefreq_windows, output_dir=f"{output_dir}/windows")

    # Print top discriminative features
    if timefreq_window_stats is not None and not timefreq_window_stats.empty:
        print("\n===== Top Discriminative Time-Frequency Features =====")
        top_features = timefreq_window_stats.head(5)
        print(top_features[["feature", "p_value", "normal_mean", "faulty_mean", "difference_pct"]])

    print("\nTime-Frequency domain window flipping evaluation complete!")


if __name__ == "__main__":
    main()