import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd
import os
from datetime import datetime
from scipy import stats
import random
import h5py
from pathlib import Path

# Import your model loading utility
from utils.baseline_xai import load_model


# Time window flipping implementation
def time_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                window_size=10, most_relevant_first=True,
                                reference_value="mild_noise",
                                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on a single time series sample.
    Implements best practices for normalized data reference values.
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    '''
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()
        '''
    # In time_window_flipping_single and frequency_window_flipping_single:

    # Get original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        # Use the provided true class if given, otherwise fall back to predicted class
        if target_class is None:
            # Only use predicted class if no target was provided
            target_class = torch.argmax(original_prob).item()
            print("Warning: No target class provided, using predicted class")

        # Get confidence for the target class (true class in our implementation)
        original_score = original_prob[target_class].item()

        # Log what we're doing for verification
        predicted_class = torch.argmax(original_prob).item()
        if predicted_class != target_class:
            print(f"Note: Model prediction ({predicted_class}) differs from target class ({target_class})")
            print(f"Tracking confidence for target class: {target_class}")

    # Compute attributions
    attributions = attribution_method(model, sample)

    # If attributions is a tuple (as in some LRP methods), take the first element
    if isinstance(attributions, tuple):
        attributions = attributions[0]

    # Ensure attributions is a tensor
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, device=device)

    # Move to CPU for numpy operations
    attributions_np = attributions.detach().cpu().numpy()

    # Set reference value for flipping based on specified strategy
    if reference_value is None or reference_value == "extreme":
        # Use extreme values (recommended for normalized data)
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_mean = channel_data.mean().item()
            data_std = channel_data.std().item()
            # Use value that's 10 standard deviations away from mean
            reference_value[c] = torch.full_like(channel_data, data_mean + 10 * data_std)
    elif reference_value == "shift":
        # Use a constant shift (3 standard deviations)
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_mean = channel_data.mean().item()
            data_std = channel_data.std().item()
            # Shift by 3 standard deviations in the opposite direction from the mean
            shift_direction = -1 if data_mean >= 0 else 1
            reference_value[c] = torch.full_like(channel_data, data_mean + (shift_direction * 3 * data_std))
    elif reference_value == "additive_noise":
        # Use the original signal plus noise
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Add noise to the original signal (noise level = 1x std)
            reference_value[c] = channel_data + torch.randn_like(channel_data) * (data_std * 0.60)
    elif reference_value == "noise":
        # Use high-variance random noise
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Generate random noise with high variance
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 10)
    elif reference_value == "zero":
        # Explicitly use zeros as reference values
        reference_value = torch.zeros_like(sample)
    elif reference_value == "mild_noise":
        # Use milder noise (less variance than full noise)
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Use smaller noise multiplier (0.5 instead of 10.0)
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 0.50)
    elif reference_value == "invert":
        # Invert the signal (flip around the mean)
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_mean = channel_data.mean().item()
            # Invert around the mean: mean + (mean - value)
            reference_value[c] = 2 * data_mean - channel_data
    elif reference_value == "minmax":
        # Use channel-specific opposite extremes
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            min_val = channel_data.min().item()
            max_val = channel_data.max().item()
            data_range = max_val - min_val
            # Use extreme values beyond min/max
            reference_value[c] = torch.full_like(channel_data, max_val + data_range)
    elif isinstance(reference_value, (int, float)):
        # Use specified constant value
        reference_value = torch.full_like(sample, reference_value)

    # Calculate number of windows
    n_windows = time_steps // window_size
    if time_steps % window_size > 0:
        n_windows += 1

    # Calculate window importance by averaging relevance within each window
    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            # Average absolute attribution within the window
            if attributions_np.shape[0] == n_channels:  # If attribution has channel dimension
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[channel, start_idx:end_idx])
                )
            else:  # If attribution is channel-agnostic
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[start_idx:end_idx])
                )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get flipped sample
        flipped_sample = sample.clone()

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            flipped_sample[channel_idx, start_idx:end_idx] = reference_value[channel_idx, start_idx:end_idx]

        # Get model output
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    return scores, flipped_pcts


def time_window_flipping_single_class_specific(model, sample, attribution_method, target_class=None, n_steps=20,
                                               window_size=10, most_relevant_first=True,
                                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis with class-specific reference values.
    Uses mild_noise for good class (0) and zero for bad class (1).

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        target_class: Target class to track
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        device: Device to run computations on

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # Compute attributions
    attributions = attribution_method(model, sample)

    # If attributions is a tuple (as in some LRP methods), take the first element
    if isinstance(attributions, tuple):
        attributions = attributions[0]

    # Ensure attributions is a tensor
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, device=device)

    # Move to CPU for numpy operations
    attributions_np = attributions.detach().cpu().numpy()

    # Set class-specific reference value based on target class
    if target_class == 0:  # Good/Normal class
        # Use mild noise as reference for normal samples
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Add mild noise to the signal
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 0.5)

        print(f"Using mild_noise reference for normal class sample")
    else:  # Bad/Faulty class (class 1)
        # Use zeros as reference for faulty samples
        reference_value = torch.zeros_like(sample)

        print(f"Using zero reference for faulty class sample")

    # Calculate window importance by averaging relevance within each window
    n_windows = time_steps // window_size
    if time_steps % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            # Average absolute attribution within the window
            if attributions_np.shape[0] == n_channels:  # If attribution has channel dimension
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[channel, start_idx:end_idx])
                )
            else:  # If attribution is channel-agnostic
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[start_idx:end_idx])
                )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get flipped sample
        flipped_sample = sample.clone()

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            flipped_sample[channel_idx, start_idx:end_idx] = reference_value[channel_idx, start_idx:end_idx]

        # Get model output
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    return scores, flipped_pcts
def time_window_flipping_batch(model, samples, attribution_methods, n_steps=10,
                               window_size=40, most_relevant_first=True,
                               reference_value="mild_noise", max_samples=None,
                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on a batch of time series samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
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
                # Pass the true class label to track
                scores, flipped_pcts = time_window_flipping_single(
                    model=model,
                    sample=sample,
                    attribution_method=lambda m, s: attribution_func(m, s, target_class),
                    target_class=target_class,  # Pass the true class label
                    n_steps=n_steps,
                    window_size=window_size,
                    most_relevant_first=most_relevant_first,
                    reference_value=reference_value,
                    device=device
                )

                # Store results
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": scores,
                    "flipped_pcts": flipped_pcts,
                    "auc": np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
                })

            except Exception as e:
                print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                # Store empty result to maintain sample count consistency
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results


def time_window_flipping_batch_class_specific(model, samples, attribution_methods, n_steps=10,
                                              window_size=40, most_relevant_first=True,
                                              max_samples=None,
                                              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on a batch of time series samples using class-specific reference values.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        max_samples: Maximum number of samples to process (None = all samples)
        device: Device to run computations on

    Returns:
        results: Dictionary with aggregated results for each method
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
                # Compute scores for this sample using the current method with class-specific references
                scores, flipped_pcts = time_window_flipping_single_class_specific(
                    model=model,
                    sample=sample,
                    attribution_method=lambda m, s: attribution_func(m, s, target_class),
                    target_class=target_class,  # Pass the true class label
                    n_steps=n_steps,
                    window_size=window_size,
                    most_relevant_first=most_relevant_first,
                    device=device
                )

                # Store results
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": scores,
                    "flipped_pcts": flipped_pcts,
                    "auc": np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
                })

            except Exception as e:
                print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
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

        # Collect all AUC values
        valid_samples = [s for s in samples if s["scores"] is not None]

        if not valid_samples:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = method_results
            continue

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

            # Get flipped percentages from common x-axis
            method_results["flipped_pcts"] = common_x
            method_results["avg_scores"] = avg_scores

            # Extract AUC values and calculate mean/std
            method_results["auc_values"] = [s["auc"] for s in valid_samples if not np.isnan(s["auc"])]
            method_results["mean_auc"] = np.mean(method_results["auc_values"])
            method_results["std_auc"] = np.std(method_results["auc_values"])

            # Store class-specific metrics
            method_results["class_specific"] = class_specific

        except Exception as e:
            print(f"Error aggregating results for {method_name}: {e}")
            method_results["flipped_pcts"] = None
            method_results["avg_scores"] = None
            method_results["auc_values"] = [s["auc"] for s in valid_samples if not np.isnan(s["auc"])]
            method_results["mean_auc"] = np.mean(method_results["auc_values"]) if method_results["auc_values"] else None
            method_results["std_auc"] = np.std(method_results["auc_values"]) if method_results["auc_values"] else None

        # Store results for this method
        agg_results[method_name] = method_results

    return agg_results


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value='mild_noise'):
    """
    Plot aggregated window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

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
        f'Aggregate Window Flipping Results\n(Flipping {flip_order} windows first, \n Reference Value Method: {reference_value})',
        fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    return fig


def plot_class_specific_results(agg_results, most_relevant_first=True, reference_value='mild_noise'):
    """
    Plot class-specific window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

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

        # If no class-specific data, just plot the overall result
        if not results["class_specific"] or all(cls not in results["class_specific"] for cls in ["class_0", "class_1"]):
            if results["avg_scores"] is not None:
                ax.plot(results["flipped_pcts"], results["avg_scores"],
                        label=f"Overall (AUC: {results['mean_auc']:.4f})")

        ax.set_title(f'{method_name}')
        ax.set_xlabel('Percentage of Windows Flipped (%)')
        ax.set_ylabel('Prediction Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.legend()

    flip_order = "most important" if most_relevant_first else "least important"
    fig.suptitle(
        f'Class-Specific Window Flipping Results\n(Flipping {flip_order} windows first, \n Reference Value Method: {reference_value})',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    return fig


def plot_auc_by_class(agg_results, most_relevant_first=True, reference_value='mild_noise'):
    """
    Plot AUC values separated by class.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

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
                 f'(Flipping {flip_order} windows first, \n Reference Value Method: {reference_value})')
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


def extract_important_time_windows(model, sample, attribution_method, target_class=None,
                                   n_windows=10, window_size=10, sampling_rate=400,
                                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extract the most important windows from time domain based on attribution scores.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions
        target_class: Target class for explanation (default: model's prediction)
        n_windows: Number of top windows to extract
        window_size: Size of time windows
        sampling_rate: Sampling rate in Hz
        device: Device to run computations on

    Returns:
        important_windows: List of dictionaries containing window info
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

    # Get attribution scores
    attributions = attribution_method(model, sample, target_class)

    # If attributions is a tuple, take the first element
    if isinstance(attributions, tuple):
        attributions = attributions[0]

    # Ensure attributions is a tensor
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, device=device)

    # Convert to numpy
    attributions_np = attributions.detach().cpu().numpy()
    input_signal = sample.detach().cpu().numpy()

    # Calculate number of windows
    n_windows_per_channel = time_steps // window_size
    if time_steps % window_size > 0:
        n_windows_per_channel += 1

    # Calculate window importance
    window_importance = np.zeros((n_channels, n_windows_per_channel))

    for channel in range(n_channels):
        for window_idx in range(n_windows_per_channel):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            # Use absolute values of relevance for importance
            if attributions_np.shape[0] == n_channels:
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[channel, start_idx:end_idx])
                )
            else:
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[start_idx:end_idx])
                )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)[::-1]  # Descending order

    # Take top n_windows indices
    top_indices = sorted_indices[:n_windows]

    # Convert flat indices to channel, window indices
    channel_indices = top_indices // n_windows_per_channel
    window_indices = top_indices % n_windows_per_channel

    # Extract important windows
    important_windows = []

    for i in range(len(top_indices)):
        channel_idx = channel_indices[i]
        window_idx = window_indices[i]

        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, time_steps)

        # Extract time domain data
        time_data = input_signal[channel_idx, start_idx:end_idx]

        window_info = {
            'time_data': time_data,
            'time_start': start_idx,
            'time_end': end_idx,
            'channel': channel_idx,
            'relevance': flat_importance[top_indices[i]],
            'window_idx': window_idx,
            'avg_amplitude': np.mean(np.abs(time_data)),
            'max_amplitude': np.max(np.abs(time_data)),
            'std_amplitude': np.std(time_data),
            'skewness': stats.skew(time_data) if len(time_data) > 2 else 0,
            'kurtosis': stats.kurtosis(time_data) if len(time_data) > 2 else 0,
            'window_size': end_idx - start_idx
        }

        # Add spectral features for this window
        if len(time_data) > 1:
            # Compute FFT for this window
            window_fft = np.fft.rfft(time_data)
            window_fft_magnitude = np.abs(window_fft)

            # Find peak frequency
            if len(window_fft_magnitude) > 0:
                peak_idx = np.argmax(window_fft_magnitude)
                window_info['peak_freq'] = peak_idx * (sampling_rate / 2) / len(window_fft_magnitude)
                window_info['spectral_energy'] = np.sum(window_fft_magnitude ** 2)

        important_windows.append(window_info)

    return important_windows


def collect_important_time_windows(model, samples, attribution_method,
                                   n_samples_per_class=10, n_windows=10, window_size=10,
                                   sampling_rate=400,
                                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Collect important time windows from multiple samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions
        n_samples_per_class: Number of samples to process per class
        n_windows: Number of top windows to extract per sample
        window_size: Size of time windows
        sampling_rate: Sampling rate in Hz
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
            windows = extract_important_time_windows(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                sampling_rate=sampling_rate,
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

    # Process class 1 samples
    n_class_1 = min(n_samples_per_class, len(class_1_samples))
    for i in range(n_class_1):
        sample, target = class_1_samples[i]

        try:
            windows = extract_important_time_windows(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                sampling_rate=sampling_rate,
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

    return all_windows


def visualize_time_windows(windows, output_dir="./results/time_windows"):
    """
    Visualize the important time windows.

    Args:
        windows: List of window dictionaries from extract_important_time_windows
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group windows by class
    normal_windows = [w for w in windows if w['class'] == 0]
    faulty_windows = [w for w in windows if w['class'] == 1]

    # Create a scatter plot of time position vs. amplitude, colored by relevance
    plt.figure(figsize=(12, 10))

    # Plot normal windows
    plt.subplot(2, 1, 1)
    relevances = [w['relevance'] for w in normal_windows]
    time_starts = [w['time_start'] for w in normal_windows]
    amplitudes = [w['avg_amplitude'] for w in normal_windows]

    plt.scatter(time_starts, amplitudes, c=relevances, cmap='viridis',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Time Windows - Normal Samples')
    plt.xlabel('Time Position')
    plt.ylabel('Average Amplitude')
    plt.grid(alpha=0.3)

    # Plot faulty windows
    plt.subplot(2, 1, 2)
    relevances = [w['relevance'] for w in faulty_windows]
    time_starts = [w['time_start'] for w in faulty_windows]
    amplitudes = [w['avg_amplitude'] for w in faulty_windows]

    plt.scatter(time_starts, amplitudes, c=relevances, cmap='plasma',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Time Windows - Faulty Samples')
    plt.xlabel('Time Position')
    plt.ylabel('Average Amplitude')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_windows_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create histograms of window positions
    plt.figure(figsize=(12, 6))

    plt.hist([w['time_start'] for w in normal_windows], bins=20, alpha=0.5,
             label='Normal', density=True)
    plt.hist([w['time_start'] for w in faulty_windows], bins=20, alpha=0.5,
             label='Faulty', density=True)

    plt.title('Distribution of Important Window Positions')
    plt.xlabel('Time Position')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{output_dir}/time_position_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create channel distribution plot
    plt.figure(figsize=(10, 6))

    # Count windows by channel for each class
    channels = sorted(set(w['channel'] for w in windows))

    normal_counts = [len([w for w in normal_windows if w['channel'] == c]) for c in channels]
    faulty_counts = [len([w for w in faulty_windows if w['channel'] == c]) for c in channels]

    x = np.arange(len(channels))
    width = 0.35

    plt.bar(x - width / 2, normal_counts, width, label='Normal')
    plt.bar(x + width / 2, faulty_counts, width, label='Faulty')

    plt.xlabel('Channel')
    plt.ylabel('Count')
    plt.title('Channel Distribution of Important Time Windows')
    plt.xticks(x, [f'Channel {c}' for c in channels])
    plt.legend()

    plt.savefig(f"{output_dir}/channel_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Display example windows
    plt.figure(figsize=(15, 10))

    # Plot a subset of windows
    n_to_plot = min(5, len(normal_windows), len(faulty_windows))

    for i in range(n_to_plot):
        # Normal window
        plt.subplot(2, n_to_plot, i + 1)
        if i < len(normal_windows):
            window = normal_windows[i]
            plt.plot(window['time_data'])
            plt.title(f"Normal #{i + 1}\nRel: {window['relevance']:.4f}")

        # Faulty window
        plt.subplot(2, n_to_plot, n_to_plot + i + 1)
        if i < len(faulty_windows):
            window = faulty_windows[i]
            plt.plot(window['time_data'])
            plt.title(f"Faulty #{i + 1}\nRel: {window['relevance']:.4f}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/example_windows.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_time_windows(windows, output_dir="./results/time_windows"):
    """
    Analyze the important time windows and identify distinguishing features.

    Args:
        windows: List of window dictionaries
        output_dir: Directory to save results

    Returns:
        stats_df: DataFrame with feature statistics
    """
    import pandas as pd
    from scipy import stats
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{k: v for k, v in w.items() if not isinstance(v, np.ndarray)}
                       for w in windows])

    # Calculate additional features
    if 'time_data' in windows[0]:
        for i, window in enumerate(windows):
            time_data = window['time_data']
            if len(time_data) > 1:
                # Zero crossing rate
                zero_crossings = np.where(np.diff(np.signbit(time_data)))[0].size
                df.loc[i, 'zero_crossing_rate'] = zero_crossings / len(time_data)

                # Energy
                df.loc[i, 'energy'] = np.sum(time_data ** 2)

                # Peak to average ratio
                if np.mean(np.abs(time_data)) > 0:
                    df.loc[i, 'peak_to_avg'] = np.max(np.abs(time_data)) / np.mean(np.abs(time_data))

                # Spectral centroid
                fft = np.fft.rfft(time_data)
                if np.sum(np.abs(fft)) > 0:
                    freqs = np.fft.rfftfreq(len(time_data), d=1.0 / 400)  # Assuming 400Hz sampling rate
                    spectral_centroid = np.sum(freqs * np.abs(fft)) / np.sum(np.abs(fft))
                    df.loc[i, 'spectral_centroid'] = spectral_centroid

    # Analyze class differences
    features = ['relevance', 'avg_amplitude', 'max_amplitude', 'std_amplitude',
                'skewness', 'kurtosis', 'peak_freq', 'spectral_energy',
                'zero_crossing_rate', 'energy', 'peak_to_avg', 'spectral_centroid']

    # Filter out features not in the DataFrame
    features = [f for f in features if f in df.columns]

    class_stats = []
    for feature in features:
        if feature not in df.columns:
            continue

        normal_values = df[df['class'] == 0][feature]
        faulty_values = df[df['class'] == 1][feature]

        # Filter out NaN values
        normal_values = normal_values.dropna()
        faulty_values = faulty_values.dropna()

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
        else:
            print(f"Not enough data for feature {feature}")

    # Sort by statistical significance
    class_stats.sort(key=lambda x: x['p_value'])
    class_stats_df = pd.DataFrame(class_stats)

    # Save results
    class_stats_df.to_csv(f"{output_dir}/time_feature_stats.csv", index=False)
    df.to_csv(f"{output_dir}/time_windows_data.csv", index=False)

    # Create summary visualizations
    plt.close('all')
    plt.figure(figsize=(12, 8))

    # Plot top 5 most discriminative features
    top_features = class_stats_df.head(5)['feature'].tolist()

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


def run_time_window_flipping_evaluation(model, samples, attribution_methods,
                                        n_steps=10, window_size=40,
                                        most_relevant_first=True, reference_value="mild_noise",
                                        max_samples=None, output_dir="./results",
                                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete window flipping evaluation on test set and save results.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
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
    filename_prefix = f"{output_dir}/time_window_flipping_{flip_order}_{timestamp}"

    print(f"Starting time domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}")

    # Run window flipping on all samples
    results = time_window_flipping_batch(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
        max_samples=max_samples,
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
    print("\nTime Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


def run_class_specific_window_flipping_evaluation(model, samples, attribution_methods,
                                                  n_steps=10, window_size=40,
                                                  most_relevant_first=True, max_samples=None,
                                                  output_dir="./results",
                                                  device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete window flipping evaluation with class-specific reference values.
    Uses mild_noise for good/normal class and zeros for bad/faulty class.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        max_samples: Maximum number of samples to process (None = all)
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
    filename_prefix = f"{output_dir}/time_window_flipping_class_specific_{flip_order}_{timestamp}"

    print(f"Starting class-specific time domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference values: mild_noise for normal class, zero for faulty class")

    # Run window flipping on all samples
    results = time_window_flipping_batch_class_specific(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        max_samples=max_samples,
        device=device
    )

    print("Computing aggregated results...")

    # Aggregate results
    agg_results = aggregate_results(results)

    print("Creating visualizations...")

    # Plot and save aggregated results
    plt.close('all')
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value='class_specific')
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first, reference_value='class_specific')
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Plot and save AUC by class
    plt.close('all')
    auc_fig = plot_auc_by_class(agg_results, most_relevant_first, reference_value='class_specific')
    if auc_fig:
        auc_fig.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
        plt.close(auc_fig)

    # Save numerical results to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nClass-Specific Time Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results
def visualize_window_flipping_sample(model, sample, target, attribution_method,
                                     n_steps=5, window_size=40, most_relevant_first=True,
                                     reference_value="mild_noise", save_path=None,
                                     device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of window flipping for a single time series sample.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        target: Target label for the sample
        attribution_method: Function that generates attributions for the input
        n_steps: Number of steps to visualize
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        save_path: Optional path to save the figure
        device: Device to run computations on

    Returns:
        fig: Matplotlib figure
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute attributions
    attributions = attribution_method(model, sample, target)

    # If attributions is a tuple (as in some LRP methods), take the first element
    if isinstance(attributions, tuple):
        attributions = attributions[0]

    # Ensure attributions is a tensor
    if not isinstance(attributions, torch.Tensor):
        attributions = torch.tensor(attributions, device=device)

    # Move to CPU for numpy operations
    attributions_np = attributions.detach().cpu().numpy()
    sample_np = sample.detach().cpu().numpy()

    # Set reference value for flipping
    if reference_value == "additive_noise":
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Add noise to the original signal (noise level = 0.6x std)
            reference_value[c] = channel_data + torch.randn_like(channel_data) * (data_std * 0.60)
        reference_np = reference_value.detach().cpu().numpy()

    elif reference_value == "zero":
        reference_np = np.zeros_like(sample_np)

    elif reference_value == "mild_noise":
        # Use milder noise (less variance than full noise)
        reference_value = torch.zeros_like(sample)
        for c in range(n_channels):
            channel_data = sample[c]
            data_std = channel_data.std().item()
            # Use smaller noise multiplier (0.5 instead of 10.0)
            reference_value[c] = torch.randn_like(channel_data) * (data_std * 0.50)
        reference_np = reference_value.detach().cpu().numpy()

    else:
        # Default to zero
        reference_np = np.zeros_like(sample_np)

    # Calculate number of windows
    n_windows = time_steps // window_size
    if time_steps % window_size > 0:
        n_windows += 1

    # Calculate window importance by averaging relevance within each window
    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            # Average absolute attribution within the window
            if attributions_np.shape[0] == n_channels:
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[channel, start_idx:end_idx])
                )
            else:
                window_importance[channel, window_idx] = np.mean(
                    np.abs(attributions_np[start_idx:end_idx])
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
        target_class = target
        original_score = original_prob[target_class].item()

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Store flipped samples and scores
    flipped_samples = [sample_np.copy()]
    scores = [original_score]
    flipped_windows_count = [0]
    flipped_pcts = [0.0]
    flipped_windows_masks = [np.zeros((n_channels, time_steps), dtype=bool)]

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get flipped sample
        flipped_sample = sample_np.copy()
        flipped_mask = np.zeros((n_channels, time_steps), dtype=bool)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, time_steps)

            flipped_sample[channel_idx, start_idx:end_idx] = reference_np[channel_idx, start_idx:end_idx]
            flipped_mask[channel_idx, start_idx:end_idx] = True

        # Get model output
        flipped_tensor = torch.tensor(flipped_sample, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Store results
        flipped_samples.append(flipped_sample)
        scores.append(score)
        flipped_windows_count.append(n_windows_to_flip)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        flipped_windows_masks.append(flipped_mask)

    # Create visualization
    plt.close('all')
    fig = plt.figure(figsize=(16, 4 * n_channels))

    # Calculate optimal layout
    n_plots = n_steps + 1
    n_cols = min(6, n_plots)  # Max 6 columns
    n_rows = (n_plots + n_cols - 1) // n_cols * n_channels

    # Set title
    fig.suptitle(f'Window Flipping Progression - {"Most" if most_relevant_first else "Least"} Important First\n' +
                 f'Sample Class: {"Good" if target == 0 else "Bad"}, Reference: {reference_value}',
                 fontsize=16)

    # Create grid spec to organize subplots with proper spacing
    from matplotlib import gridspec
    gs = gridspec.GridSpec(n_channels, n_plots, figure=fig)

    # Define channel names
    channel_names = ['X-axis', 'Y-axis', 'Z-axis']

    # Plot each channel and step
    for channel in range(n_channels):
        for step in range(n_plots):
            # Create axis in the grid
            ax = fig.add_subplot(gs[channel, step])

            # Plot the signal
            ax.plot(flipped_samples[step][channel], 'b-', linewidth=1.0)

            # If not first step, add colored background for flipped windows
            if step > 0:
                # Plot shaded areas for flipped windows
                for i in range(time_steps):
                    if flipped_windows_masks[step][channel, i]:
                        ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.5)

            # Set titles and labels
            if step == 0:
                ax.set_ylabel(f'{channel_names[channel]}\nAmplitude', fontsize=10)
                title = f'Original'
            else:
                title = f'{flipped_pcts[step]:.1f}% Flipped'

            # Add score to title
            score_color = 'green' if scores[step] > 0.5 else 'red'
            title += f'\nScore: {scores[step]:.3f}'

            ax.set_title(title, fontsize=10)

            # Only show x-axis ticks for bottom row
            if channel < n_channels - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time', fontsize=8)

            # Set axis limits consistently for all plots
            ymin = np.min(sample_np[channel]) - 0.1
            ymax = np.max(sample_np[channel]) + 0.1
            ax.set_ylim([ymin, ymax])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_both_classes(model, samples, attribution_method, n_steps=5, window_size=40,
                           most_relevant_first=True, output_dir="./results",
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize window flipping for one good sample and one bad sample.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions
        n_steps: Number of steps to visualize
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        output_dir: Directory to save visualizations
        device: Device to run on
    """
    # Find one sample of each class
    good_sample = None
    bad_sample = None

    # Loop through samples to find one of each class
    for sample_data, target in samples:
        if target == 0 and good_sample is None:  # Good/normal class
            good_sample = (sample_data, target)
            print("Found good sample")
        elif target == 1 and bad_sample is None:  # Bad/faulty class
            bad_sample = (sample_data, target)
            print("Found bad sample")

        # Break if we have one of each
        if good_sample is not None and bad_sample is not None:
            break

    # Visualize good sample with class-specific reference (mild_noise)
    if good_sample is not None:
        good_sample_data, good_target = good_sample
        print("\n===== Visualizing Good/Normal Sample (Class 0) =====")
        print("Using mild_noise as reference value")

        plt.close('all')
        good_fig = visualize_window_flipping_sample(
            model=model,
            sample=good_sample_data,
            target=good_target,
            attribution_method=attribution_method,
            n_steps=n_steps,
            window_size=window_size,
            most_relevant_first=most_relevant_first,
            reference_value="mild_noise",
            save_path=f"{output_dir}/window_flipping_good_sample.png",
            device=device
        )
        plt.close(good_fig)
    else:
        print("No good/normal sample found!")

    # Visualize bad sample with class-specific reference (zero)
    if bad_sample is not None:
        bad_sample_data, bad_target = bad_sample
        print("\n===== Visualizing Bad/Faulty Sample (Class 1) =====")
        print("Using zero as reference value")

        plt.close('all')
        bad_fig = visualize_window_flipping_sample(
            model=model,
            sample=bad_sample_data,
            target=bad_target,
            attribution_method=attribution_method,
            n_steps=n_steps,
            window_size=window_size,
            most_relevant_first=most_relevant_first,
            reference_value="zero",
            save_path=f"{output_dir}/window_flipping_bad_sample.png",
            device=device
        )
        plt.close(bad_fig)
    else:
        print("No bad/faulty sample found!")

def main():
    """
    Main function to run time domain window flipping evaluation.
    Also includes feature extraction from important windows.
    """
    # Clean up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "results-freq-temp/results/time_domain"
    n_samples_per_class = 500  # Number of samples per class
    n_steps = 10
    window_size = 40
    sampling_rate = 400

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Import your attribution methods
    from utils.xai_implementation import compute_lrp_relevance
    from utils.baseline_xai import grad_times_input_relevance, smoothgrad_relevance, occlusion_simpler_relevance

    # Set up wrapper functions for attribution methods
    def lrp_wrapper(model, sample, target=None):
        attributions, _, _ = compute_lrp_relevance(
            model=model, sample=sample, label=target, device=device
        )
        return attributions

    def grad_input_wrapper(model, sample, target=None):
        attributions, _ = grad_times_input_relevance(
            model=model, x=sample, target=target
        )
        return attributions

    def smoothgrad_wrapper(model, sample, target=None):
        attributions, _ = smoothgrad_relevance(
            model=model, x=sample, num_samples=40, noise_level=1, target=target
        )
        return attributions

    def occlusion_wrapper(model, sample, target=None):
        attributions, _ = occlusion_simpler_relevance(
            model=model, x=sample, target=target, occlusion_type="zero", window_size=40
        )
        return attributions

    # Set up attribution methods dictionary
    attribution_methods = {
        "LRP": lrp_wrapper,
        "Gradient*Input": grad_input_wrapper,
        "SmoothGrad": smoothgrad_wrapper,
        "Occlusion": occlusion_wrapper
    }

    # Load balanced samples
    samples = load_balanced_samples(data_dir, n_samples_per_class)

    # Visualize with LRP
    plt.close('all')

    # Visualize window flipping for one good and one bad sample
    print("\n===== Visualizing Time Domain Window Flipping for Both Classes =====")
    vis_fig = visualize_both_classes(
        model=model,
        samples=samples,
        attribution_method=lrp_wrapper,
        n_steps=5,
        window_size=window_size,
        most_relevant_first=True,
        output_dir=output_dir,
        device=device
    )

    plt.close(vis_fig)


    # Evaluate with most important windows flipped first
    print("\n===== Evaluating with most important time windows flipped first =====")
    results_most_first, agg_results_most_first = run_class_specific_window_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        max_samples=len(samples),  # Use all loaded samples
        output_dir=output_dir,
        device=device
    )

    # Evaluate with least important windows flipped first
    print("\n===== Evaluating with least important time windows flipped first =====")
    results_least_first, agg_results_least_first = run_class_specific_window_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        max_samples=len(samples),  # Use all loaded samples
        output_dir=output_dir,
        device=device
    )

    # Calculate and print faithfulness ratios
    print("\n===== Time Domain Faithfulness Ratios =====")
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

    # Extract important windows using LRP
    print("\n===== Extracting Important Time Windows =====")
    time_windows = collect_important_time_windows(
        model=model,
        samples=samples,
        attribution_method=lrp_wrapper,
        n_samples_per_class=min(200, n_samples_per_class),  # Use fewer samples for feature extraction
        n_windows=10,  # Extract 10 top windows per sample
        window_size=window_size,
        sampling_rate=sampling_rate,
        device=device
    )

    # Visualize and analyze important windows
    print("\n===== Analyzing Important Time Windows =====")
    visualize_time_windows(time_windows, output_dir=f"{output_dir}/windows")
    window_stats = analyze_time_windows(time_windows, output_dir=f"{output_dir}/windows")

    # Print top discriminative features
    print("\n===== Top Discriminative Time Features =====")
    top_features = window_stats.head(5)
    print(top_features[["feature", "p_value", "normal_mean", "faulty_mean", "difference_pct"]])

    print("\nTime domain window flipping evaluation complete!")


if __name__ == "__main__":
    main()


