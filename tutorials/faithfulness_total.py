import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd
from torch.utils.data import DataLoader
import os
from datetime import datetime

# Import your model and data loading utilities
from Classification.cnn1D_model import CNN1D_Wide
from torch.utils.data import DataLoader
from utils.baseline_xai import load_model
def time_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                window_size=10, most_relevant_first=True,
                                reference_value="additive_noise",
                                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on a single time series sample.
    Implements best practices for normalized data reference values.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        target_class: Target class to track (if None, use predicted class)
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows. Options:
                         - None/extreme: Use extreme values (default, recommended for normalized data)
                         - noise: Use high-variance random noise
                         - minmax: Use channel-specific opposite extremes
                         - A specific float value
        device: Device to run computations on

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        # If target_class is not provided, use the predicted class
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

    # New reference value option for time_window_flipping_single
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
            # Use smaller noise multiplier (1.0 instead of 10.0)
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

    # Rest of the function remains the same...
    # [code for calculating window importance, flipping windows, etc.]

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

            # Track probability for the target class
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    return scores, flipped_pcts
def time_window_flipping_batch(model, test_loader, attribution_methods, n_steps=10,
                               window_size=40, most_relevant_first=True,
                               reference_value="additive_noise", max_samples=None,
                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on a batch of time series samples and aggregate results0.
    Modified to pass the true target class to time_window_flipping_single.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows (default=None, which uses zeros)
        max_samples: Maximum number of samples to process (None = all samples)
        device: Device to run computations on

    Returns:
        results0: Dictionary with aggregated results0 for each method
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
                    # Pass the true class label to track
                    scores, flipped_pcts = time_window_flipping_single(
                        model=model,
                        sample=sample,
                        attribution_method=lambda m, s: attribution_func(m, s, target),
                        target_class=target.item(),  # Pass the true class label
                        n_steps=n_steps,
                        window_size=window_size,
                        most_relevant_first=most_relevant_first,
                        reference_value=reference_value,
                        device=device
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


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value = 'additive_noise'):
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
    plt.title(f'Aggregate Window Flipping Results\n(Flipping {flip_order} windows first, \n Reference Value Method Flipped with: {reference_value})', fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def plot_auc_distribution(agg_results, most_relevant_first=True, reference_value='additive_noise'):
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
    plt.title(f'AUC Distribution Across Samples\n(Flipping {flip_order} windows first), \n Reference Value Method Flipped with: {reference_value})', fontsize=16)
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


def run_window_flipping_evaluation(model, test_loader, attribution_methods,
                                   n_steps=10, window_size=40,
                                   most_relevant_first=True, max_samples=None,
                                   output_dir="./results0",
                                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete window flipping evaluation on test set and save results0.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
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
    filename_prefix = f"{output_dir}/window_flipping_{flip_order}_{timestamp}"

    print(f"Starting window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")

    # Run window flipping on all samples
    results = time_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value="additive_noise",
        max_samples=max_samples,
        device=device
    )

    print("Computing aggregated results0...")

    # Aggregate results0
    agg_results = aggregate_results(results)

    print("Plotting results0...")

    # Plot and save aggregated results0
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value='additive_noise')
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')

    # Plot and save AUC distributions
    dist_fig = plot_auc_distribution(agg_results, most_relevant_first)
    dist_fig.savefig(f"{filename_prefix}_auc_distribution.png", dpi=300, bbox_inches='tight')

    # Save numerical results0 to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nWindow Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results





# Import attribution method wrappers
def lrp_attribution_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for LRP attribution with explicit target parameter.

    Args:
        model: Trained PyTorch model
        sample: Input sample
        target: Target class for explanation (default: model's prediction)
        device: Device to run computations on

    Returns:
        attributions: LRP relevance scores
    """
    from utils.xai_implementation import compute_lrp_relevance

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute LRP relevance
    attributions, _, _ = compute_lrp_relevance(
        model=model,
        sample=sample,
        label=target,  # Pass target as label
        device=device
    )

    return attributions


def grad_times_input_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for Gradient*Input attribution with explicit target parameter.

    Args:
        model: Trained PyTorch model
        sample: Input sample
        target: Target class for explanation (default: model's prediction)
        device: Device to run computations on

    Returns:
        attributions: Gradient*Input relevance scores
    """
    from utils.baseline_xai import grad_times_input_relevance

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute Gradient*Input relevance
    attributions, _ = grad_times_input_relevance(
        model=model,
        x=sample,
        target=target  # Pass the target
    )

    return attributions


def smoothgrad_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for SmoothGrad attribution with explicit target parameter.

    Args:
        model: Trained PyTorch model
        sample: Input sample
        target: Target class for explanation (default: model's prediction)
        device: Device to run computations on

    Returns:
        attributions: SmoothGrad relevance scores
    """
    from utils.baseline_xai import smoothgrad_relevance

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute SmoothGrad relevance
    attributions, _ = smoothgrad_relevance(
        model=model,
        x=sample,
        num_samples=40,
        noise_level=1,
        target=target  # Pass the target
    )

    return attributions


def occlusion_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for Occlusion attribution with explicit target parameter.

    Args:
        model: Trained PyTorch model
        sample: Input sample
        target: Target class for explanation (default: model's prediction)
        device: Device to run computations on

    Returns:
        attributions: Occlusion relevance scores
    """
    from utils.baseline_xai import occlusion_simpler_relevance

    # Ensure sample is a tensor on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Compute Occlusion relevance
    attributions, _ = occlusion_simpler_relevance(
        model=model,
        x=sample,
        target=target,  # Pass the target
        occlusion_type="zero",
        window_size=40  # For downsampled data
    )

    return attributions
def verify_zero_reference_hypothesis(model, test_loader, device, n_samples=100):
    """
    Test whether using zeros as reference actually changes predictions.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        device: Device to run on
        n_samples: Maximum number of samples to process

    Returns:
        Dictionary with results0 statistics
    """
    print("\n===== Verifying Zero Reference Hypothesis =====")
    model.eval()
    same_prediction_count = 0
    total_count = 0
    conf_changes = []

    # Track class-specific statistics
    class_counts = {0: 0, 1: 0}  # Assuming binary classification (good/bad)
    class_same_pred = {0: 0, 1: 0}

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if total_count >= n_samples:
                break

            current_batch_size = inputs.size(0)
            if total_count + current_batch_size > n_samples:
                # Only take what we need to reach n_samples
                inputs = inputs[:n_samples-total_count]
                targets = targets[:n_samples-total_count]
                current_batch_size = inputs.size(0)

            total_count += current_batch_size

            # Get original predictions
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Update class counts
            for c in range(2):  # For binary classification
                class_counts[c] += (preds == c).sum().item()

            # Replace inputs with zeros
            zero_inputs = torch.zeros_like(inputs)
            zero_outputs = model(zero_inputs)
            zero_probs = torch.softmax(zero_outputs, dim=1)
            zero_preds = torch.argmax(zero_outputs, dim=1)

            # Count samples where prediction didn't change
            same_prediction = (preds == zero_preds)
            same_prediction_count += same_prediction.sum().item()

            # Update class-specific same prediction counts
            for c in range(2):
                class_same_pred[c] += ((preds == c) & same_prediction).sum().item()

            # Track confidence changes
            for i in range(current_batch_size):
                pred_class = preds[i].item()
                original_conf = probs[i, pred_class].item()
                zero_conf = zero_probs[i, pred_class].item()
                conf_change = original_conf - zero_conf
                conf_changes.append((pred_class, conf_change))

            # Also test with random noise for the last batch
            if total_count >= n_samples:
                noise_inputs = torch.rand_like(inputs) * 10 - 5  # Uniform between -5 and 5
                noise_outputs = model(noise_inputs)
                noise_probs = torch.softmax(noise_outputs, dim=1)
                noise_preds = torch.argmax(noise_outputs, dim=1)

                noise_same_count = (preds == noise_preds).sum().item()
                noise_same_pct = noise_same_count / current_batch_size * 100
                print(f"Noise reference test: {noise_same_count}/{current_batch_size} samples "
                      f"({noise_same_pct:.2f}%) kept the same prediction when replaced with noise")

    # Calculate overall statistics
    same_prediction_pct = same_prediction_count / total_count * 100
    print(f"Zero reference test: {same_prediction_count}/{total_count} samples "
          f"({same_prediction_pct:.2f}%) kept the same prediction when replaced with zeros")

    # Calculate class-specific statistics
    for c in range(2):
        if class_counts[c] > 0:
            class_same_pct = class_same_pred[c] / class_counts[c] * 100
            print(f"Class {c} ('{'good' if c==0 else 'bad'}'): {class_same_pred[c]}/{class_counts[c]} "
                  f"({class_same_pct:.2f}%) kept the same prediction when replaced with zeros")

    # Calculate average confidence change
    avg_conf_change = sum(change for _, change in conf_changes) / len(conf_changes) if conf_changes else 0
    print(f"Average confidence change when replaced with zeros: {avg_conf_change:.4f}")

    # Calculate class-specific confidence changes
    class_conf_changes = {0: [], 1: []}
    for cls, change in conf_changes:
        class_conf_changes[cls].append(change)

    for c in range(2):
        if class_conf_changes[c]:
            avg_cls_change = sum(class_conf_changes[c]) / len(class_conf_changes[c])
            print(f"Class {c} ('{'good' if c==0 else 'bad'}'): Average confidence change: {avg_cls_change:.4f}")

    # Visualize confidence changes
    plt.figure(figsize=(10, 6))
    for c in range(2):
        if class_conf_changes[c]:
            plt.hist(class_conf_changes[c], alpha=0.7, bins=20,
                     label=f"Class {c} ('{'good' if c==0 else 'bad'}')")

    plt.xlabel('Confidence Change (Original - Zero Input)')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Changes When Replacing with Zeros')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("zero_reference_confidence_changes.png", dpi=300)

    return {
        'total_samples': total_count,
        'same_prediction_count': same_prediction_count,
        'same_prediction_pct': same_prediction_pct,
        'class_counts': class_counts,
        'class_same_pred': class_same_pred,
        'avg_conf_change': avg_conf_change,
        'class_conf_changes': class_conf_changes
    }
import inspect
# Main evaluation function
def main():
    # Clear CUDA cache and collect garbage before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"  # Path to your trained model
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results"
    n_steps = 20
    window_size = 20
    max_samples = 50  # Set to None to use all test samples

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Load test data
    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    print(f"Loaded test data with {len(test_loader.dataset)} samples")

    # First, verify the zero reference hypothesis
    zero_ref_results = verify_zero_reference_hypothesis(model, test_loader, device, n_samples=100)

    # Based on the results0, decide which reference value to use
    if zero_ref_results['same_prediction_pct'] > 30:  # If more than 30% predictions stay the same
        print("\n⚠️ Warning: Zero reference appears ineffective for your normalized data")
        print("But continuing with zeros as requested to isolate class tracking issue")
        reference_value = "additive_noise"  # Keep using zeros as requested
    else:
        reference_value = None  # Keep using zeros

    # Set up attribution methods - make sure these wrappers properly handle the target class
    attribution_methods = {
        "LRP": lrp_attribution_wrapper,
        "Gradient*Input": grad_times_input_wrapper,
        "SmoothGrad": smoothgrad_wrapper,
        "Occlusion": occlusion_wrapper
    }

    # Check if the attribution wrappers accept target class
    print("\nChecking if attribution methods accept target class...")
    for method_name, method in attribution_methods.items():
        if 'target' in inspect.signature(method).parameters:
            print(f"✓ {method_name} accepts target parameter")
        else:
            print(
                f"✗ {method_name} does not accept target parameter - make sure it's handled in time_window_flipping_batch")

    # Run window flipping evaluation - most important first
    print("\n===== Evaluating with most important windows flipped first =====")
    print("Tracking TRUE class probability during window flipping")
    # Change these lines in your main function

    # Run window flipping evaluation
    results_most_first, agg_results_most_first = run_window_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,  # Pass the reference value type
        max_samples=max_samples,
        output_dir=output_dir,
        device=device
    )

    # Run window flipping evaluation - least important first
    print("\n===== Evaluating with least important windows flipped first =====")
    print("Tracking TRUE class probability during window flipping")
    results_least_first, agg_results_least_first = run_window_flipping_evaluation(
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
    print("\n===== Faithfulness Ratios =====")
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

    # Add debug information about a few samples to verify we're tracking the right class
    print("\n===== Debug Information =====")
    print("Checking a few samples to verify we're tracking the true class probability:")

    sample_idx = 0
    for data, targets in test_loader:
        if sample_idx >= 3:  # Check first 3 samples
            break

        sample = data[0].to(device)
        target = targets[0].to(device)

        with torch.no_grad():
            output = model(sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            pred_class = torch.argmax(prob).item()
            true_class = target.item()

            print(f"\nSample {sample_idx}:")
            print(f"True class: {true_class}, Predicted class: {pred_class}")
            print(f"Probability for true class: {prob[true_class].item():.6f}")
            print(f"Probability for predicted class: {prob[pred_class].item():.6f}")

            # Check with all-zero input
            zero_input = torch.zeros_like(sample).to(device)
            zero_output = model(zero_input.unsqueeze(0))
            zero_prob = torch.softmax(zero_output, dim=1)[0]
            zero_pred = torch.argmax(zero_prob).item()

            print(f"All-zero input prediction: {zero_pred}")
            print(f"All-zero probability for true class: {zero_prob[true_class].item():.6f}")
            print(f"All-zero probability for original predicted class: {zero_prob[pred_class].item():.6f}")

        sample_idx += 1
if __name__ == "__main__":
    main()