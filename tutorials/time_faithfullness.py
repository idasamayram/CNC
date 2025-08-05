import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc


def time_window_flipping(model, sample, attribution_method, n_steps=20,
                         window_size=10, most_relevant_first=True,
                         reference_value=None,
                         device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform window flipping analysis on time series data to evaluate attribution method faithfulness.
    Similar to pixel flipping but deletes windows of time points rather than individual points.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first; if False, least relevant first
        reference_value: Value to replace flipped windows (default=None, which uses zeros)
        device: Device to run computations on

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

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

    # Set reference value for flipping
    if reference_value is None:
        reference_value = torch.zeros_like(sample)
    else:
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
    scores = []
    flipped_pcts = []

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    # Track the original score
    scores.append(original_score)
    flipped_pcts.append(0.0)

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


def plot_window_flipping_results(scores, flipped_pcts, most_relevant_first=True, method_name="LRP"):
    """
    Plot the results of window flipping analysis.

    Args:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
        most_relevant_first: Whether most relevant windows were flipped first
        method_name: Name of the attribution method for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(flipped_pcts, scores, 'o-', linewidth=2, markersize=8)
    plt.grid(True, linestyle='--', alpha=0.7)

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Window Flipping Analysis - {method_name}\n(Flipping {flip_order} windows first)', fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)

    # Add horizontal line at 0.5 for reference
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Add area under curve (AUC) calculation
    auc = np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
    auc_label = "AUC (lower is better)" if most_relevant_first else "AUC (higher is better)"
    plt.text(0.02, 0.02, f"{auc_label}: {auc:.4f}", transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return auc


def compare_attribution_methods_window_flipping(model, sample, attribution_methods, label=None,
                                                n_steps=20, window_size=10, most_relevant_first=True,
                                                reference_value=None,
                                                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compare multiple attribution methods using window flipping.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_methods: Dictionary of {method_name: attribution_function}
        label: Optional target label for explanations
        n_steps: Number of steps in the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: Whether to flip most relevant windows first
        reference_value: Value to replace flipped windows
        device: Device to run computations on

    Returns:
        results: Dictionary with results for each method
    """
    results = {}

    plt.figure(figsize=(12, 8))

    for method_name, attribution_func in attribution_methods.items():
        print(f"Running window flipping analysis for {method_name}...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Create a wrapped attribution function that includes the label
        def wrapped_attribution_func(model, sample):
            return attribution_func(model, sample, label)

        # Run window flipping
        scores, flipped_pcts = time_window_flipping(
            model, sample, wrapped_attribution_func, n_steps, window_size,
            most_relevant_first, reference_value, device
        )

        # Calculate AUC
        auc = np.trapz(scores, flipped_pcts) / flipped_pcts[-1]

        # Store results
        results[method_name] = {
            "scores": scores,
            "flipped_pcts": flipped_pcts,
            "auc": auc
        }

        # Plot this method
        plt.plot(flipped_pcts, scores, 'o-', linewidth=2, markersize=6, label=f"{method_name} (AUC: {auc:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Window Flipping Comparison\n(Flipping {flip_order} windows first)', fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return results


def visualize_window_flipping_progression(model, sample, attribution_method,
                                          n_steps=5, window_size=10, most_relevant_first=True,
                                          reference_value=None,
                                          device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of window flipping for time series data.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        n_steps: Number of steps to visualize
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        device: Device to run computations on
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

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
    sample_np = sample.detach().cpu().numpy()

    # Set reference value for flipping
    if reference_value is None:
        reference_value = torch.zeros_like(sample)
        reference_np = np.zeros_like(sample_np)
    else:
        reference_value = torch.full_like(sample, reference_value)
        reference_np = np.full_like(sample_np, reference_value)

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
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Set up the plot
    fig, axes = plt.subplots(n_channels, n_steps + 1, figsize=(15, 3 * n_channels))
    fig.suptitle(f'Window Flipping Progression ({"Most" if most_relevant_first else "Least"} Important First)',
                 fontsize=16)

    # If only one channel, make axes 2D
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    # Plot the original signal
    for channel in range(n_channels):
        axes[channel, 0].plot(sample_np[channel], 'b-')
        axes[channel, 0].set_title(f'Original (Score: {original_score:.3f})')
        axes[channel, 0].set_ylim([sample_np[channel].min() - 0.1, sample_np[channel].max() + 0.1])

        if channel == 0:
            axes[channel, 0].set_ylabel('Amplitude')

        if channel == n_channels - 1:
            axes[channel, 0].set_xlabel('Time')

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get flipped sample
        flipped_sample = sample_np.copy()

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

        # Get model output
        flipped_tensor = torch.tensor(flipped_sample, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Plot the flipped signal
        for channel in range(n_channels):
            axes[channel, step].plot(flipped_sample[channel], 'r-')
            axes[channel, step].set_title(
                f'{n_windows_to_flip / total_windows * 100:.1f}% Flipped (Score: {score:.3f})')
            axes[channel, step].set_ylim([sample_np[channel].min() - 0.1, sample_np[channel].max() + 0.1])

            # Add shaded regions for flipped windows
            for i in range(len(windows_to_flip)):
                if channel_indices[i] == channel:
                    window_idx = window_indices[i]
                    start_idx = window_idx * window_size
                    end_idx = min((window_idx + 1) * window_size, time_steps)
                    axes[channel, step].axvspan(start_idx, end_idx, alpha=0.2, color='gray')

            if channel == n_channels - 1:
                axes[channel, step].set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Example usage with the time series data and attribution methods

import torch
from Classification.cnn1D_model import CNN1D_Wide
from utils.baseline_xai import load_model, load_sample_data
from torch.utils.data import DataLoader


def run_window_flipping_analysis(model_path, data_dir, n_steps=10, window_size=40, most_relevant_first=True):
    """Run window flipping analysis for time series data"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()

    # Set up attribution methods
    from utils.xai_implementation import (
        compute_lrp_relevance, compute_basic_dft_lrp
    )
    from utils.baseline_xai import (
        grad_times_input_relevance, smoothgrad_relevance, occlusion_simpler_relevance
    )

    # Wrapper functions for attribution methods
    def lrp_wrapper(model, sample, label=None):
        attributions, _, _ = compute_lrp_relevance(
            model=model, sample=sample, label=label, device=device
        )
        return attributions

    def dft_lrp_wrapper(model, sample, label=None):
        relevance_time, _, _, _, _, _ = compute_basic_dft_lrp(
            model=model, sample=sample, label=label, device=device,
            signal_length=sample.shape[1], leverage_symmetry=True, precision=32, sampling_rate=400
        )
        return relevance_time

    def grad_input_wrapper(model, sample, label=None):
        attributions, _ = grad_times_input_relevance(
            model=model, x=sample, target=label
        )
        return attributions

    def smoothgrad_wrapper(model, sample, label=None):
        attributions, _ = smoothgrad_relevance(
            model=model, x=sample, num_samples=40, noise_level=1, target=label
        )
        return attributions

    def occlusion_wrapper(model, sample, label=None):
        attributions, _ = occlusion_simpler_relevance(
            model=model, x=sample, target=label, occlusion_type="zero", window_size=40
        )
        return attributions

    attribution_methods = {
        "LRP": lrp_wrapper,
        "DFT-LRP": dft_lrp_wrapper,
        "Gradient*Input": grad_input_wrapper,
        "SmoothGrad": smoothgrad_wrapper,
        "Occlusion": occlusion_wrapper
    }

    # Load a sample
    samples, labels, _ = load_sample_data(data_dir, num_samples=1)
    sample = samples[0].to(device)
    label = labels[0]

    print(f"Sample shape: {sample.shape}, Label: {label}")

    # Visualize progression for one method
    print("Visualizing window flipping progression for LRP...")
    visualize_window_flipping_progression(
        model=model,
        sample=sample,
        attribution_method=lambda model, sample: lrp_wrapper(model, sample, label),
        n_steps=5,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        device=device
    )

    # Compare methods
    print("Comparing attribution methods using window flipping...")
    results = compare_attribution_methods_window_flipping(
        model=model,
        sample=sample,
        attribution_methods=attribution_methods,
        label=label,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        device=device
    )

    # Print summary
    print("\nWindow Flipping Evaluation Summary:")
    print("-" * 50)
    print(f"{'Method':<20} {'AUC':<10}")
    print("-" * 50)

    for method_name, result in results.items():
        print(f"{method_name:<20} {result['auc']:.4f}")

    return results


# Example execution
if __name__ == "__main__":
    model_path = "../cnn1d_model_wide_new.ckpt"
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"

    # Run with most relevant windows flipped first
    print("\n===== Flipping Most Important Windows First =====")
    results_most_first = run_window_flipping_analysis(
        model_path=model_path,
        data_dir=data_dir,
        n_steps=10,
        window_size=40,
        most_relevant_first=True
    )

    # Run with least relevant windows flipped first
    print("\n===== Flipping Least Important Windows First =====")
    results_least_first = run_window_flipping_analysis(
        model_path=model_path,
        data_dir=data_dir,
        n_steps=10,
        window_size=40,
        most_relevant_first=False
    )