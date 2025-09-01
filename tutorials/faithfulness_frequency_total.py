import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import gc
import pandas as pd
import os
from datetime import datetime
from utils.dft_lrp import EnhancedDFTLRP
import seaborn as sns

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd
import os
from datetime import datetime
from utils.dft_lrp import EnhancedDFTLRP


def improved_frequency_window_flipping(model, sample, attribution_method, target_class=None, n_steps=20,
                                       window_size=10, most_relevant_first=True, reference_value=None,
                                       device="cuda" if torch.cuda.is_available() else "cpu",
                                       leverage_symmetry=True, sampling_rate=400):
    """
    Improved window flipping that properly handles symmetry and preserves energy.
    """
    import numpy as np
    import torch
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Determine frequency domain dimensions
    freq_length = signal_freq.shape[1]  # time_steps//2 + 1 if leverage_symmetry=True

    # Create DFT-LRP object for transformations (needed for exact symmetry handling)
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_inverse=True  # Need inverse transform
    )

    # Prepare reference value for frequency domain
    if reference_value == "zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

            # Special case for DC and Nyquist if leveraging symmetry
            if leverage_symmetry:
                if freq_length > 0:
                    reference_value_freq[c, 0] = 1e-10 * np.sign(signal_freq[c, 0].real)
                if freq_length > 1:
                    reference_value_freq[c, -1] = 1e-10 * np.sign(signal_freq[c, -1].real)

    elif reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "invert":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            # Invert magnitude and phase
            magnitude = np.abs(signal_freq[c])
            phase = np.angle(signal_freq[c]) + np.pi  # Invert phase

            reference_value_freq[c] = magnitude * np.exp(1j * phase)

            # Special case for DC and Nyquist
            if leverage_symmetry:
                if freq_length > 0:
                    reference_value_freq[c, 0] = -signal_freq[c, 0].real  # Keep DC real
                if freq_length > 1:
                    reference_value_freq[c, -1] = -signal_freq[c, -1].real  # Keep Nyquist real

    elif reference_value == "noise":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5
            phase = np.random.uniform(-np.pi, np.pi, freq_length)

            if leverage_symmetry:
                # Ensure DC and Nyquist have zero phase (real values)
                if freq_length > 0:
                    phase[0] = 0
                if freq_length > 1:
                    phase[-1] = 0

            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    else:  # Default to magnitude_zero
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    # Calculate window importance using absolute relevance values
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values of relevance for importance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Track full model predictions at each step
    all_class_probs = []
    with torch.no_grad():
        all_probs = torch.softmax(original_output, dim=1)[0].cpu().numpy()
        all_class_probs.append(all_probs)

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        n_windows_to_flip = min(step * windows_per_step, total_windows)
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Replace frequency components with reference
            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # Transform back to time domain
        flipped_time = np.zeros((n_channels, time_steps))

        try:
            # First try with numpy's IFFT for simpler code
            for c in range(n_channels):
                if leverage_symmetry:
                    # When leveraging symmetry, use irfft which expects half-spectrum
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    # Without symmetry, use standard ifft
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        except Exception as e:
            print(f"Error in numpy IFFT: {e}")
            print("Trying DFT-LRP inverse transform...")

            # Fallback to DFT-LRP's inverse transform which properly handles symmetry
            try:
                if hasattr(dftlrp, 'inverse_fourier_layer') and dftlrp.inverse_fourier_layer is not None:
                    for c in range(n_channels):
                        # Format data for DFT-LRP based on symmetry
                        if leverage_symmetry:
                            # When using symmetry, we need to separate real and imaginary parts correctly
                            # DC and Nyquist components should be real
                            # First prepare the real part (all components)
                            freq_real = flipped_freq[c].real

                            # Then prepare imaginary part (excluding DC and Nyquist)
                            if freq_length > 2:  # Only if we have components between DC and Nyquist
                                freq_imag = flipped_freq[c, 1:-1].imag
                            else:
                                freq_imag = np.array([])

                            # Concatenate according to DFT-LRP's expected format
                            freq_data = np.concatenate([freq_real, freq_imag])
                        else:
                            # Without symmetry, simply concatenate real and imaginary parts
                            freq_data = np.concatenate([flipped_freq[c].real, flipped_freq[c].imag])

                        # Convert to tensor
                        freq_tensor = torch.tensor(freq_data, dtype=torch.float32).unsqueeze(0)

                        if dftlrp.cuda:
                            freq_tensor = freq_tensor.cuda()

                        # Apply inverse transform
                        with torch.no_grad():
                            time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                        # Get result
                        flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                else:
                    raise ValueError("DFT-LRP inverse transform not available")
            except Exception as e:
                print(f"Error in DFT-LRP inverse: {e}")
                print("Using fallback method...")

                # Ultimate fallback - perturb original signal
                for c in range(n_channels):
                    flipped_time[c] = input_signal[c] * 0.5 + np.random.normal(0, input_signal[c].std() * 0.1,
                                                                               time_steps)

        # Normalize flipped signal to match original energy
        for c in range(n_channels):
            # Get statistics of original signal
            orig_std = input_signal[c].std()
            orig_mean = input_signal[c].mean()

            # Normalize the flipped signal to match original energy
            if flipped_time[c].std() > 0:  # Avoid division by zero
                flipped_time[c] = ((flipped_time[c] - flipped_time[c].mean()) /
                                   flipped_time[c].std() * orig_std + orig_mean)
            else:
                print(f"Warning: Channel {c} has zero standard deviation after flipping")
                # Use noise with correct statistics as fallback
                flipped_time[c] = np.random.normal(orig_mean, orig_std, time_steps)

        # Convert to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]

            # Save full probability distribution
            all_class_probs.append(prob.cpu().numpy())

            # Get score for target class (which could be true label or predicted)
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts, all_class_probs


def frequency_window_flipping_batch(model, test_loader, attribution_methods, n_steps=20, window_size=10,
                                    most_relevant_first=True, reference_value=None, max_samples=None,
                                    leverage_symmetry=True, sampling_rate=400,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run window flipping on a batch of samples with improved error handling.
    Compares flipped model output with true class labels instead of predicted labels.
    """
    from tqdm import tqdm
    import numpy as np
    import torch
    import gc

    # Initialize results0 dictionary
    results = {method_name: [] for method_name in attribution_methods.keys()}

    # Track number of processed samples
    n_processed = 0

    print("Starting frequency domain window flipping batch processing:")
    print(f"  - Window size: {window_size} frequency bins")
    print(f"  - Reference value: {reference_value}")
    print(f"  - Most relevant first: {most_relevant_first}")

    # Process batches
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
        # Process each sample in the batch
        for i in range(len(data)):
            if max_samples is not None and n_processed >= max_samples:
                print(f"Reached maximum number of samples ({max_samples})")
                break

            # Get single sample and true label
            sample = data[i]
            true_target = targets[i].item()  # Use true class label for evaluation

            # Process with each attribution method
            for method_name, attribution_method in attribution_methods.items():
                try:
                    # Run window flipping for this sample
                    scores, flipped_pcts, all_class_probs = improved_frequency_window_flipping(
                        model=model,
                        sample=sample,
                        attribution_method=attribution_method,
                        target_class=true_target,  # Use true class label
                        n_steps=n_steps,
                        window_size=window_size,
                        most_relevant_first=most_relevant_first,
                        reference_value=reference_value,
                        leverage_symmetry=leverage_symmetry,
                        sampling_rate=sampling_rate,
                        device=device
                    )

                    # Compute AUC if we have valid results0
                    if len(scores) > 1 and len(flipped_pcts) > 1:
                        auc = np.trapz(y=scores, x=flipped_pcts) / 100.0
                    else:
                        auc = float('nan')

                    # Get predictions for each class at each step
                    # (Assuming binary classification with classes 0 and 1)
                    class_predictions = {
                        'class_0': [probs[0] for probs in all_class_probs],
                        'class_1': [probs[1] for probs in all_class_probs]
                    }

                    # Check if model prediction changes at any point
                    original_pred = np.argmax(all_class_probs[0])
                    pred_changes = False
                    for probs in all_class_probs[1:]:
                        if np.argmax(probs) != original_pred:
                            pred_changes = True
                            break

                    # Store results0
                    results[method_name].append({
                        "sample_idx": n_processed,
                        "scores": scores,
                        "flipped_pcts": flipped_pcts,
                        "auc": auc,
                        "target": true_target,  # True class label
                        "original_pred": original_pred,  # Model's initial prediction
                        "pred_changes": pred_changes,  # Whether prediction changes during flipping
                        "class_predictions": class_predictions  # Predictions for each class
                    })

                except Exception as e:
                    print(f"Error processing sample {n_processed} with method {method_name}: {str(e)}")
                    # Add an empty result to keep counts consistent
                    results[method_name].append({
                        "sample_idx": n_processed,
                        "scores": None,
                        "flipped_pcts": None,
                        "auc": float('nan'),
                        "target": true_target,  # True class label
                        "error": str(e)
                    })

            # Increment processed count
            n_processed += 1

            # Clean up memory after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Break if we've processed enough samples
        if max_samples is not None and n_processed >= max_samples:
            break

    return results


def aggregate_results(results):
    """
    Aggregate window flipping results0 across samples.
    More robust version that handles errors and includes class-specific metrics.
    """
    import numpy as np

    # Initialize aggregate results0
    agg_results = {}

    for method_name, samples in results.items():
        # Filter out samples with errors
        valid_samples = [s for s in samples if s["scores"] is not None and len(s["scores"]) > 1]

        if len(valid_samples) == 0:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = {
                "avg_scores": None,
                "flipped_pcts": None,
                "auc_values": [],
                "mean_auc": None,
                "std_auc": None,
                "class_specific": {}  # Add class-specific metrics
            }
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

            # Calculate class-specific metrics (if present)
            class_specific = {}

            # Check if we have class predictions in the results0
            if "class_predictions" in valid_samples[0]:
                # Get all classes
                classes = list(valid_samples[0]["class_predictions"].keys())

                for cls in classes:
                    # Extract scores for this class
                    class_interpolated_scores = []

                    for sample in valid_samples:
                        if "class_predictions" in sample and cls in sample["class_predictions"]:
                            cls_scores = sample["class_predictions"][cls]
                            flipped_pcts = sample["flipped_pcts"]

                            if len(flipped_pcts) > 1 and len(cls_scores) > 1:
                                if np.all(np.diff(flipped_pcts) > 0):
                                    # Interpolate to common x-axis
                                    from scipy.interpolate import interp1d
                                    f = interp1d(flipped_pcts, cls_scores, bounds_error=False, fill_value="extrapolate")
                                    class_interpolated_scores.append(f(common_x))

                    # Average interpolated scores for this class
                    if len(class_interpolated_scores) > 0:
                        class_avg_scores = np.mean(class_interpolated_scores, axis=0)

                        # Calculate AUC for this class
                        class_auc = np.trapz(class_avg_scores, common_x) / 100.0

                        class_specific[cls] = {
                            "avg_scores": class_avg_scores,
                            "auc": class_auc
                        }

            # Calculate prediction changes statistics
            if "pred_changes" in valid_samples[0]:
                change_counts = sum(1 for s in valid_samples if s.get("pred_changes", False))
                change_percent = (change_counts / len(valid_samples)) * 100

                class_specific["pred_changes"] = {
                    "count": change_counts,
                    "percent": change_percent
                }

            # Calculate metrics by true class
            class_0_samples = [s for s in valid_samples if s["target"] == 0]
            class_1_samples = [s for s in valid_samples if s["target"] == 1]

            if class_0_samples:
                class_0_auc = np.mean([s["auc"] for s in class_0_samples if not np.isnan(s["auc"])])
                class_specific["class_0_metrics"] = {"mean_auc": class_0_auc, "count": len(class_0_samples)}

            if class_1_samples:
                class_1_auc = np.mean([s["auc"] for s in class_1_samples if not np.isnan(s["auc"])])
                class_specific["class_1_metrics"] = {"mean_auc": class_1_auc, "count": len(class_1_samples)}

            # Store aggregated results0
            agg_results[method_name] = {
                "avg_scores": avg_scores,
                "flipped_pcts": common_x,
                "auc_values": auc_values,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "class_specific": class_specific
            }

        except Exception as e:
            print(f"Error aggregating results0 for {method_name}: {e}")
            agg_results[method_name] = {
                "avg_scores": None,
                "flipped_pcts": None,
                "auc_values": auc_values,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "class_specific": {}
            }

    return agg_results


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value='zero'):
    """
    Plot aggregated window flipping results0.
    Enhanced to show class-specific results0.
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
    plt.title(f'Aggregate Frequency Window Flipping Results\n(Flipping {flip_order} windows first, '
              f'Reference: {reference_value})', fontsize=16)
    plt.xlabel('Percentage of Frequency Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score for True Class', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Return the figure for saving
    return plt.gcf()


def plot_class_specific_results(agg_results, most_relevant_first=True, reference_value='zero'):
    """
    Plot class-specific window flipping results0.
    Shows how flipping affects predictions for each class separately.
    """
    # Set up the figure - one plot per method
    method_count = len(agg_results)
    fig, axes = plt.subplots(1, method_count, figsize=(6 * method_count, 6), squeeze=False)

    for i, (method_name, results) in enumerate(agg_results.items()):
        ax = axes[0, i]

        # Check if we have class-specific results0
        if "class_specific" in results and results["class_specific"]:
            class_specific = results["class_specific"]

            # Plot for class 0 if available
            if "class_0" in class_specific and class_specific["class_0"]["avg_scores"] is not None:
                ax.plot(results["flipped_pcts"], class_specific["class_0"]["avg_scores"],
                        label=f"Class 0 (AUC: {class_specific['class_0']['auc']:.4f})",
                        color='blue')

            # Plot for class 1 if available
            if "class_1" in class_specific and class_specific["class_1"]["avg_scores"] is not None:
                ax.plot(results["flipped_pcts"], class_specific["class_1"]["avg_scores"],
                        label=f"Class 1 (AUC: {class_specific['class_1']['auc']:.4f})",
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
    fig.suptitle(f'Class-Specific Window Flipping Results\n(Flipping {flip_order} windows first, '
                 f'Reference: {reference_value})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    return fig


def plot_prediction_changes(agg_results, most_relevant_first=True, reference_value='zero'):
    """
    Plot statistics about prediction changes during window flipping.
    Shows how often model predictions change as windows are flipped.
    """
    # Extract prediction change percentages
    methods = []
    change_pcts = []

    for method_name, results in agg_results.items():
        if "class_specific" in results and "pred_changes" in results["class_specific"]:
            methods.append(method_name)
            change_pcts.append(results["class_specific"]["pred_changes"]["percent"])

    if not methods:
        print("No prediction change data available")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(methods, change_pcts, color='teal', alpha=0.7)

    ax.set_ylabel('Percentage of Samples with Prediction Changes (%)')
    ax.set_title(f'Prediction Changes During Window Flipping\n'
                 f'(Flipping {("most" if most_relevant_first else "least")} important windows first, '
                 f'Reference: {reference_value})')

    # Add text labels on top of bars
    for i, v in enumerate(change_pcts):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)

    ax.set_ylim(0, max(change_pcts) * 1.2)  # Add some space for text labels
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig


def plot_auc_by_class(agg_results, most_relevant_first=True, reference_value='zero'):
    """
    Plot AUC values separated by class.
    Shows how explanation quality differs between classes.
    """
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
    ax.set_title(f'AUC by Class for Different Attribution Methods\n'
                 f'(Flipping {("most" if most_relevant_first else "least")} important windows first, '
                 f'Reference: {reference_value})')
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


def run_frequency_window_flipping_evaluation(model, test_loader, attribution_methods,
                                             n_steps=10, window_size=10,
                                             most_relevant_first=True, reference_value=None, max_samples=None,
                                             leverage_symmetry=True, sampling_rate=400,
                                             output_dir="./results0",
                                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency domain window flipping evaluation on test set and save results0.
    Enhanced to track true class scores and prediction changes.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"

    # Include reference value in filename
    ref_type = "zero"
    if reference_value == "noise":
        ref_type = "noise"
    elif reference_value == "mild_noise":
        ref_type = "mild_noise"
    elif reference_value == "extreme":
        ref_type = "extreme"
    elif reference_value == "shift":
        ref_type = "shift"
    elif reference_value == "invert":
        ref_type = "invert"
    elif reference_value == "magnitude_zero":
        ref_type = "magnitude_zero"
    elif reference_value == "phase_zero":
        ref_type = "phase_zero"

    filename_prefix = f"{output_dir}/freq_window_flipping_{flip_order}_{ref_type}_{timestamp}"

    print(f"Starting frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}, Leverage symmetry: {leverage_symmetry}")

    # Run window flipping on all samples
    results = frequency_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    # Compute aggregated results0
    print("Computing aggregated results0...")
    agg_results = aggregate_results(results)

    # Create and save plots
    print("Creating visualizations...")

    # Fix 1: Close any existing plots before creating new ones
    plt.close('all')

    # Plot 1: Traditional aggregate plot
    fig1 = plot_aggregate_results(agg_results, most_relevant_first, reference_value)
    fig1.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Class-specific results0
    try:
        fig2 = plot_class_specific_results(agg_results, most_relevant_first, reference_value)
        if fig2 is not None:
            fig2.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
    except Exception as e:
        print(f"Error creating class-specific plot: {e}")

    # Plot 3: Prediction changes
    try:
        fig3 = plot_prediction_changes(agg_results, most_relevant_first, reference_value)
        if fig3 is not None:
            fig3.savefig(f"{filename_prefix}_prediction_changes.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
    except Exception as e:
        print(f"Error creating prediction changes plot: {e}")

    # Plot 4: AUC by class
    try:
        fig4 = plot_auc_by_class(agg_results, most_relevant_first, reference_value)
        if fig4 is not None:
            fig4.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)
    except Exception as e:
        print(f"Error creating AUC by class plot: {e}")

    # Save numerical results0 to CSV files
    print("Saving numerical results0...")

    # Create basic aggregate results0 CSV
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

            # Add prediction change metrics
            if "pred_changes" in class_specific:
                method_data["Pred_Change_Count"] = class_specific["pred_changes"]["count"]
                method_data["Pred_Change_Pct"] = class_specific["pred_changes"]["percent"]

        agg_data.append(method_data)

    # Save to CSV
    pd.DataFrame(agg_data).to_csv(f"{filename_prefix}_aggregate.csv", index=False)

    # Save detailed sample results0
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

            # Add prediction info if available
            if "original_pred" in sample:
                sample_data["Original_Pred"] = sample["original_pred"]

            if "pred_changes" in sample:
                sample_data["Prediction_Changes"] = sample["pred_changes"]

            samples_data.append(sample_data)

    if samples_data:
        pd.DataFrame(samples_data).to_csv(f"{filename_prefix}_samples.csv", index=False)

    # Print summary table
    print("\n" + "=" * 60)
    print("Frequency Domain Window Flipping Evaluation Summary:")
    print("=" * 60)

    headers = ["Method", "Overall AUC", "Normal AUC", "Faulty AUC", "Pred Changes"]
    print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15}")
    print("-" * 80)

    for method_name, method_results in agg_results.items():
        # Get basic AUC
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')

        # Get class-specific metrics
        class_specific = method_results.get("class_specific", {})
        normal_auc = class_specific.get("class_0_metrics", {}).get("mean_auc", float('nan'))
        faulty_auc = class_specific.get("class_1_metrics", {}).get("mean_auc", float('nan'))

        # Get prediction changes
        pred_changes = class_specific.get("pred_changes", {}).get("percent", float('nan'))

        # Print row
        print(f"{method_name:<20} {mean_auc:<15.4f} {normal_auc:<15.4f} {faulty_auc:<15.4f} {pred_changes:<15.1f}%")

    print("=" * 60)

    return results, agg_results


def compare_most_least_important(model, test_loader, attribution_methods,
                                 n_steps=10, window_size=10, reference_value=None,
                                 max_samples=None, leverage_symmetry=True, sampling_rate=400,
                                 output_dir="./results0", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compare flipping most important vs. least important windows and calculate faithfulness ratio.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of attribution methods
        n_steps: Number of flipping steps
        window_size: Size of frequency windows
        reference_value: Value to use for flipped windows
        max_samples: Maximum number of samples to process
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz
        output_dir: Directory to save results0
        device: Device to use for computation

    Returns:
        faithfulness_results: Dictionary with faithfulness ratios for each method
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Running evaluation with most important windows flipped first...")
    results_most, agg_results_most = run_frequency_window_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        reference_value=reference_value,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        output_dir=output_dir,
        device=device
    )

    print("Running evaluation with least important windows flipped first...")
    results_least, agg_results_least = run_frequency_window_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        reference_value=reference_value,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        output_dir=output_dir,
        device=device
    )

    # Calculate faithfulness ratios
    faithfulness_results = {}

    for method_name in attribution_methods.keys():
        if (method_name in agg_results_most and
                method_name in agg_results_least and
                agg_results_most[method_name]["mean_auc"] is not None and
                agg_results_least[method_name]["mean_auc"] is not None):

            most_auc = agg_results_most[method_name]["mean_auc"]
            least_auc = agg_results_least[method_name]["mean_auc"]

            # Calculate faithfulness ratio (least/most)
            if most_auc > 0:
                faithfulness_ratio = least_auc / most_auc
            else:
                faithfulness_ratio = float('nan')

            # Store in results0
            faithfulness_results[method_name] = {
                "most_auc": most_auc,
                "least_auc": least_auc,
                "faithfulness_ratio": faithfulness_ratio
            }

            # Add class-specific ratios if available
            if ("class_specific" in agg_results_most[method_name] and
                    "class_specific" in agg_results_least[method_name]):

                class_specific_most = agg_results_most[method_name]["class_specific"]
                class_specific_least = agg_results_least[method_name]["class_specific"]

                # Class 0 (normal) faithfulness
                if ("class_0_metrics" in class_specific_most and
                        "class_0_metrics" in class_specific_least):

                    most_class0_auc = class_specific_most["class_0_metrics"]["mean_auc"]
                    least_class0_auc = class_specific_least["class_0_metrics"]["mean_auc"]

                    if most_class0_auc > 0:
                        class0_ratio = least_class0_auc / most_class0_auc
                    else:
                        class0_ratio = float('nan')

                    faithfulness_results[method_name]["class0_most_auc"] = most_class0_auc
                    faithfulness_results[method_name]["class0_least_auc"] = least_class0_auc
                    faithfulness_results[method_name]["class0_faithfulness_ratio"] = class0_ratio

                # Class 1 (faulty) faithfulness
                if ("class_1_metrics" in class_specific_most and
                        "class_1_metrics" in class_specific_least):

                    most_class1_auc = class_specific_most["class_1_metrics"]["mean_auc"]
                    least_class1_auc = class_specific_least["class_1_metrics"]["mean_auc"]

                    if most_class1_auc > 0:
                        class1_ratio = least_class1_auc / most_class1_auc
                    else:
                        class1_ratio = float('nan')

                    faithfulness_results[method_name]["class1_most_auc"] = most_class1_auc
                    faithfulness_results[method_name]["class1_least_auc"] = least_class1_auc
                    faithfulness_results[method_name]["class1_faithfulness_ratio"] = class1_ratio

    # Create faithfulness plot
    plt.figure(figsize=(12, 8))
    methods = list(faithfulness_results.keys())
    ratios = [faithfulness_results[m]["faithfulness_ratio"] for m in methods]

    plt.bar(methods, ratios, color='teal', alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--')

    plt.title(f'Faithfulness Ratio (Least AUC / Most AUC)\nReference: {reference_value}', fontsize=16)
    plt.ylabel('Faithfulness Ratio', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # Add text labels on bars
    for i, v in enumerate(ratios):
        plt.text(i, v + 0.05, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/faithfulness_ratio_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create class-specific faithfulness plot if data is available
    if any("class0_faithfulness_ratio" in faithfulness_results[m] for m in methods):
        plt.figure(figsize=(14, 8))

        # Extract ratios
        overall_ratios = [faithfulness_results[m].get("faithfulness_ratio", float('nan')) for m in methods]
        class0_ratios = [faithfulness_results[m].get("class0_faithfulness_ratio", float('nan')) for m in methods]
        class1_ratios = [faithfulness_results[m].get("class1_faithfulness_ratio", float('nan')) for m in methods]

        # Set up bar positions
        x = np.arange(len(methods))
        width = 0.25

        # Create grouped bar chart
        plt.bar(x - width, overall_ratios, width, label='Overall', color='teal', alpha=0.7)
        plt.bar(x, class0_ratios, width, label='Normal (Class 0)', color='lightgreen', alpha=0.7)
        plt.bar(x + width, class1_ratios, width, label='Faulty (Class 1)', color='salmon', alpha=0.7)

        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.title(f'Class-Specific Faithfulness Ratios\nReference: {reference_value}', fontsize=16)
        plt.ylabel('Faithfulness Ratio (Least AUC / Most AUC)', fontsize=14)
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_specific_faithfulness_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Save faithfulness results0 to CSV
    faithfulness_df = pd.DataFrame([
        {
            "Method": method,
            "Most_AUC": results["most_auc"],
            "Least_AUC": results["least_auc"],
            "Faithfulness_Ratio": results["faithfulness_ratio"],
            "Class0_Most_AUC": results.get("class0_most_auc", float('nan')),
            "Class0_Least_AUC": results.get("class0_least_auc", float('nan')),
            "Class0_Faithfulness_Ratio": results.get("class0_faithfulness_ratio", float('nan')),
            "Class1_Most_AUC": results.get("class1_most_auc", float('nan')),
            "Class1_Least_AUC": results.get("class1_least_auc", float('nan')),
            "Class1_Faithfulness_Ratio": results.get("class1_faithfulness_ratio", float('nan'))
        }
        for method, results in faithfulness_results.items()
    ])

    faithfulness_df.to_csv(f"{output_dir}/faithfulness_results_{timestamp}.csv", index=False)

    # Print faithfulness summary table
    print("\n" + "=" * 80)
    print("Frequency Domain Faithfulness Results:")
    print("=" * 80)
    print(f"{'Method':<20} {'Most AUC':<10} {'Least AUC':<10} {'Ratio':<10} {'Normal Ratio':<15} {'Faulty Ratio':<15}")
    print("-" * 80)

    for method in methods:
        results = faithfulness_results[method]
        most_auc = results["most_auc"]
        least_auc = results["least_auc"]
        ratio = results["faithfulness_ratio"]

        normal_ratio = results.get("class0_faithfulness_ratio", float('nan'))
        faulty_ratio = results.get("class1_faithfulness_ratio", float('nan'))

        print(
            f"{method:<20} {most_auc:<10.4f} {least_auc:<10.4f} {ratio:<10.2f} {normal_ratio:<15.2f} {faulty_ratio:<15.2f}")

    print("=" * 80)

    return faithfulness_results


def extract_important_freq_windows(model, sample, attribution_method, target_class=None,
                                   n_windows=10, window_size=10,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   leverage_symmetry=True, sampling_rate=400):
    """
    Extract the most important windows from frequency domain based on attribution scores.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates frequency domain attributions
        target_class: Target class for explanation (default: model's prediction)
        n_windows: Number of top windows to extract
        window_size: Size of frequency windows
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz

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

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Get frequency domain dimensions
    freq_length = relevance_freq.shape[1]

    # Calculate window importance
    n_windows_per_channel = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows_per_channel += 1

    window_importance = np.zeros((n_channels, n_windows_per_channel))

    for channel in range(n_channels):
        for window_idx in range(n_windows_per_channel):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values of relevance for importance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

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
        end_idx = min((window_idx + 1) * window_size, freq_length)

        # Extract frequency data
        freq_data = signal_freq[channel_idx, start_idx:end_idx]

        # Calculate frequency range for this window
        if freqs is not None and len(freqs) >= end_idx:
            freq_range = (freqs[start_idx], freqs[end_idx - 1])
        else:
            # Calculate frequencies if not provided
            if leverage_symmetry:
                # For real signals with symmetry, frequencies are from 0 to Nyquist
                nyquist = sampling_rate / 2
                freq_range = (start_idx * nyquist / freq_length, (end_idx - 1) * nyquist / freq_length)
            else:
                # For complex signals without symmetry, frequencies are from -sampling_rate/2 to sampling_rate/2
                freq_range = (
                    (start_idx - freq_length / 2) * sampling_rate / freq_length,
                    (end_idx - 1 - freq_length / 2) * sampling_rate / freq_length
                )

        # Calculate statistics for this window
        magnitude = np.abs(freq_data)
        phase = np.angle(freq_data)

        window_info = {
            'freq_data': freq_data,
            'magnitude': magnitude,
            'phase': phase,
            'freq_start': start_idx,
            'freq_end': end_idx,
            'freq_range_hz': freq_range,
            'channel': channel_idx,
            'relevance': flat_importance[top_indices[i]],
            'window_idx': window_idx,
            'avg_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude),
            'avg_phase': np.mean(phase),
            'window_size': end_idx - start_idx
        }

        # Add additional features
        # Find peak frequency in this window
        if len(magnitude) > 0:
            peak_idx = np.argmax(magnitude)
            if freqs is not None and start_idx + peak_idx < len(freqs):
                window_info['peak_freq_hz'] = freqs[start_idx + peak_idx]
            else:
                # Calculate peak frequency if freqs not provided
                if leverage_symmetry:
                    window_info['peak_freq_hz'] = (start_idx + peak_idx) * sampling_rate / (2 * freq_length)
                else:
                    window_info['peak_freq_hz'] = (start_idx + peak_idx - freq_length / 2) * sampling_rate / freq_length
        else:
            window_info['peak_freq_hz'] = 0

        important_windows.append(window_info)

    return important_windows

def collect_important_freq_windows(model, test_loader, attribution_method,
                                   n_samples_per_class=10, n_windows=10, window_size=10,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   leverage_symmetry=True, sampling_rate=400):
    """
    Collect important frequency windows from multiple samples.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_method: Function that generates frequency domain attributions
        n_samples_per_class: Number of samples to process per class
        n_windows: Number of top windows to extract per sample
        window_size: Size of frequency windows
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz

    Returns:
        all_windows: List of dictionaries containing window info
    """
    all_windows = []

    # Keep track of samples processed per class
    samples_count = {0: 0, 1: 0}  # Assuming binary classification (0=normal, 1=faulty)
    total_processed = 0

    # Process each batch
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing samples")):
        # Process each sample in the batch
        for i in range(len(data)):
            sample = data[i]
            target = targets[i].item()

            # Skip if we have enough samples for this class
            if samples_count[target] >= n_samples_per_class:
                continue

            # Extract important windows
            try:
                windows = extract_important_freq_windows(
                    model=model,
                    sample=sample,
                    attribution_method=attribution_method,
                    target_class=target,
                    n_windows=n_windows,
                    window_size=window_size,
                    device=device,
                    leverage_symmetry=leverage_symmetry,
                    sampling_rate=sampling_rate
                )

                # Add sample metadata to each window
                for window in windows:
                    window['sample_idx'] = total_processed
                    window['class'] = target
                    window['class_name'] = 'normal' if target == 0 else 'faulty'

                all_windows.extend(windows)

                # Update counts
                samples_count[target] += 1
                total_processed += 1

            except Exception as e:
                print(f"Error processing sample {total_processed}: {str(e)}")

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Check if we have enough samples
        if all(count >= n_samples_per_class for count in samples_count.values()):
            print(f"Collected windows from {samples_count[0]} normal and {samples_count[1]} faulty samples")
            break

    return all_windows


def visualize_freq_windows(windows, output_dir="./results0/freq_windows"):
    """
    Visualize the important frequency windows.

    Args:
        windows: List of window dictionaries from extract_important_freq_windows
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group windows by class
    normal_windows = [w for w in windows if w['class'] == 0]
    faulty_windows = [w for w in windows if w['class'] == 1]

    # Create a scatter plot of frequency vs. magnitude, colored by relevance
    plt.figure(figsize=(12, 10))

    # Plot normal windows
    plt.subplot(2, 1, 1)
    relevances = [w['relevance'] for w in normal_windows]
    peak_freqs = [w['peak_freq_hz'] for w in normal_windows]
    magnitudes = [w['avg_magnitude'] for w in normal_windows]

    plt.scatter(peak_freqs, magnitudes, c=relevances, cmap='viridis',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Frequency Windows - Normal Samples')
    plt.xlabel('Peak Frequency (Hz)')
    plt.ylabel('Average Magnitude')
    plt.grid(alpha=0.3)

    # Plot faulty windows
    plt.subplot(2, 1, 2)
    relevances = [w['relevance'] for w in faulty_windows]
    peak_freqs = [w['peak_freq_hz'] for w in faulty_windows]
    magnitudes = [w['avg_magnitude'] for w in faulty_windows]

    plt.scatter(peak_freqs, magnitudes, c=relevances, cmap='plasma',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Frequency Windows - Faulty Samples')
    plt.xlabel('Peak Frequency (Hz)')
    plt.ylabel('Average Magnitude')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/freq_windows_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create histograms of peak frequencies
    plt.figure(figsize=(12, 6))

    plt.hist([w['peak_freq_hz'] for w in normal_windows], bins=20, alpha=0.5,
             label='Normal', density=True)
    plt.hist([w['peak_freq_hz'] for w in faulty_windows], bins=20, alpha=0.5,
             label='Faulty', density=True)

    plt.title('Distribution of Peak Frequencies in Important Windows')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"{output_dir}/peak_freq_distribution.png", dpi=300, bbox_inches='tight')
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
    plt.title('Channel Distribution of Important Frequency Windows')
    plt.xticks(x, [f'Channel {c}' for c in channels])
    plt.legend()

    plt.savefig(f"{output_dir}/channel_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create heatmap of frequency ranges
    # Group frequencies into bins
    max_freq = max(w['freq_range_hz'][1] for w in windows)
    freq_bins = np.linspace(0, max_freq, 20)

    # Count windows in each frequency bin for each class
    normal_counts = np.zeros(len(freq_bins) - 1)
    faulty_counts = np.zeros(len(freq_bins) - 1)

    for window in normal_windows:
        freq_start, freq_end = window['freq_range_hz']
        for i in range(len(freq_bins) - 1):
            if freq_start <= freq_bins[i + 1] and freq_end >= freq_bins[i]:
                normal_counts[i] += 1

    for window in faulty_windows:
        freq_start, freq_end = window['freq_range_hz']
        for i in range(len(freq_bins) - 1):
            if freq_start <= freq_bins[i + 1] and freq_end >= freq_bins[i]:
                faulty_counts[i] += 1

    # Create heatmap
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.bar(freq_bins[:-1], normal_counts, width=freq_bins[1] - freq_bins[0],
            align='edge', alpha=0.7)
    plt.title('Frequency Distribution - Normal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.bar(freq_bins[:-1], faulty_counts, width=freq_bins[1] - freq_bins[0],
            align='edge', alpha=0.7)
    plt.title('Frequency Distribution - Faulty')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/frequency_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot magnitudes vs. frequencies for each window
    plt.figure(figsize=(15, 10))

    # Plot a subset of windows
    n_to_plot = min(5, len(normal_windows), len(faulty_windows))

    for i in range(n_to_plot):
        # Normal window
        plt.subplot(2, n_to_plot, i + 1)
        if i < len(normal_windows):
            window = normal_windows[i]
            plt.bar(range(len(window['magnitude'])), window['magnitude'])
            plt.title(f"Normal #{i + 1}\nPeak: {window['peak_freq_hz']:.1f} Hz")

        # Faulty window
        plt.subplot(2, n_to_plot, n_to_plot + i + 1)
        if i < len(faulty_windows):
            window = faulty_windows[i]
            plt.bar(range(len(window['magnitude'])), window['magnitude'])
            plt.title(f"Faulty #{i + 1}\nPeak: {window['peak_freq_hz']:.1f} Hz")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/window_magnitudes.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_freq_windows(windows, output_dir="./results0/freq_windows"):
    """
    Analyze the important frequency windows and identify distinguishing features.

    Args:
        windows: List of window dictionaries
        output_dir: Directory to save results0
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{k: v for k, v in w.items() if not isinstance(v, (np.ndarray, complex))}
                       for w in windows])

    # Add derived features
    df['magnitude_std'] = [np.std(w['magnitude']) for w in windows]
    df['phase_std'] = [np.std(w['phase']) for w in windows]
    df['magnitude_skew'] = [stats.skew(w['magnitude']) if len(w['magnitude']) > 2 else 0 for w in windows]

    # Calculate spectral entropy
    def spectral_entropy(magnitude):
        if np.sum(magnitude) == 0:
            return 0
        psd = magnitude / np.sum(magnitude)
        psd_clean = psd[psd > 0]  # Remove zeros
        if len(psd_clean) == 0:
            return 0
        return -np.sum(psd_clean * np.log2(psd_clean))

    df['spectral_entropy'] = [spectral_entropy(w['magnitude']) for w in windows]

    # Calculate frequency range width
    df['freq_width'] = df['freq_range_hz'].apply(lambda x: x[1] - x[0])

    # Analyze class differences
    features = ['peak_freq_hz', 'avg_magnitude', 'max_magnitude', 'avg_phase',
                'magnitude_std', 'phase_std', 'magnitude_skew', 'spectral_entropy', 'freq_width']

    class_stats = []
    for feature in features:
        normal_values = df[df['class'] == 0][feature]
        faulty_values = df[df['class'] == 1][feature]

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

    # Save results0
    class_stats_df.to_csv(f"{output_dir}/freq_feature_stats.csv", index=False)
    df.to_csv(f"{output_dir}/freq_windows_data.csv", index=False)

    # Create summary visualizations
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
    plt.figure(figsize=(12, 10))

    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
    plt.close()

    return class_stats_df


def main():
    """
    Main function to extract and analyze important frequency windows.
    Also performs window flipping evaluations with aggregated results0.
    """
    # Clean up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results/freq_windows_new3"
    n_samples_per_class = 10
    n_windows = 10
    window_size = 20
    n_steps = 20  # Number of steps for window flipping
    sampling_rate = 400
    leverage_symmetry = True
    max_samples = 10  # For batch processing

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    from utils.baseline_xai import load_model
    model = load_model(model_path, device)
    model.eval()

    # Load test data
    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    print(f"Loaded test data with {len(test_loader.dataset)} samples")

    # Import your frequency domain attribution methods
    from utils.xai_implementation import compute_basic_dft_lrp, compute_dft_gradient_input, compute_dft_smoothgrad, \
        compute_dft_occlusion

    # Set up attribution methods dictionary
    attribution_methods = {
        "DFT-LRP": compute_basic_dft_lrp,
        "DFT-GradInput": compute_dft_gradient_input,
        "DFT-SmoothGrad": compute_dft_smoothgrad,
        "DFT-Occlusion": compute_dft_occlusion
    }

    # First, test the improved frequency window flipping on a single sample
    print("\n===== Testing Improved Frequency Window Flipping =====")

    # Get a single sample
    for batch_idx, (data, targets) in enumerate(test_loader):
        sample = data[0].to(device)
        target = targets[0].item()
        break

    # Run improved window flipping
    scores, flipped_pcts, all_class_probs = improved_frequency_window_flipping(
        model=model,
        sample=sample,
        attribution_method=compute_basic_dft_lrp,
        target_class=target,
        n_steps=10,
        window_size=window_size,
        most_relevant_first=True,
        reference_value="zero",
        device=device,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Plot results0
    plt.figure(figsize=(10, 6))
    plt.plot(flipped_pcts, scores, marker='o')
    plt.title('Frequency Window Flipping Results')
    plt.xlabel('Percentage of Windows Flipped (%)')
    plt.ylabel('Model Confidence Score')
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/freq_window_flipping_test.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Window flipping results0: starting score = {scores[0]:.4f}, final score = {scores[-1]:.4f}")

    # Run window flipping evaluation - most important first
    print("\n===== Evaluating with most important frequency windows flipped first =====")
    print("Tracking TRUE class probability during window flipping")


    # Run comprehensive evaluation
    print("\n===== Running Comprehensive Faithfulness Evaluation =====")
    faithfulness_results = compare_most_least_important(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        reference_value="zero",
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        output_dir=output_dir,
        device=device
    )

    # Collect important frequency windows
    print("\n===== Collecting Important Frequency Windows =====")
    windows = collect_important_freq_windows(
        model=model,
        test_loader=test_loader,
        attribution_method=compute_basic_dft_lrp,
        n_samples_per_class=n_samples_per_class,
        n_windows=n_windows,
        window_size=window_size,
        device=device,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Visualize windows
    print("\n===== Visualizing Frequency Windows =====")
    visualize_freq_windows(windows, output_dir)

    # Analyze windows
    print("\n===== Analyzing Frequency Windows =====")
    stats = analyze_freq_windows(windows, output_dir)

    # Print top discriminative features
    print("\n===== Top Discriminative Frequency Features =====")
    print(stats.head(5))

    # Save windows to file
    # Remove complex data and numpy arrays before saving
    windows_data = [{k: v for k, v in w.items() if not isinstance(v, (np.ndarray, complex))}
                    for w in windows]
    pd.DataFrame(windows_data).to_csv(f"{output_dir}/important_freq_windows.csv", index=False)

    print(f"\nAnalysis complete. Results saved to {output_dir}")

    # Try different reference values for comparison
    reference_values = ["zero", "magnitude_zero", "noise"]

    for ref_value in reference_values:
        print(f"\n===== Testing reference value: {ref_value} =====")
        # Run a quick test with this reference value
        results, agg_results = run_frequency_window_flipping_evaluation(
            model=model,
            test_loader=test_loader,
            attribution_methods={"DFT-LRP": compute_basic_dft_lrp},  # Use just one method for quick test
            n_steps=10,
            window_size=window_size,
            most_relevant_first=True,
            reference_value=ref_value,
            max_samples=10,  # Use fewer samples for quick tests
            leverage_symmetry=leverage_symmetry,
            sampling_rate=sampling_rate,
            output_dir=f"{output_dir}/reference_tests",
            device=device
        )


if __name__ == "__main__":
    main()