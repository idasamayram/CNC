# time_freq_flipping.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from scipy import signal


def tf_window_flipping(model, sample, relevance_timefreq, signal_timefreq, freqs,
                       n_steps=20, bottom_to_top=True,
                       time_windows=None, window_size=40,
                       reference_value=0.0,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform time-frequency window flipping, starting from bottom frequencies and moving up.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (n_channels, time_steps)
        relevance_timefreq: Time-frequency relevance map from DFT-LRP or similar (n_channels, n_freq, n_time)
        signal_timefreq: Time-frequency signal representation
        freqs: Frequency values for the spectrogram
        n_steps: Number of steps to divide the flipping process
        bottom_to_top: If True, flip from lowest to highest frequencies; if False, highest to lowest
        time_windows: Optional pre-computed time window mapping
        window_size: Size of time windows to flip if time_windows is None
        reference_value: Value to replace flipped points
        device: Device to run computations on

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped regions at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    sample_np = sample.detach().cpu().numpy()

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get the shape of the time-frequency representation
    n_freq, n_time_frames = relevance_timefreq.shape[1], relevance_timefreq.shape[2]

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    # Initialize scores and percentages
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate frequency steps
    freq_per_step = max(1, n_freq // n_steps)

    # Calculate approximate window size in time domain
    if time_windows is None:
        # Map from time frames in spectrogram to time points in original signal
        time_window_size = time_steps // n_time_frames if n_time_frames > 0 else 1
        time_windows = []
        for t in range(n_time_frames):
            start = t * time_window_size
            end = min((t + 1) * time_window_size, time_steps)
            time_windows.append((start, end))

    # Iterate through frequency bands from bottom to top (or top to bottom)
    freq_indices = np.arange(n_freq)
    if not bottom_to_top:
        freq_indices = freq_indices[::-1]  # Reverse to go from top to bottom

    for step in range(1, n_steps + 1):
        print(f"Step {step}/{n_steps}")

        # Calculate number of frequency bands to flip
        n_freqs_to_flip = min(step * freq_per_step, n_freq)
        freq_to_flip = freq_indices[:n_freqs_to_flip]

        # Create flipped signal
        flipped_signal = sample_np.copy()
        flipped_mask = np.zeros_like(sample_np, dtype=bool)

        # For each frequency band to flip
        for freq_idx in freq_to_flip:
            # For each time frame
            for time_idx in range(n_time_frames):
                # Get corresponding time window in original signal
                start, end = time_windows[time_idx]

                # For each channel
                for channel in range(n_channels):
                    # Get relevance for this TF point
                    relevance = abs(relevance_timefreq[channel, freq_idx, time_idx])

                    # Only flip if relevance is above a threshold (optional)
                    # if relevance > relevance_threshold:

                    # Apply flipping to time window
                    flipped_signal[channel, start:end] = reference_value
                    flipped_mask[channel, start:end] = True

        # Get model prediction for flipped signal
        flipped_tensor = torch.tensor(flipped_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Calculate percentage of flipped points
        flipped_pct = np.sum(flipped_mask) / flipped_mask.size * 100.0

        # Store results0
        scores.append(score)
        flipped_pcts.append(flipped_pct)

        print(f"  Flipped {flipped_pct:.2f}% of signal, score = {score:.4f}")

    return scores, flipped_pcts


def visualize_tf_flipping(model, sample, relevance_timefreq, signal_timefreq, freqs,
                          n_steps=5, bottom_to_top=True, reference_value=0.0,
                          device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize time-frequency flipping with spectrograms and signals.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (n_channels, time_steps)
        relevance_timefreq: Time-frequency relevance map
        signal_timefreq: Time-frequency signal representation
        freqs: Frequency values for the spectrogram
        n_steps: Number of visualization steps
        bottom_to_top: If True, flip from lowest to highest frequencies
        reference_value: Value to replace flipped points
        device: Device to run computations on
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    sample_np = sample.detach().cpu().numpy()

    # Get shapes
    n_channels, time_steps = sample.shape
    n_freq, n_time_frames = relevance_timefreq.shape[1], relevance_timefreq.shape[2]

    # Original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    # Calculate frequency steps
    freq_per_step = max(1, n_freq // n_steps)

    # Map from spectrogram time frames to signal time points
    time_window_size = time_steps // n_time_frames if n_time_frames > 0 else 1
    time_windows = []
    for t in range(n_time_frames):
        start = t * time_window_size
        end = min((t + 1) * time_window_size, time_steps)
        time_windows.append((start, end))

    # Frequency indices order (bottom to top or top to bottom)
    freq_indices = np.arange(n_freq)
    if not bottom_to_top:
        freq_indices = freq_indices[::-1]

    # Set up the figure - 3 columns per step (signal, spectrogram, relevance)
    fig, axes = plt.subplots(n_channels, n_steps + 1, figsize=(5 * (n_steps + 1), 3 * n_channels))
    fig.suptitle(f'Time-Frequency Window Flipping ({("Bottom->Top" if bottom_to_top else "Top->Bottom")})',
                 fontsize=16)

    # If only one channel, make axes 2D
    if n_channels == 1:
        axes = axes.reshape(1, -1)

    # Plot original signal for each channel
    for channel in range(n_channels):
        axes[channel, 0].plot(sample_np[channel], 'b-')
        axes[channel, 0].set_title(f'Original (Score: {original_score:.3f})')
        axes[channel, 0].set_ylim([sample_np[channel].min() - 0.1, sample_np[channel].max() + 0.1])

        if channel == 0:
            axes[channel, 0].set_ylabel('Amplitude')

        if channel == n_channels - 1:
            axes[channel, 0].set_xlabel('Time')

    # Iterate through steps
    for step in range(1, n_steps + 1):
        # Calculate number of frequency bands to flip
        n_freqs_to_flip = min(step * freq_per_step, n_freq)
        freq_to_flip = freq_indices[:n_freqs_to_flip]

        # Create flipped signal
        flipped_signal = sample_np.copy()
        flipped_mask = np.zeros_like(sample_np, dtype=bool)

        # For each frequency band to flip
        for freq_idx in freq_to_flip:
            # For each time frame
            for time_idx in range(n_time_frames):
                # Get corresponding time window
                start, end = time_windows[time_idx]

                # For each channel
                for channel in range(n_channels):
                    # Apply flipping to time window
                    flipped_signal[channel, start:end] = reference_value
                    flipped_mask[channel, start:end] = True

        # Get model prediction for flipped signal
        flipped_tensor = torch.tensor(flipped_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Calculate percentage of flipped points
        flipped_pct = np.sum(flipped_mask) / flipped_mask.size * 100.0

        # Plot the flipped signals
        for channel in range(n_channels):
            # Plot time domain signal with flipped regions
            axes[channel, step].plot(flipped_signal[channel], 'r-')
            axes[channel, step].set_title(f'{flipped_pct:.1f}% Flipped (Score: {score:.3f})')
            axes[channel, step].set_ylim([sample_np[channel].min() - 0.1, sample_np[channel].max() + 0.1])

            # Highlight flipped regions
            for time_idx in range(n_time_frames):
                start, end = time_windows[time_idx]
                if np.any(flipped_mask[channel, start:end]):
                    axes[channel, step].axvspan(start, end, alpha=0.2, color='gray')

            if channel == n_channels - 1:
                axes[channel, step].set_xlabel('Time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compare_tf_flipping_directions(model, sample, attribution_methods,
                                   n_steps=10, reference_value=0.0,
                                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compare bottom-to-top vs. top-to-bottom time-frequency flipping for different methods.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        attribution_methods: Dictionary of {method_name: (relevance_timefreq, signal_timefreq, freqs)}
        n_steps: Number of steps in flipping process
        reference_value: Value to replace flipped points
        device: Device to run computations on

    Returns:
        results_dict: Dictionary with results0 for each method and direction
    """
    results = {}

    # Create separate plots for bottom-to-top and top-to-bottom
    plt.figure(figsize=(12, 8))
    plt.title('Bottom-to-Top Frequency Flipping', fontsize=16)
    plt.xlabel('Percentage of Signal Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Process each attribution method with bottom-to-top flipping
    for method_name, (relevance_tf, signal_tf, freqs) in attribution_methods.items():
        print(f"Running bottom-to-top flipping for {method_name}...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Run bottom-to-top flipping
        scores_b2t, flipped_pcts_b2t = tf_window_flipping(
            model=model,
            sample=sample,
            relevance_timefreq=relevance_tf,
            signal_timefreq=signal_tf,
            freqs=freqs,
            n_steps=n_steps,
            bottom_to_top=True,
            reference_value=reference_value,
            device=device
        )

        # Calculate AUC
        auc_b2t = np.trapz(scores_b2t, flipped_pcts_b2t) / flipped_pcts_b2t[-1]

        # Store results0
        results[f"{method_name}_b2t"] = {
            "scores": scores_b2t,
            "flipped_pcts": flipped_pcts_b2t,
            "auc": auc_b2t
        }

        # Plot this method
        plt.plot(flipped_pcts_b2t, scores_b2t, 'o-', linewidth=2, markersize=6,
                 label=f"{method_name} (AUC: {auc_b2t:.4f})")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Create another plot for top-to-bottom flipping
    plt.figure(figsize=(12, 8))
    plt.title('Top-to-Bottom Frequency Flipping', fontsize=16)
    plt.xlabel('Percentage of Signal Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Process each attribution method with top-to-bottom flipping
    for method_name, (relevance_tf, signal_tf, freqs) in attribution_methods.items():
        print(f"Running top-to-bottom flipping for {method_name}...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Run top-to-bottom flipping
        scores_t2b, flipped_pcts_t2b = tf_window_flipping(
            model=model,
            sample=sample,
            relevance_timefreq=relevance_tf,
            signal_timefreq=signal_tf,
            freqs=freqs,
            n_steps=n_steps,
            bottom_to_top=False,
            reference_value=reference_value,
            device=device
        )

        # Calculate AUC
        auc_t2b = np.trapz(scores_t2b, flipped_pcts_t2b) / flipped_pcts_t2b[-1]

        # Store results0
        results[f"{method_name}_t2b"] = {
            "scores": scores_t2b,
            "flipped_pcts": flipped_pcts_t2b,
            "auc": auc_t2b
        }

        # Plot this method
        plt.plot(flipped_pcts_t2b, scores_t2b, 'o-', linewidth=2, markersize=6,
                 label=f"{method_name} (AUC: {auc_t2b:.4f})")

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print summary of results0
    print("\nTime-Frequency Flipping Results Summary:")
    print("-" * 70)
    print(f"{'Method':<20} {'Bottom-to-Top AUC':<20} {'Top-to-Bottom AUC':<20} {'Difference':<10}")
    print("-" * 70)

    for method_name, _ in attribution_methods.items():
        b2t_auc = results[f"{method_name}_b2t"]["auc"]
        t2b_auc = results[f"{method_name}_t2b"]["auc"]
        diff = b2t_auc - t2b_auc
        print(f"{method_name:<20} {b2t_auc:.4f}{' ':<14} {t2b_auc:.4f}{' ':<14} {diff:.4f}")

    return results


def selective_frequency_flipping(model, sample, relevance_timefreq, signal_timefreq, freqs,
                                 frequency_bands=[(0, 50), (50, 100), (100, 200)],
                                 reference_value=0.0,
                                 device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the importance of different frequency bands by selectively flipping them.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        relevance_timefreq: Time-frequency relevance map
        signal_timefreq: Time-frequency signal representation
        freqs: Frequency values for the spectrogram
        frequency_bands: List of (min_freq, max_freq) tuples defining frequency bands
        reference_value: Value to replace flipped points
        device: Device to run computations on

    Returns:
        band_results: Dictionary with results0 for each frequency band
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    sample_np = sample.detach().cpu().numpy()

    # Get shapes
    n_channels, time_steps = sample.shape
    n_freq, n_time_frames = relevance_timefreq.shape[1], relevance_timefreq.shape[2]

    # Map from spectrogram time frames to signal time points
    time_window_size = time_steps // n_time_frames if n_time_frames > 0 else 1
    time_windows = []
    for t in range(n_time_frames):
        start = t * time_window_size
        end = min((t + 1) * time_window_size, time_steps)
        time_windows.append((start, end))

    # Original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    # Results for each band
    band_results = {
        "original_score": original_score,
        "bands": {}
    }

    # Test each frequency band
    for band_idx, (min_freq, max_freq) in enumerate(frequency_bands):
        print(f"Testing frequency band {band_idx + 1}/{len(frequency_bands)}: {min_freq}-{max_freq} Hz")

        # Find frequency indices that fall within this band
        freq_indices = [i for i, f in enumerate(freqs) if min_freq <= f <= max_freq]

        if not freq_indices:
            print(f"  No frequencies found in band {min_freq}-{max_freq} Hz. Skipping.")
            continue

        # Create flipped signal by zeroing out this frequency band
        flipped_signal = sample_np.copy()
        flipped_mask = np.zeros_like(sample_np, dtype=bool)

        # For each frequency in the band
        for freq_idx in freq_indices:
            # For each time frame
            for time_idx in range(n_time_frames):
                # Get corresponding time window
                start, end = time_windows[time_idx]

                # For each channel
                for channel in range(n_channels):
                    # Apply flipping to time window
                    flipped_signal[channel, start:end] = reference_value
                    flipped_mask[channel, start:end] = True

        # Get model prediction for flipped signal
        flipped_tensor = torch.tensor(flipped_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Calculate percentage of flipped points
        flipped_pct = np.sum(flipped_mask) / flipped_mask.size * 100.0

        # Store results0
        band_results["bands"][f"{min_freq}-{max_freq}Hz"] = {
            "score": score,
            "flipped_pct": flipped_pct,
            "score_change": original_score - score,
            "freq_indices": freq_indices
        }

        print(f"  Score: {score:.4f} (Change: {original_score - score:.4f})")

    # Plot results0
    plt.figure(figsize=(10, 6))

    band_names = list(band_results["bands"].keys())
    score_changes = [band_results["bands"][band]["score_change"] for band in band_names]

    plt.bar(band_names, score_changes, color='skyblue')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Impact of Flipping Different Frequency Bands', fontsize=16)
    plt.xlabel('Frequency Band', fontsize=14)
    plt.ylabel('Score Change (Original - Flipped)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(score_changes):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

    return band_results


# Example usage
def run_tf_analysis(model, sample, relevance_timefreq, signal_timefreq, freqs,
                    method_name="DFT-LRP"):
    """
    Run a complete time-frequency flipping analysis for a single method.

    Args:
        model: Trained model
        sample: Time series input tensor
        relevance_timefreq: Time-frequency relevance map
        signal_timefreq: Time-frequency signal representation
        freqs: Frequency values for the spectrogram
        method_name: Name of the method being analyzed

    Returns:
        results0: Dictionary with analysis results0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # 1. Visualize time-frequency flipping progression
    print(f"1. Visualizing time-frequency flipping for {method_name}...")
    visualize_tf_flipping(
        model=model,
        sample=sample,
        relevance_timefreq=relevance_timefreq,
        signal_timefreq=signal_timefreq,
        freqs=freqs,
        n_steps=5,
        bottom_to_top=True,  # Start from lowest frequencies
        device=device
    )

    # 2. Run bottom-to-top flipping
    print(f"\n2. Running bottom-to-top frequency flipping for {method_name}...")
    scores_b2t, flipped_pcts_b2t = tf_window_flipping(
        model=model,
        sample=sample,
        relevance_timefreq=relevance_timefreq,
        signal_timefreq=signal_timefreq,
        freqs=freqs,
        n_steps=10,
        bottom_to_top=True,
        device=device
    )

    # Calculate AUC
    auc_b2t = np.trapz(scores_b2t, flipped_pcts_b2t) / flipped_pcts_b2t[-1]
    results["bottom_to_top"] = {
        "scores": scores_b2t,
        "flipped_pcts": flipped_pcts_b2t,
        "auc": auc_b2t
    }

    # Plot results0
    plt.figure(figsize=(10, 6))
    plt.plot(flipped_pcts_b2t, scores_b2t, 'o-', linewidth=2)
    plt.title(f'Bottom-to-Top Frequency Flipping - {method_name} (AUC: {auc_b2t:.4f})', fontsize=16)
    plt.xlabel('Percentage of Signal Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3. Run top-to-bottom flipping
    print(f"\n3. Running top-to-bottom frequency flipping for {method_name}...")
    scores_t2b, flipped_pcts_t2b = tf_window_flipping(
        model=model,
        sample=sample,
        relevance_timefreq=relevance_timefreq,
        signal_timefreq=signal_timefreq,
        freqs=freqs,
        n_steps=10,
        bottom_to_top=False,  # Start from highest frequencies
        device=device
    )

    # Calculate AUC
    auc_t2b = np.trapz(scores_t2b, flipped_pcts_t2b) / flipped_pcts_t2b[-1]
    results["top_to_bottom"] = {
        "scores": scores_t2b,
        "flipped_pcts": flipped_pcts_t2b,
        "auc": auc_t2b
    }

    # Plot results0
    plt.figure(figsize=(10, 6))
    plt.plot(flipped_pcts_t2b, scores_t2b, 'o-', linewidth=2)
    plt.title(f'Top-to-Bottom Frequency Flipping - {method_name} (AUC: {auc_t2b:.4f})', fontsize=16)
    plt.xlabel('Percentage of Signal Flipped (%)', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 4. Compare the approaches
    print(f"\n4. Comparing approaches for {method_name}:")
    print(f"Bottom-to-Top AUC: {auc_b2t:.4f}")
    print(f"Top-to-Bottom AUC: {auc_t2b:.4f}")
    print(f"Difference: {auc_b2t - auc_t2b:.4f}")

    # 5. Analyze specific frequency bands
    print("\n5. Analyzing specific frequency bands...")

    # Determine frequency bands based on the actual frequencies
    max_freq = max(freqs)
    frequency_bands = [
        (0, max_freq * 0.1),  # Low frequencies (0-10%)
        (max_freq * 0.1, max_freq * 0.3),  # Low-mid frequencies (10-30%)
        (max_freq * 0.3, max_freq * 0.7),  # Mid frequencies (30-70%)
        (max_freq * 0.7, max_freq)  # High frequencies (70-100%)
    ]

    band_results = selective_frequency_flipping(
        model=model,
        sample=sample,
        relevance_timefreq=relevance_timefreq,
        signal_timefreq=signal_timefreq,
        freqs=freqs,
        frequency_bands=frequency_bands,
        device=device
    )

    results["band_analysis"] = band_results

    return results


def direct_window_flipping(model, sample, attribution_method, n_steps=20, window_size=40,
                           most_relevant_first=True, reference_value=None,
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    A simpler approach that directly flips windows in the time domain based on their average relevance.

    Args:
        model: Trained model
        sample: Input sample (shape: [3, time_steps])
        attribution_method: Function to compute attributions in time domain
        n_steps: Number of flipping steps
        window_size: Size of each window to flip
        most_relevant_first: Whether to flip most relevant windows first
        reference_value: Value to replace flipped windows with (default: 0)
        device: Device to run on

    Returns:
        scores: Model scores at each step
        flipped_pcts: Percentage of signal flipped at each step
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Compute relevance in time domain
    relevance = attribution_method(model, sample)
    if isinstance(relevance, tuple):
        relevance = relevance[0]
    if not isinstance(relevance, torch.Tensor):
        relevance = torch.tensor(relevance, device=device)

    # Convert to numpy for processing
    sample_np = sample.detach().cpu().numpy()
    relevance_np = relevance.detach().cpu().numpy()

    # Get dimensions
    n_channels, time_steps = sample_np.shape

    # Calculate window importance
    n_windows = time_steps // window_size
    if time_steps % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for c in range(n_channels):
        for w in range(n_windows):
            start = w * window_size
            end = min((w + 1) * window_size, time_steps)
            # Average absolute relevance in this window
            window_importance[c, w] = np.mean(np.abs(relevance_np[c, start:end]))

    # Flatten and sort
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track results0
    scores = []
    flipped_pcts = []

    # Original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = torch.argmax(original_prob).item()
        original_score = original_prob[target_class].item()

    scores.append(original_score)
    flipped_pcts.append(0)

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Reference value
    if reference_value is None:
        reference_value = 0.0

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        print(f"Step {step}/{n_steps}")

        # How many windows to flip
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Create flipped signal
        flipped_signal = sample_np.copy()

        # Get window indices to flip
        indices_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert to channel, window indices
        channel_indices = indices_to_flip // n_windows
        window_indices = indices_to_flip % n_windows

        # Apply flipping
        for i in range(len(indices_to_flip)):
            c = channel_indices[i]
            w = window_indices[i]
            start = w * window_size
            end = min((w + 1) * window_size, time_steps)
            flipped_signal[c, start:end] = reference_value

        # Get model prediction
        flipped_tensor = torch.tensor(flipped_signal, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pct = n_windows_to_flip / total_windows * 100
        flipped_pcts.append(flipped_pct)

        print(f"  Flipped {flipped_pct:.2f}% of windows, score = {score:.4f}")

    return scores, flipped_pcts