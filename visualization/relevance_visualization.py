import scipy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
import scipy.signal

#Visualize time domain signal with relevance heatmap and relevance over time
def visualize_with_label_attribution(
        signal,
        attributions,
        label,
        method_name,
        cmap="bwr"):
    """
    Visualize signal with relevance heatmap and relevance over time in a 3x2 grid,
    including the label and average attribution for each axis over time.
    Args:
        signal: Original input signal (shape: (3, time_steps)).
        attributions: Importance values for explanation (shape: (3, time_steps)).
        label: True label for the sample (e.g., "Good" or "Bad").
        method_name: Name of the explanation method (e.g., "Integrated Gradients").
        cmap: Colormap for relevance (default: "bwr").
    """
    def calculate_average_attribution(attributions):
        """
        Calculate the average attribution for each axis.
        Args:
            attributions: Attribution values (shape: (3, time_steps)).
        Returns:
            A list of average attribution values for each axis.
        """
        averages = [np.mean(attr) for attr in attributions]
        return averages

    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()

    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()

    # Calculate average attribution for each axis
    avg_attributions = calculate_average_attribution(attributions)

    axes_labels = ["X", "Y", "Z"]
    n_axes = signal.shape[0]
    ncols = 2  # 2 columns: signal+heatmap, relevance
    nrows = n_axes
    figsize = (10, 6)  # Add extra space for column titles
    fig = plt.figure(figsize=figsize)

    # Create a gridspec with space for column titles
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.2] + [1] * nrows)

    # Define column titles
    column_titles = [
        "Time Domain Signal\nwith Relevance Heatmap",
        "Time Domain\nRelevance"
    ]

    # Add column titles
    for col, title in enumerate(column_titles):
        ax_title = fig.add_subplot(gs[0, col])
        ax_title.text(0.5, 0.5, title, ha='center', va='center', fontsize=10)
        ax_title.axis('off')

    # Create a grid of subplots with shared x-axes per column
    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if i == 0:  # First row
                ax[i, j] = fig.add_subplot(gs[i + 1, j])
            else:  # Share x-axis with the first row
                ax[i, j] = fig.add_subplot(gs[i + 1, j], sharex=ax[0, j])

    # Label text
    label_text = f"Label: {'Good' if label == 0 else 'Bad'}"

    for i in range(n_axes):  # Loop over axes: X, Y, Z
        time_steps = np.arange(signal[i].shape[0])

        # Find the maximum absolute value for the current axis, using percentile to avoid outliers
        max_abs_value = np.percentile(np.abs(attributions[i]), 99)
        print(f"Maximum Absolute Attribution in Axis {i}: {max_abs_value}")

        norm = plt.Normalize(vmin=-max_abs_value, vmax=max_abs_value)
        cmap_obj = colormaps[cmap]

        # Left column: Signal + Relevance Heatmap
        for t in range(len(time_steps) - 1):
            ax[i, 0].axvspan(time_steps[t], time_steps[t + 1], color=cmap_obj(norm(attributions[i][t])), alpha=0.5)

        ax[i, 0].plot(time_steps, signal[i], color="black", linewidth=0.8, label="Signal")
        if i == n_axes - 1:
            ax[i, 0].set_xlabel("Time Steps", fontsize=8)
        ax[i, 0].set_ylabel(f"{axes_labels[i]}\nSignal Value", fontsize=8)
        ax[i, 0].legend(fontsize=6, loc="upper right", prop={'size': 6}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 0].grid(visible=False)
        ax[i, 0].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 0].get_xticklabels(), visible=False)

        # Right column: Relevance over Time
        ax[i, 1].fill_between(range(len(attributions[i])), attributions[i], where=attributions[i] > 0, color='red',
                              alpha=0.5, label='Positive Relevance')
        ax[i, 1].fill_between(range(len(attributions[i])), attributions[i], where=attributions[i] < 0, color='blue',
                              alpha=0.5, label='Negative Relevance')
        if i == n_axes - 1:
            ax[i, 1].set_xlabel("Time Steps", fontsize=8)
        ax[i, 1].set_ylabel(f"{axes_labels[i]}\nRelevance Value", fontsize=8)
        ax[i, 1].legend(fontsize=6, loc="upper right", prop={'size': 6}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 1].grid(visible=False)
        ax[i, 1].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 1].get_xticklabels(), visible=False)

    fig.suptitle(f" Explanation for Time Domain Relevance\n with {method_name} - {label_text}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




# visualize DFT-XAI results for time and frequency domain
def visualize_xai_dft(
        relevance_time,
        relevance_freq,
        signal_freq,
        input_signal,
        freqs,
        predicted_label,
        axes_names=["X", "Y", "Z"],
        k_max=200,  # Maximum frequency in Hz
        signal_length=2000,
        sampling_rate=400,  # Sampling rate in Hz
        cmap="bwr",
        method=None# Colormap for relevance heatmap
):
    """
    Visualize LRP relevances in time and frequency domains for vibration data.

    Args:
        relevance_time: Numpy array with time-domain relevances
        relevance_freq: Numpy array with frequency-domain relevances
        signal_freq: Numpy array with frequency-domain signal
        input_signal: Numpy array with the input signal
        freqs: Frequency bins
        predicted_label: Predicted label (0 for "Good", 1 for "Bad")
        axes_names: Names of the axes (X, Y, Z)
        k_max: Maximum frequency to plot (in Hz)
        signal_length: Length of the signal
        sampling_rate: Sampling rate of the data in Hz
        cmap: Colormap for relevance heatmap
    """
    # Convert tensors to numpy if necessary
    if isinstance(relevance_time, torch.Tensor):
        relevance_time = relevance_time.detach().cpu().numpy()
    if isinstance(relevance_freq, torch.Tensor):
        relevance_freq = relevance_freq.detach().cpu().numpy()
    if isinstance(signal_freq, torch.Tensor):
        signal_freq = signal_freq.detach().cpu().numpy()
    if isinstance(input_signal, torch.Tensor):
        input_signal = input_signal.detach().cpu().numpy()
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.detach().cpu().numpy()

    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    ncols = 4  # 4 columns: signal+heatmap (time), relevance (time), signal+heatmap (freq), relevance (freq)
    nrows = n_axes
    figsize = (ncols * 5, nrows * 3 + 1)  # Add extra space for column titles
    fig = plt.figure(figsize=figsize, dpi=600)

    # Create a gridspec with space for column titles
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.2] + [1] * nrows)

    # Define column titles
    column_titles = [
        "Time Domain Signal\nwith Relevance Heatmap",
        "Time Domain\nRelevance",
        "Frequency Domain Signal\nwith Relevance Heatmap",
        "Frequency Domain\nRelevance"
    ]

    # Add column titles
    for col, title in enumerate(column_titles):
        ax_title = fig.add_subplot(gs[0, col])
        ax_title.text(0.5, 0.5, title, ha='center', va='center', fontsize=14)
        ax_title.axis('off')

    # Create a grid of subplots with shared x-axes per column
    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if i == 0:  # First row
                ax[i, j] = fig.add_subplot(gs[i + 1, j])
            else:  # Share x-axis with the first row
                ax[i, j] = fig.add_subplot(gs[i + 1, j], sharex=ax[0, j])

    def replace_positive(x, positive=True):
        """Replace positive (or negative if positive=False) values with zero."""
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for each axis to display
    avg_relevances_time = [np.mean(np.abs(rel)) for rel in relevance_time]

    # For frequency domain, use real part of relevance
    relevance_freq_real = relevance_freq.real if np.iscomplexobj(relevance_freq) else relevance_freq
    avg_relevances_freq = [np.mean(np.abs(rel)) for rel in relevance_freq_real]

    # Label text
    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    # Plot for each axis
    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        # Find the maximum absolute relevance for symmetric colormap
        max_abs_relevance_time = np.percentile(np.abs(relevance_time_axis), 99)
        norm_time = plt.Normalize(vmin=-max_abs_relevance_time, vmax=max_abs_relevance_time)
        cmap_obj = colormaps[cmap]

        # Plot heatmap as background
        for t in range(len(x_time) - 1):
            ax[i, 0].axvspan(x_time[t], x_time[t + 1], color=cmap_obj(norm_time(relevance_time_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 0].plot(x_time, signal_time_axis, color="black", linewidth=0.8, label="Signal")
        if i == n_axes - 1:
            ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel(f"{axes_names[i]}\nAmplitude", fontsize=12)
        ax[i, 0].grid(visible=False)
        ax[i, 0].legend(fontsize=6, loc="upper right", prop={'size': 10}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 0].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 0].get_xticklabels(), visible=False)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        if i == n_axes - 1:
            ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel(f"{axes_names[i]}\nRelevance", fontsize=12)
        ax[i, 1].grid(visible=False)
        ax[i, 1].legend(fontsize=6, loc="upper right", prop={'size': 10}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 1].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 1].get_xticklabels(), visible=False)

        # Frequency domain: Signal with Relevance Heatmap
        freq_range = (freqs >= 0) & (freqs <= k_max)
        x_freq = freqs[freq_range]
        signal_freq_axis = np.abs(signal_freq[i, :len(x_freq)])
        relevance_freq_real_axis = relevance_freq_real[i, :len(x_freq)]

        # Maximum magnitude for normalization
        max_abs_relevance_freq_mag = np.percentile(np.abs(relevance_freq_real_axis), 99)
        norm_freq_mag = plt.Normalize(vmin=-max_abs_relevance_freq_mag, vmax=max_abs_relevance_freq_mag)

        for t in range(len(x_freq) - 1):
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq_mag(relevance_freq_real_axis[t])),
                             alpha=0.5)

        # Plot signal on top
        ax[i, 2].plot(x_freq, signal_freq_axis, color="black", linewidth=0.8, label="Signal")
        if i == n_axes - 1:
            ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel(f"{axes_names[i]}\nMagnitude", fontsize=12)
        ax[i, 2].grid(visible=False)
        ax[i, 2].legend(fontsize=6, loc="upper right", prop={'size': 10}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 2].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 2].get_xticklabels(), visible=False)

        # Frequency domain: Relevance (using only real part)
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis), color="blue", label="Negative")
        if i == n_axes - 1:
            ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel(f"{axes_names[i]}\nRelevance", fontsize=12)
        ax[i, 3].grid(visible=False)
        ax[i, 3].legend(fontsize=6, loc="upper right", prop={'size': 10}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 3].tick_params(axis='both', which='major', labelsize=8)
        if i < n_axes - 1:
            plt.setp(ax[i, 3].get_xticklabels(), visible=False)

    fig.suptitle(f"{method} Explanation - {label_text}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




# visualize DFT-XAI results for time, frequency and time-frequency domain with spectrogram used for signal and STDFT-LRP for relevance
def visualize_hybrid_timefreq(
        relevance_time,
        relevance_freq,
        signal_freq,
        relevance_timefreq,
        signal_timefreq,
        input_signal,
        freqs,
        predicted_label,
        axes_names=["X", "Y", "Z"],
        k_max=200,  # Maximum frequency in Hz
        signal_length=2000,
        sampling_rate=400,  # Sampling rate in Hz
        cmap="bwr",
        method =None# Colormap for relevance heatmap
):
    """
    Hybrid visualization that uses:
    - EnhancedDFTLRP for time and frequency domain relevances
    - scipy's spectrogram for time-frequency signal visualization
    - DFT-LRP results0 for time-frequency relevance visualization

    Args:
        relevance_time: Numpy array with time-domain relevances
        relevance_freq: Numpy array with frequency-domain relevances
        signal_freq: Numpy array with frequency-domain signal
        relevance_timefreq: Numpy array with time-frequency relevances
        signal_timefreq: Numpy array with time-frequency signal (will be replaced with scipy spectrogram)
        input_signal: Numpy array with the input signal
        freqs: Frequency bins
        predicted_label: Predicted label (0 for "Good", 1 for "Bad")
        axes_names: Names of the axes (X, Y, Z)
        k_max: Maximum frequency to plot (in Hz)
        signal_length: Length of the signal
        sampling_rate: Sampling rate of the data in Hz
        cmap: Colormap for relevance heatmap
    """
    # Convert tensors to numpy if necessary
    if isinstance(relevance_time, torch.Tensor):
        relevance_time = relevance_time.detach().cpu().numpy()
    if isinstance(relevance_freq, torch.Tensor):
        relevance_freq = relevance_freq.detach().cpu().numpy()
    if isinstance(signal_freq, torch.Tensor):
        signal_freq = signal_freq.detach().cpu().numpy()
    if isinstance(input_signal, torch.Tensor):
        input_signal = input_signal.detach().cpu().numpy()
    if isinstance(freqs, torch.Tensor):
        freqs = freqs.detach().cpu().numpy()
    if isinstance(relevance_timefreq, torch.Tensor):
        relevance_timefreq = relevance_timefreq.detach().cpu().numpy()
    if isinstance(signal_timefreq, torch.Tensor):
        signal_timefreq = signal_timefreq.detach().cpu().numpy()

    # Calculate spectrograms using scipy for better visualization
    spectrograms = []
    for i in range(input_signal.shape[0]):
        # Compute signal spectrogram with scipy
        f, t, Sxx = scipy.signal.spectrogram(
            input_signal[i],
            fs=sampling_rate,
            nperseg=256,  # Standard window size
            noverlap=128,  # 50% overlap
            nfft=signal_length
        )
        spectrograms.append((f, t, Sxx))

    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    ncols = 6  # 6 columns: time signal, time relevance, freq signal, freq relevance, TF signal, TF relevance
    nrows = n_axes
    figsize = (ncols * 6, nrows * 4 + 1)  # Add extra space for column titles
    fig = plt.figure(figsize=figsize, dpi=450)

    # Create a gridspec with space for column titles
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[0.2] + [1] * nrows)

    # Define column titles
    column_titles = [
        "Time Domain Signal\nwith Relevance Heatmap",
        "Time Domain\nRelevance",
        "Frequency Domain Signal\nwith Relevance Heatmap",
        "Frequency Domain\nRelevance",
        "Time-Frequency\nSignal Spectrogram",
        "Time-Frequency\nRelevance Map"
    ]

    # Add column titles
    for col, title in enumerate(column_titles):
        ax_title = fig.add_subplot(gs[0, col])
        ax_title.text(0.5, 0.5, title, ha='center', va='center', fontsize=16)
        ax_title.axis('off')

    # Create a grid of subplots with shared x-axes per column
    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if i == 0:  # First row
                ax[i, j] = fig.add_subplot(gs[i + 1, j])
            else:  # Share x-axis with the first row
                ax[i, j] = fig.add_subplot(gs[i + 1, j], sharex=ax[0, j])

    def replace_positive(x, positive=True):
        """Replace positive (or negative if positive=False) values with zero."""
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for both domains
    def calculate_average_relevance(relevances):
        return [np.mean(np.abs(rel.real if np.iscomplexobj(rel) else rel)) for rel in relevances]

    avg_relevances_time = calculate_average_relevance(relevance_time)

    # For frequency domain, use real part if complex
    if np.iscomplexobj(relevance_freq):
        relevance_freq_for_avg = relevance_freq.real
    else:
        relevance_freq_for_avg = relevance_freq
    avg_relevances_freq = calculate_average_relevance(relevance_freq_for_avg)

    # Label text
    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    # Get frequency range and subset frequencies for plotting
    freq_range = (freqs >= 0) & (freqs <= k_max)
    freq_subset = freqs[freq_range]

    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        # Find maximum absolute relevance with percentile to avoid outliers
        max_abs_relevance_time = np.percentile(np.abs(relevance_time_axis), 99)
        norm_time = plt.Normalize(vmin=-max_abs_relevance_time, vmax=max_abs_relevance_time)
        cmap_obj = colormaps[cmap]

        # Plot heatmap as background
        for t in range(len(x_time) - 1):
            ax[i, 0].axvspan(x_time[t], x_time[t + 1], color=cmap_obj(norm_time(relevance_time_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 0].plot(x_time, signal_time_axis, color="black", linewidth=0.8, label="Signal")
        # Only show x-axis label on the bottom row
        if i == n_axes - 1:
            ax[i, 0].set_xlabel("Time (s)", fontsize=14)
        ax[i, 0].set_ylabel(f"{axes_names[i]}\nAmplitude", fontsize=14)

        # Remove x-tick labels for all but the bottom row
        if i < n_axes - 1:
            plt.setp(ax[i, 0].get_xticklabels(), visible=False)

        ax[i, 0].grid(visible=False)
        ax[i, 0].legend(fontsize=10, loc="upper right", prop={'size': 12}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 0].tick_params(axis='both', which='major', labelsize=12)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        if i == n_axes - 1:
            ax[i, 1].set_xlabel("Time (s)", fontsize=14)
        ax[i, 1].set_ylabel(f"{axes_names[i]}\nRelevance", fontsize=14)

        if i < n_axes - 1:
            plt.setp(ax[i, 1].get_xticklabels(), visible=False)

        ax[i, 1].grid(visible=False)
        ax[i, 1].legend(fontsize=10, loc="upper right", prop={'size': 12}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 1].tick_params(axis='both', which='major', labelsize=12)

        # Frequency domain: Signal with Relevance Heatmap
        x_freq = freq_subset
        try:
            # Handle potentially incompatible shapes
            freq_shape = signal_freq[i].shape[0] if signal_freq[i].ndim > 0 else len(signal_freq[i])
            usable_freq = min(len(x_freq), freq_shape)
            signal_freq_axis = np.abs(signal_freq[i, :usable_freq])

            # Check if relevance_freq is complex and extract real part if needed
            if np.iscomplexobj(relevance_freq[i]):
                relevance_freq_axis = relevance_freq[i, :usable_freq].real
            else:
                relevance_freq_axis = relevance_freq[i, :usable_freq]

            # Use percentile for normalization to handle outliers
            max_abs_relevance_freq = np.percentile(np.abs(relevance_freq_axis), 99)
            norm_freq = plt.Normalize(vmin=-max_abs_relevance_freq, vmax=max_abs_relevance_freq)

            # Plot frequency domain with relevance heatmap
            for t in range(len(x_freq[:usable_freq]) - 1):
                ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq(relevance_freq_axis[t])), alpha=0.5)

            # Plot signal on top
            ax[i, 2].plot(x_freq[:usable_freq], signal_freq_axis, color="black", linewidth=0.8, label="Signal")

            # Frequency domain: Relevance
            ax[i, 3].fill_between(x_freq[:usable_freq], replace_positive(relevance_freq_axis, positive=False),
                                  color="red", label="Positive")
            ax[i, 3].fill_between(x_freq[:usable_freq], replace_positive(relevance_freq_axis), color="blue",
                                  label="Negative")

        except Exception as e:
            print(f"Error plotting frequency domain for axis {i}: {e}")
            # Create empty plots
            ax[i, 2].text(0.5, 0.5, "Frequency Domain Error", ha="center", va="center")
            ax[i, 3].text(0.5, 0.5, "Frequency Domain Error", ha="center", va="center")

        # Set titles and labels for frequency domain
        if i == n_axes - 1:
            ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=14)
            ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=14)

        if i < n_axes - 1:
            plt.setp(ax[i, 2].get_xticklabels(), visible=False)
            plt.setp(ax[i, 3].get_xticklabels(), visible=False)

        ax[i, 2].set_ylabel(f"{axes_names[i]}\nMagnitude", fontsize=14)
        ax[i, 2].grid(visible=False)
        ax[i, 2].legend(fontsize=10, loc="upper right", prop={'size': 12}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 2].tick_params(axis='both', which='major', labelsize=12)

        ax[i, 3].set_ylabel(f"{axes_names[i]}\nRelevance", fontsize=14)
        ax[i, 3].grid(visible=False)
        ax[i, 3].legend(fontsize=10, loc="upper right", prop={'size': 12}, handlelength=1, handletextpad=0.5,
                        borderpad=0.3, frameon=True, framealpha=0.8)
        ax[i, 3].tick_params(axis='both', which='major', labelsize=12)

        # Time-frequency domain:
        # 1. Signal visualization: Use scipy's spectrogram
        f, t, Sxx = spectrograms[i]
        # Limit frequency to k_max
        f_mask = f <= k_max
        im1 = ax[i, 4].pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-10),
                                  shading='gouraud', cmap='viridis')
        if i == n_axes - 1:
            ax[i, 4].set_xlabel("Time (s)", fontsize=14)

        if i < n_axes - 1:
            plt.setp(ax[i, 4].get_xticklabels(), visible=False)

        ax[i, 4].set_ylabel(f"{axes_names[i]}\nFrequency (Hz)", fontsize=14)
        ax[i, 4].set_ylim(0, k_max)
        plt.colorbar(im1, ax=ax[i, 4], label="Power/dB")

        # 2. Relevance visualization: Use DFT-LRP relevance_timefreq if available
        try:
            if relevance_timefreq is not None and not np.all(relevance_timefreq[i] == 0):
                # Get the actual number of frames
                n_frames = relevance_timefreq.shape[2]
                total_time = signal_length / sampling_rate
                time_steps = np.linspace(0, total_time, n_frames)

                # Use freq_subset for proper frequency indexing
                freq_subset_len = min(len(freq_subset), relevance_timefreq.shape[1])

                # Relevance time-frequency map - use real part if complex
                if np.iscomplexobj(relevance_timefreq):
                    rel_data = relevance_timefreq[i, :freq_subset_len, :].real
                else:
                    rel_data = relevance_timefreq[i, :freq_subset_len, :]

                # Set symmetric color scale based on percentile to avoid outliers
                max_abs = np.percentile(np.abs(rel_data), 99)

                im2 = ax[i, 5].pcolormesh(
                    time_steps,
                    freq_subset[:freq_subset_len],
                    rel_data,
                    shading='gouraud',
                    cmap='coolwarm',
                    vmin=-max_abs,
                    vmax=max_abs
                )
                if i == n_axes - 1:
                    ax[i, 5].set_xlabel("Time (s)", fontsize=14)

                if i < n_axes - 1:
                    plt.setp(ax[i, 5].get_xticklabels(), visible=False)

                ax[i, 5].set_ylabel(f"{axes_names[i]}\nFrequency (Hz)", fontsize=14)
                ax[i, 5].grid(visible=False)
                ax[i, 5].set_ylim(0, k_max)
                plt.colorbar(im2, ax=ax[i, 5], label="Relevance")
            else:
                # Fallback: Create relevance spectrograms by weighting signal with relevance
                weighted_signal = input_signal[i] * relevance_time[i]
                f_rel, t_rel, Sxx_rel = scipy.signal.spectrogram(
                    weighted_signal, fs=sampling_rate, nperseg=256, noverlap=128
                )
                # Use relevance sign to color the spectrogram
                im2 = ax[i, 5].pcolormesh(
                    t_rel,
                    f_rel[f_mask],
                    np.sign(weighted_signal.mean()) * Sxx_rel[f_mask],
                    shading='gouraud',
                    cmap='coolwarm'
                )
                if i == n_axes - 1:
                    ax[i, 5].set_xlabel("Time (s)", fontsize=14)

                if i < n_axes - 1:
                    plt.setp(ax[i, 5].get_xticklabels(), visible=False)

                ax[i, 5].set_ylabel(f"{axes_names[i]}\nFrequency (Hz)", fontsize=14)
                ax[i, 5].set_ylim(0, k_max)
                plt.colorbar(im2, ax=ax[i, 5], label="Weighted Relevance")
        except Exception as e:
            print(f"Error plotting time-frequency relevance for axis {i}: {e}")
            # Fallback: Create relevance spectrograms by weighting signal with relevance
            try:
                weighted_signal = input_signal[i] * relevance_time[i]
                f_rel, t_rel, Sxx_rel = scipy.signal.spectrogram(
                    weighted_signal, fs=sampling_rate, nperseg=256, noverlap=128
                )
                # Use absolute value to show relevance magnitude
                im2 = ax[i, 5].pcolormesh(t_rel, f_rel[f_mask], np.abs(Sxx_rel[f_mask]),
                                          shading='gouraud', cmap='coolwarm')
                if i == n_axes - 1:
                    ax[i, 5].set_xlabel("Time (s)", fontsize=14)

                if i < n_axes - 1:
                    plt.setp(ax[i, 5].get_xticklabels(), visible=False)

                ax[i, 5].set_ylabel(f"{axes_names[i]}\nFrequency (Hz)", fontsize=14)
                ax[i, 5].set_ylim(0, k_max)
                plt.colorbar(im2, ax=ax[i, 5], label="Relevance")
            except Exception as e2:
                print(f"Error creating direct relevance spectrogram: {e2}")
                ax[i, 5].text(0.5, 0.5, "Time-Frequency Relevance Error", ha="center", va="center")

    fig.suptitle(f"{method} Explanation - {label_text}", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
