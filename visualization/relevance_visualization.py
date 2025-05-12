import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps

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
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))  # 3 rows, 2 columns
    label_text = f"Label: {'Good' if label == 0 else 'Bad'}"



    for i in range(3):  # Loop over axes: X, Y, Z
        time_steps = np.arange(signal[i].shape[0])


        # Find the maximum absolute value for the current axis
        max_abs_value = np.max(np.abs(attributions[i]))
        print(f" Maximum Absolute Attribution in Axis {i}: {max_abs_value}")

        # Map attributions to colors using bwr colormap, scaling between -max_abs_value and +max_abs_value
        norm = plt.Normalize(vmin=-max_abs_value, vmax=max_abs_value)
        cmap = colormaps['bwr']

        # Normalize using -max_abs_value to +max_abs_value to keep zero as white
        '''relevance_colors = plt.cm.get_cmap(cmap)((attributions[i] + max_abs_value) / (2 * max_abs_value))



        # Left column: Signal + Relevance Heatmap
        relevance_colors = plt.cm.get_cmap(cmap)((attributions[i] - np.min(attributions[i])) /
                                                 (np.max(attributions[i]) - np.min(attributions[i])))  # Map to colormap'''

        for t in range(len(time_steps) - 1):
            axs[i, 0].axvspan(time_steps[t], time_steps[t + 1],  color=cmap(norm(attributions[i][t])), alpha=0.5)

        axs[i, 0].plot(time_steps, signal[i], color="black", linewidth=0.8, label="Signal")  # Thinner signal line
        axs[i, 0].set_title(f"{method_name} Heatmap for {axes_labels[i]}-Axis\n{label_text}, Average Attribution{avg_attributions[i]:.4f}")
        axs[i, 0].set_xlabel("Time Steps")
        axs[i, 0].set_ylabel("Signal Value")
        axs[i, 0].legend()

        # Right column: Relevance over Time
        axs[i, 1].bar(time_steps, attributions[i], color=["red" if val > 0 else "blue" for val in attributions[i]],
                      alpha=0.8, width=1.0)
        axs[i, 1].set_title(f"{method_name} Relevance Over Time for {axes_labels[i]}-Axis\n{label_text}")
        axs[i, 1].set_xlabel("Time Steps")
        axs[i, 1].set_ylabel("Relevance Value")

    fig.suptitle(f"Explanation for {method_name} - {label_text}", fontsize=16)  # Add overall title with label
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.show()


# 8️⃣ Visualize LRP Relevances for a Single Sample
# ------------------------
def visualize_lrp_single_sample(
        signal,
        relevance,
        label,
        sample_idx=0,
        axes_names=["X", "Y", "Z"]):
    """
    Visualize a single sample's signal and LRP relevance for each axis.

    Args:
        signal: Numpy array of shape (3, 10000) for the time series
        relevance: Numpy array of shape (3, 10000) for LRP relevances
        label: Integer label (0 or 1) for the sample
        sample_idx: Index of the sample (for title, defaults to 0 for single sample)
        axes_names: List of axis names for labeling
    """
    n_axes = signal.shape[0]
    fig, axs = plt.subplots(n_axes, 2, figsize=(12, 4 * n_axes))

    for i in range(n_axes):
        # Plot signal
        axs[i, 0].plot(signal[i], label=f"Signal ({axes_names[i]})")
        axs[i, 0].set_title(f"Signal - Axis {axes_names[i]} (Sample {sample_idx}, Label: {'Good' if label == 0 else 'Bad'})")
        axs[i, 0].set_xlabel("Time Step")
        axs[i, 0].set_ylabel("Amplitude")
        axs[i, 0].legend()

        # Plot relevance (positive in blue, negative in red)
        axs[i, 1].fill_between(range(len(relevance[i])), relevance[i], where=relevance[i] > 0, color='red', alpha=0.5, label='Positive Relevance')
        axs[i, 1].fill_between(range(len(relevance[i])), relevance[i], where=relevance[i] < 0, color='blue', alpha=0.5, label='Negative Relevance')
        axs[i, 1].set_title(f"LRP Relevance - Axis {axes_names[i]}")
        axs[i, 1].set_xlabel("Time Step")
        axs[i, 1].set_ylabel("Relevance")
        axs[i, 1].legend()

    plt.tight_layout()
    plt.show()

# 6️⃣ Visualize LRP Relevances in Time and Frequency Domains

def visualize_lrp_dft(
    relevance_time,
    relevance_freq,
    signal_freq,
    input_signal,
    freqs,
    predicted_label,
    axes_names=["X", "Y", "Z"],
    k_max=1000,  # Maximum frequency in Hz
    signal_length=2000,
    sampling_rate=400,  # Sampling rate in Hz
    cmap="bwr"  # Colormap for relevance heatmap
):
    """
    Visualize LRP relevances in time and frequency domains for each axis with relevance heatmaps
    for both time-domain and frequency-domain signals.

    Args:
        relevance_time: Numpy array of shape (3, signal_length) with time-domain relevances
        relevance_freq: Numpy array of shape (3, freq_bins) with frequency-domain relevances
        signal_freq: Numpy array of shape (3, freq_bins) with frequency-domain signal magnitudes
        input_signal: Numpy array of shape (3, signal_length) with the input signal
        freqs: Frequency bins (length freq_bins)
        predicted_label: Predicted or true label (0 for "Good", 1 for "Bad")
        axes_names: Names of the axes (X, Y, Z)
        k_max: Maximum frequency to plot (in Hz)
        signal_length: Length of the signal
        sampling_rate: Sampling rate of the data in Hz
        cmap: Colormap for relevance heatmap (default: "bwr")
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
    nrows, ncols = n_axes, 4  # 4 columns: signal+heatmap (time), relevance (time), signal+heatmap (freq), relevance (freq)
    figsize = (ncols * 6, nrows * 5)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    def replace_positive(x, positive=True):
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for both domains to display
    def calculate_average_relevance(relevances):
        return [np.mean(np.abs(rel)) for rel in relevances]

    avg_relevances_time = calculate_average_relevance(relevance_time)
    avg_relevances_freq = calculate_average_relevance(relevance_freq)

    # Label text
    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    # Plot for each axis
    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        # Find the maximum absolute relevance for symmetric colormap
        max_abs_relevance_time = np.max(np.abs(relevance_time_axis))
        norm_time = plt.Normalize(vmin=-max_abs_relevance_time, vmax=max_abs_relevance_time)
        cmap_obj = colormaps[cmap]

        # Plot heatmap as background
        for t in range(len(x_time) - 1):
            ax[i, 0].axvspan(x_time[t], x_time[t + 1], color=cmap_obj(norm_time(relevance_time_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 0].plot(x_time, signal_time_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(
            f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}",
            fontsize=14
        )
        ax[i, 0].legend(fontsize=10, loc="upper right")
        ax[i, 0].grid(True)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red", label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel("Relevance", fontsize=12)
        ax[i, 1].set_title(f"LRP Relevance (Time) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 1].legend(fontsize=10, loc="upper right")
        ax[i, 1].grid(True)

        # Frequency domain: Signal with Relevance Heatmap
        freq_range = (freqs >= 0) & (freqs <= k_max)
        x_freq = freqs[freq_range]
        signal_freq_axis = np.abs(signal_freq[i, :len(x_freq)])
        relevance_freq_axis = relevance_freq[i, :len(x_freq)]

        # Find the maximum absolute relevance for symmetric colormap
        max_abs_relevance_freq = np.max(np.abs(relevance_freq_axis))
        norm_freq = plt.Normalize(vmin=-max_abs_relevance_freq, vmax=max_abs_relevance_freq)

        # Plot heatmap as background
        for t in range(len(x_freq) - 1):
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq(relevance_freq_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 2].plot(x_freq, signal_freq_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel("Magnitude", fontsize=12)
        ax[i, 2].set_title(
            f"Signal with LRP Heatmap (Freq) - Axis {axes_names[i]}\nAvg Relevance: {avg_relevances_freq[i]:.4f}",
            fontsize=14
        )
        ax[i, 2].legend(fontsize=10, loc="upper right")
        ax[i, 2].grid(True)

        # Frequency domain: Relevance
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_axis, positive=False), color="red", label="Positive")
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_axis), color="blue", label="Negative")
        ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel("Relevance", fontsize=12)
        ax[i, 3].set_title(f"LRP Relevance (Freq) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 3].legend(fontsize=10, loc="upper right")
        ax[i, 3].grid(True)

    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# 7️⃣ Visualize LRP Relevances in Time and Frequency Domains

def visualize_lrp_fft(
    relevance_time,
    relevance_freq,
    signal_freq,
    relevance_timefreq,
    signal_timefreq,
    input_signal,
    freqs,
    predicted_label,
    axes_names=["X", "Y", "Z"],
    k_max=200,
    signal_length=2000,
    sampling_rate=400,
    cmap="bwr"  # Colormap for relevance heatmaps
):
    """
    Visualize LRP relevances in time and frequency domains with heatmaps for time and frequency signals.

    Args:
        relevance_time: Numpy array of shape (3, signal_length) with time-domain relevances
        relevance_freq: Numpy array of shape (3, freq_bins) with frequency-domain relevances
        signal_freq: Numpy array of shape (3, freq_bins) with frequency-domain signal magnitudes
        relevance_timefreq: Numpy array of shape (3, freq_bins, time_steps) with time-frequency relevances
        signal_timefreq: Numpy array of shape (3, freq_bins, time_steps) with time-frequency signal
        input_signal: Numpy array of shape (3, signal_length) with the input signal
        freqs: Frequency bins (length freq_bins)
        predicted_label: Predicted or true label (0 for "Good", 1 for "Bad")
        axes_names: Names of the axes (X, Y, Z)
        k_max: Maximum frequency to plot (in Hz)
        signal_length: Length of the signal
        sampling_rate: Sampling rate of the data in Hz
        cmap: Colormap for relevance heatmaps (default: "bwr")
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
    if relevance_timefreq is not None and isinstance(relevance_timefreq, torch.Tensor):
        relevance_timefreq = relevance_timefreq.detach().cpu().numpy()
    if signal_timefreq is not None and isinstance(signal_timefreq, torch.Tensor):
        signal_timefreq = signal_timefreq.detach().cpu().numpy()

    n_axes = input_signal.shape[0]
    ncols = 6 if signal_timefreq is not None else 4
    nrows = n_axes
    figsize = (ncols * 6, nrows * 5)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    def replace_positive(x, positive=True):  ##replaces positive(or negative if positive=false) values with zero
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for both domains to display
    def calculate_average_relevance(relevances):
        return [np.mean(np.abs(rel.real if np.iscomplexobj(rel) else rel)) for rel in relevances]

    avg_relevances_time = calculate_average_relevance(relevance_time)
    avg_relevances_freq = calculate_average_relevance(relevance_freq)

    # Label text
    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        # Find the maximum absolute relevance for symmetric colormap
        max_abs_relevance_time = np.max(np.abs(relevance_time_axis))
        norm_time = plt.Normalize(vmin=-max_abs_relevance_time, vmax=max_abs_relevance_time)
        cmap_obj = colormaps[cmap]

        # Plot heatmap as background
        for t in range(len(x_time) - 1):
            ax[i, 0].axvspan(x_time[t], x_time[t + 1], color=cmap_obj(norm_time(relevance_time_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 0].plot(x_time, signal_time_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(
            f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}",
            fontsize=14
        )
        ax[i, 0].legend(fontsize=10, loc="upper right")
        ax[i, 0].grid(True)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red", label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel("Relevance", fontsize=12)
        ax[i, 1].set_title(f"LRP Relevance (Time) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 1].legend(fontsize=10, loc="upper right")
        ax[i, 1].grid(True)

        # Frequency domain: Signal with Relevance Heatmap
        freq_range = (freqs >= 0) & (freqs <= k_max)
        x_freq = freqs[freq_range]
        signal_freq_axis = np.abs(signal_freq[i, :len(x_freq)])
        relevance_freq_axis = relevance_freq[i, :len(x_freq)].real  # Use real part for signed relevance

        # Find the maximum absolute relevance for symmetric colormap
        max_abs_relevance_freq = np.max(np.abs(relevance_freq_axis))
        norm_freq = plt.Normalize(vmin=-max_abs_relevance_freq, vmax=max_abs_relevance_freq)

        # Plot heatmap as background
        for t in range(len(x_freq) - 1):
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq(relevance_freq_axis[t])), alpha=0.5)

        # Plot signal on top
        ax[i, 2].plot(x_freq, signal_freq_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel("Magnitude", fontsize=12)
        ax[i, 2].set_title(
            f"Signal with LRP Heatmap (Freq) - Axis {axes_names[i]}\nAvg Relevance: {avg_relevances_freq[i]:.4f}",
            fontsize=14
        )
        ax[i, 2].legend(fontsize=10, loc="upper right")
        ax[i, 2].grid(True)

        # Frequency domain: Relevance
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_axis, positive=False), color="red", label="Positive")
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_axis), color="blue", label="Negative")
        ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel("Relevance", fontsize=12)
        ax[i, 3].set_title(f"LRP Relevance (Freq) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 3].legend(fontsize=10, loc="upper right")
        ax[i, 3].grid(True)

        if signal_timefreq is not None:
            total_time = signal_length / sampling_rate
            time_steps = np.linspace(0, total_time, 20)  # 20 frames
            freq_subset = freqs[freq_range]

            im1 = ax[i, 4].imshow(
                np.abs(signal_timefreq[i, :len(freq_subset), :].T),
                aspect="auto",
                origin="lower",
                extent=[time_steps[0], time_steps[-1], 0, k_max],
                cmap='viridis'
            )
            ax[i, 4].set_xlabel("Time (s)", fontsize=12)
            ax[i, 4].set_ylabel("Frequency (Hz)", fontsize=12)
            ax[i, 4].set_title(f"Signal (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
            ax[i, 4].grid(True)
            plt.colorbar(im1, ax=ax[i, 4], label="Magnitude")

            im2 = ax[i, 5].imshow(
                relevance_timefreq[i, :len(freq_subset), :].real.T,  # Use real part for signed heatmap
                aspect="auto",
                origin="lower",
                extent=[0, total_time, 0, k_max],
                cmap='coolwarm',
                vmin=-np.max(np.abs(relevance_timefreq[i].real)),
                vmax=np.max(np.abs(relevance_timefreq[i].real))
            )
            ax[i, 5].set_xlabel("Time (s)", fontsize=12)
            ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
            ax[i, 5].set_title(f"LRP Relevance (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
            ax[i, 5].grid(True)
            plt.colorbar(im2, ax=ax[i, 5], label="Relevance")

    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()