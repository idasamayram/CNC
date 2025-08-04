import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
import scipy

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

def visualize_lrp_dft_old(
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
        relevance_freq: Numpy array of shape (3, freq_bins) with frequency-domain relevances (can be complex)
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
        """
        Replace positive (or negative if positive=False) values with zero in the array.
        """
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for both domains to display
    def calculate_average_relevance(relevances):
        return [np.mean(np.abs(rel)) for rel in relevances]

    avg_relevances_time = calculate_average_relevance(relevance_time)
    # For frequency, use magnitude for average relevance

    relevance_freq_real = relevance_freq.real

    '''
    # use this if you want to use the magnitude of the complex relevance not only real part
    if np.iscomplexobj(relevance_freq):
        relevance_freq_real = relevance_freq.real
        relevance_freq_magnitude = np.abs(relevance_freq)
    else:
        relevance_freq_real = relevance_freq
        relevance_freq_magnitude = np.abs(relevance_freq)
        '''

    avg_relevances_freq = calculate_average_relevance(relevance_freq_real)

    # Label text
    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    # Plot for each axis
    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        # Find the maximum absolute relevance for symmetric colormap
        # max_abs_relevance_time = np.max(np.abs(relevance_time_axis))
        max_abs_relevance_time = np.percentile(np.abs(relevance_time_axis), 99)

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
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red", label="Positive") #replaces negative values with zero => only positive values are shown in red
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative") #replaces positive values with zero => only negative values are shown in blue
        ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel("Relevance", fontsize=12)
        ax[i, 1].set_title(f"LRP Relevance (Time) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 1].legend(fontsize=10, loc="upper right")
        ax[i, 1].grid(True)

        # Frequency domain: Signal with Relevance Heatmap
        freq_range = (freqs >= 0) & (freqs <= k_max)
        x_freq = freqs[freq_range]
        signal_freq_axis = np.abs(signal_freq[i, :len(x_freq)])
        relevance_freq_real_axis = relevance_freq_real[i, :len(x_freq)]

        '''
        relevance_freq_mag_axis = relevance_freq_magnitude[i, :len(x_freq)]
        relevance_freq_mag_sign_axis = np.sign(relevance_freq_real[i, :len(x_freq)]) * relevance_freq_magnitude[i,:len(x_freq)]
        '''

        # Maximum magnitude for normalization (intensity)
        # max_abs_relevance_freq_mag = np.max(relevance_freq_mag_axis) # for absolute magnitude using complex relevance
        max_abs_relevance_freq_mag = np.percentile(np.abs(relevance_freq_real_axis), 99) # to avoid outliers using percentile

        # max_abs_relevance_freq_mag = np.max(relevance_freq_real_axis)


        norm_freq_mag = plt.Normalize(vmin=-max_abs_relevance_freq_mag, vmax=max_abs_relevance_freq_mag)
        # For color (red/blue), use sign of real part
        # For intensity, use magnitude

        for t in range(len(x_freq) - 1):
            # Choose color by sign of real part, intensity by normalized magnitude
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq_mag(relevance_freq_real_axis[t])), alpha=0.5)

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

        # Frequency domain: Relevance (using only real part here )
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis, positive=False), color="red", label="Positive")
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis), color="blue", label="Negative")
        ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel("Relevance", fontsize=12)
        ax[i, 3].set_title(f"LRP Relevance (Freq) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 3].legend(fontsize=10, loc="upper right")
        ax[i, 3].grid(True)


    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# a bit cleaned up version, best so far for time and freq domain
def visualize_lrp_dft(
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
        cmap="bwr"  # Colormap for relevance heatmap
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
    nrows, ncols = n_axes, 4  # 4 columns: signal+heatmap (time), relevance (time), signal+heatmap (freq), relevance (freq)
    figsize = (ncols * 5, nrows * 4)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

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
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(
            f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}",
            fontsize=14
        )
        ax[i, 0].legend(fontsize=10, loc="upper right")
        ax[i, 0].grid(True)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red",
                              label="Positive")
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
        relevance_freq_real_axis = relevance_freq_real[i, :len(x_freq)]

        # Maximum magnitude for normalization
        max_abs_relevance_freq_mag = np.percentile(np.abs(relevance_freq_real_axis), 99)
        norm_freq_mag = plt.Normalize(vmin=-max_abs_relevance_freq_mag, vmax=max_abs_relevance_freq_mag)

        for t in range(len(x_freq) - 1):
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq_mag(relevance_freq_real_axis[t])),
                             alpha=0.5)

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

        # Frequency domain: Relevance (using only real part)
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 3].fill_between(x_freq, replace_positive(relevance_freq_real_axis), color="blue", label="Negative")
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
            num_frames = signal_timefreq.shape[-1]  # Use actual number of frames
            time_steps = np.linspace(0, total_time, num_frames)
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
                relevance_timefreq[i, :len(freq_subset), :].real.T,
                aspect="auto",
                origin="lower",
                extent=[time_steps[0], time_steps[-1], 0, k_max],
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

# 9️⃣ Visualize LRP Relevances in Time and Frequency Domains

def visualize_lrp_dft_extended(
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
    cmap="bwr"
):
    """
    Visualize LRP relevances in time, frequency, and time-frequency domains.
    
    Args:
        relevance_time: Numpy array of shape (3, signal_length) with time-domain relevances
        relevance_freq: Numpy array of shape (3, freq_bins) with frequency-domain relevances
        signal_freq: Numpy array of shape (3, freq_bins) with frequency-domain signal
        relevance_timefreq: Numpy array of shape (3, freq_bins, time_steps) with time-frequency relevances or None
        signal_timefreq: Numpy array of shape (3, freq_bins, time_steps) with time-frequency signal or None
        input_signal: Numpy array of shape (3, signal_length) with the input signal
        freqs: Frequency bins (for visualization)
        predicted_label: Predicted or true label (0 for "Good", 1 for "Bad")
        axes_names: Names of the axes (X, Y, Z)
        k_max: Maximum frequency to plot (in Hz)
        signal_length: Length of the signal
        sampling_rate: Sampling rate of the data in Hz
        cmap: Colormap for relevance heatmaps (default: "bwr")
    """
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

    # Check if time-frequency data is available
    has_timefreq = signal_timefreq is not None and relevance_timefreq is not None
    
    n_axes = input_signal.shape[0]
    ncols = 6 if has_timefreq else 4
    nrows = n_axes
    figsize = (ncols * 6, nrows * 5)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    def replace_positive(x, positive=True):
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    def calculate_average_relevance(relevances):
        return [np.mean(np.abs(rel.real if np.iscomplexobj(rel) else rel)) for rel in relevances]

    avg_relevances_time = calculate_average_relevance(relevance_time)
    avg_relevances_freq = calculate_average_relevance(relevance_freq)

    label_text = f"Label: {'Good' if predicted_label == 0 else 'Bad'}"

    # Get frequency range and subset frequencies for plotting
    freq_range = (freqs >= 0) & (freqs <= k_max)
    freq_subset = freqs[freq_range]

    for i in range(n_axes):
        # Time domain: Signal with Relevance Heatmap
        x_time = np.linspace(0, signal_length / sampling_rate, signal_length)
        signal_time_axis = input_signal[i]
        relevance_time_axis = relevance_time[i]

        max_abs_relevance_time = np.max(np.abs(relevance_time_axis))
        norm_time = plt.Normalize(vmin=-max_abs_relevance_time, vmax=max_abs_relevance_time)
        cmap_obj = colormaps[cmap]

        for t in range(len(x_time) - 1):
            ax[i, 0].axvspan(x_time[t], x_time[t + 1], color=cmap_obj(norm_time(relevance_time_axis[t])), alpha=0.5)
        ax[i, 0].plot(x_time, signal_time_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}", fontsize=14)
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
        x_freq = freq_subset
        signal_freq_axis = np.abs(signal_freq[i, :len(x_freq)])
        
        # Check if relevance_freq is complex and extract real part if needed
        if np.iscomplexobj(relevance_freq[i]):
            relevance_freq_axis = relevance_freq[i, :len(x_freq)].real
        else:
            relevance_freq_axis = relevance_freq[i, :len(x_freq)]

        max_abs_relevance_freq = np.max(np.abs(relevance_freq_axis))
        norm_freq = plt.Normalize(vmin=-max_abs_relevance_freq, vmax=max_abs_relevance_freq)

        for t in range(len(x_freq) - 1):
            ax[i, 2].axvspan(x_freq[t], x_freq[t + 1], color=cmap_obj(norm_freq(relevance_freq_axis[t])), alpha=0.5)
        ax[i, 2].plot(x_freq, signal_freq_axis, color="black", linewidth=0.8, label="Signal")
        ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel("Magnitude", fontsize=12)
        ax[i, 2].set_title(f"Signal with LRP Heatmap (Freq) - Axis {axes_names[i]}\nAvg Relevance: {avg_relevances_freq[i]:.4f}", fontsize=14)
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

        # Add time-frequency visualizations if available
        if has_timefreq:
            # Try to get the actual number of time frames from the data
            if signal_timefreq[i].shape[-1] > 0:
                n_frames = signal_timefreq[i].shape[-1]
                total_time = signal_length / sampling_rate
                time_steps = np.linspace(0, total_time, n_frames)
                
                # Plot signal time-frequency representation
                try:
                    # Use freq_subset length for proper frequency indexing
                    freq_subset_len = len(freq_subset)
                    
                    im1 = ax[i, 4].imshow(
                        np.abs(signal_timefreq[i, :freq_subset_len, :].T),
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
                    
                    # Extract real part if complex
                    rel_data = relevance_timefreq[i, :freq_subset_len, :].real if np.iscomplexobj(relevance_timefreq[i]) else relevance_timefreq[i, :freq_subset_len, :]
                    
                    # Set symmetric color scale based on maximum absolute value
                    max_abs = np.max(np.abs(rel_data))
                    
                    im2 = ax[i, 5].imshow(
                        rel_data.T,
                        aspect="auto",
                        origin="lower",
                        extent=[time_steps[0], time_steps[-1], 0, k_max],
                        cmap='coolwarm',
                        vmin=-max_abs,
                        vmax=max_abs
                    )
                    ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                    ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                    ax[i, 5].set_title(f"LRP Relevance (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
                    ax[i, 5].grid(True)
                    plt.colorbar(im2, ax=ax[i, 5], label="Relevance")
                    
                except Exception as e:
                    print(f"Error plotting time-frequency data for axis {i}: {e}")
                    # Create empty plots with error message
                    ax[i, 4].text(0.5, 0.5, "Time-Frequency Data Error", ha="center", va="center")
                    ax[i, 5].text(0.5, 0.5, "Time-Frequency Data Error", ha="center", va="center")

    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



# best functiopn to visualize time-freq signal and relevance  with DFTLRP so far, uses dft-LRP time-frequency data if available, otherwise computes spectrograms directly
def visualize_lrp_with_spectrograms(
        relevance_time,
        relevance_freq,
        signal_freq,
        relevance_timefreq,
        signal_timefreq,
        input_signal,
        freqs,
        predicted_label,
        axes_names=["X", "Y", "Z"],
        k_max=500,  # Maximum frequency in Hz
        signal_length=2000,
        sampling_rate=400,  # Sampling rate in Hz
        cmap="bwr"  # Colormap for relevance heatmap
):
    """
    Visualize LRP relevances in time, frequency, and time-frequency domains
    with proper spectrograms for the time-frequency domain.
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

    # Check if time-frequency data is available and valid
    has_timefreq = (signal_timefreq is not None and
                    relevance_timefreq is not None and
                    not np.all(signal_timefreq == 0))

    if not has_timefreq:
        # If DFT-LRP time-frequency data is unavailable, compute spectrograms directly
        print("DFT-LRP time-frequency data not available. Computing spectrograms directly.")
        has_spectrograms = True

        # Calculate spectrograms for each axis using scipy
        spectrograms = []
        relevance_spectrograms = []

        for i in range(input_signal.shape[0]):
            # Compute signal spectrogram
            f, t, Sxx = scipy.signal.spectrogram(
                input_signal[i],
                fs=sampling_rate,
                nperseg=256,  # Standard window size
                noverlap=128,  # 50% overlap
                nfft=signal_length
            )
            spectrograms.append((f, t, Sxx))

            # Weight the signal by the relevance to get relevance spectrogram
            weighted_signal = input_signal[i] * relevance_time[i]
            f_rel, t_rel, Sxx_rel = scipy.signal.spectrogram(
                weighted_signal,
                fs=sampling_rate,
                nperseg=256,
                noverlap=128,
                nfft=signal_length
            )
            relevance_spectrograms.append((f_rel, t_rel, Sxx_rel))
    else:
        # DFT-LRP time-frequency data is available
        has_spectrograms = False

    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    ncols = 6  # 6 columns: time signal, time relevance, freq signal, freq relevance, TF signal, TF relevance
    nrows = n_axes
    figsize = (ncols * 4, nrows * 3.5)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

    def replace_positive(x, positive=True):
        """Replace positive (or negative if positive=False) values with zero."""
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    # Calculate average relevance for both domains to display
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
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(
            f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}",
            fontsize=14)
        ax[i, 0].legend(fontsize=10, loc="upper right")
        ax[i, 0].grid(True)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel("Relevance", fontsize=12)
        ax[i, 1].set_title(f"LRP Relevance (Time) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 1].legend(fontsize=10, loc="upper right")
        ax[i, 1].grid(True)

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
        ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel("Magnitude", fontsize=12)
        ax[i, 2].set_title(
            f"Signal with LRP Heatmap (Freq) - Axis {axes_names[i]}\nAvg Relevance: {avg_relevances_freq[i]:.4f}",
            fontsize=14)
        ax[i, 2].legend(fontsize=10, loc="upper right")
        ax[i, 2].grid(True)

        ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel("Relevance", fontsize=12)
        ax[i, 3].set_title(f"LRP Relevance (Freq) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 3].legend(fontsize=10, loc="upper right")
        ax[i, 3].grid(True)

        # Time-frequency domain
        if has_spectrograms:
            # Plot signal spectrogram
            f, t, Sxx = spectrograms[i]
            # Limit frequency to k_max
            f_mask = f <= k_max
            im1 = ax[i, 4].pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-10),
                                      shading='gouraud', cmap='viridis')
            ax[i, 4].set_xlabel("Time (s)", fontsize=12)
            ax[i, 4].set_ylabel("Frequency (Hz)", fontsize=12)
            ax[i, 4].set_title(f"Signal Spectrogram - Axis {axes_names[i]}", fontsize=14)
            ax[i, 4].set_ylim(0, k_max)
            plt.colorbar(im1, ax=ax[i, 4], label="Power/dB")

            # Plot relevance spectrogram with a different colormap
            f_rel, t_rel, Sxx_rel = relevance_spectrograms[i]
            # Use absolute value to show relevance magnitude
            im2 = ax[i, 5].pcolormesh(t_rel, f_rel[f_mask], np.abs(Sxx_rel[f_mask]),
                                      shading='gouraud', cmap='coolwarm')
            ax[i, 5].set_xlabel("Time (s)", fontsize=12)
            ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
            ax[i, 5].set_title(f"LRP Relevance Spectrogram - Axis {axes_names[i]}", fontsize=14)
            ax[i, 5].set_ylim(0, k_max)
            plt.colorbar(im2, ax=ax[i, 5], label="Relevance")

        elif has_timefreq:
            try:
                # Get the actual number of frames
                n_frames = signal_timefreq.shape[2]
                total_time = signal_length / sampling_rate
                time_steps = np.linspace(0, total_time, n_frames)

                # Use freq_subset for proper frequency indexing
                freq_subset_len = min(len(freq_subset), signal_timefreq.shape[1])

                # Signal time-frequency magnitude - use pcolormesh instead of imshow for better spectrogram visualization
                signal_tf_data = np.abs(signal_timefreq[i, :freq_subset_len, :])
                im1 = ax[i, 4].pcolormesh(
                    time_steps,
                    freq_subset[:freq_subset_len],
                    signal_tf_data,
                    shading='gouraud',
                    cmap='viridis'
                )
                ax[i, 4].set_xlabel("Time (s)", fontsize=12)
                ax[i, 4].set_ylabel("Frequency (Hz)", fontsize=12)
                ax[i, 4].set_title(f"Signal (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
                plt.colorbar(im1, ax=ax[i, 4], label="Magnitude")

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
                ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                ax[i, 5].set_title(f"LRP Relevance (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
                plt.colorbar(im2, ax=ax[i, 5], label="Relevance")

            except Exception as e:
                print(f"Error plotting time-frequency data for axis {i}: {e}")
                # Create spectrograms directly if time-frequency data has issues
                try:
                    f, t, Sxx = scipy.signal.spectrogram(input_signal[i], fs=sampling_rate, nperseg=256, noverlap=128)
                    # Limit frequency to k_max
                    f_mask = f <= k_max
                    im1 = ax[i, 4].pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-10),
                                              shading='gouraud', cmap='viridis')
                    ax[i, 4].set_xlabel("Time (s)", fontsize=12)
                    ax[i, 4].set_ylabel("Frequency (Hz)", fontsize=12)
                    ax[i, 4].set_title(f"Signal Spectrogram - Axis {axes_names[i]}", fontsize=14)
                    plt.colorbar(im1, ax=ax[i, 4], label="Power/dB")

                    # Weight the signal by the relevance to get relevance spectrogram
                    weighted_signal = input_signal[i] * relevance_time[i]
                    f_rel, t_rel, Sxx_rel = scipy.signal.spectrogram(
                        weighted_signal, fs=sampling_rate, nperseg=256, noverlap=128
                    )
                    # Use absolute value to show relevance magnitude
                    im2 = ax[i, 5].pcolormesh(t_rel, f_rel[f_mask], np.abs(Sxx_rel[f_mask]),
                                              shading='gouraud', cmap='coolwarm')
                    ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                    ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                    ax[i, 5].set_title(f"LRP Relevance Spectrogram - Axis {axes_names[i]}", fontsize=14)
                    plt.colorbar(im2, ax=ax[i, 5], label="Relevance")
                except Exception as e2:
                    print(f"Error creating direct spectrograms: {e2}")
                    ax[i, 4].text(0.5, 0.5, "Time-Frequency Data Error", ha="center", va="center")
                    ax[i, 5].text(0.5, 0.5, "Time-Frequency Data Error", ha="center", va="center")
        else:
            # No time-frequency data available at all
            ax[i, 4].text(0.5, 0.5, "Time-Frequency Data Not Available", ha="center", va="center")
            ax[i, 5].text(0.5, 0.5, "Time-Frequency Data Not Available", ha="center", va="center")

    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import colormaps
import scipy.signal

# this will ony use scipy signal spectrograms for signal itself and  DFT-LRP for time-frequency relevance visualization
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
        cmap="bwr"  # Colormap for relevance heatmap
):
    """
    Hybrid visualization that uses:
    - EnhancedDFTLRP for time and frequency domain relevances
    - scipy's spectrogram for time-frequency signal visualization
    - DFT-LRP results for time-frequency relevance visualization

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
    figsize = (ncols * 4, nrows * 3.5)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)

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
        ax[i, 0].set_xlabel("Time (s)", fontsize=12)
        ax[i, 0].set_ylabel("Amplitude", fontsize=12)
        ax[i, 0].set_title(
            f"Signal with LRP Heatmap (Time) - Axis {axes_names[i]}\n{label_text}, Avg Relevance: {avg_relevances_time[i]:.4f}",
            fontsize=14)
        ax[i, 0].legend(fontsize=10, loc="upper right")
        ax[i, 0].grid(True)

        # Time domain: Relevance
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis, positive=False), color="red",
                              label="Positive")
        ax[i, 1].fill_between(x_time, replace_positive(relevance_time_axis), color="blue", label="Negative")
        ax[i, 1].set_xlabel("Time (s)", fontsize=12)
        ax[i, 1].set_ylabel("Relevance", fontsize=12)
        ax[i, 1].set_title(f"LRP Relevance (Time) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 1].legend(fontsize=10, loc="upper right")
        ax[i, 1].grid(True)

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
        ax[i, 2].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 2].set_ylabel("Magnitude", fontsize=12)
        ax[i, 2].set_title(
            f"Signal with LRP Heatmap (Freq) - Axis {axes_names[i]}\nAvg Relevance: {avg_relevances_freq[i]:.4f}",
            fontsize=14)
        ax[i, 2].legend(fontsize=10, loc="upper right")
        ax[i, 2].grid(True)

        ax[i, 3].set_xlabel("Frequency (Hz)", fontsize=12)
        ax[i, 3].set_ylabel("Relevance", fontsize=12)
        ax[i, 3].set_title(f"LRP Relevance (Freq) - Axis {axes_names[i]}", fontsize=14)
        ax[i, 3].legend(fontsize=10, loc="upper right")
        ax[i, 3].grid(True)

        # Time-frequency domain:
        # 1. Signal visualization: Use scipy's spectrogram
        f, t, Sxx = spectrograms[i]
        # Limit frequency to k_max
        f_mask = f <= k_max
        im1 = ax[i, 4].pcolormesh(t, f[f_mask], 10 * np.log10(Sxx[f_mask] + 1e-10),
                                  shading='gouraud', cmap='viridis')
        ax[i, 4].set_xlabel("Time (s)", fontsize=12)
        ax[i, 4].set_ylabel("Frequency (Hz)", fontsize=12)
        ax[i, 4].set_title(f"Signal Spectrogram - Axis {axes_names[i]}", fontsize=14)
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
                ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                ax[i, 5].set_title(f"LRP Relevance (Time-Freq) - Axis {axes_names[i]}", fontsize=14)
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
                ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                ax[i, 5].set_title(f"LRP Relevance Spectrogram - Axis {axes_names[i]}", fontsize=14)
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
                ax[i, 5].set_xlabel("Time (s)", fontsize=12)
                ax[i, 5].set_ylabel("Frequency (Hz)", fontsize=12)
                ax[i, 5].set_title(f"LRP Relevance Spectrogram (Fallback) - Axis {axes_names[i]}", fontsize=14)
                ax[i, 5].set_ylim(0, k_max)
                plt.colorbar(im2, ax=ax[i, 5], label="Relevance")
            except Exception as e2:
                print(f"Error creating direct relevance spectrogram: {e2}")
                ax[i, 5].text(0.5, 0.5, "Time-Frequency Relevance Error", ha="center", va="center")

    fig.suptitle(f"LRP Explanation - {label_text}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ... (other visualization functions) ...

def visualize_crp_heatmap(
    signal,
    crp_heatmap,
    label,
    axes_names=["X", "Y", "Z"],
    sample_idx=0,
    method_name="CRP"
):
    """
    Visualize CRP heatmap as a color overlay behind the signal, with a colorbar.
    Args:
        signal: (3, time_steps) numpy array or torch tensor
        crp_heatmap: (3, time_steps) numpy array or torch tensor
        label: int (0=Good, 1=Bad)
        axes_names: list of str
        sample_idx: sample index for title
        method_name: label for plot
    """
    # Convert to numpy
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(crp_heatmap, torch.Tensor):
        crp_heatmap = crp_heatmap.detach().cpu().numpy()

    # Defensive shape handling
    signal = np.asarray(signal)
    crp_heatmap = np.asarray(crp_heatmap)

    # Squeeze batch if needed
    if signal.ndim == 3 and signal.shape[0] == 1:
        signal = signal.squeeze(0)
    if crp_heatmap.ndim == 3 and crp_heatmap.shape[0] == 1:
        crp_heatmap = crp_heatmap.squeeze(0)

    # Ensure 2D (axes, time)
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
    if crp_heatmap.ndim == 1:
        crp_heatmap = crp_heatmap.reshape(1, -1)

    # Match axes count
    if signal.shape[0] != crp_heatmap.shape[0]:
        # If one is single axis, repeat to match the other
        if signal.shape[0] == 1 and crp_heatmap.shape[0] > 1:
            signal = np.repeat(signal, crp_heatmap.shape[0], axis=0)
        elif crp_heatmap.shape[0] == 1 and signal.shape[0] > 1:
            crp_heatmap = np.repeat(crp_heatmap, signal.shape[0], axis=0)
        else:
            raise ValueError(f"Shape mismatch: signal {signal.shape}, crp_heatmap {crp_heatmap.shape}")

    # CRP debug: print stats
    print("[DEBUG] signal shape:", signal.shape)
    print("[DEBUG] crp_heatmap shape:", crp_heatmap.shape)
    print("[DEBUG] crp_heatmap min:", np.min(crp_heatmap), "max:", np.max(crp_heatmap), "mean:", np.mean(crp_heatmap))

    n_axes = signal.shape[0]
    fig, axs = plt.subplots(n_axes, 1, figsize=(12, 3 * n_axes))
    if n_axes == 1:
        axs = [axs]

    for i in range(n_axes):
        time_steps = np.arange(signal[i].shape[0])
        max_abs_rel = np.max(np.abs(crp_heatmap[i]))
        # Avoid zero norm (show neutral color if all zero)
        if max_abs_rel == 0:
            max_abs_rel = 1e-8
        norm = plt.Normalize(vmin=-max_abs_rel, vmax=max_abs_rel)
        cmap = colormaps['bwr']

        # Color background: relevance as heatmap
        for t in range(len(time_steps) - 1):
            axs[i].axvspan(
                time_steps[t], time_steps[t + 1],
                color=cmap(norm(crp_heatmap[i][t])),
                alpha=0.6
            )

        # Plot signal on top
        axs[i].plot(time_steps, signal[i], color='black', linewidth=1.0, label="Signal")

        axs[i].set_title(
            f"{method_name} Heatmap for {axes_names[i]}-Axis (Sample {sample_idx}, Label: {'Good' if label==0 else 'Bad'})"
        )
        axs[i].set_xlabel("Time Step")
        axs[i].set_ylabel("Amplitude")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axs[i], orientation='vertical', label='Relevance')

        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_crp_timefreq(signal, crp_heatmap, label, sample_rate=400, axes_names=["X", "Y", "Z"]):
    """
    Visualize CRP heatmap in both time and frequency domains
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    if isinstance(crp_heatmap, torch.Tensor):
        crp_heatmap = crp_heatmap.detach().cpu().numpy()

    n_axes = signal.shape[0]
    fig, axs = plt.subplots(n_axes, 3, figsize=(18, 4 * n_axes))

    for i in range(n_axes):
        # Time domain - original from your function
        time_steps = np.arange(signal[i].shape[0])
        time_in_sec = time_steps / sample_rate

        # Using your existing code for the heatmap background
        max_abs_rel = np.max(np.abs(crp_heatmap[i]))
        norm = plt.Normalize(vmin=-max_abs_rel, vmax=max_abs_rel)
        cmap = colormaps['bwr']

        # 1. Time domain signal with heatmap
        for t in range(len(time_steps) - 1):
            axs[i, 0].axvspan(time_in_sec[t], time_in_sec[t + 1],
                              color=cmap(norm(crp_heatmap[i][t])),
                              alpha=0.6)
        axs[i, 0].plot(time_in_sec, signal[i], color='black', linewidth=0.8)
        axs[i, 0].set_title(f"{axes_names[i]}-Axis: Time Domain with CRP")
        axs[i, 0].set_xlabel("Time (s)")

        # 2. Frequency domain with CRP emphasis
        # Calculate FFT
        n = len(signal[i])
        fft_signal = np.fft.rfft(signal[i])
        fft_crp = np.fft.rfft(crp_heatmap[i] * signal[i])  # Weighted by relevance
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)

        # Plot frequency spectrum
        axs[i, 1].plot(freqs, np.abs(fft_signal), 'gray', alpha=0.7, label='Original')
        axs[i, 1].plot(freqs, np.abs(fft_crp), 'r', alpha=0.7, label='CRP Weighted')
        axs[i, 1].set_title(f"{axes_names[i]}-Axis: Frequency Emphasis")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_xlim(0, sample_rate / 2)
        axs[i, 1].legend()

        # 3. Spectrogram with CRP overlay
        # Calculate spectrogram
        f, t, Sxx = scipy.signal.spectrogram(signal[i], fs=sample_rate,
                                             nperseg=128, noverlap=64)

        # Resize CRP to match spectrogram time bins
        from scipy.ndimage import zoom
        zoom_factor = len(t) / len(crp_heatmap[i])
        crp_resized = zoom(crp_heatmap[i], zoom_factor, order=1)

        # Create masked array for overlay
        abs_crp = np.abs(crp_resized) / np.max(np.abs(crp_resized))

        # Plot spectrogram
        im = axs[i, 2].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), cmap='viridis')

        # Overlay contours of significant CRP areas
        levels = np.linspace(0.5, 1.0, 5)
        ct = axs[i, 2].contour(t, [f[0]] * len(t),
                               [abs_crp] * len(f[0]), levels=levels,
                               colors=['r', 'r', 'r', 'r', 'r'],
                               alpha=0.7, linewidths=1)

        axs[i, 2].set_title(f"{axes_names[i]}-Axis: Spectrogram with CRP Contours")
        axs[i, 2].set_xlabel("Time (s)")
        axs[i, 2].set_ylabel("Frequency (Hz)")

    plt.colorbar(im, ax=axs[:, 2].tolist())
    plt.tight_layout()
    plt.show()