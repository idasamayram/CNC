import os
import shutil
import random
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils import data_loader_utils
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


#  function to load vibration data from an HDF5 file
def load_data(file_path):
    """
    Load vibration data from an HDF5 file.
    """
    with h5py.File(file_path, 'r') as file:
        df = file['vibration_data'][:]
    return pd.DataFrame({'X': df[:, 0], 'Y': df[:, 1], 'Z': df[:, 2]})

# function to find all .h5 files in a directory
def find_all_h5s_in_dir(s_dir):
    """
    list all .h5 files in a directory
    """

    fileslist = []
    for root, dirs, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(file)
    return fileslist

# function to load H5 files and calculate the duration in seconds
def load_h5_files_and_calculate_duration(data_root):
    data_list = []
    labels = []

    for machine in ['M01', 'M02', 'M03']:
        for operation in os.listdir(os.path.join(data_root, machine)):
            if os.path.isdir(os.path.join(data_root, machine, operation)):
                for label in ['good', 'bad']:
                    data_path = os.path.join(data_root, machine, operation, label)
                    files = find_all_h5s_in_dir(data_path)

                    for file in files:
                        file_path = os.path.join(data_path, file)
                        with h5py.File(file_path, 'r') as f:
                            vibration_data = f['vibration_data'][:]
                            samples_s = len(vibration_data) / 2000  # Assuming a data sampling frequency of 2000 Hz

                        data_list.append({
                            'Machine': machine,
                            'Operation': operation,
                            'Sample Type': label,
                            'File Name': file,
                            'Duration (s)': samples_s
                        })

    return data_list

# function to read data from two datafiles and plot them
def datafile_read(files, dataset_labels=None, plotting=True):
    """Loads and plots data from multiple datafiles in separate subplots for each axis with dataset labels.

    Keyword Arguments:
        files {list} -- List of file paths
        dataset_labels {list} -- List of dataset labels for legends (optional)

    Returns:
        list of ndarrays -- List of raw data arrays
    """
    all_data = []
    max_samples = 0

    for file in files:
        with h5py.File(file, 'r') as f:
            vibration_data = f['vibration_data'][:]
        all_data.append(vibration_data)
        max_samples = max(max_samples, len(vibration_data))

    # Interpolation for the x-axis plot
    freq = 2000
    samples_s = max_samples / freq
    samples = np.linspace(0, samples_s, max_samples)

    if plotting:
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        for i, data in enumerate(all_data):
            if len(data) < max_samples:
                padding = np.full((max_samples - len(data), 3), np.nan)
                data = np.vstack((data, padding))

            # Determine label based on file name
            file_path = files[i].lower()
            if 'good' in file_path:
                label = 'Good'
                color = 'blue'
                linestyle = '-'
                alpha = 0.7
            elif 'bad' in file_path:
                label = 'Bad'
                color = 'red'
                linestyle = '--'
                alpha = 0.7
            else:
                label = f'Dataset {i + 1}'
                color = 'gray'
                linestyle = '-.'
                alpha = 0.5

            axs[0].plot(samples, data[:, 0], label=f'X-axis - {label}', color=color,
                        alpha=alpha, linewidth=1.5, linestyle=linestyle)
            axs[1].plot(samples, data[:, 1], label=f'Y-axis - {label}', color=color,
                        alpha=alpha, linewidth=1.5, linestyle=linestyle)
            axs[2].plot(samples, data[:, 2], label=f'Z-axis - {label}', color=color,
                        alpha=alpha, linewidth=1.5, linestyle=linestyle)

        axs[0].set_ylabel('X-axis Vibration Data')
        axs[1].set_ylabel('Y-axis Vibration Data')
        axs[2].set_ylabel('Z-axis Vibration Data')
        axs[2].set_xlabel('Time [sec]')

        for ax in axs:
            ax.locator_params(axis='y', nbins=10)
            ax.grid()
            ax.legend()

        plt.show()

    return all_data

# function to load data from multiple directories and concatenate them
def load_and_concatenate_data(path_to_dataset, process_names, machines, labels):
    """
    Load data from multiple directories and concatenate them.
    """
    X_data = []
    y_data = []

    for process_name in process_names:
        for machine in machines:
            for label in labels:
                data_path = os.path.join(path_to_dataset, machine, process_name, label)
                data_list, data_label = data_loader_utils.load_tool_research_data(data_path, add_additional_label=False, label=label)
                # Concatenating
                X_data.extend(data_list)
                y_data.extend(data_label)

    return X_data, y_data

# function to separete good and bad data
def separate_good_bad_data(X_data, y_data):
    """
    Separate good and bad data from the dataset.
    """
    good_X_data = []
    good_y_data = []
    bad_X_data = []
    bad_y_data = []

    for i, label in enumerate(y_data):
        if label == "good":
            good_X_data.append(X_data[i])
            good_y_data.append(label)
        elif label == "bad":
            bad_X_data.append(X_data[i])
            bad_y_data.append(label)
        else:
            # Handle other labels if necessary
            pass

    return good_X_data, good_y_data, bad_X_data, bad_y_data

# function to create a DataFrame from the vibration data
def create_dataframe_from_vibration_data(data_dir):
    """
    Create a DataFrame from the vibration data.
    """
    # if data_dir is a string, convert it to a Path object
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    # check if the directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"The specified directory does not exist: {data_dir}")
    with h5py.File(data_dir, 'r') as file:
        vibration_data = file['vibration_data'][:]


    vibration_df = pd.DataFrame({'X': vibration_data[:, 0], 'Y': vibration_data[:, 1], 'Z': vibration_data[:, 2]})
    time = len(vibration_df) / 2000

    return vibration_df, time

# Define a function to group the dataset into 'good' and 'bad' folders
def group_dataset(machines, selected_operations, new_root, data_root):
    print(f"Data root exists: {data_root.exists()}")
    if not data_root.exists():
        raise FileNotFoundError(f"The specified data root does not exist: {data_root}")


    grouped_data = []
    selected_data_dir = Path(new_root) / "Selected_data_grouped"
    selected_data_dir.mkdir(exist_ok=True)
    print(f"Directory '{selected_data_dir}' created")

    bad_data_dir = selected_data_dir / "bad"
    good_data_dir = selected_data_dir / "good"

    bad_data_dir.mkdir(exist_ok=True)
    print(f"Directory '{bad_data_dir}' created")

    good_data_dir.mkdir(exist_ok=True)
    print(f"Directory '{good_data_dir}' created")

    for machine in machines:
        for operation in selected_operations:
            bad_folder = Path(data_root) / machine / operation / 'bad'
            good_folder = Path(data_root) / machine / operation / 'good'

            # Get the list of bad files
            if bad_folder.exists():
                bad_files = [file for file in bad_folder.iterdir() if file.is_file() and file.suffix == '.h5']

                for file in bad_files:

                    if not bad_folder.exists():
                        print(f"Skipping missing folder: {bad_folder}")
                        continue
                    dest_bad = bad_data_dir / file.name
                    try:
                        shutil.copy2(file, dest_bad)
                        print(f"Copied '{file}' to '{dest_bad}'")
                    except shutil.Error as e:
                        print(f"Error occurred while copying file: {e}")
                    except IOError as e:
                        print(f"Error occurred while accessing file: {e.strerror}")

                    grouped_data.append((machine, operation, 'bad', file.name))

        # Get the list of good files
            if good_folder.exists():
                good_files = [file for file in good_folder.iterdir() if file.is_file() and file.suffix == '.h5']

                for file in good_files:
                    dest_good = good_data_dir / file.name
                    try:
                        shutil.copy2(file, dest_good)
                        print(f"Copied '{file}' to '{dest_good}'")
                    except shutil.Error as e:
                        print(f"Error occurred while copying file: {e}")
                    except IOError as e:
                        print(f"Error occurred while accessing file: {e.strerror}")

                    grouped_data.append((machine, operation, 'good', file.name))

    return grouped_data

# function to Normalize the data based on good data
def normalize_data(input_root, output_root):
    """
    Normalize the vibration data based on the mean and standard deviation of good samples.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"The specified input directory does not exist: {input_root}")

    output_root.mkdir(exist_ok=True)

    # Get list of all machine-operation combinations
    machine_operations = {}

    for label in ['bad', 'good']:
        input_folder = input_root / label
        if not input_folder.exists():
            print(f"Skipping missing folder: {input_folder}")
            continue

        for file_path in input_folder.glob("*.h5"):
            file_name = file_path.name
            parts = file_name.split('_')  # Example: M01_Aug_2019_OP01_002.h5
            machine = parts[0]  # e.g., M01
            operation = parts[3]  # e.g., OP01
            key = f"{machine}_{operation}"

            if key not in machine_operations:
                machine_operations[key] = []
            machine_operations[key].append(file_path)

    # Step 1: Compute mean & std per machine-operation
    machine_operation_stats = {}
    for key, files in machine_operations.items():
        all_data = []
        for file_path in files:
            if 'bad' in str(file_path):  # Skip bad samples
                continue
            with h5py.File(file_path, 'r') as file:
                vibration_data = file['vibration_data'][:]
                all_data.append(vibration_data)
        if not all_data:  # Skip if no good samples
            continue
        all_data = np.vstack(all_data)
        mean_vals = np.mean(all_data, axis=0)
        std_vals = np.std(all_data, axis=0)
        std_vals[std_vals == 0] = 1  # Avoid division by zero
        machine_operation_stats[key] = (mean_vals, std_vals)
        print(f"Computed Mean & Std for {key} → Mean: {mean_vals}, Std: {std_vals}")


    # Step 2: Normalize each file using its machine-operation group stats
    for key, files in machine_operations.items():
        mean_vals, std_vals = machine_operation_stats[key]

        for file_path in files:
            with h5py.File(file_path, 'r') as file:
                vibration_data = file['vibration_data'][:]

                # Debug: Print mean/std of a sample before normalization
                print(f"\nBefore Normalization ({file_path.name}):")
                print("Mean:", np.mean(vibration_data, axis=0))
                print("Std:", np.std(vibration_data, axis=0))

                # Normalize using machine-operation-specific mean & std
                normalized_data = (vibration_data - mean_vals) / std_vals

                # Debug: Print mean/std after normalization to verify correctness
                print(f"After Normalization ({file_path.name}):")
                print("Mean:", np.mean(normalized_data, axis=0))
                print("Std:", np.std(normalized_data, axis=0))

                # Define output path using pathlib
                label = "bad" if "bad" in str(file_path) else "good"
                output_folder = output_root / label
                output_folder.mkdir(parents=True, exist_ok=True)

                output_file_path = output_folder / file_path.name

                # Save the normalized data
                with h5py.File(output_file_path, 'w') as new_file:
                    new_file.create_dataset('vibration_data', data=normalized_data)

                    print(f"✅ Saved normalized file: {output_file_path}")

    print("\n✅ Normalization complete! All files saved in:", output_root)

# function to compute the FFT of a signal
def compute_fft(signal, sampling_rate=2000):
    """Calculates the FFT of a signal.

    Args:
        signal: The time-domain signal (a NumPy array).
        sampling_rate: The sampling rate of the signal in Hz.

    Returns:
        (frequencies, fft_values): A tuple containing:
            * frequencies: An array of frequencies corresponding to the FFT bins.
            * fft_values: An array of complex FFT values.
    """

    signal_length = len(signal)
    fft_values = np.fft.fft(signal)  # Calculate the FFT
    frequencies = np.fft.fftfreq(signal_length, d=1/sampling_rate)  # Frequencies

    # Keep only positive frequencies for real-world signals
    positive_indices = frequencies >= 0
    frequencies = frequencies[positive_indices]
    fft_values = fft_values[positive_indices]

    return frequencies, fft_values

# function to plot FFT comparison between good and bad samples
def plot_fft_comparison(good_h5, bad_h5, good_sample_name, bad_sample_name, sampling_freq=2000):
    """
    Loads vibration data from good and bad H5 files, computes FFT for each axis,
    and plots a comparison between them.

    Args:
        good_h5: Path to the H5 file containing good sample data
        bad_h5: Path to the H5 file containing bad sample data
        good_sample_name: Name identifier for the good sample
        bad_sample_name: Name identifier for the bad sample
        sampling_freq: Sampling frequency in Hz (default: 2000)

    Returns:
        matplotlib figure object
    """
    # Load and compute FFT for bad sample
    with h5py.File(bad_h5, 'r') as f:
        bad_data_x = f['vibration_data'][:, 0]
        bad_data_y = f['vibration_data'][:, 1]
        bad_data_z = f['vibration_data'][:, 2]

    freqs_bad_x, magnitudes_bad_x = compute_fft(bad_data_x, sampling_freq)
    freqs_bad_y, magnitudes_bad_y = compute_fft(bad_data_y, sampling_freq)
    freqs_bad_z, magnitudes_bad_z = compute_fft(bad_data_z, sampling_freq)

    # Load and compute FFT for good sample
    with h5py.File(good_h5, 'r') as f1:
        good_data_x = f1['vibration_data'][:, 0]
        good_data_y = f1['vibration_data'][:, 1]
        good_data_z = f1['vibration_data'][:, 2]

    freqs_good_x, magnitudes_good_x = compute_fft(good_data_x, sampling_freq)
    freqs_good_y, magnitudes_good_y = compute_fft(good_data_y, sampling_freq)
    freqs_good_z, magnitudes_good_z = compute_fft(good_data_z, sampling_freq)

    # Extract machine and operation info
    parts = good_sample_name.split('_')  # Example: M01_Aug_2019_OP01_002
    machine = parts[0]  # e.g., M01
    operation = parts[3]  # e.g., OP01

    # Plot FFT magnitudes for comparison
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(freqs_good_x, np.abs(magnitudes_good_x), label=f'Good: {good_sample_name}', color='black')
    plt.plot(freqs_bad_x, np.abs(magnitudes_bad_x), label=f'Bad: {bad_sample_name}', color='magenta', linestyle='dashdot',
             alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Magnitude Comparison (X-axis) - {operation} - {machine}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(freqs_good_y, np.abs(magnitudes_good_y), label=f'Good: {good_sample_name}', color='black')
    plt.plot(freqs_bad_y, np.abs(magnitudes_bad_y),label=f'Bad: {bad_sample_name}', color='magenta', linestyle='dashdot',
             alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Magnitude Comparison (Y-axis) - {operation} - {machine}')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(freqs_good_z, np.abs(magnitudes_good_z), label=f'Good: {good_sample_name}', color='black')
    plt.plot(freqs_bad_z, np.abs(magnitudes_bad_z), label=f'Bad: {bad_sample_name}', color='magenta', linestyle='dashdot',
             alpha=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Magnitude Comparison (Z-axis) - {operation} - {machine}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    return fig

# function to compute the FFT of a signal, handling both single-axis and multi-axis signals
def compute_fft_total(signal, sampling_rate=2000):
    """Calculates the FFT of a signal.

    Args:
        signal: The time-domain signal (a NumPy array). Can be single-axis or multi-axis.
        sampling_rate: The sampling rate of the signal in Hz.

    Returns:
        If signal is 1D:
            (frequencies, fft_values): A tuple containing frequencies and complex FFT values.
        If signal is 2D (multi-axis):
            (frequencies, fft_values): A tuple where fft_values is a list of FFT values for each axis.
    """

    def compute_fft_single_axis(signal, sampling_rate=2000):
        """Helper function to calculate FFT for a single-axis signal."""
        signal_length = len(signal)
        fft_values = np.fft.fft(signal)  # Calculate the FFT
        frequencies = np.fft.fftfreq(signal_length, d=1 / sampling_rate)  # Frequencies

        # Keep only positive frequencies for real-world signals
        positive_indices = frequencies >= 0
        frequencies = frequencies[positive_indices]
        fft_values = fft_values[positive_indices]

        return frequencies, fft_values


    # Check if the signal is multi-dimensional (has multiple axes)
    if len(np.shape(signal)) > 1 and np.shape(signal)[1] > 1:
        # Multi-axis signal (e.g., X, Y, Z vibration data)
        # Process each axis separately
        frequencies = None
        fft_values = []

        for axis in range(np.shape(signal)[1]):
            axis_signal = signal[:, axis]
            freqs, fft = compute_fft_single_axis(axis_signal, sampling_rate)

            # Store the frequency array only once (it's the same for all axes)
            if frequencies is None:
                frequencies = freqs

            fft_values.append(fft)

        return frequencies, fft_values
    else:
        # Single-axis signal
        return compute_fft_single_axis(signal, sampling_rate)

# function to visualize the FFT of a 3-axis vibration signal in both 2D and 3D
def visualize_fft_3d(data_dir, sampling_rate=2000, title="3D FFT Visualization"):
    """
    Visualizes the FFT of a 3-axis vibration signal in both 2D plots and a 3D plot.

    Args:
        signal: Multi-axis signal with shape (time_steps, 3) for X, Y, Z axes
        sampling_rate: The sampling rate in Hz
        title: Title for the plots
    """

    with h5py.File(data_dir, 'r') as f:
        vibration_data = f['vibration_data'][:]
        # Compute FFT for all three axes

    frequencies, fft_values = compute_fft_total(vibration_data, sampling_rate)

    # Get magnitude (absolute value) of the complex FFT values
    magnitudes = [np.abs(fft) for fft in fft_values]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 2D plots for each axis
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(frequencies, magnitudes[0])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('X-axis FFT')
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frequencies, magnitudes[1])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Y-axis FFT')
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(frequencies, magnitudes[2])
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Z-axis FFT')
    ax3.grid(True)

    # 3D visualization
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    # Create a frequency threshold to reduce clutter (e.g., first 100 frequencies)
    cutoff = min(100, len(frequencies))
    x = magnitudes[0][:cutoff]
    y = magnitudes[1][:cutoff]
    z = magnitudes[2][:cutoff]
    freq = frequencies[:cutoff]

    # 3D scatter plot with frequency-based coloring
    scatter = ax4.scatter(x, y, z, c=freq, cmap='viridis',
                          s=10, alpha=0.7)

    # Add a color bar to show frequency mapping
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Frequency (Hz)')

    ax4.set_xlabel('X-axis Magnitude')
    ax4.set_ylabel('Y-axis Magnitude')
    ax4.set_zlabel('Z-axis Magnitude')
    ax4.set_title('3D FFT Visualization')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig

# function to trim and create 5-second windows from time series data
def trim_and_window(time_series, label, window_size=10000, sampling_rate=2000, max_trim_seconds=3, overlap_slide=5000):
    """
    Trim the time series (for GOOD samples) and create 5-second windows.
    GOOD samples: Trim up to 3 seconds from both ends (adjust if signal is short), non-overlapping windows.
    BAD samples: No trimming, overlapping windows (2.5-second slide).

    Args:
        time_series (np.array): The time series data (shape: [time_steps, 3] for X, Y, Z).
        label (str): "GOOD" or "BAD" indicating the class of the time series.
        window_size (int): The size of the sliding window.
        sampling_rate (int): The sampling rate of the time series in Hz.
        max_trim_seconds (int): Maximum seconds to trim from each side for GOOD samples.
        overlap_slide (int): Number of data points to slide for overlapping windows, BAD samples (default: 5000 for 2.5 seconds).

    Returns:
        List of 5-second windows.
    """
    max_trim_points = max_trim_seconds * sampling_rate  # Maximum data points to trim (6000 points)

    total_len = time_series.shape[0]  # Total number of data points
    print(f"Total length of time series: {total_len} data points ({total_len / sampling_rate:.2f} seconds)")

    # Minimum length to produce at least one 5-second window
    min_length = window_size  # 10,000 points (5 seconds)

    if total_len < min_length:
        print(f"Time series too short to create a 5-second window: {total_len} data points ({total_len / sampling_rate:.2f} seconds), skipping")
        return []

    if label == 'BAD':
        # No trimming for bad samples to preserve anomaly information
        trimmed_series = time_series
        print("No trimming for BAD class")

        # Overlapping windows (2.5-second slide)
        windows = []
        for i in range(0, total_len - window_size + 1, overlap_slide):
            window = trimmed_series[i:i + window_size]
            if window.shape[0] == window_size:  # Ensure full window
                windows.append(window)
        print(f"Number of overlapping 5-second windows (1-second slide) for BAD class: {len(windows)}")

    elif label == 'GOOD':
        # Dynamically adjust trimming based on signal length
        # Goal: Ensure at least one 5-second window after trimming
        # Ideal trim: 3 seconds each side (6000 points)
        # Minimum length to trim 2 seconds each side: 2 * 4000 + 10,000 = 18,000 points (9 seconds)  or 2 * 6,000 + 10,000 = 22,000 points (11 seconds)

        if total_len >= 2 * max_trim_points + window_size:  # Can trim full 3 seconds
            trim_points = max_trim_points
            trimmed_series = time_series[trim_points:total_len - trim_points]
            print(f"Trimmed {trim_points} data points ({max_trim_seconds} seconds) from each side for GOOD class")
        else:
            # Adjust trim to ensure at least one window
            # Need: total_len - 2 * trim_points >= window_size
            # So: 2 * trim_points <= total_len - window_size
            # trim_points <= (total_len - window_size) / 2
            max_possible_trim = (total_len - window_size) // 2
            trim_points = min(max_possible_trim, max_trim_points)  # Don't trim more than max_trim_points
            if trim_points < 0:  # If signal is too short to trim and get a window, this was caught earlier
                trim_points = 0
            trimmed_series = time_series[trim_points:total_len - trim_points]
            print(f"Adjusted trim for short signal: Trimmed {trim_points} data points ({trim_points / sampling_rate:.2f} seconds) from each side for GOOD class")

        print(f"Trimmed time series length: {trimmed_series.shape[0]} data points ({trimmed_series.shape[0] / sampling_rate:.2f} seconds)")

        # Non-overlapping windows (5-second slide)
        windows = []
        for i in range(0, trimmed_series.shape[0] - window_size + 1, window_size):
            window = trimmed_series[i:i + window_size]
            if window.shape[0] == window_size:  # Ensure full window
                windows.append(window)
        print(f"Number of non-overlapping 5-second windows for GOOD class: {len(windows)}")

    return windows

# function to process and save 5-second windows for GOOD and BAD samples
def process_and_save_windows(label, output_root, folder):
    """
    Processes and saves the 5-second windows for a given label (GOOD or BAD) as .h5 files.

    Args:
        label (str): 'good' or 'bad'.
        folder (Path): Path to the 'good' or 'bad' folder.
        output_root: Path to the root directory where processed windows will be saved.
    """
    # Create save directory
    save_dir = output_root / label
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {label.upper()} files in {folder}")

    # Process each h5 file in the folder
    for h5_file in folder.glob("*.h5"):
        print(f"Processing file: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            if 'vibration_data' not in f:
                print(f"No 'vibration_data' dataset in file: {h5_file}")
                continue

            vibration_data = f['vibration_data'][:]
            print(f"Loaded time series data of shape {vibration_data.shape}")

            # Trim and create windows
            windows = trim_and_window(vibration_data, label.upper())

            # Save each window as a new .h5 file
            for idx, window in enumerate(windows):
                window_filename = save_dir / f"{h5_file.stem}_window_{idx}.h5"
                with h5py.File(window_filename, 'w') as hf:
                    hf.create_dataset('vibration_data', data=window)
                    print(f"Saved window {idx} for {h5_file.stem} to {window_filename}")

# function to create 5-second windows for GOOD and BAD samples apply
def window_dataset(input_root, output_root):
    """
    Process the dataset to create 5-second windows for GOOD and BAD samples.
    Saves the processed windows in the specified output directory.

    Args:
        input_root: Path to the root directory containing 'good' and 'bad' folders.
        output_root: Path to the root directory where processed windows will be saved.
    """
    for label in ['good', 'bad']:
        folder = input_root / label
        if folder.exists():
            process_and_save_windows(label, output_root, folder)
        else:
            print(f"Folder not found: {folder}")


    print("\n✅ Windowing complete! All files saved in:", output_root)

# Function to downsample a single time series using NumPy
def downsample_time_series(vibration_data, original_sample_length, target_sample_length):
    """
    Downsample the vibration data using average pooling with NumPy.

    Args:
        vibration_data (np.array): Shape (10000, 3) - original vibration data (X, Y, Z axes).
        original_samples (int): Original number of samples (10000).
        target_samples (int): Target number of samples after downsampling (2000).

    Returns:
        downsampled_data (np.array): Shape (2000, 3) - downsampled vibration data.
    """
    # Validate input shape
    if vibration_data.shape[0] != original_sample_length:
        raise ValueError(f"Expected {original_sample_length} samples, but got {vibration_data.shape[0]} sample length")
    if vibration_data.shape[1] != 3:
        raise ValueError(f"Expected 3 axes, but got {vibration_data.shape[1]} axes")

    # Calculate downsampling factor
    factor = original_sample_length // target_sample_length  # Should be 10

    # Reshape data to (target_samples, factor, 3) and average along the factor axis
    reshaped_data = vibration_data.reshape(target_sample_length, factor, 3)  # Shape: (2000, 5, 3)
    downsampled_data = np.mean(reshaped_data, axis=1)  # Shape: (2000, 3)

    return downsampled_data

# Function to process and save downsampled files
def downsample_and_save_files(input_root, output_root, original_length=10000, target_length=2000):
    """
    Downsample all .h5 files in the windowed_root directory and save them to downsampled_root.
    """
    # Iterate through both 'bad' and 'good' folders
    for sample_type in ["bad", "good"]:
        input_folder = input_root / sample_type
        output_folder = output_root / sample_type

        # Create output folder
        output_folder.mkdir(exist_ok=True)
        print(f"Directory '{output_folder}' created")

        if not input_folder.exists():
            print(f"Skipping missing folder: {input_folder}")
            continue

        # Iterate through all .h5 files
        for h5_file in input_folder.glob("*.h5"):
            print(f"Processing file: {h5_file}")

            # Load the original data
            with h5py.File(h5_file, 'r') as f:
                vibration_data = f['vibration_data'][:]  # Shape: (10000, 3)
                print(f"Loaded time series data of shape {vibration_data.shape}")

            # Downsample the data
            try:
                downsampled_data = downsample_time_series(vibration_data, original_length, target_length)
                print(f"Downsampled time series data to shape {downsampled_data.shape}")
            except ValueError as e:
                print(f"Error downsampling {h5_file}: {e}")
                continue

            # Define the new filename with "_downsampled" appended
            new_filename = f"{h5_file.stem}_downsampled.h5"
            output_path = output_folder / new_filename

            # Save the downsampled data
            with h5py.File(output_path, 'w') as hf:
                hf.create_dataset('vibration_data', data=downsampled_data)
                print(f"Saved downsampled file to {output_path}")

# Function to verify the downsampled data size and duration
def verify_downsampled_data(output_root_down, output_file_name="downsampled_data_summary.csv"):
    """
    Verify the downsampled data by checking the shape and duration of each file.
    """
    # Verify the downsampled data
    print("\nVerifying downsampled data:")
    output_root_down = Path(output_root_down)

    if not output_root_down.exists():
        raise FileNotFoundError(f"The specified output directory does not exist: {output_root_down}")

    data_list = []

    for sample_type in ["bad", "good"]:
        folder_path = output_root_down / sample_type
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder_path}")
            continue

        for file_path in folder_path.glob("*.h5"):
            file_name = file_path.name
            with h5py.File(file_path, 'r') as file:
                vibration_data = file['vibration_data'][:]
                num_samples = vibration_data.shape[0]
                duration = num_samples / 400  # Duration in seconds
                data_list.append({
                    "Sample Type": sample_type,
                    "File Name": file_name,
                    "Duration (s)": duration,
                    "Shape": vibration_data.shape
                })

    df_data_list= pd.DataFrame(data_list)
    print(f' Data Length:{len(df_data_list)}')

    # Save to CSV for reference
    df_data_list.to_csv(output_file_name, index=False)
    print(f'\n✅ Downsampled data summary saved to {output_file_name}')

    return df_data_list

# Function to extract unseen data from the dataset
def extract_unseen_data(data_root, new_root, seen_machines, seen_operations):
    """
    Identifies and copies data from:
    1. All operations of unseen_grouped machines (e.g., all data from M03)
    2. Unseen operations from seen machines (e.g., OP13 from M01, M02)

    Args:
        data_root (str or Path): Path to the original dataset
        new_root (str or Path): Path to save the unseen_grouped data
        seen_machines (list): List of machine names that have already been used (e.g., ['M01', 'M02'])
        seen_operations (list): List of operation names that have already been used (e.g., ['OP01', 'OP02'])
    """

    data_root = Path(data_root)
    new_root = Path(new_root)

    if not data_root.exists():
        raise FileNotFoundError(f"The specified data root does not exist: {data_root}")

    new_root.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{new_root}' created")

    # Create good and bad folders
    good_data_dir = new_root / "good"
    bad_data_dir = new_root / "bad"
    good_data_dir.mkdir(exist_ok=True)
    bad_data_dir.mkdir(exist_ok=True)

    # Iterate through all machines and operations in the dataset
    for machine in data_root.iterdir():
        if machine.is_dir() and machine.name not in seen_machines:
            # Unseen machine: copy all operations
            print(f"Copying all operations from unseen_grouped machine: {machine.name}")
            for operation in machine.iterdir():
                if operation.is_dir():
                    for label in ['good', 'bad']:
                        label_folder = operation / label
                        if label_folder.exists():
                            for file in label_folder.glob("*.h5"):
                                dest_file = (good_data_dir if label == 'good' else bad_data_dir) / file.name
                                shutil.copy2(file, dest_file)
                                print(f"Copied {file} to {dest_file}")

        elif machine.is_dir() and machine.name in seen_machines:
            # Seen machine: copy unseen_grouped operations
            print(f"Seen machine: {machine.name}. Checking for unseen_grouped operations.")
            for operation in machine.iterdir():
                if operation.is_dir() and operation.name not in seen_operations:
                    print(f"Copying unseen_grouped operation: {operation.name} from machine {machine.name}")
                    for label in ['good', 'bad']:
                        label_folder = operation / label
                        if label_folder.exists():
                            for file in label_folder.glob("*.h5"):
                                dest_file = (good_data_dir if label == 'good' else bad_data_dir) / file.name
                                shutil.copy2(file, dest_file)
                                print(f"Copied {file} to {dest_file}")

    print("\n✅ Unseen data extraction complete! All files saved in:", new_root)



