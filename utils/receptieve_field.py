import matplotlib.pyplot as plt
import numpy as np
import torch

# receptieve field only for one axis, here the cnn1d_wide
def calculate_receptive_field_cnn1d_wide():
    """
    Calculates and returns the receptive field for each layer of CNN1D_Wide
    """
    # Define layer properties: (name, kernel_size, stride, dilation)
    # Only include layers that affect the receptive field (conv and pooling)
    layers = [
        {"name": "Input", "kernel_size": 1, "stride": 1, "dilation": 1},
        {"name": "Conv1D-1", "kernel_size": 25, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-1", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-2", "kernel_size": 15, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-2", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-3", "kernel_size": 9, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-3", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-4", "kernel_size": 5, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-4", "kernel_size": 2, "stride": 2, "dilation": 1},
    ]

    # Calculate cumulative metrics
    receptive_field = 1  # Start with a receptive field of 1 (a single point)
    jump = 1  # How many input steps does one output step correspond to
    output_size = 2000  # Assuming input is 2000 time steps

    # For each layer, calculate how it affects the receptive field
    receptive_fields = [1]  # Input layer
    jumps = [1]  # Input layer
    output_sizes = [output_size]  # Input layer

    for i in range(1, len(layers)):
        layer = layers[i]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        dilation = layer["dilation"]

        # Update receptive field: RF_new = RF_old + (kernel_size - 1) * jump * dilation
        receptive_field = receptive_field + (kernel_size - 1) * jump * dilation
        receptive_fields.append(receptive_field)

        # Update jump: jump_new = jump_old * stride
        jump = jump * stride
        jumps.append(jump)

        # Update output size: output_size_new = (output_size_old - kernel_size * dilation + dilation) / stride + 1
        output_size = (output_size - kernel_size * dilation + dilation) // stride + 1
        output_sizes.append(output_size)

    # Create a table with the calculated values
    rf_table = []
    for i, layer in enumerate(layers):
        rf_table.append({
            "Layer": layer["name"],
            "Kernel Size": layer["kernel_size"] if i > 0 else "-",
            "Stride": layer["stride"] if i > 0 else "-",
            "Receptive Field": receptive_fields[i],
            "Jump": jumps[i],
            "Output Size": output_sizes[i]
        })

    return rf_table

# receptieve field only for one axis, here the cnn1d_ds_wide
def calculate_receptive_field_cnn1d_ds_wide():
    """
    Calculates and returns the receptive field for each layer of CNN1D_DS_Wide
    """
    # Define layer properties: (name, kernel_size, stride, dilation)
    # Only include layers that affect the receptive field (conv and pooling)
    layers = [
        {"name": "Input", "kernel_size": 1, "stride": 1, "dilation": 1},
        {"name": "Conv1D-1", "kernel_size": 25, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-1", "kernel_size": 3, "stride": 2, "dilation": 1},
        {"name": "Conv1D-2", "kernel_size": 15, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-2", "kernel_size": 3, "stride": 2, "dilation": 1},
        {"name": "Conv1D-3", "kernel_size": 9, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-3", "kernel_size": 3, "stride": 2, "dilation": 1},
    ]

    # Calculate cumulative metrics
    receptive_field = 1  # Start with a receptive field of 1 (a single point)
    jump = 1  # How many input steps does one output step correspond to
    output_size = 2000  # Assuming input is 2000 time steps

    # For each layer, calculate how it affects the receptive field
    receptive_fields = [1]  # Input layer
    jumps = [1]  # Input layer
    output_sizes = [output_size]  # Input layer

    for i in range(1, len(layers)):
        layer = layers[i]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        dilation = layer["dilation"]

        # Update receptive field: RF_new = RF_old + (kernel_size - 1) * jump * dilation
        receptive_field = receptive_field + (kernel_size - 1) * jump * dilation
        receptive_fields.append(receptive_field)

        # Update jump: jump_new = jump_old * stride
        jump = jump * stride
        jumps.append(jump)

        # Update output size: output_size_new = (output_size_old - kernel_size * dilation + dilation) / stride + 1
        output_size = (output_size - kernel_size * dilation + dilation) // stride + 1
        output_sizes.append(output_size)

    # Create a table with the calculated values
    rf_table = []
    for i, layer in enumerate(layers):
        rf_table.append({
            "Layer": layer["name"],
            "Kernel Size": layer["kernel_size"] if i > 0 else "-",
            "Stride": layer["stride"] if i > 0 else "-",
            "Receptive Field": receptive_fields[i],
            "Jump": jumps[i],
            "Output Size": output_sizes[i]
        })

    return rf_table

# Function to calculate receptive field relative to original time steps
def receptive_field_in_seconds(rf_steps, sampling_rate=400):
    """
    Convert receptive field size in time steps to seconds

    Parameters:
    -----------
    rf_steps : int
        Receptive field size in time steps
    sampling_rate : int, optional
        Sampling rate in Hz, default is 400 Hz

    Returns:
    --------
    float
        Receptive field size in seconds
    """
    return rf_steps / sampling_rate

# receptieve field only for one axis, here the cnn1d_wide including seconds
def calculate_receptive_field_cnn1d_wide_sec(sampling_rate=400):
    """
    Calculates and returns the receptive field for each layer of CNN1D_Wide

    Parameters:
    -----------
    sampling_rate : int, optional
        Sampling rate in Hz, default is 400 Hz
    """
    # Define layer properties
    layers = [
        {"name": "Input", "kernel_size": 1, "stride": 1, "dilation": 1},
        {"name": "Conv1D-1", "kernel_size": 25, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-1", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-2", "kernel_size": 15, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-2", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-3", "kernel_size": 9, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-3", "kernel_size": 4, "stride": 4, "dilation": 1},
        {"name": "Conv1D-4", "kernel_size": 5, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-4", "kernel_size": 2, "stride": 2, "dilation": 1},
    ]

    # Calculate receptive field for each layer
    receptive_field = 1
    jump = 1
    output_size = 2000

    receptive_fields = [1]
    jumps = [1]
    output_sizes = [output_size]

    for i in range(1, len(layers)):
        layer = layers[i]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        dilation = layer["dilation"]

        receptive_field = receptive_field + (kernel_size - 1) * jump * dilation
        receptive_fields.append(receptive_field)

        jump = jump * stride
        jumps.append(jump)

        output_size = (output_size - kernel_size * dilation + dilation) // stride + 1
        output_sizes.append(output_size)

    # Create a table with the calculated values
    rf_table = []
    for i, layer in enumerate(layers):
        rf_in_seconds = receptive_fields[i] / sampling_rate  # Convert to seconds

        rf_table.append({
            "Layer": layer["name"],
            "Kernel Size": layer["kernel_size"] if i > 0 else "-",
            "Stride": layer["stride"] if i > 0 else "-",
            "RF (steps)": receptive_fields[i],
            "RF (sec)": rf_in_seconds,
            "Jump": jumps[i],
            "Out Size": output_sizes[i]
        })

    return rf_table

# receptieve field only for one axis, here the cnn1d_ds_wide including seconds
def calculate_receptive_field_cnn1d_ds_wide_sec(sampling_rate=400):
    """
    Calculates and returns the receptive field for each layer of CNN1D_DS_Wide

    Parameters:
    -----------
    sampling_rate : int, optional
        Sampling rate in Hz, default is 400 Hz
    """
    # Define layer properties
    layers = [
        {"name": "Input", "kernel_size": 1, "stride": 1, "dilation": 1},
        {"name": "Conv1D-1", "kernel_size": 25, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-1", "kernel_size": 3, "stride": 2, "dilation": 1},
        {"name": "Conv1D-2", "kernel_size": 15, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-2", "kernel_size": 3, "stride": 2, "dilation": 1},
        {"name": "Conv1D-3", "kernel_size": 9, "stride": 1, "dilation": 1},
        {"name": "MaxPool1D-3", "kernel_size": 3, "stride": 2, "dilation": 1},
    ]

    # Calculate receptive field for each layer
    receptive_field = 1
    jump = 1
    output_size = 2000

    receptive_fields = [1]
    jumps = [1]
    output_sizes = [output_size]

    for i in range(1, len(layers)):
        layer = layers[i]
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        dilation = layer["dilation"]

        receptive_field = receptive_field + (kernel_size - 1) * jump * dilation
        receptive_fields.append(receptive_field)

        jump = jump * stride
        jumps.append(jump)

        output_size = (output_size - kernel_size * dilation + dilation) // stride + 1
        output_sizes.append(output_size)

    # Create a table with the calculated values
    rf_table = []
    for i, layer in enumerate(layers):
        rf_in_seconds = receptive_fields[i] / sampling_rate  # Convert to seconds

        rf_table.append({
            "Layer": layer["name"],
            "Kernel Size": layer["kernel_size"] if i > 0 else "-",
            "Stride": layer["stride"] if i > 0 else "-",
            "RF (steps)": receptive_fields[i],
            "RF (sec)": rf_in_seconds,
            "Jump": jumps[i],
            "Out Size": output_sizes[i]
        })

    return rf_table


def plot_receptive_field_cnn1d_wide(save_path=None, actual_signal=None):
    """
    Creates a visual representation of receptive field growth through CNN1D_Wide layers
    with fixed alignment issues
    """
    rf_table = calculate_receptive_field_cnn1d_wide()

    # Extract data for visualization
    layer_names = [item["Layer"] for item in rf_table]
    receptive_fields = [item["Receptive Field"] for item in rf_table]
    output_sizes = [item["Output Size"] for item in rf_table]
    jumps = [item["Jump"] for item in rf_table]

    # Create a colormap for the bars
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layer_names)))

    # Create figure with more appropriate size and layout
    fig = plt.figure(figsize=(15, 12))

    # Create main plot for receptive field
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    bars = ax1.barh(range(len(layer_names)), receptive_fields, color=colors, height=0.7)

    # Annotate bars with RF size
    for i, (bar, rf) in enumerate(zip(bars, receptive_fields)):
        ax1.text(
            rf + 5, bar.get_y() + bar.get_height()/2,
            f"{rf}", va='center', fontweight='bold'
        )

    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel('Receptive Field Size (Time Steps)')
    ax1.set_title('Growth of Receptive Field')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Create a table for numerical values
    ax_table = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
    table_data = [[item["Layer"],
                  item["Kernel Size"],
                  item["Stride"],
                  item["Receptive Field"],
                  item["Jump"],
                  item["Output Size"]] for item in rf_table]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Layer", "K", "S", "RF", "Jump", "Out Size"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.axis('off')
    ax_table.set_title('Layer-wise RF Calculations')

    # Create a plot showing the output feature map sizes - MOVED TO MIDDLE ROW
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1)
    ax3.plot(range(len(layer_names)), output_sizes, 'bo-', linewidth=2, markersize=8)
    for i, size in enumerate(output_sizes):
        ax3.text(i, size + max(output_sizes)*0.05, f"{size}", ha='center')

    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45, ha='right')
    ax3.set_ylabel('Feature Map Size (Time Steps)')
    ax3.set_title('Output Feature Map Size')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add model architecture diagram in its own dedicated subplot - SEPARATE FROM CHART
    ax_arch = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    model_architecture = """
    Input (3, 2000)
       ↓
    Conv1D (k=25, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=15, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=9, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=5, s=1) → ReLU → MaxPool1D (k=2, s=2) → Dropout
       ↓
    Global Average Pooling
       ↓
    FC (128→64) → ReLU → Dropout
       ↓
    FC (64→2)
    """
    ax_arch.text(0.5, 0.5, model_architecture, fontfamily='monospace', fontsize=11,
                ha='center', va='center', transform=ax_arch.transAxes)
    ax_arch.axis('off')
    ax_arch.set_title('CNN1D_Wide Architecture')

    # Create visualization of the actual receptive field - MOVED TO BOTTOM ROW, FULL WIDTH
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    ''''# Generate a synthetic signal for visualization (or use your actual signal here)
    time_steps = 2000
    x = np.linspace(0, time_steps-1, time_steps)
    # Create a signal with various frequency components
    signal = (np.sin(x/10) + 0.5*np.sin(x/5) + 0.3*np.sin(x/20) +
              0.2*np.random.normal(0, 1, time_steps))

    # Normalize the signal
    signal = (signal - signal.min()) / (signal.max() - signal.min())'''

    # Generate signal data (either synthetic or use provided actual signal)
    time_steps = 2000
    x = np.linspace(0, time_steps-1, time_steps)

    # Use actual signal for visualization (assuming you have a signal in a variable called 'actual_signal')
    if actual_signal is not None:
        # Use the provided actual signal
        signal = actual_signal
        if len(signal.shape) > 1:
            # If multi-dimensional, take one axis (e.g., X-axis)
            signal = signal[0]
    else:
        # Create a synthetic signal with various frequency components
        signal = (np.sin(x/10) + 0.5*np.sin(x/5) + 0.3*np.sin(x/20) +
                  0.2*np.random.normal(0, 1, time_steps))

    # Normalize the signal for visualization
    signal = (signal - signal.min()) / (signal.max() - signal.min())

    # Plot the signal
    ax4.plot(x, signal, 'k-', linewidth=0.8, alpha=0.7)



    # Highlight receptive fields at different layers
    center_point = time_steps // 2
    highlight_layers = [0, 2, 4, 6, 8]  # Input, after Conv1+Pool1, after Conv2+Pool2, etc.

    for i, layer_idx in enumerate(highlight_layers):
        rf = receptive_fields[layer_idx]
        half_rf = rf // 2
        color = plt.cm.viridis(0.1 + 0.8 * i / len(highlight_layers))

        # Highlight the receptive field
        ax4.axvspan(
            max(0, center_point - half_rf),
            min(time_steps-1, center_point + half_rf),
            color=color,
            alpha=0.3,
            label=f"{layer_names[layer_idx]}: RF={rf}"
        )

    ax4.set_xlim([center_point - 350, center_point + 350])  # Zoom in to see details
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Signal Amplitude')
    ax4.set_title('Receptive Field Visualization on Signal')
    ax4.legend(loc='upper right', fontsize=8)

    # Add title with final receptive field size
    plt.suptitle(f"CNN1D_Wide: Receptive Field Analysis (Final RF = {receptive_fields[-1]} time steps)",
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, rf_table


def plot_receptive_field_cnn1d_ds_wide(save_path=None, actual_signal=None):
    """
    Creates a visual representation of receptive field growth through CNN1D_DS_Wide layers
    with improved layout to avoid alignment issues

    Args:
        save_path: Path to save the generated figure
        actual_signal: Optional real signal data to use instead of synthetic
    """
    rf_table = calculate_receptive_field_cnn1d_ds_wide()

    # Extract data for visualization
    layer_names = [item["Layer"] for item in rf_table]
    receptive_fields = [item["Receptive Field"] for item in rf_table]
    output_sizes = [item["Output Size"] for item in rf_table]
    jumps = [item["Jump"] for item in rf_table]

    # Create a colormap for the bars
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layer_names)))

    # Create figure with more appropriate size and layout
    fig = plt.figure(figsize=(15, 12))

    # Create main plot for receptive field
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=1)
    bars = ax1.barh(range(len(layer_names)), receptive_fields, color=colors, height=0.7)

    # Annotate bars with RF size
    for i, (bar, rf) in enumerate(zip(bars, receptive_fields)):
        ax1.text(
            rf + 3, bar.get_y() + bar.get_height()/2,
            f"{rf}", va='center', fontweight='bold'
        )

    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel('Receptive Field Size (Time Steps)')
    ax1.set_title('Growth of Receptive Field')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Create a table for numerical values
    ax_table = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
    table_data = [[item["Layer"],
                  item["Kernel Size"],
                  item["Stride"],
                  item["Receptive Field"],
                  item["Jump"],
                  item["Output Size"]] for item in rf_table]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Layer", "K", "S", "RF", "Jump", "Out Size"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.axis('off')
    ax_table.set_title('Layer-wise RF Calculations')

    # Create a plot showing the output feature map sizes - MOVED TO MIDDLE ROW
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=1)
    ax3.plot(range(len(layer_names)), output_sizes, 'bo-', linewidth=2, markersize=8)
    for i, size in enumerate(output_sizes):
        ax3.text(i, size + max(output_sizes)*0.05, f"{size}", ha='center')

    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45, ha='right')
    ax3.set_ylabel('Feature Map Size (Time Steps)')
    ax3.set_title('Output Feature Map Size')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add model architecture diagram in its own dedicated subplot - SEPARATE FROM CHART
    ax_arch = plt.subplot2grid((3, 3), (1, 1), colspan=2)
    model_architecture = """

    Input (3, 2000)
       ↓
    Conv1D (k=25, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Conv1D (k=15, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Conv1D (k=9, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Global Average Pooling
       ↓
    FC (64→64) → ReLU → Dropout(0.3)
       ↓
    FC (64→2)
    """
    ax_arch.text(0.5, 0.5, model_architecture, fontfamily='monospace', fontsize=11,
                ha='center', va='center', transform=ax_arch.transAxes)
    ax_arch.axis('off')
    ax_arch.set_title('CNN1D_DS_Wide Architecture')

    # Create visualization of the actual receptive field - MOVED TO BOTTOM ROW, FULL WIDTH
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

    # Generate signal data (either synthetic or use provided actual signal)
    time_steps = 2000
    x = np.linspace(0, time_steps-1, time_steps)

    if actual_signal is not None:
        # Use the provided actual signal
        signal = actual_signal
        if len(signal.shape) > 1:
            # If multi-dimensional, take one axis (e.g., X-axis)
            signal = signal[0]
    else:
        # Create a synthetic signal with various frequency components
        signal = (np.sin(x/10) + 0.5*np.sin(x/5) + 0.3*np.sin(x/20) +
                  0.2*np.random.normal(0, 1, time_steps))

    # Normalize the signal for visualization
    signal = (signal - signal.min()) / (signal.max() - signal.min())

    # Plot the signal
    ax4.plot(x, signal, 'k-', linewidth=0.8, alpha=0.7)

    # Highlight receptive fields at different layers
    center_point = time_steps // 2
    highlight_layers = [0, 2, 4, 6]  # Input, after Conv1+Pool1, after Conv2+Pool2, etc.

    for i, layer_idx in enumerate(highlight_layers):
        rf = receptive_fields[layer_idx]
        half_rf = rf // 2
        color = plt.cm.viridis(0.1 + 0.8 * i / len(highlight_layers))

        # Highlight the receptive field
        ax4.axvspan(
            max(0, center_point - half_rf),
            min(time_steps-1, center_point + half_rf),
            color=color,
            alpha=0.3,
            label=f"{layer_names[layer_idx]}: RF={rf}"
        )

    ax4.set_xlim([center_point - 350, center_point + 350])  # Zoom in to see details
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Signal Amplitude')
    ax4.set_title('Receptive Field Visualization on Signal')
    ax4.legend(loc='upper right', fontsize=8)

    # Add title with final receptive field size
    plt.suptitle(f"CNN1D_DS_Wide: Receptive Field Analysis (Final RF = {receptive_fields[-1]} time steps)",
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, rf_table

def plot_receptive_field_cnn1d_ds_wide_multiaxis(signal_data, save_path=None, sampling_rate=400):
    """
    Creates a visual representation of receptive field growth through CNN1D_DS_Wide layers
    with support for multi-axis signals and including receptive field in seconds

    Parameters:
    -----------
    signal_data : numpy.ndarray
        Signal data with shape (3, 2000) containing the 3 axes of vibration data
    save_path : str, optional
        Path to save the figure
    sampling_rate : int, optional
        Sampling rate in Hz, default is 400 Hz
    """
    rf_table = calculate_receptive_field_cnn1d_ds_wide_sec(sampling_rate)

    # Extract data for visualization
    layer_names = [item["Layer"] for item in rf_table]
    receptive_fields = [item["RF (steps)"] for item in rf_table]
    receptive_fields_sec = [item["RF (sec)"] for item in rf_table]
    output_sizes = [item["Out Size"] for item in rf_table]
    jumps = [item["Jump"] for item in rf_table]

    # Create a colormap for the bars and receptive fields
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layer_names)))

    # Create figure with more appropriate size
    fig = plt.figure(figsize=(15, 18))

    # Create main plot for receptive field
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=1)
    bars = ax1.barh(range(len(layer_names)), receptive_fields, color=colors, height=0.7)

    # Annotate bars with RF size in steps and seconds
    for i, (bar, rf, rf_sec) in enumerate(zip(bars, receptive_fields, receptive_fields_sec)):
        ax1.text(
            rf + 5, bar.get_y() + bar.get_height()/2,
            f"{rf} steps ({rf_sec:.4f} s)", va='center', fontweight='bold'
        )

    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel('Receptive Field Size (Time Steps)')
    ax1.set_title('Growth of Receptive Field')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Create a table for numerical values
    ax_table = plt.subplot2grid((4, 3), (0, 2), rowspan=1)
    table_data = [[item["Layer"],
                  item["Kernel Size"],
                  item["Stride"],
                  f"{item['RF (steps)']} ({item['RF (sec)']:.4f}s)",
                  item["Jump"],
                  item["Out Size"]] for item in rf_table]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Layer", "K", "S", "RF (steps/sec)", "Jump", "Out Size"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.axis('off')
    ax_table.set_title('Layer-wise RF Calculations')

    # Create a plot showing the output feature map sizes
    ax3 = plt.subplot2grid((4, 3), (1, 0), colspan=1)
    ax3.plot(range(len(layer_names)), output_sizes, 'bo-', linewidth=2, markersize=8)
    for i, size in enumerate(output_sizes):
        ax3.text(i, size + max(output_sizes)*0.05, f"{size}", ha='center')

    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45, ha='right')
    ax3.set_ylabel('Feature Map Size (Time Steps)')
    ax3.set_title('Output Feature Map Size')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add model architecture diagram in its own dedicated subplot
    ax_arch = plt.subplot2grid((4, 3), (1, 1), colspan=2)
    model_architecture = """
    Input (3, 2000)
       ↓
    Conv1D (k=25, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Conv1D (k=15, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Conv1D (k=9, s=1) → GroupNorm → ReLU → MaxPool1D (k=3, s=2)
       ↓
    Global Average Pooling
       ↓
    FC (64→64) → ReLU → Dropout(0.3)
       ↓
    FC (64→2)
    """
    ax_arch.text(0.5, 0.5, model_architecture, fontfamily='monospace', fontsize=11,
                ha='center', va='center', transform=ax_arch.transAxes)
    ax_arch.axis('off')
    ax_arch.set_title('CNN1D_DS_Wide Architecture')

    # Get signal dimensions and verify shape
    if signal_data.shape[0] != 3 or len(signal_data.shape) != 2:
        raise ValueError(f"Expected signal_data shape (3, time_steps), got {signal_data.shape}")

    time_steps = signal_data.shape[1]
    center_point = time_steps // 2

    # Highlight layers for receptive field visualization
    highlight_layers = list(range(len(layer_names)))  # All layers

    # Create a visualization for each axis
    axis_indices = [0, 1, 2]  # Indices for the three axes

    for i, axis_idx in enumerate(axis_indices):
        ax = plt.subplot2grid((4, 3), (2 + i//3, i % 3))

        # Plot the signal for this axis
        x = np.arange(time_steps)
        signal = signal_data[axis_idx]

        ax.plot(x, signal, 'k-', linewidth=0.8, alpha=0.7)

        # Create an array of colors with increasing opacity
        alphas = np.linspace(0.15, 0.45, len(highlight_layers))

        # Visualize receptive fields with different alpha levels for each layer
        for j, layer_idx in enumerate(highlight_layers):
            rf = receptive_fields[layer_idx]
            rf_sec = receptive_fields_sec[layer_idx]
            half_rf = rf // 2

            color = colors[layer_idx].copy()

            # Draw receptive field overlay
            ax.axvspan(
                max(0, center_point - half_rf),
                min(time_steps-1, center_point + half_rf),
                color=color,
                alpha=alphas[j],
                label=f"{layer_names[layer_idx]}: RF={rf} ({rf_sec:.4f}s)"
            )

        # Set axis limits to zoom in on the center portion
        ax.set_xlim([center_point - 350, center_point + 350])

        # Set labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title(f'Receptive Field on Axis {axis_idx+1}')

        # Add legend for the first axis only to avoid repetition
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Add title with final receptive field size in both steps and seconds
    final_rf = receptive_fields[-1]
    final_rf_sec = receptive_fields_sec[-1]
    plt.suptitle(f"CNN1D_DS_Wide: Receptive Field Analysis (Final RF = {final_rf} steps / {final_rf_sec:.4f} seconds)",
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, rf_table

def plot_receptive_field_cnn1d_wide_multiaxis(signal_data, save_path=None, sampling_rate=400):
    """
    Creates a visual representation of receptive field growth through CNN1D_Wide layers
    with support for multi-axis signals and including receptive field in seconds

    Parameters:
    -----------
    signal_data : numpy.ndarray
        Signal data with shape (3, 2000) containing the 3 axes of vibration data
    save_path : str, optional
        Path to save the figure
    sampling_rate : int, optional
        Sampling rate in Hz, default is 400 Hz
    """
    rf_table = calculate_receptive_field_cnn1d_wide_sec(sampling_rate)

    # Extract data for visualization
    layer_names = [item["Layer"] for item in rf_table]
    receptive_fields = [item["RF (steps)"] for item in rf_table]
    receptive_fields_sec = [item["RF (sec)"] for item in rf_table]
    output_sizes = [item["Out Size"] for item in rf_table]
    jumps = [item["Jump"] for item in rf_table]

    # Create a colormap for the bars and receptive fields
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(layer_names)))

    # Create figure with more appropriate size
    fig = plt.figure(figsize=(15, 18))

    # Create main plot for receptive field
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=1)
    bars = ax1.barh(range(len(layer_names)), receptive_fields, color=colors, height=0.7)

    # Annotate bars with RF size in steps and seconds
    for i, (bar, rf, rf_sec) in enumerate(zip(bars, receptive_fields, receptive_fields_sec)):
        ax1.text(
            rf + 20, bar.get_y() + bar.get_height()/2,
            f"{rf} steps ({rf_sec:.4f} s)", va='center', fontweight='bold'
        )

    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel('Receptive Field Size (Time Steps)')
    ax1.set_title('Growth of Receptive Field')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)

    # Create a table for numerical values
    ax_table = plt.subplot2grid((4, 3), (0, 2), rowspan=1)
    table_data = [[item["Layer"],
                  item["Kernel Size"],
                  item["Stride"],
                  f"{item['RF (steps)']} ({item['RF (sec)']:.4f}s)",
                  item["Jump"],
                  item["Out Size"]] for item in rf_table]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Layer", "K", "S", "RF (steps/sec)", "Jump", "Out Size"],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.axis('off')
    ax_table.set_title('Layer-wise RF Calculations')

    # Create a plot showing the output feature map sizes
    ax3 = plt.subplot2grid((4, 3), (1, 0), colspan=1)
    ax3.plot(range(len(layer_names)), output_sizes, 'bo-', linewidth=2, markersize=8)
    for i, size in enumerate(output_sizes):
        ax3.text(i, size + max(output_sizes)*0.05, f"{size}", ha='center')

    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45, ha='right')
    ax3.set_ylabel('Feature Map Size (Time Steps)')
    ax3.set_title('Output Feature Map Size')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Add model architecture diagram in its own dedicated subplot
    ax_arch = plt.subplot2grid((4, 3), (1, 1), colspan=2)
    model_architecture = """
    Input (3, 2000)
       ↓
    Conv1D (k=25, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=15, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=9, s=1) → ReLU → MaxPool1D (k=4, s=4) → Dropout
       ↓
    Conv1D (k=5, s=1) → ReLU → MaxPool1D (k=2, s=2) → Dropout
       ↓
    Global Average Pooling
       ↓
    FC (128→64) → ReLU → Dropout
       ↓
    FC (64→2)
    """
    ax_arch.text(0.5, 0.5, model_architecture, fontfamily='monospace', fontsize=11,
                ha='center', va='center', transform=ax_arch.transAxes)
    ax_arch.axis('off')
    ax_arch.set_title('CNN1D_Wide Architecture')

    # Get signal dimensions and verify shape
    if signal_data.shape[0] != 3 or len(signal_data.shape) != 2:
        raise ValueError(f"Expected signal_data shape (3, time_steps), got {signal_data.shape}")

    time_steps = signal_data.shape[1]
    center_point = time_steps // 2

    # Highlight layers for receptive field visualization
    highlight_layers = list(range(len(layer_names)))  # All layers

    # Create a visualization for each axis
    axis_indices = [0, 1, 2]  # Indices for the three axes

    for i, axis_idx in enumerate(axis_indices):
        ax = plt.subplot2grid((4, 3), (2 + i//3, i % 3))

        # Plot the signal for this axis
        x = np.arange(time_steps)
        signal = signal_data[axis_idx]

        ax.plot(x, signal, 'k-', linewidth=0.8, alpha=0.7)

        # Create an array of colors with increasing opacity
        alphas = np.linspace(0.15, 0.45, len(highlight_layers))

        # Visualize receptive fields with different alpha levels for each layer
        for j, layer_idx in enumerate(highlight_layers):
            rf = receptive_fields[layer_idx]
            rf_sec = receptive_fields_sec[layer_idx]
            half_rf = rf // 2

            color = colors[layer_idx].copy()

            # Draw receptive field overlay
            ax.axvspan(
                max(0, center_point - half_rf),
                min(time_steps-1, center_point + half_rf),
                color=color,
                alpha=alphas[j],
                label=f"{layer_names[layer_idx]}: RF={rf} ({rf_sec:.4f}s)"
            )

        # Set axis limits to zoom in on the center portion
        ax.set_xlim([center_point - 350, center_point + 350])

        # Set labels and title
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Signal Amplitude')
        ax.set_title(f'Receptive Field on Axis {axis_idx+1}')

        # Add legend for the first axis only to avoid repetition
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)

    # Add title with final receptive field size in both steps and seconds
    final_rf = receptive_fields[-1]
    final_rf_sec = receptive_fields_sec[-1]
    plt.suptitle(f"CNN1D_Wide: Receptive Field Analysis (Final RF = {final_rf} steps / {final_rf_sec:.4f} seconds)",
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, rf_table