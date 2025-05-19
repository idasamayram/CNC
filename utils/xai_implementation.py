import numpy as np
import torch
from utils.lrp_utils import zennit_relevance  # Import from DFT-LRP utils
import utils.lrp_utils as lrp_utils  # For zennit_relevance
from utils.fft_lrp import FFTLRP
from numpy.fft import fftfreq
from utils.dft_lrp import DFTLRP


def compute_lrp_relevance(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute LRP relevances for a single vibration sample.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, 10000) for the vibration data   (3,2000) for downsampled
        label: Optional integer label (0 or 1) for the sample. If None, use model prediction as target
        device: Torch device (CPU or CUDA)

    Returns:
        relevance: Numpy array of shape (3, 10000) containing LRP relevances
        input_signal: Numpy array of the input signal (for visualization)
        predicted_label: Predicted label if label is None
    """
    # Ensure sample is a PyTorch tensor with shape (1, 3, 10000)
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension and move to device
    else:
        sample = sample.to(device).unsqueeze(0)  # Assume it's already a tensor, add batch dimension and move to device

    # Move model to the specified device (already done in your script, but ensure consistency)
    model = model.to(device)
    model.eval()

    # Debug: Print device information
    print(f"Sample device: {sample.device}")
    print(f"Model device: {next(model.parameters()).device}")

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            outputs = model(sample)
            _, predicted_label = torch.max(outputs, 1)  # Get predicted class (0 or 1)
        target = predicted_label.item()
    else:
        target = label  # Use provided label as target
        target = torch.tensor([target], device=device)  # Ensure target is on the same device

    # Debug: Print target device
    print(f"Target device: {target.device}")

    # Compute LRP relevances using Zennit
    try:
        relevance = zennit_relevance(
            input=sample,
            model=model,
            target=target,  # Target is already on the correct device
            attribution_method="lrp",  # Use LRP
            zennit_choice="EpsilonPlus",  # Use EpsilonPlus rule (stable for neural networks)
            rel_is_model_out=True,  # Relevance is model output (logits)
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        print(f"Error in zennit_relevance: {e}")
        raise

    # Remove batch dimension and convert to numpy
    relevance = relevance.squeeze(0)  # Shape: (3, 10000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 10000)

    return relevance, input_signal, target.item() if label is None else label


'''
def compute_dft_lrp_relevance(
    model,
    sample,
    label=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    signal_length=2000,  # Updated to match downsampled data
    leverage_symmetry=True,
    precision=32,
    create_stdft=False,
    create_inverse=False,
    sampling_rate=400  # Adjusted based on downsampling assumption
):
    """
    Compute LRP relevances for a single vibration sample in both time and frequency domains.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, 10000) for the vibration data
        label: Optional integer label (0 or 1). If None, use model prediction
        device: Torch device (CPU or CUDA)
        signal_length: Length of the signal (10000 for your dataset)
        batch_size: Batch size for DFT-LRP processing
        leverage_symmetry: Use symmetry in DFT (reduces frequency bins to positive frequencies)
        precision: 32 or 16 for DFTLRP
        create_stdft: Whether to create STDFT layers (not needed for now)
        create_inverse: Whether to create inverse DFT layers (not needed for now)
        sampling_rate: Sampling rate of the data in Hz (e.g., 10000 for 10 kHz)

    Returns:
        relevance_time: Numpy array of shape (3, 10000) with time-domain relevances  [3,2000 for downssampled]
        relevance_freq: Numpy array of shape (3, 5001) with frequency-domain relevances
        signal_freq: Numpy array of shape (3, 5001) with frequency-domain signal
        input_signal: Numpy array of shape (3, 10000) with the input signal
        freqs: Frequency bins (for visualization)
        predicted_label: Predicted label if label is None
    """
    # Ensure sample is a PyTorch tensor with shape (1, 3, 10000)
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        sample = sample.to(device).unsqueeze(0)

    model = model.to(device)
    model.eval()

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            outputs = model(sample)
            _, predicted_label = torch.max(outputs, 1)
        target = predicted_label.item()
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)

    # Compute LRP relevances in the time domain using Zennit
    relevance_time = lrp_utils.zennit_relevance(
        input=sample,
        model=model,
        target=target,
        attribution_method="lrp",
        zennit_choice="EpsilonPlus",
        rel_is_model_out=True,
        cuda=(device == "cuda")
    )
    relevance_time = relevance_time.squeeze(0)  # Shape: (3, 10000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 10000)

    # Initialize DFTLRP for frequency-domain propagation
    dftlrp = DFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=create_stdft,
        create_inverse=create_inverse
    )

    # Prepare for frequency-domain propagation
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    print(n_axes)
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length  # 5001 with symmetry
    print(freq_length)
    signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.empty((n_axes, freq_length))

    # Process each axis separately
    for axis in range(n_axes):
        signal_axis = input_signal[axis:axis+1, :]
        relevance_axis = relevance_time[axis:axis+1, :]
        signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
            relevance=relevance_axis,
            signal=signal_axis,
            real=False,
            short_time=False,
            epsilon=1e-6
        )
        signal_freq[axis] = signal_freq_axis[0]
        relevance_freq[axis] = relevance_freq_axis[0]

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0/sampling_rate)[:freq_length]  # Scaled by sampling rate

    # Clean up to free memory
    del dftlrp

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target.item() if label is None else label
'''
def compute_dft_lrp_relevance(
    model,
    sample,
    label=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    signal_length=2000,
    leverage_symmetry=True,
    precision=32,
    create_stdft=False,
    create_inverse=False,
    sampling_rate=400
):
    # Ensure sample is a PyTorch tensor with shape (1, 3, 2000)
    if isinstance(sample, np.ndarray):
        assert sample.shape == (3, 2000), f"Expected sample shape (3, 2000), got {sample.shape}"
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        assert sample.shape[1:] == (3, 2000), f"Expected sample shape [batch, 3, 2000], got {sample.shape}"
        sample = sample.to(device)

    model = model.to(device)
    model.eval()

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            outputs = model(sample)
            _, predicted_label = torch.max(outputs, 1)
        target = predicted_label.item()
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)

    # Compute LRP relevances in the time domain using Zennit
    relevance_time = lrp_utils.zennit_relevance(
        input=sample,
        model=model,
        target=target,
        attribution_method="lrp",
        zennit_choice="EpsilonPlus",
        rel_is_model_out=True,
        cuda=(device == "cuda")
    )
    assert relevance_time.shape == (1, 3, 2000), f"Expected relevance_time shape (1, 3, 2000), got {relevance_time.shape}"
    relevance_time = relevance_time.squeeze(0)  # Shape: (3, 2000)
    assert relevance_time.shape == (3, 2000), f"Expected squeezed relevance_time shape (3, 2000), got {relevance_time.shape}"

    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 2000)
    assert input_signal.shape == (3, 2000), f"Expected input_signal shape (3, 2000), got {input_signal.shape}"

    # Initialize DFTLRP for frequency-domain propagation
    dftlrp = DFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=create_stdft,
        create_inverse=create_inverse
    )

    # Process all axes together
    signal_freq, relevance_freq = dftlrp.dft_lrp(
        relevance=relevance_time[np.newaxis, :],  # Shape: [1, 3, 2000]
        signal=input_signal[np.newaxis, :],       # Shape: [1, 3, 2000]
        real=False,
        short_time=False,
        epsilon=1e-6
    )
    assert signal_freq.shape == (1, 3, signal_length // 2 + 1), f"Expected signal_freq shape (1, 3, {signal_length // 2 + 1}), got {signal_freq.shape}"
    assert relevance_freq.shape == (1, 3, signal_length // 2 + 1), f"Expected relevance_freq shape (1, 3, {signal_length // 2 + 1}), got {relevance_freq.shape}"
    signal_freq = signal_freq.squeeze(0)  # Shape: [3, 1001]
    relevance_freq = relevance_freq.squeeze(0)  # Shape: [3, 1001]
    assert signal_freq.shape == (3, signal_length // 2 + 1), f"Expected squeezed signal_freq shape (3, {signal_length // 2 + 1}), got {signal_freq.shape}"
    assert relevance_freq.shape == (3, signal_length // 2 + 1), f"Expected squeezed relevance_freq shape (3, {signal_length // 2 + 1}), got {relevance_freq.shape}"

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0/sampling_rate)[:signal_length // 2 + 1]


    # Clean up to free memory
    del dftlrp

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target if label is None else label

def compute_fft_lrp_relevance(
    model,
    sample,
    label=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    signal_length=2000,
    batch_size=1,
    leverage_symmetry=True,
    sampling_rate=400,
    compute_timefreq=True,
    window_shift=1,
    window_width=128,
    window_shape="rectangle"
):
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        sample = sample.to(device).unsqueeze(0)
    print(f"Input sample shape: {sample.shape}")

    model = model.to(device)
    model.eval()

    if label is None:
        with torch.no_grad():
            outputs = model(sample)
            _, predicted_label = torch.max(outputs, 1)
        target = predicted_label.item()
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)
    print(f"Target label: {target}")

    relevance_time_tensor = lrp_utils.zennit_relevance(
        input=sample,
        model=model,
        target=target,
        attribution_method="lrp",
        zennit_choice="EpsilonPlus",
        rel_is_model_out=True,
        cuda=(device == "cuda")
    )
    print(f"relevance_time_tensor shape before squeeze: {relevance_time_tensor.shape}")

    if relevance_time_tensor.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got shape {relevance_time_tensor.shape}")
    relevance_time = relevance_time_tensor.squeeze(0)
    print(f"relevance_time shape after squeeze: {relevance_time.shape}")

    input_signal = sample.squeeze(0).detach().cpu().numpy()
    print(f"input_signal shape: {input_signal.shape}")

    if relevance_time.shape != input_signal.shape:
        raise ValueError(f"Shape mismatch: relevance_time {relevance_time.shape}, input_signal {input_signal.shape}")

    fftlrp = FFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        window_shift=window_shift,
        window_width=window_width,
        window_shape="rectangle",
        create_inverse=True,
        create_transpose_inverse=True,
        create_forward=True,
        create_stdft=compute_timefreq
    )

    n_axes = input_signal.shape[0]
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.empty((n_axes, freq_length), dtype=np.complex128)

    for axis in range(n_axes):
        signal_axis = input_signal[axis:axis+1, :]
        relevance_axis = relevance_time[axis:axis+1, :]
        signal_hat, relevance_hat = fftlrp.fft_lrp(
            relevance=relevance_axis,
            signal=signal_axis,
            short_time=False,
            epsilon=1e-6,
            real=False
        )
        signal_freq[axis] = signal_hat.squeeze(0)
        relevance_freq[axis] = relevance_hat.squeeze(0)

    freqs = np.fft.rfftfreq(signal_length, d=1.0/sampling_rate)

    relevance_timefreq = None
    signal_timefreq = None
    if compute_timefreq:
        signal_timefreq = np.empty((n_axes, freq_length, 20), dtype=np.complex128)  # Fixed to 20
        relevance_timefreq = np.empty((n_axes, freq_length, 20), dtype=np.complex128)
        for axis in range(n_axes):  # Loop over all axes
            signal_axis = input_signal[axis]
            relevance_axis = relevance_time[axis]
            signal_hat, relevance_hat = fftlrp.fft_lrp(
                relevance=relevance_axis[np.newaxis, :],
                signal=signal_axis[np.newaxis, :],
                short_time=True,
                epsilon=1e-6,
                real=False
            )
            signal_timefreq[axis] = signal_hat.squeeze(0)[:, :20]  # Take first 20 frames
            relevance_timefreq[axis] = relevance_hat.squeeze(0)[:, :20]
        print(f"signal_timefreq shape after selection: {signal_timefreq.shape}, relevance_timefreq shape after selection: {relevance_timefreq.shape}")

    del fftlrp
    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target if label is None else label)