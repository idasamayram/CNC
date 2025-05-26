import numpy as np
import torch
from utils.lrp_utils import zennit_relevance  # Import from DFT-LRP utils
import utils.lrp_utils as lrp_utils  # For zennit_relevance
from utils.fft_lrp import FFTLRP
from numpy.fft import fftfreq
from utils.dft_lrp import DFTLRP
import gc

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
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(
            0)  # Add batch dimension and move to device
    else:
        sample = sample.to(device).unsqueeze(0)  # Assume it's already a tensor, add batch dimension and move to device

    # Move model to the specified device (already done in your script, but ensure consistency)
    model = model.to(device)
    model.eval()

    # Make sure model and sample are on the same device
    if next(model.parameters()).device != sample.device:
        print(f"Warning: Model device ({next(model.parameters()).device}) doesn't match sample device ({sample.device})")
        print("Moving model to match sample device")
        model = model.to(sample.device)

    # Debug: Print device information
    print(f"Sample device: {sample.device}")
    print(f"Model device: {next(model.parameters()).device}")

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample)
                _, predicted_label = torch.max(outputs, 1)  # Get predicted class (0 or 1)
                target = predicted_label.item()
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {e}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
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
        # Try to recover by falling back to CPU if possible
        if device == "cuda":
            print("Falling back to CPU for LRP computation")
            device = "cpu"
            model = model.cpu()
            sample = sample.cpu()
            if isinstance(target, torch.Tensor):
                target = target.cpu()
            try:
                relevance = zennit_relevance(
                    input=sample,
                    model=model,
                    target=target,
                    attribution_method="lrp",
                    zennit_choice="EpsilonPlus",
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    # Remove batch dimension and convert to numpy
    relevance = relevance.squeeze(0)  # Shape: (3, 10000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 10000)

    return relevance, input_signal, target.item() if isinstance(target, torch.Tensor) else target


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

    # Make sure model and sample are on the same device
    model = model.to(device)
    model.eval()

    if next(model.parameters()).device != sample.device:
        print(f"Warning: Model device ({next(model.parameters()).device}) doesn't match sample device ({sample.device})")
        print("Moving model to match sample device")
        model = model.to(sample.device)

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample)
                _, predicted_label = torch.max(outputs, 1)
                target = predicted_label.item()
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {e}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)

    # Compute LRP relevances in the time domain using Zennit
    try:
        relevance_time = lrp_utils.zennit_relevance(
            input=sample,
            model=model,
            target=target,
            attribution_method="lrp",
            zennit_choice="EpsilonPlus",
            rel_is_model_out=True,
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        print(f"Error in zennit_relevance: {e}")
        # Try to recover by falling back to CPU if possible
        if device == "cuda":
            print("Falling back to CPU for LRP computation")
            device = "cpu"
            model = model.cpu()
            sample = sample.cpu()
            if isinstance(target, torch.Tensor):
                target = target.cpu()
            try:
                relevance_time = lrp_utils.zennit_relevance(
                    input=sample,
                    model=model,
                    target=target,
                    attribution_method="lrp",
                    zennit_choice="EpsilonPlus",
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    relevance_time = relevance_time.squeeze(0)  # Shape: (3, 10000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 10000)

    print(f"Input sample shape: {sample.shape}")  # Should be [1, 3, 2000]
    print(f"Relevance time shape: {relevance_time.shape}")  # Should be [1, 3, 2000] before squeeze, [3, 2000] after
    print(f"Input signal shape: {input_signal.shape}")  # Should be [3, 2000]

    # Verify shapes are consistent
    if relevance_time.shape != input_signal.shape:
        raise ValueError(f"Shape mismatch: relevance_time {relevance_time.shape}, input_signal {input_signal.shape}")

    # Initialize DFTLRP for frequency-domain propagation
    try:
        dftlrp = DFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=(device == "cuda"),
            create_stdft=create_stdft,
            create_inverse=create_inverse
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing DFTLRP: {e}")

    # Prepare for frequency-domain propagation
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    print(f'Number of axis is: {n_axes}')
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length  # 5001 with symmetry
    print(f'Frequency length is:{freq_length}')
    signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.empty((n_axes, freq_length))

    # Process each axis separately
    for axis in range(n_axes):
        try:
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
        except Exception as e:
            print(f"Error processing axis {axis}: {e}")
            # Fill with zeros if processing fails
            signal_freq[axis] = np.zeros(freq_length, dtype=np.complex128)
            relevance_freq[axis] = np.zeros(freq_length)

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0/sampling_rate)[:freq_length]  # Scaled by sampling rate

    # Clean up to free memory
    try:
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target.item() if isinstance(target, torch.Tensor) else target


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
    # Convert sample to tensor if it's a numpy array
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        sample = sample.to(device).unsqueeze(0)

    # Verify input shape
    if len(sample.shape) != 3 or sample.shape[1] != 3:
        raise ValueError(f"Expected input shape (batch_size, 3, time_steps), got {sample.shape}")

    print(f"Input sample shape: {sample.shape}")

    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Make sure model and sample are on the same device
    if next(model.parameters()).device != sample.device:
        print(f"Warning: Model device ({next(model.parameters()).device}) doesn't match sample device ({sample.device})")
        print("Moving model to match sample device")
        model = model.to(sample.device)

    # Determine target (label or model prediction)
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample)
                _, predicted_label = torch.max(outputs, 1)
                target = predicted_label.item()
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {str(e)}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)
    print(f"Target label: {target}")

    # Compute relevances in time domain using Zennit
    try:
        relevance_time_tensor = lrp_utils.zennit_relevance(
            input=sample,
            model=model,
            target=target,
            attribution_method="lrp",
            zennit_choice="EpsilonPlus",
            rel_is_model_out=True,
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        print(f"Error in zennit_relevance: {e}")
        # Try to recover by falling back to CPU if possible
        if device == "cuda":
            print("Falling back to CPU for LRP computation")
            device = "cpu"
            model = model.cpu()
            sample = sample.cpu()
            if isinstance(target, torch.Tensor):
                target = target.cpu()
            try:
                relevance_time_tensor = lrp_utils.zennit_relevance(
                    input=sample,
                    model=model,
                    target=target,
                    attribution_method="lrp",
                    zennit_choice="EpsilonPlus",
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    print(f"relevance_time_tensor shape before squeeze: {relevance_time_tensor.shape}")

    # Verify batch dimension before squeezing
    if relevance_time_tensor.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got shape {relevance_time_tensor.shape}")

    relevance_time = relevance_time_tensor.squeeze(0)
    print(f"relevance_time shape after squeeze: {relevance_time.shape}")

    # Extract input signal and verify shapes
    input_signal = sample.squeeze(0).detach().cpu().numpy()
    print(f"input_signal shape: {input_signal.shape}")

    # Check for shape consistency between relevance and input signal
    if relevance_time.shape != input_signal.shape:
        raise ValueError(f"Shape mismatch: relevance_time {relevance_time.shape}, input_signal {input_signal.shape}")

    # Create FFTLRP instance
    try:
        fftlrp = FFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=32,
            cuda=(device == "cuda"),
            window_shift=window_shift,
            window_width=window_width,
            window_shape=window_shape,
            create_inverse=True,
            create_transpose_inverse=True,
            create_forward=True,
            create_stdft=compute_timefreq
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing FFTLRP: {str(e)}")

    # Prepare arrays for results
    n_axes = input_signal.shape[0]
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length

    # Initialize arrays with proper shapes
    signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.empty((n_axes, freq_length), dtype=np.complex128)

    # Process each axis separately
    for axis in range(n_axes):
        signal_axis = input_signal[axis:axis+1, :]
        relevance_axis = relevance_time[axis:axis+1, :]

        try:
            signal_hat, relevance_hat = fftlrp.fft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                short_time=False,
                epsilon=1e-6,
                real=False
            )

            # Verify output shapes before assignment
            if signal_hat.shape != (1, freq_length):
                raise ValueError(f"Expected signal_hat shape (1, {freq_length}), got {signal_hat.shape}")

            signal_freq[axis] = signal_hat.squeeze(0)
            relevance_freq[axis] = relevance_hat.squeeze(0)
        except Exception as e:
            print(f"Error in FFT-LRP computation for axis {axis}: {str(e)}")
            # Fill with zeros if processing fails
            signal_freq[axis] = np.zeros(freq_length, dtype=np.complex128)
            relevance_freq[axis] = np.zeros(freq_length, dtype=np.complex128)

    # Compute frequency bins for visualization
    freqs = np.fft.rfftfreq(signal_length, d=1.0/sampling_rate)

    # Initialize time-frequency arrays
    relevance_timefreq = None
    signal_timefreq = None

    # Compute time-frequency representations if requested
    if compute_timefreq:
        # Calculate the actual number of frames based on parameters
        n_frames = (signal_length - window_width) // window_shift + 1

        # Cap at 20 frames if needed for memory efficiency
        n_frames = min(20, n_frames)

        # Create arrays with proper shapes
        signal_timefreq = np.empty((n_axes, freq_length, n_frames), dtype=np.complex128)
        relevance_timefreq = np.empty((n_axes, freq_length, n_frames), dtype=np.complex128)

        for axis in range(n_axes):
            signal_axis = input_signal[axis]
            relevance_axis = relevance_time[axis]

            try:
                signal_hat, relevance_hat = fftlrp.fft_lrp(
                    relevance=relevance_axis[np.newaxis, :],
                    signal=signal_axis[np.newaxis, :],
                    short_time=True,
                    epsilon=1e-6,
                    real=False
                )

                # Check actual output shape
                actual_frames = signal_hat.shape[2] if len(signal_hat.shape) > 2 else 1
                frames_to_use = min(n_frames, actual_frames)

                # Handle potential shape mismatch
                if len(signal_hat.shape) < 3 or signal_hat.shape[0] != 1 or signal_hat.shape[1] != freq_length:
                    print(f"Warning: Unexpected shape for signal_hat: {signal_hat.shape}")
                    # Reshape or pad if necessary to match expected dimensions
                    if len(signal_hat.shape) == 2 and signal_hat.shape[0] == 1:
                        # Only one frame, need to reshape
                        signal_timefreq[axis, :, 0] = signal_hat.squeeze(0)[:freq_length]
                        relevance_timefreq[axis, :, 0] = relevance_hat.squeeze(0)[:freq_length]
                    else:
                        # Unable to handle this shape, fill with zeros
                        signal_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)
                        relevance_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)
                        print(f"Warning: Cannot handle signal_hat shape {signal_hat.shape}, filling with zeros")
                else:
                    # Normal case - copy the available frames
                    signal_timefreq[axis, :, :frames_to_use] = signal_hat.squeeze(0)[:, :frames_to_use]
                    relevance_timefreq[axis, :, :frames_to_use] = relevance_hat.squeeze(0)[:, :frames_to_use]
            except Exception as e:
                print(f"Error in time-frequency computation for axis {axis}: {str(e)}")
                # Fill with zeros if processing fails
                signal_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)
                relevance_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)

        print(f"signal_timefreq shape: {signal_timefreq.shape}, relevance_timefreq shape: {relevance_timefreq.shape}")

    # Clean up to free memory
    try:
        del fftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target if isinstance(target, int) else target.item())
