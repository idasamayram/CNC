import numpy as np
import torch
from utils.lrp_utils import zennit_relevance, zennit_relevance_lrp # Import from DFT-LRP utils
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
    # Ensure sample is a PyTorch tensor with shape (1, 3, 2000)
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
        relevance = zennit_relevance_lrp(
            input=sample,
            model=model,
            target=target,  # Target is already on the correct device
            RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
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
                relevance = zennit_relevance_lrp(
                    input=sample,
                    model=model,
                    target=target,
                    RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    # Remove batch dimension and convert to numpy
    relevance = relevance.squeeze(0)  # Shape: (3, 2000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 2000)

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
        relevance_time = zennit_relevance_lrp(
            input=sample,
            model=model,
            target=target,
            RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
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
                relevance_time = zennit_relevance_lrp(
                    input=sample,
                    model=model,
                    target=target,
                    RuleComposite="CustomFirstLayerMap",  # Use custom layer map for cnn1d network
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


def compute_dft_lrp_relevance_once(
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
    """
    Compute LRP relevances for a single vibration sample in both time and frequency domains,
    processing all axes at once with fallback to per-axis processing.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, 2000) for the vibration data
        label: Optional integer label (0 or 1). If None, use model prediction
        device: Torch device (CPU or CUDA)
        signal_length: Length of the signal
        leverage_symmetry: Use symmetry in DFT (reduces frequency bins to positive frequencies)
        precision: 32 or 16 for DFTLRP
        create_stdft: Whether to create STDFT layers
        create_inverse: Whether to create inverse DFT layers
        sampling_rate: Sampling rate of the data in Hz

    Returns:
        relevance_time: Numpy array of shape (3, signal_length) with time-domain relevances
        relevance_freq: Numpy array of shape (3, freq_bins) with frequency-domain relevances
        signal_freq: Numpy array of shape (3, freq_bins) with frequency-domain signal
        input_signal: Numpy array of shape (3, signal_length) with the input signal
        freqs: Frequency bins (for visualization)
        predicted_label: Predicted label if label is None
    """
    # Ensure sample is a PyTorch tensor with shape (1, 3, 2000)
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        sample = sample.to(device).unsqueeze(0)

    # Make sure model and sample are on the same device
    model = model.to(device)
    model.eval()

    if next(model.parameters()).device != sample.device:
        print(
            f"Warning: Model device ({next(model.parameters()).device}) doesn't match sample device ({sample.device})")
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
        relevance_time = zennit_relevance_lrp(
            input=sample,
            model=model,
            target=target,
            RuleComposite="CustomFirstLayerMap",  # Using the custom layer map
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
                relevance_time = zennit_relevance_lrp(
                    input=sample,
                    model=model,
                    target=target,
                    RuleComposite="CustomFirstLayerMap",
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    # Convert to numpy arrays and remove batch dimension
    if isinstance(relevance_time, torch.Tensor):
        relevance_time = relevance_time.squeeze(0).cpu().numpy()
    else:
        relevance_time = relevance_time.squeeze(0)

    input_signal = sample.squeeze(0).detach().cpu().numpy()

    print(f"Input sample shape: {sample.shape}")
    print(f"Relevance time shape: {relevance_time.shape}")
    print(f"Input signal shape: {input_signal.shape}")

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
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length

    # Try to process all axes at once
    try:
        print("Attempting to process all axes at once with DFT-LRP...")

        # Reshape to include batch dimension while keeping axes separate
        batch_signal = np.reshape(input_signal, (1, n_axes, signal_length))
        batch_relevance = np.reshape(relevance_time, (1, n_axes, signal_length))

        # Process all axes simultaneously through the modified DFT-LRP function
        # Note: This assumes the DFTLRP class has been modified to handle multi-axis inputs
        try:
            batch_signal_freq, batch_relevance_freq = dftlrp.dft_lrp_multi_axis_with_per_axis_norm(
                relevance=batch_relevance,
                signal=batch_signal,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            # Extract results
            signal_freq = batch_signal_freq.squeeze(0)  # Shape: (3, freq_length)
            relevance_freq = batch_relevance_freq.squeeze(0)  # Shape: (3, freq_length)

            print(
                f"Successfully processed all axes together. signal_freq shape: {signal_freq.shape}, relevance_freq shape: {relevance_freq.shape}")

        except AttributeError:
            # If dft_lrp_multi_axis doesn't exist, fall back to per-axis processing
            raise NotImplementedError("Multi-axis DFT-LRP not implemented in DFTLRP class")

    except (NotImplementedError, Exception) as e:
        print(f"Could not process all axes at once: {e}")
        print("Falling back to per-axis processing...")

        # Initialize arrays for per-axis processing
        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately
        for axis in range(n_axes):
            try:
                print(f"Processing axis {axis}...")
                signal_axis = input_signal[axis:axis + 1, :]
                relevance_axis = relevance_time[axis:axis + 1, :]

                signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=False,
                    epsilon=1e-6
                )

                signal_freq[axis] = signal_freq_axis[0]
                relevance_freq[axis] = relevance_freq_axis[0]
                print(f"Completed axis {axis} processing")

            except Exception as axis_e:
                print(f"Error processing axis {axis}: {axis_e}")
                # Fill with zeros if processing fails
                signal_freq[axis] = np.zeros(freq_length, dtype=np.complex128)
                relevance_freq[axis] = np.zeros(freq_length)

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]  # Scaled by sampling rate

    # Clean up to free memory
    try:
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target.item() if isinstance(target,
                                                                                                         torch.Tensor) else target
def compute_dft_lrp_relevance_with_timefreq(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        create_stdft=True,
        create_inverse=True,
        sampling_rate=400,
        window_shift=1,
        window_width=128,
        window_shape="rectangle"
):
    """
    Compute LRP relevances for a single vibration sample in both time and frequency domains,
    including time-frequency representations.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, 2000) for the vibration data
        label: Optional integer label (0 or 1). If None, use model prediction
        device: Torch device (CPU or CUDA)
        signal_length: Length of the signal (default 2000 for downsampled data)
        leverage_symmetry: Use symmetry in DFT (reduces frequency bins to positive frequencies)
        precision: 32 or 16 for DFTLRP
        create_stdft: Whether to create STDFT layers for time-frequency analysis
        create_inverse: Whether to create inverse DFT layers
        sampling_rate: Sampling rate of the data in Hz
        window_shift: Shift between adjacent windows for STDFT
        window_width: Width of the window for STDFT
        window_shape: Shape of the window function (rectangle, hann, etc.)

    Returns:
        relevance_time: Numpy array with time-domain relevances
        relevance_freq: Numpy array with frequency-domain relevances
        signal_freq: Numpy array with frequency-domain signal
        relevance_timefreq: Numpy array with time-frequency relevances
        signal_timefreq: Numpy array with time-frequency signal
        input_signal: Numpy array with the input signal
        freqs: Frequency bins (for visualization)
        predicted_label: Predicted label if label is None
    """
    # Memory management - clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Convert to CPU numpy first for preprocessing if sample is on GPU
    is_tensor = isinstance(sample, torch.Tensor)
    if is_tensor and sample.device.type == 'cuda':
        sample_np = sample.detach().cpu().numpy()
    elif is_tensor:
        sample_np = sample.numpy()
    else:
        sample_np = sample

    # Handle input shape - ensure we have shape (3, signal_length)
    if sample_np.ndim == 3:  # If batch dimension is present
        sample_np = sample_np.squeeze(0)  # Remove batch dimension

    # Create tensor with proper shape (1, 3, signal_length) on specified device
    if is_tensor:
        sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Move model to the specified device and evaluate mode
    model = model.to(device)
    model.eval()

    # Get target label (model prediction or provided label)
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample_tensor)
                _, predicted_label = torch.max(outputs, 1)
                target = predicted_label.item()
                target_tensor = predicted_label.to(device)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("CUDA out of memory during prediction. Falling back to CPU.")
                    device = "cpu"
                    model = model.cpu()
                    sample_tensor = sample_tensor.cpu()
                    outputs = model(sample_tensor)
                    _, predicted_label = torch.max(outputs, 1)
                    target = predicted_label.item()
                    target_tensor = predicted_label
                else:
                    raise RuntimeError(f"Error during model prediction: {e}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target_tensor = torch.tensor([target], device=device)

    # Compute LRP relevances in the time domain
    try:
        relevance_time_tensor = zennit_relevance_lrp(
            input=sample_tensor,
            model=model,
            target=target_tensor,
            RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
            rel_is_model_out=True,
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Falling back to CPU for LRP computation.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()
            if isinstance(target_tensor, torch.Tensor):
                target_tensor = target_tensor.cpu()

            # Try again on CPU
            relevance_time_tensor = zennit_relevance_lrp(
                input=sample_tensor,
                model=model,
                target=target_tensor,
                RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
                rel_is_model_out=True,
                cuda=False
            )
        else:
            raise

    # Move results to CPU and convert to numpy to free GPU memory
    relevance_time = relevance_time_tensor.squeeze(0).detach().cpu().numpy() if isinstance(relevance_time_tensor,
                                                                                           torch.Tensor) else relevance_time_tensor.squeeze(
        0)
    input_signal = sample_tensor.squeeze(0).detach().cpu().numpy() if isinstance(sample_tensor,
                                                                                 torch.Tensor) else sample_tensor.squeeze(
        0)

    # Clear GPU memory after getting time-domain results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute frequency domain results without using GPU if possible
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Initialize frequency domain results
    signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.empty((n_axes, freq_length))

    # Initialize time-frequency variables
    relevance_timefreq = None
    signal_timefreq = None

    # Use a separate function to handle DFT-LRP computation to better manage memory
    def process_single_axis(axis, signal, relevance, use_stdft=False):
        # Create DFTLRP object with reduced memory usage
        try:
            # Only create on CPU if we've fallen back or if GPU is low on memory
            use_cuda = (
                                   device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
                0).total_memory * 0.7

            dftlrp = DFTLRP(
                signal_length=signal_length,
                leverage_symmetry=leverage_symmetry,
                precision=precision,
                cuda=use_cuda,
                create_stdft=use_stdft and create_stdft,
                create_inverse=create_inverse,
                window_shift=window_shift,
                window_width=window_width,
                window_shape=window_shape
            )

            # Process frequency domain
            signal_axis = signal[axis:axis + 1, :]
            relevance_axis = relevance[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            # Process time-frequency domain if needed
            signal_tf_axis = None
            relevance_tf_axis = None

            if use_stdft:
                signal_tf_axis, relevance_tf_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=True,
                    epsilon=1e-6
                )

            # Clean up immediately
            del dftlrp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return signal_freq_axis, relevance_freq_axis, signal_tf_axis, relevance_tf_axis

        except Exception as e:
            print(f"Error processing axis {axis}: {e}")
            return None, None, None, None

    # Process frequency domain for all axes
    for axis in range(n_axes):
        signal_freq_axis, relevance_freq_axis, _, _ = process_single_axis(axis, input_signal, relevance_time, False)

        if signal_freq_axis is not None:
            signal_freq[axis] = signal_freq_axis[0]
            relevance_freq[axis] = relevance_freq_axis[0]
        else:
            # Fill with zeros if processing fails
            signal_freq[axis] = np.zeros(freq_length, dtype=np.complex128)
            relevance_freq[axis] = np.zeros(freq_length)

    # Process time-frequency domain if requested
    if create_stdft:
        # Calculate number of time frames with a cap for memory efficiency
        n_frames = min(20, (signal_length - window_width) // window_shift + 1)

        # Initialize time-frequency arrays
        signal_timefreq = np.empty((n_axes, freq_length, n_frames), dtype=np.complex128)
        relevance_timefreq = np.empty((n_axes, freq_length, n_frames), dtype=np.complex128)

        for axis in range(n_axes):
            # Start with empty arrays
            signal_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)
            relevance_timefreq[axis, :, :] = np.zeros((freq_length, n_frames), dtype=np.complex128)

            # Process time-frequency for this axis
            _, _, signal_tf_axis, relevance_tf_axis = process_single_axis(axis, input_signal, relevance_time, True)

            # Handle results (if available)
            if signal_tf_axis is not None and relevance_tf_axis is not None:
                # Check and handle shape
                if len(signal_tf_axis.shape) == 3:  # Expected (1, freq, frames)
                    actual_frames = signal_tf_axis.shape[2]
                    frames_to_use = min(n_frames, actual_frames)
                    signal_timefreq[axis, :, :frames_to_use] = signal_tf_axis.squeeze(0)[:, :frames_to_use]
                    relevance_timefreq[axis, :, :frames_to_use] = relevance_tf_axis.squeeze(0)[:, :frames_to_use]
                elif len(signal_tf_axis.shape) == 2 and signal_tf_axis.shape[0] == 1:
                    # Just one frame
                    signal_timefreq[axis, :, 0] = signal_tf_axis.squeeze(0)[:freq_length]
                    relevance_timefreq[axis, :, 0] = relevance_tf_axis.squeeze(0)[:freq_length]

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target.item() if isinstance(target, torch.Tensor) else target)

def compute_dft_lrp_relevance_2(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        sampling_rate=400
):
    """
    Compute LRP relevances for a single vibration sample in both time and frequency domains
    using a refined approach to get more meaningful frequency relevance values.
    """
    # Ensure sample is a PyTorch tensor with shape (1, 3, 2000)
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        sample = sample.to(device).unsqueeze(0)

    # Make sure model and sample are on the same device
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
    relevance_time = zennit_relevance_lrp(
        input=sample,
        model=model,
        target=target,
        RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
        rel_is_model_out=True,
        cuda=(device == "cuda")
    )

    relevance_time = relevance_time.squeeze(0)  # Shape: (3, 2000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 2000)

    # Convert relevance_time to numpy if it's a tensor
    if not isinstance(relevance_time, np.ndarray):
        relevance_time_np = relevance_time.detach().cpu().numpy()
    else:
        relevance_time_np = relevance_time

    # Initialize arrays for frequency domain results
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length

    signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
    relevance_freq = np.zeros((n_axes, freq_length), dtype=np.float32)

    # Improved approach for calculating frequency domain relevance
    for axis in range(n_axes):
        # FFT of the signal
        signal_freq[axis] = np.fft.rfft(input_signal[axis]) if leverage_symmetry else np.fft.fft(input_signal[axis])

        # Get signal magnitude in frequency domain
        signal_magnitude = np.abs(signal_freq[axis])

        # Method: Use a combination of relevance distribution and signal energy
        # 1. Compute total relevance in time domain
        total_relevance = np.sum(np.abs(relevance_time_np[axis]))

        # 2. Compute normalized signal energy distribution
        signal_energy = signal_magnitude ** 2
        total_energy = np.sum(signal_energy)

        if total_energy > 0:  # Avoid division by zero
            # 3. Distribute relevance based on normalized signal energy
            energy_ratio = signal_energy / total_energy
            relevance_freq[axis] = energy_ratio * total_relevance

            # 4. Apply additional frequency-selective weighting (optional)
            # This can emphasize certain frequency ranges that are important for the specific application
            # For example, if you know certain frequency bands are more relevant for your task:

            # Define frequency bands of interest (example: 0-10Hz, 10-50Hz, 50-200Hz)
            if leverage_symmetry:
                freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate)
            else:
                freqs = np.fft.fftfreq(signal_length, d=1.0 / sampling_rate)

            # Example: Apply a weight multiplier to emphasize mid-frequency components
            # Adjust these bands based on your specific application domain knowledge
            low_band = (freqs <= 10)  # Low frequency band (0-10 Hz)
            mid_band = (freqs > 10) & (freqs <= 50)  # Mid frequency band (10-50 Hz)
            high_band = (freqs > 50)  # High frequency band (>50 Hz)

            # Apply different weights to different bands (example weights)
            band_weights = np.ones_like(freqs, dtype=float)
            band_weights[low_band] *= 0.8  # Slightly de-emphasize very low frequencies
            band_weights[mid_band] *= 1.5  # Emphasize mid-frequencies
            band_weights[high_band] *= 1.0  # Normal weight for high frequencies

            # Apply band weighting to relevance
            relevance_freq[axis] = relevance_freq[axis] * band_weights
        else:
            # If signal energy is zero, distribute relevance evenly
            relevance_freq[axis] = np.ones(freq_length) * (total_relevance / freq_length)

        print(f"Axis {axis} frequency relevance stats:")
        print(f"  Min: {relevance_freq[axis].min()}")
        print(f"  Max: {relevance_freq[axis].max()}")
        print(f"  Mean: {relevance_freq[axis].mean()}")
        print(f"  Sum: {np.sum(relevance_freq[axis])}")
        print(f"  Non-zero values: {np.count_nonzero(relevance_freq[axis])}/{len(relevance_freq[axis])}")

    # Compute frequency bins for visualization
    if leverage_symmetry:
        freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate)
    else:
        freqs = np.fft.fftfreq(signal_length, d=1.0 / sampling_rate)

    # Convert relevance_time to numpy if it's a tensor
    if not isinstance(relevance_time, np.ndarray):
        relevance_time = relevance_time.detach().cpu().numpy()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target.item() if isinstance(target,
                                                                                                         torch.Tensor) else target


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
        relevance_time_tensor = zennit_relevance_lrp(
            input=sample,
            model=model,
            target=target,
            RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
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
                relevance_time_tensor = zennit_relevance_lrp(
                    input=sample,
                    model=model,
                    target=target,
                    RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
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
