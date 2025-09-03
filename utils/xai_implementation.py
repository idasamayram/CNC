import numpy as np
import torch
from utils.lrp_utils import  zennit_relevance_lrp
from numpy.fft import fftfreq
from utils.dft_lrp import EnhancedDFTLRP
import gc
from visualization.relevance_visualization import  visualize_hybrid_timefreq, visualize_xai_dft


# designated function to compute DFT of XAI methods

def compute_basic_dft_lrp(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
):
    """
    Compute LRP relevances for a single vibration sample in time and frequency domains only.
    This is a simplified version that doesn't compute time-frequency representations.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, signal_length) for the vibration data
        label: Optional integer label (0 or 1). If None, use model prediction
        device: Torch device (CPU or CUDA)
        signal_length: Length of the signal (default 2000 for downsampled data)
        leverage_symmetry: Use symmetry in DFT (reduces frequency bins to positive frequencies)
        precision: 32 or 16 for DFTLRP
        sampling_rate: Sampling rate of the data in Hz

    Returns:
        tuple containing:
        - relevance_time: Numpy array with time-domain relevances
        - relevance_freq: Numpy array with frequency-domain relevances
        - signal_freq: Numpy array with frequency-domain signal
        - input_signal: Numpy array with the input signal
        - freqs: Frequency bins (for visualization)
        - target: The target label used for LRP
    """
    import torch

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
            RuleComposite="CustomLayerMap",  # Use custom layer map for CNN1D model
            rel_is_model_out=True,
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Falling back to CPU for LRP computation.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()
            target_tensor = target_tensor.cpu() if isinstance(target_tensor, torch.Tensor) else target_tensor

            relevance_time_tensor = zennit_relevance_lrp(
                input=sample_tensor,
                model=model,
                target=target_tensor,
                RuleComposite="CustomLayerMap",
                rel_is_model_out=True,
                cuda=False
            )
        else:
            raise

    # Move to CPU and convert to numpy
    if isinstance(relevance_time_tensor, torch.Tensor):
        relevance_time = relevance_time_tensor.squeeze(0).cpu().numpy()
    else:
        relevance_time = relevance_time_tensor.squeeze(0)

    input_signal = sample_tensor.squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del relevance_time_tensor
    del sample_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Compute frequency domain relevance
    print("Computing frequency domain relevance...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (
                       device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
        0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT yet
            create_inverse=False,  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Clean up to free memory
    try:
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


def compute_dft_vanilla_gradient(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
):
    """
    Compute vanilla gradient relevances in time and frequency domains.
    This applies DFT to vanilla gradient method.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz

    Returns:
        relevance_time: Time domain gradient relevance
        relevance_freq: Frequency domain gradient relevance
        signal_freq: Frequency domain representation of the signal
        input_signal: Original input signal
        freqs: Frequency bins
        target: Target class used for explanation
    """
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Prepare input tensor
    if isinstance(sample, np.ndarray):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sample_tensor = sample.clone().to(device).unsqueeze(0)

    # Set requires_grad for gradient computation
    sample_tensor.requires_grad_(True)

    # Prepare model
    model = model.to(device)
    model.eval()

    # Get prediction or use provided label
    if label is None:
        with torch.no_grad():
            output = model(sample_tensor)
            _, prediction = torch.max(output, 1)
            target = prediction.item()
    else:
        target = label if isinstance(label, int) else label.item()

    # Compute gradients
    output = model(sample_tensor)

    # Zero gradients
    if hasattr(model, 'zero_grad'):
        model.zero_grad()
    else:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    # Compute gradient w.r.t target class
    target_output = output[0, target]
    target_output.backward(retain_graph=True)

    # Get gradient from input tensor (vanilla gradient)
    gradient = sample_tensor.grad.clone().detach()
    relevance_time = gradient.squeeze(0).cpu().numpy()

    # Get input signal
    input_signal = sample_tensor.squeeze(0).detach().cpu().numpy()

    # Create DFT-LRP object to leverage its DFT capabilities
    dft_handler = EnhancedDFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=False,  # No need for STDFT
        create_inverse=False  # No need for inverse transformation
    )

    # Use the DFT-LRP object to compute frequency domain representations
    # We call dft_lrp but since we're providing our own gradient-based relevance,
    # it's effectively just applying DFT to the gradient relevance
    try:
        signal_freq, relevance_freq = dft_handler.dft_lrp(
            relevance=relevance_time,
            signal=input_signal,
            short_time=False,
            epsilon=1e-6,
            real=False
        )
    except Exception as e:
        print(f"Error computing frequency domain representations: {e}")
        # Fallback: use numpy's FFT directly
        n_axes = input_signal.shape[0]  # 3 for X, Y, Z
        freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        for axis in range(n_axes):
            # FFT of the signal
            signal_freq[axis] = np.fft.rfft(input_signal[axis]) if leverage_symmetry else np.fft.fft(input_signal[axis])

            # FFT of the relevance
            rel_freq = np.fft.rfft(relevance_time[axis]) if leverage_symmetry else np.fft.fft(relevance_time[axis])

            # Take magnitude for relevance frequency
            relevance_freq[axis] = np.abs(rel_freq)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate) if leverage_symmetry else np.fft.fftfreq(
        signal_length, d=1.0 / sampling_rate)

    # Clean up
    del sample_tensor, gradient, dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


def compute_dft_gradient_input(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
):
    """
    Compute Gradient×Input relevances in time and frequency domains.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz

    Returns:
        relevance_time: Time domain Gradient×Input relevance
        relevance_freq: Frequency domain Gradient×Input relevance
        signal_freq: Frequency domain representation of the signal
        input_signal: Original input signal
        freqs: Frequency bins
        target: Target class used for explanation
    """
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Prepare input tensor
    if isinstance(sample, np.ndarray):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sample_tensor = sample.clone().to(device).unsqueeze(0)

    # Set requires_grad for gradient computation
    sample_tensor.requires_grad_(True)

    # Prepare model
    model = model.to(device)
    model.eval()

    # Get prediction or use provided label
    if label is None:
        with torch.no_grad():
            output = model(sample_tensor)
            _, prediction = torch.max(output, 1)
            target = prediction.item()
    else:
        target = label if isinstance(label, int) else label.item()

    # Compute gradients
    output = model(sample_tensor)

    # Zero gradients
    if hasattr(model, 'zero_grad'):
        model.zero_grad()
    else:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

    # Compute gradient w.r.t target class
    target_output = output[0, target]
    target_output.backward(retain_graph=True)

    # Get gradient from input tensor
    gradient = sample_tensor.grad.clone().detach()

    # Compute Gradient×Input relevance
    grad_input = gradient * sample_tensor
    relevance_time = grad_input.detach().squeeze(0).cpu().numpy()

    # Get input signal
    input_signal = sample_tensor.squeeze(0).detach().cpu().numpy()

    # Create DFT-LRP object to leverage its DFT capabilities
    dft_handler = EnhancedDFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=False
    )

    # Use the DFT-LRP object to compute frequency domain representations
    try:
        signal_freq, relevance_freq = dft_handler.dft_lrp(
            relevance=relevance_time,
            signal=input_signal,
            short_time=False,
            epsilon=1e-6,
            real=False
        )
    except Exception as e:
        print(f"Error computing frequency domain representations: {e}")
        # Fallback: use numpy's FFT directly
        n_axes = input_signal.shape[0]  # 3 for X, Y, Z
        freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        for axis in range(n_axes):
            # FFT of the signal
            signal_freq[axis] = np.fft.rfft(input_signal[axis]) if leverage_symmetry else np.fft.fft(input_signal[axis])

            # FFT of the relevance
            rel_freq = np.fft.rfft(relevance_time[axis]) if leverage_symmetry else np.fft.fft(relevance_time[axis])

            # Take magnitude for relevance frequency
            relevance_freq[axis] = np.abs(rel_freq)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate) if leverage_symmetry else np.fft.fftfreq(
        signal_length, d=1.0 / sampling_rate)

    # Clean up
    del sample_tensor, gradient, grad_input, dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


def compute_dft_smoothgrad(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        num_samples=40,
        noise_level=1.0
):
    """
    Compute SmoothGrad relevances in time and frequency domains.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz
        num_samples: Number of noisy samples for SmoothGrad
        noise_level: Noise level for SmoothGrad

    Returns:
        relevance_time: Time domain SmoothGrad relevance
        relevance_freq: Frequency domain SmoothGrad relevance
        signal_freq: Frequency domain representation of the signal
        input_signal: Original input signal
        freqs: Frequency bins
        target: Target class used for explanation
    """
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Prepare input tensor
    if isinstance(sample, np.ndarray):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sample_tensor = sample.clone().to(device).unsqueeze(0)

    # Prepare model
    model = model.to(device)
    model.eval()

    # Get prediction or use provided label
    if label is None:
        with torch.no_grad():
            output = model(sample_tensor)
            _, prediction = torch.max(output, 1)
            target = prediction.item()
    else:
        target = label if isinstance(label, int) else label.item()

    # Compute standard deviation for noise addition
    noise_std = torch.std(sample_tensor)

    # Initialize accumulated gradients tensor
    accumulated_gradients = torch.zeros_like(sample_tensor)

    # Compute gradients for multiple noisy samples
    for i in range(num_samples):
        # Create noisy sample (use original for first iteration)
        if i == 0:
            noisy_sample = sample_tensor.clone()
        else:
            noise = torch.randn_like(sample_tensor) * noise_level * noise_std
            noisy_sample = sample_tensor.clone() + noise

        # Compute gradients for this noisy sample
        noisy_sample.requires_grad_(True)

        # Forward pass
        output = model(noisy_sample)

        # Zero gradients
        if hasattr(model, 'zero_grad'):
            model.zero_grad()
        else:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        # Backward pass
        target_output = output[0, target]
        target_output.backward(retain_graph=True)

        # Accumulate gradients
        accumulated_gradients += noisy_sample.grad.clone().detach()

        # Free memory
        del noisy_sample, output

    # Average the accumulated gradients
    smoothgrad = accumulated_gradients / num_samples

    # Compute SmoothGrad relevance (gradient × input)
    smoothgrad_input = smoothgrad * sample_tensor
    relevance_time = smoothgrad_input.squeeze(0).cpu().numpy()

    # Get input signal
    input_signal = sample_tensor.squeeze(0).detach().cpu().numpy()

    # Create DFT-LRP object to leverage its DFT capabilities
    dft_handler = EnhancedDFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=False
    )

    # Use the DFT-LRP object to compute frequency domain representations
    try:
        signal_freq, relevance_freq = dft_handler.dft_lrp(
            relevance=relevance_time,
            signal=input_signal,
            short_time=False,
            epsilon=1e-6,
            real=False
        )
    except Exception as e:
        print(f"Error computing frequency domain representations: {e}")
        # Fallback: use numpy's FFT directly
        n_axes = input_signal.shape[0]  # 3 for X, Y, Z
        freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        for axis in range(n_axes):
            # FFT of the signal
            signal_freq[axis] = np.fft.rfft(input_signal[axis]) if leverage_symmetry else np.fft.fft(input_signal[axis])

            # FFT of the relevance
            rel_freq = np.fft.rfft(relevance_time[axis]) if leverage_symmetry else np.fft.fft(relevance_time[axis])

            # Take magnitude for relevance frequency
            relevance_freq[axis] = np.abs(rel_freq)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate) if leverage_symmetry else np.fft.fftfreq(
        signal_length, d=1.0 / sampling_rate)

    # Clean up
    del sample_tensor, accumulated_gradients, smoothgrad, smoothgrad_input, dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


def compute_dft_occlusion(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        occlusion_type="zero",
        window_size=5  # Window size for occlusion (40 for downsampled data)
):
    """
    Compute occlusion-based relevances in time and frequency domains.
    This function applies DFT to occlusion-based XAI method.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz
        occlusion_type: Type of occlusion to use ("zero", "one", "mone", "flip")
        window_size: Size of the occlusion window

    Returns:
        tuple containing:
        - relevance_time: Time domain occlusion relevance
        - relevance_freq: Frequency domain occlusion relevance
        - signal_freq: Frequency domain signal
        - input_signal: Original input signal
        - freqs: Frequency bins
        - target: Target label used for explanation
    """
    import torch
    import numpy as np
    import gc
    from utils.dft_lrp import EnhancedDFTLRP
    from numpy.fft import fftfreq

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
    sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Move model to the specified device and evaluate mode
    model = model.to(device)
    model.eval()

    # Define occlusion functions
    def zero(val):
        return torch.zeros_like(val)

    def one(val):
        return torch.ones_like(val)

    def mone(val):
        return -torch.ones_like(val)

    def flip(val):
        return -val

    occlusion_fxns = {"zero": zero, "one": one, "mone": mone, "flip": flip}
    assert occlusion_type in occlusion_fxns, f"Invalid occlusion type: {occlusion_type}"

    # Get prediction or use provided label
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample_tensor)
                _, predicted_label = torch.max(outputs, 1)
                target = predicted_label.item()
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print("CUDA out of memory during prediction. Falling back to CPU.")
                    device = "cpu"
                    model = model.cpu()
                    sample_tensor = sample_tensor.cpu()
                    outputs = model(sample_tensor)
                    _, predicted_label = torch.max(outputs, 1)
                    target = predicted_label.item()
                else:
                    raise RuntimeError(f"Error during model prediction: {e}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label

    # Get the original prediction before occlusion
    with torch.no_grad():
        pred_0 = model(sample_tensor)

    # Prepare attribution mask (same shape as input)
    attributions = torch.zeros_like(sample_tensor)




    for i in range(0, sample_tensor.shape[2], window_size):  # Slide occlusion window

        # Apply occlusion for each feature/axis separately
        for feature_idx in range(sample_tensor.shape[1]):  # X, Y, Z axes
            x_copy = sample_tensor.clone()  # Copy original input

            # Apply occlusion only to the current feature/axis
            x_copy[:,feature_idx, i:i + window_size] = occlusion_fxns[occlusion_type](
                x_copy[:,feature_idx, i:i + window_size])


            # Get new prediction after occlusion
            # Get new prediction after occlusion
            with torch.no_grad():
                pred = model(x_copy)

            # Compute attribution for this axis only
            attributions[:,feature_idx, i:i + window_size] = pred_0[target] - pred[target]


    # Extract relevance map (remove batch dimension)
    relevance_time = attributions.squeeze(0).cpu().numpy()

    # Get input signal (remove batch dimension)
    input_signal = sample_tensor.squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del attributions, sample_tensor, pred_0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Compute frequency domain relevance
    print("Computing frequency domain relevance...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (device == "cuda") and torch.cuda.is_available() and \
               torch.cuda.memory_allocated() < torch.cuda.get_device_properties(0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT
            create_inverse=False  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


# General function to compute DFT-XAI relevances (can be used for all baselines and LRP)

def compute_dft_for_xai_method(
        xai_method_func,
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400,
        **xai_kwargs
):
    """
    General function to compute DFT for any XAI method.

    Args:
        xai_method_func: Function that computes relevance maps (e.g., grad_times_input_relevance)
        model: Trained CNN1D model
        sample: Input tensor
        label: Optional target label
        device: Computing device
        signal_length: Length of the signal
        leverage_symmetry: Whether to use symmetry in DFT
        precision: Computation precision
        sampling_rate: Signal sampling rate
        xai_kwargs: Additional arguments for the XAI method function

    Returns:
        relevance_time: Time domain relevance from the XAI method
        relevance_freq: Frequency domain relevance
        signal_freq: Frequency domain signal
        input_signal: Original input signal
        freqs: Frequency bins
        target: Target class
    """
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Prepare input tensor for the XAI method
    if isinstance(sample, np.ndarray):
        sample_tensor = torch.tensor(sample, dtype=torch.float32).to(device)
    else:
        sample_tensor = sample.clone().to(device)

    # Compute relevance using the provided XAI method function
    relevance_time, target = xai_method_func(model, sample_tensor, target=label, **xai_kwargs)

    # Convert to numpy for DFT processing
    if isinstance(relevance_time, torch.Tensor):
        relevance_time = relevance_time.cpu().numpy()

    # Get input signal
    if isinstance(sample, torch.Tensor):
        input_signal = sample.detach().cpu().numpy()
    else:
        input_signal = sample

    # Create DFT-LRP object to leverage its DFT capabilities
    dft_handler = EnhancedDFTLRP(
        signal_length=signal_length,
        leverage_symmetry=leverage_symmetry,
        precision=precision,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=False
    )

    # Use the DFT-LRP object to compute frequency domain representations
    try:
        signal_freq, relevance_freq = dft_handler.dft_lrp(
            relevance=relevance_time,
            signal=input_signal,
            short_time=False,
            epsilon=1e-6,
            real=False
        )
    except Exception as e:
        print(f"Error computing frequency domain representations: {e}")
        # Fallback: use numpy's FFT directly
        n_axes = input_signal.shape[0]  # Usually 3 for X, Y, Z
        freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        for axis in range(n_axes):
            # FFT of the signal
            signal_freq[axis] = np.fft.rfft(input_signal[axis]) if leverage_symmetry else np.fft.fft(input_signal[axis])

            # FFT of the relevance
            rel_freq = np.fft.rfft(relevance_time[axis]) if leverage_symmetry else np.fft.fft(relevance_time[axis])

            # Take magnitude for relevance frequency
            relevance_freq[axis] = np.abs(rel_freq)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(signal_length, d=1.0 / sampling_rate) if leverage_symmetry else np.fft.fftfreq(
        signal_length, d=1.0 / sampling_rate)

    # Clean up
    del dft_handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target


# designated function to compute STDFT of XAI methods

def compute_enhanced_dft_lrp(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=16,
        sampling_rate=400,
        window_shift=None,  # Will be set automatically based on window_width
        window_width=128,
        window_shape="rectangle"
):
    """
    Compute LRP relevances in time, frequency, and time-frequency domains
    with improved memory management, error handling, and spectrogram visualization.
    """
    import torch

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
            RuleComposite="CustomLayerMap",  # Use custom layer map for CNN1D model
            rel_is_model_out=True,
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Falling back to CPU for LRP computation.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()
            target_tensor = target_tensor.cpu() if isinstance(target_tensor, torch.Tensor) else target_tensor

            relevance_time_tensor = zennit_relevance_lrp(
                input=sample_tensor,
                model=model,
                target=target_tensor,
                RuleComposite="CustomLayerMap",
                rel_is_model_out=True,
                cuda=False
            )
        else:
            raise

    # Move to CPU and convert to numpy
    if isinstance(relevance_time_tensor, torch.Tensor):
        relevance_time = relevance_time_tensor.squeeze(0).cpu().numpy()
    else:
        relevance_time = relevance_time_tensor.squeeze(0)

    input_signal = sample_tensor.squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del relevance_time_tensor
    del sample_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Compute frequency domain relevance
    print("Computing frequency domain relevance...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (
                           device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
        0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT yet
            create_inverse=False,  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Step 2: Compute time-frequency domain relevance
    print("Computing time-frequency domain relevance...")

    # Set window parameters to better match spectrogram settings
    if window_width is None:
        window_width = 256  # Standard spectrogram window size

    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap is standard for spectrograms

    # Calculate number of time frames based on window parameters
    n_frames = max(1, (signal_length - window_width) // window_shift + 1)
    print(f"Creating time-frequency domain with {n_frames} frames")

    # Initialize time-frequency arrays
    signal_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)
    relevance_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)

    try:
        # Use CPU for computation to avoid CUDA errors
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=False,  # Use CPU for STDFT to avoid CUDA errors
            window_shift=window_shift,
            window_width=window_width,
            window_shape=window_shape,
            create_dft=False,  # No need for regular DFT
            create_inverse=False  # No need for inverse transformation
        )

        # Process each axis separately
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            try:
                signal_tf_axis, relevance_tf_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=True,
                    epsilon=1e-6
                )

                # Debug shapes
                print(f"Axis {axis} signal_tf_axis shape: {signal_tf_axis.shape}")
                print(f"Axis {axis} relevance_tf_axis shape: {relevance_tf_axis.shape}")

                # Handle different possible shapes
                if signal_tf_axis is not None and signal_tf_axis.ndim >= 2:
                    if signal_tf_axis.ndim == 3:  # Expected (batch, frames, freq)
                        # Get the actual shape
                        actual_freq = min(signal_tf_axis.shape[2], freq_length)
                        actual_frames = min(signal_tf_axis.shape[1], n_frames)

                        # Transpose to get (batch, freq, frames) which is correct for spectrogram visualization
                        signal_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            signal_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                        relevance_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            relevance_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                    elif signal_tf_axis.ndim == 2:
                        if signal_tf_axis.shape[0] == 1:  # Shape (batch, freq*frames)
                            # Reshape to get proper time-frequency shape
                            reshaped_signal = signal_tf_axis[0].reshape(n_frames, -1)
                            reshaped_relevance = relevance_tf_axis[0].reshape(n_frames, -1)

                            # Transpose to get (freq, frames)
                            actual_freq = min(reshaped_signal.shape[1], freq_length)
                            actual_frames = min(reshaped_signal.shape[0], n_frames)

                            signal_timefreq[axis, :actual_freq, :actual_frames] = reshaped_signal[:actual_frames,
                                                                                  :actual_freq].T
                            relevance_timefreq[axis, :actual_freq, :actual_frames] = reshaped_relevance[:actual_frames,
                                                                                     :actual_freq].T
            except Exception as e:
                print(f"Error computing time-frequency domain for axis {axis}: {e}")

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in time-frequency domain computation: {e}")

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target)


def compute_enhanced_dft_gradient(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=16,
        sampling_rate=400,
        window_shift=None,  # Will be set automatically based on window_width
        window_width=128,
        window_shape="rectangle"
):
    """
    Compute vanilla gradient relevances in time, frequency, and time-frequency domains
    with improved memory management, error handling, and spectrogram visualization.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz
        window_shift: Shift between adjacent windows for STDFT analysis
        window_width: Width of the window for STDFT analysis
        window_shape: Shape of the window function for STDFT

    Returns:
        tuple containing:
        - relevance_time: Time domain gradient relevance
        - relevance_freq: Frequency domain gradient relevance
        - signal_freq: Frequency domain representation of the signal
        - relevance_timefreq: Time-frequency domain gradient relevance
        - signal_timefreq: Time-frequency domain representation of the signal
        - input_signal: Original input signal
        - freqs: Frequency bins (for visualization)
        - target: Target class used for explanation
    """
    import torch
    from numpy.fft import fftfreq
    import numpy as np
    import gc

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
    sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Set requires_grad for gradient computation
    sample_tensor.requires_grad_(True)

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

    # Compute gradient relevances in the time domain
    try:
        # Forward pass
        outputs = model(sample_tensor)

        # Zero gradients
        model.zero_grad()

        # Backward pass for target class
        outputs[0, target].backward()

        # Get gradient from input tensor (vanilla gradient)
        gradient = sample_tensor.grad.clone()

        # Convert to numpy array
        relevance_time = gradient.squeeze(0).cpu().numpy()

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Falling back to CPU for gradient computation.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()
            sample_tensor.requires_grad_(True)

            # Retry on CPU
            outputs = model(sample_tensor)
            model.zero_grad()
            outputs[0, target].backward()
            gradient = sample_tensor.grad.clone()
            relevance_time = gradient.squeeze(0).cpu().numpy()
        else:
            raise

    # Get input signal
    input_signal = sample_tensor.detach().squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del gradient
    del sample_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Compute frequency domain relevance
    print("Computing frequency domain relevance...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (
                           device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
        0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT yet
            create_inverse=False,  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Step 2: Compute time-frequency domain relevance
    print("Computing time-frequency domain relevance...")

    # Set window parameters to better match spectrogram settings
    if window_width is None:
        window_width = 256  # Standard spectrogram window size

    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap is standard for spectrograms

    # Calculate number of time frames based on window parameters
    n_frames = max(1, (signal_length - window_width) // window_shift + 1)
    print(f"Creating time-frequency domain with {n_frames} frames")

    # Initialize time-frequency arrays
    signal_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)
    relevance_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)

    try:
        # Use CPU for computation to avoid CUDA errors
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=False,  # Use CPU for STDFT to avoid CUDA errors
            window_shift=window_shift,
            window_width=window_width,
            window_shape=window_shape,
            create_dft=False,  # No need for regular DFT
            create_inverse=False  # No need for inverse transformation
        )

        # Process each axis separately
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            try:
                signal_tf_axis, relevance_tf_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=True,
                    epsilon=1e-6
                )

                # Debug shapes
                print(f"Axis {axis} signal_tf_axis shape: {signal_tf_axis.shape}")
                print(f"Axis {axis} relevance_tf_axis shape: {relevance_tf_axis.shape}")

                # Handle different possible shapes
                if signal_tf_axis is not None and signal_tf_axis.ndim >= 2:
                    if signal_tf_axis.ndim == 3:  # Expected (batch, frames, freq)
                        # Get the actual shape
                        actual_freq = min(signal_tf_axis.shape[2], freq_length)
                        actual_frames = min(signal_tf_axis.shape[1], n_frames)

                        # Transpose to get (batch, freq, frames) which is correct for spectrogram visualization
                        signal_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            signal_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                        relevance_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            relevance_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                    elif signal_tf_axis.ndim == 2:
                        if signal_tf_axis.shape[0] == 1:  # Shape (batch, freq*frames)
                            # Reshape to get proper time-frequency shape
                            reshaped_signal = signal_tf_axis[0].reshape(n_frames, -1)
                            reshaped_relevance = relevance_tf_axis[0].reshape(n_frames, -1)

                            # Transpose to get (freq, frames)
                            actual_freq = min(reshaped_signal.shape[1], freq_length)
                            actual_frames = min(reshaped_signal.shape[0], n_frames)

                            signal_timefreq[axis, :actual_freq, :actual_frames] = reshaped_signal[:actual_frames,
                                                                                  :actual_freq].T
                            relevance_timefreq[axis, :actual_freq, :actual_frames] = reshaped_relevance[:actual_frames,
                                                                                     :actual_freq].T
            except Exception as e:
                print(f"Error computing time-frequency domain for axis {axis}: {e}")

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in time-frequency domain computation: {e}")

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target)


def compute_enhanced_dft_gradient_input(
        model,
        sample,
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=16,
        sampling_rate=400,
        window_shift=None,  # Will be set automatically based on window_width
        window_width=128,
        window_shape="rectangle"
):
    """
    Compute Gradient×Input relevances in time, frequency, and time-frequency domains
    with improved memory management, error handling, and spectrogram visualization.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz
        window_shift: Shift between adjacent windows for STDFT analysis
        window_width: Width of the window for STDFT analysis
        window_shape: Shape of the window function for STDFT

    Returns:
        tuple containing:
        - relevance_time: Time domain gradient×input relevance
        - relevance_freq: Frequency domain gradient×input relevance
        - signal_freq: Frequency domain representation of the signal
        - relevance_timefreq: Time-frequency domain gradient×input relevance
        - signal_timefreq: Time-frequency domain representation of the signal
        - input_signal: Original input signal
        - freqs: Frequency bins (for visualization)
        - target: Target class used for explanation
    """
    import torch
    from numpy.fft import fftfreq
    import numpy as np
    import gc

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
    sample_tensor = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(device)

    # Set requires_grad for gradient computation
    sample_tensor.requires_grad_(True)

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

    # Compute gradient×input relevances in the time domain
    try:
        # Forward pass
        outputs = model(sample_tensor)

        # Zero gradients
        model.zero_grad()

        # Backward pass for target class
        outputs[0, target].backward()

        # Get gradient from input tensor
        gradient = sample_tensor.grad.clone()

        # Compute gradient×input
        grad_input = gradient * sample_tensor

        # Convert to numpy array
        relevance_time = grad_input.squeeze(0).detach().cpu().numpy()

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory. Falling back to CPU for gradient computation.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()
            sample_tensor.requires_grad_(True)

            # Retry on CPU
            outputs = model(sample_tensor)
            model.zero_grad()
            outputs[0, target].backward()
            gradient = sample_tensor.grad.clone()
            grad_input = gradient * sample_tensor
            relevance_time = grad_input.squeeze(0).detach().cpu().numpy()
        else:
            raise

    # Get input signal
    input_signal = sample_tensor.detach().squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del gradient, grad_input
    del sample_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Compute frequency domain relevance
    print("Computing frequency domain relevance...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (
                           device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
        0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT yet
            create_inverse=False,  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Step 2: Compute time-frequency domain relevance
    print("Computing time-frequency domain relevance...")

    # Set window parameters to better match spectrogram settings
    if window_width is None:
        window_width = 256  # Standard spectrogram window size

    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap is standard for spectrograms

    # Calculate number of time frames based on window parameters
    n_frames = max(1, (signal_length - window_width) // window_shift + 1)
    print(f"Creating time-frequency domain with {n_frames} frames")

    # Initialize time-frequency arrays
    signal_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)
    relevance_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)

    try:
        # Use CPU for computation to avoid CUDA errors
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=False,  # Use CPU for STDFT to avoid CUDA errors
            window_shift=window_shift,
            window_width=window_width,
            window_shape=window_shape,
            create_dft=False,  # No need for regular DFT
            create_inverse=False  # No need for inverse transformation
        )

        # Process each axis separately
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            try:
                signal_tf_axis, relevance_tf_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=True,
                    epsilon=1e-6
                )

                # Debug shapes
                print(f"Axis {axis} signal_tf_axis shape: {signal_tf_axis.shape}")
                print(f"Axis {axis} relevance_tf_axis shape: {relevance_tf_axis.shape}")

                # Handle different possible shapes
                if signal_tf_axis is not None and signal_tf_axis.ndim >= 2:
                    if signal_tf_axis.ndim == 3:  # Expected (batch, frames, freq)
                        # Get the actual shape
                        actual_freq = min(signal_tf_axis.shape[2], freq_length)
                        actual_frames = min(signal_tf_axis.shape[1], n_frames)

                        # Transpose to get (batch, freq, frames) which is correct for spectrogram visualization
                        signal_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            signal_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                        relevance_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            relevance_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                    elif signal_tf_axis.ndim == 2:
                        if signal_tf_axis.shape[0] == 1:  # Shape (batch, freq*frames)
                            # Reshape to get proper time-frequency shape
                            reshaped_signal = signal_tf_axis[0].reshape(n_frames, -1)
                            reshaped_relevance = relevance_tf_axis[0].reshape(n_frames, -1)

                            # Transpose to get (freq, frames)
                            actual_freq = min(reshaped_signal.shape[1], freq_length)
                            actual_frames = min(reshaped_signal.shape[0], n_frames)

                            signal_timefreq[axis, :actual_freq, :actual_frames] = reshaped_signal[:actual_frames,
                                                                                  :actual_freq].T
                            relevance_timefreq[axis, :actual_freq, :actual_frames] = reshaped_relevance[:actual_frames,
                                                                                     :actual_freq].T
            except Exception as e:
                print(f"Error computing time-frequency domain for axis {axis}: {e}")

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in time-frequency domain computation: {e}")

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target)


# General function to compute STDFT-XAI relevances (can be used for all baselines and LRP)

def compute_enhanced_dft_for_xai_method(
        model,
        sample,
        xai_method="lrp",
        # Options: "lrp", "gradient", "gradient_input", "smoothgrad", "integrated_gradients", "occlusion"
        label=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        signal_length=2000,
        leverage_symmetry=True,
        precision=16,
        sampling_rate=400,
        window_shift=None,  # Will be set automatically based on window_width
        window_width=128,
        window_shape="rectangle",
        **xai_kwargs  # Additional arguments specific to the XAI method
):
    """
    Generalized function to compute any XAI method relevances in time, frequency, and time-frequency domains.

    Args:
        model: Trained CNN1D model
        sample: Input tensor of shape (3, signal_length)
        xai_method: The XAI method to use ("lrp", "gradient", "gradient_input", "smoothgrad", "integrated_gradients", "occlusion")
        label: Optional target label (if None, uses model prediction)
        device: Computing device (cuda/cpu)
        signal_length: Length of the input signal
        leverage_symmetry: Whether to use symmetry properties of DFT
        precision: Precision for computation (32 or 16)
        sampling_rate: Sampling rate of the signal in Hz
        window_shift: Shift between adjacent windows for STDFT analysis
        window_width: Width of the window for STDFT analysis
        window_shape: Shape of the window function for STDFT
        xai_kwargs: Additional arguments specific to the XAI method, such as:
            - num_samples: For SmoothGrad (default: 40)
            - noise_level: For SmoothGrad (default: 1.0)
            - steps: For Integrated Gradients (default: 50)
            - occlusion_type: For occlusion (default: "zero")
            - window_size: For occlusion (default: 40)

    Returns:
        tuple containing:
        - relevance_time: Time domain XAI relevance
        - relevance_freq: Frequency domain XAI relevance
        - signal_freq: Frequency domain representation of the signal
        - relevance_timefreq: Time-frequency domain XAI relevance
        - signal_timefreq: Time-frequency domain representation of the signal
        - input_signal: Original input signal
        - freqs: Frequency bins (for visualization)
        - target: Target class used for explanation
    """
    import torch
    from numpy.fft import fftfreq
    import numpy as np
    import gc

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

    # Compute relevances in the time domain based on the selected XAI method
    try:
        if xai_method.lower() == "lrp":
            # Layer-wise Relevance Propagation
            from utils.lrp_utils import zennit_relevance_lrp
            relevance_time_tensor = zennit_relevance_lrp(
                input=sample_tensor,
                model=model,
                target=target_tensor,
                RuleComposite="CustomLayerMap",
                rel_is_model_out=True,
                cuda=(device == "cuda")
            )
            if relevance_time_tensor.ndim== 3:
                relevance_time = relevance_time_tensor.squeeze(0)
            else:
                relevance_time = relevance_time_tensor

        elif xai_method.lower() == "gradient":
            # Vanilla Gradient
            sample_tensor.requires_grad_(True)
            outputs = model(sample_tensor)
            model.zero_grad()
            outputs[0, target].backward()
            gradient = sample_tensor.grad.clone()
            relevance_time = gradient.squeeze(0).cpu().numpy()

        elif xai_method.lower() == "gradient_input":
            # Gradient * Input
            sample_tensor.requires_grad_(True)
            outputs = model(sample_tensor)
            model.zero_grad()
            outputs[0, target].backward()
            gradient = sample_tensor.grad.clone()
            grad_input = gradient * sample_tensor
            relevance_time = grad_input.squeeze(0).detach().cpu().numpy()

        elif xai_method.lower() == "smoothgrad":
            # SmoothGrad
            num_samples = xai_kwargs.get("num_samples", 40)
            noise_level = xai_kwargs.get("noise_level", 1.0)

            # Compute standard deviation for noise addition
            noise_std = torch.std(sample_tensor)

            # Initialize accumulated gradients tensor
            accumulated_gradients = torch.zeros_like(sample_tensor)

            # Compute gradients for multiple noisy samples
            for i in range(num_samples):
                # Create noisy sample (use original for first iteration)
                if i == 0:
                    noisy_sample = sample_tensor.clone()
                else:
                    noise = torch.randn_like(sample_tensor) * noise_level * noise_std
                    noisy_sample = sample_tensor.clone() + noise

                # Compute gradients for this noisy sample
                noisy_sample.requires_grad_(True)

                # Forward pass
                output = model(noisy_sample)

                # Zero gradients
                model.zero_grad()

                # Backward pass
                target_output = output[0, target]
                target_output.backward(retain_graph=(i < num_samples - 1))

                # Accumulate gradients
                accumulated_gradients += noisy_sample.grad.clone().detach()

                # Free memory
                del noisy_sample, output

            # Average the accumulated gradients
            smoothgrad = accumulated_gradients / num_samples

            # Compute SmoothGrad relevance (gradient × input)
            smoothgrad_input = smoothgrad * sample_tensor
            relevance_time = smoothgrad_input.squeeze(0).cpu().numpy()

        elif xai_method.lower() == "integrated_gradients":
            # Integrated Gradients
            steps = xai_kwargs.get("steps", 50)

            # Initialize baseline (zeros)
            baseline = torch.zeros_like(sample_tensor)

            # Initialize accumulated gradients
            accumulated_gradients = torch.zeros_like(sample_tensor)

            # Compute gradients at different interpolation steps
            for step in range(steps):
                # Create interpolated sample
                alpha = step / (steps - 1)
                interpolated_sample = baseline + alpha * (sample_tensor - baseline)
                interpolated_sample.requires_grad_(True)

                # Forward pass
                output = model(interpolated_sample)

                # Zero gradients
                model.zero_grad()

                # Backward pass
                target_output = output[0, target]
                target_output.backward(retain_graph=(step < steps - 1))

                # Accumulate gradients
                accumulated_gradients += interpolated_sample.grad.clone().detach()

                # Free memory
                del interpolated_sample, output

            # Compute average gradient and multiply by (sample - baseline)
            avg_gradient = accumulated_gradients / steps
            ig_relevance = avg_gradient * (sample_tensor - baseline)
            relevance_time = ig_relevance.squeeze(0).cpu().numpy()

        elif xai_method.lower() == "occlusion":
            # Occlusion-based explanation
            occlusion_type = xai_kwargs.get("occlusion_type", "zero")
            window_size = xai_kwargs.get("window_size", 40)

            # Occlusion functions
            def zero_occlude(val):
                return torch.zeros_like(val)

            def one_occlude(val):
                return torch.ones_like(val)

            def minus_one_occlude(val):
                return -torch.ones_like(val)

            def flip_occlude(val):
                return -val

            occlusion_funcs = {
                "zero": zero_occlude,
                "one": one_occlude,
                "mone": minus_one_occlude,
                "flip": flip_occlude
            }

            if occlusion_type not in occlusion_funcs:
                raise ValueError(
                    f"Invalid occlusion type: {occlusion_type}. Choose from {list(occlusion_funcs.keys())}")

            occlude_func = occlusion_funcs[occlusion_type]

            # Get the original prediction
            with torch.no_grad():
                original_output = model(sample_tensor)
                original_score = original_output[0, target].item()

            # Initialize attribution mask
            relevance = torch.zeros_like(sample_tensor)

            # Iterate over time steps in window_size chunks
            for i in range(0, signal_length, window_size):
                occluded_sample = sample_tensor.clone()

                # Compute attribution: Original - Occluded
                # Apply the difference to each axis individually
                for feature_idx in range(sample_tensor.shape[1]):  # For each axis (X, Y, Z)
                    # Create a temporary occluded sample with occlusion ONLY in this feature/axis
                    single_axis_occluded = sample_tensor.clone()
                    end_idx = min(i + window_size, signal_length)
                    single_axis_occluded[0, feature_idx, i:end_idx] = occlude_func(
                        single_axis_occluded[0, feature_idx, i:end_idx])

                    # Get prediction for occlusion in just this axis
                    with torch.no_grad():
                        single_axis_output = model(single_axis_occluded)
                        single_axis_score = single_axis_output[0, target].item()

                    # Assign attribution to just this axis
                    relevance[0, feature_idx, i:end_idx] = original_score - single_axis_score

            relevance_time = relevance.squeeze(0).cpu().numpy()

        else:
            raise ValueError(
                f"Unknown XAI method: {xai_method}. Supported methods: 'lrp', 'gradient', 'gradient_input', 'smoothgrad', 'integrated_gradients', 'occlusion'")

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"CUDA out of memory during {xai_method} computation. Falling back to CPU.")
            device = "cpu"
            model = model.cpu()
            sample_tensor = sample_tensor.cpu()

            # Recursive call with CPU device
            return compute_enhanced_dft_for_xai_method(
                model=model,
                sample=sample_tensor.squeeze(0),
                xai_method=xai_method,
                label=target,
                device="cpu",
                signal_length=signal_length,
                leverage_symmetry=leverage_symmetry,
                precision=precision,
                sampling_rate=sampling_rate,
                window_shift=window_shift,
                window_width=window_width,
                window_shape=window_shape,
                **xai_kwargs
            )
        else:
            raise


    # Get input signal
    input_signal = sample_tensor.detach().squeeze(0).cpu().numpy()

    # Free memory after time-domain computation
    del sample_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Compute frequency domain relevance
    print(f"Computing frequency domain relevance for {xai_method}...")
    freq_length = signal_length // 2 + 1 if leverage_symmetry else signal_length
    n_axes = input_signal.shape[0]  # 3 (X, Y, Z)

    # Use CPU for computation to avoid CUDA errors
    use_cuda = (
                           device == "cuda") and torch.cuda.is_available() and torch.cuda.memory_allocated() < torch.cuda.get_device_properties(
        0).total_memory * 0.7

    try:
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=use_cuda,
            create_stdft=False,  # No need for STDFT yet
            create_inverse=False,  # No need for inverse transformation
        )

        signal_freq = np.empty((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.empty((n_axes, freq_length))

        # Process each axis separately for better memory management
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            signal_freq_axis, relevance_freq_axis = dftlrp.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                real=False,
                short_time=False,
                epsilon=1e-6
            )

            signal_freq[axis] = signal_freq_axis[0] if signal_freq_axis.ndim > 1 else signal_freq_axis
            relevance_freq[axis] = relevance_freq_axis[0] if relevance_freq_axis.ndim > 1 else relevance_freq_axis

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in frequency domain computation: {e}")
        # Create empty arrays if computation fails
        signal_freq = np.zeros((n_axes, freq_length), dtype=np.complex128)
        relevance_freq = np.zeros((n_axes, freq_length))

    # Step 2: Compute time-frequency domain relevance
    print(f"Computing time-frequency domain relevance for {xai_method}...")

    # Set window parameters to better match spectrogram settings
    if window_width is None:
        window_width = 256  # Standard spectrogram window size

    if window_shift is None:
        window_shift = window_width // 2  # 50% overlap is standard for spectrograms

    # Calculate number of time frames based on window parameters
    n_frames = max(1, (signal_length - window_width) // window_shift + 1)
    print(f"Creating time-frequency domain with {n_frames} frames")

    # Initialize time-frequency arrays
    signal_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)
    relevance_timefreq = np.zeros((n_axes, freq_length, n_frames), dtype=np.complex128)

    try:
        # Use CPU for computation to avoid CUDA errors
        dftlrp = EnhancedDFTLRP(
            signal_length=signal_length,
            leverage_symmetry=leverage_symmetry,
            precision=precision,
            cuda=False,  # Use CPU for STDFT to avoid CUDA errors
            window_shift=window_shift,
            window_width=window_width,
            window_shape=window_shape,
            create_dft=False,  # No need for regular DFT
            create_inverse=False  # No need for inverse transformation
        )

        # Process each axis separately
        for axis in range(n_axes):
            signal_axis = input_signal[axis:axis + 1, :]
            relevance_axis = relevance_time[axis:axis + 1, :]

            try:
                signal_tf_axis, relevance_tf_axis = dftlrp.dft_lrp(
                    relevance=relevance_axis,
                    signal=signal_axis,
                    real=False,
                    short_time=True,
                    epsilon=1e-6
                )

                # Debug shapes
                print(f"Axis {axis} signal_tf_axis shape: {signal_tf_axis.shape}")
                print(f"Axis {axis} relevance_tf_axis shape: {relevance_tf_axis.shape}")

                # Handle different possible shapes
                if signal_tf_axis is not None and signal_tf_axis.ndim >= 2:
                    if signal_tf_axis.ndim == 3:  # Expected (batch, frames, freq)
                        # Get the actual shape
                        actual_freq = min(signal_tf_axis.shape[2], freq_length)
                        actual_frames = min(signal_tf_axis.shape[1], n_frames)

                        # Transpose to get (batch, freq, frames) which is correct for spectrogram visualization
                        signal_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            signal_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                        relevance_timefreq[axis, :actual_freq, :actual_frames] = np.transpose(
                            relevance_tf_axis[0, :actual_frames, :actual_freq], (1, 0))
                    elif signal_tf_axis.ndim == 2:
                        if signal_tf_axis.shape[0] == 1:  # Shape (batch, freq*frames)
                            # Reshape to get proper time-frequency shape
                            reshaped_signal = signal_tf_axis[0].reshape(n_frames, -1)
                            reshaped_relevance = relevance_tf_axis[0].reshape(n_frames, -1)

                            # Transpose to get (freq, frames)
                            actual_freq = min(reshaped_signal.shape[1], freq_length)
                            actual_frames = min(reshaped_signal.shape[0], n_frames)

                            signal_timefreq[axis, :actual_freq, :actual_frames] = reshaped_signal[:actual_frames,
                                                                                  :actual_freq].T
                            relevance_timefreq[axis, :actual_freq, :actual_frames] = reshaped_relevance[:actual_frames,
                                                                                     :actual_freq].T
            except Exception as e:
                print(f"Error computing time-frequency domain for axis {axis}: {e}")

        # Clean up memory
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Error in time-frequency domain computation: {e}")

    # Compute frequency bins for visualization
    freqs = fftfreq(signal_length, d=1.0 / sampling_rate)[:freq_length]

    # Final cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (relevance_time, relevance_freq, signal_freq, relevance_timefreq, signal_timefreq,
            input_signal, freqs, target)
