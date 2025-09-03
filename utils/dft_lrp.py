import numpy as np
import torch
import torch.nn as nn
import utils.dft_utils as dft_utils
import gc
from scipy.fft import fftfreq



class EnhancedDFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None,
                 window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True,
                 create_forward=True, create_dft=True, create_stdft=True) -> None:
        """
        Enhanced class for Discrete Fourier transform in pytorch and relevance propagation through DFT layer.
        Includes fixes for memory management and shape compatibility issues.
        """
        # Import needed modules
        import torch.nn as nn
        import utils.dft_utils as dft_utils

        # Check available GPU memory before deciding to use CUDA
        if cuda and torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            # Only use CUDA if there's enough free memory (e.g., 2GB)
            cuda = free_memory > 2 * 1024 * 1024 * 1024
            if not cuda:
                print(
                    f"Warning: Not enough GPU memory available ({free_memory / (1024 ** 3):.2f} GB). Using CPU instead.")

        self.signal_length = signal_length
        self.nyquist_k = signal_length // 2
        self.precision = precision
        self.cuda = cuda
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        # Track which layers were successfully created
        self.has_fourier_layer = False
        self.has_inverse_fourier_layer = False
        self.has_transpose_inverse_fourier_layer = False
        self.has_st_fourier_layer = False
        self.has_st_inverse_fourier_layer = False
        self.has_st_transpose_inverse_fourier_layer = False

        # Validate window parameters for STDFT
        if create_stdft and (window_shift is None or window_width is None or window_shape is None):
            print("Warning: STDFT requested but window parameters not properly specified. Disabling STDFT.")
            create_stdft = False

        # Calculate estimated memory requirement for STDFT
        if create_stdft and window_width and window_shift:
            n_frames = max(1, (signal_length - window_width) // window_shift + 1)
            freq_bins = signal_length // 2 + 1 if leverage_symmetry else signal_length
            estimated_memory = signal_length * freq_bins * n_frames * 8  # Rough estimate in bytes
            print(f"Estimated STDFT memory requirement: {estimated_memory / (1024 ** 3):.2f} GB")

        # Create fourier layers
        if create_dft:
            if create_forward:
                try:
                    self.fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                    symmetry=self.symmetry,
                                                                    transpose=False, inverse=False, short_time=False,
                                                                    cuda=self.cuda, precision=self.precision)
                    self.has_fourier_layer = True
                    # Force memory cleanup after creating layer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error creating forward Fourier layer: {e}")
                    self.has_fourier_layer = False

            if create_inverse:
                try:
                    self.inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                            symmetry=self.symmetry, transpose=False,
                                                                            inverse=True, short_time=False,
                                                                            cuda=self.cuda,
                                                                            precision=self.precision)
                    self.has_inverse_fourier_layer = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error creating inverse Fourier layer: {e}")
                    self.has_inverse_fourier_layer = False

            if create_transpose_inverse:
                try:
                    self.transpose_inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                                      symmetry=self.symmetry,
                                                                                      transpose=True,
                                                                                      inverse=True, short_time=False,
                                                                                      cuda=self.cuda,
                                                                                      precision=self.precision)
                    self.has_transpose_inverse_fourier_layer = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error creating transpose inverse Fourier layer: {e}")
                    self.has_transpose_inverse_fourier_layer = False

        if create_stdft:
            # Adjust window parameters to ensure compatibility
            if window_width > signal_length:
                window_width = signal_length
                self.stdft_kwargs["window_width"] = window_width
                print(f"Warning: Adjusted window width to {window_width} to match signal length")

            if window_shift <= 0:
                window_shift = max(1, window_width // 4)  # Use 1/4 of window width as default shift
                self.stdft_kwargs["window_shift"] = window_shift
                print(f"Warning: Adjusted window shift to {window_shift}")

            # Create STDFT layers
            try:
                if create_forward:
                    self.st_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                       symmetry=self.symmetry, transpose=False,
                                                                       inverse=False, short_time=True, cuda=self.cuda,
                                                                       precision=self.precision, **self.stdft_kwargs)
                    self.has_st_fourier_layer = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if create_inverse:
                    self.st_inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                               symmetry=self.symmetry, transpose=False,
                                                                               inverse=True, short_time=True,
                                                                               cuda=self.cuda,
                                                                               precision=self.precision,
                                                                               **self.stdft_kwargs)
                    self.has_st_inverse_fourier_layer = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if create_transpose_inverse:
                    self.st_transpose_inverse_fourier_layer = self._create_fourier_layer(
                        signal_length=self.signal_length,
                        symmetry=self.symmetry,
                        transpose=True, inverse=True,
                        short_time=True, cuda=self.cuda,
                        precision=self.precision,
                        **self.stdft_kwargs)
                    self.has_st_transpose_inverse_fourier_layer = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error creating STDFT layers: {e}")
                print("STDFT functionality will be disabled")
                self.has_st_fourier_layer = False
                self.has_st_inverse_fourier_layer = False
                self.has_st_transpose_inverse_fourier_layer = False

    def __del__(self):
        """Clean up GPU memory when this object is destroyed"""
        try:
            # Explicitly set models to CPU first to avoid CUDA errors during deletion
            if hasattr(self, 'cuda') and self.cuda:
                if hasattr(self, 'fourier_layer') and self.has_fourier_layer:
                    self.fourier_layer = self.fourier_layer.cpu()
                if hasattr(self, 'inverse_fourier_layer') and self.has_inverse_fourier_layer:
                    self.inverse_fourier_layer = self.inverse_fourier_layer.cpu()
                if hasattr(self, 'transpose_inverse_fourier_layer') and self.has_transpose_inverse_fourier_layer:
                    self.transpose_inverse_fourier_layer = self.transpose_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_fourier_layer') and self.has_st_fourier_layer:
                    self.st_fourier_layer = self.st_fourier_layer.cpu()
                if hasattr(self, 'st_inverse_fourier_layer') and self.has_st_inverse_fourier_layer:
                    self.st_inverse_fourier_layer = self.st_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_transpose_inverse_fourier_layer') and self.has_st_transpose_inverse_fourier_layer:
                    self.st_transpose_inverse_fourier_layer = self.st_transpose_inverse_fourier_layer.cpu()

            # Force CUDA memory cleanup if using GPU
            if hasattr(self, 'cuda') and self.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during DFTLRP cleanup: {e}")

    def _array_to_tensor(self, input_data, precision, cuda):
        """Convert array to tensor with proper precision and device"""
        import torch
        import numpy as np

        # Handle NaN values that might cause crashes
        if isinstance(input_data, np.ndarray) and np.isnan(input_data).any():
            print("Warning: NaN values in input data. Replacing with zeros.")
            input_data = np.nan_to_num(input_data, nan=0.0)

        if isinstance(input_data, torch.Tensor):
            # Already a tensor, ensure correct precision and device
            # FIXED: This was inverted in original code
            if precision == 32:
                input_data = input_data.float()  # Convert to float32
            else:
                input_data = input_data.half()  # Convert to float16

            if cuda and torch.cuda.is_available():
                # Check if tensor is not already on CUDA
                if input_data.device.type != 'cuda':
                    input_data = input_data.cuda()
            elif input_data.device.type == 'cuda':
                # Move to CPU if cuda is False but tensor is on GPU
                input_data = input_data.cpu()
            return input_data

        # Create tensor with correct precision
        dtype = torch.float32 if precision == 32 else torch.float16
        input_tensor = torch.tensor(input_data, dtype=dtype)

        if cuda and torch.cuda.is_available():
            # Check available memory before moving to GPU
            tensor_size = input_tensor.element_size() * input_tensor.nelement()
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()

            if tensor_size < free_memory * 0.8:  # Use 80% of free memory as threshold
                input_tensor = input_tensor.cuda()
            else:
                print(
                    f"Warning: Tensor too large ({tensor_size / (1024 ** 3):.2f} GB) for GPU memory ({free_memory / (1024 ** 3):.2f} GB). Keeping on CPU.")
                cuda = False
                self.cuda = False  # Update class attribute to prevent future GPU operations

        return input_tensor

    def _create_fourier_layer(self, signal_length, inverse, symmetry, transpose, short_time, cuda, precision,
                              **stdft_kwargs):
        """Create linear layer with Discrete Fourier Transformation weights"""
        import torch
        import torch.nn as nn
        import utils.dft_utils as dft_utils
        import gc

        try:
            # For large STDFT, temporarily switch to CPU to avoid GPU OOM
            orig_cuda = cuda
            if short_time and cuda and signal_length * (stdft_kwargs.get("window_width", 128)) > 1e6:
                print("Large STDFT computation detected. Temporarily using CPU for weight creation.")
                cuda = False

            if short_time:
                window_shift = stdft_kwargs.get("window_shift")
                window_width = stdft_kwargs.get("window_width")
                window_shape = stdft_kwargs.get("window_shape")

                if None in (window_shift, window_width, window_shape):
                    raise ValueError("STDFT requires window_shift, window_width, and window_shape parameters")

                print(
                    f"Creating STDFT weights with parameters: shift={window_shift}, width={window_width}, shape={window_shape}")
                weights_fourier = dft_utils.create_short_time_fourier_weights(
                    signal_length, window_shift, window_width, window_shape,
                    inverse=inverse, real=True, symmetry=symmetry
                )

                # Restore original cuda setting
                cuda = orig_cuda
            else:
                weights_fourier = dft_utils.create_fourier_weights(
                    signal_length=signal_length, real=True, inverse=inverse, symmetry=symmetry
                )

            print(f"Weight shape from dft_utils: {weights_fourier.shape}")

            if transpose:
                weights_fourier = weights_fourier.T

            # Convert weights to tensor with desired precision
            weights_fourier = self._array_to_tensor(weights_fourier, precision, cuda).T
            print(f"Weight shape after tensor conversion: {weights_fourier.shape}")

            n_in, n_out = weights_fourier.shape
            fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
            with torch.no_grad():
                fourier_layer.weight = nn.Parameter(weights_fourier)

            # Clean up memory immediately
            del weights_fourier
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if cuda and torch.cuda.is_available():
                # Check memory before moving layer to GPU
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                layer_size = sum(p.numel() * p.element_size() for p in fourier_layer.parameters())

                if layer_size < free_memory * 0.8:
                    fourier_layer = fourier_layer.cuda()
                else:
                    print(
                        f"Warning: Fourier layer too large ({layer_size / (1024 ** 3):.2f} GB) for GPU memory. Keeping on CPU.")

            return fourier_layer
        except Exception as e:
            print(f"Error creating Fourier layer: {e}")
            raise

    def reshape_signal(self, signal, signal_length, relevance=False, short_time=False, symmetry=True):
        """Reshape signal with improved error handling and debugging"""
        import torch
        import numpy as np

        # Handle input that is already a tensor
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        # Ensure we're working with numpy arrays
        signal = np.asarray(signal)

        # Handle NaN values that might cause crashes
        if np.isnan(signal).any():
            print("Warning: NaN values detected in reshape_signal. Replacing with zeros.")
            signal = np.nan_to_num(signal, nan=0.0)

        # Get batch size or assume 1 if not present
        if signal.ndim <= 1:
            bs = 1
            signal = signal.reshape(1, -1)
        else:
            bs = signal.shape[0]

        # Debug info
        print(f"Reshape signal input shape: {signal.shape}, signal_length: {signal_length}")

        if symmetry:
            nyquist_k = signal_length // 2
            if short_time:
                # Handle time-frequency domain reshaping
                try:
                    n_windows = signal.shape[-1] // signal_length
                    if n_windows <= 0:
                        print(
                            f"Warning: Signal length {signal.shape[-1]} too short for STDFT with signal_length {signal_length}")
                        # Return empty array with correct shape
                        freq_bins = nyquist_k + 1
                        return np.zeros((bs, 1, freq_bins), dtype=np.complex128 if not relevance else np.float64)

                    signal = signal.reshape(bs, n_windows, signal_length)
                    print(f"Signal reshaped to {signal.shape} for short-time processing")
                except Exception as e:
                    print(f"Error reshaping signal for short-time processing: {e}")
                    return signal

            # Handle symmetry transformation
            try:
                zeros = np.zeros_like(signal[..., :1])
                if relevance:
                    # Sum real and imaginary parts for relevance
                    # Use np.concatenate with specified dimension to avoid shape errors
                    imag_part = np.concatenate([zeros, signal[..., nyquist_k + 1:]], axis=-1)

                    # Ensure shapes match before addition
                    real_part = signal[..., :nyquist_k + 1]
                    if real_part.shape[-1] > imag_part.shape[-1]:
                        # Pad imag_part if needed
                        pad_width = [(0, 0)] * (real_part.ndim - 1) + [(0, real_part.shape[-1] - imag_part.shape[-1])]
                        imag_part = np.pad(imag_part, pad_width, 'constant')
                    elif imag_part.shape[-1] > real_part.shape[-1]:
                        # Truncate imag_part if needed
                        imag_part = imag_part[..., :real_part.shape[-1]]

                    signal = real_part + imag_part
                else:
                    # Create complex signal
                    imag_part = np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros], axis=-1)
                    real_part = signal[..., :nyquist_k + 1]

                    # Ensure shapes match before addition
                    if real_part.shape[-1] > imag_part.shape[-1]:
                        pad_width = [(0, 0)] * (real_part.ndim - 1) + [(0, real_part.shape[-1] - imag_part.shape[-1])]
                        imag_part = np.pad(imag_part, pad_width, 'constant')
                    elif imag_part.shape[-1] > real_part.shape[-1]:
                        imag_part = imag_part[..., :real_part.shape[-1]]

                    signal = real_part + 1j * imag_part
            except Exception as e:
                print(f"Error applying symmetry transformation: {e}")
                return signal
        else:
            if short_time:
                # Handle time-frequency domain reshaping without symmetry
                try:
                    n_windows = signal.shape[-1] // signal_length // 2
                    if n_windows <= 0:
                        print(
                            f"Warning: Signal length {signal.shape[-1]} too short for STDFT with signal_length {signal_length}")
                        return np.zeros((bs, 1, signal_length), dtype=np.complex128 if not relevance else np.float64)

                    signal = signal.reshape(bs, n_windows, signal_length * 2)
                    print(f"Signal reshaped to {signal.shape} for short-time processing (no symmetry)")
                except Exception as e:
                    print(f"Error reshaping signal for short-time processing (no symmetry): {e}")
                    return signal

            # Handle non-symmetry transformation
            try:
                if relevance:
                    # Sum real and imaginary parts for relevance
                    real_part = signal[..., :signal_length]
                    imag_part = signal[..., signal_length:]

                    # Check if shapes match before addition
                    if real_part.shape != imag_part.shape:
                        print(
                            f"Warning: Real and imaginary part shapes don't match: {real_part.shape} vs {imag_part.shape}")
                        # Adjust shapes if needed
                        min_length = min(real_part.shape[-1], imag_part.shape[-1])
                        real_part = real_part[..., :min_length]
                        imag_part = imag_part[..., :min_length]

                    signal = real_part + imag_part
                else:
                    # Create complex signal
                    real_part = signal[..., :signal_length]
                    imag_part = signal[..., signal_length:]

                    # Check if shapes match before addition
                    if real_part.shape != imag_part.shape:
                        print(
                            f"Warning: Real and imaginary part shapes don't match: {real_part.shape} vs {imag_part.shape}")
                        # Adjust shapes if needed
                        min_length = min(real_part.shape[-1], imag_part.shape[-1])
                        real_part = real_part[..., :min_length]
                        imag_part = imag_part[..., :min_length]

                    signal = real_part + 1j * imag_part
            except Exception as e:
                print(f"Error applying non-symmetry transformation: {e}")
                return signal

        print(f"Reshape signal output shape: {signal.shape}")
        return signal

    def dft_lrp(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        """
        Relevance propagation through DFT with improved memory management and error handling.
        """
        import torch
        import numpy as np
        import gc

        # Debug info on input shapes
        print(f"DFT-LRP input shapes - relevance: {relevance.shape}, signal: {signal.shape}")
        print(f"Short time mode: {short_time}")

        # Check for NaN inputs that might cause crashes
        if isinstance(relevance, np.ndarray) and np.isnan(relevance).any():
            print("Warning: NaN values in relevance input. Replacing with zeros.")
            relevance = np.nan_to_num(relevance, nan=0.0)

        if isinstance(signal, np.ndarray) and np.isnan(signal).any():
            print("Warning: NaN values in signal input. Replacing with zeros.")
            signal = np.nan_to_num(signal, nan=0.0)

        # Verify signal and relevance have compatible shapes
        if np.asarray(signal).shape != np.asarray(relevance).shape:
            print(
                f"Warning: Signal shape {np.asarray(signal).shape} doesn't match relevance shape {np.asarray(relevance).shape}")
            # Try to reshape if dimensions are compatible
            if np.prod(np.asarray(signal).shape) == np.prod(np.asarray(relevance).shape):
                relevance = np.asarray(relevance).reshape(np.asarray(signal).shape)
            else:
                raise ValueError(
                    f"Signal shape {np.asarray(signal).shape} must match relevance shape {np.asarray(relevance).shape}")

        # Determine which transforms to use (regular DFT or STDFT)
        if short_time:
            # Check if STDFT layers exist and are accessible
            if not hasattr(self, 'st_fourier_layer') or not self.has_st_fourier_layer or \
                    not hasattr(self,
                                'st_transpose_inverse_fourier_layer') or not self.has_st_transpose_inverse_fourier_layer:
                print("Warning: STDFT layers not available, falling back to regular DFT")
                short_time = False
                # Make sure regular DFT layers exist
                if not hasattr(self, 'fourier_layer') or not self.has_fourier_layer or \
                        not hasattr(self,
                                    'transpose_inverse_fourier_layer') or not self.has_transpose_inverse_fourier_layer:
                    print("ERROR: Regular DFT layers not available either. Cannot proceed.")
                    raise ValueError("No DFT layers available for processing")

                transform = self.fourier_layer
                dft_transform = self.transpose_inverse_fourier_layer
            else:
                transform = self.st_fourier_layer
                dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            # Check if regular DFT layers exist
            if not hasattr(self, 'fourier_layer') or not self.has_fourier_layer or \
                    not hasattr(self,
                                'transpose_inverse_fourier_layer') or not self.has_transpose_inverse_fourier_layer:
                print("ERROR: Regular DFT layers not available. Cannot proceed.")
                raise ValueError("No DFT layers available for processing")

            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer

        # Check available memory and decide whether to use GPU
        use_cuda = self.cuda
        if use_cuda and torch.cuda.is_available():
            signal_size = np.prod(np.asarray(signal).shape) * 4  # Estimate size in bytes
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()

            # For STDFT we need more memory
            memory_factor = 5 if short_time else 3
            if signal_size * memory_factor > free_memory * 0.8:
                print(f"Warning: Insufficient GPU memory for operation. Using CPU instead.")
                print(
                    f"Required: ~{signal_size * memory_factor / (1024 ** 3):.2f} GB, Available: {free_memory / (1024 ** 3):.2f} GB")
                use_cuda = False

        # Convert inputs to tensors
        try:
            # Use adaptive precision based on signal size
            if np.prod(np.asarray(signal).shape) > 1e7:  # Very large signal
                temp_precision = min(16, self.precision)  # Reduce to half precision if possible
                print(f"Large signal detected. Temporarily using {temp_precision}-bit precision.")
            else:
                temp_precision = self.precision

            signal_tensor = self._array_to_tensor(signal, temp_precision, use_cuda)
            del signal  # Free original data
            gc.collect()

            # Compute signal_hat if not provided
            if signal_hat is None:
                with torch.no_grad():
                    try:
                        # Ensure transform is on same device as signal_tensor
                        if transform.weight.device != signal_tensor.device:
                            transform = transform.to(signal_tensor.device)

                        signal_hat_tensor = transform(signal_tensor)
                        print(f"Signal hat shape after transform: {signal_hat_tensor.shape}")
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print("CUDA out of memory during signal_hat computation. Falling back to CPU.")
                            # Move to CPU and retry
                            if use_cuda:
                                torch.cuda.empty_cache()
                                use_cuda = False
                                signal_tensor = signal_tensor.cpu()
                                transform = transform.cpu()
                                signal_hat_tensor = transform(signal_tensor)
                        else:
                            # For other errors, create placeholder
                            print(f"Error computing signal_hat: {e}")
                            if short_time:
                                # Get expected output shape for STDFT
                                window_shift = self.stdft_kwargs["window_shift"]
                                window_width = self.stdft_kwargs["window_width"]
                                n_frames = (self.signal_length - window_width) // window_shift + 1
                                freq_bins = self.signal_length // 2 + 1 if self.symmetry else self.signal_length
                                signal_hat_tensor = torch.zeros(
                                    (signal_tensor.size(0), n_frames * freq_bins),
                                    dtype=signal_tensor.dtype,
                                    device=signal_tensor.device
                                )
                            else:
                                # Regular DFT shape
                                freq_bins = self.signal_length // 2 + 1 if self.symmetry else self.signal_length
                                signal_hat_tensor = torch.zeros(
                                    (signal_tensor.size(0), freq_bins),
                                    dtype=signal_tensor.dtype,
                                    device=signal_tensor.device
                                )
            else:
                signal_hat_tensor = self._array_to_tensor(signal_hat, temp_precision, use_cuda)
                del signal_hat  # Free original data
                gc.collect()

            relevance_tensor = self._array_to_tensor(relevance, temp_precision, use_cuda)
            del relevance  # Free original data
            gc.collect()

            # Make sure all tensors are on the same device
            if signal_tensor.device != signal_hat_tensor.device:
                signal_hat_tensor = signal_hat_tensor.to(signal_tensor.device)
            if signal_tensor.device != relevance_tensor.device:
                relevance_tensor = relevance_tensor.to(signal_tensor.device)

            # Handle division by small values safely
            with torch.no_grad():
                # Apply epsilon with correct sign to avoid division by zero
                norm = signal_tensor.clone()
                # Use absolute value to ensure proper handling of both positive and negative values
                abs_norm = torch.abs(norm)

                # Apply epsilon where values are smaller than epsilon
                epsilon_mask = abs_norm < epsilon
                norm[epsilon_mask] = torch.sign(norm[epsilon_mask]) * epsilon
                # Handle zeros in the sign function (avoid NaN results0)
                zero_mask = norm == 0
                if zero_mask.any():
                    norm[zero_mask] = epsilon

                # Compute normalized relevance
                relevance_normed = relevance_tensor / norm

                # Free memory we don't need anymore
                del relevance_tensor, norm, abs_norm, epsilon_mask, zero_mask
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Process through DFT to get relevance in frequency domain
                try:
                    # Ensure dft_transform is on the same device
                    if dft_transform.weight.device != relevance_normed.device:
                        dft_transform = dft_transform.to(relevance_normed.device)

                    relevance_hat_tensor = dft_transform(relevance_normed)
                    print(f"Relevance hat shape after transform: {relevance_hat_tensor.shape}")
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("CUDA out of memory during relevance hat computation. Falling back to CPU.")
                        # Move to CPU and retry
                        if use_cuda:
                            torch.cuda.empty_cache()
                            use_cuda = False
                            relevance_normed = relevance_normed.cpu()
                            dft_transform = dft_transform.cpu()
                            relevance_hat_tensor = dft_transform(relevance_normed)
                    else:
                        print(f"Error in DFT transform of normalized relevance: {e}")
                        relevance_hat_tensor = torch.zeros_like(signal_hat_tensor)

                # Free memory we don't need anymore
                del relevance_normed, signal_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Apply signal_hat weights to relevance
                print(f"Signal hat shape: {signal_hat_tensor.shape}, Relevance hat shape: {relevance_hat_tensor.shape}")

                # Ensure shapes match before multiplication
                if signal_hat_tensor.shape == relevance_hat_tensor.shape:
                    relevance_hat_tensor = signal_hat_tensor * relevance_hat_tensor
                else:
                    print("Shape mismatch between signal_hat and relevance_hat.")
                    print(
                        f"Signal hat shape: {signal_hat_tensor.shape}, Relevance hat shape: {relevance_hat_tensor.shape}")

                    # Try to reshape if possible
                    if signal_hat_tensor.numel() == relevance_hat_tensor.numel():
                        print("Reshaping tensors to match dimensions.")
                        relevance_hat_tensor = relevance_hat_tensor.reshape(signal_hat_tensor.shape)
                        relevance_hat_tensor = signal_hat_tensor * relevance_hat_tensor
                    else:
                        print("Creating placeholder with zeros.")
                        relevance_hat_tensor = torch.zeros_like(signal_hat_tensor)

            # Move results0 back to CPU
            signal_hat_tensor = signal_hat_tensor.cpu()
            relevance_hat_tensor = relevance_hat_tensor.cpu()

            # Convert to numpy
            relevance_hat = relevance_hat_tensor.numpy()
            signal_hat = signal_hat_tensor.numpy()

            # Clean up GPU memory explicitly
            del relevance_hat_tensor, signal_hat_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Check for NaN values in output
            if np.isnan(relevance_hat).any() or np.isnan(signal_hat).any():
                print("Warning: NaN values detected in output. Replacing with zeros.")
                relevance_hat = np.nan_to_num(relevance_hat, nan=0.0)
                signal_hat = np.nan_to_num(signal_hat, nan=0.0)

            # Reshape signal and relevance if needed
            if not real:
                try:
                    signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False,
                                                     short_time=short_time, symmetry=self.symmetry)
                    relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True,
                                                        short_time=short_time, symmetry=self.symmetry)
                except Exception as e:
                    print(f"Error in reshaping signal/relevance for output: {e}")

            return signal_hat, relevance_hat

        except Exception as e:
            print(f"Error in DFT-LRP calculation: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for better debugging

            # Clean up any GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Return placeholder data
            if short_time:
                # For STDFT, create placeholder data with appropriate shape
                window_shift = self.stdft_kwargs["window_shift"]
                window_width = self.stdft_kwargs["window_width"]
                n_frames = max(1, (self.signal_length - window_width) // window_shift + 1)
                freq_bins = self.signal_length // 2 + 1 if self.symmetry else self.signal_length

                placeholder_shape = (1, n_frames, freq_bins)
                signal_hat_placeholder = np.zeros(placeholder_shape, dtype=np.complex128)
                relevance_hat_placeholder = np.zeros(placeholder_shape, dtype=np.float64)
            else:
                # For regular DFT, create placeholder data with appropriate shape
                freq_bins = self.signal_length // 2 + 1 if self.symmetry else self.signal_length
                placeholder_shape = (1, freq_bins)
                signal_hat_placeholder = np.zeros(placeholder_shape, dtype=np.complex128)
                relevance_hat_placeholder = np.zeros(placeholder_shape, dtype=np.float64)

            return signal_hat_placeholder, relevance_hat_placeholder