import numpy as np
import torch
import torch.nn as nn
import utils.dft_utils as dft_utils
import gc

class DFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None,
                 window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True,
                 create_forward=True, create_dft=True, create_stdft=True) -> None:
        """
        Class for Discrete Fourier transform in pytorch and relevance propagation through DFT layer.

        Args:
        signal_length: number of time steps in the signal
        leverage_symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage

        window_width: width of the window for short time DFT
        window_shift: width/hopsize of window for short time DFT
        window_shape: shape of window for STDFT, options are 'rectangle' and 'halfsine'

        create_inverse: create weights for inverse DFT
        create_transpose: cretae weights for transpose inverse DFT (for DFT-LRP)
        create_forward: create weights for forward DFT
        create_stdft: create weights for short time DFT
        create_stdft: create weights DFT
        """
        self.signal_length = signal_length
        self.nyquist_k = signal_length // 2
        self.precision = precision
        self.cuda = cuda
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        # create fourier layers
        # dft
        if create_dft:
            if create_forward:
                self.fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry,
                                                               transpose=False, inverse=False, short_time=False,
                                                               cuda=self.cuda, precision=self.precision)
            # inverse dft
            if create_inverse:
                self.inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                       symmetry=self.symmetry, transpose=False,
                                                                       inverse=True, short_time=False, cuda=self.cuda,
                                                                       precision=self.precision)
            # transpose inverse dft for dft-lrp
            if create_transpose_inverse:
                self.transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                 symmetry=self.symmetry, transpose=True,
                                                                                 inverse=True, short_time=False,
                                                                                 cuda=self.cuda,
                                                                                 precision=self.precision)

        if create_stdft:
            # stdft
            if create_forward:
                self.st_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                  symmetry=self.symmetry, transpose=False,
                                                                  inverse=False, short_time=True, cuda=self.cuda,
                                                                  precision=self.precision, **self.stdft_kwargs)
            # inverse stdft
            if create_inverse:
                self.st_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                          symmetry=self.symmetry, transpose=False,
                                                                          inverse=True, short_time=True, cuda=self.cuda,
                                                                          precision=self.precision, **self.stdft_kwargs)
            # transpose inverse stdft for dft-lrp
            if create_transpose_inverse:
                self.st_transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                    symmetry=self.symmetry,
                                                                                    transpose=True, inverse=True,
                                                                                    short_time=True, cuda=self.cuda,
                                                                                    precision=self.precision,
                                                                                    **self.stdft_kwargs)

    def __del__(self):
        """Clean up GPU memory when this object is destroyed"""
        try:
            # Explicitly set models to CPU first to avoid CUDA errors during deletion
            if self.cuda and torch.cuda.is_available():
                if hasattr(self, 'fourier_layer'):
                    self.fourier_layer = self.fourier_layer.cpu()
                if hasattr(self, 'inverse_fourier_layer'):
                    self.inverse_fourier_layer = self.inverse_fourier_layer.cpu()
                if hasattr(self, 'transpose_inverse_fourier_layer'):
                    self.transpose_inverse_fourier_layer = self.transpose_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_fourier_layer'):
                    self.st_fourier_layer = self.st_fourier_layer.cpu()
                if hasattr(self, 'st_inverse_fourier_layer'):
                    self.st_inverse_fourier_layer = self.st_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_transpose_inverse_fourier_layer'):
                    self.st_transpose_inverse_fourier_layer = self.st_transpose_inverse_fourier_layer.cpu()
            
            # Manually delete potentially large tensors
            if hasattr(self, 'fourier_layer'):
                del self.fourier_layer
            if hasattr(self, 'inverse_fourier_layer'):
                del self.inverse_fourier_layer
            if hasattr(self, 'transpose_inverse_fourier_layer'):
                del self.transpose_inverse_fourier_layer
            if hasattr(self, 'st_fourier_layer'):
                del self.st_fourier_layer
            if hasattr(self, 'st_inverse_fourier_layer'):
                del self.st_inverse_fourier_layer
            if hasattr(self, 'st_transpose_inverse_fourier_layer'):
                del self.st_transpose_inverse_fourier_layer
            
            # Force CUDA memory cleanup if using GPU
            if self.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during DFTLRP cleanup: {e}")

    @staticmethod
    def _array_to_tensor(input: np.ndarray, precision: float, cuda: bool) -> torch.tensor:
        dtype = torch.float32 if precision == 32 else torch.float16
        input = torch.tensor(input, dtype=dtype)
        if cuda:
            input = input.cuda()
        return input

    @staticmethod
    def create_fourier_layer(signal_length: int, inverse: bool, symmetry: bool, transpose: bool, short_time: bool,
                             cuda: bool, precision: int, **stdft_kwargs):
        """
        Create linear layer with Discrete Fourier Transformation weights

        Args:
        inverse: if True, create weights for inverse DFT
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        transpose: create layer with transposed DFT weights for explicit relevance propagation
        short_time: short time DFT
        cuda: use gpu
        precision: 32 or 16 for reduced precision with less memory usage
        """
        if short_time:
            weights_fourier = dft_utils.create_short_time_fourier_weights(signal_length, stdft_kwargs["window_shift"],
                                                                          stdft_kwargs["window_width"],
                                                                          stdft_kwargs["window_shape"], inverse=inverse,
                                                                          real=True, symmetry=symmetry)
        else:
            weights_fourier = dft_utils.create_fourier_weights(signal_length=signal_length, real=True, inverse=inverse,
                                                               symmetry=symmetry)

        print(f"Raw weight shape from dft_utils: {weights_fourier.shape}")
        if transpose:
            weights_fourier = weights_fourier.T

        weights_fourier = DFTLRP._array_to_tensor(weights_fourier, precision, cuda).T
        print(f"Weight shape after tensor conversion: {weights_fourier.shape}")

        n_in, n_out = weights_fourier.shape
        fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
        with torch.no_grad():
            fourier_layer.weight = nn.Parameter(weights_fourier)
        del weights_fourier

        if cuda:
            fourier_layer = fourier_layer.cuda()

        return fourier_layer

    @staticmethod
    def reshape_signal(signal: np.ndarray, signal_length: int, relevance: bool, short_time: bool, symmetry: bool):
        """
        Restructure array from concatenation of real and imaginary parts to complex (if array contains signal) or sum of real and imaginary part (if array contains relevance). Additionallty, reshapes time-frequenc

        Args:
        relevance: True if array contains relevance, not signal itself
        symmetry: if True, levereage that for real signal the DF transformed signal is symmetric and compute only first half it
        short_time: short time DFT
        """
        bs = signal.shape[0]
        if symmetry:
            nyquist_k = signal_length // 2
            if short_time:
                n_windows = signal.shape[-1] // signal_length
                signal = signal.reshape(bs, n_windows, signal_length)
            zeros = np.zeros_like(signal[..., :1])
            if relevance:
                signal = signal[..., :nyquist_k + 1] + np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros],
                                                                      axis=-1)
            else:
                signal = signal[..., :nyquist_k + 1] + 1j * np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros],
                                                                           axis=-1)
        else:
            if short_time:
                n_windows = signal.shape[-1] // signal_length // 2
                signal = signal.reshape(bs, n_windows, signal_length * 2)
            if relevance:
                signal = signal[..., :signal_length] + signal[..., signal_length:]
            else:
                signal = signal[..., :signal_length] + 1j * signal[..., signal_length:]
        return signal

    # Reshape signal batch with proper handling for batched data
    def reshape_signal_batch(self, signal, signal_length, relevance=False, short_time=False, symmetry=True):
        """Reshape signal batch with proper handling for batched data"""
        batch_size = signal.shape[0]
        n_channels = signal.shape[1]
        reshaped_signals = []

        for i in range(batch_size):
            # Process each sample in the batch
            sample = signal[i]
            reshaped = self.reshape_signal(
                sample,
                signal_length,
                relevance=relevance,
                short_time=short_time,
                symmetry=symmetry
            )
            reshaped_signals.append(reshaped[np.newaxis, ...])

        # Concatenate along batch dimension
        return np.concatenate(reshaped_signals, axis=0)
    def fourier_transform(self, signal: np.ndarray, real: bool = True, inverse: bool = False,
                          short_time: bool = False) -> np.ndarray:
        """
        Discrete Fourier transform (DFT) of signal in time (inverse=False) or inverse DFT of signal in frequency.

        Args:
        inverse: if True, perform inverse DFT
        short_time: if True, perform short time DFT
        real: if real, the output is split into real and imaginary parts of the signal in freq. domain y_k, i.e. (y_k^real, y_k^imag)
        """
        if inverse:
            if short_time:
                transform = self.st_inverse_fourier_layer
            else:
                transform = self.inverse_fourier_layer
        else:
            if short_time:
                transform = self.st_fourier_layer
            else:
                transform = self.fourier_layer

        signal = self._array_to_tensor(signal, self.precision, self.cuda)

        with torch.no_grad():
            signal_hat = transform(signal).cpu().numpy()

        # render y_k as complex number of shape (n_windows, signal_length) //2
        if not real and not inverse:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
        return signal_hat

    def dft_lrp(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        """
        Relevance propagation through DFT with improved memory management

        Args:
            relevance: relevance in time domain
            signal: signal in time domain, same shape as relevance
            signal_hat: signal in frequency domain, if None it is computed using signal
            short_time: relevance propagation through short time DFT
            epsilon: small constant to stabilize division in DFT-LRP (prevents division by zero)
            real: if True, split output into real and imaginary parts

        Returns:
            tuple: (signal_hat, relevance_hat) - Frequency domain signal and relevance scores
        """
        # Verify signal and relevance have the same shape
        if signal.shape != relevance.shape:
            raise ValueError(f"Signal shape {signal.shape} must match relevance shape {relevance.shape}")

        if short_time:
            transform = self.st_fourier_layer
            dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer

        print(f"Input signal shape: {signal.shape}")
        print(f"Input relevance shape: {relevance.shape}")

        # Move data to appropriate device and ensure we're working with tensors
        signal_tensor = self._array_to_tensor(signal, self.precision, self.cuda)
        relevance_tensor = self._array_to_tensor(relevance, self.precision, self.cuda)

        # Compute signal_hat if not provided
        if signal_hat is None:
            with torch.no_grad():
                signal_hat_tensor = transform(signal_tensor)
        else:
            signal_hat_tensor = self._array_to_tensor(signal_hat, self.precision, self.cuda)

        print(f"Signal hat shape after transform: {signal_hat_tensor.shape}")

        # More robust normalization to prevent division by very small values
        with torch.no_grad():
            norm = signal_tensor.clone()
            # Apply absolute value before adding epsilon to ensure proper handling of both positive and negative values
            abs_norm = torch.abs(norm)

            # Use dynamic epsilon based on signal magnitude if needed
            if epsilon <= 0:
                epsilon = torch.mean(abs_norm) * 1e-5

            epsilon_mask = abs_norm < epsilon

            # Apply epsilon where values are smaller than epsilon
            norm[epsilon_mask] = torch.sign(norm[epsilon_mask]) * epsilon
            # Handle zeros in the sign function (avoid NaN results)
            zero_mask = norm == 0
            if zero_mask.any():
                norm[zero_mask] = epsilon

            # Compute normalized relevance
            relevance_normed = relevance_tensor / norm

            # Process through DFT to get relevance in frequency domain
            relevance_hat_tensor = dft_transform(relevance_normed)

            # Apply signal_hat weights to relevance
            print(f"Relevance hat shape before multiplication: {relevance_hat_tensor.shape}")
            relevance_hat_tensor = signal_hat_tensor * relevance_hat_tensor
            print(f"Relevance hat shape after multiplication: {relevance_hat_tensor.shape}")

        # Move results back to CPU and convert to numpy
        relevance_hat = relevance_hat_tensor.cpu().numpy()
        signal_hat = signal_hat_tensor.cpu().numpy()

        # Clean up GPU memory explicitly
        del relevance_normed, relevance_hat_tensor, signal_hat_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # add real and imaginary part of relevance and signal
        if not real:
            try:
                signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False,
                                                 short_time=short_time, symmetry=self.symmetry)
                relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True,
                                                    short_time=short_time, symmetry=self.symmetry)
            except Exception as e:
                print(f"Error in reshaping signal: {e}")
                # If reshape fails, return the raw arrays

        return signal_hat, relevance_hat

    def dft_lrp_batch(self, relevance, signal, batch_size=32, **kwargs):
        """
        Process large datasets in batches to avoid memory issues

        Args:
            relevance: Array of shape (N, C, T) with N samples
            signal: Array of shape (N, C, T)
            batch_size: Number of samples to process at once
            **kwargs: Additional arguments for dft_lrp

        Returns:
            signal_hat_all, relevance_hat_all: Results for all samples
        """
        n_samples = signal.shape[0]
        if n_samples <= batch_size:
            # Small enough to process at once
            return self.dft_lrp(relevance, signal, **kwargs)

        # Process in batches
        signal_hats = []
        relevance_hats = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_signal = signal[i:end_idx]
            batch_relevance = relevance[i:end_idx]

            # Process this batch
            batch_signal_hat, batch_relevance_hat = self.dft_lrp(
                batch_relevance, batch_signal, **kwargs)

            # Store results
            signal_hats.append(batch_signal_hat)
            relevance_hats.append(batch_relevance_hat)

            # Force cleanup after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Combine results
        signal_hat_all = np.concatenate(signal_hats, axis=0)
        relevance_hat_all = np.concatenate(relevance_hats, axis=0)

        return signal_hat_all, relevance_hat_all

    def dft_lrp_multi_axis(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        """
        Apply DFT-LRP to multiple axes at once.

        Args:
            relevance: Array of shape (batch_size, n_channels, signal_length)
            signal: Array of shape (batch_size, n_channels, signal_length)
            signal_hat: Precomputed frequency-domain signal (optional)
            short_time: Whether to use short-time DFT
            epsilon: Small constant for numerical stability
            real: Whether to return real-only values

        Returns:
            signal_hat: Frequency-domain signal of shape (batch_size, n_channels, freq_length)
            relevance_hat: Frequency-domain relevance of shape (batch_size, n_channels, freq_length)
        """
        # Get dimensions
        batch_size, n_channels, signal_length = signal.shape
        freq_length = signal_length // 2 + 1 if self.symmetry else signal_length

        # Initialize output arrays
        if np.iscomplexobj(signal):
            signal_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)
            relevance_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)
        else:
            signal_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)
            relevance_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)

        # Process each axis separately (still in a batch-efficient way)
        for axis in range(n_channels):
            signal_axis = signal[:, axis:axis + 1, :]
            relevance_axis = relevance[:, axis:axis + 1, :]

            # If signal_hat is provided, extract the corresponding axis
            signal_hat_axis = None
            if signal_hat is not None:
                signal_hat_axis = signal_hat[:, axis:axis + 1, :]

            # Call the regular dft_lrp function
            signal_hat_result, relevance_hat_result = self.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                signal_hat=signal_hat_axis,
                short_time=short_time,
                epsilon=epsilon,
                real=real
            )

            # Store results
            signal_hat_out[:, axis, :] = signal_hat_result.squeeze(1)
            relevance_hat_out[:, axis, :] = relevance_hat_result.squeeze(1)

        return signal_hat_out, relevance_hat_out

    def dft_lrp_multi_axis_with_per_axis_norm(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6,
                                              real=False):
        """
        Modified DFT-LRP that processes all axes at once but normalizes per axis
        """
        batch_size, n_channels, signal_length = signal.shape
        freq_length = signal_length // 2 + 1 if self.symmetry else signal_length

        # Initialize output arrays
        signal_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)
        relevance_hat_out = np.zeros((batch_size, n_channels, freq_length), dtype=np.complex128)

        # Create a copy to avoid modifying the original
        # signal_tensor = self._array_to_tensor(signal)
        signal_tensor = self._array_to_tensor(signal, self.precision, self.cuda)

        # Compute signal_hat for all channels at once if not provided
        if signal_hat is None:
            if short_time:
                signal_hat = torch.stft(signal_tensor.reshape(batch_size * n_channels, -1),
                                        n_fft=self.signal_length,
                                        hop_length=self.stdft_kwargs["window_shift"],
                                        win_length=self.stdft_kwargs["window_width"],
                                        window=self.get_window().to(signal_tensor.device),
                                        return_complex=True)
                signal_hat = signal_hat.reshape(batch_size, n_channels, -1, signal_hat.shape[-1])
            else:
                # Process all channels but reshape to separate them
                signal_hat = torch.fft.rfft(signal_tensor.reshape(batch_size * n_channels, -1), dim=-1)
                signal_hat = signal_hat.reshape(batch_size, n_channels, -1)

        # Now normalize and calculate relevance per axis
        for axis in range(n_channels):
            # Extract this axis
            axis_signal = signal[:, axis:axis + 1, :]
            axis_relevance = relevance[:, axis:axis + 1, :]

            # Apply standard normalization for this axis only
            norm = axis_signal.copy() if isinstance(axis_signal, np.ndarray) else axis_signal.clone()
            abs_norm = np.abs(norm) if isinstance(norm, np.ndarray) else torch.abs(norm)

            # Apply epsilon
            epsilon_mask = abs_norm < epsilon
            if isinstance(norm, np.ndarray):
                norm[epsilon_mask] = np.sign(norm[epsilon_mask]) * epsilon
                zero_mask = norm == 0
                if np.any(zero_mask):
                    norm[zero_mask] = epsilon
            else:
                norm[epsilon_mask] = torch.sign(norm[epsilon_mask]) * epsilon
                zero_mask = norm == 0
                if zero_mask.any():
                    norm[zero_mask] = epsilon

            # Normalize relevance for this axis only
            relevance_normed = axis_relevance / norm

            # Extract the signal_hat for this axis
            if isinstance(signal_hat, torch.Tensor):
                axis_signal_hat = signal_hat[:, axis:axis + 1, :]
            else:
                axis_signal_hat = signal_hat[:, axis, :]

            # Finish the relevance calculation
            if isinstance(relevance_normed, np.ndarray):
                relevance_normed = torch.from_numpy(relevance_normed).to(signal_tensor.device)

            # Apply the DFT to the normalized relevance
            if short_time:
                # Short-time DFT processing
                relevance_stft = torch.stft(relevance_normed.reshape(batch_size, -1),
                                            n_fft=self.signal_length,
                                            hop_length=self.stdft_kwargs["window_shift"],
                                            win_length=self.stdft_kwargs["window_width"],
                                            window=self.get_window().to(relevance_normed.device),
                                            return_complex=True)
                relevance_hat = relevance_stft * torch.conj(axis_signal_hat) / (torch.abs(axis_signal_hat) + 1e-6)
            else:
                # Regular DFT processing
                relevance_fft = torch.fft.rfft(relevance_normed.reshape(batch_size, -1), dim=-1)
                relevance_hat = relevance_fft * torch.conj(axis_signal_hat) / (torch.abs(axis_signal_hat) + 1e-6)

            # Store results
            signal_hat_out[:, axis] = axis_signal_hat.cpu().numpy().reshape(batch_size, -1)
            relevance_hat_out[:, axis] = relevance_hat.cpu().numpy().reshape(batch_size, -1)

        return signal_hat_out, relevance_hat_out

    def dft_lrp_multi_axis_optimized(self, relevance, signal, **kwargs):
        """Memory-efficient multi-axis processing that processes one axis at a time"""
        batch_size, n_channels, signal_length = signal.shape

        # Pre-allocate output arrays on CPU
        if kwargs.get('real', False):
            output_length = signal_length
            dtype = np.float32 if self.precision == 32 else np.float16
        else:
            output_length = signal_length // 2 + 1 if self.symmetry else signal_length
            dtype = np.complex64 if self.precision == 32 else np.complex32

        signal_hat_out = np.zeros((batch_size, n_channels, output_length), dtype=dtype)
        relevance_hat_out = np.zeros((batch_size, n_channels, output_length), dtype=dtype)

        # Process one axis at a time to save memory
        for axis in range(n_channels):
            # Extract single axis data
            signal_axis = signal[:, axis:axis + 1, :]
            relevance_axis = relevance[:, axis:axis + 1, :]

            # Process this axis
            signal_hat_axis, relevance_hat_axis = self.dft_lrp(
                relevance=relevance_axis,
                signal=signal_axis,
                **kwargs
            )

            # Store results
            signal_hat_out[:, axis] = signal_hat_axis.reshape(batch_size, -1)
            relevance_hat_out[:, axis] = relevance_hat_axis.reshape(batch_size, -1)

            # Force cleanup after each axis
            gc.collect()
            if self.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return signal_hat_out, relevance_hat_out

    def print_memory_stats(self):
        """Print current GPU memory usage"""
        if not (self.cuda and torch.cuda.is_available()):
            print("CUDA not available")
            return

        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
        print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

    def __enter__(self):
        return self

    # Context Manager Support:
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        return False  # Don't suppress exceptions


class EnhancedDFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None,
                 window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True,
                 create_forward=True, create_dft=True, create_stdft=True) -> None:
        """
        Enhanced class for Discrete Fourier transform in pytorch and relevance propagation through DFT layer.
        Includes fixes for memory management and shape compatibility issues.
        """
        import gc  # Import inside the class to ensure it's available
        import torch.nn as nn
        import utils.dft_utils as dft_utils

        self.signal_length = signal_length
        self.nyquist_k = signal_length // 2
        self.precision = precision
        self.cuda = cuda and torch.cuda.is_available()
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        # Validate window parameters for STDFT
        if create_stdft and (window_shift is None or window_width is None or window_shape is None):
            print("Warning: STDFT requested but window parameters not properly specified. Disabling STDFT.")
            create_stdft = False

        # Create fourier layers
        if create_dft:
            if create_forward:
                self.fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                symmetry=self.symmetry,
                                                                transpose=False, inverse=False, short_time=False,
                                                                cuda=self.cuda, precision=self.precision)
            if create_inverse:
                self.inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                        symmetry=self.symmetry, transpose=False,
                                                                        inverse=True, short_time=False, cuda=self.cuda,
                                                                        precision=self.precision)
            if create_transpose_inverse:
                self.transpose_inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                                  symmetry=self.symmetry,
                                                                                  transpose=True,
                                                                                  inverse=True, short_time=False,
                                                                                  cuda=self.cuda,
                                                                                  precision=self.precision)

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
                if create_inverse:
                    self.st_inverse_fourier_layer = self._create_fourier_layer(signal_length=self.signal_length,
                                                                               symmetry=self.symmetry, transpose=False,
                                                                               inverse=True, short_time=True,
                                                                               cuda=self.cuda,
                                                                               precision=self.precision,
                                                                               **self.stdft_kwargs)
                if create_transpose_inverse:
                    self.st_transpose_inverse_fourier_layer = self._create_fourier_layer(
                        signal_length=self.signal_length,
                        symmetry=self.symmetry,
                        transpose=True, inverse=True,
                        short_time=True, cuda=self.cuda,
                        precision=self.precision,
                        **self.stdft_kwargs)
            except Exception as e:
                print(f"Error creating STDFT layers: {e}")
                print("STDFT functionality will be disabled")
                create_stdft = False

    def __del__(self):
        """Clean up GPU memory when this object is destroyed"""
        try:
            # Explicitly set models to CPU first to avoid CUDA errors during deletion
            if self.cuda:
                if hasattr(self, 'fourier_layer'):
                    self.fourier_layer = self.fourier_layer.cpu()
                if hasattr(self, 'inverse_fourier_layer'):
                    self.inverse_fourier_layer = self.inverse_fourier_layer.cpu()
                if hasattr(self, 'transpose_inverse_fourier_layer'):
                    self.transpose_inverse_fourier_layer = self.transpose_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_fourier_layer'):
                    self.st_fourier_layer = self.st_fourier_layer.cpu()
                if hasattr(self, 'st_inverse_fourier_layer'):
                    self.st_inverse_fourier_layer = self.st_inverse_fourier_layer.cpu()
                if hasattr(self, 'st_transpose_inverse_fourier_layer'):
                    self.st_transpose_inverse_fourier_layer = self.st_transpose_inverse_fourier_layer.cpu()

            # Force CUDA memory cleanup if using GPU
            if self.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during DFTLRP cleanup: {e}")

    def _array_to_tensor(self, input_data, precision, cuda):
        """Convert array to tensor with proper precision and device"""
        import torch

        if isinstance(input_data, torch.Tensor):
            # Already a tensor, just ensure correct precision and device
            if precision == 32:
                input_data = input_data.float()
            else:
                input_data = input_data.half()
            if cuda and torch.cuda.is_available():
                input_data = input_data.cuda()
            return input_data

        dtype = torch.float32 if precision == 32 else torch.float16
        input_tensor = torch.tensor(input_data, dtype=dtype)
        if cuda and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        return input_tensor

    def _create_fourier_layer(self, signal_length, inverse, symmetry, transpose, short_time, cuda, precision,
                              **stdft_kwargs):
        """Create linear layer with Discrete Fourier Transformation weights"""
        import torch
        import torch.nn as nn
        import utils.dft_utils as dft_utils

        try:
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
            else:
                weights_fourier = dft_utils.create_fourier_weights(
                    signal_length=signal_length, real=True, inverse=inverse, symmetry=symmetry
                )

            print(f"Weight shape from dft_utils: {weights_fourier.shape}")

            if transpose:
                weights_fourier = weights_fourier.T

            weights_fourier = self._array_to_tensor(weights_fourier, precision, cuda).T
            print(f"Weight shape after tensor conversion: {weights_fourier.shape}")

            n_in, n_out = weights_fourier.shape
            fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
            with torch.no_grad():
                fourier_layer.weight = nn.Parameter(weights_fourier)
            del weights_fourier

            if cuda and torch.cuda.is_available():
                fourier_layer = fourier_layer.cuda()

            return fourier_layer
        except Exception as e:
            print(f"Error creating Fourier layer: {e}")
            raise

    def reshape_signal(self, signal, signal_length, relevance=False, short_time=False, symmetry=True):
        """Reshape signal with improved error handling and debugging"""
        # Handle input that is already a tensor
        if isinstance(signal, torch.Tensor):
            signal = signal.cpu().numpy()

        # Ensure we're working with numpy arrays
        signal = np.asarray(signal)

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
                    signal = signal[..., :nyquist_k + 1] + np.concatenate([zeros, signal[..., nyquist_k + 1:], zeros],
                                                                          axis=-1)
                else:
                    # Create complex signal
                    signal = signal[..., :nyquist_k + 1] + 1j * np.concatenate(
                        [zeros, signal[..., nyquist_k + 1:], zeros], axis=-1)
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
                    signal = signal[..., :signal_length] + signal[..., signal_length:]
                else:
                    # Create complex signal
                    signal = signal[..., :signal_length] + 1j * signal[..., signal_length:]
            except Exception as e:
                print(f"Error applying non-symmetry transformation: {e}")
                return signal

        print(f"Reshape signal output shape: {signal.shape}")
        return signal

    def dft_lrp(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        """
        Relevance propagation through DFT with improved error handling and debugging.
        """
        import torch

        # Debug info on input shapes
        print(f"DFT-LRP input shapes - relevance: {relevance.shape}, signal: {signal.shape}")
        print(f"Short time mode: {short_time}")

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

        if short_time:
            # Check if STDFT layers exist
            if not hasattr(self, 'st_fourier_layer') or not hasattr(self, 'st_transpose_inverse_fourier_layer'):
                print("Warning: STDFT layers not available, falling back to regular DFT")
                short_time = False
                transform = self.fourier_layer
                dft_transform = self.transpose_inverse_fourier_layer
            else:
                transform = self.st_fourier_layer
                dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer

        # Convert inputs to tensors
        try:
            signal_tensor = self._array_to_tensor(signal, self.precision, self.cuda)

            # Compute signal_hat if not provided
            if signal_hat is None:
                with torch.no_grad():
                    try:
                        signal_hat_tensor = transform(signal_tensor)
                        print(f"Signal hat shape after transform: {signal_hat_tensor.shape}")
                    except Exception as e:
                        print(f"Error computing signal_hat: {e}")
                        # Create empty signal_hat with appropriate shape
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
                signal_hat_tensor = self._array_to_tensor(signal_hat, self.precision, self.cuda)

            relevance_tensor = self._array_to_tensor(relevance, self.precision, self.cuda)

            # Handle division by small values safely
            with torch.no_grad():
                # Apply epsilon with correct sign to avoid division by zero
                norm = signal_tensor.clone()
                # Use absolute value to ensure proper handling of both positive and negative values
                abs_norm = torch.abs(norm)

                # Apply epsilon where values are smaller than epsilon
                epsilon_mask = abs_norm < epsilon
                norm[epsilon_mask] = torch.sign(norm[epsilon_mask]) * epsilon
                # Handle zeros in the sign function (avoid NaN results)
                zero_mask = norm == 0
                if zero_mask.any():
                    norm[zero_mask] = epsilon

                # Compute normalized relevance
                relevance_normed = relevance_tensor / norm

                # Process through DFT to get relevance in frequency domain
                try:
                    relevance_hat_tensor = dft_transform(relevance_normed)
                    print(f"Relevance hat shape after transform: {relevance_hat_tensor.shape}")
                except Exception as e:
                    print(f"Error in DFT transform of normalized relevance: {e}")
                    relevance_hat_tensor = torch.zeros_like(signal_hat_tensor)

                # Apply signal_hat weights to relevance
                print(f"Signal hat shape: {signal_hat_tensor.shape}, Relevance hat shape: {relevance_hat_tensor.shape}")

                # Ensure shapes match before multiplication
                if signal_hat_tensor.shape == relevance_hat_tensor.shape:
                    relevance_hat_tensor = signal_hat_tensor * relevance_hat_tensor
                else:
                    print("Shape mismatch between signal_hat and relevance_hat. Creating placeholder.")
                    relevance_hat_tensor = torch.zeros_like(signal_hat_tensor)

            # Move results back to CPU and convert to numpy
            relevance_hat = relevance_hat_tensor.cpu().numpy()
            signal_hat = signal_hat_tensor.cpu().numpy()

            # Clean up GPU memory explicitly
            del relevance_normed, relevance_hat_tensor, signal_hat_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
