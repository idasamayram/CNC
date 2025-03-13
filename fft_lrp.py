import torch
import numpy as np
from utils import fft_utils  # Assumed to contain get_window; replace if unavailable

class FFTLRP:
    def __init__(
        self,
        signal_length,
        leverage_symmetry=True,
        precision=32,
        cuda=True,
        window_shift=1,
        window_width=128,
        window_shape="rectangle",
        create_inverse=True,
        create_transpose_inverse=True,
        create_forward=True,
        create_stdft=True
    ):
        """
        Initialize the FFTLRP class for Layer-wise Relevance Propagation in frequency domain.

        Args:
            signal_length (int): Length of the input signal (e.g., 2000).
            leverage_symmetry (bool): Whether to use symmetry in FFT (rfft instead of fft).
            precision (int): Floating-point precision (32 or 64).
            cuda (bool): Whether to use CUDA if available.
            window_shift (int): Hop length for STFT.
            window_width (int): Window width for STFT.
            window_shape (str): Shape of the window (e.g., "rectangle").
            create_inverse (bool): Whether to create inverse transform.
            create_transpose_inverse (bool): Whether to create transpose inverse transform.
            create_forward (bool): Whether to create forward transform.
            create_stdft (bool): Whether to create STFT transform.
        """
        self.signal_length = signal_length
        self.symmetry = leverage_symmetry
        self.precision = precision
        self.cuda = cuda and torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.stdft_kwargs = {
            "window_shift": window_shift,
            "window_width": window_width,
            "window_shape": window_shape
        }
        self.create_inverse = create_inverse
        self.create_transpose_inverse = create_transpose_inverse
        self.create_forward = create_forward
        self.create_stdft = create_stdft

        # Set precision for tensors
        self.dtype = torch.float32 if self.precision == 32 else torch.float64

    def _array_to_tensor(self, arr):
        """
        Convert a numpy array to a PyTorch tensor with the correct device and dtype.

        Args:
            arr (np.ndarray): Input array.

        Returns:
            torch.Tensor: Tensor on the specified device.
        """
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr).to(dtype=self.dtype)
        return arr.to(self.device)

    def fourier_transform(self, signal, real=False, inverse=False, short_time=False):
        """
        Compute the Fourier transform (or STFT) of the signal.

        Args:
            signal (np.ndarray or torch.Tensor): Input signal.
            real (bool): Whether to return real values (not used in this implementation).
            inverse (bool): Whether to compute the inverse transform.
            short_time (bool): Whether to compute the Short-Time Fourier Transform (STFT).

        Returns:
            torch.Tensor: Transformed signal (frequency or time-frequency domain).
        """
        signal_tensor = self._array_to_tensor(signal)
        print(f"fourier_transform: signal_tensor shape = {signal_tensor.shape}")

        if short_time:
            window = fft_utils.get_window(self.stdft_kwargs["window_width"], self.stdft_kwargs["window_shape"])
            signal_hat = torch.stft(
                signal_tensor,
                n_fft=self.signal_length,
                hop_length=self.stdft_kwargs["window_shift"],
                win_length=self.stdft_kwargs["window_width"],
                window=window.to(signal_tensor.device),
                return_complex=True
            )
            print(f"fourier_transform (STFT): signal_hat shape = {signal_hat.shape}")
        else:
            if inverse:
                signal_hat = torch.fft.irfft(signal_tensor, n=self.signal_length, dim=-1)
            else:
                signal_hat = torch.fft.rfft(signal_tensor, dim=-1)
            print(f"fourier_transform (FFT): signal_hat shape = {signal_hat.shape}")

        return signal_hat

    def reshape_signal(self, signal, signal_length, relevance=False, short_time=False, symmetry=True):
        """
        Reshape the transformed signal or relevance to the expected format.

        Args:
            signal (np.ndarray or torch.Tensor): Transformed signal or relevance.
            signal_length (int): Length of the original signal.
            relevance (bool): Whether the input is relevance (affects handling).
            short_time (bool): Whether the signal is in time-frequency domain (STFT).
            symmetry (bool): Whether to leverage symmetry (rfft-style).

        Returns:
            np.ndarray: Reshaped signal or relevance.
        """
        signal = self._array_to_tensor(signal)
        print(f"reshape_signal: input signal shape = {signal.shape}")

        if short_time:
            # Time-frequency domain: shape (batch, freq_bins, time_frames)
            if signal.dim() == 3:  # (batch, freq, time)
                signal = signal.cpu().numpy()
            else:
                raise ValueError(f"Unexpected shape for short_time=True: {signal.shape}")
        else:
            # Frequency domain: shape (batch, freq_bins)
            freq_length = signal_length // 2 + 1 if symmetry else signal_length
            if signal.dim() == 2:  # (batch, freq)
                if signal.shape[-1] != freq_length:
                    raise ValueError(f"Frequency dimension mismatch: got {signal.shape[-1]}, expected {freq_length}")
                signal = signal.cpu().numpy()
            else:
                raise ValueError(f"Unexpected shape for short_time=False: {signal.shape}")

        print(f"reshape_signal: output signal shape = {signal.shape}")
        return signal

    def fft_lrp(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        """
        Compute Layer-wise Relevance Propagation in the frequency or time-frequency domain.

        Args:
            relevance (np.ndarray): Relevance scores in the time domain.
            signal (np.ndarray): Input signal in the time domain.
            signal_hat (np.ndarray, optional): Precomputed Fourier transform of the signal.
            short_time (bool): Whether to compute in time-frequency domain (STFT).
            epsilon (float): Small value for numerical stability.
            real (bool): Whether to return real values (not used).

        Returns:
            tuple: (signal_hat, relevance_hat)
                - signal_hat: Transformed signal in frequency or time-frequency domain.
                - relevance_hat: Relevance scores in frequency or time-frequency domain.
        """
        print(f"fft_lrp: Input relevance shape = {relevance.shape}, signal shape = {signal.shape}")
        signal_tensor = self._array_to_tensor(signal)
        relevance_tensor = self._array_to_tensor(relevance)
        print(f"fft_lrp: relevance_tensor shape = {relevance_tensor.shape}, signal_tensor shape = {signal_tensor.shape}")

        # Compute the Fourier transform of the signal if not provided
        if signal_hat is None:
            signal_hat = self.fourier_transform(signal, real=real, inverse=False, short_time=short_time)
        signal_hat_tensor = self._array_to_tensor(signal_hat)
        print(f"fft_lrp: signal_hat shape = {signal_hat.shape}")

        # Normalize relevance in the time domain
        norm = signal_tensor + epsilon
        relevance_normed = relevance_tensor / norm
        print(f"fft_lrp: relevance_normed shape = {relevance_normed.shape}")

        # Compute relevance in the frequency or time-frequency domain
        if short_time:
            window = fft_utils.get_window(self.stdft_kwargs["window_width"], self.stdft_kwargs["window_shape"])
            relevance_stft = torch.stft(
                relevance_normed,
                n_fft=self.signal_length,
                hop_length=self.stdft_kwargs["window_shift"],
                win_length=self.stdft_kwargs["window_width"],
                window=window.to(relevance_normed.device),
                return_complex=True
            )
            print(f"fft_lrp: relevance_stft shape = {relevance_stft.shape}")
            relevance_hat = torch.abs(relevance_stft) * torch.abs(signal_hat_tensor)
        else:
            relevance_fft = torch.fft.rfft(relevance_normed, dim=-1)
            print(f"fft_lrp: relevance_fft shape = {relevance_fft.shape}")
            relevance_hat = torch.abs(relevance_fft) * torch.abs(signal_hat_tensor)

        print(f"fft_lrp: relevance_hat shape (before cpu) = {relevance_hat.shape}")
        relevance_hat = relevance_hat.cpu().numpy()
        print(f"fft_lrp: relevance_hat shape (after cpu) = {relevance_hat.shape}")

        # Reshape the outputs if needed
        if not real:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time, symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True, short_time=short_time, symmetry=self.symmetry)
            print(f"fft_lrp: relevance_hat shape (after reshape) = {relevance_hat.shape}")

        return signal_hat, relevance_hat

    def __del__(self):
        """
        Clean up any resources (e.g., CUDA memory).
        """
        pass  # Add cleanup if needed (e.g., clearing CUDA cache)