import torch
import numpy as np
from utils import fft_utils  # Assumed to contain get_window; replace if unavailable


class FFTLRP:
    def __init__(self, signal_length, leverage_symmetry=True, precision=32, cuda=True, window_shift=1, window_width=128,
                 window_shape="rectangle", create_inverse=True, create_transpose_inverse=True, create_forward=True, create_stdft=True):
        self.signal_length = signal_length
        self.symmetry = leverage_symmetry
        self.precision = precision
        self.cuda = cuda and torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}
        self.create_inverse = create_inverse
        self.create_transpose_inverse = create_transpose_inverse
        self.create_forward = create_forward
        self.create_stdft = create_stdft
        self.dtype = torch.float32 if self.precision == 32 else torch.float64

    def _array_to_tensor(self, arr):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr).to(dtype=self.dtype)
        return arr.to(self.device)

    def fourier_transform(self, signal, real=False, inverse=False, short_time=False):
        signal_tensor = self._array_to_tensor(signal)
        print(f"fourier_transform: signal_tensor shape = {signal_tensor.shape}")
        if short_time:
            window = fft_utils.get_window(self.stdft_kwargs["window_width"], self.stdft_kwargs["window_shape"])
            signal_hat = torch.stft(signal_tensor, n_fft=self.signal_length, hop_length=self.stdft_kwargs["window_shift"], win_length=self.stdft_kwargs["window_width"], window=window.to(signal_tensor.device), return_complex=True)
            print(f"fourier_transform (STFT): signal_hat shape = {signal_hat.shape}")
        else:
            if inverse:
                signal_hat = torch.fft.irfft(signal_tensor, n=self.signal_length, dim=-1)
            else:
                signal_hat = torch.fft.rfft(signal_tensor, dim=-1)
            print(f"fourier_transform (FFT): signal_hat shape = {signal_hat.shape}")
        return signal_hat

    def reshape_signal(self, signal, signal_length, relevance=False, short_time=False, symmetry=True):
        signal = self._array_to_tensor(signal)
        print(f"reshape_signal: input signal shape = {signal.shape}")
        if short_time:
            if signal.dim() == 3:
                signal = signal.cpu().numpy()
            else:
                raise ValueError(f"Unexpected shape for short_time=True: {signal.shape}")
        else:
            freq_length = signal_length // 2 + 1 if symmetry else signal_length
            if signal.dim() == 2:
                if signal.shape[-1] != freq_length:
                    raise ValueError(f"Frequency dimension mismatch: got {signal.shape[-1]}, expected {freq_length}")
                signal = signal.cpu().numpy()
            else:
                raise ValueError(f"Unexpected shape for short_time=False: {signal.shape}")
        print(f"reshape_signal: output signal shape = {signal.shape}")
        return signal

    def fft_lrp(self, relevance, signal, signal_hat=None, short_time=False, epsilon=1e-6, real=False):
        print(f"fft_lrp: Input relevance shape = {relevance.shape}, signal shape = {signal.shape}")
        signal_tensor = self._array_to_tensor(signal)
        relevance_tensor = self._array_to_tensor(relevance)
        print(f"fft_lrp: relevance_tensor shape = {relevance_tensor.shape}, signal_tensor shape = {signal_tensor.shape}")

        if signal_hat is None:
            signal_hat = self.fourier_transform(signal, real=real, inverse=False, short_time=short_time)
        signal_hat_tensor = self._array_to_tensor(signal_hat)
        print(f"fft_lrp: signal_hat shape = {signal_hat.shape}")

        norm = signal_tensor + epsilon
        relevance_normed = relevance_tensor / norm
        print(f"fft_lrp: relevance_normed shape = {relevance_normed.shape}")

        # Apply 1/sqrt(N) normalization to match DFT-LRP
        norm_factor = 1.0 / np.sqrt(self.signal_length)


        if short_time:
            relevance_stft = torch.stft(relevance_normed, n_fft=self.signal_length,
                                        hop_length=self.stdft_kwargs["window_shift"],
                                        win_length=self.stdft_kwargs["window_width"],
                                        window=fft_utils.get_window(self.stdft_kwargs["window_width"],
                                                                    self.stdft_kwargs["window_shape"]).to(relevance_normed.device),
                                        return_complex=True)
            print(f"fft_lrp: relevance_stft real part = {relevance_stft.real}")
            print(f"fft_lrp: relevance_stft imag part = {relevance_stft.imag}")
            relevance_hat = relevance_stft * torch.conj(signal_hat_tensor) / (torch.abs(signal_hat_tensor) + 1e-6)
            relevance_hat = relevance_hat * norm_factor  # Normalize STFT output
            signal_hat_tensor = signal_hat_tensor * norm_factor  # Normalize signal_hat
            print(f"fft_lrp: signal_hat magnitude = {torch.abs(signal_hat)}")
            print(f"fft_lrp: relevance_hat magnitude = {torch.abs(relevance_hat)}")
            print(f"fft_lrp: relevance_hat real part before cpu = {relevance_hat.real}")

        else:
            relevance_fft = torch.fft.rfft(relevance_normed, dim=-1) * norm_factor  # Normalize FFT output
            signal_hat_tensor = signal_hat_tensor * norm_factor  # Normalize signal_hat
            relevance_hat = relevance_fft * torch.conj(signal_hat_tensor) / (torch.abs(signal_hat_tensor) + 1e-6)
            print(f"fft_lrp: relevance_fft real part = {relevance_fft.real}")
            print(f"fft_lrp: relevance_fft imag part = {relevance_fft.imag}")
            print(f"fft_lrp: relevance_hat real part before cpu = {relevance_hat.real}")

        print(f"fft_lrp: relevance_hat shape (before cpu) = {relevance_hat.shape}")
        relevance_hat = relevance_hat.cpu().numpy()
        print(f"fft_lrp: relevance_hat shape (after cpu) = {relevance_hat.shape}")

        if not real:
            signal_hat = self.reshape_signal(signal_hat_tensor, self.signal_length, relevance=False, short_time=short_time, symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True, short_time=short_time, symmetry=self.symmetry)
            print(f"fft_lrp: relevance_hat shape (after reshape) = {relevance_hat.shape}")

        return signal_hat, relevance_hat

    def __del__(self):
        pass