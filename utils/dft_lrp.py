import numpy as np
import torch
import torch.nn as nn
import utils.dft_utils as dft_utils


class DFTLRP():
    def __init__(self, signal_length, precision=32, cuda=True, leverage_symmetry=False, window_shift=None,
                 window_width=None, window_shape=None, create_inverse=True, create_transpose_inverse=True,
                 create_forward=True, create_dft=True, create_stdft=True) -> None:
        self.signal_length = signal_length
        self.nyquist_k = signal_length // 2
        self.precision = precision
        self.cuda = cuda
        self.symmetry = leverage_symmetry
        self.stdft_kwargs = {"window_shift": window_shift, "window_width": window_width, "window_shape": window_shape}

        if create_dft:
            if create_forward:
                self.fourier_layer = self.create_fourier_layer(signal_length=self.signal_length, symmetry=self.symmetry,
                                                               transpose=False, inverse=False, short_time=False,
                                                               cuda=self.cuda, precision=self.precision)
            if create_inverse:
                self.inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                       symmetry=self.symmetry, transpose=False,
                                                                       inverse=True, short_time=False, cuda=self.cuda,
                                                                       precision=self.precision)
            if create_transpose_inverse:
                self.transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                 symmetry=self.symmetry, transpose=True,
                                                                                 inverse=True, short_time=False,
                                                                                 cuda=self.cuda,
                                                                                 precision=self.precision)

        if create_stdft:
            if create_forward:
                self.st_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                  symmetry=self.symmetry,
                                                                  transpose=False, inverse=False, short_time=True,
                                                                  cuda=self.cuda, precision=self.precision,
                                                                  **self.stdft_kwargs)
            if create_inverse:
                self.st_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                          symmetry=self.symmetry, transpose=False,
                                                                          inverse=True, short_time=True,
                                                                          cuda=self.cuda, precision=self.precision,
                                                                          **self.stdft_kwargs)
            if create_transpose_inverse:
                self.st_transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                    symmetry=self.symmetry,
                                                                                    transpose=True,
                                                                                    inverse=True, short_time=True,
                                                                                    cuda=self.cuda,
                                                                                    precision=self.precision,
                                                                                    **self.stdft_kwargs)

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

        n_out, n_in = weights_fourier.shape  # Adjusted to match nn.Linear convention
        expected_out = signal_length if not symmetry else signal_length // 2 + 1
        assert n_out == expected_out, f"Expected output dimension {expected_out}, got {n_out}"
        fourier_layer = torch.nn.Linear(n_in, n_out, bias=False)
        with torch.no_grad():
            fourier_layer.weight = nn.Parameter(weights_fourier)
        del weights_fourier

        if cuda:
            fourier_layer = fourier_layer.cuda()

        return fourier_layer

    @staticmethod
    def reshape_signal(signal: np.ndarray, signal_length: int, relevance: bool, short_time: bool, symmetry: bool):
        bs = signal.shape[0]
        if symmetry:
            nyquist_k = signal_length // 2
            if short_time:
                n_windows = signal.shape[-1] // signal_length
                signal = signal.reshape(bs, n_windows, signal_length)
            if relevance:
                pass  # Keep real-valued for relevance
            else:
                zeros = np.zeros_like(signal)
                signal = signal + 1j * zeros  # Convert to complex with zero imaginary part
        else:
            if short_time:
                n_windows = signal.shape[-1] // signal_length // 2
                signal = signal.reshape(bs, n_windows, signal_length * 2)
            if relevance:
                signal = signal[..., :signal_length] + signal[..., signal_length:]
            else:
                signal = signal[..., :signal_length] + 1j * signal[..., signal_length:]
        return signal

    def fourier_transform(self, signal: np.ndarray, real: bool = True, inverse: bool = False,
                          short_time: bool = False) -> np.ndarray:
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

        if len(signal.shape) == 3:
            batch_size, channels, signal_len = signal.shape
            signal = signal.reshape(batch_size * channels, signal_len)
            signal = self._array_to_tensor(signal, self.precision, self.cuda)
            with torch.no_grad():
                signal_hat = transform(signal).cpu().numpy()
            freq_length = self.signal_length // 2 + 1 if self.symmetry else self.signal_length
            signal_hat = signal_hat.reshape(batch_size, channels, freq_length)
        else:
            signal = self._array_to_tensor(signal, self.precision, self.cuda)
            with torch.no_grad():
                signal_hat = transform(signal).cpu().numpy()

        if not real and not inverse:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
        return signal_hat

    def dft_lrp(self, relevance: np.ndarray, signal: np.ndarray, signal_hat=None, short_time=False, epsilon=1e-6,
                real=False) -> np.ndarray:
        if short_time:
            transform = self.st_fourier_layer
            dft_transform = self.st_transpose_inverse_fourier_layer
        else:
            transform = self.fourier_layer
            dft_transform = self.transpose_inverse_fourier_layer

        input_shape = signal.shape
        print(f"Input signal shape: {input_shape}")
        print(f"Input relevance shape: {relevance.shape}")
        if len(input_shape) == 3:
            batch_size, channels, signal_len = input_shape
            signal = signal.reshape(batch_size * channels, signal_len)
            relevance = relevance.reshape(batch_size * channels, signal_len)
        else:
            batch_size, signal_len = input_shape
            channels = 1

        signal = self._array_to_tensor(signal, self.precision, self.cuda)
        if signal_hat is None:
            signal_hat = transform(signal)
        else:
            signal_hat = self._array_to_tensor(signal_hat, self.precision, self.cuda)
        print(f"Signal hat shape after transform: {signal_hat.shape}")

        relevance = self._array_to_tensor(relevance, self.precision, self.cuda)
        norm = signal + epsilon
        relevance_normed = relevance / norm

        with torch.no_grad():
            relevance_hat = dft_transform(relevance_normed)
            print(f"Relevance hat shape before multiplication: {relevance_hat.shape}")
            relevance_hat = signal_hat * relevance_hat
            print(f"Relevance hat shape after multiplication: {relevance_hat.shape}")

        freq_length = self.signal_length // 2 + 1 if self.symmetry else self.signal_length
        print(f"Expected freq_length: {freq_length}")
        relevance_hat = relevance_hat.cpu().numpy().reshape(batch_size, channels, freq_length)
        signal_hat = signal_hat.detach().numpy().reshape(batch_size, channels, freq_length)
        print(f"Signal shape after transform: {signal_hat.shape}")
        print(f"Relevance hat shape after transform: {relevance_hat.shape}")

        if not real:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True,
                                                short_time=short_time, symmetry=self.symmetry)

        return signal_hat, relevance_hat