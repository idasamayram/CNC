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
        self.freq_length = self.signal_length // 2 + 1 if self.symmetry else self.signal_length

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
                                                                  symmetry=self.symmetry, transpose=False,
                                                                  inverse=False, short_time=True, cuda=self.cuda,
                                                                  precision=self.precision, **self.stdft_kwargs)
            if create_inverse:
                self.st_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                          symmetry=self.symmetry, transpose=False,
                                                                          inverse=True, short_time=True, cuda=self.cuda,
                                                                          precision=self.precision, **self.stdft_kwargs)
            if create_transpose_inverse:
                self.st_transpose_inverse_fourier_layer = self.create_fourier_layer(signal_length=self.signal_length,
                                                                                    symmetry=self.symmetry,
                                                                                    transpose=True, inverse=True,
                                                                                    short_time=True, cuda=self.cuda,
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

        # Do not transpose weights here; nn.Linear expects weights as [out_features, in_features]
        weights_fourier = DFTLRP._array_to_tensor(weights_fourier, precision, cuda)

        n_in, n_out = weights_fourier.shape  # Should be [signal_length, freq_length] for forward, [freq_length, signal_length] for inverse
        if transpose:
            weights_fourier = weights_fourier.T
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

        signal = self._array_to_tensor(signal, self.precision, self.cuda)

        # Handle 3D input [batch_size, channels, signal_length]
        if len(signal.shape) == 3:
            batch_size, channels, signal_len = signal.shape
            # Reshape to [batch_size * channels, signal_length] for linear layer
            signal = signal.view(batch_size * channels, signal_len)
            with torch.no_grad():
                signal_hat = transform(signal)
            # Reshape back to [batch_size, channels, freq_length]
            signal_hat = signal_hat.view(batch_size, channels, -1).cpu().numpy()
        else:
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

        signal = self._array_to_tensor(signal, self.precision, self.cuda)
        relevance = self._array_to_tensor(relevance, self.precision, self.cuda)

        # Handle 3D input [batch_size, channels, signal_length]
        if len(signal.shape) == 3:
            batch_size, channels, signal_len = signal.shape
            # Reshape to [batch_size * channels, signal_length] for linear layer
            signal = signal.view(batch_size * channels, signal_len)
            relevance = relevance.view(batch_size * channels, signal_len)
        else:
            batch_size, signal_len = signal.shape
            channels = 1

        if signal_hat is None:
            with torch.no_grad():
                signal_hat = transform(signal)
        else:
            signal_hat = self._array_to_tensor(signal_hat, self.precision, self.cuda)

        norm = signal + epsilon
        relevance_normed = relevance / norm

        with torch.no_grad():
            relevance_hat = dft_transform(relevance_normed)
            relevance_hat = signal_hat * relevance_hat

        # Reshape back to [batch_size, channels, freq_length] if 3D
        if len(signal.shape) == 2:  # After view
            relevance_hat = relevance_hat.view(batch_size, channels, -1)
            signal_hat = signal_hat.view(batch_size, channels, -1)

        relevance_hat = relevance_hat.cpu().numpy()
        signal_hat = signal_hat.cpu().numpy()

        if not real:
            signal_hat = self.reshape_signal(signal_hat, self.signal_length, relevance=False, short_time=short_time,
                                             symmetry=self.symmetry)
            relevance_hat = self.reshape_signal(relevance_hat, self.signal_length, relevance=True,
                                                short_time=short_time, symmetry=self.symmetry)
        print(f"signal shape after transform: {signal_hat.shape}")
        print(f"relevance_hat shape after transform: {relevance_hat.shape}")

        return signal_hat, relevance_hat