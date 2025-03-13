import numpy as np
import torch

def create_fft_weights(signal_length, inverse=False, symmetry=False, real=False):
    """
    Placeholder for FFT weights. FFT is computed directly with torch.fft.rfft.
    """
    freq_length = signal_length // 2 + 1 if symmetry else signal_length
    return np.eye(freq_length, signal_length)

def rectangle_window(m, width, signal_length, shift):
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = 1 / np.sqrt(shift)
    return w_nm

def halfsine_window(m, width, signal_length, shift=None):
    w_nm = np.zeros(signal_length)
    w_nm[m:m+width] = np.sin(np.pi / width * (np.arange(width) + 0.5))
    return w_nm

WINDOWS = {"rectangle": rectangle_window, "halfsine": halfsine_window}

def create_window_mask(shift, width, signal_length, window_function):
    ms = np.arange(0, signal_length - width + 1, shift)
    W_mn = [window_function(m, width, signal_length, shift)[np.newaxis] for m in ms]
    W_mn = np.concatenate(W_mn, axis=0)
    return W_mn.transpose((1, 0))

def create_short_time_fft_weights(signal_length, shift, window_width, window_shape, inverse=False, real=False, symmetry=False):
    """
    Placeholder for STFT weights. STFT is computed directly with torch.stft.
    """
    # We don't need to precompute weights; return a dummy matrix
    freq_length = signal_length // 2 + 1 if symmetry else signal_length
    n_windows = (signal_length - window_width) // shift + 1
    return np.zeros((signal_length, n_windows * freq_length))

def get_window(window_length, window_type):
    if window_type not in WINDOWS:
        raise ValueError("Available window types: rectangle, halfsine")
    window_func = WINDOWS[window_type]
    window = window_func(0, window_length, window_length, 1)
    return torch.tensor(window, dtype=torch.float32)