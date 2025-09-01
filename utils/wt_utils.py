import torch
import torch.nn as nn
import numpy as np
import pywt
import gc


class WaveletLayer(nn.Module):
    """PyTorch layer for discrete wavelet transform"""

    def __init__(self, signal_length, wavelet='db4', level=None, mode='symmetric', cuda=True):
        """
        Initialize wavelet transform layer.

        Args:
            signal_length: Length of input signal
            wavelet: Wavelet type (e.g., 'db4', 'sym4', 'haar')
            level: Decomposition level (None for maximum level)
            mode: Signal extension mode
            cuda: Whether to use CUDA for computation
        """
        super(WaveletLayer, self).__init__()
        self.signal_length = signal_length
        self.wavelet_name = wavelet
        self.mode = mode
        self.cuda = cuda and torch.cuda.is_available()

        # Calculate maximum decomposition level if not specified
        if level is None:
            self.level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet).dec_len)
        else:
            self.level = level

        print(f"Wavelet layer initialized with {wavelet} wavelet, level {self.level}")

        # Create wavelet filter coefficients
        wavelet_obj = pywt.Wavelet(wavelet)
        self.dec_lo = nn.Parameter(torch.FloatTensor(wavelet_obj.dec_lo), requires_grad=False)
        self.dec_hi = nn.Parameter(torch.FloatTensor(wavelet_obj.dec_hi), requires_grad=False)

        # Store sizes of coefficients at each level for reconstruction
        self._coeff_sizes = []

    def forward(self, x):
        """
        Forward pass applies wavelet transform.

        Args:
            x: Input tensor of shape (batch_size, channels, signal_length)

        Returns:
            list of tensors for each decomposition level, with structure:
            [approximation_coeff, [detail_level1, detail_level2, ..., detail_levelN]]
        """
        batch_size, channels, signal_length = x.shape
        device = x.device

        # Move to CPU for PyWavelets processing
        x_np = x.detach().cpu().numpy()

        # Initialize output lists
        approx_coeffs = []
        detail_coeffs = [[] for _ in range(self.level)]

        # Process each sample in batch
        for b in range(batch_size):
            # Process each channel
            sample_approx = []
            sample_details = [[] for _ in range(self.level)]

            for c in range(channels):
                # Apply wavelet transform
                coeffs = pywt.wavedec(x_np[b, c], self.wavelet_name, mode=self.mode, level=self.level)

                # Store coefficient sizes for reconstruction
                if b == 0 and c == 0:
                    self._coeff_sizes = [coeff.shape[0] for coeff in coeffs]

                # Split into approximation and detail coefficients
                approx = coeffs[0]
                details = coeffs[1:]

                sample_approx.append(approx)
                for level in range(self.level):
                    sample_details[level].append(details[level])

            # Stack channels
            approx_coeffs.append(np.stack(sample_approx))
            for level in range(self.level):
                detail_coeffs[level].append(np.stack(sample_details[level]))

        # Stack batches
        approx_coeffs = np.stack(approx_coeffs)
        detail_coeffs = [np.stack(level_coeffs) for level_coeffs in detail_coeffs]

        # Convert to tensors and move to original device
        approx_tensor = torch.tensor(approx_coeffs, dtype=torch.float32, device=device)
        detail_tensors = [torch.tensor(detail_coeff, dtype=torch.float32, device=device)
                          for detail_coeff in detail_coeffs]

        return [approx_tensor, detail_tensors]

    def get_coeff_sizes(self):
        """Return coefficient sizes for reconstruction"""
        return self._coeff_sizes


class InverseWaveletLayer(nn.Module):
    """PyTorch layer for inverse discrete wavelet transform"""

    def __init__(self, signal_length, wavelet='db4', level=None, mode='symmetric', cuda=True):
        """
        Initialize inverse wavelet transform layer.

        Args:
            signal_length: Length of input signal
            wavelet: Wavelet type (e.g., 'db4', 'sym4', 'haar')
            level: Decomposition level (None for maximum level)
            mode: Signal extension mode
            cuda: Whether to use CUDA for computation
        """
        super(InverseWaveletLayer, self).__init__()
        self.signal_length = signal_length
        self.wavelet_name = wavelet
        self.mode = mode
        self.cuda = cuda and torch.cuda.is_available()

        # Calculate maximum decomposition level if not specified
        if level is None:
            self.level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet).dec_len)
        else:
            self.level = level

        print(f"Inverse wavelet layer initialized with {wavelet} wavelet, level {self.level}")

        # Create wavelet filter coefficients
        wavelet_obj = pywt.Wavelet(wavelet)
        self.rec_lo = nn.Parameter(torch.FloatTensor(wavelet_obj.rec_lo), requires_grad=False)
        self.rec_hi = nn.Parameter(torch.FloatTensor(wavelet_obj.rec_hi), requires_grad=False)

    def forward(self, coeffs_list, coeff_sizes=None):
        """
        Forward pass applies inverse wavelet transform.

        Args:
            coeffs_list: List containing [approximation_coeff, [detail_level1, detail_level2, ..., detail_levelN]]
            coeff_sizes: List of coefficient sizes from forward transform

        Returns:
            Reconstructed signal tensor of shape (batch_size, channels, signal_length)
        """
        approx_coeff = coeffs_list[0]
        detail_coeffs = coeffs_list[1]

        # Move to CPU for PyWavelets processing
        approx_np = approx_coeff.detach().cpu().numpy()
        detail_np = [detail.detach().cpu().numpy() for detail in detail_coeffs]

        batch_size, channels, _ = approx_np.shape
        device = approx_coeff.device

        # Initialize output tensor
        output = np.zeros((batch_size, channels, self.signal_length))

        # Process each sample in batch
        for b in range(batch_size):
            # Process each channel
            for c in range(channels):
                # Combine coefficients
                coeffs = [approx_np[b, c]]
                for level in range(len(detail_np)):
                    coeffs.append(detail_np[level][b, c])

                # Apply inverse wavelet transform
                reconstructed = pywt.waverec(coeffs, self.wavelet_name, mode=self.mode)

                # Handle potential size difference
                if reconstructed.shape[0] > self.signal_length:
                    reconstructed = reconstructed[:self.signal_length]
                elif reconstructed.shape[0] < self.signal_length:
                    pad_width = self.signal_length - reconstructed.shape[0]
                    reconstructed = np.pad(reconstructed, (0, pad_width), mode='constant')

                output[b, c] = reconstructed

        # Convert to tensor and move to original device
        output_tensor = torch.tensor(output, dtype=torch.float32, device=device)

        return output_tensor


class WaveletLRP:
    def __init__(self, signal_length, wavelet='db4', level=None, mode='symmetric',
                 precision=32, cuda=True, create_inverse=True, create_transpose_inverse=True):
        """
        Class for Wavelet transform in PyTorch and relevance propagation through Wavelet layer.

        Args:
            signal_length: Length of input signal
            wavelet: Wavelet type (e.g., 'db4', 'sym4', 'haar')
            level: Decomposition level (None for maximum level)
            mode: Signal extension mode
            precision: 32 or 16 for reduced precision
            cuda: Whether to use CUDA
            create_inverse: Whether to create inverse transform layer
            create_transpose_inverse: Whether to create transpose inverse layer for LRP
        """
        self.signal_length = signal_length
        self.wavelet_name = wavelet
        self.mode = mode
        self.precision = precision
        self.cuda = cuda and torch.cuda.is_available()

        # Calculate level if not specified
        if level is None:
            self.level = pywt.dwt_max_level(signal_length, pywt.Wavelet(wavelet).dec_len)
        else:
            self.level = level

        print(f"WaveletLRP initialized with {wavelet} wavelet, level {self.level}, cuda={self.cuda}")

        # Create wavelet layers
        self.wavelet_layer = WaveletLayer(
            signal_length=signal_length,
            wavelet=wavelet,
            level=self.level,
            mode=mode,
            cuda=cuda
        )

        if create_inverse:
            self.inverse_wavelet_layer = InverseWaveletLayer(
                signal_length=signal_length,
                wavelet=wavelet,
                level=self.level,
                mode=mode,
                cuda=cuda
            )

        if create_transpose_inverse:
            # For LRP, we need a transposed version of the wavelet transform
            # This is approximated by using the inverse wavelet transform
            self.transpose_inverse_wavelet_layer = InverseWaveletLayer(
                signal_length=signal_length,
                wavelet=wavelet,
                level=self.level,
                mode=mode,
                cuda=cuda
            )

        # Move to GPU if available
        if self.cuda:
            self.wavelet_layer = self.wavelet_layer.cuda()
            if create_inverse:
                self.inverse_wavelet_layer = self.inverse_wavelet_layer.cuda()
            if create_transpose_inverse:
                self.transpose_inverse_wavelet_layer = self.transpose_inverse_wavelet_layer.cuda()

    # ... [other methods remain the same] ...

    def wavelet_lrp(self, relevance, signal, signal_wavelet=None, epsilon=1e-6):
        """
        Relevance propagation through Wavelet transform.

        Args:
            relevance: Relevance in time domain
            signal: Signal in time domain
            signal_wavelet: Signal in wavelet domain (if already computed)
            epsilon: Small constant for numerical stability

        Returns:
            tuple: (signal_wavelet, relevance_wavelet)
        """
        # Convert inputs to tensors
        relevance_tensor = self._array_to_tensor(relevance, self.precision, self.cuda)
        signal_tensor = self._array_to_tensor(signal, self.precision, self.cuda)

        # Compute wavelet transform of signal if not provided
        if signal_wavelet is None:
            with torch.no_grad():
                signal_wavelet = self.wavelet_layer(signal_tensor)

        # Handle division by small values safely
        with torch.no_grad():
            # Apply epsilon with correct sign to avoid division by zero
            norm = signal_tensor.clone()
            abs_norm = torch.abs(norm)

            # Apply epsilon where values are smaller than epsilon
            epsilon_mask = abs_norm < epsilon
            norm[epsilon_mask] = torch.sign(norm[epsilon_mask]) * epsilon

            # Handle zeros in the sign function (avoid NaN results0)
            zero_mask = (norm == 0)  # This is a tensor of booleans
            if torch.any(zero_mask):  # Use torch.any instead of calling zero_mask as a function
                norm[zero_mask] = epsilon

            # Compute normalized relevance
            relevance_normed = relevance_tensor / norm

            # Process through wavelet transform
            relevance_wavelet_coeffs = self.wavelet_layer(relevance_normed)

            # Apply signal weights to relevance in wavelet domain
            # For approximation coefficients
            relevance_wavelet_coeffs[0] = relevance_wavelet_coeffs[0] * signal_wavelet[0]

            # For detail coefficients at each level
            for i in range(len(relevance_wavelet_coeffs[1])):
                relevance_wavelet_coeffs[1][i] = relevance_wavelet_coeffs[1][i] * signal_wavelet[1][i]

        return signal_wavelet, relevance_wavelet_coeffs