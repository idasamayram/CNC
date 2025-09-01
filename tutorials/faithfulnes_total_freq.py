import argparse
import torch
import os
from datetime import datetime
from pathlib import Path
from utils.baseline_xai import load_model
from utils.xai_implementation import compute_basic_dft_lrp, compute_dft_gradient_input, compute_dft_smoothgrad, \
    compute_dft_occlusion
import pandas as pd
import numpy as np
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd





def frequency_window_flipping_with_dftlrp(model, sample, attribution_method, target_class=None, n_steps=20,
                                          window_size=10, most_relevant_first=True, reference_value=None,
                                          device="cuda" if torch.cuda.is_available() else "cpu",
                                          leverage_symmetry=True, sampling_rate=400):
    """
    Perform window flipping analysis in the frequency domain using EnhancedDFTLRP.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions (like compute_basic_dft_lrp)
        target_class: Target class to track (if None, use predicted class)
        n_steps: Number of steps for window flipping
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Type of reference value to use for flipping
        device: Device to run on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz
    """
    import numpy as np
    import torch
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # 1. Get frequency domain attributions using the provided attribution method
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Check if we got valid results0
    if relevance_freq is None or signal_freq is None:
        print("Attribution method failed to provide frequency domain values")
        return [original_score], [0.0]

    # 2. Create DFT-LRP object for transformations
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_inverse=True  # Need inverse transform
    )

    # 3. Determine frequency domain dimensions
    freq_length = signal_freq.shape[1]  # This will be time_steps//2 + 1 with leverage_symmetry=True

    # 4. Prepare reference value for frequency domain
    if reference_value == "zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "noise":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5

            # When using symmetry, keep DC and Nyquist components real
            if leverage_symmetry:
                # Complex noise for all components
                phase = np.random.uniform(-np.pi, np.pi, freq_length)
                # Ensure DC and Nyquist have zero phase (real values)
                if freq_length > 0:
                    phase[0] = 0  # DC component
                if freq_length > 1:
                    phase[-1] = 0  # Nyquist component

                reference_value_freq[c] = magnitude * np.exp(1j * phase)
            else:
                real_part = np.random.normal(0, magnitude, freq_length)
                imag_part = np.random.normal(0, magnitude, freq_length)
                reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "invert":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            # Invert magnitude and phase, but preserve constraints
            magnitude = np.abs(signal_freq[c])
            phase = np.angle(signal_freq[c]) + np.pi  # Invert phase

            if leverage_symmetry:
                # Ensure DC and Nyquist remain real
                if freq_length > 0:
                    phase[0] = 0 if signal_freq[c, 0].real < 0 else np.pi  # Keep DC real but invert
                if freq_length > 1:
                    phase[-1] = 0 if signal_freq[c, -1].real < 0 else np.pi  # Keep Nyquist real but invert

            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            # Keep phase but set magnitude to near-zero
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

            # Special case for DC and Nyquist if leveraging symmetry
            if leverage_symmetry:
                if freq_length > 0:
                    # DC component must be real
                    reference_value_freq[c, 0] = 1e-10 * np.sign(signal_freq[c, 0].real)
                if freq_length > 1:
                    # Nyquist component must be real
                    reference_value_freq[c, -1] = 1e-10 * np.sign(signal_freq[c, -1].real)

    elif reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)

    else:  # Default to magnitude_zero if not specified
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    # 5. Calculate window importance using absolute relevance values
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values of relevance for importance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

    # 6. Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # 7. Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # 8. Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # 9. Iteratively flip windows
    for step in range(1, n_steps + 1):
        n_windows_to_flip = min(step * windows_per_step, total_windows)
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Replace frequency components with reference
            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # 10. Transform back to time domain using EnhancedDFTLRP's inverse transform
        try:
            # Convert to tensor for DFT-LRP
            flipped_time = np.zeros((n_channels, time_steps))

            if hasattr(dftlrp, 'inverse_fourier_layer') and dftlrp.inverse_fourier_layer is not None:
                # Use DFT-LRP's inverse transform
                for c in range(n_channels):
                    # Convert to tensor for EnhancedDFTLRP
                    if leverage_symmetry:
                        # Need to reshape the frequency data to match the expected format
                        freq_tensor = torch.tensor(np.concatenate([
                            flipped_freq[c].real,  # Real part first
                            flipped_freq[c, 1:-1].imag  # Then imaginary part (excluding DC and Nyquist)
                        ]), dtype=torch.float32)

                        # Add batch dimension
                        freq_tensor = freq_tensor.unsqueeze(0)

                        if dftlrp.cuda:
                            freq_tensor = freq_tensor.cuda()

                        # Apply inverse transform
                        with torch.no_grad():
                            time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                        # Get result
                        flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                    else:
                        # Non-symmetry case requires different handling
                        # Split into real and imaginary parts
                        freq_real = flipped_freq[c].real
                        freq_imag = flipped_freq[c].imag

                        # Concatenate as expected by DFT-LRP
                        freq_tensor = torch.tensor(np.concatenate([freq_real, freq_imag]), dtype=torch.float32)
                        freq_tensor = freq_tensor.unsqueeze(0)

                        if dftlrp.cuda:
                            freq_tensor = freq_tensor.cuda()

                        # Apply inverse transform
                        with torch.no_grad():
                            time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                        # Get result
                        flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
            else:
                # Fallback to numpy IFFT if DFT-LRP inverse not available
                print("DFT-LRP inverse transform not available, falling back to numpy IFFT")
                for c in range(n_channels):
                    if leverage_symmetry:
                        flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                    else:
                        flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        except Exception as e:
            print(f"Error in inverse transform: {e}")
            print("Falling back to numpy IFFT")

            # Fallback to numpy IFFT
            for c in range(n_channels):
                if leverage_symmetry:
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # 11. Convert back to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # Normalize the flipped signal to match the original signal energy
        for c in range(n_channels):
            # Get statistics of original signal
            orig_std = input_signal[c].std()
            orig_mean = input_signal[c].mean()

            # Normalize the flipped signal to match original energy
            if flipped_time[c].std() > 0:  # Avoid division by zero
                flipped_time[c] = ((flipped_time[c] - flipped_time[c].mean()) /
                                   flipped_time[c].std() * orig_std + orig_mean)
            else:
                print(f"Warning: Channel {c} has zero standard deviation after flipping")


        # 12. Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts

def frequency_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                     window_size=10, most_relevant_first=True, reference_value=None,
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     leverage_symmetry=True, sampling_rate=400):
    """
    Perform window flipping analysis in the frequency domain on a single time series sample.
    """
    import numpy as np
    import torch
    import gc

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        # Track softmax probability for the target class
        original_score = original_prob[target_class].item()

    try:
        # First attempt - try to get the frequency domain attributions
        relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
            model=model,
            sample=sample,
            label=target_class,
            device=device,
            signal_length=time_steps,
            leverage_symmetry=leverage_symmetry,
            sampling_rate=sampling_rate
        )

        # Check if dimensions make sense
        if relevance_freq.shape[1] == 0:
            print(f"Warning: Empty frequency relevance with shape {relevance_freq.shape}")
            return [original_score], [0.0]

    except Exception as e:
        print(f"Error computing frequency domain attributions: {e}")
        print("Returning default scores without flipping")
        return [original_score], [0.0]

    # Get the number of frequency bins
    freq_length = relevance_freq.shape[1]  # This will be time_steps//2 + 1 if leverage_symmetry=True
    print(f"Working with frequency domain shape: {relevance_freq.shape}")

    # Prepare reference value in frequency domain
    if reference_value is None or reference_value == "extreme":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).mean() * 10
            phase = np.random.uniform(-np.pi, np.pi, freq_length)
            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "zero":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)

    elif reference_value == "noise":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "invert":
        reference_value_freq = -signal_freq.copy()

    elif reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)


    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    elif reference_value == "phase_zero":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            reference_value_freq[c] = np.abs(signal_freq[c])

    elif reference_value == "random_complex":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).mean() * 3
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    # Calculate window importance - use absolute relevance values
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values for relevance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        n_windows_to_flip = min(step * windows_per_step, total_windows)
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Replace frequency components with reference
            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # Transform back to time domain using numpy's IFFT for simplicity
        flipped_time = np.zeros((n_channels, time_steps))

        try:
            for c in range(n_channels):
                if leverage_symmetry:
                    # For real signals with leverage_symmetry=True
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    # For complex signals or when not leveraging symmetry
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real
        except Exception as e:
            print(f"Error in inverse transform: {e}")
            # If inverse transform fails, skip this step
            scores.append(scores[-1])  # Use previous score
            flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
            continue

        # Convert to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # Get model prediction
        with torch.no_grad():
            try:
                output = model(flipped_sample.unsqueeze(0))
                prob = torch.softmax(output, dim=1)[0]
                score = prob[target_class].item()
            except Exception as e:
                print(f"Error in model prediction: {e}")
                score = scores[-1]  # Use previous score

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts
'''

def frequency_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                     window_size=10, most_relevant_first=True,
                                     reference_value=None,
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     leverage_symmetry=True,
                                     sampling_rate=400):
    """
    Perform window flipping analysis in the frequency domain on a single time series sample.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions in frequency domain (DFT-XAI functions)
        target_class: Target class to track (if None, use predicted class)
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    import numpy as np
    import torch
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        # If target_class is not provided, use the predicted class
        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        # Track softmax probability for the target class
        original_score = original_prob[target_class].item()

    # 1. Use the provided attribution method to get frequency domain attributions
    # This will return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target
    _, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Determine frequency length based on symmetry
    freq_length = relevance_freq.shape[1]

    # Create a DFT-LRP object for inverse transforms
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=True  # Need inverse transform
    )

    # 2. Prepare reference value for frequency domain
    if reference_value is None or reference_value == "extreme":
        # Use extreme values - large magnitude with random phase
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).mean() * 10  # 10x average magnitude
            phase = np.random.uniform(-np.pi, np.pi, freq_length)
            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "mild_noise":
        # Use mild noise - random complex values with 1x std magnitude
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std()  # 1x standard deviation
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "noise":
        # Use high variance noise - random complex values with 5x std magnitude
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5  # 5x standard deviation
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "shift":
        # Shift phase by pi (inverse phase)
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c])
            phase = np.angle(signal_freq[c]) + np.pi  # Shift phase by pi
            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "zero":
        # Set to zero in frequency domain (flat line in time domain)
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)

    elif reference_value == "invert":
        # Invert the frequency components (negate real and imaginary parts)
        reference_value_freq = -signal_freq.copy()

    elif reference_value == "magnitude_zero":
        # Keep phase but set magnitude to zero
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            # Use very small magnitude (not exactly zero to avoid numerical issues)
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    elif reference_value == "phase_zero":
        # Keep magnitude but set phase to zero
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c])
            reference_value_freq[c] = magnitude  # Phase = 0 is just the real part = magnitude

    elif reference_value == "random_complex":
        # Completely random complex values with high magnitude
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).mean() * 3  # 3x mean magnitude
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif isinstance(reference_value, (int, float)):
        # Constant value in time domain = impulse in frequency domain
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        if freq_length > 0:  # Set DC component to create constant value
            reference_value_freq[:, 0] = reference_value * time_steps

    # 3. Calculate frequency window importance by averaging relevance within each window
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    # Calculate window importance
    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute relevance within the window
            window_importance[channel, window_idx] = np.mean(relevance_freq[channel, start_idx:end_idx])

    # 4. Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # 5. Track model outputs
    scores = [original_score]  # Start with original softmax probability
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # 6. Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # 7. Transform back to time domain
        try:
            # Use DFT-LRP's inverse transform if available
            flipped_time = np.zeros((n_channels, time_steps))

            for c in range(n_channels):
                if hasattr(dftlrp, 'inverse_transform'):
                    flipped_time[c] = dftlrp.inverse_transform(flipped_freq[c])
                else:
                    # Fallback to numpy's IFFT
                    if leverage_symmetry:
                        flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                    else:
                        flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        except Exception as e:
            print(f"Error in inverse transform: {e}. Falling back to numpy IFFT.")

            # Fallback using numpy IFFT
            flipped_time = np.zeros((n_channels, time_steps))

            for c in range(n_channels):
                if leverage_symmetry:
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # 8. Convert back to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # 9. Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            # Calculate softmax probability for the target class
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts


'''

'''def frequency_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                     window_size=10, most_relevant_first=True,
                                     reference_value=None,
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     leverage_symmetry=True,
                                     sampling_rate=400):
    """
    Perform window flipping analysis in the frequency domain on a single time series sample.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions in frequency domain (DFT-XAI functions)
        target_class: Target class to track (if None, use predicted class)
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows. Options:
                         - None/extreme: Use extreme values (default)
                         - noise: Use high-variance random noise
                         - mild_noise: Use noise with 1x std magnitude
                         - shift: Use shifted values in opposite direction from mean
                         - zero: Set frequency components to zero
                         - invert: Invert phase of the frequency components
                         - magnitude_zero: Keep phase but set magnitude to zero
                         - phase_zero: Keep magnitude but set phase to zero
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz

    Returns:
        scores: List of model outputs at each flipping step
        flipped_pcts: List of percentages of flipped windows at each step
    """
    import numpy as np
    import torch
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        # If target_class is not provided, use the predicted class
        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # 1. Use the provided attribution method to get frequency domain attributions
    # This will return relevance_time, relevance_freq, signal_freq, input_signal, freqs, target
    _, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Determine frequency length based on symmetry
    freq_length = relevance_freq.shape[1]

    # Create a DFT-LRP object for inverse transforms
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_stdft=False,
        create_inverse=True  # Need inverse transform
    )

    # 2. Prepare reference value for frequency domain
    if reference_value is None or reference_value == "extreme":
        # Use extreme values - large magnitude with random phase
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).mean() * 10  # 10x average magnitude
            phase = np.random.uniform(-np.pi, np.pi, freq_length)
            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "mild_noise":
        # Use mild noise - random complex values with 1x std magnitude
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std()  # 1x standard deviation
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "noise":
        # Use high variance noise - random complex values with 5x std magnitude
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5  # 5x standard deviation
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif reference_value == "shift":
        # Shift phase by pi (inverse phase)
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c])
            phase = np.angle(signal_freq[c]) + np.pi  # Shift phase by pi
            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    elif reference_value == "zero":
        # Set to zero in frequency domain (flat line in time domain)
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)

    elif reference_value == "invert":
        # Invert the frequency components (negate real and imaginary parts)
        reference_value_freq = -signal_freq.copy()

    elif reference_value == "magnitude_zero":
        # Keep phase but set magnitude to zero
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            # Use very small magnitude (not exactly zero to avoid numerical issues)
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    elif reference_value == "phase_zero":
        # Keep magnitude but set phase to zero
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c])
            reference_value_freq[c] = magnitude  # Phase = 0 is just the real part = magnitude

    elif reference_value == "random_complex":
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        for c in range(n_channels):
            # Completely random complex values
            magnitude = np.abs(signal_freq[c]).mean()  # Use mean magnitude as reference
            real_part = np.random.normal(0, magnitude, freq_length)
            imag_part = np.random.normal(0, magnitude, freq_length)
            reference_value_freq[c] = real_part + 1j * imag_part

    elif isinstance(reference_value, (int, float)):
        # Constant value in time domain = impulse in frequency domain
        reference_value_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        if freq_length > 0:  # Set DC component to create constant value
            reference_value_freq[:, 0] = reference_value * time_steps

    # 3. Calculate frequency window importance by averaging relevance within each window
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    # Calculate window importance
    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute relevance within the window
            window_importance[channel, window_idx] = np.mean(relevance_freq[channel, start_idx:end_idx])

    # 4. Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # 5. Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # 6. Iteratively flip windows
    for step in range(1, n_steps + 1):
        # Calculate how many windows to flip at this step
        n_windows_to_flip = min(step * windows_per_step, total_windows)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # 7. Transform back to time domain
        try:
            # Use DFT-LRP's inverse transform if available
            flipped_time = np.zeros((n_channels, time_steps))

            for c in range(n_channels):
                if hasattr(dftlrp, 'inverse_transform'):
                    flipped_time[c] = dftlrp.inverse_transform(flipped_freq[c])
                else:
                    # Fallback to numpy's IFFT
                    if leverage_symmetry:
                        flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                    else:
                        flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        except Exception as e:
            print(f"Error in inverse transform: {e}. Falling back to numpy IFFT.")

            # Fallback using numpy IFFT
            flipped_time = np.zeros((n_channels, time_steps))

            for c in range(n_channels):
                if leverage_symmetry:
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # 8. Convert back to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # 9. Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]

            # Track probability for the target class
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts
'''


def improved_frequency_window_flipping(model, sample, attribution_method, target_class=None, n_steps=20,
                                       window_size=10, most_relevant_first=True, reference_value=None,
                                       device="cuda" if torch.cuda.is_available() else "cpu",
                                       leverage_symmetry=True, sampling_rate=400):
    """
    Improved window flipping that properly handles symmetry and preserves energy.
    """
    import numpy as np
    import torch
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Get original prediction to determine target class if not provided
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Determine frequency domain dimensions
    freq_length = signal_freq.shape[1]  # time_steps//2 + 1 if leverage_symmetry=True

    # Create DFT-LRP object for transformations (needed for exact symmetry handling)
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_inverse=True  # Need inverse transform
    )

    # Prepare reference value for frequency domain
    if reference_value == "zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

            # Special case for DC and Nyquist if leveraging symmetry
            if leverage_symmetry:
                if freq_length > 0:
                    reference_value_freq[c, 0] = 1e-10 * np.sign(signal_freq[c, 0].real)
                if freq_length > 1:
                    reference_value_freq[c, -1] = 1e-10 * np.sign(signal_freq[c, -1].real)

    elif reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "invert":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            # Invert magnitude and phase
            magnitude = np.abs(signal_freq[c])
            phase = np.angle(signal_freq[c]) + np.pi  # Invert phase

            reference_value_freq[c] = magnitude * np.exp(1j * phase)

            # Special case for DC and Nyquist
            if leverage_symmetry:
                if freq_length > 0:
                    reference_value_freq[c, 0] = -signal_freq[c, 0].real  # Keep DC real
                if freq_length > 1:
                    reference_value_freq[c, -1] = -signal_freq[c, -1].real  # Keep Nyquist real

    elif reference_value == "noise":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5
            phase = np.random.uniform(-np.pi, np.pi, freq_length)

            if leverage_symmetry:
                # Ensure DC and Nyquist have zero phase (real values)
                if freq_length > 0:
                    phase[0] = 0
                if freq_length > 1:
                    phase[-1] = 0

            reference_value_freq[c] = magnitude * np.exp(1j * phase)

    else:  # Default to magnitude_zero
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    # Calculate window importance using absolute relevance values
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values of relevance for importance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Iteratively flip windows
    for step in range(1, n_steps + 1):
        n_windows_to_flip = min(step * windows_per_step, total_windows)
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Replace frequency components with reference
            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # Transform back to time domain
        flipped_time = np.zeros((n_channels, time_steps))

        try:
            # First try with numpy's IFFT for simpler code
            for c in range(n_channels):
                if leverage_symmetry:
                    # When leveraging symmetry, use irfft which expects half-spectrum
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    # Without symmetry, use standard ifft
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        except Exception as e:
            print(f"Error in numpy IFFT: {e}")
            print("Trying DFT-LRP inverse transform...")

            # Fallback to DFT-LRP's inverse transform which properly handles symmetry
            try:
                if hasattr(dftlrp, 'inverse_fourier_layer') and dftlrp.inverse_fourier_layer is not None:
                    for c in range(n_channels):
                        # Format data for DFT-LRP based on symmetry
                        if leverage_symmetry:
                            # When using symmetry, we need to separate real and imaginary parts correctly
                            # DC and Nyquist components should be real
                            # First prepare the real part (all components)
                            freq_real = flipped_freq[c].real

                            # Then prepare imaginary part (excluding DC and Nyquist)
                            if freq_length > 2:  # Only if we have components between DC and Nyquist
                                freq_imag = flipped_freq[c, 1:-1].imag
                            else:
                                freq_imag = np.array([])

                            # Concatenate according to DFT-LRP's expected format
                            freq_data = np.concatenate([freq_real, freq_imag])
                        else:
                            # Without symmetry, simply concatenate real and imaginary parts
                            freq_data = np.concatenate([flipped_freq[c].real, flipped_freq[c].imag])

                        # Convert to tensor
                        freq_tensor = torch.tensor(freq_data, dtype=torch.float32).unsqueeze(0)

                        if dftlrp.cuda:
                            freq_tensor = freq_tensor.cuda()

                        # Apply inverse transform
                        with torch.no_grad():
                            time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                        # Get result
                        flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                else:
                    raise ValueError("DFT-LRP inverse transform not available")
            except Exception as e:
                print(f"Error in DFT-LRP inverse: {e}")
                print("Using fallback method...")

                # Ultimate fallback - perturb original signal
                for c in range(n_channels):
                    flipped_time[c] = input_signal[c] * 0.5 + np.random.normal(0, input_signal[c].std() * 0.1,
                                                                               time_steps)

        # Normalize flipped signal to match original energy
        for c in range(n_channels):
            # Get statistics of original signal
            orig_std = input_signal[c].std()
            orig_mean = input_signal[c].mean()

            # Normalize the flipped signal to match original energy
            if flipped_time[c].std() > 0:  # Avoid division by zero
                flipped_time[c] = ((flipped_time[c] - flipped_time[c].mean()) /
                                   flipped_time[c].std() * orig_std + orig_mean)
            else:
                print(f"Warning: Channel {c} has zero standard deviation after flipping")
                # Use noise with correct statistics as fallback
                flipped_time[c] = np.random.normal(orig_mean, orig_std, time_steps)

        # Convert to tensor for model inference
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        # Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results0
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts
def frequency_window_flipping_batch(model, test_loader, attribution_methods, n_steps=20, window_size=10,
                                    most_relevant_first=True, reference_value=None, max_samples=None,
                                    leverage_symmetry=True, sampling_rate=400,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run window flipping on a batch of samples with improved error handling.
    """
    from tqdm import tqdm
    import numpy as np
    import torch
    import gc

    # Initialize results0 dictionary
    results = {method_name: [] for method_name in attribution_methods.keys()}

    # Track number of processed samples
    n_processed = 0

    print("Starting frequency domain window flipping batch processing:")
    print(f"  - Window size: {window_size} frequency bins")
    print(f"  - Reference value: {reference_value}")
    print(f"  - Most relevant first: {most_relevant_first}")

    # Process batches
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
        # Process each sample in the batch
        for i in range(len(data)):
            if max_samples is not None and n_processed >= max_samples:
                print(f"Reached maximum number of samples ({max_samples})")
                break

            # Get single sample
            sample = data[i]
            target = targets[i]

            # Process with each attribution method
            for method_name, attribution_method in attribution_methods.items():
                try:
                    # Run window flipping for this sample
                    scores, flipped_pcts = frequency_window_flipping_single(
                        model=model,
                        sample=sample,
                        attribution_method=attribution_method,
                        target_class=target,
                        n_steps=n_steps,
                        window_size=window_size,
                        most_relevant_first=most_relevant_first,
                        reference_value=reference_value,
                        leverage_symmetry=leverage_symmetry,
                        sampling_rate=sampling_rate,
                        device=device
                    )

                    # Compute AUC if we have valid results0
                    if len(scores) > 1 and len(flipped_pcts) > 1:
                        auc = np.trapz(y=scores, x=flipped_pcts) / 100.0
                    else:
                        auc = float('nan')

                    # Store results0
                    results[method_name].append({
                        "sample_idx": n_processed,
                        "scores": scores,
                        "flipped_pcts": flipped_pcts,
                        "auc": auc,
                        "target": target.item() if isinstance(target, torch.Tensor) else target
                    })

                except Exception as e:
                    print(f"Error processing sample {n_processed} with method {method_name}: {str(e)}")
                    # Add an empty result to keep counts consistent
                    results[method_name].append({
                        "sample_idx": n_processed,
                        "scores": None,
                        "flipped_pcts": None,
                        "auc": float('nan'),
                        "target": target.item() if isinstance(target, torch.Tensor) else target,
                        "error": str(e)
                    })

            # Increment processed count
            n_processed += 1

            # Clean up memory after each sample
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Break if we've processed enough samples
        if max_samples is not None and n_processed >= max_samples:
            break

    return results

'''
def frequency_window_flipping_batch(model, test_loader, attribution_methods, n_steps=10,
                                    window_size=10, most_relevant_first=True,
                                    reference_value=None, max_samples=None,
                                    leverage_symmetry=True, sampling_rate=400,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform frequency domain window flipping analysis on a batch of time series samples and aggregate results0.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows (default=None, which uses extreme values)
        max_samples: Maximum number of samples to process (None = all samples)
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz
        device: Device to run computations on

    Returns:
        results0: Dictionary with aggregated results0 for each method
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    import gc

    # Initialize results0 storage
    results0 = {method_name: [] for method_name in attribution_methods}

    # Print configuration
    print(f"Starting frequency domain window flipping batch processing:")
    print(f"  - Window size: {window_size} frequency bins")
    print(f"  - Reference value: {reference_value}")
    print(f"  - Most relevant first: {most_relevant_first}")

    # Keep track of the current sample count
    sample_count = 0

    # Process each batch in the test loader
    for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
        # Process each sample in the batch
        for i in range(len(data)):
            sample = data[i].to(device)
            target = targets[i].to(device)

            # Increment sample counter
            sample_count += 1

            # Print progress every 10 samples
            if sample_count % 10 == 0:
                print(f"Processing sample {sample_count}")

            # Check if we've reached the maximum number of samples
            if max_samples is not None and sample_count > max_samples:
                break

            # Process each attribution method
            for method_name, attribution_func in attribution_methods.items():
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                try:
                    # Compute scores for this sample using the current method
                    scores, flipped_pcts = frequency_window_flipping_single(
                        model=model,
                        sample=sample,
                        attribution_method=attribution_func,
                        target_class=target.item(),
                        n_steps=n_steps,
                        window_size=window_size,
                        most_relevant_first=most_relevant_first,
                        reference_value=reference_value,
                        leverage_symmetry=leverage_symmetry,
                        sampling_rate=sampling_rate,
                        device=device
                    )

                    # Calculate AUC for this sample
                    auc = np.trapz(scores, flipped_pcts) / flipped_pcts[-1]

                    # Store results0
                    results0[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": scores,
                        "flipped_pcts": flipped_pcts,
                        "auc": auc
                    })

                except Exception as e:
                    print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                    # Store empty result to maintain sample count consistency
                    results0[method_name].append({
                        "sample_idx": sample_count - 1,
                        "scores": None,
                        "flipped_pcts": None,
                        "auc": float('nan')
                    })

        # Check if we've reached the maximum number of samples
        if max_samples is not None and sample_count >= max_samples:
            print(f"Reached maximum number of samples ({max_samples})")
            break

    return results0

'''
'''
def run_frequency_window_flipping_evaluation(model, test_loader, attribution_methods,
                                             n_steps=10, window_size=10,
                                             most_relevant_first=True, reference_value=None, max_samples=None,
                                             leverage_symmetry=True, sampling_rate=400,
                                             output_dir="./results0",
                                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency domain window flipping evaluation on test set and save results0.

    Args:
        model: Trained PyTorch model
        test_loader: DataLoader with test samples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz
        output_dir: Directory to save results0
        device: Device to run computations on

    Returns:
        results0: Dictionary with individual sample results0
        agg_results: Dictionary with aggregated results0
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"

    # Include reference value in filename
    ref_type = "zero"
    if reference_value == "noise":
        ref_type = "noise"
    elif reference_value == "mild_noise":
        ref_type = "mild_noise"
    elif reference_value == "extreme":
        ref_type = "extreme"
    elif reference_value == "shift":
        ref_type = "shift"
    elif reference_value == "invert":
        ref_type = "invert"
    elif reference_value == "magnitude_zero":
        ref_type = "magnitude_zero"
    elif reference_value == "phase_zero":
        ref_type = "phase_zero"

    filename_prefix = f"{output_dir}/freq_window_flipping_{flip_order}_{ref_type}_{timestamp}"

    print(f"Starting frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}, Leverage symmetry: {leverage_symmetry}")

    # Run window flipping on all samples
    results0 = frequency_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    # Compute aggregated results0
    print("Computing aggregated results0...")
    agg_results = aggregate_results(results0)  # Using your existing aggregate_results function

    print("Plotting results0...")
    # Plot and save aggregated results0
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value)  # Using your existing plot function
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')

    # Plot and save AUC distributions
    dist_fig = plot_auc_distribution(agg_results, most_relevant_first, reference_value)  # Using your existing plot function
    dist_fig.savefig(f"{filename_prefix}_auc_distribution.png", dpi=300, bbox_inches='tight')

    # Save numerical results0 to CSV
    save_results_to_csv(agg_results, results0, filename_prefix)  # Using your existing save function

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nFrequency Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results0, agg_results

'''


def run_frequency_window_flipping_evaluation(model, test_loader, attribution_methods,
                                             n_steps=10, window_size=10,
                                             most_relevant_first=True, reference_value=None, max_samples=None,
                                             leverage_symmetry=True, sampling_rate=400,
                                             output_dir="./results0",
                                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency domain window flipping evaluation on test set and save results0.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import datetime

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"

    # Include reference value in filename
    ref_type = "zero"
    if reference_value == "noise":
        ref_type = "noise"
    elif reference_value == "mild_noise":
        ref_type = "mild_noise"
    elif reference_value == "extreme":
        ref_type = "extreme"
    elif reference_value == "shift":
        ref_type = "shift"
    elif reference_value == "invert":
        ref_type = "invert"
    elif reference_value == "magnitude_zero":
        ref_type = "magnitude_zero"
    elif reference_value == "phase_zero":
        ref_type = "phase_zero"

    filename_prefix = f"{output_dir}/freq_window_flipping_{flip_order}_{ref_type}_{timestamp}"

    print(f"Starting frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}, Leverage symmetry: {leverage_symmetry}")

    # Run window flipping on all samples
    results = frequency_window_flipping_batch(
        model=model,
        test_loader=test_loader,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        reference_value=reference_value,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    # Compute aggregated results0
    print("Computing aggregated results0...")
    agg_results = aggregate_results(results)

    print("Plotting results0...")

    # Fix 1: Close any existing plots before creating new ones
    plt.close('all')

    # Fix 2: Create aggregate plot and immediately save it
    agg_fig = plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order_text = "most important" if most_relevant_first else "least important"
    plt.title(
        f'Aggregate Window Flipping Results\n(Flipping {flip_order_text} windows first, \n Reference Value Method Flipped with: {reference_value})',
        fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Fix 3: Save the figure directly
    plt.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    print(f"Saved aggregate plot to {filename_prefix}_aggregate_plot.png")

    # Fix 4: Show the plot and then close it to prevent blank plots
    if 'DISPLAY' in os.environ:  # Only show if running in a graphical environment
        plt.show()
    plt.close()

    # Fix 5: Create AUC distribution plot with the same approach
    plt.figure(figsize=(14, 6))

    # Count number of methods with valid data
    valid_methods = [m for m, r in agg_results.items() if len(r["auc_values"]) > 0]
    n_methods = len(valid_methods)

    if n_methods == 0:
        print("No valid methods to plot")
    else:
        # Create boxplots
        boxes = []
        labels = []

        for method_name, results in agg_results.items():
            if len(results["auc_values"]) == 0:
                continue

            boxes.append(results["auc_values"])
            labels.append(method_name)

        plt.boxplot(boxes, tick_labels=labels)  # Updated parameter name

        flip_order_text = "most important" if most_relevant_first else "least important"
        plt.title(
            f'AUC Distribution Across Samples\n(Flipping {flip_order_text} windows first), \n Reference Value Method Flipped with: {reference_value})',
            fontsize=16)
        plt.ylabel('AUC Value', fontsize=14)
        plt.grid(True, linestyle='--', axis='y', alpha=0.7)

        # Add text with mean and std
        for i, method_name in enumerate(labels):
            mean_auc = agg_results[method_name]["mean_auc"]
            std_auc = agg_results[method_name]["std_auc"]
            plt.text(i + 1, plt.ylim()[0] + 0.05, f"μ={mean_auc:.4f}\nσ={std_auc:.4f}",
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # Fix 6: Save immediately and then close
        plt.savefig(f"{filename_prefix}_auc_distribution.png", dpi=300, bbox_inches='tight')
        print(f"Saved AUC distribution plot to {filename_prefix}_auc_distribution.png")

        if 'DISPLAY' in os.environ:
            plt.show()
        plt.close()

    # Save numerical results0 to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nFrequency Domain Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


def aggregate_results(results):
    """
    Aggregate window flipping results0 across samples.
    More robust version that handles errors.
    """
    import numpy as np

    # Initialize aggregate results0
    agg_results = {}

    for method_name, samples in results.items():
        # Filter out samples with errors
        valid_samples = [s for s in samples if s["scores"] is not None and len(s["scores"]) > 1]

        if len(valid_samples) == 0:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = {
                "avg_scores": None,
                "flipped_pcts": None,
                "auc_values": [],
                "mean_auc": None,
                "std_auc": None
            }
            continue

        # Extract AUC values
        auc_values = [s["auc"] for s in valid_samples if not np.isnan(s["auc"])]

        # Calculate mean and std of AUC
        if len(auc_values) > 0:
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)
        else:
            mean_auc = None
            std_auc = None

        # Interpolate scores to a common x-axis for averaging
        try:
            # Find common x-axis (flipped percentages)
            common_x = np.linspace(0, 100, 101)  # 0% to 100% in 1% increments

            # Interpolate each sample to common x-axis
            interpolated_scores = []

            for sample in valid_samples:
                flipped_pcts = sample["flipped_pcts"]
                scores = sample["scores"]

                # Check if we have enough points for interpolation
                if len(flipped_pcts) > 1 and len(scores) > 1:
                    # Ensure flipped_pcts is strictly increasing
                    if np.all(np.diff(flipped_pcts) > 0):
                        # Interpolate to common x-axis
                        from scipy.interpolate import interp1d
                        f = interp1d(flipped_pcts, scores, bounds_error=False, fill_value="extrapolate")
                        interpolated_scores.append(f(common_x))

            # Average interpolated scores
            if len(interpolated_scores) > 0:
                avg_scores = np.mean(interpolated_scores, axis=0)
            else:
                avg_scores = None

            # Store aggregated results0
            agg_results[method_name] = {
                "avg_scores": avg_scores,
                "flipped_pcts": common_x,
                "auc_values": auc_values,
                "mean_auc": mean_auc,
                "std_auc": std_auc
            }

        except Exception as e:
            print(f"Error aggregating results0 for {method_name}: {e}")
            agg_results[method_name] = {
                "avg_scores": None,
                "flipped_pcts": None,
                "auc_values": auc_values,
                "mean_auc": mean_auc,
                "std_auc": std_auc
            }

    return agg_results
'''def aggregate_results(results0):
    """
    Aggregate window flipping results0 across all samples.

    Args:
        results0: Dictionary with results0 for each method and sample

    Returns:
        agg_results: Dictionary with aggregated results0
    """
    agg_results = {}

    for method_name, samples in results0.items():
        # Initialize storage for this method
        method_results = {
            "avg_scores": [],
            "flipped_pcts": None,
            "auc_values": [],
            "mean_auc": None,
            "std_auc": None
        }

        # Collect all AUC values
        valid_samples = [s for s in samples if s["scores"] is not None]

        if not valid_samples:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = method_results
            continue

        # Get flipped percentages from first sample (should be the same for all)
        method_results["flipped_pcts"] = valid_samples[0]["flipped_pcts"]

        # Initialize scores array with correct dimensions
        n_steps = len(valid_samples[0]["scores"])
        all_scores = np.zeros((len(valid_samples), n_steps))

        # Collect scores and AUC values
        for i, sample in enumerate(valid_samples):
            all_scores[i] = sample["scores"]
            method_results["auc_values"].append(sample["auc"])

        # Calculate average scores at each flipping step
        method_results["avg_scores"] = np.mean(all_scores, axis=0)

        # Calculate mean and std of AUC values
        method_results["mean_auc"] = np.mean(method_results["auc_values"])
        method_results["std_auc"] = np.std(method_results["auc_values"])

        # Store results0 for this method
        agg_results[method_name] = method_results

    return agg_results
'''


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value = 'complete_zero'):
    """
    Plot aggregated window flipping results0.

    Args:
        agg_results: Dictionary with aggregated results0 for each method
        most_relevant_first: Whether most relevant windows were flipped first
    """
    plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'Aggregate Window Flipping Results\n(Flipping {flip_order} windows first, \n Reference Value Method Flipped with: {reference_value})', fontsize=16)
    plt.xlabel('Percentage of Time Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def plot_auc_distribution(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot distribution of AUC values for each method.

    Args:
        agg_results: Dictionary with aggregated results0 for each method
        most_relevant_first: Whether most relevant windows were flipped first
    """
    plt.figure(figsize=(14, 6))

    # Count number of methods with valid data
    valid_methods = [m for m, r in agg_results.items() if len(r["auc_values"]) > 0]
    n_methods = len(valid_methods)

    if n_methods == 0:
        print("No valid methods to plot")
        return

    # Create boxplots
    boxes = []
    labels = []

    for method_name, results in agg_results.items():
        if len(results["auc_values"]) == 0:
            continue

        boxes.append(results["auc_values"])
        labels.append(method_name)

    plt.boxplot(boxes, tick_labels=labels)  # Updated parameter name

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(f'AUC Distribution Across Samples\n(Flipping {flip_order} windows first), \n Reference Value Method Flipped with: {reference_value})', fontsize=16)
    plt.ylabel('AUC Value', fontsize=14)
    plt.grid(True, linestyle='--', axis='y', alpha=0.7)

    # Add text with mean and std
    for i, method_name in enumerate(labels):
        mean_auc = agg_results[method_name]["mean_auc"]
        std_auc = agg_results[method_name]["std_auc"]
        plt.text(i + 1, plt.ylim()[0] + 0.05, f"μ={mean_auc:.4f}\nσ={std_auc:.4f}",
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

    return plt.gcf()  # Return the figure for saving


def save_results_to_csv(agg_results, results, filename_prefix):
    """
    Save results0 to CSV files.

    Args:
        agg_results: Dictionary with aggregated results0
        results: Dictionary with individual sample results0
        filename_prefix: Prefix for filenames
    """
    import pandas as pd

    # Save aggregated results0
    agg_data = []
    for method_name, method_results in agg_results.items():
        agg_data.append({
            "Method": method_name,
            "Mean AUC": method_results["mean_auc"] if "mean_auc" in method_results else None,
            "Std AUC": method_results["std_auc"] if "std_auc" in method_results else None,
            "Num Samples": len(method_results["auc_values"]) if "auc_values" in method_results else 0
        })

    agg_df = pd.DataFrame(agg_data)
    agg_df.to_csv(f"{filename_prefix}_aggregate.csv", index=False)
    print(f"Saved aggregate results0 to {filename_prefix}_aggregate.csv")

    # Save individual sample results0
    samples_data = []
    try:
        for method_name, samples in results.items():
            if not isinstance(samples, list):
                print(f"Warning: Expected list for method {method_name}, got {type(samples)}. Skipping.")
                continue

            for sample_idx, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    print(
                        f"Warning: Expected dict for sample {sample_idx} in method {method_name}, got {type(sample)}. Skipping.")
                    continue

                if "scores" not in sample or sample["scores"] is None:
                    continue

                samples_data.append({
                    "Method": method_name,
                    "Sample Index": sample.get("sample_idx", sample_idx),
                    "AUC": sample.get("auc", float('nan'))
                })

        if samples_data:  # Only save if we have data
            samples_df = pd.DataFrame(samples_data)
            samples_df.to_csv(f"{filename_prefix}_samples.csv", index=False)
            print(f"Saved sample results0 to {filename_prefix}_samples.csv")
        else:
            print("No valid sample data to save")

    except Exception as e:
        print(f"Error saving sample results0: {str(e)}")
        print("Continuing execution despite error in saving sample results0")





# Wrapper functions for different DFT-based attribution methods to match your time domain wrappers

def dft_lrp_wrapper(model, sample, target=None):
    """
    Wrapper for DFT-LRP attribution method.
    """

    device = sample.device
    signal_length = sample.shape[1]

    _, relevance_freq, signal_freq, _, freqs, _ = compute_basic_dft_lrp(
        model=model,
        sample=sample,
        label=target,
        device=str(device),
        signal_length=signal_length,
        leverage_symmetry=True,
        sampling_rate=400
    )

    return relevance_freq, signal_freq, freqs


def dft_grad_times_input_wrapper(model, sample, target=None):
    """
    Wrapper for DFT Gradient×Input attribution method.
    """

    device = sample.device
    signal_length = sample.shape[1]

    _, relevance_freq, signal_freq, _, freqs, _ = compute_dft_gradient_input(
        model=model,
        sample=sample,
        label=target,
        device=str(device),
        signal_length=signal_length,
        leverage_symmetry=True,
        sampling_rate=400
    )

    return relevance_freq, signal_freq, freqs


def dft_smoothgrad_wrapper(model, sample, target=None):
    """
    Wrapper for DFT SmoothGrad attribution method.
    """

    device = sample.device
    signal_length = sample.shape[1]

    _, relevance_freq, signal_freq, _, freqs, _ = compute_dft_smoothgrad(
        model=model,
        sample=sample,
        label=target,
        device=str(device),
        signal_length=signal_length,
        leverage_symmetry=True,
        sampling_rate=400,
        num_samples=20,
        noise_level=0.2
    )

    return relevance_freq, signal_freq, freqs


def dft_occlusion_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Fixed wrapper for DFT Occlusion attribution method that handles window size properly.
    """
    # Set a fixed, smaller window size for occlusion that's appropriate for time series
    occlusion_window_size = 5  # Use a small window size for occlusion

    signal_length = sample.shape[1]

    try:
        # Call the DFT occlusion function with the fixed window size
        _, relevance_freq, signal_freq, _, freqs, _ = compute_dft_occlusion(
            model=model,
            sample=sample,
            label=target,
            device=device,
            signal_length=signal_length,
            leverage_symmetry=True,
            sampling_rate=400,
            occlusion_type="zero",
            window_size=occlusion_window_size  # Use fixed window size for occlusion
        )

        return relevance_freq, signal_freq, freqs

    except Exception as e:
        print(f"Error in DFT-Occlusion: {str(e)}")
        # Create empty placeholders with correct shapes
        n_channels = sample.shape[0]
        freq_length = signal_length // 2 + 1  # For leverage_symmetry=True

        # Return empty arrays with correct shapes
        relevance_freq = np.zeros((n_channels, freq_length), dtype=np.float32)
        signal_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        freqs = np.fft.rfftfreq(signal_length, d=1.0 / 400)

        return relevance_freq, signal_freq, freqs


def verify_frequency_reference_hypothesis(model, test_loader, device, n_samples=50, reference_types=None):
    """
    Verify that different frequency domain reference values work as expected.
    Tests various reference values to find the one that effectively
    changes model predictions when frequencies are replaced.

    Args:
        model: Trained model
        test_loader: DataLoader with test samples
        device: Device to run on
        n_samples: Number of samples to test
        reference_types: List of reference types to test, if None uses default list

    Returns:
        Dictionary with results0 for each reference type
    """
    import torch
    import numpy as np
    from utils.dft_lrp import EnhancedDFTLRP

    print("\n===== Verifying frequency domain reference types =====")

    # List of reference values to test
    if reference_types is None:
        reference_types = ["zero", "noise", "mild_noise", "shift", "invert", "magnitude_zero", "phase_zero",
                           "random_complex", "complete_zero"]

    results = {}

    # Process samples
    sample_count = 0
    changed_predictions = {ref_type: 0 for ref_type in reference_types}

    # Get n_samples from the dataloader
    for batch_idx, (data, targets) in enumerate(test_loader):
        for i in range(len(data)):
            if sample_count >= n_samples:
                break

            sample = data[i].to(device)
            sample_count += 1

            # Get original prediction
            with torch.no_grad():
                original_output = model(sample.unsqueeze(0))
                original_pred = torch.argmax(original_output, 1).item()

            # Get sample as numpy array
            sample_np = sample.detach().cpu().numpy()
            n_channels, time_steps = sample_np.shape

            # Setup DFT-LRP for transformations
            dftlrp = EnhancedDFTLRP(
                signal_length=time_steps,
                leverage_symmetry=True,
                precision=32,
                cuda=(device == "cuda"),
                create_inverse=True
            )

            # Transform to frequency domain
            signal_freq = np.zeros((n_channels, time_steps // 2 + 1), dtype=np.complex128)
            for c in range(n_channels):
                signal_freq[c] = np.fft.rfft(sample_np[c])

            # Test each reference type
            for ref_type in reference_types:
                # Create reference values based on type
                if ref_type == "zero":
                    ref_freq = np.zeros_like(signal_freq)

                elif ref_type == "noise":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        magnitude = np.abs(signal_freq[c]).std() * 5
                        ref_freq[c] = np.random.normal(0, magnitude, signal_freq.shape[1]) + \
                                      1j * np.random.normal(0, magnitude, signal_freq.shape[1])

                elif ref_type == "mild_noise":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        magnitude = np.abs(signal_freq[c]).std()
                        ref_freq[c] = np.random.normal(0, magnitude, signal_freq.shape[1]) + \
                                      1j * np.random.normal(0, magnitude, signal_freq.shape[1])

                elif ref_type == "shift":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        magnitude = np.abs(signal_freq[c])
                        phase = np.angle(signal_freq[c]) + np.pi
                        ref_freq[c] = magnitude * np.exp(1j * phase)

                elif ref_type == "invert":
                    ref_freq = -signal_freq

                elif ref_type == "magnitude_zero":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        phase = np.angle(signal_freq[c])
                        ref_freq[c] = 1e-10 * np.exp(1j * phase)

                elif ref_type == "phase_zero":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        ref_freq[c] = np.abs(signal_freq[c])

                elif ref_type == "random_complex":
                    ref_freq = np.zeros_like(signal_freq)
                    for c in range(n_channels):
                        # Completely random complex values with high magnitude
                        magnitude = np.abs(signal_freq[c]).mean() * 3  # 3x mean magnitude
                        real_part = np.random.normal(0, magnitude, signal_freq.shape[1])
                        imag_part = np.random.normal(0, magnitude, signal_freq.shape[1])
                        ref_freq[c] = real_part + 1j * imag_part

                # Convert back to time domain
                modified_time = np.zeros_like(sample_np)
                for c in range(n_channels):
                    modified_time[c] = np.fft.irfft(ref_freq[c], n=time_steps)

                # Get prediction
                modified_tensor = torch.tensor(modified_time, dtype=torch.float32).to(device)

                with torch.no_grad():
                    modified_output = model(modified_tensor.unsqueeze(0))
                    modified_pred = torch.argmax(modified_output, 1).item()

                # Check if prediction changed
                if modified_pred != original_pred:
                    changed_predictions[ref_type] += 1

            # Print progress
            if sample_count % 5 == 0:
                print(f"Verified {sample_count}/{n_samples} samples")

    # Calculate percentages
    for ref_type in reference_types:
        pct_changed = (changed_predictions[ref_type] / n_samples) * 100
        results[ref_type] = {
            "changed": changed_predictions[ref_type],
            "total": n_samples,
            "pct_changed": pct_changed
        }

        print(f"{ref_type} reference test: {changed_predictions[ref_type]}/{n_samples} samples "
              f"({pct_changed:.2f}%) changed prediction when replaced")

    return results


def visualize_frequency_flipping(model, sample, attribution_method, target_class=None, n_steps=5,
                                 window_size=10, most_relevant_first=True, reference_value="complete_zero",
                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                 leverage_symmetry=True, sampling_rate=400):
    """
    Visualize frequency domain window flipping to check if transformations are working correctly.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions
        target_class: Target class to track (if None, use predicted class)
        n_steps: Number of steps for visualization (use small number for clarity)
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Type of reference value to use
        device: Device to run on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import gc
    from utils.dft_lrp import EnhancedDFTLRP

    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

        original_score = original_prob[target_class].item()
        print(f"Original confidence for class {target_class}: {original_score:.4f}")

    # Get frequency domain attributions
    print("Computing frequency domain attributions...")
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        label=target_class,
        device=device,
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Check shapes
    print(f"Input signal shape: {input_signal.shape}")
    print(f"Time domain relevance shape: {relevance_time.shape}")
    print(f"Frequency domain signal shape: {signal_freq.shape}")
    print(f"Frequency domain relevance shape: {relevance_freq.shape}")

    # Create DFT-LRP object for inverse transforms
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_inverse=True
    )

    # Determine frequency domain dimensions
    freq_length = signal_freq.shape[1]

    # Prepare reference value for frequency domain
    if reference_value == "zero":
        reference_value_freq = np.zeros_like(signal_freq)
    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)
    elif reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)

    elif reference_value == "noise":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5
            reference_value_freq[c] = np.random.normal(0, magnitude, signal_freq.shape[1]) + \
                                      1j * np.random.normal(0, magnitude, signal_freq.shape[1])
    else:
        # Default to magnitude_zero
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)

    # Calculate window importance
    n_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Use absolute values of relevance for importance
            window_importance[channel, window_idx] = np.mean(np.abs(relevance_freq[channel, start_idx:end_idx]))

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Calculate windows to flip per step
    total_windows = n_channels * n_windows
    windows_per_step = max(1, total_windows // n_steps)

    # Setup visualization
    # We'll show:
    # 1. Original time domain signal
    # 2. Original frequency magnitude
    # 3. Flipped frequency magnitude
    # 4. Reconstructed time domain signal
    # for each step

    fig = plt.figure(figsize=(18, n_steps * 5))
    gs = GridSpec(n_steps, 4, figure=fig)

    # Store scores
    scores = [original_score]

    # Show original signal (first step = no flipping)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    # Time domain signal - original
    for c in range(n_channels):
        ax1.plot(np.arange(time_steps), input_signal[c], label=f'Channel {c}')
    ax1.set_title("Original Time Domain Signal")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    # Frequency domain magnitude - original
    for c in range(n_channels):
        ax2.plot(freqs[:freq_length], np.abs(signal_freq[c]), label=f'Channel {c}')
    ax2.set_title("Original Frequency Domain Magnitude")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_yscale('log')
    ax2.legend()

    # Frequency domain phase - original
    for c in range(n_channels):
        ax3.plot(freqs[:freq_length], np.angle(signal_freq[c]), label=f'Channel {c}')
    ax3.set_title("Original Frequency Domain Phase")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Phase (radians)")
    ax3.legend()

    # Show relevance in frequency domain
    for c in range(n_channels):
        ax4.plot(freqs[:freq_length], np.abs(relevance_freq[c]), label=f'Channel {c}')
    ax4.set_title("Frequency Domain Relevance (Magnitude)")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Relevance")
    ax4.legend()

    # Iteratively flip windows and visualize
    for step in range(1, n_steps):
        n_windows_to_flip = min(step * windows_per_step, total_windows)
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Create a copy of original frequency representation
        flipped_freq = signal_freq.copy()

        # Mark which bins are being flipped
        flipped_bins = np.zeros((n_channels, freq_length), dtype=bool)

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Mark which bins are being flipped
            flipped_bins[channel_idx, start_idx:end_idx] = True

            # Replace frequency components with reference
            flipped_freq[channel_idx, start_idx:end_idx] = reference_value_freq[channel_idx, start_idx:end_idx]

        # Transform back to time domain
        try:
            # First try numpy IFFT for simplicity
            flipped_time = np.zeros((n_channels, time_steps))
            for c in range(n_channels):
                if leverage_symmetry:
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real
        except Exception as e:
            print(f"Error in numpy IFFT: {e}")
            print("Trying DFT-LRP inverse transform...")

            try:
                # Try DFT-LRP inverse transform
                flipped_time = np.zeros((n_channels, time_steps))

                if hasattr(dftlrp, 'inverse_fourier_layer') and dftlrp.inverse_fourier_layer is not None:
                    for c in range(n_channels):
                        if leverage_symmetry:
                            # Need to reshape for DFT-LRP
                            freq_real = flipped_freq[c].real
                            freq_imag = np.concatenate([
                                np.zeros(1),  # DC is real
                                flipped_freq[c, 1:-1].imag,
                                np.zeros(1)  # Nyquist is real
                            ])

                            # Concatenate as expected by DFT-LRP
                            freq_data = np.concatenate([freq_real, freq_imag])
                            freq_tensor = torch.tensor(freq_data, dtype=torch.float32)
                            freq_tensor = freq_tensor.unsqueeze(0)

                            if dftlrp.cuda:
                                freq_tensor = freq_tensor.cuda()

                            # Apply inverse transform
                            with torch.no_grad():
                                time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                            # Get result
                            flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                        else:
                            # Non-symmetry case
                            freq_real = flipped_freq[c].real
                            freq_imag = flipped_freq[c].imag

                            # Concatenate
                            freq_data = np.concatenate([freq_real, freq_imag])
                            freq_tensor = torch.tensor(freq_data, dtype=torch.float32)
                            freq_tensor = freq_tensor.unsqueeze(0)

                            if dftlrp.cuda:
                                freq_tensor = freq_tensor.cuda()

                            # Apply inverse transform
                            with torch.no_grad():
                                time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                            # Get result
                            flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
                else:
                    raise ValueError("DFT-LRP inverse transform not available")
            except Exception as e:
                print(f"Error in DFT-LRP inverse transform: {e}")
                print("Using zeros as fallback")
                flipped_time = np.zeros((n_channels, time_steps))

        # Get model prediction
        flipped_sample = torch.tensor(flipped_time, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(flipped_sample.unsqueeze(0))
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()
            scores.append(score)

            predicted_class = torch.argmax(prob).item()
            predicted_score = prob[predicted_class].item()

        # Print statistics to verify flipping
        print(
            f"\nStep {step}: Flipped {n_windows_to_flip}/{total_windows} windows ({n_windows_to_flip / total_windows * 100:.1f}%)")
        print(f"Original vs Flipped stats:")
        for c in range(n_channels):
            orig_mag_mean = np.abs(signal_freq[c]).mean()
            flip_mag_mean = np.abs(flipped_freq[c]).mean()
            orig_time_std = input_signal[c].std()
            flip_time_std = flipped_time[c].std()

            print(
                f"  Channel {c}: Freq Mag {orig_mag_mean:.4e} → {flip_mag_mean:.4e}, Time StdDev {orig_time_std:.4f} → {flip_time_std:.4f}")

            # Count how many bins were actually modified
            bins_flipped = np.sum(flipped_bins[c])
            print(f"  Channel {c}: {bins_flipped}/{freq_length} frequency bins flipped")

        print(f"Model confidence: {original_score:.4f} → {score:.4f}")
        print(f"Predicted class: {predicted_class} (confidence: {predicted_score:.4f})")

        # Plot the flipped results0
        ax1 = fig.add_subplot(gs[step, 0])
        ax2 = fig.add_subplot(gs[step, 1])
        ax3 = fig.add_subplot(gs[step, 2])
        ax4 = fig.add_subplot(gs[step, 3])

        # Time domain signal - flipped
        for c in range(n_channels):
            ax1.plot(np.arange(time_steps), flipped_time[c], label=f'Channel {c}')
        ax1.set_title(f"Flipped Time Domain Signal (Step {step})")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax1.legend()

        # Frequency domain magnitude - flipped
        for c in range(n_channels):
            # Plot original magnitude in light color
            ax2.plot(freqs[:freq_length], np.abs(signal_freq[c]), linestyle='--', alpha=0.3, label=f'Ch{c} Orig')

            # Plot flipped magnitude
            ax2.plot(freqs[:freq_length], np.abs(flipped_freq[c]), label=f'Ch{c} Flipped')

            # Highlight flipped regions
            flipped_x = freqs[:freq_length][flipped_bins[c]]
            flipped_y = np.abs(flipped_freq[c])[flipped_bins[c]]
            ax2.scatter(flipped_x, flipped_y, color='red', s=10, alpha=0.5)

        ax2.set_title(f"Flipped Frequency Domain Magnitude (Step {step})")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_yscale('log')
        ax2.legend()

        # Frequency domain phase - flipped
        for c in range(n_channels):
            # Plot original phase in light color
            ax3.plot(freqs[:freq_length], np.angle(signal_freq[c]), linestyle='--', alpha=0.3, label=f'Ch{c} Orig')

            # Plot flipped phase
            ax3.plot(freqs[:freq_length], np.angle(flipped_freq[c]), label=f'Ch{c} Flipped')

            # Highlight flipped regions
            flipped_x = freqs[:freq_length][flipped_bins[c]]
            flipped_y = np.angle(flipped_freq[c])[flipped_bins[c]]
            ax3.scatter(flipped_x, flipped_y, color='red', s=10, alpha=0.5)

        ax3.set_title(f"Flipped Frequency Domain Phase (Step {step})")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Phase (radians)")
        ax3.legend()

        # Time domain comparison
        for c in range(n_channels):
            # Plot original signal
            ax4.plot(np.arange(time_steps), input_signal[c], linestyle='--', alpha=0.3, label=f'Ch{c} Orig')

            # Plot flipped signal
            ax4.plot(np.arange(time_steps), flipped_time[c], label=f'Ch{c} Flipped')

        ax4.set_title(f"Time Domain Comparison (Step {step})")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Amplitude")
        ax4.legend()

    # Add overall title with confidence info
    plt.suptitle(f"Frequency Window Flipping Visualization\n" +
                 f"Reference: {reference_value}, Window Size: {window_size}, Most Relevant First: {most_relevant_first}\n" +
                 f"Target Class: {target_class}, Confidence: {original_score:.4f} → {scores[-1]:.4f}",
                 fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure
    plt.savefig(
        f"freq_window_flipping_vis_{reference_value}_{window_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        dpi=600, bbox_inches='tight')

    # Clean up
    plt.close(fig)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Return scores for verification
    return scores


def analyze_reference_impact(model, test_loader, device="cuda", n_samples=5):
    """
    Analyze how different reference values impact the signal properties and model predictions.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Get some samples
    samples = []
    targets = []
    for batch_idx, (data, target) in enumerate(test_loader):
        for i in range(len(data)):
            if len(samples) >= n_samples:
                break
            samples.append(data[i])
            targets.append(target[i])
        if len(samples) >= n_samples:
            break

    # Reference types to test
    reference_types = [
        "zero", "magnitude_zero", "complete_zero", "invert",
        "noise", "mild_noise", "random_complex"
    ]

    # Metrics to track
    metrics = {
        ref_type: {
            "time_domain_energy_ratio": [],
            "freq_domain_energy_ratio": [],
            "confidence_ratio": []
        } for ref_type in reference_types
    }

    # Process each sample
    for idx, (sample, target) in enumerate(zip(samples, targets)):
        sample = sample.to(device)
        n_channels, time_steps = sample.shape

        # Get original prediction
        with torch.no_grad():
            original_output = model(sample.unsqueeze(0))
            original_prob = torch.softmax(original_output, dim=1)[0]
            original_score = original_prob[target].item()

        # Original signal properties
        sample_np = sample.cpu().numpy()
        original_energy_time = np.sum(sample_np ** 2)

        # Transform to frequency domain
        signal_freq = np.zeros((n_channels, time_steps // 2 + 1), dtype=np.complex128)
        for c in range(n_channels):
            signal_freq[c] = np.fft.rfft(sample_np[c])

        original_energy_freq = np.sum(np.abs(signal_freq) ** 2)

        print(f"\nAnalyzing sample {idx + 1}/{len(samples)}")
        print(f"Original time domain energy: {original_energy_time:.4e}")
        print(f"Original frequency domain energy: {original_energy_freq:.4e}")
        print(f"Original confidence: {original_score:.4f}")

        # Test each reference type
        for ref_type in reference_types:
            # Create reference frequency domain
            if ref_type == "zero":
                ref_freq = np.zeros_like(signal_freq)
            elif ref_type == "magnitude_zero":
                ref_freq = np.zeros_like(signal_freq)
                for c in range(n_channels):
                    phase = np.angle(signal_freq[c])
                    ref_freq[c] = 1e-10 * np.exp(1j * phase)
            elif ref_type == "complete_zero":
                ref_freq = np.zeros_like(signal_freq)
            elif ref_type == "invert":
                ref_freq = -signal_freq
            elif ref_type == "noise":
                ref_freq = np.zeros_like(signal_freq)
                for c in range(n_channels):
                    magnitude = np.abs(signal_freq[c]).std() * 5
                    phase = np.random.uniform(-np.pi, np.pi, signal_freq.shape[1])
                    ref_freq[c] = magnitude * np.exp(1j * phase)
            elif ref_type == "mild_noise":
                ref_freq = np.zeros_like(signal_freq)
                for c in range(n_channels):
                    magnitude = np.abs(signal_freq[c]).std()
                    phase = np.random.uniform(-np.pi, np.pi, signal_freq.shape[1])
                    ref_freq[c] = magnitude * np.exp(1j * phase)
            elif ref_type == "random_complex":
                ref_freq = np.zeros_like(signal_freq)
                for c in range(n_channels):
                    magnitude = np.abs(signal_freq[c]).mean() * 3
                    phase = np.random.uniform(-np.pi, np.pi, signal_freq.shape[1])
                    ref_freq[c] = magnitude * np.exp(1j * phase)

            # Transform back to time domain
            ref_time = np.zeros_like(sample_np)
            for c in range(n_channels):
                ref_time[c] = np.fft.irfft(ref_freq[c], n=time_steps)

            # Calculate energy ratios
            ref_energy_time = np.sum(ref_time ** 2)
            ref_energy_freq = np.sum(np.abs(ref_freq) ** 2)

            time_ratio = ref_energy_time / original_energy_time
            freq_ratio = ref_energy_freq / original_energy_freq

            # Get model confidence
            ref_tensor = torch.tensor(ref_time, dtype=torch.float32).to(device)
            with torch.no_grad():
                ref_output = model(ref_tensor.unsqueeze(0))
                ref_prob = torch.softmax(ref_output, dim=1)[0]
                ref_score = ref_prob[target].item()

            confidence_ratio = ref_score / original_score

            # Store metrics
            metrics[ref_type]["time_domain_energy_ratio"].append(time_ratio)
            metrics[ref_type]["freq_domain_energy_ratio"].append(freq_ratio)
            metrics[ref_type]["confidence_ratio"].append(confidence_ratio)

            print(f"{ref_type}: Time energy ratio: {time_ratio:.4e}, "
                  f"Freq energy ratio: {freq_ratio:.4e}, "
                  f"Confidence: {ref_score:.4f} ({confidence_ratio:.2f}x)")

    # Calculate averages
    print("\nAverages across all samples:")
    print(f"{'Reference':<15} {'Time Energy':<15} {'Freq Energy':<15} {'Confidence':<15}")
    print("-" * 60)

    for ref_type in reference_types:
        avg_time = np.mean(metrics[ref_type]["time_domain_energy_ratio"])
        avg_freq = np.mean(metrics[ref_type]["freq_domain_energy_ratio"])
        avg_conf = np.mean(metrics[ref_type]["confidence_ratio"])

        print(f"{ref_type:<15} {avg_time:<15.4e} {avg_freq:<15.4e} {avg_conf:<15.4f}")

    return metrics
def main():
    """
    Main function to run frequency domain window flipping evaluation.
    """



    # Clean up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    parser = argparse.ArgumentParser(description='Run frequency domain window flipping evaluation')

    # Model and data paths
    parser.add_argument('--model-path', type=str, required=False,
                        default="../cnn1d_model_test_newest.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=False,
                        default="../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD",
                        help='Directory containing data')
    parser.add_argument('--output-dir', type=str, required=False,
                        default="./results0",
                        help='Output directory for results0')

    # Window flipping parameters
    parser.add_argument('--n-steps', type=int, default=10,
                        help='Number of steps for window flipping')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Size of frequency windows for flipping')
    parser.add_argument('--max-samples', type=int, default=5,
                        help='Maximum number of samples to process')
    parser.add_argument('--reference', type=str, default="complete_zero",
                        help='Reference value for flipping: zero, noise, mild_noise, shift, invert, random_complex, etc. or "auto" to determine the best')
    parser.add_argument('--sampling-rate', type=int, default=400,
                        help='Sampling rate of the data in Hz')

    # Added option to skip reference verification
    parser.add_argument('--skip-verification', action='store_true',
                        help='Skip reference value verification (use specified reference directly)')

    # Device
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use (cuda/cpu)')

    # Parse arguments
    args = parser.parse_args()

    # Print configuration
    print("\n===== Frequency Domain Window Flipping Evaluation =====")
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Window Size: {args.window_size}")
    print(f"Reference Value: {args.reference}")
    print(f"Sampling Rate: {args.sampling_rate}")
    print(f"Max Samples: {args.max_samples}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = "../cnn1d_model_test_newest.ckpt"  # Path to your trained model
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results"



    from utils.dataloader import stratified_group_split
    _, _, test_loader, _ = stratified_group_split(data_dir)

    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path, args.device)

    # Set up DFT-based attribution methods
    dft_attribution_methods = {
        "DFT-LRP": compute_basic_dft_lrp,
        "DFT-GradInput": compute_dft_gradient_input,
        "DFT-SmoothGrad": compute_dft_smoothgrad,
        "DFT-Occlusion": compute_dft_occlusion
    }

    # Determine reference value to use
    reference_value = args.reference

    # Only do verification if reference is "auto" or if not skipping verification
    if reference_value == "auto" or not args.skip_verification:
        print("\nVerifying frequency domain reference values...")
        # Use fewer samples for quick verification
        verification_samples = min(30, args.max_samples if args.max_samples else 30)

        # Define reference types to test
        reference_types = ["zero", "noise", "mild_noise", "shift", "invert", "magnitude_zero", "phase_zero",
                           "random_complex", "complete_zero"]

        # Run verification
        ref_results = verify_frequency_reference_hypothesis(model, test_loader, args.device,
                                                            n_samples=verification_samples,
                                                            reference_types=reference_types)

        # Print verification results0 in a table format
        print("\nReference Value Verification Results:")
        print("-" * 60)
        print(f"{'Reference Type':<15} {'Changed Predictions':<20} {'Percentage':<10}")
        print("-" * 60)

        best_pct = 0
        best_ref = None

        for ref_type, metrics in ref_results.items():
            print(f"{ref_type:<15} {metrics['changed']}/{metrics['total']:<20} {metrics['pct_changed']:.2f}%")
            if metrics['pct_changed'] > best_pct:
                best_pct = metrics['pct_changed']
                best_ref = ref_type

        print("-" * 60)

        # If using auto reference, select the best one
        if reference_value == "auto":
            reference_value = best_ref
            print(f"\nAutomatically selected reference: {reference_value} "
                  f"({best_pct:.2f}% samples changed prediction)")
        else:
            print(f"\nUsing specified reference: {reference_value}")
            print(f"Note: Best reference would be {best_ref} "
                  f"({best_pct:.2f}% samples changed prediction)")
    else:
        print(f"\nSkipping verification, using specified reference: {reference_value}")

    # Run window flipping evaluation - most important first
    print("\n===== Evaluating with most important frequency windows flipped first =====")
    print("Tracking TRUE class probability during window flipping")
    print(f"Using reference value: {reference_value}")



    # First, analyze the impact of different reference types
    print("\n===== Analyzing impact of different reference types =====")
    reference_metrics = analyze_reference_impact(
        model=model,
        test_loader=test_loader,
        device=args.device,
        n_samples=5  # Use a small number for quick analysis
    )


    results_most_first, agg_results_most_first = run_frequency_window_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=dft_attribution_methods,
        n_steps=args.n_steps,
        window_size=args.window_size,
        most_relevant_first=True,
        max_samples=args.max_samples,
        reference_value=reference_value,
        leverage_symmetry=True,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
        device=args.device
    )

    # Run window flipping evaluation - least important first
    print("\n===== Evaluating with least important frequency windows flipped first =====")
    print("Tracking TRUE class probability during window flipping")
    print(f"Using reference value: {reference_value}")

    results_least_first, agg_results_least_first = run_frequency_window_flipping_evaluation(
        model=model,
        test_loader=test_loader,
        attribution_methods=dft_attribution_methods,
        n_steps=args.n_steps,
        window_size=args.window_size,
        most_relevant_first=False,
        max_samples=args.max_samples,
        reference_value=reference_value,
        leverage_symmetry=True,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
        device=args.device
    )

    # Calculate and print faithfulness ratios
    print("\n===== Faithfulness Ratios (Least/Most AUC) =====")
    print(f"{'Method':<20} {'Most AUC':<10} {'Least AUC':<10} {'Ratio':<10}")
    print("-" * 60)

    for method in dft_attribution_methods.keys():
        most_auc = agg_results_most_first[method]["mean_auc"]
        least_auc = agg_results_least_first[method]["mean_auc"]
        ratio = least_auc / most_auc if most_auc > 0 else float('nan')

        print(f"{method:<20} {most_auc:<10.4f} {least_auc:<10.4f} {ratio:<10.4f}")

    print("\nEvaluation complete!")

    # Example usage:
    sample, target = next(iter(test_loader))
    sample = sample[0]  # Get first sample
    target = target[0]  # Get first target

    # Run visualization with the complete_zero reference (most aggressive)
    scores = visualize_frequency_flipping(
        model=model,
        sample=sample,
        attribution_method=compute_basic_dft_lrp,
        target_class=target,
        n_steps=5,  # Use 5 steps for clear visualization
        window_size=10,
        most_relevant_first=True,
        reference_value="complete_zero",  # Try the most aggressive option
        leverage_symmetry=True,
        sampling_rate=400
    )

    print("Confidence scores after each step:")
    for i, score in enumerate(scores):
        print(f"Step {i}: {score:.4f}")


# If run as a script
if __name__ == "__main__":
    main()