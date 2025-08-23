import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import pandas as pd
import os
import random
import h5py
from scipy import stats
from pathlib import Path
from datetime import datetime
from utils.dft_lrp import EnhancedDFTLRP

# Import your model loading utility
from utils.baseline_xai import load_model, occlusion_signal_relevance


def frequency_window_flipping_single(model, sample, attribution_method, target_class=None, n_steps=20,
                                     window_size=10, most_relevant_first=True,
                                     reference_value="complete_zero",
                                     device="cuda" if torch.cuda.is_available() else "cpu",
                                     leverage_symmetry=True,
                                     sampling_rate=400):
    """
    Perform window flipping analysis in the frequency domain on a single time series sample.
    Properly handles symmetry properties of the DFT.
    """
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

    # Get frequency domain attributions
    # Call with correct parameter names (target_class instead of label)
    attribution_results = attribution_method(
        model=model,
        sample=sample,
        target=target_class    )

    # Unpack results - these functions typically return:
    # (relevance_time, relevance_freq, signal_freq, input_signal, freqs, target)
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_results

    # Determine frequency length
    freq_length = signal_freq.shape[1]

    # Create DFT-LRP object for transformations with proper symmetry handling
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_inverse=True  # Need inverse transform
    )

    # Prepare reference value for frequency domain
    if reference_value == "complete_zero":
        reference_value_freq = np.zeros_like(signal_freq)
    elif reference_value == "magnitude_zero":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            phase = np.angle(signal_freq[c])
            reference_value_freq[c] = 1e-10 * np.exp(1j * phase)
    elif reference_value == "noise":
        reference_value_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 5
            phase = np.random.uniform(-np.pi, np.pi, freq_length)

            # Handle symmetry constraints if needed
            if leverage_symmetry:
                # For real signals with symmetry, DC and Nyquist must be real
                if freq_length > 0:
                    phase[0] = 0  # DC component
                if freq_length > 1:
                    phase[-1] = 0  # Nyquist component

            reference_value_freq[c] = magnitude * np.exp(1j * phase)
    else:
        # Default to complete zero
        reference_value_freq = np.zeros_like(signal_freq)

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
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

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

        # Transform back to time domain using EnhancedDFTLRP's inverse transform
        flipped_time = np.zeros((n_channels, time_steps))

        # Use the EnhancedDFTLRP class for proper inverse transform with symmetry
        for c in range(n_channels):
            # Format frequency data for EnhancedDFTLRP
            if leverage_symmetry:
                # When using symmetry, need to handle real signal constraints
                freq_real = flipped_freq[c].real

                if freq_length > 2:  # If we have components between DC and Nyquist
                    freq_imag = flipped_freq[c, 1:-1].imag

                    # Format as expected by EnhancedDFTLRP: [real parts, imag parts]
                    freq_data = np.concatenate([freq_real, freq_imag])
                else:
                    # Special case with only DC and possibly Nyquist
                    freq_data = freq_real

                # Convert to tensor
                freq_tensor = torch.tensor(freq_data, dtype=torch.float32).unsqueeze(0).to(device)

                # Apply EnhancedDFTLRP's inverse transform
                with torch.no_grad():
                    time_tensor = dftlrp.inverse_fourier_layer(freq_tensor)

                # Get result
                flipped_time[c] = time_tensor.cpu().numpy().squeeze(0)
            else:
                # Without symmetry, directly use numpy's ifft
                flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # Get model output for flipped sample
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)

    return scores, flipped_pcts


def improved_frequency_guided_time_window_flipping_single(model, sample, attribution_method, target_class=None,
                                                          n_steps=20, window_size=40, most_relevant_first=True,
                                                          device="cuda" if torch.cuda.is_available() else "cpu",
                                                          leverage_symmetry=True, sampling_rate=400,
                                                          freq_windows_to_consider=0.5):  # Consider top 50% by default
    """
    Perform frequency-guided time window flipping with improved efficiency:
    1. Identify important frequency components using attribution method
    2. Sort frequency windows by importance and only process the most important ones
    3. Map those frequencies back to time domain using EnhancedDFTLRP
    4. Apply class-specific reference values: mild_noise for normal, zero for faulty
    5. Measure impact on model prediction

    Args:
        freq_windows_to_consider: Fraction of frequency windows to consider (0-1)
                                 or absolute count (if > 1)
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()
            print(f"No target class provided, using predicted class: {target_class}")

        original_score = original_prob[target_class].item()
        print(f"Original score for class {target_class}: {original_score:.4f}")

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        target=target_class
    )

    # Create DFT-LRP object for transforms
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_forward=True,
        create_inverse=True,
        create_transpose_inverse=False
    )

    # Determine frequency length based on symmetry
    freq_length = signal_freq.shape[1]
    print(f"Frequency length: {freq_length}, Time steps: {time_steps}")

    # Calculate frequency window importance
    n_freq_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_freq_windows += 1

    freq_window_importance = np.zeros((n_channels, n_freq_windows))

    for channel in range(n_channels):
        for window_idx in range(n_freq_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute attribution within the window
            freq_window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort frequency window importance
    flat_importance = freq_window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Determine how many frequency windows to consider
    if freq_windows_to_consider <= 1.0:
        # Interpret as a fraction
        num_windows_to_consider = int(len(sorted_indices) * freq_windows_to_consider)
    else:
        # Interpret as an absolute count
        num_windows_to_consider = min(int(freq_windows_to_consider), len(sorted_indices))

    # Get the most important frequency windows
    important_freq_windows = sorted_indices[:num_windows_to_consider]

    # Convert flat indices back to channel, window indices
    freq_channel_indices = important_freq_windows // n_freq_windows
    freq_window_indices = important_freq_windows % n_freq_windows

    print(f"Considering {num_windows_to_consider} most important frequency windows out of {len(sorted_indices)} total")

    # Calculate time window size - similar to what we use in time domain
    time_window_size = time_steps // 20  # Reasonable size for time windows
    n_time_windows = time_steps // time_window_size
    if time_steps % time_window_size > 0:
        n_time_windows += 1

    # Map each frequency window to time windows
    # This is a mapping of importance - which time windows are influenced by important frequencies
    time_window_importance = np.zeros((n_channels, n_time_windows))

    # For each important frequency window, calculate how it affects time windows
    for i in range(len(important_freq_windows)):
        channel = freq_channel_indices[i]
        freq_window_idx = freq_window_indices[i]

        # Skip if this frequency window has zero importance
        if freq_window_importance[channel, freq_window_idx] == 0:
            continue

        # Get frequency indices for this window
        start_idx = freq_window_idx * window_size
        end_idx = min((freq_window_idx + 1) * window_size, freq_length)

        # Create a frequency domain mask that isolates just this frequency window
        freq_mask = np.zeros_like(signal_freq[channel], dtype=np.complex128)
        freq_mask[start_idx:end_idx] = signal_freq[channel, start_idx:end_idx]

        # Convert to tensor for EnhancedDFTLRP with careful device handling
        if leverage_symmetry:
            # Need to handle the symmetry properly
            if dftlrp.has_inverse_fourier_layer:
                # Format for inverse DFT with symmetry
                # Convert from complex to real representation that dftlrp expects
                if dftlrp.cuda:
                    # If using CUDA, ensure data is on the correct device
                    freq_real = torch.tensor(np.real(freq_mask), dtype=torch.float32).to(device)

                    # For symmetry, prepare properly - depends on dftlrp's expected format
                    # This is based on how the EnhancedDFTLRP class is implemented
                    if freq_length > 1:
                        if freq_length == time_steps // 2 + 1:
                            # Standard half-spectrum format for real signals
                            freq_imag = torch.tensor(np.imag(freq_mask[1:-1]), dtype=torch.float32).to(device)
                            # Concatenate real and imaginary parts as expected by dftlrp
                            freq_data = torch.cat([freq_real, freq_imag], dim=0).to(device)
                        else:
                            # Handle non-standard spectrum length
                            freq_imag = torch.tensor(np.imag(freq_mask[1:]), dtype=torch.float32).to(device)
                            freq_data = torch.cat([freq_real, freq_imag], dim=0).to(device)
                    else:
                        # Only DC component
                        freq_data = freq_real.to(device)

                    # Use EnhancedDFTLRP for inverse transform, ensuring all tensors are on same device
                    try:
                        time_tensor = dftlrp.inverse_fourier_layer(freq_data.unsqueeze(0))
                        time_contribution = time_tensor.cpu().numpy().squeeze(0)
                    except RuntimeError as e:
                        print(f"Error using dftlrp inverse: {e}")
                        print("Falling back to torch.fft.irfft")
                        # Fallback to torch's irfft
                        freq_complex = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                        time_tensor = torch.fft.irfft(freq_complex, n=time_steps)
                        time_contribution = time_tensor.cpu().numpy()
                else:
                    # CPU path
                    # Fallback to numpy's irfft for CPU mode
                    time_contribution = np.fft.irfft(freq_mask, n=time_steps)
            else:
                # No inverse layer in DFTLRP - fallback to standard methods
                if device == "cuda":
                    freq_tensor = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                    time_tensor = torch.fft.irfft(freq_tensor, n=time_steps)
                    time_contribution = time_tensor.cpu().numpy()
                else:
                    time_contribution = np.fft.irfft(freq_mask, n=time_steps)
        else:
            # Without symmetry
            if dftlrp.has_inverse_fourier_layer:
                if dftlrp.cuda:
                    try:
                        freq_tensor = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                        # For non-symmetry case, we need the full complex signal
                        freq_data_real = torch.real(freq_tensor)
                        freq_data_imag = torch.imag(freq_tensor)
                        freq_data = torch.cat([freq_data_real, freq_data_imag], dim=0).to(device)

                        time_tensor = dftlrp.inverse_fourier_layer(freq_data.unsqueeze(0))
                        time_contribution = time_tensor.cpu().numpy().squeeze(0)
                    except RuntimeError as e:
                        print(f"Error using dftlrp inverse (non-symmetry): {e}")
                        print("Falling back to torch.fft.ifft")
                        freq_tensor = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                        time_tensor = torch.fft.ifft(freq_tensor, n=time_steps).real
                        time_contribution = time_tensor.cpu().numpy()
                else:
                    # CPU path
                    time_contribution = np.fft.ifft(freq_mask, n=time_steps).real
            else:
                # Standard ifft
                if device == "cuda":
                    freq_tensor = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                    time_tensor = torch.fft.ifft(freq_tensor, n=time_steps).real
                    time_contribution = time_tensor.cpu().numpy()
                else:
                    time_contribution = np.fft.ifft(freq_mask, n=time_steps).real

        # Calculate which time windows are affected by this frequency
        # Weight by both frequency importance and time domain magnitude
        time_importance = np.abs(time_contribution) * freq_window_importance[channel, freq_window_idx]

        # Accumulate importance for each time window
        for time_window_idx in range(n_time_windows):
            start_t = time_window_idx * time_window_size
            end_t = min((time_window_idx + 1) * time_window_size, time_steps)

            # Sum importance within this window
            window_importance = np.sum(time_importance[start_t:end_t])
            time_window_importance[channel, time_window_idx] += window_importance

    # Normalize time window importance for each channel
    for channel in range(n_channels):
        if np.max(time_window_importance[channel]) > 0:
            time_window_importance[channel] /= np.max(time_window_importance[channel])

    # Flatten and sort time window importance
    flat_time_importance = time_window_importance.flatten()
    sorted_time_indices = np.argsort(flat_time_importance)

    if most_relevant_first:
        sorted_time_indices = sorted_time_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_time_windows

    # Create exponential progression for more gradual steps
    flip_percentages = np.linspace(0, 1, n_steps + 1) ** 1.5
    windows_per_step = [int(total_windows * pct) for pct in flip_percentages[1:]]

    # Iteratively flip time windows
    for step, n_windows_to_flip in enumerate(windows_per_step, 1):
        # Get flipped sample
        flipped_sample = input_signal.copy()

        # Get windows to flip
        windows_to_flip = sorted_time_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_time_windows
        window_indices = windows_to_flip % n_time_windows

        # Apply class-specific reference values to time domain
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * time_window_size
            end_idx = min((window_idx + 1) * time_window_size, time_steps)

            # Apply class-specific reference values:
            if target_class == 0:  # Normal/good class
                # Use mild noise
                window_data = input_signal[channel_idx, start_idx:end_idx]
                data_std = np.std(window_data)
                flipped_sample[channel_idx, start_idx:end_idx] = np.random.normal(
                    0, data_std * 0.5, end_idx - start_idx
                )
            else:  # Faulty/bad class
                # Use zero
                flipped_sample[channel_idx, start_idx:end_idx] = 0.0

        # Convert to tensor for model inference
        flipped_tensor = torch.tensor(flipped_sample, dtype=torch.float32, device=device).unsqueeze(0)

        # Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        print(f"Step {step}: Score after flipping {n_windows_to_flip} windows: {score:.4f} ({flipped_pcts[-1]:.1f}%)")

    # Clean up
    del dftlrp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts
def frequency_guided_time_window_flipping_single(model, sample, attribution_method, target_class=None,
                                                 n_steps=20, window_size=40, most_relevant_first=True,
                                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                                 leverage_symmetry=True, sampling_rate=400):
    """
    Perform frequency-guided time window flipping:
    1. Identify important frequency components using attribution method
    2. Map those frequencies back to time domain using EnhancedDFTLRP
    3. Apply class-specific reference values: mild_noise for normal, zero for faulty
    4. Measure impact on model prediction
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()
            print(f"No target class provided, using predicted class: {target_class}")

        original_score = original_prob[target_class].item()
        print(f"Original score for class {target_class}: {original_score:.4f}")

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        target=target_class
    )

    # Create DFT-LRP object for transforms
    dftlrp = EnhancedDFTLRP(
        signal_length=time_steps,
        leverage_symmetry=leverage_symmetry,
        precision=32,
        cuda=(device == "cuda"),
        create_forward=True,
        create_inverse=True,
        create_transpose_inverse=False
    )

    # Determine frequency length based on symmetry
    freq_length = signal_freq.shape[1]
    print(f"Frequency length: {freq_length}, Time steps: {time_steps}")

    # Calculate frequency window importance
    n_freq_windows = freq_length // window_size
    if freq_length % window_size > 0:
        n_freq_windows += 1

    freq_window_importance = np.zeros((n_channels, n_freq_windows))

    for channel in range(n_channels):
        for window_idx in range(n_freq_windows):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute attribution within the window
            freq_window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort frequency window importance
    flat_importance = freq_window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Find corresponding time windows for each important frequency window
    # Each important frequency window maps to multiple time windows

    # Calculate time window size - similar to what we use in time domain
    time_window_size = time_steps // 20  # Reasonable size for time windows
    n_time_windows = time_steps // time_window_size
    if time_steps % time_window_size > 0:
        n_time_windows += 1

    # Map each frequency window to time windows
    # This is a mapping of importance - which time windows are influenced by important frequencies
    time_window_importance = np.zeros((n_channels, n_time_windows))

    # For each channel, calculate how each frequency window affects time windows
    for channel in range(n_channels):
        for freq_window_idx in range(n_freq_windows):
            # Skip if this frequency window has zero importance
            if freq_window_importance[channel, freq_window_idx] == 0:
                continue

            # Get frequency indices for this window
            start_idx = freq_window_idx * window_size
            end_idx = min((freq_window_idx + 1) * window_size, freq_length)

            # Create a frequency domain mask that isolates just this frequency window
            freq_mask = np.zeros_like(signal_freq[channel], dtype=np.complex128)
            freq_mask[start_idx:end_idx] = signal_freq[channel, start_idx:end_idx]

            # Convert to tensor for EnhancedDFTLRP
            # Create a frequency domain mask that isolates just this frequency window
            freq_mask = np.zeros_like(signal_freq[channel], dtype=np.complex128)
            freq_mask[start_idx:end_idx] = signal_freq[channel, start_idx:end_idx]

            # Convert to tensor for EnhancedDFTLRP
            if leverage_symmetry:
                # Need to handle the symmetry properly
                if dftlrp.has_inverse_fourier_layer:
                    # Format for inverse DFT with symmetry
                    # Convert from complex to real representation that dftlrp expects
                    if dftlrp.cuda:
                        # If using CUDA, ensure data is on the correct device
                        freq_real = torch.tensor(np.real(freq_mask), dtype=torch.float32).to(device)

                        # For symmetry, prepare properly - depends on dftlrp's expected format
                        # This is based on how the EnhancedDFTLRP class is implemented
                        if freq_length > 1:
                            if freq_length == time_steps // 2 + 1:
                                # Standard half-spectrum format for real signals
                                freq_imag = torch.tensor(np.imag(freq_mask[1:-1]), dtype=torch.float32).to(device)
                                # Concatenate real and imaginary parts as expected by dftlrp
                                freq_data = torch.cat([freq_real, freq_imag], dim=0).to(device)
                            else:
                                # Handle non-standard spectrum length
                                freq_imag = torch.tensor(np.imag(freq_mask[1:]), dtype=torch.float32).to(device)
                                freq_data = torch.cat([freq_real, freq_imag], dim=0).to(device)
                        else:
                            # Only DC component
                            freq_data = freq_real.to(device)

                        # Use EnhancedDFTLRP for inverse transform, ensuring all tensors are on same device
                        try:
                            time_tensor = dftlrp.inverse_fourier_layer(freq_data.unsqueeze(0))
                            time_contribution = time_tensor.cpu().numpy().squeeze(0)
                        except RuntimeError as e:
                            print(f"Error using dftlrp inverse: {e}")
                            print("Falling back to torch.fft.irfft")
                            # Fallback to torch's irfft
                            freq_complex = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                            time_tensor = torch.fft.irfft(freq_complex, n=time_steps)
                            time_contribution = time_tensor.cpu().numpy()
                    else:
                        # CPU path
                        # Fallback to numpy's irfft for CPU mode
                        time_contribution = np.fft.irfft(freq_mask, n=time_steps)
                else:
                    # Fallback to numpy
                    time_contribution = np.fft.irfft(freq_mask, n=time_steps)
            else:
                # Without symmetry
                if dftlrp.has_inverse_fourier_layer:
                    freq_tensor = torch.tensor(freq_mask, dtype=torch.complex64).to(device)
                    time_contribution = torch.fft.ifft(freq_tensor, n=time_steps).real.cpu().numpy()
                else:
                    time_contribution = np.fft.ifft(freq_mask, n=time_steps).real

            # Calculate which time windows are affected by this frequency
            # Weight by both frequency importance and time domain magnitude
            time_importance = np.abs(time_contribution) * freq_window_importance[channel, freq_window_idx]

            # Accumulate importance for each time window
            for time_window_idx in range(n_time_windows):
                start_t = time_window_idx * time_window_size
                end_t = min((time_window_idx + 1) * time_window_size, time_steps)

                # Sum importance within this window
                window_importance = np.sum(time_importance[start_t:end_t])
                time_window_importance[channel, time_window_idx] += window_importance

    # Normalize time window importance for each channel
    for channel in range(n_channels):
        if np.max(time_window_importance[channel]) > 0:
            time_window_importance[channel] /= np.max(time_window_importance[channel])

    # Flatten and sort time window importance
    flat_time_importance = time_window_importance.flatten()
    sorted_time_indices = np.argsort(flat_time_importance)

    if most_relevant_first:
        sorted_time_indices = sorted_time_indices[::-1]

    # Track model outputs
    scores = [original_score]
    flipped_pcts = [0.0]

    # Calculate windows to flip per step
    total_windows = n_channels * n_time_windows

    # Create exponential progression for more gradual steps
    flip_percentages = np.linspace(0, 1, n_steps + 1) ** 1.5
    windows_per_step = [int(total_windows * pct) for pct in flip_percentages[1:]]

    # Iteratively flip time windows
    for step, n_windows_to_flip in enumerate(windows_per_step, 1):
        # Get flipped sample
        flipped_sample = input_signal.copy()

        # Get windows to flip
        windows_to_flip = sorted_time_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_time_windows
        window_indices = windows_to_flip % n_time_windows

        # Apply class-specific reference values to time domain
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * time_window_size
            end_idx = min((window_idx + 1) * time_window_size, time_steps)

            # Apply class-specific reference values:
            if target_class == 0:  # Normal/good class
                # Use mild noise
                window_data = input_signal[channel_idx, start_idx:end_idx]
                data_std = np.std(window_data)
                flipped_sample[channel_idx, start_idx:end_idx] = np.random.normal(
                    0, data_std * 0.5, end_idx - start_idx
                )
            else:  # Faulty/bad class
                # Use zero
                flipped_sample[channel_idx, start_idx:end_idx] = 0.0

        # Convert to tensor for model inference
        flipped_tensor = torch.tensor(flipped_sample, dtype=torch.float32, device=device).unsqueeze(0)

        # Get model output for flipped sample
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Track results
        scores.append(score)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        print(f"Step {step}: Score after flipping {n_windows_to_flip} windows: {score:.4f} ({flipped_pcts[-1]:.1f}%)")

    # Clean up
    del dftlrp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return scores, flipped_pcts

def frequency_window_flipping_batch(model, samples, attribution_methods, n_steps=10,
                                    window_size=10, most_relevant_first=True,
                                    reference_value="complete_zero", max_samples=None,
                                    leverage_symmetry=True, sampling_rate=400,
                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform frequency domain window flipping analysis on a batch of time series samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz
        device: Device to run computations on

    Returns:
        results: Dictionary with results for each method and sample
    """
    # Initialize results storage
    results = {method_name: [] for method_name in attribution_methods}

    # Keep track of the current sample count
    sample_count = 0

    # Process each sample
    for sample_idx, (sample_data, target) in enumerate(tqdm(samples, desc="Processing samples")):
        # Skip if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break

        # Move sample to device
        sample = sample_data.to(device)
        target_class = target

        # Increment sample counter
        sample_count += 1

        # Print progress every 10 samples
        if sample_count % 10 == 0:
            print(f"Processing sample {sample_count}/{len(samples) if max_samples is None else max_samples}")

        # Process each attribution method
        for method_name, attribution_func in attribution_methods.items():
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                # Compute scores for this sample using the current method
                # Compute scores for this sample using the current method
                scores, flipped_pcts = frequency_window_flipping_single(
                    model=model,
                    sample=sample,
                    attribution_method=attribution_func,
                    target_class=target_class,  # Pass the target class correctly
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

                # Store results
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": scores,
                    "flipped_pcts": flipped_pcts,
                    "auc": auc
                })

            except Exception as e:
                print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                # Store empty result to maintain sample count consistency
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results


def frequency_guided_time_window_flipping_batch(model, samples, attribution_methods, n_steps=20,
                                                window_size=40, most_relevant_first=True,
                                                max_samples=None, leverage_symmetry=True,
                                                sampling_rate=400,
                                                device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Perform frequency-guided time window flipping on a batch of time series samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        max_samples: Maximum number of samples to process (None = all samples)
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz
        device: Device to run computations on

    Returns:
        results: Dictionary with aggregated results for each method
    """
    # Initialize results storage
    results = {method_name: [] for method_name in attribution_methods}

    # Keep track of the current sample count
    sample_count = 0

    # Process each sample
    for sample_idx, (sample_data, target) in enumerate(tqdm(samples, desc="Processing samples")):
        # Skip if we've reached max_samples
        if max_samples is not None and sample_count >= max_samples:
            break

        # Move sample to device
        sample = sample_data.to(device)
        target_class = target

        # Increment sample counter
        sample_count += 1

        # Print progress every 10 samples
        if sample_count % 10 == 0:
            print(f"Processing sample {sample_count}/{len(samples) if max_samples is None else max_samples}")

        # Process each attribution method
        for method_name, attribution_func in attribution_methods.items():
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            try:
                # Compute scores for this sample using the current method
                print(f"Processing sample {sample_count}, method: {method_name}")
                scores, flipped_pcts = improved_frequency_guided_time_window_flipping_single(
                    model=model,
                    sample=sample,
                    attribution_method=attribution_func,  # Use the specific function, not the dictionary
                    target_class=target_class,
                    n_steps=n_steps,
                    window_size=window_size,
                    most_relevant_first=most_relevant_first,
                    device=device,
                    leverage_symmetry=leverage_symmetry,
                    sampling_rate=sampling_rate
                )

                # Store results
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": scores,
                    "flipped_pcts": flipped_pcts,
                    "auc": np.trapz(scores, flipped_pcts) / flipped_pcts[-1]
                })

            except Exception as e:
                print(f"Error processing sample {sample_count - 1} with method {method_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Store empty result to maintain sample count consistency
                results[method_name].append({
                    "sample_idx": sample_count - 1,
                    "target": target_class,
                    "scores": None,
                    "flipped_pcts": None,
                    "auc": float('nan')
                })

    return results
def aggregate_results(results):
    """
    Aggregate window flipping results across all samples.

    Args:
        results: Dictionary with results for each method and sample

    Returns:
        agg_results: Dictionary with aggregated results
    """
    agg_results = {}

    for method_name, samples in results.items():
        # Initialize storage for this method
        method_results = {
            "avg_scores": [],
            "flipped_pcts": None,
            "auc_values": [],
            "mean_auc": None,
            "std_auc": None,
            "class_specific": {}  # Add class-specific metrics
        }

        # Filter out samples with errors
        valid_samples = [s for s in samples if s["scores"] is not None]

        if not valid_samples:
            print(f"No valid samples for method {method_name}")
            agg_results[method_name] = method_results
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

            # Calculate class-specific metrics
            class_specific = {}

            # Check for class 0 (normal) samples
            class_0_samples = [s for s in valid_samples if s["target"] == 0]
            if class_0_samples:
                class_0_aucs = [s["auc"] for s in class_0_samples if not np.isnan(s["auc"])]
                class_specific["class_0_metrics"] = {
                    "mean_auc": np.mean(class_0_aucs),
                    "std_auc": np.std(class_0_aucs),
                    "count": len(class_0_samples)
                }

                # Interpolate class 0 scores
                class_0_scores = []
                for sample in class_0_samples:
                    if len(sample["flipped_pcts"]) > 1 and len(sample["scores"]) > 1:
                        if np.all(np.diff(sample["flipped_pcts"]) > 0):
                            f = interp1d(sample["flipped_pcts"], sample["scores"], bounds_error=False,
                                         fill_value="extrapolate")
                            class_0_scores.append(f(common_x))

                if class_0_scores:
                    class_specific["class_0"] = {"avg_scores": np.mean(class_0_scores, axis=0)}
                    class_specific["class_0"]["auc"] = np.trapz(class_specific["class_0"]["avg_scores"],
                                                                common_x) / 100.0

            # Check for class 1 (faulty) samples
            class_1_samples = [s for s in valid_samples if s["target"] == 1]
            if class_1_samples:
                class_1_aucs = [s["auc"] for s in class_1_samples if not np.isnan(s["auc"])]
                class_specific["class_1_metrics"] = {
                    "mean_auc": np.mean(class_1_aucs),
                    "std_auc": np.std(class_1_aucs),
                    "count": len(class_1_samples)
                }

                # Interpolate class 1 scores
                class_1_scores = []
                for sample in class_1_samples:
                    if len(sample["flipped_pcts"]) > 1 and len(sample["scores"]) > 1:
                        if np.all(np.diff(sample["flipped_pcts"]) > 0):
                            f = interp1d(sample["flipped_pcts"], sample["scores"], bounds_error=False,
                                         fill_value="extrapolate")
                            class_1_scores.append(f(common_x))

                if class_1_scores:
                    class_specific["class_1"] = {"avg_scores": np.mean(class_1_scores, axis=0)}
                    class_specific["class_1"]["auc"] = np.trapz(class_specific["class_1"]["avg_scores"],
                                                                common_x) / 100.0

            # Store results
            method_results["avg_scores"] = avg_scores
            method_results["flipped_pcts"] = common_x
            method_results["auc_values"] = auc_values
            method_results["mean_auc"] = mean_auc
            method_results["std_auc"] = std_auc
            method_results["class_specific"] = class_specific

        except Exception as e:
            print(f"Error aggregating results for {method_name}: {e}")
            method_results["avg_scores"] = None
            method_results["flipped_pcts"] = None
            method_results["auc_values"] = auc_values
            method_results["mean_auc"] = mean_auc
            method_results["std_auc"] = std_auc
            method_results["class_specific"] = {}

        # Store results for this method
        agg_results[method_name] = method_results

    return agg_results


def plot_aggregate_results(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot aggregated window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

    Returns:
        fig: Figure object
    """
    # Close any existing figures first to avoid white plots
    plt.close('all')

    # Create new figure
    fig = plt.figure(figsize=(12, 8))

    for method_name, results in agg_results.items():
        if results["avg_scores"] is None or len(results["avg_scores"]) == 0:
            print(f"Skipping {method_name} - no valid data")
            continue

        # Plot average scores with confidence interval
        plt.plot(results["flipped_pcts"], results["avg_scores"],
                 label=f"{method_name} (AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f})")

    flip_order = "most important" if most_relevant_first else "least important"
    plt.title(
        f'Aggregate Frequency Window Flipping Results\n(Flipping {flip_order} windows first, \n Reference Value Method: {reference_value})',
        fontsize=16)
    plt.xlabel('Percentage of Frequency Windows Flipped (%)', fontsize=14)
    plt.ylabel('Average Prediction Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    return fig


def plot_class_specific_results(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot class-specific window flipping results.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

    Returns:
        fig: Figure object
    """
    # Close any existing figures first
    plt.close('all')

    # Set up the figure - one plot per method
    method_count = len(agg_results)
    fig, axes = plt.subplots(1, method_count, figsize=(6 * method_count, 6), squeeze=False)

    for i, (method_name, results) in enumerate(agg_results.items()):
        ax = axes[0, i]

        # Check if we have class-specific results
        if "class_specific" in results and results["class_specific"]:
            class_specific = results["class_specific"]

            # Plot for class 0 if available
            if "class_0" in class_specific and "avg_scores" in class_specific["class_0"]:
                ax.plot(results["flipped_pcts"], class_specific["class_0"]["avg_scores"],
                        label=f"Normal (AUC: {class_specific['class_0']['auc']:.4f})",
                        color='blue')

            # Plot for class 1 if available
            if "class_1" in class_specific and "avg_scores" in class_specific["class_1"]:
                ax.plot(results["flipped_pcts"], class_specific["class_1"]["avg_scores"],
                        label=f"Faulty (AUC: {class_specific['class_1']['auc']:.4f})",
                        color='red')

        # If no class-specific data, just plot the overall result if available
        elif results["avg_scores"] is not None and len(results["avg_scores"]) > 0:
            mean_auc = results.get("mean_auc")
            if mean_auc is not None:
                ax.plot(results["flipped_pcts"], results["avg_scores"],
                        label=f"Overall (AUC: {mean_auc:.4f})")
            else:
                ax.plot(results["flipped_pcts"], results["avg_scores"],
                        label="Overall (AUC: N/A)")

        ax.set_title(f'{method_name}')
        ax.set_xlabel('Percentage of Windows Flipped (%)')
        ax.set_ylabel('Prediction Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax.legend()

    flip_order = "most important" if most_relevant_first else "least important"
    fig.suptitle(
        f'Class-Specific Frequency-Guided Window Flipping Results\n(Flipping {flip_order} windows first)',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    return fig

def plot_auc_by_class(agg_results, most_relevant_first=True, reference_value='complete_zero'):
    """
    Plot AUC values separated by class.

    Args:
        agg_results: Dictionary with aggregated results for each method
        most_relevant_first: Whether most relevant windows were flipped first

    Returns:
        fig: Figure object
    """
    # Close any existing figures
    plt.close('all')

    # Prepare data
    methods = []
    class0_aucs = []
    class0_counts = []
    class1_aucs = []
    class1_counts = []

    for method_name, results in agg_results.items():
        if "class_specific" in results:
            class_specific = results["class_specific"]

            if "class_0_metrics" in class_specific:
                methods.append(method_name)
                class0_aucs.append(class_specific["class_0_metrics"]["mean_auc"])
                class0_counts.append(class_specific["class_0_metrics"]["count"])

                if "class_1_metrics" in class_specific:
                    class1_aucs.append(class_specific["class_1_metrics"]["mean_auc"])
                    class1_counts.append(class_specific["class_1_metrics"]["count"])
                else:
                    class1_aucs.append(0)
                    class1_counts.append(0)

    if not methods:
        print("No class-specific AUC data available")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.35

    rects1 = ax.bar(x - width / 2, class0_aucs, width, label=f'Normal (n={class0_counts[0]})', color='lightgreen')
    rects2 = ax.bar(x + width / 2, class1_aucs, width, label=f'Faulty (n={class1_counts[0]})', color='salmon')

    ax.set_ylabel('Mean AUC')
    flip_order = "most important" if most_relevant_first else "least important"
    ax.set_title(f'AUC by Class for Different Attribution Methods\n'
                 f'(Flipping {flip_order} frequency windows first, \n Reference Value Method: {reference_value})')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Add text labels
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()

    return fig


def extract_important_freq_windows(model, sample, attribution_method, target_class=None,
                                   n_windows=10, window_size=10,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   leverage_symmetry=True, sampling_rate=400):
    """
    Extract the most important windows from frequency domain based on attribution scores.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (channels, time_steps)
        attribution_method: Function that generates attributions
        target_class: Target class for explanation (default: model's prediction)
        n_windows: Number of top windows to extract
        window_size: Size of frequency windows
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz

    Returns:
        important_windows: List of dictionaries containing window info
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    n_channels, time_steps = sample.shape

    # Get original prediction
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]

        if target_class is None:
            target_class = torch.argmax(original_prob).item()

    # Get frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        target=target_class# Changed from 'label' to 'target_class'

    )

    # Determine frequency length
    freq_length = relevance_freq.shape[1]

    # Calculate window importance
    n_windows_per_channel = freq_length // window_size
    if freq_length % window_size > 0:
        n_windows_per_channel += 1

    window_importance = np.zeros((n_channels, n_windows_per_channel))

    for channel in range(n_channels):
        for window_idx in range(n_windows_per_channel):
            start_idx = window_idx * window_size
            end_idx = min((window_idx + 1) * window_size, freq_length)

            # Average absolute attribution within the window
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)[::-1]  # Descending order

    # Take top n_windows indices
    top_indices = sorted_indices[:n_windows]

    # Convert flat indices to channel, window indices
    channel_indices = top_indices // n_windows_per_channel
    window_indices = top_indices % n_windows_per_channel

    # Extract important windows
    important_windows = []

    for i in range(len(top_indices)):
        channel_idx = channel_indices[i]
        window_idx = window_indices[i]

        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, freq_length)

        # Extract frequency data for this window
        freq_data = signal_freq[channel_idx, start_idx:end_idx]

        # Calculate frequency range for this window in Hz
        freq_range = (freqs[start_idx], freqs[end_idx - 1]) if len(freqs) >= end_idx else (0, 0)

        # Calculate statistics for this window
        magnitude = np.abs(freq_data)
        phase = np.angle(freq_data)

        window_info = {
            'freq_data': freq_data,
            'magnitude': magnitude,
            'phase': phase,
            'freq_start': freq_range[0],
            'freq_end': freq_range[1],
            'channel': channel_idx,
            'relevance': flat_importance[top_indices[i]],
            'window_idx': window_idx,
            'avg_magnitude': np.mean(magnitude),
            'max_magnitude': np.max(magnitude),
            'avg_phase': stats.circmean(phase),
            'phase_std': stats.circstd(phase),
            'window_size': end_idx - start_idx
        }

        # Find peak frequency in this window
        if len(magnitude) > 0:
            peak_idx = np.argmax(magnitude)
            window_info['peak_freq_hz'] = freqs[start_idx + peak_idx]

            # Calculate spectral centroid (frequency center of mass)
            if np.sum(magnitude) > 0:  # Avoid division by zero
                freqs_window = freqs[start_idx:end_idx]
                window_info['spectral_centroid'] = np.sum(freqs_window * magnitude) / np.sum(magnitude)

                # Calculate spectral bandwidth (spread around centroid)
                centroid = window_info['spectral_centroid']
                window_info['spectral_bandwidth'] = np.sqrt(
                    np.sum(((freqs_window - centroid) ** 2) * magnitude) / np.sum(magnitude)
                )

                # Calculate spectral flatness (ratio of geometric mean to arithmetic mean)
                # Higher values indicate more noise-like signal
                if np.all(magnitude > 0):  # Avoid log of zero
                    geo_mean = np.exp(np.mean(np.log(magnitude)))
                    arith_mean = np.mean(magnitude)
                    window_info['spectral_flatness'] = geo_mean / arith_mean if arith_mean > 0 else 0

        important_windows.append(window_info)

    return important_windows


def collect_important_freq_windows(model, samples, attribution_method,
                                   n_samples_per_class=10, n_windows=10, window_size=10,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   leverage_symmetry=True, sampling_rate=400):
    """
    Collect important frequency windows from multiple samples.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions
        n_samples_per_class: Number of samples to process per class
        n_windows: Number of top windows to extract per sample
        window_size: Size of frequency windows
        device: Device to run computations on
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz

    Returns:
        all_windows: List of dictionaries containing window info
    """
    all_windows = []

    # Separate samples by class
    class_0_samples = [(s, t) for s, t in samples if t == 0]
    class_1_samples = [(s, t) for s, t in samples if t == 1]

    # Process class 0 samples
    n_class_0 = min(n_samples_per_class, len(class_0_samples))
    for i in range(n_class_0):
        sample, target = class_0_samples[i]

        try:
            windows = extract_important_freq_windows(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                device=device,
                leverage_symmetry=leverage_symmetry,
                sampling_rate=sampling_rate
            )

            # Add sample metadata to each window
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 0
                window['class_name'] = 'normal'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing normal sample {i}: {str(e)}")

    # Process class 1 samples
    n_class_1 = min(n_samples_per_class, len(class_1_samples))
    for i in range(n_class_1):
        sample, target = class_1_samples[i]

        try:
            windows = extract_important_freq_windows(
                model=model,
                sample=sample,
                attribution_method=attribution_method,
                target_class=target,
                n_windows=n_windows,
                window_size=window_size,
                device=device,
                leverage_symmetry=leverage_symmetry,
                sampling_rate=sampling_rate
            )

            # Add sample metadata to each window
            for window in windows:
                window['sample_idx'] = i
                window['class'] = 1
                window['class_name'] = 'faulty'

            all_windows.extend(windows)

        except Exception as e:
            print(f"Error processing faulty sample {i}: {str(e)}")

    return all_windows


def visualize_freq_windows(windows, output_dir="./results/freq_windows"):
    """
    Visualize the important frequency windows.

    Args:
        windows: List of window dictionaries from extract_important_freq_windows
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group windows by class
    normal_windows = [w for w in windows if w['class'] == 0]
    faulty_windows = [w for w in windows if w['class'] == 1]

    # Create a scatter plot of frequency vs. magnitude, colored by relevance
    plt.close('all')
    plt.figure(figsize=(12, 10))

    # Plot normal windows
    plt.subplot(2, 1, 1)
    relevances = [w['relevance'] for w in normal_windows]
    peak_freqs = [w['peak_freq_hz'] for w in normal_windows]
    magnitudes = [w['avg_magnitude'] for w in normal_windows]

    plt.scatter(peak_freqs, magnitudes, c=relevances, cmap='viridis',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Frequency Windows - Normal Samples')
    plt.xlabel('Peak Frequency (Hz)')
    plt.ylabel('Average Magnitude')
    plt.grid(alpha=0.3)
    plt.xscale('log')  # Log scale for frequency

    # Plot faulty windows
    plt.subplot(2, 1, 2)
    relevances = [w['relevance'] for w in faulty_windows]
    peak_freqs = [w['peak_freq_hz'] for w in faulty_windows]
    magnitudes = [w['avg_magnitude'] for w in faulty_windows]

    plt.scatter(peak_freqs, magnitudes, c=relevances, cmap='plasma',
                alpha=0.7, s=100, edgecolors='w')
    plt.colorbar(label='Relevance')
    plt.title('Important Frequency Windows - Faulty Samples')
    plt.xlabel('Peak Frequency (Hz)')
    plt.ylabel('Average Magnitude')
    plt.grid(alpha=0.3)
    plt.xscale('log')  # Log scale for frequency

    plt.tight_layout()
    plt.savefig(f"{output_dir}/freq_windows_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create histograms of peak frequencies
    plt.figure(figsize=(12, 6))
    plt.close('all')

    plt.hist([w['peak_freq_hz'] for w in normal_windows], bins=20, alpha=0.5,
             label='Normal', density=True)
    plt.hist([w['peak_freq_hz'] for w in faulty_windows], bins=20, alpha=0.5,
             label='Faulty', density=True)

    plt.title('Distribution of Peak Frequencies in Important Windows')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log')  # Log scale for frequency

    plt.savefig(f"{output_dir}/peak_freq_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create channel distribution plot
    plt.close('all')
    plt.figure(figsize=(10, 6))

    # Count windows by channel for each class
    channels = sorted(set(w['channel'] for w in windows))

    normal_counts = [len([w for w in normal_windows if w['channel'] == c]) for c in channels]
    faulty_counts = [len([w for w in faulty_windows if w['channel'] == c]) for c in channels]

    x = np.arange(len(channels))
    width = 0.35

    plt.bar(x - width / 2, normal_counts, width, label='Normal')
    plt.bar(x + width / 2, faulty_counts, width, label='Faulty')

    plt.xlabel('Channel')
    plt.ylabel('Count')
    plt.title('Channel Distribution of Important Frequency Windows')
    plt.xticks(x, [f'Channel {c}' for c in channels])
    plt.legend()

    plt.savefig(f"{output_dir}/channel_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_freq_windows(windows, output_dir="./results/freq_windows"):
    """
    Analyze the important frequency windows and identify distinguishing features.

    Args:
        windows: List of window dictionaries
        output_dir: Directory to save results
    """
    import pandas as pd
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{k: v for k, v in w.items() if not isinstance(v, np.ndarray)}
                       for w in windows])

    # Add derived features
    if len(df) > 0:
        df['magnitude_std'] = [np.std(w['magnitude']) for w in windows]
        df['peak_to_avg_ratio'] = df['max_magnitude'] / df['avg_magnitude'].replace(0, np.nan)

        # Calculate spectral energy
        for i, window in enumerate(windows):
            if 'magnitude' in window:
                df.loc[i, 'spectral_energy'] = np.sum(window['magnitude'] ** 2)

                # Calculate spectral entropy
                magnitude = window['magnitude']
                if np.sum(magnitude) > 0:
                    psd = magnitude / np.sum(magnitude)
                    psd_clean = psd[psd > 0]  # Remove zeros
                    if len(psd_clean) > 0:
                        df.loc[i, 'spectral_entropy'] = -np.sum(psd_clean * np.log2(psd_clean))

    # Analyze class differences
    features = ['peak_freq_hz', 'avg_magnitude', 'max_magnitude', 'magnitude_std',
                'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness',
                'spectral_energy', 'spectral_entropy', 'peak_to_avg_ratio',
                'avg_phase', 'phase_std']

    # Filter out features not in the DataFrame
    features = [f for f in features if f in df.columns]

    class_stats = []
    for feature in features:
        normal_values = df[df['class'] == 0][feature].dropna()
        faulty_values = df[df['class'] == 1][feature].dropna()

        # Calculate separation statistics
        if len(normal_values) > 0 and len(faulty_values) > 0:
            try:
                t_stat, p_value = stats.ttest_ind(normal_values, faulty_values, equal_var=False)

                class_stats.append({
                    'feature': feature,
                    't_statistic': abs(t_stat),
                    'p_value': p_value,
                    'normal_mean': normal_values.mean(),
                    'faulty_mean': faulty_values.mean(),
                    'normal_std': normal_values.std(),
                    'faulty_std': faulty_values.std(),
                    'difference_pct': ((faulty_values.mean() - normal_values.mean()) /
                                       normal_values.mean() * 100) if normal_values.mean() != 0 else 0
                })
            except Exception as e:
                print(f"Error calculating statistics for {feature}: {e}")

    # Sort by statistical significance
    if class_stats:
        class_stats.sort(key=lambda x: x['p_value'])
        class_stats_df = pd.DataFrame(class_stats)

        # Save results
        class_stats_df.to_csv(f"{output_dir}/freq_feature_stats.csv", index=False)
        df.to_csv(f"{output_dir}/freq_windows_data.csv", index=False)

        # Create summary visualizations
        plt.close('all')
        plt.figure(figsize=(12, 8))

        # Plot top 5 most discriminative features
        top_features = class_stats_df.head(5)['feature'].tolist()

        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i + 1)

            sns.boxplot(x='class_name', y=feature, data=df)
            plt.title(f"{feature}\np={class_stats_df[class_stats_df['feature'] == feature]['p_value'].values[0]:.4f}")

            if i >= 2:
                plt.xlabel('Class')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/top_discriminative_features.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create feature correlation matrix
        plt.close('all')
        plt.figure(figsize=(12, 10))

        # Filter numeric features
        numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]

        if numeric_features:
            corr = df[numeric_features].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Matrix')

            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()

        return class_stats_df
    else:
        print("No statistics could be calculated")
        return None


def load_balanced_samples(data_dir, n_samples_per_class=10):
    """
    Load balanced samples from good and bad subfolders.

    Args:
        data_dir: Path to data directory
        n_samples_per_class: Number of samples to load per class

    Returns:
        samples: List of (sample, target) tuples
    """
    data_dir = Path(data_dir)
    samples = []

    # Load samples from good subfolder
    good_dir = data_dir / "good"
    good_files = list(good_dir.glob("*.h5"))

    if len(good_files) < n_samples_per_class:
        print(f"Warning: Only {len(good_files)} good samples available, requested {n_samples_per_class}")

    # Randomly sample
    selected_good = random.sample(good_files, min(n_samples_per_class, len(good_files)))

    # Load good samples
    for file_path in selected_good:
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3) for downsampled

        # Transpose to (3, 2000) for CNN
        data = np.transpose(data, (1, 0))
        samples.append((torch.tensor(data, dtype=torch.float32), 0))  # 0 = good class

    # Load samples from bad subfolder
    bad_dir = data_dir / "bad"
    bad_files = list(bad_dir.glob("*.h5"))

    if len(bad_files) < n_samples_per_class:
        print(f"Warning: Only {len(bad_files)} bad samples available, requested {n_samples_per_class}")

    # Randomly sample
    selected_bad = random.sample(bad_files, min(n_samples_per_class, len(bad_files)))

    # Load bad samples
    for file_path in selected_bad:
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3) for downsampled

        # Transpose to (3, 2000) for CNN
        data = np.transpose(data, (1, 0))
        samples.append((torch.tensor(data, dtype=torch.float32), 1))  # 1 = bad class

    print(f"Loaded {len(selected_good)} good samples and {len(selected_bad)} bad samples")
    return samples


def save_results_to_csv(agg_results, results, filename_prefix):
    """
    Save results to CSV files.

    Args:
        agg_results: Dictionary with aggregated results
        results: Dictionary with individual sample results
        filename_prefix: Prefix for filenames
    """
    # Save aggregated results
    agg_data = []
    for method_name, method_results in agg_results.items():
        # Basic method metrics
        method_data = {
            "Method": method_name,
            "Mean_AUC": method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan'),
            "Std_AUC": method_results["std_auc"] if method_results["std_auc"] is not None else float('nan'),
            "Num_Samples": len(method_results["auc_values"])
        }

        # Add class-specific metrics if available
        if "class_specific" in method_results:
            class_specific = method_results["class_specific"]

            # Add class-0 metrics
            if "class_0_metrics" in class_specific:
                method_data["Class0_AUC"] = class_specific["class_0_metrics"]["mean_auc"]
                method_data["Class0_Count"] = class_specific["class_0_metrics"]["count"]

            # Add class-1 metrics
            if "class_1_metrics" in class_specific:
                method_data["Class1_AUC"] = class_specific["class_1_metrics"]["mean_auc"]
                method_data["Class1_Count"] = class_specific["class_1_metrics"]["count"]

        agg_data.append(method_data)

    # Save to CSV
    pd.DataFrame(agg_data).to_csv(f"{filename_prefix}_aggregate.csv", index=False)

    # Save detailed sample results
    samples_data = []
    for method_name, samples in results.items():
        for sample in samples:
            if sample["scores"] is None:
                continue

            # Basic sample data
            sample_data = {
                "Method": method_name,
                "Sample_Index": sample["sample_idx"],
                "True_Class": sample["target"],
                "AUC": sample["auc"]
            }

            samples_data.append(sample_data)

    if samples_data:
        pd.DataFrame(samples_data).to_csv(f"{filename_prefix}_samples.csv", index=False)


def run_frequency_window_flipping_evaluation(model, samples, attribution_methods,
                                             n_steps=10, window_size=10,
                                             most_relevant_first=True, reference_value="complete_zero",
                                             leverage_symmetry=True, sampling_rate=400,
                                             max_samples=None, output_dir="./results",
                                             device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency domain window flipping evaluation on test set and save results.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of frequency windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate in Hz
        max_samples: Maximum number of samples to process (None = all)
        output_dir: Directory to save results
        device: Device to run computations on

    Returns:
        results: Dictionary with individual sample results
        agg_results: Dictionary with aggregated results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/freq_window_flipping_{flip_order}_{timestamp}"

    print(f"Starting frequency domain window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}")

    # Run window flipping on all samples
    results = frequency_window_flipping_batch(
        model=model,
        samples=samples,
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

    print("Computing aggregated results...")

    # Aggregate results
    agg_results = aggregate_results(results)

    print("Creating visualizations...")

    # Plot and save aggregated results
    plt.close('all')
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first, reference_value)
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Plot and save AUC by class
    plt.close('all')
    auc_fig = plot_auc_by_class(agg_results, most_relevant_first, reference_value)
    if auc_fig:
        auc_fig.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
        plt.close(auc_fig)

    # Save numerical results to CSV
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


def run_frequency_guided_window_flipping_evaluation(model, samples, attribution_methods,
                                                    n_steps=20, window_size=40,
                                                    most_relevant_first=True, reference_value="zero",
                                                    max_samples=None, output_dir="./results",
                                                    leverage_symmetry=True, sampling_rate=400,
                                                    device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run complete frequency-guided time window flipping evaluation.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_methods: Dictionary of {method_name: attribution_function}
        n_steps: Number of steps to divide the flipping process
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        reference_value: Value to replace flipped windows
        max_samples: Maximum number of samples to process (None = all)
        output_dir: Directory to save results
        leverage_symmetry: Whether to use symmetry in DFT
        sampling_rate: Sampling rate of the signal in Hz
        device: Device to run computations on

    Returns:
        results: Dictionary with individual sample results
        agg_results: Dictionary with aggregated results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flip_order = "most_first" if most_relevant_first else "least_first"
    filename_prefix = f"{output_dir}/freq_guided_window_flipping_{flip_order}_{reference_value}_{timestamp}"

    print(f"Starting frequency-guided time window flipping evaluation with {len(attribution_methods)} methods")
    print(f"Settings: n_steps={n_steps}, window_size={window_size}, most_relevant_first={most_relevant_first}")
    print(f"Reference value: {reference_value}, leverage_symmetry={leverage_symmetry}, sampling_rate={sampling_rate}")

    # Run window flipping on all samples
    results = frequency_guided_time_window_flipping_batch(
        model=model,
        samples=samples,
        attribution_methods=attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=most_relevant_first,
        max_samples=max_samples,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        device=device
    )

    print("Computing aggregated results...")

    # Aggregate results
    agg_results = aggregate_results(results)

    print("Creating visualizations...")

    # Plot and save aggregated results
    plt.close('all')
    agg_fig = plot_aggregate_results(agg_results, most_relevant_first, reference_value=reference_value)
    agg_fig.savefig(f"{filename_prefix}_aggregate_plot.png", dpi=300, bbox_inches='tight')
    plt.close(agg_fig)

    # Plot and save class-specific results
    plt.close('all')
    class_fig = plot_class_specific_results(agg_results, most_relevant_first, reference_value=reference_value)
    class_fig.savefig(f"{filename_prefix}_class_specific.png", dpi=300, bbox_inches='tight')
    plt.close(class_fig)

    # Plot and save AUC by class
    plt.close('all')
    auc_fig = plot_auc_by_class(agg_results, most_relevant_first, reference_value=reference_value)
    if auc_fig:
        auc_fig.savefig(f"{filename_prefix}_auc_by_class.png", dpi=300, bbox_inches='tight')
        plt.close(auc_fig)

    # Save numerical results to CSV
    save_results_to_csv(agg_results, results, filename_prefix)

    print(f"Results saved with prefix: {filename_prefix}")

    # Print summary
    print("\nFrequency-Guided Time Window Flipping Evaluation Summary:")
    print("-" * 60)
    print(f"{'Method':<20} {'Mean AUC':<10} {'Std Dev':<10} {'Samples':<10}")
    print("-" * 60)

    for method_name, method_results in agg_results.items():
        mean_auc = method_results["mean_auc"] if method_results["mean_auc"] is not None else float('nan')
        std_auc = method_results["std_auc"] if method_results["std_auc"] is not None else float('nan')
        n_samples = len(method_results["auc_values"])

        print(f"{method_name:<20} {mean_auc:<10.4f} {std_auc:<10.4f} {n_samples:<10}")

    return results, agg_results


'''
def visualize_frequency_window_flipping_sample(model, sample, target, attribution_method,
                                               n_steps=5, window_size=10, most_relevant_first=True,
                                               reference_value="complete_zero", save_path=None,
                                               leverage_symmetry=True, sampling_rate=400,
                                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of frequency domain window flipping for a single time series sample.
    Fixed implementation to properly display frequency range and use proper flipping steps.
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        target=target
    )

    # Determine frequency length
    freq_length = relevance_freq.shape[1]

    # Set reference value for frequency domain
    reference_freq = np.zeros_like(signal_freq)

    # Calculate window importance by averaging relevance within each window
    effective_window_size = max(1, min(window_size, freq_length // 20))
    n_windows = freq_length // effective_window_size
    if freq_length % effective_window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * effective_window_size
            end_idx = min((window_idx + 1) * effective_window_size, freq_length)

            # Average absolute attribution within the window
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = target
        original_score = original_prob[target_class].item()

    # Calculate windows to flip per step - use exponential progression for more gradual steps
    total_windows = n_channels * n_windows

    # Store flipped samples and scores
    flipped_samples = [input_signal.copy()]  # Time domain
    flipped_freqs = [signal_freq.copy()]  # Frequency domain
    scores = [original_score]
    flipped_windows_count = [0]
    flipped_pcts = [0.0]
    flipped_windows_masks = [np.zeros((n_channels, freq_length), dtype=bool)]

    # Determine steps for visualization
    if n_steps > 5:
        # If many steps, select a representative subset
        step_indices = np.linspace(1, total_windows, n_steps, dtype=int)
    else:
        # Use exponential progression
        step_indices = [int(total_windows * ((i + 1) / n_steps) ** 1.5) for i in range(n_steps)]
        step_indices = [min(s, total_windows) for s in step_indices]

    # Iteratively flip windows
    for step_idx, n_windows_to_flip in enumerate(step_indices):
        # Get flipped sample in frequency domain
        flipped_freq = signal_freq.copy()
        flipped_mask = np.zeros((n_channels, freq_length), dtype=bool)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * effective_window_size
            end_idx = min((window_idx + 1) * effective_window_size, freq_length)

            flipped_freq[channel_idx, start_idx:end_idx] = reference_freq[channel_idx, start_idx:end_idx]
            flipped_mask[channel_idx, start_idx:end_idx] = True

        # Transform back to time domain using numpy IFFT
        flipped_time = np.zeros_like(input_signal)
        for c in range(n_channels):
            if leverage_symmetry:
                flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
            else:
                flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # Get model output
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Store results
        flipped_samples.append(flipped_time)
        flipped_freqs.append(flipped_freq)
        scores.append(score)
        flipped_windows_count.append(n_windows_to_flip)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        flipped_windows_masks.append(flipped_mask)

    # Create visualization - 3 channels x (n_steps+1) columns
    # For each cell, we'll show time domain + frequency domain plots
    plt.close('all')
    fig = plt.figure(figsize=(16, 3 * n_channels))

    # Set title
    fig.suptitle(f'Frequency Window Flipping Progression - Most Important First\n' +
                 f'Sample Class: {"Good" if target == 0 else "Bad"}, Reference: {reference_value}',
                 fontsize=16)

    # Create grid spec to organize subplots
    from matplotlib import gridspec
    gs = gridspec.GridSpec(n_channels * 2, len(flipped_samples), figure=fig)

    # Define channel names
    channel_names = ['X-axis', 'Y-axis', 'Z-axis']

    # Plot original and flipped signals
    for step in range(len(flipped_samples)):
        for channel in range(n_channels):
            # Time domain row
            ax_time = fig.add_subplot(gs[channel * 2, step])

            # Plot time domain signal
            ax_time.plot(flipped_samples[step][channel], 'b-')

            # Set title and labels
            if step == 0:
                title = "Original"
                ax_time.set_ylabel(f"{channel_names[channel]}\nAmplitude")
            else:
                title = f"{flipped_pcts[step]:.1f}% Flipped"

            # Add score to title
            title += f"\nScore: {scores[step]:.3f}"
            ax_time.set_title(title)

            # Frequency domain row
            ax_freq = fig.add_subplot(gs[channel * 2 + 1, step])

            # Calculate max frequency to show based on sampling rate
            nyquist = sampling_rate / 2

            # Plot frequency magnitude (log scale)
            freq_magnitude = np.abs(flipped_freqs[step][channel])

            # Plot with proper frequency axis
            ax_freq.semilogy(freqs, freq_magnitude)

            # Highlight flipped regions with gray background
            if step > 0:
                for i in range(freq_length):
                    if flipped_windows_masks[step][channel, i]:
                        ax_freq.axvspan(freqs[i] - 0.5, freqs[i] + 0.5, color='lightgray', alpha=0.5)

            # Set x-axis limits to show full frequency range up to Nyquist
            ax_freq.set_xlim(0, nyquist)

            if step == 0:
                ax_freq.set_ylabel("Magnitude (log)")

            # Only add x-label for bottom row
            if channel == n_channels - 1:
                ax_time.set_xlabel("Time")
                ax_freq.set_xlabel("Frequency (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

'''


def visualize_frequency_window_flipping_sample(model, sample, target, attribution_method,
                                               n_steps=5, window_size=10, most_relevant_first=True,
                                               reference_value="complete_zero", save_path=None,
                                               leverage_symmetry=True, sampling_rate=400,
                                               device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize the progression of frequency domain window flipping for a single time series sample.
    Fixed implementation to properly display frequency range and use proper flipping steps.
    Added support for mild_noise reference value.
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get the shape of the input
    n_channels, time_steps = sample.shape

    # Compute frequency domain attributions
    relevance_time, relevance_freq, signal_freq, input_signal, freqs, _ = attribution_method(
        model=model,
        sample=sample,
        target=target
    )

    # Determine frequency length
    freq_length = relevance_freq.shape[1]

    # Set reference value for frequency domain based on selected option
    if reference_value == "complete_zero":
        # Zero out frequency components completely
        reference_freq = np.zeros_like(signal_freq)
    elif reference_value == "mild_noise":
        # Use mild noise in frequency domain
        reference_freq = np.zeros_like(signal_freq)
        for c in range(n_channels):
            magnitude = np.abs(signal_freq[c]).std() * 0.5  # Use 0.5 as noise scaling factor
            phase = np.random.uniform(-np.pi, np.pi, freq_length)

            # Handle symmetry constraints if needed
            if leverage_symmetry:
                # For real signals with symmetry, DC and Nyquist must be real
                if freq_length > 0:
                    phase[0] = 0  # DC component
                if freq_length > 1 and freq_length == time_steps // 2 + 1:
                    phase[-1] = 0  # Nyquist component

            reference_freq[c] = magnitude * np.exp(1j * phase)
    else:
        # Default to complete zero for any unrecognized reference value
        reference_freq = np.zeros_like(signal_freq)
        print(f"Warning: Unrecognized reference value '{reference_value}'. Using complete_zero instead.")

    # Calculate window importance by averaging relevance within each window
    effective_window_size = max(1, min(window_size, freq_length // 20))
    n_windows = freq_length // effective_window_size
    if freq_length % effective_window_size > 0:
        n_windows += 1

    window_importance = np.zeros((n_channels, n_windows))

    for channel in range(n_channels):
        for window_idx in range(n_windows):
            start_idx = window_idx * effective_window_size
            end_idx = min((window_idx + 1) * effective_window_size, freq_length)

            # Average absolute attribution within the window
            window_importance[channel, window_idx] = np.mean(
                np.abs(relevance_freq[channel, start_idx:end_idx])
            )

    # Flatten and sort window importance
    flat_importance = window_importance.flatten()
    sorted_indices = np.argsort(flat_importance)

    # If flipping most relevant first, reverse the order
    if most_relevant_first:
        sorted_indices = sorted_indices[::-1]

    # Original prediction for reference
    with torch.no_grad():
        original_output = model(sample.unsqueeze(0))
        original_prob = torch.softmax(original_output, dim=1)[0]
        target_class = target
        original_score = original_prob[target_class].item()

    # Calculate windows to flip per step - use exponential progression for more gradual steps
    total_windows = n_channels * n_windows

    # Store flipped samples and scores
    flipped_samples = [input_signal.copy()]  # Time domain
    flipped_freqs = [signal_freq.copy()]  # Frequency domain
    scores = [original_score]
    flipped_windows_count = [0]
    flipped_pcts = [0.0]
    flipped_windows_masks = [np.zeros((n_channels, freq_length), dtype=bool)]

    # Determine steps for visualization
    if n_steps > 5:
        # If many steps, select a representative subset
        step_indices = np.linspace(1, total_windows, n_steps, dtype=int)
    else:
        # Use exponential progression
        step_indices = [int(total_windows * ((i + 1) / n_steps) ** 1.5) for i in range(n_steps)]
        step_indices = [min(s, total_windows) for s in step_indices]

    # Iteratively flip windows
    for step_idx, n_windows_to_flip in enumerate(step_indices):
        # Get flipped sample in frequency domain
        flipped_freq = signal_freq.copy()
        flipped_mask = np.zeros((n_channels, freq_length), dtype=bool)

        # Get windows to flip
        windows_to_flip = sorted_indices[:n_windows_to_flip]

        # Convert flat indices to channel, window indices
        channel_indices = windows_to_flip // n_windows
        window_indices = windows_to_flip % n_windows

        # Set flipped windows to reference value
        for i in range(len(windows_to_flip)):
            channel_idx = channel_indices[i]
            window_idx = window_indices[i]

            start_idx = window_idx * effective_window_size
            end_idx = min((window_idx + 1) * effective_window_size, freq_length)

            flipped_freq[channel_idx, start_idx:end_idx] = reference_freq[channel_idx, start_idx:end_idx]
            flipped_mask[channel_idx, start_idx:end_idx] = True

        # Create DFT-LRP object for inverse transform
        dftlrp = EnhancedDFTLRP(
            signal_length=time_steps,
            leverage_symmetry=leverage_symmetry,
            precision=32,
            cuda=(device == "cuda"),
            create_inverse=True
        )

        # Transform back to time domain
        flipped_time = np.zeros_like(input_signal)
        for c in range(n_channels):
            try:
                # Try to use EnhancedDFTLRP for inverse transform if available
                if dftlrp.has_inverse_fourier_layer:
                    # Convert complex frequency data to format expected by inverse layer
                    if leverage_symmetry:
                        # Format for EnhancedDFTLRP with symmetry
                        freq_data = torch.tensor(flipped_freq[c], dtype=torch.complex64).to(device)
                        time_tensor = torch.fft.irfft(freq_data, n=time_steps).to(device)
                        flipped_time[c] = time_tensor.cpu().numpy()
                    else:
                        # Format for full complex spectrum
                        freq_data = torch.tensor(flipped_freq[c], dtype=torch.complex64).to(device)
                        time_tensor = torch.fft.ifft(freq_data, n=time_steps).real.to(device)
                        flipped_time[c] = time_tensor.cpu().numpy()
                else:
                    # Fallback to numpy IFFT
                    if leverage_symmetry:
                        flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                    else:
                        flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real
            except Exception as e:
                print(f"Error in inverse transform: {e}, falling back to numpy IFFT")
                # Fallback to numpy's IFFT functions
                if leverage_symmetry:
                    flipped_time[c] = np.fft.irfft(flipped_freq[c], n=time_steps)
                else:
                    flipped_time[c] = np.fft.ifft(flipped_freq[c], n=time_steps).real

        # Get model output
        flipped_tensor = torch.tensor(flipped_time, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(flipped_tensor)
            prob = torch.softmax(output, dim=1)[0]
            score = prob[target_class].item()

        # Store results
        flipped_samples.append(flipped_time)
        flipped_freqs.append(flipped_freq)
        scores.append(score)
        flipped_windows_count.append(n_windows_to_flip)
        flipped_pcts.append(n_windows_to_flip / total_windows * 100.0)
        flipped_windows_masks.append(flipped_mask)

        # Clean up DFTLRP
        del dftlrp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Create visualization - 3 channels x (n_steps+1) columns
    # For each cell, we'll show time domain + frequency domain plots
    plt.close('all')
    fig = plt.figure(figsize=(16, 3 * n_channels))

    # Set title
    fig.suptitle(f'Frequency Window Flipping Progression - Most Important First\n' +
                 f'Sample Class: {"Good" if target == 0 else "Bad"}, Reference: {reference_value}',
                 fontsize=16)

    # Create grid spec to organize subplots
    from matplotlib import gridspec
    gs = gridspec.GridSpec(n_channels * 2, len(flipped_samples), figure=fig)

    # Define channel names
    channel_names = ['X-axis', 'Y-axis', 'Z-axis']

    # Plot original and flipped signals
    for step in range(len(flipped_samples)):
        for channel in range(n_channels):
            # Time domain row
            ax_time = fig.add_subplot(gs[channel * 2, step])

            # Plot time domain signal
            ax_time.plot(flipped_samples[step][channel], 'b-')

            # Set title and labels
            if step == 0:
                title = "Original"
                ax_time.set_ylabel(f"{channel_names[channel]}\nAmplitude")
            else:
                title = f"{flipped_pcts[step]:.1f}% Flipped"

            # Add score to title
            title += f"\nScore: {scores[step]:.3f}"
            ax_time.set_title(title)

            # Frequency domain row
            ax_freq = fig.add_subplot(gs[channel * 2 + 1, step])

            # Calculate max frequency to show based on sampling rate
            nyquist = sampling_rate / 2

            # Plot frequency magnitude (log scale)
            freq_magnitude = np.abs(flipped_freqs[step][channel])

            # Plot with proper frequency axis
            ax_freq.semilogy(freqs, freq_magnitude)

            # Highlight flipped regions with gray background
            if step > 0:
                for i in range(freq_length):
                    if flipped_windows_masks[step][channel, i]:
                        ax_freq.axvspan(freqs[i] - 0.5, freqs[i] + 0.5, color='lightgray', alpha=0.5)

            # Set x-axis limits to show full frequency range up to Nyquist
            ax_freq.set_xlim(0, nyquist)

            if step == 0:
                ax_freq.set_ylabel("Magnitude (log)")

            # Only add x-label for bottom row
            if channel == n_channels - 1:
                ax_time.set_xlabel("Time")
                ax_freq.set_xlabel("Frequency (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
def visualize_both_classes(model, samples, attribution_method, n_steps=5, window_size=40,
                           most_relevant_first=True, output_dir="./results",
                           device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Visualize window flipping for one good sample and one bad sample.

    Args:
        model: Trained PyTorch model
        samples: List of (sample, target) tuples
        attribution_method: Function that generates attributions
        n_steps: Number of steps to visualize
        window_size: Size of time windows to flip
        most_relevant_first: If True, flip most relevant windows first
        output_dir: Directory to save visualizations
        device: Device to run on
    """
    # Find one sample of each class
    good_sample = None
    bad_sample = None

    # Loop through samples to find one of each class
    for sample_data, target in samples:
        if target == 0 and good_sample is None:  # Good/normal class
            good_sample = (sample_data, target)
            print("Found good sample")
        elif target == 1 and bad_sample is None:  # Bad/faulty class
            bad_sample = (sample_data, target)
            print("Found bad sample")

        # Break if we have one of each
        if good_sample is not None and bad_sample is not None:
            break

    # Visualize good sample with class-specific reference (mild_noise)
    if good_sample is not None:
        good_sample_data, good_target = good_sample
        print("\n===== Visualizing Good/Normal Sample (Class 0) =====")
        print("Using mild_noise as reference value")

        plt.close('all')
        good_fig = visualize_frequency_window_flipping_sample(
            model=model,
            sample=good_sample_data,
            target=good_target,
            attribution_method=attribution_method,
            n_steps=n_steps,
            window_size=window_size,
            most_relevant_first=most_relevant_first,
            reference_value="mild_noise",
            save_path=f"{output_dir}/window_flipping_good_sample.png",
            device=device
        )
        plt.close(good_fig)
    else:
        print("No good/normal sample found!")

    # Visualize bad sample with class-specific reference (zero)
    if bad_sample is not None:
        bad_sample_data, bad_target = bad_sample
        print("\n===== Visualizing Bad/Faulty Sample (Class 1) =====")
        print("Using zero as reference value")

        plt.close('all')
        bad_fig = visualize_frequency_window_flipping_sample(
            model=model,
            sample=bad_sample_data,
            target=bad_target,
            attribution_method=attribution_method,
            n_steps=n_steps,
            window_size=window_size,
            most_relevant_first=most_relevant_first,
            reference_value="complete_zero",
            save_path=f"{output_dir}/window_flipping_bad_sample.png",
            device=device
        )
        plt.close(bad_fig)
    else:
        print("No bad/faulty sample found!")



def dft_lrp_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for DFT-LRP attribution method.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor
        target: Optional target class
        device: Device to run on

    Returns:
        The full tuple from compute_basic_dft_lrp for frequency domain flipping
    """
    from utils.xai_implementation import compute_basic_dft_lrp

    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_basic_dft_lrp(
        model=model,
        sample=sample,
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        precision=32,
        sampling_rate=400
    )


def dft_gradient_input_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for DFT-GradInput attribution method.
    """
    from utils.xai_implementation import compute_dft_gradient_input

    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_gradient_input(
        model=model,
        sample=sample,
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        sampling_rate=400
    )


def dft_smoothgrad_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Wrapper for DFT-SmoothGrad attribution method.
    """
    from utils.xai_implementation import compute_dft_smoothgrad

    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    return compute_dft_smoothgrad(
        model=model,
        sample=sample,
        label=target,
        device=device,
        signal_length=sample.shape[1],
        leverage_symmetry=True,
        sampling_rate=400,
        num_samples=20,
        noise_level=0.2
    )


def dft_occlusion_wrapper(model, sample, target=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Fixed wrapper for DFT Occlusion attribution method that handles window size properly.
    """
    from utils.xai_implementation import compute_dft_for_xai_method

    # Ensure sample is on the correct device
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device)
    else:
        sample = sample.to(device)

    # Use a small, fixed window size for occlusion to avoid conflicts

    try:
        return compute_dft_for_xai_method(
            xai_method_func = occlusion_signal_relevance,
            model=model,
            sample=sample,
            label=target,
            device=device,
            signal_length=2000,
            leverage_symmetry=True,
            sampling_rate=400,
            occlusion_type="zero"
        )
    except Exception as e:
        print(f"Error in DFT-Occlusion: {e}")
        # Create empty placeholder results with correct shapes
        n_channels, time_steps = sample.shape
        freq_length = time_steps // 2 + 1  # For leverage_symmetry=True

        # Empty results with correct shapes
        relevance_time = np.zeros((n_channels, time_steps), dtype=np.float32)
        relevance_freq = np.zeros((n_channels, freq_length), dtype=np.float32)
        signal_freq = np.zeros((n_channels, freq_length), dtype=np.complex128)
        freqs = np.fft.rfftfreq(time_steps, d=1.0 / 400)

        return relevance_time, relevance_freq, signal_freq, sample.cpu().numpy(), freqs, target




def main():
    """
    Main function to run frequency domain window flipping evaluation.
    Also includes feature extraction from important frequency windows.
    """
    # Clean up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Configuration
    model_path = "../cnn1d_model_test_newest.ckpt"
    data_dir = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    output_dir = "./results/freq_domain"
    n_samples_per_class = 5  # Number of samples per class
    n_steps = 10
    window_size = 10
    sampling_rate = 400
    leverage_symmetry = True
    reference_value = "complete_zero"  # Options: zero, complete_zero, magnitude_zero, noise, etc.

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, device)
    model.eval()


    # Set up frequency attribution methods dictionary
    freq_attribution_methods = {
        "DFT-LRP": dft_lrp_wrapper,
        "DFT-GradInput": dft_gradient_input_wrapper,
        "DFT-SmoothGrad": dft_smoothgrad_wrapper,
        "DFT-Occlusion": dft_occlusion_wrapper
    }

    # Load balanced samples
    samples = load_balanced_samples(data_dir, n_samples_per_class)

    # Visualize frequency window flipping for a single sample
    print("\n===== Visualizing Frequency Domain Window Flipping =====")
    sample_data, target = samples[0]

    # Visualize with DFT-LRP
    plt.close('all')
    # Visualize with LRP
    plt.close('all')

    # Visualize window flipping for one good and one bad sample
    print("\n===== Visualizing Time Domain Window Flipping for Both Classes =====")
    visualize_both_classes(
        model=model,
        samples=samples,
        attribution_method=dft_lrp_wrapper,
        n_steps=5,
        window_size=window_size,
        most_relevant_first=True,
        output_dir=output_dir,
        device=device
    )
    plt.close('all')

    # Evaluate with most important windows flipped first
    print("\n===== Evaluating with most important frequency windows flipped first =====")
    results_most_first, agg_results_most_first = run_frequency_guided_window_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=freq_attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=True,
        reference_value=reference_value,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        max_samples=len(samples),  # Use all loaded samples
        output_dir=output_dir,
        device=device
    )

    # Evaluate with least important windows flipped first
    print("\n===== Evaluating with least important frequency windows flipped first =====")
    results_least_first, agg_results_least_first = run_frequency_guided_window_flipping_evaluation(
        model=model,
        samples=samples,
        attribution_methods=freq_attribution_methods,
        n_steps=n_steps,
        window_size=window_size,
        most_relevant_first=False,
        reference_value=reference_value,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate,
        max_samples=len(samples),  # Use all loaded samples
        output_dir=output_dir,
        device=device
    )

    # Calculate and print faithfulness ratios
    print("\n===== Frequency Domain Faithfulness Ratios =====")
    print("(Ratio of AUC when flipping least important windows first vs. most important first)")
    print("Higher ratio indicates better explanation faithfulness")
    print("-" * 60)
    print(f"{'Method':<20} {'AUC Ratio':<12} {'Most First':<12} {'Least First':<12}")
    print("-" * 60)

    for method_name in freq_attribution_methods.keys():
        most_auc = agg_results_most_first[method_name]["mean_auc"]
        least_auc = agg_results_least_first[method_name]["mean_auc"]

        if np.isnan(most_auc) or np.isnan(least_auc) or most_auc == 0:
            ratio = float('nan')
        else:
            ratio = least_auc / most_auc

        print(f"{method_name:<20} {ratio:<12.4f} {most_auc:<12.4f} {least_auc:<12.4f}")

    # Extract important frequency windows using DFT-LRP
    print("\n===== Extracting Important Frequency Windows =====")
    freq_windows = collect_important_freq_windows(
        model=model,
        samples=samples,
        attribution_method=dft_lrp_wrapper,
        n_samples_per_class=min(5, n_samples_per_class),  # Use fewer samples for feature extraction
        n_windows=10,  # Extract 10 top windows per sample
        window_size=window_size,
        device=device,
        leverage_symmetry=leverage_symmetry,
        sampling_rate=sampling_rate
    )

    # Visualize and analyze important frequency windows
    print("\n===== Analyzing Important Frequency Windows =====")
    visualize_freq_windows(freq_windows, output_dir=f"{output_dir}/windows")
    freq_window_stats = analyze_freq_windows(freq_windows, output_dir=f"{output_dir}/windows")

    # Print top discriminative features
    if freq_window_stats is not None and not freq_window_stats.empty:
        print("\n===== Top Discriminative Frequency Features =====")
        top_features = freq_window_stats.head(5)
        print(top_features[["feature", "p_value", "normal_mean", "faulty_mean", "difference_pct"]])

    print("\nFrequency domain window flipping evaluation complete!")


if __name__ == "__main__":
    main()