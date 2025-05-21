import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import h5py
from pathlib import Path
from Classification.cnn1D_model import CNN1D_DS, FrequencyDomainCNN


class LRP:
    """
    Layer-wise Relevance Propagation for CNN1D models with DFT integration
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize LRP with a trained model

        Args:
            model: Trained PyTorch CNN1D model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Register hooks to capture activations and gradients
        self.activations = {}
        self.gradients = {}
        self.handles = []

        # Register hooks for all convolutional and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                self.handles.append(
                    module.register_forward_hook(self._save_activation_hook(name))
                )
                self.handles.append(
                    module.register_backward_hook(self._save_gradient_hook(name))
                )

    def _save_activation_hook(self, name):
        """Create a hook to save activations"""

        def hook(module, input, output):
            self.activations[name] = output

        return hook

    def _save_gradient_hook(self, name):
        """Create a hook to save gradients"""

        def hook(module, grad_input, grad_output):
            self.gradients[name] = grad_output[0]

        return hook

    def close(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()

    def _preprocess_dft(self, x):
        """
        Apply DFT to input signal and prepare for feature visualization

        Args:
            x: Input time-domain signal [batch, channels, time]

        Returns:
            Frequency-domain representation [batch, channels, frequencies]
        """
        # Apply FFT along the time dimension
        x_freq = torch.fft.rfft(x, dim=2)
        # Get magnitude (amplitude) spectrum
        x_freq_mag = torch.abs(x_freq)
        return x_freq_mag

    def explain(self, input_tensor, target_class=None, alpha=1, epsilon=1e-7):
        """
        Generate LRP explanation for the input

        Args:
            input_tensor: Input tensor [batch, channels, time]
            target_class: Target class index. If None, uses model's prediction
            alpha: Alpha parameter for LRP-α rule (default: 1)
            epsilon: Small constant for numerical stability

        Returns:
            relevance_time: Time-domain relevance scores
            relevance_freq: Frequency-domain relevance scores
        """
        # Step 1: Ensure input is on correct device
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Step 2: Forward pass to get predictions
        self.model.zero_grad()
        output = self.model(input_tensor)

        # Step 3: Get target class (prediction or specified)
        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        # Step 4: Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)

        # Step 5: Backward pass to get gradients
        output.backward(gradient=one_hot)

        # Step 6: Apply LRP to get relevance scores in time domain
        relevance_time = self._compute_lrp_relevance(input_tensor, alpha, epsilon)

        # Step 7: Apply DFT to get frequency domain relevance
        input_freq = self._preprocess_dft(input_tensor.detach())
        relevance_freq = self._map_relevance_to_frequency(relevance_time, input_freq)

        return relevance_time, relevance_freq

    def _compute_lrp_relevance(self, input_tensor, alpha, epsilon):
        """
        Compute LRP relevance scores in time domain
        """
        # Basic Gradient*Input method (can be replaced with more sophisticated LRP rules)
        gradients = input_tensor.grad.detach()
        # Apply LRP-α rule: R = α * (x * grad) / (x * grad + epsilon)
        relevance = alpha * input_tensor.detach() * gradients
        relevance = relevance / (torch.abs(relevance) + epsilon)

        return relevance

    def _map_relevance_to_frequency(self, relevance_time, input_freq):
        """
        Map time-domain relevance to frequency domain

        Args:
            relevance_time: Time-domain relevance scores
            input_freq: Frequency-domain representation

        Returns:
            Frequency-domain relevance scores
        """
        # Apply FFT to relevance scores
        relevance_freq = torch.fft.rfft(relevance_time, dim=2)
        relevance_freq_mag = torch.abs(relevance_freq)

        # Weight by frequency components
        weighted_relevance = relevance_freq_mag * input_freq

        # Normalize
        total = torch.sum(weighted_relevance, dim=2, keepdim=True)
        normalized_relevance = weighted_relevance / (total + 1e-10)

        return normalized_relevance

    def plot_explanation(self, input_tensor, target_class=None, sample_idx=0,
                         channel_names=None, cmap='seismic', figsize=(18, 12)):
        """
        Create plots of the LRP explanation

        Args:
            input_tensor: Input tensor [batch, channels, time]
            target_class: Target class index. If None, uses model's prediction
            sample_idx: Index of sample to explain in batch
            channel_names: List of names for input channels
            cmap: Colormap for relevance visualization
            figsize: Figure size

        Returns:
            Figure with the explanation
        """
        # Get relevance scores
        relevance_time, relevance_freq = self.explain(input_tensor, target_class)

        # Select single example from batch
        input_data = input_tensor[sample_idx].cpu().detach().numpy()
        relevance_time_data = relevance_time[sample_idx].cpu().detach().numpy()
        relevance_freq_data = relevance_freq[sample_idx].cpu().detach().numpy()

        # Get prediction and class
        with torch.no_grad():
            output = self.model(input_tensor)

        pred_class = torch.argmax(output[sample_idx]).item()
        pred_prob = torch.softmax(output[sample_idx], dim=0)[pred_class].item()

        if target_class is None:
            target_class = pred_class
        else:
            target_class = target_class[sample_idx].item() if isinstance(target_class, torch.Tensor) else target_class

        # Setup channel names
        n_channels = input_data.shape[0]
        if channel_names is None:
            channel_names = [f"Channel {i + 1}" for i in range(n_channels)]

        # Create figure with subplots for time and frequency domain
        fig, axes = plt.subplots(n_channels, 3, figsize=figsize)
        fig.suptitle(f'DFT-LRP Explanation: Class {target_class} ' +
                     f'(Prediction: {pred_class}, Confidence: {pred_prob:.2f})',
                     fontsize=16)

        time_steps = np.arange(input_data.shape[1])
        freq_steps = np.arange(relevance_freq_data.shape[1])

        for i in range(n_channels):
            # Plot original signal
            axes[i, 0].plot(time_steps, input_data[i], 'b-')
            axes[i, 0].set_title(f"Original Signal - {channel_names[i]}")
            axes[i, 0].set_xlabel("Time")

            # Plot time-domain relevance
            im = axes[i, 1].imshow(
                relevance_time_data[i].reshape(1, -1),
                aspect='auto',
                cmap=cmap,
                interpolation='nearest'
            )
            axes[i, 1].set_title(f"Time-Domain Relevance - {channel_names[i]}")
            axes[i, 1].set_xlabel("Time")
            fig.colorbar(im, ax=axes[i, 1])

            # Plot frequency-domain relevance
            im = axes[i, 2].imshow(
                relevance_freq_data[i].reshape(1, -1),
                aspect='auto',
                cmap=cmap,
                interpolation='nearest'
            )
            axes[i, 2].set_title(f"Frequency-Domain Relevance - {channel_names[i]}")
            axes[i, 2].set_xlabel("Frequency")
            fig.colorbar(im, ax=axes[i, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


class DFT_LRP:
    """
    Discrete Fourier Transform enhanced Layer-wise Relevance Propagation
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize DFT-LRP with a trained CNN1D model

        Args:
            model: Trained PyTorch CNN1D model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Dictionary to store activations and relevance scores
        self.activations = {}
        self.relevance = {}

        # Register forward hooks for all layers
        self.handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.ReLU, nn.MaxPool1d, nn.AdaptiveAvgPool1d, nn.Dropout)):
                self.handles.append(
                    module.register_forward_hook(self._save_activation_hook(name))
                )

    def _save_activation_hook(self, name):
        """Create a hook to save activations"""

        def hook(module, input, output):
            self.activations[name] = output

        return hook

    def close(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()

    def _compute_dft_relevance(self, input_tensor, target_class=None, epsilon=1e-9):
        """
        Forward pass to compute DFT relevance propagation

        Args:
            input_tensor: Input tensor [batch, channels, time]
            target_class: Target class index. If None, uses model's prediction
            epsilon: Small value for numerical stability

        Returns:
            Relevance scores for the input tensor
        """
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        # Initialize relevance with model's output (one-hot encoded)
        batch_size = output.shape[0]
        R = torch.zeros_like(output)
        for i in range(batch_size):
            R[i, target_class[i]] = output[i, target_class[i]]

        # Get layer names in reverse order for the backward pass
        layer_names = list(self.activations.keys())
        layer_names.reverse()

        # Save output relevance
        self.relevance['output'] = R.clone()

        # Backward pass through the layers
        for i, name in enumerate(layer_names):
            if i == 0:  # Skip the last layer (output layer)
                continue

            current_layer_type = name.split('.')[-2] if '.' in name else name

            # Get the activation for the current layer
            activation = self.activations[name]

            # Apply different LRP rules based on layer type
            if 'conv' in name:
                # Apply LRP-ε rule for convolutional layers
                R = self._lrp_epsilon_rule(name, R, activation, epsilon)
            elif 'fc' in name or 'linear' in name:
                # Apply LRP-ε rule for linear layers
                R = self._lrp_epsilon_rule(name, R, activation, epsilon)
            elif 'relu' in name:
                # Pass relevance through ReLU
                R = R * (activation > 0).float()
            elif 'pool' in name:
                # Special rule for pooling layers
                R = self._lrp_pooling_rule(name, R, activation)

            # Store relevance for this layer
            self.relevance[name] = R.clone()

        # Final relevance is the relevance for the input features
        input_relevance = self.relevance[layer_names[-1]]

        # Map to frequency domain
        input_freq = torch.fft.rfft(input_tensor, dim=2)
        input_freq_mag = torch.abs(input_freq)

        # Compute frequency relevance using the input relevance
        freq_relevance = torch.fft.rfft(input_relevance, dim=2)
        freq_relevance_mag = torch.abs(freq_relevance)

        # Weight frequency relevance by input frequency magnitude
        weighted_freq_relevance = freq_relevance_mag * input_freq_mag

        # Normalize the frequency relevance
        norm = torch.sum(weighted_freq_relevance, dim=2, keepdim=True) + epsilon
        normalized_freq_relevance = weighted_freq_relevance / norm

        return input_relevance, normalized_freq_relevance

    def _lrp_epsilon_rule(self, layer_name, R_upper, activations, epsilon):
        """
        Implement LRP-ε rule: R_i = a_i * (R_j / (∑_i a_i + ε))
        """
        # Find the corresponding module in the model
        parts = layer_name.split('.')
        module = self.model
        for part in parts:
            try:
                module = getattr(module, part)
            except AttributeError:
                pass  # Skip if not found (happens with numbered layers)

        if isinstance(module, nn.Conv1d):
            # Get the module weights and reshape for the computation
            weights = module.weight
            bias = module.bias if module.bias is not None else 0

            # Get input to this layer (from previous layer's activations)
            prev_layer_name = self._get_previous_layer_name(layer_name)
            if prev_layer_name:
                lower_activations = self.activations[prev_layer_name]
            else:
                # If no previous layer found, use model input
                lower_activations = activations

            # Compute denominator for LRP-ε rule
            # We compute this using a forward pass through the layer
            with torch.no_grad():
                denominator = self._forward_pass_through_layer(module, lower_activations)
                denominator = denominator + epsilon * (denominator >= 0).float() - epsilon * (denominator < 0).float()

            # Compute relevance for lower layer
            R_lower = torch.autograd.grad(
                outputs=denominator,
                inputs=lower_activations,
                grad_outputs=R_upper / denominator,
                retain_graph=True
            )[0] * lower_activations

            return R_lower

        elif isinstance(module, nn.Linear):
            # Convert to tensor
            weights = module.weight
            bias = module.bias if module.bias is not None else 0

            # Get input to this layer
            prev_layer_name = self._get_previous_layer_name(layer_name)
            if prev_layer_name:
                lower_activations = self.activations[prev_layer_name]
            else:
                lower_activations = activations

            # Compute denominator for LRP-ε rule
            with torch.no_grad():
                denominator = self._forward_pass_through_layer(module, lower_activations)
                denominator = denominator + epsilon * (denominator >= 0).float() - epsilon * (denominator < 0).float()

            # Compute relevance for lower layer
            R_lower = torch.autograd.grad(
                outputs=denominator,
                inputs=lower_activations,
                grad_outputs=R_upper / denominator,
                retain_graph=True
            )[0] * lower_activations

            return R_lower

        else:
            # For other layer types, pass the relevance as is
            return R_upper

    def _lrp_pooling_rule(self, layer_name, R_upper, activations):
        """
        LRP rule for pooling layers - backpropagate relevance proportionally
        """
        # Find previous layer's activations
        prev_layer_name = self._get_previous_layer_name(layer_name)
        if not prev_layer_name:
            return R_upper

        prev_activations = self.activations[prev_layer_name]

        # If it's a global pooling layer, we need to redistribute relevance
        if isinstance(self._get_module_from_name(layer_name), nn.AdaptiveAvgPool1d):
            # For adaptive average pooling, distribute evenly
            return torch.nn.functional.interpolate(
                R_upper,
                size=prev_activations.shape[2],
                mode='linear'
            )
        elif isinstance(self._get_module_from_name(layer_name), nn.MaxPool1d):
            # For max pooling, only assign relevance to the maximum values
            with torch.no_grad():
                # Create a mask for max locations
                pool_size = self._get_module_from_name(layer_name).kernel_size
                stride = self._get_module_from_name(layer_name).stride

                R_expanded = torch.zeros_like(prev_activations)

                # Iterate through each channel and find max locations
                batch_size, channels, time_len = prev_activations.shape

                for b in range(batch_size):
                    for c in range(channels):
                        for t in range(0, time_len - pool_size + 1, stride):
                            if t + pool_size <= time_len:
                                window = prev_activations[b, c, t:t + pool_size]
                                max_idx = torch.argmax(window) + t
                                out_idx = t // stride
                                if out_idx < R_upper.shape[2]:
                                    R_expanded[b, c, max_idx] = R_upper[b, c, out_idx]

                return R_expanded

        # Default: pass the relevance unchanged
        return R_upper

    def _forward_pass_through_layer(self, module, input_tensor):
        """Helper to compute forward pass through a single layer"""
        with torch.no_grad():
            if isinstance(module, nn.Conv1d):
                return nn.functional.conv1d(
                    input_tensor,
                    module.weight,
                    module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups
                )
            elif isinstance(module, nn.Linear):
                # Make sure input tensor is properly flattened for linear layer
                input_shape = input_tensor.shape
                if len(input_shape) > 2:
                    input_tensor = input_tensor.view(input_shape[0], -1)
                return nn.functional.linear(input_tensor, module.weight, module.bias)
            else:
                return module(input_tensor)

    def _get_module_from_name(self, name):
        """Helper to get module from its name"""
        parts = name.split('.')
        module = self.model
        for part in parts:
            try:
                module = getattr(module, part)
            except AttributeError:
                # Try numeric index for sequential modules
                if part.isdigit():
                    module = module[int(part)]
                else:
                    pass  # Skip if not found
        return module

    def _get_previous_layer_name(self, current_layer_name):
        """Helper to find the previous layer name in activation order"""
        layer_names = list(self.activations.keys())
        try:
            idx = layer_names.index(current_layer_name)
            if idx > 0:
                return layer_names[idx - 1]
        except ValueError:
            pass
        return None

    def explain(self, input_tensor, target_class=None, epsilon=1e-9):
        """
        Generate DFT-LRP explanation for input tensor

        Args:
            input_tensor: Input tensor [batch, channels, time]
            target_class: Target class index (optional)
            epsilon: Small value for numerical stability

        Returns:
            Time domain relevance and frequency domain relevance
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        # Clear cached activations
        self.activations = {}
        self.relevance = {}

        # Forward pass to compute activations
        self.model(input_tensor)

        # Compute relevance scores
        time_relevance, freq_relevance = self._compute_dft_relevance(
            input_tensor, target_class, epsilon
        )

        return time_relevance, freq_relevance

    def plot_dft_lrp(self, input_tensor, target_class=None, sample_idx=0,
                     axislabels=None, cmap='viridis', figsize=(15, 12)):
        """
        Plot DFT-LRP explanation

        Args:
            input_tensor: Input tensor [batch, channels, time]
            target_class: Target class index (optional)
            sample_idx: Index of sample to explain
            axislabels: Labels for axes
            cmap: Colormap for heatmaps
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Get relevance scores
        time_relevance, freq_relevance = self.explain(input_tensor, target_class)

        # Get sample to explain
        x_sample = input_tensor[sample_idx].detach().cpu().numpy()
        time_rel_sample = time_relevance[sample_idx].detach().cpu().numpy()
        freq_rel_sample = freq_relevance[sample_idx].detach().cpu().numpy()

        # Get prediction and confidence
        with torch.no_grad():
            output = self.model(input_tensor)

        pred_class = torch.argmax(output[sample_idx]).item()
        confidence = torch.softmax(output[sample_idx], dim=0)[pred_class].item()

        # Set target class for explanation
        if target_class is None:
            target_class = pred_class
        elif isinstance(target_class, torch.Tensor):
            target_class = target_class[sample_idx].item()

        # Set up channel labels
        n_channels = x_sample.shape[0]
        if axislabels is None:
            axislabels = [f'Axis {i + 1}' for i in range(n_channels)]

        # Create figure
        fig, axes = plt.subplots(n_channels, 3, figsize=figsize)
        fig.suptitle(f'DFT-LRP Explanation\nPrediction: Class {pred_class} (Confidence: {confidence:.2f})',
                     fontsize=16)

        # Time steps and frequency bins
        time_steps = np.arange(x_sample.shape[1])
        freq_bins = np.arange(freq_rel_sample.shape[1])

        for i in range(n_channels):
            # Plot original signal
            axes[i, 0].plot(time_steps, x_sample[i], 'b-')
            axes[i, 0].set_title(f'Original Signal - {axislabels[i]}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Amplitude')

            # Plot time-domain relevance
            im = axes[i, 1].imshow(time_rel_sample[i].reshape(1, -1),
                                   aspect='auto', cmap=cmap,
                                   extent=[0, len(time_steps), 0, 1])
            axes[i, 1].set_title(f'Time-Domain Relevance - {axislabels[i]}')
            axes[i, 1].set_xlabel('Time')
            plt.colorbar(im, ax=axes[i, 1])

            # Plot frequency-domain relevance
            im = axes[i, 2].imshow(freq_rel_sample[i].reshape(1, -1),
                                   aspect='auto', cmap=cmap,
                                   extent=[0, len(freq_bins), 0, 1])
            axes[i, 2].set_title(f'Frequency-Domain Relevance - {axislabels[i]}')
            axes[i, 2].set_xlabel('Frequency')
            plt.colorbar(im, ax=axes[i, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig


# Example usage
def analyze_vibration_sample(model_path, data_file, channel_names=None, class_names=None):
    """
    Analyze a vibration sample with DFT-LRP

    Args:
        model_path: Path to the trained model checkpoint
        data_file: Path to the h5 file containing vibration data
        channel_names: Names of the input channels (default: ['X', 'Y', 'Z'])
        class_names: Names of the classes (default: ['Good', 'Bad'])

    Returns:
        Figure with the explanation
    """
    # Default values
    if channel_names is None:
        channel_names = ['X', 'Y', 'Z']
    if class_names is None:
        class_names = ['Good', 'Bad']

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D_DS()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load data
    with h5py.File(data_file, 'r') as f:
        data = f['vibration_data'][:]  # Shape (2000, 3)

    # Preprocess data
    data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Initialize DFT-LRP
    dft_lrp = DFT_LRP(model, device)

    # Get explanation
    fig = dft_lrp.plot_dft_lrp(input_tensor, axislabels=channel_names, figsize=(15, 12))

    # Clean up
    dft_lrp.close()

    return fig


def analyze_dataset_samples(model_path, data_dir, num_samples=5):
    """
    Analyze multiple samples from a dataset

    Args:
        model_path: Path to the trained model checkpoint
        data_dir: Directory containing the dataset
        num_samples: Number of samples to analyze

    Returns:
        List of figures with explanations
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D_DS()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create dataset
    class SimpleVibrationDataset(Dataset):
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.file_paths = []
            self.labels = []

            # Get good samples
            good_dir = self.data_dir / 'good'
            if good_dir.exists():
                for file_path in good_dir.glob('*.h5'):
                    self.file_paths.append(file_path)
                    self.labels.append(0)  # 0 = good

            # Get bad samples
            bad_dir = self.data_dir / 'bad'
            if bad_dir.exists():
                for file_path in bad_dir.glob('*.h5'):
                    self.file_paths.append(file_path)
                    self.labels.append(1)  # 1 = bad

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            with h5py.File(file_path, 'r') as f:
                data = f['vibration_data'][:]  # Shape (2000, 3)

            data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN
            label = self.labels[idx]

            return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    # Create dataset and dataloader
    dataset = SimpleVibrationDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize DFT-LRP
    dft_lrp = DFT_LRP(model, device)

    # Analyze samples
    figures = []
    channel_names = ['X', 'Y', 'Z']

    for i, (inputs, labels) in enumerate(dataloader):
        if i >= num_samples:
            break

        # Get explanation
        fig = dft_lrp.plot_dft_lrp(
            inputs,
            target_class=labels,
            axislabels=channel_names,
            figsize=(15, 12)
        )
        figures.append(fig)

    # Clean up
    dft_lrp.close()

    return figures


if __name__ == "__main__":
    # Example usage
    model_path = '../cnn1d_freq.ckpt'
    data_file = 'path/to/your/data.h5'
    analyze_vibration_sample(model_path, data_file)

    # Analyze multiple samples from a dataset
    data_dir = 'path/to/your/dataset'
    figures = analyze_dataset_samples(model_path, data_dir, num_samples=5)

    for fig in figures:
        plt.show(fig)