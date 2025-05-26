import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from captum.attr import LayerGradCam, LRP, IntegratedGradients, DeepLift
from captum.attr import visualization as viz


class ModelExplainer:
    """
    Utility class to explain CNN model predictions using various XAI techniques
    including Layer-wise Relevance Propagation (LRP)
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # For storing activations and gradients
        self.activations = {}
        self.gradients = {}
        
        # Register hooks for getting intermediate activations
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture activations and gradients"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for CNN layers
        if hasattr(self.model, 'time_cnn'):
            # For FrequencyDomainCNN
            for name, module in self.model.time_cnn.feature_extractor.named_children():
                if isinstance(module, nn.Conv1d):
                    module.register_forward_hook(get_activation(f'time_conv_{name}'))
                    module.register_backward_hook(get_gradient(f'time_conv_{name}'))
            
            # For frequency branch
            self.model.freq_conv1.register_forward_hook(get_activation('freq_conv1'))
            self.model.freq_conv1.register_backward_hook(get_gradient('freq_conv1'))
            self.model.freq_conv2.register_forward_hook(get_activation('freq_conv2'))
            self.model.freq_conv2.register_backward_hook(get_gradient('freq_conv2'))
            
        elif hasattr(self.model, 'conv_block'):
            # For CNN_1d model
            for i, module in enumerate(self.model.conv_block):
                if isinstance(module, nn.Conv1d):
                    module.register_forward_hook(get_activation(f'conv_{i}'))
                    module.register_backward_hook(get_gradient(f'conv_{i}'))
                    
    def get_stored_activations(self):
        """Get activations dictionary stored by the model"""
        if hasattr(self.model, 'get_activations'):
            return self.model.get_activations()
        return self.activations
        
    def explain_prediction_lrp(self, input_data, target_class=None):
        """
        Explain prediction using Layer-wise Relevance Propagation (LRP)
        
        Args:
            input_data: Model input as tensor or tuple of tensors
            target_class: Target class index for which to compute attributions
                          If None, uses the model's predicted class
        
        Returns:
            Dictionary with LRP attributions for different layers
        """
        # Prepare input for captum
        if isinstance(input_data, (tuple, list)):
            time_input = input_data[0].to(self.device)
            freq_input = input_data[1].to(self.device)
            input_tensor = time_input  # LRP will work on time domain data
        else:
            input_tensor = input_data.to(self.device)
        
        # Initialize LRP
        lrp = LRP(self.model)
        
        # Forward pass to get prediction
        if isinstance(input_data, (tuple, list)):
            output = self.model((time_input, freq_input))
        else:
            output = self.model(input_tensor)
            
        # If target class not provided, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Compute attributions using LRP
        attributions = lrp.attribute(input_tensor, target=target_class)
        
        return {
            'input': input_tensor.cpu().numpy(),
            'attributions': attributions.cpu().numpy(),
            'target_class': target_class,
            'activations': self.get_stored_activations()
        }
    
    def explain_with_gradcam(self, input_data, target_layer, target_class=None):
        """
        Explain prediction using Grad-CAM
        
        Args:
            input_data: Model input as tensor or tuple of tensors
            target_layer: Target layer for Grad-CAM visualization
            target_class: Target class index for attributions
        
        Returns:
            Dictionary with Grad-CAM results
        """
        if isinstance(input_data, (tuple, list)):
            time_input = input_data[0].to(self.device)
            freq_input = input_data[1].to(self.device)
            
            # Forward pass to get prediction
            output = self.model((time_input, freq_input))
            
            # If target class not provided, use the predicted class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            # Run backward pass to get gradients
            output[:, target_class].backward()
            
            # Get activations and gradients
            activations = self.activations[target_layer]
            gradients = self.gradients[target_layer]
            
            # Compute GradCAM
            pooled_gradients = torch.mean(gradients, dim=[0, 2])
            for i in range(activations.shape[1]):
                activations[:, i, :] *= pooled_gradients[i]
            
            # Average over channels
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            return {
                'input': time_input.cpu().numpy(),
                'heatmap': heatmap.cpu().numpy(),
                'target_class': target_class,
                'prediction': output.softmax(dim=1).detach().cpu().numpy()
            }
        else:
            # Handle single input tensor case
            input_tensor = input_data.to(self.device)
            
            # Forward pass to get prediction
            output = self.model(input_tensor)
            
            # If target class not provided, use the predicted class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            # Run backward pass to get gradients
            output[:, target_class].backward()
            
            # Get activations and gradients
            activations = self.activations[target_layer]
            gradients = self.gradients[target_layer]
            
            # Compute GradCAM
            pooled_gradients = torch.mean(gradients, dim=[0, 2])
            for i in range(activations.shape[1]):
                activations[:, i, :] *= pooled_gradients[i]
            
            # Average over channels
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            return {
                'input': input_tensor.cpu().numpy(),
                'heatmap': heatmap.cpu().numpy(),
                'target_class': target_class,
                'prediction': output.softmax(dim=1).detach().cpu().numpy()
            }
            
    def explain_with_integrated_gradients(self, input_data, target_class=None):
        """
        Explain prediction using Integrated Gradients
        
        Args:
            input_data: Model input as tensor or tuple of tensors
            target_class: Target class index for attributions
        
        Returns:
            Dictionary with IG attributions
        """
        if isinstance(input_data, (tuple, list)):
            time_input = input_data[0].to(self.device)
            freq_input = input_data[1].to(self.device)
            
            # Create a custom forward function for captum
            def forward_func(time_inp):
                return self.model((time_inp, freq_input))
                
            # Initialize integrated gradients
            ig = IntegratedGradients(forward_func)
            
            # Forward pass to get prediction
            output = self.model((time_input, freq_input))
            
            # If target class not provided, use the predicted class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            # Compute attributions
            attributions = ig.attribute(time_input, target=target_class)
            
            return {
                'input': time_input.cpu().numpy(),
                'attributions': attributions.cpu().numpy(),
                'target_class': target_class,
                'prediction': output.softmax(dim=1).detach().cpu().numpy()
            }
        else:
            input_tensor = input_data.to(self.device)
            
            # Initialize integrated gradients
            ig = IntegratedGradients(self.model)
            
            # Forward pass to get prediction
            output = self.model(input_tensor)
            
            # If target class not provided, use the predicted class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            # Compute attributions
            attributions = ig.attribute(input_tensor, target=target_class)
            
            return {
                'input': input_tensor.cpu().numpy(),
                'attributions': attributions.cpu().numpy(),
                'target_class': target_class,
                'prediction': output.softmax(dim=1).detach().cpu().numpy()
            }
    
    def visualize_attribution(self, attribution_dict, method_name="LRP", show_original=True, 
                              axis=-1, figsize=(12, 8)):
        """
        Visualize attribution results
        
        Args:
            attribution_dict: Dictionary with attribution results
            method_name: Name of the attribution method
            show_original: Whether to show original input
            axis: Axis to visualize for the vibration data (0, 1, 2 for X, Y, Z)
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # If axis is -1, visualize all axes
        if axis == -1:
            n_axes = attribution_dict['input'].shape[1]  # Number of axes (usually 3)
            for i in range(n_axes):
                # Original signal
                if show_original:
                    plt.subplot(n_axes, 2, i*2 + 1)
                    plt.plot(attribution_dict['input'][0, i])
                    plt.title(f"Original Signal (Axis {i})")
                    plt.grid(True)
                
                # Attribution visualization
                plt.subplot(n_axes, 2, i*2 + 2)
                plt.plot(attribution_dict['attributions'][0, i], 'r')
                plt.title(f"{method_name} Attribution (Axis {i})")
                plt.grid(True)
        else:
            # Original signal for selected axis
            if show_original:
                plt.subplot(1, 2, 1)
                plt.plot(attribution_dict['input'][0, axis])
                plt.title(f"Original Signal (Axis {axis})")
                plt.grid(True)
            
            # Attribution visualization for selected axis
            plt.subplot(1, 2, 2)
            plt.plot(attribution_dict['attributions'][0, axis], 'r')
            plt.title(f"{method_name} Attribution (Axis {axis})")
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()

    def visualize_feature_maps(self, layer_name=None):
        """
        Visualize feature maps from a specific layer or all layers
        
        Args:
            layer_name: Name of layer to visualize. If None, visualize all.
        """
        activations = self.get_stored_activations()
        
        if layer_name is not None and layer_name in activations:
            self._plot_feature_maps(activations[layer_name], layer_name)
        else:
            for name, activation in activations.items():
                # Skip non-convolutional activations
                if len(activation.shape) != 3:  # [batch, channels, length]
                    continue
                self._plot_feature_maps(activation, name)
    
    def _plot_feature_maps(self, activation, layer_name):
        """Helper to plot feature maps of a layer"""
        activation = activation.cpu().numpy()
        n_channels = min(16, activation.shape[1])  # Show max 16 channels
        
        # Determine grid size
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Feature Maps: {layer_name}", fontsize=16)
        
        for i in range(n_channels):
            plt.subplot(grid_size, grid_size, i+1)
            plt.plot(activation[0, i])
            plt.title(f"Channel {i}")
            plt.grid(True)
            plt.xticks([])
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


# Example usage code
if __name__ == "__main__":
    import sys
    sys.path.append("E:/Thesis/Datasets/CNC/Classification")
    
    from cnn1d_new_freq import FrequencyDomainCNN, VibrationDataset
    
    # Load a trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrequencyDomainCNN().to(device)
    model.load_state_dict(torch.load("../efficient_cnn1d_freq_model.ckpt", map_location=device))
    model.eval()
    
    # Load a sample from test set
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data"
    dataset = VibrationDataset(data_directory)
    
    # Get one sample
    sample_idx = 10  # Choose any index
    inputs, label = dataset[sample_idx]
    
    # Initialize explainer
    explainer = ModelExplainer(model, device)
    
    # Get LRP explanation
    lrp_result = explainer.explain_prediction_lrp(inputs)
    
    # Visualize
    explainer.visualize_attribution(lrp_result, "LRP")
    
    # Visualize feature maps
    explainer.visualize_feature_maps("freq_conv1")
