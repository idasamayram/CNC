import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from torch.utils.data import DataLoader

# Required imports for CRP
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import CondAttribution
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.concepts import ChannelConcept

# Import your model and dataset classes
# If the code below is in a different file, you'll need to import these from your model file
from Classification.cnn1D_model import CNN1D_DS, VibrationDataset


# Helper function to get layer names
def get_layer_names(model, layer_types):
    """Get names of layers of specified types in the model."""
    layer_names = []
    for name, module in model.named_modules():
        if any(isinstance(module, layer_type) for layer_type in layer_types):
            layer_names.append(name)
    return layer_names


# Helper function to get number of channels in a layer
def get_layer_channels(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Conv1d):
                return module.out_channels
            elif isinstance(module, nn.Linear):
                return module.out_features
    return 0


def compute_crp_relevance(model, sample, target_label=None, device="cuda"):
    """
    Apply CRP to a single vibration signal with improved handling for CNN1D models.
    Args:
        model: your trained CNN1D_DS model
        sample: torch tensor (shape: [3, 2000] or [1, 3, 2000])
        target_label: int or None (if None, use model prediction)
        device: 'cuda' or 'cpu'
    Returns:
        crp_heatmap: np.ndarray, same shape as input signal
        attr_obj: full CRP attribution output (activations, relevances, etc.)
    """
    # Ensure correct shape and device
    if isinstance(sample, torch.Tensor):
        if sample.ndim == 2:
            sample = sample.unsqueeze(0)  # add batch dimension
        sample = sample.to(device)
        sample = sample.clone().detach().requires_grad_(True)
    else:
        raise ValueError("Input must be a torch.Tensor")

    model = model.to(device)
    model.eval()

    # Get prediction if label not provided
    if target_label is None:
        with torch.no_grad():
            out = model(sample)
            target_label = out.argmax(1).item()

    # Setup CRP
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    attribution = CondAttribution(model)
    cc = ChannelConcept()

    # Find Conv1d and Linear layers in your model
    layer_names = get_layer_names(model, [nn.Conv1d, nn.Linear])

    # Option 1: Use the default approach (no conditions)
    # This will generate a standard relevance heatmap for the target class
    conditions = [{"y": [target_label]}]

    # Option 2: To analyze specific features, use multiple channels
    # Find a layer with significant feature representations (often middle layers work well)
    # feature_layer = layer_names[len(layer_names)//2]  # Pick a middle layer
    # Get the number of channels in that layer
    # num_channels = get_layer_channels(model, feature_layer)
    # conditions = [{feature_layer: list(range(num_channels)), "y": [target_label]}]

    # Option 3: Try multiple conditions to see which produces better heatmaps
    # conditions = []
    # for layer_name in layer_names:
    #     conditions.append({layer_name: [], "y": [target_label]})

    attr = attribution(sample, conditions, composite, record_layer=layer_names)

    # Analyze the relevance of each channel/concept in the selected layer
    if len(layer_names) > 0:
        # Get a meaningful layer to analyze
        analysis_layer = layer_names[-2] if len(layer_names) > 1 else layer_names[-1]
        if analysis_layer in attr.relevances:
            # Find the most relevant channels
            rel_c = cc.attribute(attr.relevances[analysis_layer])
            most_relevant_channels = torch.argsort(rel_c, descending=True)
            print(f"Most relevant channels in {analysis_layer}: {most_relevant_channels[:5].cpu().numpy()}")

    # Move to numpy for further analysis/plotting
    crp_map = attr.heatmap.squeeze().detach().cpu().numpy()
    return crp_map, attr


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your trained model
    model = CNN1D_DS()
    model_path = "../cnn1d_model.ckpt"  # Update with your model path
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load a few samples for analysis - using your dataset class
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data"  # Update with your data path
    dataset = VibrationDataset(data_directory)

    # Create a small test loader to get a few samples
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Analyze multiple samples to compare good vs bad predictions
    num_samples = 5  # Adjust as needed
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))

    for i, (data, label) in enumerate(test_loader):
        if i >= num_samples:
            break

        data = data.to(device)
        true_label = label.item()  # 0=good, 1=bad
        label_name = "bad" if true_label == 1 else "good"

        # Get model prediction
        with torch.no_grad():
            output = model(data)
            pred_label = output.argmax(1).item()
            pred_name = "bad" if pred_label == 1 else "good"

        # Generate CRP heatmap for the predicted class
        crp_map, _ = compute_crp_relevance(model, data, target_label=pred_label, device=device)

        # Plot the input signal
        axes[i, 0].set_title(f"Sample {i + 1}: True={label_name}, Pred={pred_name}")
        for c in range(data.shape[1]):  # For each channel (x, y, z)
            axes[i, 0].plot(data[0, c].cpu().numpy(), label=f'Channel {c + 1}')
        axes[i, 0].legend()
        axes[i, 0].set_xlabel('Time Steps')
        axes[i, 0].set_ylabel('Amplitude')

        # Plot the CRP heatmap
        im = axes[i, 1].imshow(crp_map, aspect='auto', cmap='coolwarm')
        axes[i, 1].set_title(f"CRP Relevance Heatmap (class={pred_name})")
        axes[i, 1].set_xlabel('Time Steps')
        axes[i, 1].set_ylabel('Channels')
        plt.colorbar(im, ax=axes[i, 1])

    plt.tight_layout()
    plt.savefig('crp_analysis_results.png')
    plt.show()

    # In-depth analysis for one example
    print("\nDetailed analysis for a specific sample:")
    # Get a sample (could modify to select one of each class)
    for data, label in test_loader:
        true_label = label.item()
        break  # Just take the first sample

    # Analyze both classes for comparison
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    for idx, target_class in enumerate([0, 1]):  # 0=good, 1=bad
        class_name = "good" if target_class == 0 else "bad"

        # Generate CRP heatmap specifically for this target class
        crp_map, attr = compute_crp_relevance(model, data, target_label=target_class, device=device)

        # Plot as a heatmap
        im = axes[idx].imshow(crp_map, aspect='auto', cmap='coolwarm')
        axes[idx].set_title(f"CRP Relevance for Class '{class_name}'")
        axes[idx].set_xlabel('Time Steps')
        axes[idx].set_ylabel('Channels')
        plt.colorbar(im, ax=axes[idx])

        # Find the most influential time points
        top_indices = []
        for c in range(crp_map.shape[0]):
            # Get indices of top 5 values in this channel
            channel_indices = np.argsort(crp_map[c])[-5:]
            top_indices.append((c, channel_indices))
            print(f"Class {class_name}, Channel {c + 1}: Top 5 influential time points at {channel_indices}")

    plt.tight_layout()
    plt.savefig('crp_class_comparison.png')
    plt.show()

    # Optional: Layer-wise relevance analysis
    print("\nLayer-wise relevance analysis:")
    _, attr = compute_crp_relevance(model, data, device=device)

    # Analyzing relevance for each layer
    layer_relevances = []
    for layer_name, relevance in attr.relevances.items():
        if isinstance(relevance, torch.Tensor):
            total_rel = relevance.abs().sum().item()
            layer_relevances.append((layer_name, total_rel))
            print(f"Layer {layer_name}: Total relevance = {total_rel:.4f}")

    # Plot layer-wise relevance
    if layer_relevances:
        names, values = zip(*layer_relevances)
        plt.figure(figsize=(12, 6))
        plt.bar(names, values)
        plt.xlabel('Layer Name')
        plt.ylabel('Total Absolute Relevance')
        plt.title('Layer-wise Relevance Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('layer_relevance.png')
        plt.show()


if __name__ == "__main__":
    main()