import torch
import torch.nn as nn
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

import torch

def compute_crp_relevance(model, sample, target_label=None, device="cuda"):
    """
    Apply CRP to a single vibration signal with improved handling for CNN1D models.
    Args:
        model: your trained CNN1D model
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
    print(layer_names)

    # Option 1: Use the default approach (no conditions)
    # This will generate a standard relevance heatmap for the target class
    conditions = [{"y": [target_label]}]

    # Option 2: To analyze specific features, use multiple channels
    # Find a layer with significant feature representations (often middle layers work well)
    feature_layer = layer_names[len(layer_names)//2]  # Pick a middle layer
    print(feature_layer)
    # Get the number of channels in that layer
    num_channels = get_layer_channels(model, feature_layer)
    print(num_channels)
    #conditions = [{feature_layer: list(range(num_channels)), "y": [target_label]}]
    #print(conditions)


    # Option 3: Try multiple conditions to see which produces better heatmaps
    # conditions = []
    # for layer_name in layer_names:
    #     conditions.append({layer_name: [], "y": [target_label]})

    attr = attribution(sample, conditions, composite, record_layer=layer_names)

    # Analyze the relevance of each channel/concept in the selected layer
    if len(layer_names) > 0:
        # Get a meaningful layer to analyze
        analysis_layer = layer_names[-3] if len(layer_names) > 1 else layer_names[-1]
        if analysis_layer in attr.relevances:
            # Find the most relevant channels
            rel_c = cc.attribute(attr.relevances[analysis_layer])
            most_relevant_channels = torch.argsort(rel_c, descending=True)
            print(f"Most relevant channels in {analysis_layer}: {most_relevant_channels[:5].cpu().numpy()}")

    # Move to numpy for further analysis/plotting
    crp_map = attr.heatmap.squeeze().detach().cpu().numpy()
    return crp_map, attr


# Helper function to get number of channels in a layer
def get_layer_channels(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Conv1d):
                return module.out_channels
            elif isinstance(module, nn.Linear):
                return module.out_features
    return 0