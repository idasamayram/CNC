import torch
import torch.nn as nn
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

def compute_crp_relevance(model, sample, target_label=None, device="cuda"):
    """
    Apply CRP to a single vibration signal.
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


        # This is the important line:
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

    # Pick a layer for concept analysis, e.g., last Conv1d before FC
    record_layer = layer_names[-2] if len(layer_names) > 1 else layer_names[-1]

    # For each channel in the selected layer, compute CRP wrt the target class
    # Here, channel 0 as example; in practice, loop over all or most relevant channels
    conditions = [{record_layer: [0], "y": [target_label]}]

    attr = attribution(sample, conditions, composite, record_layer=layer_names)
    # attr.heatmap is your CRP map for the sample
    # attr.activations, attr.relevances give deeper info

    # Move to numpy for further analysis/plotting if needed
    crp_map = attr.heatmap.squeeze().detach().cpu().numpy()
    return crp_map, attr

