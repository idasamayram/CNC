import torch
import torch.nn as nn
import zennit.composites
import zennit.rules
import zennit.core
import zennit.attribution
from zennit.types import Linear
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import gc
from zennit.composites import LayerMapComposite, SpecialFirstLayerMapComposite
from zennit.rules import Epsilon, ZPlus, Pass, Gamma, AlphaBeta,ZBox, Flat
from zennit.composites import EpsilonPlus, layer_map_base
from zennit.core import Hook
from zennit.core import Composite


class GroupNormPass_fullcheck(zennit.core.Hook):
    def backward(self, module, grad_input, grad_output):
        """Pass through all gradients in the proper format"""
        # Check how many gradients PyTorch expects (based on grad_input)
        if grad_input is None or len(grad_input) == 0:
            return (grad_output[0],)  # Default case

        # Return a tuple with the same number of elements as grad_input
        return tuple(grad_output[i] if i < len(grad_output) else None
                     for i in range(len(grad_input)))


# First create the custom hook for GroupNorm
class GroupNormPass(zennit.core.Hook):
    '''Custom Pass rule specifically for GroupNorm layers.'''

    def backward(self, module, grad_input, grad_output):
        '''Pass through the upper gradient with correct formatting.'''
        return (grad_output[0],)

custom_layer_map_cnn1d = [
    # Pass through activations
    (nn.ReLU, zennit.rules.Pass()),

    # Max Pooling - use Norm rule
    (nn.MaxPool1d, zennit.rules.Norm()),

    # Convolutional layers - use ZPlus rule for better visual interpretability
    (nn.Conv1d, zennit.rules.Gamma()),

    # Group normalization - pass through
    (nn.BatchNorm1d, zennit.rules.Pass()),

    # Group normalization - use our custom pass rule
    (nn.GroupNorm, GroupNormPass()),  # <-- This is the key change

    # Adaptive pooling - use Norm rule
    (nn.AdaptiveAvgPool1d, zennit.rules.Norm()),

    # Fully connected layers - use Epsilon rule
    (nn.Linear, zennit.rules.Epsilon(epsilon=0))
]

custom_first_map = [
    (nn.Conv1d, zennit.rules.AlphaBeta(alpha=2, beta=1))
]


def one_hot(output, index=0, cuda=True):
    '''Get the one-hot encoded value at the provided indices in dim=1'''
    device = output.device  # Use the same device as the output
    values = output[torch.arange(output.shape[0]), index]  # Indexing on the same device
    print(f'values shape:{values.shape}')
    
    # Handle case where index is a tensor
    if isinstance(index, torch.Tensor) and len(index) > 1:
        # Create mask where each row has a 1 at the position specified by the corresponding index
        mask = torch.zeros_like(output)
        for i, idx in enumerate(index):
            mask[i, idx] = 1.0
        out = values[:, None] * mask
    else:
        # Standard case with a single index or list of indices
        mask = torch.eye(output.shape[1], device=device)[index]  # Create eye matrix on the same device
        out = values[:, None] * mask
        
    return out


def zennit_relevance(input, model, target, attribution_method="lrp", zennit_choice="EpsilonPlus", rel_is_model_out=True,
                     cuda=True):
    """
    Compute relevance scores using various attribution methods.
    
    Args:
        input: Input tensor to explain
        model: PyTorch model
        target: Target class for explanation
        attribution_method: Method to use ("lrp", "gxi", "sensitivity", "ig")
        zennit_choice: LRP rule to use when attribution_method="lrp"
        rel_is_model_out: Whether relevance is model output
        cuda: Whether to use GPU
    
    Returns:
        relevance: Attribution scores with same shape as input
    """
    # Use the device of the input tensor directly
    device = input.device
    input = input.clone().detach().requires_grad_(True).to(device)  # Explicitly move to the input's device
    print(f"Input device in zennit_relevance: {input.device}")  # Debug device

    # Verify target is on the same device as input
    if isinstance(target, torch.Tensor) and target.device != device:
        print(f"Warning: Target device ({target.device}) doesn't match input device ({device})")
        target = target.to(device)

    try:
        if attribution_method == "lrp":
            relevance = zennit_relevance_lrp(input, model, target, zennit_choice, rel_is_model_out, cuda=(device.type == "cuda"))
        elif attribution_method == "gxi" or attribution_method == "sensitivity":
            attributer = zennit.attribution.Gradient(model)
            _, relevance = attributer(input, partial(one_hot, index=target, cuda=(device.type == "cuda")))
            relevance = relevance.detach().cpu().numpy()
            if attribution_method == "gxi":
                relevance = relevance * input.detach().cpu().numpy()
            elif attribution_method == "sensitivity":
                relevance = relevance ** 2
        elif attribution_method == "ig":
            attributer = zennit.attribution.IntegratedGradients(model)
            _, relevance = attributer(input, partial(one_hot, index=target, cuda=(device.type == "cuda")))
            relevance = relevance.detach().cpu().numpy()
    except Exception as e:
        print(f"Error in zennit_relevance with method {attribution_method}: {e}")
        if cuda and device.type == "cuda":
            print("Trying to fall back to CPU")
            input_cpu = input.cpu()
            model_cpu = model.cpu()
            target_cpu = target.cpu() if isinstance(target, torch.Tensor) else target
            
            try:
                # Call the appropriate function based on attribution method instead of recursively calling this function
                if attribution_method == "lrp":
                    return zennit_relevance_lrp(input_cpu, model_cpu, target_cpu, zennit_choice, rel_is_model_out, False)
                elif attribution_method == "gxi" or attribution_method == "sensitivity":
                    attributer = zennit.attribution.Gradient(model_cpu)
                    _, relevance = attributer(input_cpu, partial(one_hot, index=target_cpu, cuda=False))
                    relevance = relevance.detach().cpu().numpy()
                    if attribution_method == "gxi":
                        relevance = relevance * input_cpu.detach().cpu().numpy()
                    elif attribution_method == "sensitivity":
                        relevance = relevance ** 2
                    return relevance
                elif attribution_method == "ig":
                    attributer = zennit.attribution.IntegratedGradients(model_cpu)
                    _, relevance = attributer(input_cpu, partial(one_hot, index=target_cpu, cuda=False))
                    return relevance.detach().cpu().numpy()
            except Exception as e2:
                print(f"CPU fallback also failed: {e2}")
                raise
        raise
        
    # Clean up to free memory
    torch.cuda.empty_cache()
    gc.collect()
    
    return relevance


def zennit_relevance_lrp(input, model, target, RuleComposite=None, rel_is_model_out=True, cuda=True):
    """
    Compute Layer-wise Relevance Propagation using Zennit.
    
    Args:
        input: Input tensor to explain
        model: PyTorch model
        target: Target class for explanation
        zennit_choice: LRP rule to use ("EpsilonPlus" or "EpsilonAlpha2Beta1")
        rel_is_model_out: Whether relevance is model output
        cuda: Whether to use GPU
        
    Returns:
        relevance: LRP attribution scores with same shape as input
    """
    device = input.device  # Use the same device as the input

    # Ensure input requires grad
    input = input.clone().detach().requires_grad_(True).to(device)  # Explicitly move to the input's device
    print(f"Input device in zennit_relevance: {input.device}")  # Debug device


    if RuleComposite == "EpsilonPlus" or RuleComposite == None:
        lrp = EpsilonPlus()
    elif RuleComposite == "CustomLayerMap":
        lrp = LayerMapComposite(custom_layer_map_cnn1d)
    elif RuleComposite == "CustomFirstLayerMap":
        lrp = SpecialFirstLayerMapComposite(
                layer_map=custom_layer_map_cnn1d,
                first_map=custom_first_map,
             )

    # Print model structure to see what layers we're working with
    print("Model structure:")
    for name, module in model.named_modules():
        print(f"  - {name}: {type(module).__name__}")


    # Register hooks for rules to all modules that apply

    try:
        lrp.register(model)
        print("Successfully registered hooks")


        # Execute the hooked/modified model
        print(f"Input device in zennit_relevance_lrp: {input.device}")  # Debug device
        output = model(input)
        print(f"Model output shape: {output.shape}")


        target_output = one_hot(output.detach(), target, cuda=(device.type == "cuda"))  # Use updated one_hot
        print(f"Target output shape: {target_output.shape, target_output}")  # Debug target output shape


        if not rel_is_model_out:
            if isinstance(target, torch.Tensor) and len(target) > 1:
                # Handle batched targets
                for i, t in enumerate(target):
                    target_output[i, t] = torch.sign(output[i, t])
            else:
                # Single target case
                target_output[:, target] = torch.sign(output[:, target])

        # Compute the attribution via the gradient
        # The grad_outputs parameter specifies how to weight the gradients of each output element
        # In this case, target_output is a one-hot vector that selects and weighs the target class



        relevance = torch.autograd.grad(output, input, grad_outputs=target_output)[0]

        # Remove all hooks, undoing the modification
        lrp.remove()
    except Exception as e:
        # Make sure we clean up hooks even if there's an error
        try:
            lrp.remove()
        except:
            pass
        raise e

    return relevance.cpu().numpy()
