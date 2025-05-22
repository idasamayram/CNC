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


def one_hot(output, index=0, cuda=True):
    '''Get the one-hot encoded value at the provided indices in dim=1'''
    device = output.device  # Use the same device as the output
    values = output[torch.arange(output.shape[0]), index]  # Indexing on the same device
    
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


def zennit_relevance_lrp(input, model, target, zennit_choice="EpsilonPlus", rel_is_model_out=True, cuda=True):
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
    if zennit_choice == "EpsilonPlus":
        lrp = zennit.composites.EpsilonPlus()
    elif zennit_choice == "EpsilonAlpha2Beta1":
        lrp = zennit.composites.EpsilonAlpha2Beta1()

    # Register hooks for rules to all modules that apply
    try:
        lrp.register(model)

        # Execute the hooked/modified model
        print(f"Input device in zennit_relevance_lrp: {input.device}")  # Debug device
        output = model(input)

        target_output = one_hot(output.detach(), target, cuda=(device.type == "cuda"))  # Use updated one_hot
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
