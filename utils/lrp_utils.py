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


def one_hot(output, index=0, cuda=True):
    '''Get the one-hot encoded value at the provided indices in dim=1'''
    device = output.device  # Use the same device as the output
    values = output[torch.arange(output.shape[0]), index]  # Indexing on the same device
    mask = torch.eye(output.shape[1], device=device)[index]  # Create eye matrix on the same device
    out = values[:, None] * mask
    return out


def zennit_relevance(input, model, target, attribution_method="lrp", zennit_choice="EpsilonPlus", rel_is_model_out=True,
                     cuda=True):
    # Use the device of the input tensor directly
    device = input.device
    input = input.clone().detach().requires_grad_(True).to(device)  # Explicitly move to the input's device
    print(f"Input device in zennit_relevance: {input.device}")  # Debug device

    if attribution_method == "lrp":
        relevance = zennit_relevance_lrp(input, model, target, zennit_choice, rel_is_model_out, cuda)
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
    return relevance


def zennit_relevance_lrp(input, model, target, zennit_choice="EpsilonPlus", rel_is_model_out=True, cuda=True):
    """
    zennit_choice: str, zennit rule or composite
    """
    device = input.device  # Use the same device as the input
    if zennit_choice == "EpsilonPlus":
        lrp = zennit.composites.EpsilonPlus()
    elif zennit_choice == "EpsilonAlpha2Beta1":
        lrp = zennit.composites.EpsilonAlpha2Beta1()

    # Register hooks for rules to all modules that apply
    lrp.register(model)

    # Execute the hooked/modified model
    print(f"Input device in zennit_relevance_lrp: {input.device}")  # Debug device
    output = model(input)

    target_output = one_hot(output.detach(), target, cuda=(device.type == "cuda"))  # Use updated one_hot
    if not rel_is_model_out:
        target_output[:, target] = torch.sign(output[:, target])

    # Compute the attribution via the gradient
    relevance = torch.autograd.grad(output, input, grad_outputs=target_output)[0]

    # Remove all hooks, undoing the modification
    lrp.remove()

    return relevance.cpu().numpy()