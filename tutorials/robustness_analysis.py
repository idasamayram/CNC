# time_series_robustness.py

import torch
import numpy as np
from tqdm import tqdm
import gc
import torch
from Classification.cnn1D_model import CNN1D_Wide
from utils.baseline_xai import load_model, load_sample_data
from torch.utils.data import DataLoader


def continuity_metric(model, sample, attribution_method, device="cuda" if torch.cuda.is_available() else "cpu",
                      sigma=0.01, n_perturbations=10):
    """
    Measure the continuity of the explanation method.
    A good explanation should be continuous w.r.t. small perturbations in input.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        device: Device to run computations on
        sigma: Standard deviation of the Gaussian noise for perturbation
        n_perturbations: Number of perturbations to try

    Returns:
        continuity_score: Measure of how continuous the explanation is (1 = perfect)
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get baseline explanation
    explanation_base = attribution_method(model, sample)

    # If explanation_base is a tuple (as in some LRP methods), take the first element
    if isinstance(explanation_base, tuple):
        explanation_base = explanation_base[0]

    # Ensure explanation_base is a tensor
    if not isinstance(explanation_base, torch.Tensor):
        explanation_base = torch.tensor(explanation_base, device=device, dtype=torch.float32)
    else:
        explanation_base = explanation_base.to(device).float()

    # Normalize base explanation
    explanation_base = (explanation_base - explanation_base.min()) / (
                explanation_base.max() - explanation_base.min() + 1e-8)

    max_diff = 0

    # Generate perturbed inputs and measure differences in explanations
    for _ in range(n_perturbations):
        # Add small Gaussian noise to input
        noise = torch.randn_like(sample) * sigma
        sample_perturbed = (sample + noise).detach().clone().requires_grad_()

        # Get explanation for perturbed input
        explanation_perturbed = attribution_method(model, sample_perturbed)

        # If explanation_perturbed is a tuple, take the first element
        if isinstance(explanation_perturbed, tuple):
            explanation_perturbed = explanation_perturbed[0]

        # Ensure explanation_perturbed is a tensor
        if not isinstance(explanation_perturbed, torch.Tensor):
            explanation_perturbed = torch.tensor(explanation_perturbed, device=device, dtype=torch.float32)
        else:
            explanation_perturbed = explanation_perturbed.to(device).float()

        # Normalize perturbed explanation
        explanation_perturbed = (explanation_perturbed - explanation_perturbed.min()) / (
                    explanation_perturbed.max() - explanation_perturbed.min() + 1e-8)

        # Calculate maximum absolute difference
        diff = torch.abs(explanation_base - explanation_perturbed).max().item()
        if diff > max_diff:
            max_diff = diff

    # Continuity score: 1 - max_diff (1 = perfect continuity)
    continuity_score = 1 - max_diff

    return continuity_score


def local_lipschitz_metric(model, sample, attribution_method, device="cuda" if torch.cuda.is_available() else "cpu",
                           n_samples=10, noise_std=0.01, norm_type=2):
    """
    Measure the local Lipschitz constant of the explanation method.
    A lower Lipschitz constant indicates the explanation is more robust to input changes.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_method: Function that generates attributions for the input
        device: Device to run computations on
        n_samples: Number of perturbations to try
        noise_std: Standard deviation of the Gaussian noise for perturbation
        norm_type: Type of norm to use for the Lipschitz calculation (1, 2, etc.)

    Returns:
        lipschitz: Maximum estimated Lipschitz constant
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)

    # Get baseline explanation
    explanation_base = attribution_method(model, sample)

    # If explanation_base is a tuple, take the first element
    if isinstance(explanation_base, tuple):
        explanation_base = explanation_base[0]

    # Ensure explanation_base is a tensor
    if not isinstance(explanation_base, torch.Tensor):
        explanation_base = torch.tensor(explanation_base, device=device)
    else:
        explanation_base = explanation_base.to(device)

    max_lipschitz = 0.0

    # Generate perturbed inputs and measure Lipschitz constants
    for _ in range(n_samples):
        # Add small Gaussian noise to input
        noise = torch.randn_like(sample) * noise_std
        sample_perturbed = (sample + noise).detach().clone().requires_grad_()

        # Get explanation for perturbed input
        explanation_perturbed = attribution_method(model, sample_perturbed)

        # If explanation_perturbed is a tuple, take the first element
        if isinstance(explanation_perturbed, tuple):
            explanation_perturbed = explanation_perturbed[0]

        # Ensure explanation_perturbed is a tensor
        if not isinstance(explanation_perturbed, torch.Tensor):
            explanation_perturbed = torch.tensor(explanation_perturbed, device=device)
        else:
            explanation_perturbed = explanation_perturbed.to(device)

        # Compute norm of explanation difference
        diff_exp = explanation_perturbed - explanation_base
        norm_exp = torch.norm(diff_exp.view(-1), p=norm_type)

        # Compute norm of input difference
        diff_x = sample_perturbed - sample
        norm_x = torch.norm(diff_x.view(-1), p=norm_type)

        if norm_x.item() > 0:
            lipschitz_estimate = (norm_exp / norm_x).item()
            if lipschitz_estimate > max_lipschitz:
                max_lipschitz = lipschitz_estimate

    return max_lipschitz


def evaluate_robustness(model, sample, attribution_methods,
                        device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate robustness metrics for multiple attribution methods.

    Args:
        model: Trained PyTorch model
        sample: Time series input tensor of shape (3, time_steps)
        attribution_methods: Dictionary of {method_name: attribution_function}
        device: Device to run computations on

    Returns:
        results0: Dictionary with robustness metrics for each method
    """
    results = {}

    for method_name, attribution_func in attribution_methods.items():
        print(f"Evaluating robustness metrics for {method_name}...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        try:
            # Measure continuity
            continuity = continuity_metric(model, sample, attribution_func, device)

            # Measure local Lipschitz constant
            lipschitz = local_lipschitz_metric(model, sample, attribution_func, device)

            # Store results0
            results[method_name] = {
                "continuity": continuity,
                "lipschitz": lipschitz
            }

            print(f"  Continuity: {continuity:.4f}")
            print(f"  Lipschitz: {lipschitz:.4f}")

        except Exception as e:
            print(f"  Error evaluating {method_name}: {str(e)}")
            results[method_name] = {
                "continuity": float('nan'),
                "lipschitz": float('nan')
            }

    return results

if __name__ == "__main__":
    # Example usage
    # Define your model and sample here
    # Example usage



    # Load model
    model_path = "../cnn1d_model_wide_new.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    model.eval()

    # Load data
    data_dir = "../data/final/new_selection/normalized_windowed_downsampled_data_lessBAD"
    samples, labels, _ = load_sample_data(data_dir, num_samples=1)
    sample = samples[0].to(device)
    label = labels[0]

    # Define attribution methods
    attribution_methods = {
        "LRP": lrp_attribution_wrapper,
        "DFT-LRP": dft_lrp_attribution_wrapper,
        "Gradient*Input": grad_times_input_wrapper,
        "SmoothGrad": smoothgrad_wrapper,
        "Occlusion": occlusion_wrapper
    }

    # Compare methods using pixel flipping
    results = compare_attribution_methods(
        model=model,
        sample=sample,
        attribution_methods=attribution_methods,
        label=label,
        n_steps=20,
        most_relevant_first=True,
        device=device
    )

    # Evaluate robustness
    robustness_results = evaluate_robustness(
        model=model,
        sample=sample,
        attribution_methods=attribution_methods,
        device=device
    )

    # Print summaries
    print("\nPixel Flipping Results:")
    for method_name, result in results.items():
        print(f"{method_name}: AUC = {result['auc']:.4f}")

    print("\nRobustness Results:")
    for method_name, metrics in robustness_results.items():
        print(f"{method_name}: Continuity = {metrics['continuity']:.4f}, Lipschitz = {metrics['lipschitz']:.4f}")