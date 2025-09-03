import torch
import numpy as np
from Classification.cnn1D_model import CNN1D_Wide
from pathlib import Path
import h5py
import random
from utils.lrp_utils import  zennit_relevance_lrp
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device, model=CNN1D_Wide):
    """Load a trained CNN1D_Wide model from checkpoint"""

    model = model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_single(model, x, detach=False):
    """
    Perform a single prediction on input data using the model.
    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        detach: If True, detach the tensor to avoid retaining the computational graph.
    Returns:
        prediction: Model output logits (before softmax).
        ypred: Predicted class label (0 or 1).
    """
    # if x has batch dimension, remove it
    if x.dim() == 3:
        x = x.squeeze(0)
    prediction = model.forward(x.unsqueeze(0)).to(device)  # Add batch dimension
    prediction = prediction[0]  # Remove batch dimension

    if detach:
        prediction = prediction.detach()

    ypred = torch.argmax(prediction).item()  # Predicted class

    return prediction, ypred

def load_sample_data(data_dir, num_samples=3, label=None):
    """
    Load sample data from the dataset

    Args:
        data_dir: Path to data directory
        num_samples: Number of samples to load
        label: If specified, only load samples with this label (0=good, 1=bad)

    Returns:
        samples: List of sample tensors
        labels: List of corresponding labels
        sample_paths: List of sample file paths
    """
    data_dir = Path(data_dir)
    samples = []
    labels = []
    sample_paths = []

    # Define which folders to search based on label
    if label is not None:
        label_folders = [["good", "bad"][label]]
    else:
        label_folders = ["good", "bad"]

    # Collect file paths
    for folder_name in label_folders:
        folder = data_dir / folder_name
        for file_name in folder.glob("*.h5"):
            sample_paths.append((file_name, 0 if folder_name == "good" else 1))
            if len(sample_paths) >= num_samples * 100:  # Collect extra to allow for random selection
                break

    # Randomly select samples if we have more than requested
    if len(sample_paths) > num_samples:
        random.shuffle(sample_paths)  # Set seed for reproducibility
        sample_paths = sample_paths[:num_samples]

    # Load the actual data
    for file_path, label_idx in sample_paths:
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        # Transpose to (3, 2000) for CNN
        data = np.transpose(data, (1, 0))
        samples.append(torch.tensor(data, dtype=torch.float32))
        labels.append(label_idx)

    return samples, labels, [str(path) for path, _ in sample_paths]

def gradient_relevance(model, x, target=None):
    """
    Compute gradients for the given input and target.
    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        target: Target class for explanation (default: model's prediction).
    Returns:
        grad: Gradients w.r.t. input.
        target: Target class used for gradient computation.
    """
    # Ensure x is detached and a leaf tensor, then enable gradients
    x = x.detach().clone()
    x.requires_grad = True  # Enable gradient computation

    y_pred, y = predict_single(model, x)
    if target is None:
        target = y

    # Compute gradients
    grad, = torch.autograd.grad(y_pred[target], x, y_pred[target])
    # WHICH IS EQUIVALENT TO: y_pred[target].backward(y_pred[target]),  grad = x.grad
    # introduces a scaling factor(value of y_pred[target])

    #    # Alternative way to compute gradients
    # grad = torch.autograd.grad(y_pred[target], x, retain_graph=True)[0]  # Alternative way to compute gradients
    # which is equivalent to grad, = torch.autograd.grad(y_pred[target], x, torch.ones_like(y_pred[target]))
    return grad, target

def grad_times_input_relevance(model, x, target=None):
    """
    Compute Grad*Input explanation for the given input.
    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        target: Target class for explanation (default: model's prediction).
    Returns:
        attribution: Grad*Input attributions.
    """
    grad, target = gradient_relevance(model, x, target)

    return grad * x, target  # Multiply gradients by input

def smoothgrad_relevance(model, x, num_samples=40, noise_level=1, target=None): # num_sample = 200 for normal signal, for sownsampled=40
    """
    Compute SmoothGrad explanation for the given input.
    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        num_samples: Number of noisy samples to generate (default: 200).
        noise_level: Standard deviation of added noise (default: 3).
        target: Target class for explanation (default: model's prediction).
    Returns:
        sgrad: SmoothGrad attributions (averaged gradients).
        target: Target class used for explanation.
    """
    # Ensure x is a leaf tensor with requires_grad
    x = x.clone().requires_grad_(True)  # Create a new leaf tensor with gradients enabled

    # Compute gradients for the original signal
    sgrad, target = gradient_relevance(model, x, target)
    noise_std = torch.std(x)  # Global standard deviation

    # Add noisy samples and accumulate gradients
    for i in range(1, num_samples):
        noisy_x = torch.clone(x.detach()) + torch.randn_like(x) * noise_level * noise_std
        sgrad += gradient_relevance(model, noisy_x, target)[0]

    # Average accumulated gradients
    sgrad /= num_samples

    return sgrad * x, target

def occlusion_signal_relevance(model, x, target=None, occlusion_type="zero"):
    """
    Compute occlusion-based explanation for time-series signal.
    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        target: Target class for explanation (default: model's prediction).
        occlusion_type: Type of occlusion ("zero", "one", "mone", "flip").
    Returns:
        attribution: Occlusion-based attributions (3, time_steps).
    """
    # Define occlusion functions

    def zero(val):
        return torch.zeros_like(val)

    def one(val):
        return torch.ones_like(val)

    def mone(val):
        return -torch.ones_like(val)

    def flip(val):
        return -val

    occlusion_fxns = {"zero": zero, "one": one, "mone": mone, "flip": flip}
    assert occlusion_type in occlusion_fxns, f"Invalid occlusion type: {occlusion_type}"

    # Original prediction
    x_0 = x.detach().clone()
    pred_0, y_0 = predict_single(model, x_0, detach=True)
    if target is None:
        target = y_0

    # Prepare attribution mask
    attributions = torch.zeros_like(x)

    # Iterate over time steps and axes
    for feature_idx in range(x.shape[0]):  # X, Y, Z axes
        for time_idx in range(x.shape[1]):  # Time steps
            x_copy = x.clone()
            x_copy[feature_idx, time_idx] = occlusion_fxns[occlusion_type](x_copy[feature_idx, time_idx])
            pred, _ = predict_single(model, x_copy, detach=True)
            attributions[feature_idx, time_idx] = pred_0[target] - pred[target]

    return attributions, target

def occlusion_simpler_relevance(model, x, target=None, occlusion_type="zero", window_size=20):  #windowsize=40 for downsampled, 200 for normal signal
    """
    Compute occlusion-based explanation for time-series signals.

    Args:
        model: Trained PyTorch model.
        x: Input tensor (time-series signal, shape: (3, time_steps)).
        target: Target class for explanation (default: model's prediction).
        occlusion_type: Type of occlusion ("zero", "one", "mone", "flip").
        window_size: Number of consecutive time steps to occlude.

    Returns:
        attribution: Occlusion-based attributions (3, time_steps).
    """
    def zero(val):
        return torch.zeros_like(val)

    def one(val):
        return torch.ones_like(val)

    def mone(val):
        return -torch.ones_like(val)

    def flip(val):
        return -val

    occlusion_fxns = {"zero": zero, "one": one, "mone": mone, "flip": flip}
    assert occlusion_type in occlusion_fxns, f"Invalid occlusion type: {occlusion_type}"

    # Get the original prediction before occlusion
    x_0 = x.detach().clone()
    pred_0, y_0 = predict_single(model, x_0, detach=True)
    if target is None:
        target = y_0  # Use model's prediction if no target is provided

    # Prepare attribution mask (same shape as input)
    attributions = torch.zeros_like(x)

    # Iterate over time steps in window_size chunks
    for i in range(0, x.shape[1], window_size):  # Slide occlusion window
        # Apply occlusion for each feature/axis separately
        for feature_idx in range(x.shape[0]):  # X, Y, Z axes
            x_copy = x.clone()  # Copy original input

            # Apply occlusion only to the current feature/axis
            x_copy[feature_idx, i:i + window_size] = occlusion_fxns[occlusion_type](
                x_copy[feature_idx, i:i + window_size])

            # Get new prediction after occlusion
            pred, _ = predict_single(model, x_copy, detach=True)

            # Compute attribution for this axis only
            attributions[feature_idx, i:i + window_size] = pred_0[target] - pred[target]

    return attributions, target

def compute_lrp_relevance(model, sample, label=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute LRP relevances for a single vibration sample.

    Args:
        model: Trained CNN1D model
        sample: Numpy array or torch tensor of shape (3, 10000) for the vibration data   (3,2000) for downsampled
        label: Optional integer label (0 or 1) for the sample. If None, use model prediction as target
        device: Torch device (CPU or CUDA)

    Returns:
        relevance: Numpy array of shape (3, 10000) containing LRP relevances
        input_signal: Numpy array of the input signal (for visualization)
        predicted_label: Predicted label if label is None
    """
    # Ensure sample is a PyTorch tensor with shape (1, 3, 2000)
    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(
            0)  # Add batch dimension and move to device
    else:
        sample = sample.to(device).unsqueeze(0)  # Assume it's already a tensor, add batch dimension and move to device

    # Move model to the specified device (already done in your script, but ensure consistency)
    model = model.to(device)
    model.eval()

    # Make sure model and sample are on the same device
    if next(model.parameters()).device != sample.device:
        print(f"Warning: Model device ({next(model.parameters()).device}) doesn't match sample device ({sample.device})")
        print("Moving model to match sample device")
        model = model.to(sample.device)

    # Debug: Print device information
    print(f"Sample device: {sample.device}")
    print(f"Model device: {next(model.parameters()).device}")

    # If no label provided, use model prediction as target
    if label is None:
        with torch.no_grad():
            try:
                outputs = model(sample)
                _, predicted_label = torch.max(outputs, 1)  # Get predicted class (0 or 1)
                target = predicted_label.item()
            except Exception as e:
                raise RuntimeError(f"Error during model prediction: {e}")
    else:
        target = label.item() if isinstance(label, torch.Tensor) else label
        target = torch.tensor([target], device=device)  # Ensure target is on the same device

    # Debug: Print target device
    print(f"Target device: {target.device}")

    # Compute LRP relevances using Zennit
    try:
        relevance = zennit_relevance_lrp(
            input=sample,
            model=model,
            target=target,  # Target is already on the correct device
            RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
            rel_is_model_out=True,  # Relevance is model output (logits)
            cuda=(device == "cuda")
        )
    except RuntimeError as e:
        print(f"Error in zennit_relevance: {e}")
        # Try to recover by falling back to CPU if possible
        if device == "cuda":
            print("Falling back to CPU for LRP computation")
            device = "cpu"
            model = model.cpu()
            sample = sample.cpu()
            if isinstance(target, torch.Tensor):
                target = target.cpu()
            try:
                relevance = zennit_relevance_lrp(
                    input=sample,
                    model=model,
                    target=target,
                    RuleComposite="CustomLayerMap",  # Use custom layer map for cnn1d network
                    rel_is_model_out=True,
                    cuda=False
                )
            except Exception as e2:
                raise RuntimeError(f"Error in LRP computation on CPU fallback: {e2}")
        else:
            raise

    # Remove batch dimension and convert to numpy
    relevance = relevance.squeeze(0)  # Shape: (3, 2000)
    input_signal = sample.squeeze(0).detach().cpu().numpy()  # Shape: (3, 2000)

    return relevance, input_signal, target.item() if isinstance(target, torch.Tensor) else target

def summarize_attributions(attributions):
    """
    Summarize the positive and negative attributions for each axis, including counts, averages, and totals.
    Args:
        attributions: Attribution values (shape: (3, time_steps)).
    Returns:
        summary: Dictionary with detailed attribution statistics per axis.
    """
    summary = {}
    for axis, attr in enumerate(attributions):
        # Positive relevance
        positive_values = attr[attr > 0]
        positive_count = len(positive_values)
        total_positive_relevance = np.sum(positive_values)
        average_positive_relevance = total_positive_relevance / positive_count if positive_count > 0 else 0

        # Negative relevance
        negative_values = attr[attr < 0]
        negative_count = len(negative_values)
        total_negative_relevance = np.sum(negative_values)
        average_negative_relevance = total_negative_relevance / negative_count if negative_count > 0 else 0

        # Total relevance (positive + negative)
        total_relevance = total_positive_relevance + total_negative_relevance
        total_count = positive_count + negative_count
        average_relevance = total_relevance / total_count if total_count > 0 else 0

        # Store results0
        summary[f"Axis {axis}"] = {
            "Positive Count": positive_count,

            "Total Positive Relevance": total_positive_relevance,

            "Average Positive Relevance": average_positive_relevance,

            "Negative Count": negative_count,

            "Total Negative Relevance": total_negative_relevance,

            "Average Negative Relevance": average_negative_relevance,

            "Total Relevance (Pos + Neg)": total_relevance,

            "Average Relevance (Pos + Neg)": average_relevance,
        }
    return summary

