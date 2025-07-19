import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import psutil
import os
import torch
from IPython.display import display
import time


# Add this to visualization_utils.py

# Add this to visualization_utils.py

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def save_figure(fig, save_dir, filename, dpi=300):
    """Save figure to specified directory"""
    ensure_dir(save_dir)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    return filepath
# Memory tracking function
def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


# Track execution time as a decorator
def track_time(func):
    """Decorator to track function execution time"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
            # Assuming second element is a metrics dictionary
            result[1]['execution_time'] = execution_time
        return result

    return wrapper


# Plot confusion matrix with metrics table
def plot_confmat_and_metrics(y_true, y_pred, class_names=None, title="Confusion Matrix", save_dir=None, filename=None):
    """Plot confusion matrix with metrics table and optionally save to file"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("Only works for binary classification (2 classes).")

    TN, FP, FN, TP = cm.ravel()

    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics = [
        ["Accuracy", f"{accuracy:.3f}"],
        ["Precision", f"{precision:.3f}"],
        ["Recall (TPR)", f"{recall:.3f}"],
        ["Specificity (TNR)", f"{specificity:.3f}"],
        ["F1-score", f"{f1:.3f}"],
        ["TP", TP],
        ["FP", FP],
        ["TN", TN],
        ["FN", FN],
    ]

    # Plot
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.9])

    # Confusion matrix
    ax0 = fig.add_subplot(gs[0])
    if class_names is None:
        class_names = ["Good", "Bad"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax0,
                annot_kws={'size': 18})
    ax0.set_xlabel("Predicted label")
    ax0.set_ylabel("True label")
    ax0.set_title(title, fontsize=16)

    # Table
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    table = ax1.table(
        cellText=metrics,
        colLabels=["Metric", "Value"],
        loc='center',
        cellLoc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#cccccc")
        else:
            cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#e6e6e6")
    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title.replace(' ', '_').lower()}_confusion_matrix.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {filepath}")
    plt.show()

    # Return metrics dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN
    }


# Plot learning curve
def plot_learning_curve(model_name, train_sizes, train_scores, val_scores, title_prefix="Model", save_dir=None, filename=None):
    """Plot learning curve from cross-validation results"""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')

    plt.title(f'Learning Curve - {model_name}', fontsize=16)
    plt.xlabel('Training Set Size', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()


    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title_prefix.replace(' ', '_').lower()}_learning_curve.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved learning curve to {filepath}")


    plt.show()


# Add this function to visualization_utils.py

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,
                          title_prefix="Model", save_dir=None, filename=None):
    """
    Plot training and validation loss/accuracy curves and optionally save to file
    """
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{title_prefix} Training and Validation Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{title_prefix} Training and Validation Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title_prefix.replace(' ', '_').lower()}_training_history.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {filepath}")




def plot_learning_rate_curve(learning_rates, title_prefix="Model", save_dir=None, filename=None):
    """
    Plot learning rate changes over epochs and optionally save to file
    """
    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates, marker='o', markersize=3)
    plt.title(f'Learning Rate Schedule - {title_prefix}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better visualizes LR changes
    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title_prefix.replace(' ', '_').lower()}_learning_rate_curve.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved learning rate curve to {filepath}")



# Visualize filters from CNN1D model
def visualize_cnn_filters(model, title=None, save_dir=None):
    """
    Visualize the filters from the first convolutional layer of a CNN1D model with
    detailed interpretations and save to specified directory.

    Args:
        model: The trained CNN model
        title: Optional custom title for the plot
        save_dir: Directory to save the plot
    """
    # Get the weights from the first convolutional layer
    weights = model.conv1.weight.data.cpu().numpy()

    # Determine the number of filters and their size
    n_filters, n_channels, filter_size = weights.shape

    # Create a figure - add extra height for the explanation text
    fig = plt.figure(figsize=(14, 14))

    # Create grid for filter plots (4x4 grid) in the top part of the figure
    grid_size = (4, 4)
    gs = fig.add_gridspec(nrows=5, ncols=4, height_ratios=[1, 1, 1, 1, 0.3])

    # Plot each filter
    for i in range(min(16, n_filters)):
        ax = fig.add_subplot(gs[i // 4, i % 4])

        # Plot each channel of the filter with different colors
        for c in range(n_channels):
            color = ['red', 'green', 'blue'][c]
            channel_label = ['X-axis', 'Y-axis', 'Z-axis'][c]
            ax.plot(weights[i, c], color=color, alpha=0.7, label=channel_label if i == 0 else "")

        # Compute some statistics about this filter
        filter_mean = weights[i].mean()
        filter_std = weights[i].std()

        ax.set_title(f'Filter {i + 1}\nμ={filter_mean:.2f}, σ={filter_std:.2f}')
        ax.grid(True, linestyle='--', alpha=0.6)

        # Only show legend on the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Create a dedicated text area at the bottom for the explanation
    explanation_ax = fig.add_subplot(gs[4, :])
    explanation_ax.axis('off')

    # Add explanation text in a clear, readable format
    explanation_text = """
    Filter Interpretation:

    • These filters show learned patterns that the model uses to detect features in vibration data
    • Regular patterns may indicate specific frequencies or temporal structures important for classification
    • Different patterns across X, Y, Z axes suggest the model detects directional vibration differences
    • Red = X-axis, Green = Y-axis, Blue = Z-axis vibration patterns
    """

    explanation_ax.text(0.5, 0.5, explanation_text,
                        ha='center', va='center',
                        fontsize=12,
                        bbox={"facecolor": "#f8f9fa",
                              "edgecolor": "#dee2e6",
                              "boxstyle": "round,pad=1.0",
                              "alpha": 1.0})

    # Set the overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    else:
        fig.suptitle('CNN1D First Layer Filter Visualization', fontsize=16, y=0.98)

    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        filename = f"cnn1d_filters.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved CNN filter visualization to {filepath}")

    plt.show()
    return fig

def visualize_tcn_activations(model, sample_input, layer_idx=0, title_prefix="Model",save_dir=None, filename=None):
    """
    Visualize activations of a TCN layer for a specific input sample

    Args:
        model: Trained TCN model
        sample_input: A single input sample (needs batch dimension)
        layer_idx: Index of the TCN layer to visualize
    """
    # Register hook to get intermediate activations
    activations = {}

    def hook_fn(module, input, output):
        activations['layer_' + str(layer_idx)] = output.detach().cpu()

    # Get the specific TCN layer
    target_layer = model.layers[layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)

    # Forward pass to get activations
    with torch.no_grad():
        _ = model(sample_input)

    # Remove the hook
    hook.remove()

    # Get activations and reshape
    act = activations['layer_' + str(layer_idx)]
    channels = act.shape[1]

    # Plot activations
    plt.figure(figsize=(14, 8))
    for i in range(min(16, channels)):  # Show up to 16 channels
        plt.subplot(4, 4, i + 1)
        plt.plot(act[0, i].numpy())
        plt.title(f'Channel {i + 1}')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f'TCN Layer {layer_idx} Activations', fontsize=16)
    plt.subplots_adjust(top=0.92)
    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title_prefix.replace(' ', '_').lower()}_tcn_activation.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {filepath}")

    plt.show()

def visualize_feature_distributions(X_train, y_train, title_prefix="Model", feature_names=None, top_n=10, save_dir=None, filename=None):
    """
    Visualize distributions of top features by importance

    Args:
        X_train: Feature matrix
        y_train: Labels
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    # Calculate feature importance using simple correlation
    importances = []
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        importances.append((abs(corr), i, corr))

    # Sort by absolute correlation
    importances.sort(reverse=True)

    # Plot distributions for top N features
    fig, axes = plt.subplots(min(top_n, len(importances)), 1, figsize=(10, 2 * min(top_n, len(importances))))

    for i, (abs_corr, idx, corr) in enumerate(importances[:top_n]):
        if top_n == 1:
            ax = axes
        else:
            ax = axes[i]

        # Get feature data for each class
        good_data = X_train[y_train == 0, idx]
        bad_data = X_train[y_train == 1, idx]

        # Plot distributions
        ax.hist(good_data, bins=30, alpha=0.5, label='Good', color='green')
        ax.hist(bad_data, bins=30, alpha=0.5, label='Bad', color='red')

        # Add feature name and correlation
        ax.set_title(f"{feature_names[idx]}: Correlation = {corr:.3f}")
        ax.legend()
        ax.grid(alpha=0.3)


    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = f"{title_prefix.replace(' ', '_').lower()}_feature_distributions.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved feature distributions to {filepath}")

# Create formatted metrics table
def create_metrics_table(results, save_dir=None, filename=None):
    """Create a formatted table of metrics and optionally save to CSV"""
    # Create DataFrame for metrics
    metrics_data = []

    for model_name, metrics in results.items():
        metrics_data.append({
            "Method": model_name,
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TP": metrics["TP"],
            "F1": metrics["f1"],
            "Accuracy": metrics["accuracy"],
            "TPR (Recall)": metrics["recall"],
            "TNR (Specificity)": metrics["specificity"],
            "Execution Time (s)": metrics.get("execution_time", "-"),
            "Memory Usage (MB)": metrics.get("memory_usage", "-")
        })

    # Create DataFrame and sort by accuracy
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values(by="Accuracy", ascending=False)

    # Format numeric columns
    for col in ["F1", "Accuracy", "TPR (Recall)", "TNR (Specificity)"]:
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    if "Execution Time (s)" in metrics_df.columns:
        metrics_df["Execution Time (s)"] = metrics_df["Execution Time (s)"].map(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    if "Memory Usage (MB)" in metrics_df.columns:
        metrics_df["Memory Usage (MB)"] = metrics_df["Memory Usage (MB)"].map(
            lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # Save to CSV if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = "model_metrics.csv"
        filepath = os.path.join(save_dir, filename)
        metrics_df.to_csv(filepath, index=False)
        print(f"Saved metrics to {filepath}")

    # Create a styled DataFrame for display
    styled_df = metrics_df.style.background_gradient(cmap='YlGnBu',
                                                     subset=['F1', 'Accuracy', 'TPR (Recall)', 'TNR (Specificity)'])

    # Display the table
    display(styled_df)

    return styled_df, metrics_df


# Create parameters table
'''def create_parameters_table(results):
    """Create a formatted table of best hyperparameters"""
    # Extract model types and parameters
    params_data = []

    for model_name, metrics in results.items():
        if "model" in metrics:
            model_obj = metrics["model"]
            params = {}

            if model_name == "SVM":
                params = {
                    "kernel": model_obj.named_steps['svm'].kernel,
                    "C": model_obj.named_steps['svm'].C,
                    "gamma": model_obj.named_steps['svm'].gamma
                }
            elif model_name == "Random_Forest":
                params = {
                    "n_estimators": model_obj.named_steps['rf'].n_estimators,
                    "max_depth": model_obj.named_steps['rf'].max_depth or "None",
                    "min_samples_split": model_obj.named_steps['rf'].min_samples_split
                }
            elif model_name == "Gradient_Boosting":
                params = {
                    "n_estimators": model_obj.named_steps['gb'].n_estimators,
                    "learning_rate": model_obj.named_steps['gb'].learning_rate,
                    "max_depth": model_obj.named_steps['gb'].max_depth
                }
            elif model_name == "MLP_Sklearn":
                params = {
                    "hidden_layer_sizes": str(model_obj.named_steps['mlp'].hidden_layer_sizes),
                    "learning_rate_init": model_obj.named_steps['mlp'].learning_rate_init,
                    "alpha": model_obj.named_steps['mlp'].alpha,
                    "activation": model_obj.named_steps['mlp'].activation
                }
            elif model_name in ["CNN1D_Time", "CNN1D_Frequency", "TCN"]:
                # Just examples for deep learning models - adjust as needed
                params = metrics.get("hyperparams", {})

            # Add model name and parameters
            for param_name, param_value in params.items():
                params_data.append({
                    "Model": model_name,
                    "Parameter": param_name,
                    "Value": str(param_value),
                    "Validation Accuracy": f"{metrics.get('val_accuracy', 0) * 100:.2f}%",
                    "Test Accuracy": f"{metrics.get('accuracy', 0) * 100:.2f}%"
                })

    # Create DataFrame
    params_df = pd.DataFrame(params_data)

    # Create styled DataFrame
    if not params_data:
        return None, None

    styled_df = params_df.style.background_gradient(cmap='YlGnBu', subset=['Validation Accuracy', 'Test Accuracy'])

    # Display the table
    display(styled_df)

    return styled_df, params_df


# Create performance comparison table
def create_performance_table(results):
    """Create a table showing overall performance metrics"""
    # Create DataFrame for performance comparison
    performance_data = []

    for model_name, metrics in results.items():
        if all(k in metrics for k in ["train_accuracy", "val_accuracy"]):
            row = {
                "Model": model_name,
                "Training Accuracy": f"{metrics['train_accuracy'] * 100:.2f}%",
                "Validation Accuracy": f"{metrics['val_accuracy'] * 100:.2f}%",
                "Test Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
                "F1 Score": f"{metrics['f1']:.4f}",
                "Execution Time (s)": f"{metrics.get('execution_time', 0):.2f}"
            }

            # Add training/validation loss if available
            if "train_loss" in metrics and "val_loss" in metrics:
                row["Training Loss"] = f"{metrics['train_loss']:.4f}"
                row["Validation Loss"] = f"{metrics['val_loss']:.4f}"

            # Add standard deviations if available
            if "std_train_acc" in metrics:
                row["Std Dev Train Acc"] = f"{metrics['std_train_acc'] * 100:.2f}%"
            if "std_val_acc" in metrics:
                row["Std Dev Val Acc"] = f"{metrics['std_val_acc'] * 100:.2f}%"

            # Add memory usage if available
            if "memory_usage" in metrics:
                row["Memory Usage (MB)"] = f"{metrics['memory_usage']:.2f}"

            performance_data.append(row)

    # Create DataFrame
    if not performance_data:
        print("Not enough performance data available to create the table.")
        return None, None

    performance_df = pd.DataFrame(performance_data)

    # Determine columns for gradient highlighting
    highlight_cols = [col for col in performance_df.columns
                      if any(c in col for c in ["Accuracy", "F1"])]

    # Create styled DataFrame
    styled_df = performance_df.style.background_gradient(cmap='YlGnBu', subset=highlight_cols)

    # Display the table
    display(styled_df)

    return styled_df, performance_df'''


# Updated create_parameters_table function in visualization_utils.py

def create_parameters_table(results, save_dir=None, filename=None):
    """
    Create a formatted table of best hyperparameters with enhanced NN model details

    Args:
        results: Dictionary containing results from all models

    Returns:
        styled_df: Styled DataFrame for display
        params_df: Raw DataFrame with parameters
    """
    # Extract model types and parameters
    params_data = []

    for model_name, metrics in results.items():
        # Handle traditional ML models
        if "model" in metrics:
            model_obj = metrics["model"]
            params = {}

            if model_name == "SVM":
                params = {
                    "kernel": model_obj.named_steps['svm'].kernel,
                    "C": model_obj.named_steps['svm'].C,
                    "gamma": model_obj.named_steps['svm'].gamma
                }
            elif model_name == "Random_Forest":
                params = {
                    "n_estimators": model_obj.named_steps['rf'].n_estimators,
                    "max_depth": model_obj.named_steps['rf'].max_depth or "None",
                    "min_samples_split": model_obj.named_steps['rf'].min_samples_split
                }
            elif model_name == "Gradient_Boosting":
                params = {
                    "n_estimators": model_obj.named_steps['gb'].n_estimators,
                    "learning_rate": model_obj.named_steps['gb'].learning_rate,
                    "max_depth": model_obj.named_steps['gb'].max_depth
                }
            elif model_name == "MLP_Sklearn":
                params = {
                    "hidden_layer_sizes": str(model_obj.named_steps['mlp'].hidden_layer_sizes),
                    "learning_rate_init": model_obj.named_steps['mlp'].learning_rate_init,
                    "alpha": model_obj.named_steps['mlp'].alpha,
                    "activation": model_obj.named_steps['mlp'].activation
                }

        # Handle neural network models with hyperparams dictionary
        elif "hyperparams" in metrics:
            params = metrics["hyperparams"]
        else:
            # Skip models without hyperparameters
            continue

        # Add model name and parameters
        for param_name, param_value in params.items():
            params_data.append({
                "Model": model_name,
                "Parameter": param_name,
                "Value": str(param_value),
                "Validation Accuracy": f"{metrics.get('val_accuracy', 0) * 100:.2f}%",
                "Test Accuracy": f"{metrics.get('accuracy', 0) * 100:.2f}%"
            })

    # Create DataFrame
    if not params_data:
        print("Not enough parameter data available to create the table.")
        return None, None

    params_df = pd.DataFrame(params_data)

    # Sort first by model name, then by parameter name
    params_df = params_df.sort_values(['Model', 'Parameter'])

    # Create styled DataFrame
    styled_df = params_df.style.background_gradient(
        cmap='YlGnBu', subset=['Validation Accuracy', 'Test Accuracy']
    ).set_properties(**{
        'border-color': 'black',
        'border-width': '1px',
        'border-style': 'solid'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'),
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'left')]},
        {'selector': 'caption', 'props': [('caption-side', 'top'),
                                          ('font-size', '16px'),
                                          ('font-weight', 'bold')]}
    ]).set_caption("Model Hyperparameters")

    # Display the table
    display(styled_df)

    # Save to CSV if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = "model_hyperparameters.csv"
        filepath = os.path.join(save_dir, filename)
        params_df.to_csv(filepath, index=False)
        print(f"Saved hyperparameters to {filepath}")

    return styled_df, params_df
# Updated create_performance_table function in visualization_utils.py

def create_performance_table(results, save_dir=None, filename=None):
    """
    Create a table showing overall performance metrics with enhanced NN model details

    Args:
        results: Dictionary containing results from all models

    Returns:
        styled_df: Styled DataFrame for display
        metrics_df: Raw DataFrame with metrics
    """
    # Create DataFrame for performance comparison
    performance_data = []

    for model_name, metrics in results.items():
        # Base metrics that all models should have
        row = {
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "F1 Score": f"{metrics['f1']:.4f}",
            "Precision": f"{metrics['precision']:.4f}" if 'precision' in metrics else "-",
            "Recall": f"{metrics['recall']:.4f}" if 'recall' in metrics else "-",
            "Specificity": f"{metrics['specificity']:.4f}" if 'specificity' in metrics else "-",
            "Execution Time (s)": f"{metrics.get('execution_time', 0):.2f}"
        }

        # Add training/validation metrics if available
        if all(k in metrics for k in ["train_accuracy", "val_accuracy"]):
            row["Train Accuracy"] = f"{metrics['train_accuracy'] * 100:.2f}%"
            row["Val Accuracy"] = f"{metrics['val_accuracy'] * 100:.2f}%"

        # Add loss metrics if available
        if "train_loss" in metrics and "val_loss" in metrics:
            row["Train Loss"] = f"{metrics['train_loss']:.4f}"
            row["Val Loss"] = f"{metrics['val_loss']:.4f}"

        # Add standard deviations if available
        if "std_train_acc" in metrics:
            row["Std Dev Train"] = f"{metrics['std_train_acc'] * 100:.2f}%"
        if "std_val_acc" in metrics:
            row["Std Dev Val"] = f"{metrics['std_val_acc'] * 100:.2f}%"

        # Add memory usage if available
        if "memory_usage" in metrics:
            row["Memory (MB)"] = f"{metrics['memory_usage']:.1f}"

        # Add NN-specific hyperparameters if available
        if "hyperparams" in metrics:
            hp = metrics["hyperparams"]

            # Add epochs
            if "epochs" in hp:
                row["Epochs"] = hp["epochs"]

            # Add learning rate info
            if "learning_rate" in hp:
                row["Init LR"] = f"{hp['learning_rate']:.6f}"
            if "final_learning_rate" in hp:
                row["Final LR"] = f"{hp['final_learning_rate']:.6f}"

            # Add scheduler info
            if "scheduler_type" in hp:
                row["Scheduler"] = hp["scheduler_type"]

            # Add early stopping info
            if "early_stopping" in hp:
                row["Early Stop"] = "Yes" if hp["early_stopping"] else "No"
                if hp["early_stopping"] and "patience" in hp:
                    row["Patience"] = hp["patience"]

            # Add model-specific architecture info
            if "CNN" in model_name or "TCN" in model_name:
                # For CNN & TCN models
                if "channels" in hp:
                    row["Architecture"] = f"{hp['channels']}"
                if "kernel_size" in hp:
                    row["Kernel"] = hp["kernel_size"]
                if "dropout" in hp:
                    row["Dropout"] = f"{hp['dropout']:.2f}"

            elif "MLP" in model_name:
                # For MLP models
                if "hidden_sizes" in hp:
                    row["Architecture"] = f"{hp['hidden_sizes']}"

        performance_data.append(row)

    # Create DataFrame
    if not performance_data:
        print("Not enough performance data available to create the table.")
        return None, None

    performance_df = pd.DataFrame(performance_data)

    # Define column order for better organization
    base_cols = ["Model", "Accuracy", "F1 Score", "Precision", "Recall", "Specificity"]
    training_cols = ["Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss",
                     "Std Dev Train", "Std Dev Val"]
    resource_cols = ["Execution Time (s)", "Memory (MB)"]

    # NN-specific columns
    nn_cols = ["Epochs", "Init LR", "Final LR", "Scheduler", "Early Stop", "Patience",
               "Architecture", "Kernel", "Dropout"]

    # Order columns that exist in the DataFrame
    all_cols = base_cols + training_cols + resource_cols + nn_cols
    existing_cols = [col for col in all_cols if col in performance_df.columns]

    # Add any columns that might be in the DataFrame but not in our predefined lists
    remaining_cols = [col for col in performance_df.columns if col not in existing_cols]
    ordered_cols = existing_cols + remaining_cols

    # Reorder the columns
    performance_df = performance_df[ordered_cols]

    # Sort by accuracy (descending)
    performance_df['Sort_Accuracy'] = performance_df['Accuracy'].str.rstrip('%').astype(float)
    performance_df = performance_df.sort_values('Sort_Accuracy', ascending=False)
    performance_df = performance_df.drop(columns=['Sort_Accuracy'])

    # Determine columns for gradient highlighting
    highlight_cols = [col for col in performance_df.columns
                      if any(c in col for c in ["Accuracy", "F1", "Precision", "Recall", "Specificity"])]

    # Create styled DataFrame with a better color scheme
    styled_df = performance_df.style.background_gradient(
        cmap='YlGnBu', subset=highlight_cols
    ).set_properties(**{
        'border-color': 'black',
        'border-width': '1px',
        'border-style': 'solid'
    }).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'),
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'caption', 'props': [('caption-side', 'top'),
                                          ('font-size', '16px'),
                                          ('font-weight', 'bold')]}
    ]).set_caption("Model Performance Comparison")

    # Display the table
    display(styled_df)

    # Save to CSV if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = "model_performance_comparison.csv"
        filepath = os.path.join(save_dir, filename)
        performance_df.to_csv(filepath, index=False)
        print(f"Saved performance metrics to {filepath}")

    return styled_df, performance_df


def compare_nn_training_curves(results, save_dir=None):
    """
    Compare the training and validation curves of neural network models and optionally save to file

    Args:
        results: Dictionary containing results from neural network models
        save_dir: Directory to save the plots
    """
    # Filter only neural network models with training history
    nn_models = {}
    for model_name, metrics in results.items():
        if ("CNN" in model_name or "TCN" in model_name or "MLP_PyTorch" in model_name):
            # Check if we have full training history
            if all(k in metrics for k in ["train_accuracies", "val_accuracies", "train_losses", "val_losses"]):
                nn_models[model_name] = metrics
            # Check if we at least have final metrics
            elif all(k in metrics for k in ["train_accuracy", "val_accuracy", "train_loss", "val_loss"]):
                nn_models[model_name] = metrics

    if not nn_models:
        print("No neural network models with training history found.")
        return

    # Create figure with 2 rows - top for accuracy, bottom for loss
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Color map for different models
    colors = plt.cm.tab10.colors

    # Plot accuracy curves
    for i, (model_name, metrics) in enumerate(nn_models.items()):
        color = colors[i % len(colors)]
        # Check if we have train/val histories as lists
        if all(k in metrics for k in ["train_accuracies", "val_accuracies"]) and \
                isinstance(metrics["train_accuracies"], list) and \
                len(metrics["train_accuracies"]) > 0:
            # Full history available
            train_accs = metrics["train_accuracies"]
            val_accs = metrics["val_accuracies"]
            epochs = range(1, len(train_accs) + 1)
            axes[0].plot(epochs, train_accs, color=color, linestyle='-', linewidth=2,
                         label=f"{model_name} Train")
            axes[0].plot(epochs, val_accs, color=color, linestyle='--', linewidth=2,
                         label=f"{model_name} Val")
        else:
            # Only final values available - plot as points
            axes[0].scatter(1, metrics["train_accuracy"], color=color, marker='o', s=100,
                            label=f"{model_name} Train")
            axes[0].scatter(1, metrics["val_accuracy"], color=color, marker='x', s=100,
                            label=f"{model_name} Val")

    axes[0].set_title("Training and Validation Accuracy", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Accuracy", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=12)

    # Plot loss curves
    for i, (model_name, metrics) in enumerate(nn_models.items()):
        color = colors[i % len(colors)]
        if all(k in metrics for k in ["train_losses", "val_losses"]) and \
                isinstance(metrics["train_losses"], list) and \
                len(metrics["train_losses"]) > 0:
            # Full history available
            train_losses = metrics["train_losses"]
            val_losses = metrics["val_losses"]
            epochs = range(1, len(train_losses) + 1)
            axes[1].plot(epochs, train_losses, color=color, linestyle='-', linewidth=2,
                         label=f"{model_name} Train")
            axes[1].plot(epochs, val_losses, color=color, linestyle='--', linewidth=2,
                         label=f"{model_name} Val")
        else:
            # Only final values available - plot as points
            axes[1].scatter(1, metrics["train_loss"], color=color, marker='o', s=100,
                            label=f"{model_name} Train")
            axes[1].scatter(1, metrics["val_loss"], color=color, marker='x', s=100,
                            label=f"{model_name} Val")

    axes[1].set_title("Training and Validation Loss", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=12)

    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        filepath = os.path.join(save_dir, "nn_training_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved neural network training comparison to {filepath}")

    plt.show()

    # Also compare learning rate schedules if available
    models_with_lr = [(name, metrics) for name, metrics in nn_models.items()
                      if "learning_rates" in metrics and isinstance(metrics["learning_rates"], list)
                      and len(metrics["learning_rates"]) > 0]

    if models_with_lr:
        plt.figure(figsize=(10, 6))

        for i, (model_name, metrics) in enumerate(models_with_lr):
            color = colors[i % len(colors)]
            lrs = metrics["learning_rates"]
            epochs = range(1, len(lrs) + 1)
            plt.plot(epochs, lrs, color=color, linewidth=2, label=model_name)

        plt.title("Learning Rate Schedules", fontsize=16)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Learning Rate", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=12)
        plt.yscale('log')  # Log scale often better visualizes LR changes
        plt.tight_layout()

        # Save figure if directory is specified
        if save_dir:
            ensure_dir(save_dir)
            filepath = os.path.join(save_dir, "learning_rate_comparison.png")
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved learning rate comparison to {filepath}")

        plt.show()
def plot_model_comparison(results, save_dir=None):
    """
    Plot bar chart comparing model accuracy, F1, execution time, and memory usage
    and optionally save to file
    """
    # Extract metrics for comparison
    models = []
    accuracies = []
    f1_scores = []
    exec_times = []
    memory_usages = []

    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        models.append(model_name)
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1'])
        if 'execution_time' in metrics:
            exec_times.append(metrics['execution_time'])
        if 'memory_usage' in metrics:
            memory_usages.append(metrics['memory_usage'])

    # Create figure with subplots
    n_plots = 1 + (len(exec_times) > 0) + (len(memory_usages) > 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))

    if n_plots == 1:
        axes = [axes]

    # Plot accuracy and F1 scores
    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width / 2, accuracies, width, label='Accuracy', color='#3498db')
    axes[0].bar(x + width / 2, f1_scores, width, label='F1 Score', color='#e74c3c')

    axes[0].set_ylabel('Score', fontsize=14)
    axes[0].set_title('Model Performance Comparison', fontsize=16)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name.replace('_', '\n') for name in models], fontsize=10)
    axes[0].legend(fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for i, v in enumerate(accuracies):
        axes[0].text(i - width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    for i, v in enumerate(f1_scores):
        axes[0].text(i + width / 2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    axes[0].set_ylim(0, max(max(accuracies), max(f1_scores)) * 1.15)

    # Plot execution time if available
    if len(exec_times) > 0:
        axes[1].bar(models, exec_times, color='#2ecc71')
        axes[1].set_title('Execution Time', fontsize=16)
        axes[1].set_ylabel('Time (seconds)', fontsize=14)
        axes[1].set_xticklabels([name.replace('_', '\n') for name in models], fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(exec_times):
            axes[1].text(i, v + 0.5, f'{v:.2f}s', ha='center', fontsize=9)

    # Plot memory usage if available
    if len(memory_usages) > 0 and n_plots >= 3:
        axes[2].bar(models, memory_usages, color='#9b59b6')
        axes[2].set_title('Memory Usage', fontsize=16)
        axes[2].set_ylabel('Memory (MB)', fontsize=14)
        axes[2].set_xticklabels([name.replace('_', '\n') for name in models], fontsize=10)
        axes[2].grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(memory_usages):
            axes[2].text(i, v + 0.5, f'{v:.1f}MB', ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        filepath = os.path.join(save_dir, "model_performance_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved model performance comparison to {filepath}")

    plt.show()


# Plot operation distribution
def plot_operation_split_bar(train_ops, val_ops, test_ops, title="Operation Distribution in Data Splits", save_dir=None, filename=None):
    """
    Plot bar chart showing the distribution of operations in train/val/test sets
    """
    all_ops = sorted(set(train_ops) | set(val_ops) | set(test_ops))
    n_groups = len(all_ops)
    train_counts = [train_ops.get(op, 0) for op in all_ops]
    val_counts = [val_ops.get(op, 0) for op in all_ops]
    test_counts = [test_ops.get(op, 0) for op in all_ops]

    ind = np.arange(n_groups)
    width = 0.7

    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = ax.bar(ind, train_counts, width, label='train', color="#FFCB57")
    p2 = ax.bar(ind, val_counts, width, bottom=train_counts, label='validate', color="#A348A6")
    bottom_stacked = np.array(train_counts) + np.array(val_counts)
    p3 = ax.bar(ind, test_counts, width, bottom=bottom_stacked, label='test', color="#36964A")

    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(all_ops, rotation=45)
    ax.legend()
    plt.tight_layout()

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            # Create a default filename based on the title
            filename = "operation_distribution.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved operation distribution plot to {filepath}")

    plt.show()


def plot_label_distribution(train_labels, val_labels, test_labels, title="Label Distribution in Data Splits",
                            save_dir=None, filename=None):
    """
    Plot bar chart showing the distribution of labels in train/val/test sets

    Args:
        train_labels: Labels from training set
        val_labels: Labels from validation set
        test_labels: Labels from test set
        title: Plot title
        save_dir: Directory to save the plot (optional)
        filename: Filename for saved plot (optional)
    """
    # Convert to numpy arrays if they're not already
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    # Count labels in each split
    labels = ["Good", "Bad"]
    train_counts = [np.sum(train_labels == 0), np.sum(train_labels == 1)]
    val_counts = [np.sum(val_labels == 0), np.sum(val_labels == 1)]
    test_counts = [np.sum(test_labels == 0), np.sum(test_labels == 1)]

    # Calculate percentages for text display
    train_total = len(train_labels)
    val_total = len(val_labels)
    test_total = len(test_labels)
    total_samples = train_total + val_total + test_total

    train_percentages = [count / train_total * 100 for count in train_counts]
    val_percentages = [count / val_total * 100 for count in val_counts]
    test_percentages = [count / test_total * 100 for count in test_counts]

    # Calculate overall class imbalance
    total_good = train_counts[0] + val_counts[0] + test_counts[0]
    total_bad = train_counts[1] + val_counts[1] + test_counts[1]
    good_percentage = total_good / total_samples * 100
    bad_percentage = total_bad / total_samples * 100

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot counts - left subplot
    x = np.arange(len(labels))
    width = 0.25

    train_bars = ax1.bar(x - width, train_counts, width, label='Train', color="#3498db")
    val_bars = ax1.bar(x, val_counts, width, label='Validation', color="#e74c3c")
    test_bars = ax1.bar(x + width, test_counts, width, label='Test', color="#2ecc71")

    ax1.set_title('Label Distribution by Count', fontsize=14)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(fontsize=12)

    # Add count labels above each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=10)

    add_labels(train_bars)
    add_labels(val_bars)
    add_labels(test_bars)

    # Plot percentages - right subplot
    # Stacked bar showing the percentage of total samples in each split
    split_labels = ["Train", "Validation", "Test"]
    good_counts = [train_counts[0], val_counts[0], test_counts[0]]
    bad_counts = [train_counts[1], val_counts[1], test_counts[1]]

    # Calculate percentages of the total for each segment
    good_percentages = [count / total_samples * 100 for count in good_counts]
    bad_percentages = [count / total_samples * 100 for count in bad_counts]

    # Create stacked bars
    ax2.bar(split_labels, good_percentages, color="#3498db", label="Good")
    ax2.bar(split_labels, bad_percentages, bottom=good_percentages, color="#e74c3c", label="Bad")

    # Add split percentage labels
    for i, split in enumerate(split_labels):
        total_pct = good_percentages[i] + bad_percentages[i]
        # Label for each split's total percentage
        ax2.text(i, total_pct + 1, f"{total_pct:.1f}%",
                 ha='center', fontsize=10)

        # If segment is large enough, add good/bad percentages
        if good_percentages[i] > 5:
            ax2.text(i, good_percentages[i] / 2, f"{good_percentages[i]:.1f}%",
                     ha='center', color='white', fontsize=10)
        if bad_percentages[i] > 5:
            ax2.text(i, good_percentages[i] + bad_percentages[i] / 2, f"{bad_percentages[i]:.1f}%",
                     ha='center', color='white', fontsize=10)

    ax2.set_title('Distribution Percentage by Split', fontsize=14)
    ax2.set_ylabel('Percentage of Total Samples', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(fontsize=12, loc='upper right')

    # Add overall class distribution info
    plt.figtext(0.5, 0.01,
                f"Overall class distribution: Good: {good_percentage:.1f}% ({total_good} samples), "
                f"Bad: {bad_percentage:.1f}% ({total_bad} samples), "
                f"Total: {total_samples} samples",
                ha="center", fontsize=12, bbox={"facecolor": "#f8f9fa", "alpha": 1.0, "pad": 5,
                                                "edgecolor": "#dee2e6"})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the text at the bottom

    fig.suptitle(title, fontsize=16)

    # Save figure if directory is specified
    if save_dir:
        ensure_dir(save_dir)
        if filename is None:
            filename = "label_distribution.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved label distribution plot to {filepath}")

    plt.show()

    # Return counts as dictionary for potential further use
    return {
        "train": {"good": train_counts[0], "bad": train_counts[1],
                  "ratio": train_counts[0] / train_counts[1] if train_counts[1] > 0 else float('inf')},
        "validation": {"good": val_counts[0], "bad": val_counts[1],
                       "ratio": val_counts[0] / val_counts[1] if val_counts[1] > 0 else float('inf')},
        "test": {"good": test_counts[0], "bad": test_counts[1],
                 "ratio": test_counts[0] / test_counts[1] if test_counts[1] > 0 else float('inf')},
        "total": {"good": total_good, "bad": total_bad,
                  "ratio": total_good / total_bad if total_bad > 0 else float('inf')}
    }
def save_df_to_csv(df, save_dir, filename):
    """Save DataFrame to CSV file"""
    ensure_dir(save_dir)
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved DataFrame to {filepath}")
    return filepath
# Save plots and tables as HTML report
def save_results_as_report(results, filename="model_comparison_report.html", plots_dir=None):
    """
    Save all results as an HTML report with links to saved plots

    Args:
        results: Dictionary containing results from models
        filename: Output HTML file name
        plots_dir: Directory containing saved plots
    """
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            img { max-width: 100%; margin: 10px 0; }
            .container { max-width: 1200px; margin: 0 auto; }
            .section { margin: 30px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Comparison Report</h1>
    """

    # Add metrics table
    _, metrics_df = create_metrics_table(results)
    html_content += "<div class='section'><h2>Performance Metrics</h2>"
    html_content += metrics_df.to_html(index=False, classes="dataframe")
    html_content += "</div>"

    # Add parameters table
    _, params_df = create_parameters_table(results)
    if params_df is not None:
        html_content += "<div class='section'><h2>Model Parameters</h2>"
        html_content += params_df.to_html(index=False, classes="dataframe")
        html_content += "</div>"

    # Add performance table
    _, perf_df = create_performance_table(results)
    if perf_df is not None:
        html_content += "<div class='section'><h2>Detailed Performance</h2>"
        html_content += perf_df.to_html(index=False, classes="dataframe")
        html_content += "</div>"

    # Add embedded images (if plots directory provided)
    if plots_dir:
        html_content += "<div class='section'><h2>Visualizations</h2>"

        # Add performance comparison plots
        model_perf_path = os.path.join(plots_dir, "model_performance_comparison.png")
        if os.path.exists(model_perf_path):
            html_content += f"""
                <div class='subsection'>
                    <h3>Model Performance Comparison</h3>
                    <img src='{os.path.relpath(model_perf_path, os.path.dirname(filename))}' alt='Model Performance Comparison'>
                </div>
            """

        # Add NN training comparison
        nn_training_path = os.path.join(plots_dir, "nn_training_comparison.png")
        if os.path.exists(nn_training_path):
            html_content += f"""
                <div class='subsection'>
                    <h3>Neural Network Training Comparison</h3>
                    <img src='{os.path.relpath(nn_training_path, os.path.dirname(filename))}' alt='Neural Network Training Comparison'>
                </div>
            """

        # Add learning rate comparison
        lr_path = os.path.join(plots_dir, "learning_rate_comparison.png")
        if os.path.exists(lr_path):
            html_content += f"""
                <div class='subsection'>
                    <h3>Learning Rate Schedules</h3>
                    <img src='{os.path.relpath(lr_path, os.path.dirname(filename))}' alt='Learning Rate Schedules'>
                </div>
            """

        # Add individual model plots if available
        html_content += "<h3>Individual Model Results</h3>"
        html_content += "<div style='display: flex; flex-wrap: wrap;'>"

        for plot_file in os.listdir(plots_dir):
            if plot_file.endswith(".png") and not plot_file in ["model_performance_comparison.png",
                                                                "nn_training_comparison.png",
                                                                "learning_rate_comparison.png"]:
                html_content += f"""
                    <div style='width: 48%; margin: 1%;'>
                        <h4>{os.path.splitext(plot_file)[0].replace('_', ' ').title()}</h4>
                        <img src='{os.path.relpath(os.path.join(plots_dir, plot_file), os.path.dirname(filename))}' 
                             alt='{plot_file}' style='width: 100%;'>
                    </div>
                """

        html_content += "</div>"
        html_content += "</div>"

    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """

    # Write to file
    with open(filename, 'w') as f:
        f.write(html_content)

    print(f"Report saved to {filename}")

