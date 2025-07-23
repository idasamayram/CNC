# cnn1d_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import gc
from visualization.visualization_utils import (
    track_time, plot_confmat_and_metrics, visualize_cnn_filters, get_memory_usage,
    plot_training_history, ensure_dir, save_figure
)


class CNN1D_DS_Wide(nn.Module):
    """1D CNN model for vibration data analysis"""

    def __init__(self):
        super(CNN1D_DS_Wide, self).__init__()
        # Wider kernels with GroupNorm for better receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.gn1 = nn.GroupNorm(4, 16)  # GroupNorm for better generalization
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.gn2 = nn.GroupNorm(4, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.gn3 = nn.GroupNorm(4, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.gn1(self.conv1(x))))
        x = self.pool2(self.relu(self.gn2(self.conv2(x))))
        x = self.pool3(self.relu(self.gn3(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(train_loader), correct / total


def validate_epoch(model, val_loader, criterion, device):
    """Validate after an epoch"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(val_loader), correct / total


def test_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


@track_time
def train_cnn1d_model(train_loader, val_loader, test_loader, epochs=30, lr=0.001, weight_decay=1e-4,
                      model_type="Time", early_stopping=False, patience=5, use_scheduler=False,
                      scheduler_type="plateau", scheduler_params=None, save_dir=None):
    """
    Train and evaluate a CNN1D model with early stopping and optional learning rate scheduling.
    By default, early_stopping=True and use_scheduler=False as requested.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN1D ({model_type} Domain) model on {device}...")

    # Track memory usage before training
    memory_before = get_memory_usage()

    # Initialize model
    model = CNN1D_DS_Wide().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize scheduler if requested (but default is False)
    scheduler = None
    if use_scheduler:
        if scheduler_params is None:
            scheduler_params = {}

        if scheduler_type == "step":
            step_size = scheduler_params.get("step_size", 10)
            gamma = scheduler_params.get("gamma", 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "plateau":
            factor = scheduler_params.get("factor", 0.1)
            patience_lr = scheduler_params.get("patience", 3)
            min_lr = scheduler_params.get("min_lr", 1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience_lr, min_lr=min_lr, verbose=True)
        elif scheduler_type == "cosine":
            T_max = scheduler_params.get("T_max", epochs)
            eta_min = scheduler_params.get("eta_min", 0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == "onecycle":
            steps_per_epoch = len(train_loader)
            max_lr = scheduler_params.get("max_lr", lr * 10)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # For early stopping
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    # For learning curve
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    # Training loop
    for epoch in range(epochs):
        # Training
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation
        val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Update scheduler if using
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            elif scheduler_type != "onecycle":  # OneCycleLR is updated in the train_epoch function
                scheduler.step()

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} - "
              f"LR: {current_lr:.6f} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Early stopping check
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    # Load best model if early stopping was used
    if early_stopping and best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Test evaluation
    all_preds, all_labels = test_model(model, test_loader, device)

    # Track memory usage after training
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before

    # Calculate metrics
    from sklearn.metrics import f1_score
    test_accuracy = (all_preds == all_labels).mean()
    test_f1 = f1_score(all_labels, all_preds)

    # Plot confusion matrix
    metrics_dict = plot_confmat_and_metrics(all_labels, all_preds, class_names=["Good", "Bad"],
                                            title=f"CNN1D {model_type} Confusion Matrix",
                                            save_dir=save_dir)

    # Plot learning curves using the common visualization function
    plot_training_history(
        train_losses, val_losses, train_accuracies, val_accuracies,
        title_prefix=f"CNN1D {model_type}", save_dir=save_dir
    )

    # Visualize CNN filters
    visualize_cnn_filters(model, title=f"CNN1D {model_type} Filters", save_dir=save_dir)

    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Return model and metrics - include all history data
    metrics = {
        "model": model,
        "model_type": f"CNN1D_{model_type}",
        "accuracy": test_accuracy,
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "specificity": metrics_dict["specificity"],
        "f1": test_f1,
        "TP": metrics_dict["TP"],
        "FP": metrics_dict["FP"],
        "TN": metrics_dict["TN"],
        "FN": metrics_dict["FN"],
        "train_accuracy": train_accuracies[-1],
        "val_accuracy": val_accuracy,
        "train_loss": train_losses[-1],
        "val_loss": val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
        "std_train_acc": np.std(train_accuracies[-5:]) if len(train_accuracies) >= 5 else np.std(train_accuracies),
        "std_val_acc": np.std(val_accuracies[-5:]) if len(val_accuracies) >= 5 else np.std(val_accuracies),
        "memory_usage": memory_used,
        "hyperparams": {
            "learning_rate": lr,
            "final_learning_rate": learning_rates[-1],
            "weight_decay": weight_decay,
            "epochs": epoch + 1,  # Actual epochs trained
            "scheduler_type": scheduler_type if use_scheduler else "none",
            "early_stopping": early_stopping,
            "patience": patience
        }
    }

    return model, metrics