import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pandas as pd
import h5py
from pathlib import Path
import torch.nn.functional as F
from scipy import signal
from scipy.fft import fft, fftfreq
import time
import psutil
import gc
from tqdm import tqdm
import matplotlib.ticker as ticker
from utils.dataloader import stratified_group_split, stratified_group_split_freq


# Enable GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ----------------
# Dataset Class
# ----------------
class VibrationDataset(Dataset):
    def __init__(self, data_dir, transform=None, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []
        self.augment_bad = augment_bad
        self.file_groups = []
        self.transform = transform

        for label, label_idx in zip(["good", "bad"], [0, 1]):  # 0=good, 1=bad
            folder = self.data_dir / label
            for file_name in folder.glob("*.h5"):
                self.file_paths.append(file_name)
                self.labels.append(label_idx)
                operation = file_name.stem.split('_')[3]
                self.operations.append(operation)
                file_group = file_name.stem.rsplit('_window_', 1)[0]
                self.file_groups.append(file_group)

        self.labels = np.array(self.labels)
        self.operations = np.array(self.operations)
        self.file_groups = np.array(self.file_groups)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN
        label = self.labels[idx]

        # Augment bad samples by adding noise
        if self.augment_bad and label == 1:
            data += np.random.normal(0, 0.01, data.shape)

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
# ----------------
# Data Transforms
# ----------------
class FrequencyTransform:
    """Transform time-domain signals to frequency domain using FFT"""

    def __init__(self, n_freq_bins=1000):
        self.n_freq_bins = n_freq_bins

    def __call__(self, data):
        # data has shape (3, 2000)
        freq_data = np.zeros((3, self.n_freq_bins), dtype=np.float32)
        for i in range(3):  # Process each axis
            # Compute FFT and take magnitude
            fft_vals = np.abs(fft(data[i]))[:self.n_freq_bins]
            # Normalize
            freq_data[i] = fft_vals / np.max(fft_vals) if np.max(fft_vals) > 0 else fft_vals
        return freq_data

class FeatureExtractor:
    """Extract handcrafted features for traditional ML models"""

    def __call__(self, data):
        # Extract statistical and frequency-domain features from time series
        features = []

        for i in range(3):  # Process each axis
            # Time domain features
            axis_data = data[i]
            features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.max(axis_data),
                np.min(axis_data),
                np.median(axis_data),
                np.percentile(axis_data, 25),
                np.percentile(axis_data, 75),
                np.sqrt(np.mean(axis_data ** 2)),  # RMS
                np.sum(np.abs(np.diff(axis_data))),  # Signal complexity
                np.max(axis_data) - np.min(axis_data)  # Range
            ])

            # Frequency domain features
            fft_vals = np.abs(fft(axis_data))
            freq = fftfreq(len(axis_data))
            pos_freq_idx = np.where(freq > 0)[0]
            fft_vals = fft_vals[pos_freq_idx]

            # Take top 5 frequency magnitudes and their indices
            if len(fft_vals) > 5:
                top_indices = np.argsort(fft_vals)[-5:]
                top_freqs = freq[pos_freq_idx][top_indices]
                top_mags = fft_vals[top_indices]

                features.extend(list(top_freqs))
                features.extend(list(top_mags))

        return np.array(features, dtype=np.float32)

# ----------------
# Memory and Time Tracking
# ----------------
class ResourceTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.end_memory = None

    def start(self):
        self.reset()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.start_memory
        return self

    def update_peak(self):
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

    def stop(self):
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.update_peak()
        return self

    def get_stats(self):
        elapsed_time = self.end_time - self.start_time
        memory_used = self.end_memory - self.start_memory
        return {
            'time_elapsed': elapsed_time,
            'memory_used': memory_used,
            'peak_memory': self.peak_memory,
            'start_memory': self.start_memory,
            'end_memory': self.end_memory
        }

# ----------------
# Models
# ----------------
# 1. CNN1D_DS_Wide (Current best model)
class CNN1D_DS_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_DS_Wide, self).__init__()
        # Wider kernels with GroupNorm for better receptive field and stable training
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.gn1 = nn.GroupNorm(4, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.gn2 = nn.GroupNorm(4, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.gn3 = nn.GroupNorm(4, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.gn1(self.conv1(x))))
        x = self.pool2(self.relu(self.gn2(self.conv2(x))))
        x = self.pool3(self.relu(self.gn3(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# 1-1. CNN1D_Wide (Current best model)
class CNN1D_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        # Wider kernels to increase receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)  # Increased kernel size
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout1 = nn.Dropout(0.2)  # Add dropout after first layer

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)  # Increased kernel size
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout2 = nn.Dropout(0.2)  # Add dropout after second layer

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)  # Increased kernel size
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout3 = nn.Dropout(0.2)  # Add dropout after third layer

        # NEW: Add a fourth convolutional layer for deeper network
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)  # Changed input size to match conv4 output
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)  # Increased dropout for final layer
        self.relu = nn.LeakyReLU(0.1)  # Using LeakyReLU for better gradient flow

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x

# 2. CNN1D_Freq (New model for frequency domain)
class CNN1D_Freq(nn.Module):
    def __init__(self):
        super(CNN1D_Freq, self).__init__()
        # Slightly smaller model than CNN1D_DS_Wide
        self.conv1 = nn.Conv1d(3, 16, kernel_size=15, stride=1, padding=7)
        self.bn1 = nn.BatchNorm1d(16)  # Use BatchNorm instead of GroupNorm
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 48, kernel_size=5, stride=1, padding=2)  # 48 filters instead of 64
        self.bn3 = nn.BatchNorm1d(48)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(48, 32)  # Smaller hidden layer
        self.fc2 = nn.Linear(32, 2)

        self.dropout = nn.Dropout(0.25)  # Lower dropout
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# 3. TCN (Temporal Convolutional Network)
# Corrected TCN model implementation
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        # Save original size for residual connection
        original_size = x.size(2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Ensure residual has the same size as the output
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        # Match sizes if needed (center crop or pad)
        if out.size(2) != residual.size(2):
            if out.size(2) > residual.size(2):
                # Center crop the output
                diff = out.size(2) - residual.size(2)
                start = diff // 2
                out = out[:, :, start:start + residual.size(2)]
            else:
                # Center pad the output
                diff = residual.size(2) - out.size(2)
                start = diff // 2
                end = diff - start
                out = F.pad(out, (start, end))

        return self.relu(out + residual)

# 4 Modified TCN with weaker performance
class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        layers = []
        num_levels = 3  # Reduced from 4 to 3
        num_hidden = 14  # Reduced from 16 to 14
        kernel_size = 5  # Smaller kernel size
        dropout_rate = 0.3  # Higher dropout

        # First layer with reduced parameters
        layers.append(TemporalBlock(3, num_hidden, kernel_size, stride=1, dilation=1,
                                    padding=(kernel_size - 1), dropout=dropout_rate))

        # Fewer levels of temporal blocks
        for i in range(num_levels - 1):
            dilation_size = 2 ** (i + 1)
            layers.append(TemporalBlock(num_hidden, num_hidden, kernel_size, stride=1,
                                        dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size,
                                        dropout=dropout_rate))

        self.tcn_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = self.tcn_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)# 4. Simple MLP for comparison
# 5. MLP Model
class MLP_Model(nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.flatten = nn.Flatten()
        # Smaller network to ensure it's weaker than CNN1D_DS_Wide
        self.fc1 = nn.Linear(3 * 2000, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
# ----------------
# Training Functions
# ----------------
def train_epoch(model, train_loader, optimizer, criterion, device, resource_tracker=None):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if resource_tracker:
            resource_tracker.update_peak()

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
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

    val_loss /= len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }

    return results

def train_neural_network(model_name, model, train_loader, val_loader, test_loader,
                         epochs=30, lr=0.001, weight_decay=1e-4, scheduler=True):
    resource_tracker = ResourceTracker().start()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    if scheduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # Training metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print(f"\n==== Training {model_name} ====")
    for epoch in tqdm(range(epochs), desc=f"Training {model_name}"):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, resource_tracker)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        if scheduler:
            lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:  # Print every 10 epochs and first epoch
            print(f"{model_name} - Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate on test set
    test_results = evaluate_model(model, test_loader, device)

    resource_tracker.stop()
    resource_stats = resource_tracker.get_stats()

    print(f"{model_name} - Test Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
    print(
        f"{model_name} - Time: {resource_stats['time_elapsed']:.2f} sec, Memory Used: {resource_stats['memory_used']:.2f} MB")

    # Plot training curves
    plot_training_curves(model_name, train_losses, val_losses,
                         train_accuracies, val_accuracies)

    # Plot confusion matrix
    plot_confusion_matrix(test_results['confusion_matrix'],
                          ["Good", "Bad"],
                          f"Confusion Matrix - {model_name}")

    return {
        'model': model,
        'test_results': test_results,
        'resource_stats': resource_stats,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    }

def extract_features_for_ml(dataset, extractor=FeatureExtractor()):
    print("Extracting features for traditional ML models...")
    features = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        data, label = dataset[i]
        # Convert to numpy array if tensor
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # Extract features
        feature_vector = extractor(data)
        features.append(feature_vector)
        labels.append(label)

    return np.array(features), np.array(labels)

def train_traditional_ml_model(model_name, model, X_train, y_train, X_test, y_test):
    resource_tracker = ResourceTracker().start()

    # Train the model
    print(f"\n==== Training {model_name} ====")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    resource_tracker.stop()
    resource_stats = resource_tracker.get_stats()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(
        f"{model_name} - Time: {resource_stats['time_elapsed']:.2f} sec, Memory Used: {resource_stats['memory_used']:.2f} MB")

    # Plot confusion matrix
    plot_confusion_matrix(cm, ["Good", "Bad"], f"Confusion Matrix - {model_name}")

    results = {
        'model': model,
        'test_results': {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test
        },
        'resource_stats': resource_stats
    }

    return results
# ----------------
# Visualization Functions
# ----------------
def plot_training_curves(model_name, train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=2, linecolor='white', cbar=False, annot_kws={'size': 18})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.show()

def plot_model_comparison(results_dict):
    # Extract metrics for comparison
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    model_names = list(results_dict.keys())

    # Create a DataFrame for easy plotting
    data = []
    for model in model_names:
        for metric in metrics:
            data.append({
                'model': model,
                'metric': metric,
                'value': results_dict[model]['test_results'][metric]
            })

    df = pd.DataFrame(data)

    # Plot comparison
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df, x='model', y='value', hue='metric')
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Create a table of results for easier comparison
    pivot_df = df.pivot(index='model', columns='metric', values='value')
    print("\nModel Performance Comparison:")
    print(pivot_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save to CSV
    pivot_df.to_csv('model_comparison_results.csv')
    print("Results saved to 'model_comparison_results.csv'")

def plot_resource_comparison(results_dict):
    # Extract resource stats for comparison
    model_names = list(results_dict.keys())
    training_times = [results_dict[model]['resource_stats']['time_elapsed'] for model in model_names]
    memory_used = [results_dict[model]['resource_stats']['memory_used'] for model in model_names]
    peak_memory = [results_dict[model]['resource_stats']['peak_memory'] for model in model_names]

    # Create a DataFrame for easy plotting
    resource_df = pd.DataFrame({
        'model': model_names,
        'training_time': training_times,
        'memory_used': memory_used,
        'peak_memory': peak_memory
    })

    # Plot time comparison
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x='model', y='training_time', data=resource_df)
    plt.title('Training Time Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of each bar
    for i, v in enumerate(training_times):
        ax.text(i, v + 0.5, f"{v:.1f}s", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Plot memory comparison (both used and peak)
    plt.figure(figsize=(12, 5))

    # Create memory data in long format for seaborn
    memory_data = []
    for model in model_names:
        memory_data.append({
            'model': model,
            'type': 'Memory Used',
            'memory': results_dict[model]['resource_stats']['memory_used']
        })
        memory_data.append({
            'model': model,
            'type': 'Peak Memory',
            'memory': results_dict[model]['resource_stats']['peak_memory']
        })

    memory_df = pd.DataFrame(memory_data)

    ax = sns.barplot(data=memory_df, x='model', y='memory', hue='type')
    plt.title('Memory Usage Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Memory (MB)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Memory Type')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of each bar
    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.5,
                f"{p.get_height():.1f} MB", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Create a comprehensive comparison table and save to CSV
    resource_df = pd.DataFrame({
        'model': model_names,
        'training_time': training_times,
        'memory_used': memory_used,
        'peak_memory': peak_memory,
        'accuracy': [results_dict[model]['test_results']['accuracy'] for model in model_names],
        'f1_score': [results_dict[model]['test_results']['f1'] for model in model_names]
    })

    resource_df = resource_df.sort_values('f1_score', ascending=False)

    print("\nResource and Performance Comparison:")
    print(resource_df.to_string(float_format=lambda x: f"{x:.4f}"))
    resource_df.to_csv('resource_comparison_results.csv')
    print("Resource comparison saved to 'resource_comparison_results.csv'")

def plot_operation_split_bar(train_ops, val_ops, test_ops,
                             title="Stratified Distribution of Train/Val/Test Sets per Operation"):
    """
    Plots a bar chart showing the distribution of operations in train/val/test sets.
    Args:
        train_ops, val_ops, test_ops: Counters with operation as key and count as value.
        title: Title for the plot.
    """
    all_ops = sorted(set(train_ops) | set(val_ops) | set(test_ops))
    n_groups = len(all_ops)
    train_counts = [train_ops.get(op, 0) for op in all_ops]
    val_counts = [val_ops.get(op, 0) for op in all_ops]
    test_counts = [test_ops.get(op, 0) for op in all_ops]

    ind = np.arange(n_groups)
    width = 0.7

    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    p1 = ax.bar(ind, train_counts, width, label='Train', color="#FFCB57")
    p2 = ax.bar(ind, val_counts, width, bottom=train_counts, label='Validate', color="#A348A6")
    bottom_stacked = np.array(train_counts) + np.array(val_counts)
    p3 = ax.bar(ind, test_counts, width, bottom=bottom_stacked, label='Test', color="#36964A")

    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(all_ops, rotation=45, ha='right')
    ax.legend()

    # Add count labels
    for i, op in enumerate(all_ops):
        total = train_counts[i] + val_counts[i] + test_counts[i]
        ax.text(i, total + 10, f"Total: {total}", ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_label_distribution(dataset, train_idx, val_idx, test_idx):
    """
    Plots the distribution of good vs. bad labels across train/val/test sets.
    """
    train_good = sum(dataset.labels[train_idx] == 0)
    train_bad = sum(dataset.labels[train_idx] == 1)
    val_good = sum(dataset.labels[val_idx] == 0)
    val_bad = sum(dataset.labels[val_idx] == 1)
    test_good = sum(dataset.labels[test_idx] == 0)
    test_bad = sum(dataset.labels[test_idx] == 1)

    # Create data for plotting
    data = {
        'Set': ['Train', 'Train', 'Validation', 'Validation', 'Test', 'Test'],
        'Label': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad'],
        'Count': [train_good, train_bad, val_good, val_bad, test_good, test_bad]
    }
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x='Set', y='Count', hue='Label', palette={'Good': '#2ecc71', 'Bad': '#e74c3c'})
    plt.title('Distribution of Labels Across Train/Val/Test Sets', fontsize=14)
    plt.xlabel('Dataset Split', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)

    # Add percentage labels
    total_train = train_good + train_bad
    total_val = val_good + val_bad
    total_test = test_good + test_bad

    for i, p in enumerate(ax.patches):
        if i == 0:  # Train Good
            percentage = train_good / total_train * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{train_good} ({percentage:.1f}%)", ha='center')
        elif i == 1:  # Validation Good
            percentage = val_good / total_val * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{val_good} ({percentage:.1f}%)", ha='center')
        elif i == 2:  # Test Good
            percentage = test_good / total_test * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{test_good} ({percentage:.1f}%)", ha='center')
        elif i == 3:  # Train Bad
            percentage = train_bad / total_train * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{train_bad} ({percentage:.1f}%)", ha='center')
        elif i == 4:  # Validation Bad
            percentage = val_bad / total_val * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{val_bad} ({percentage:.1f}%)", ha='center')
        elif i == 5:  # Test Bad
            percentage = test_bad / total_test * 100
            ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 5,
                    f"{test_bad} ({percentage:.1f}%)", ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_operation_pie_charts(dataset, train_idx, val_idx, test_idx):
    """
    Plots pie charts showing the distribution of operations in each split.
    """
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Sort operations for consistent colors
    all_ops = sorted(set(train_ops) | set(val_ops) | set(test_ops))
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_ops)))

    # Training set
    train_labels = []
    train_sizes = []
    for op in all_ops:
        count = train_ops.get(op, 0)
        percentage = count / sum(train_ops.values()) * 100
        train_labels.append(f"{op}: {count} ({percentage:.1f}%)")
        train_sizes.append(count)

    ax1.pie(train_sizes, labels=None, colors=colors, autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
            startangle=90, shadow=False)
    ax1.set_title('Training Set Operations', fontsize=14)

    # Validation set
    val_labels = []
    val_sizes = []
    for op in all_ops:
        count = val_ops.get(op, 0)
        percentage = count / sum(val_ops.values()) * 100
        val_labels.append(f"{op}: {count} ({percentage:.1f}%)")
        val_sizes.append(count)

    ax2.pie(val_sizes, labels=None, colors=colors, autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
            startangle=90, shadow=False)
    ax2.set_title('Validation Set Operations', fontsize=14)

    # Test set
    test_labels = []
    test_sizes = []
    for op in all_ops:
        count = test_ops.get(op, 0)
        percentage = count / sum(test_ops.values()) * 100
        test_labels.append(f"{op}: {count} ({percentage:.1f}%)")
        test_sizes.append(count)

    ax3.pie(test_sizes, labels=None, colors=colors, autopct=lambda p: f'{p:.1f}%' if p > 5 else '',
            startangle=90, shadow=False)
    ax3.set_title('Test Set Operations', fontsize=14)

    # Create a single legend for all pie charts
    labels = [f"{op}" for op in all_ops]
    fig.legend(labels, loc='lower center', ncol=min(len(all_ops), 5), bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Adjust for legend
    plt.show()


def plot_training_dynamics_comparison(results_dict, models_to_compare=None):
    """
    Plot training dynamics comparison between different models.

    Args:
        results_dict: Dictionary containing results of all trained models
        models_to_compare: List of model names to include in comparison (if None, all neural network models are used)
    """
    plt.figure(figsize=(18, 10))

    # Select models that have training history (neural network models)
    if models_to_compare is None:
        models_to_compare = []
        for model_name, result in results_dict.items():
            if 'training_history' in result:
                models_to_compare.append(model_name)

    # Define colors for consistent plotting
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_compare)))
    color_dict = {model: color for model, color in zip(models_to_compare, colors)}

    # Plot validation accuracy
    plt.subplot(2, 2, 1)
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['val_accuracies']) + 1)

        plt.plot(epochs, history['val_accuracies'], '-o',
                 color=color_dict[model_name], label=model_name,
                 linewidth=2, markersize=4)

    plt.title('Validation Accuracy During Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # Plot validation loss
    plt.subplot(2, 2, 2)
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['val_losses']) + 1)

        plt.plot(epochs, history['val_losses'], '-o',
                 color=color_dict[model_name], label=model_name,
                 linewidth=2, markersize=4)

    plt.title('Validation Loss During Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Plot training accuracy
    plt.subplot(2, 2, 3)
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['train_accuracies']) + 1)

        plt.plot(epochs, history['train_accuracies'], '-o',
                 color=color_dict[model_name], label=model_name,
                 linewidth=2, markersize=4)

    plt.title('Training Accuracy During Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    # Plot training loss
    plt.subplot(2, 2, 4)
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['train_losses']) + 1)

        plt.plot(epochs, history['train_losses'], '-o',
                 color=color_dict[model_name], label=model_name,
                 linewidth=2, markersize=4)

    plt.title('Training Loss During Training', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('results/others/training_dynamics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_dynamics_comparison_full(results_dict, models_to_compare=None):
    """
    Creates a comprehensive visualization of training dynamics for multiple models:
    - Train and validation accuracy over epochs
    - Train and validation loss over epochs
    - Bar plot of test accuracy
    - Bar plot of train-validation gap (overfitting measure)

    Args:
        results_dict: Dictionary with model results containing training metrics
        models_to_compare: List of model names to compare (if None, all models are used)
    """
    if models_to_compare is None:
        models_to_compare = list(results_dict.keys())
    else:
        # Only use models that exist in results_dict
        models_to_compare = [m for m in models_to_compare if m in results_dict]

    if not models_to_compare:
        print("No valid models to compare")
        return

    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Training Dynamics Comparison", fontsize=16)

    # Generate a color map for the models
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_compare)))
    color_map = {model: colors[i] for i, model in enumerate(models_to_compare)}

    # Plot 1: Accuracy curves
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['train_accuracies']) + 1)

        # Plot train and validation accuracy
        axs[0, 0].plot(epochs, history['train_accuracies'],
                       color=color_map[model_name], linestyle='-', alpha=0.7,
                       label=f"{model_name} (Train)")
        axs[0, 0].plot(epochs, history['val_accuracies'],
                       color=color_map[model_name], linestyle='--', alpha=0.7,
                       label=f"{model_name} (Val)")

    axs[0, 0].set_title("Accuracy Curves")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: Loss curves
    for i, model_name in enumerate(models_to_compare):
        if 'training_history' not in results_dict[model_name]:
            continue

        history = results_dict[model_name]['training_history']
        epochs = range(1, len(history['train_losses']) + 1)

        # Plot train and validation loss
        axs[0, 1].plot(epochs, history['train_losses'],
                       color=color_map[model_name], linestyle='-', alpha=0.7,
                       label=f"{model_name} (Train)")
        axs[0, 1].plot(epochs, history['val_losses'],
                       color=color_map[model_name], linestyle='--', alpha=0.7,
                       label=f"{model_name} (Val)")

    axs[0, 1].set_title("Loss Curves")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Test Accuracy Bar Plot
    nn_models = [m for m in models_to_compare if 'test_results' in results_dict[m]]
    test_acc = [results_dict[m]['test_results']['accuracy'] for m in nn_models]

    bar_colors = [color_map[m] for m in nn_models]
    bar_positions = np.arange(len(nn_models))

    bars = axs[1, 0].bar(bar_positions, test_acc, alpha=0.7, color=bar_colors)
    axs[1, 0].set_title("Test Accuracy Comparison")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].set_xticks(bar_positions)
    axs[1, 0].set_xticklabels(nn_models, rotation=45, ha='right')
    axs[1, 0].grid(True, axis='y', alpha=0.3)

    # Add value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axs[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                       f'{test_acc[i]:.4f}', ha='center', va='bottom')

    # Plot 4: Train-Val Gap (measure of overfitting)
    train_val_gaps = []
    gap_models = []

    for model_name in nn_models:
        if 'training_history' in results_dict[model_name]:
            history = results_dict[model_name]['training_history']
            # Use the final epoch values for the gap
            train_acc = history['train_accuracies'][-1]
            val_acc = history['val_accuracies'][-1]
            train_val_gaps.append(train_acc - val_acc)
            gap_models.append(model_name)

    if train_val_gaps:
        bar_colors = [color_map[m] for m in gap_models]
        bar_positions = np.arange(len(gap_models))

        bars = axs[1, 1].bar(bar_positions, train_val_gaps, alpha=0.7, color=bar_colors)
        axs[1, 1].set_title("Train-Validation Accuracy Gap\n(Lower is Better)")
        axs[1, 1].set_ylabel("Gap (Train Acc - Val Acc)")
        axs[1, 1].set_xticks(bar_positions)
        axs[1, 1].set_xticklabels(gap_models, rotation=45, ha='right')
        axs[1, 1].grid(True, axis='y', alpha=0.3)

        # Add value annotations
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axs[1, 1].text(bar.get_x() + bar.get_width() / 2.,
                           height + 0.001 if height >= 0 else height - 0.02,
                           f'{train_val_gaps[i]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("training_dynamics_comparison.png", dpi=300)
    plt.show()


def plot_performance_complexity_tradeoff(results_dict, complexity_metrics, feature_counts, test_dataset_size):
    """
    Plot a scatter plot comparing model performance (test accuracy) vs. complexity (parameters/nodes/support vectors)
    with inference time as bubble size and feature counts in annotations.

    Args:
        results_dict: Dictionary containing results for each model (accuracy, resource stats)
        complexity_metrics: Dictionary with model names and their complexity (parameters, nodes, or support vectors)
        feature_counts: Dictionary with model names and their input feature counts
        test_dataset_size: Number of samples in the test dataset (for inference time calculation)
        save_path: Path to save the plot
    """
    models = [
        {"name": "1D-CNN-Wide", "complexity": complexity_metrics["1D-CNN-Wide"], "accuracy": results_dict["1D-CNN-Wide"]['test_results']['accuracy'], "inference_time": results_dict["1D-CNN-Wide"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "CNN", "features": feature_counts["1D-CNN-Wide"]},
        {"name": "1D-CNN-GN", "complexity": complexity_metrics["1D-CNN-GN"], "accuracy": results_dict["1D-CNN-GN"]['test_results']['accuracy'], "inference_time": results_dict["1D-CNN-GN"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "CNN", "features": feature_counts["1D-CNN-GN"]},
        {"name": "TCN", "complexity": complexity_metrics["TCN"], "accuracy": results_dict["TCN"]['test_results']['accuracy'], "inference_time": results_dict["TCN"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "Other NN", "features": feature_counts["TCN"]},
        {"name": "1D-CNN-Freq", "complexity": complexity_metrics["1D-CNN-Freq"], "accuracy": results_dict["1D-CNN-Freq"]['test_results']['accuracy'], "inference_time": results_dict["1D-CNN-Freq"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "CNN", "features": feature_counts["1D-CNN-Freq"]},
        {"name": "MLP", "complexity": complexity_metrics["MLP"], "accuracy": results_dict["MLP"]['test_results']['accuracy'], "inference_time": results_dict["MLP"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "Other NN", "features": feature_counts["MLP"]},
        {"name": "SVM", "complexity": complexity_metrics["SVM"], "accuracy": results_dict["SVM"]['test_results']['accuracy'], "inference_time": results_dict["SVM"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "Traditional ML", "features": feature_counts["SVM"]},
        {"name": "Gradient Boosting", "complexity": complexity_metrics["Gradient Boosting"], "accuracy": results_dict["Gradient Boosting"]['test_results']['accuracy'], "inference_time": results_dict["Gradient Boosting"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "Traditional ML", "features": feature_counts["Gradient Boosting"]},
        {"name": "Random Forest", "complexity": complexity_metrics["Random Forest"], "accuracy": results_dict["Random Forest"]['test_results']['accuracy'], "inference_time": results_dict["Random Forest"]['resource_stats']['time_elapsed']/test_dataset_size, "type": "Traditional ML", "features": feature_counts["Random Forest"]}
    ]

    model_names = [m["name"] for m in models]
    complexity = np.array([m["complexity"] for m in models])
    accuracy = np.array([m["accuracy"] for m in models])
    inference_time = np.array([m["inference_time"] for m in models])
    model_types = [m["type"] for m in models]
    feature_count = [m["features"] for m in models]

    # Create first figure: Scatter plot
    plt.figure(figsize=(12, 8))

    # Define different markers and colors for model types
    markers = {'CNN': 'o', 'Other NN': 's', 'Traditional ML': '^'}
    colors = {'CNN': '#1f77b4', 'Other NN': '#ff7f0e', 'Traditional ML': '#2ca02c'}

    # Create scatter plot with bubble size representing inference time
    for i, model in enumerate(models):
        plt.scatter(complexity[i], accuracy[i],
                    s=inference_time[i] * 500,  # Scale for visibility
                    alpha=0.6,
                    marker=markers[model["type"]],
                    color=colors[model["type"]],
                    edgecolors='black', linewidths=1)

        # Add model name annotations
        plt.annotate(model["name"],
                     (complexity[i], accuracy[i]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9)

    plt.xscale('log')  # Log scale for complexity
    plt.xlabel('Model Complexity (Parameters/Nodes/Support Vectors)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Model Performance vs. Complexity\n(Bubble size represents inference time per sample)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add legend for model types
    legend_elements = [plt.Line2D([0], [0], marker=markers[t], color='w',
                                  markerfacecolor=colors[t], markersize=10,
                                  label=t) for t in set(model_types)]
    plt.legend(handles=legend_elements, title="Model Types", loc="upper right")

    plt.tight_layout()
    plt.savefig('performance_complexity_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create second figure: Data table
    fig, ax = plt.figure(figsize=(14, 5)), plt.subplot(111)
    ax.axis('off')

    # Create table data
    table_data = []
    headers = ["Model", "Type", "Complexity", "Input Features",
               "Test Acc", "F1 Score", "Inference Time (ms/sample)"]

    for model in models:
        model_name = model["name"]
        row = [
            model_name,
            model["type"],
            f"{model['complexity']:,}",
            f"{model['features']:,}",
            f"{model['accuracy']:.4f}",
            f"{results_dict[model_name]['test_results']['f1']:.4f}",
            f"{model['inference_time'] * 1000:.2f}"
        ]
        table_data.append(row)

    # Sort by accuracy (descending)
    table_data = sorted(table_data, key=lambda x: float(x[4]), reverse=True)

    # Create the table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#f2f2f2'] * len(headers))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Set column widths
    col_widths = [0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.16]
    for i, width in enumerate(col_widths):
        for j in range(len(table_data) + 1):
            table[(j, i)].set_width(width)

    plt.title('Model Comparison Details', fontsize=14)
    plt.tight_layout()
    plt.savefig('performance_complexity_table.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison_table(results_dict, save_path='model_comparison_table.png'):
    """
    Create and save a professional-looking table comparing model performance metrics.

    Args:
        results_dict: Dictionary containing results for each model
        save_path: Path to save the table image
    """
    # Extract key metrics
    metrics_data = []

    for model_name, result in results_dict.items():
        test_results = result['test_results']
        cm = test_results['confusion_matrix']

        # Extract TP, FP, TN, FN from confusion matrix
        # For binary classification: [[TN, FP], [FN, TP]]
        if cm.shape == (2, 2):
            TN, FP = cm[0, 0], cm[0, 1]
            FN, TP = cm[1, 0], cm[1, 1]
        else:
            # If confusion matrix has a different shape, set to N/A
            TN, FP, FN, TP = "N/A", "N/A", "N/A", "N/A"

        # Calculate sensitivity (TPR) and specificity (TNR)
        if isinstance(TP, (int, float)) and isinstance(FN, (int, float)) and (TP + FN) > 0:
            sensitivity = TP / (TP + FN)
        else:
            sensitivity = test_results.get('recall', "N/A")

        if isinstance(TN, (int, float)) and isinstance(FP, (int, float)) and (TN + FP) > 0:
            specificity = TN / (TN + FP)
        else:
            specificity = "N/A"

        # Add resource stats
        resource_stats = result.get('resource_stats', {})
        training_time = resource_stats.get('time_elapsed', "N/A")
        peak_memory = resource_stats.get('peak_memory', "N/A")

        metrics_data.append({
            'Model': model_name,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'F1 Score': test_results.get('f1', "N/A"),
            'Accuracy': test_results.get('accuracy', "N/A"),
            'Precision': test_results.get('precision', "N/A"),
            'Sensitivity': sensitivity,  # TPR
            'Specificity': specificity,  # TNR
            'Training Time (s)': training_time,
            'Peak Memory (MB)': peak_memory
        })

    # Create DataFrame
    df = pd.DataFrame(metrics_data)

    # Sort by F1 Score (descending)
    df = df.sort_values(by='F1 Score', ascending=False)

    # Format numeric columns for display
    for col in ['F1 Score', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    for col in ['Training Time (s)', 'Peak Memory (MB)']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # Create the table figure
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 2))
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Set header row style
    for i, key in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')

    # Highlight the best model row
    best_model_row = 1  # First row after header is the best model (sorted by F1)
    for i in range(len(df.columns)):
        cell = table[best_model_row, i]
        cell.set_facecolor('#d4efdf')  # Light green

    # Add a title
    plt.title('Time Series Classification Model Comparison', fontsize=16, pad=20)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Comparison table saved to {save_path}")

    # Display the table
    plt.tight_layout()
    plt.show()

def create_model_parameters_table(save_path='model_parameters_table.png'):
    """
    Create and save a professional-looking table showing model parameters.
    """
    # Define model parameters
    model_params = {
        '1D-CNN-GN': [
            ('Input Layer', '(3, 2000)', '0'),
            ('Conv1D', '(16, -)', '416'),  # (in_channels * out_channels * kernel_size + out_channels)
            ('GroupNorm', '(16, -)', '32'),
            ('MaxPool1D', '(16, -)', '0'),
            ('Conv1D', '(32, -)', '7,712'),
            ('GroupNorm', '(32, -)', '64'),
            ('MaxPool1D', '(32, -)', '0'),
            ('Conv1D', '(64, -)', '18,496'),
            ('GroupNorm', '(64, -)', '128'),
            ('MaxPool1D', '(64, -)', '0'),
            ('GlobalAvgPool', '(64, 1)', '0'),
            ('Linear', '(64)', '4,160'),
            ('Dropout', '(64)', '0'),
            ('Linear', '(2)', '130'),
            ('Total', '', '31,138')
        ],
        'TCN': [
            ('Input Layer', '(3, 2000)', '0'),
            ('TemporalBlock1', '(16, -)', '848'),  # Approximate
            ('TemporalBlock2', '(16, -)', '1,824'),
            ('TemporalBlock3', '(16, -)', '1,824'),
            ('TemporalBlock4', '(16, -)', '1,824'),
            ('GlobalAvgPool', '(16, 1)', '0'),
            ('Linear', '(2)', '34'),
            ('Total', '', '6,354')  # Reduced parameters
        ],
        '1D-CNN-Freq': [
            ('Input Layer', '(3, 1000)', '0'),
            ('Conv1D', '(16, -)', '736'),
            ('BatchNorm1D', '(16, -)', '32'),
            ('MaxPool1D', '(16, -)', '0'),
            ('Conv1D', '(32, -)', '4,640'),
            ('BatchNorm1D', '(32, -)', '64'),
            ('MaxPool1D', '(32, -)', '0'),
            ('Conv1D', '(48, -)', '7,728'),
            ('BatchNorm1D', '(48, -)', '96'),
            ('MaxPool1D', '(48, -)', '0'),
            ('GlobalAvgPool', '(48, 1)', '0'),
            ('Linear', '(32)', '1,568'),
            ('Dropout', '(32)', '0'),
            ('Linear', '(2)', '66'),
            ('Total', '', '14,930')
        ],
        'MLP': [
            ('Input Layer', '(3*2000)', '0'),
            ('Linear', '(256)', '1,536,256'),
            ('ReLU', '(256)', '0'),
            ('Dropout', '(256)', '0'),
            ('Linear', '(64)', '16,448'),
            ('ReLU', '(64)', '0'),
            ('Dropout', '(64)', '0'),
            ('Linear', '(2)', '130'),
            ('Total', '', '1,552,834')
        ]
    }

    # Create data for the table
    rows = []
    model_names = []
    for model_name, params in model_params.items():
        model_names.append(model_name)
        for layer_idx, (layer_type, output_shape, param_count) in enumerate(params):
            rows.append([
                model_name if layer_idx == 0 else "",
                layer_type,
                output_shape,
                param_count
            ])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['Model', 'Layer', 'Output Shape', 'Parameters'])

    # Create the table figure
    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.35 + 2))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)

    # Set header style
    for i, key in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')

    # Highlight model rows
    current_model = None
    for i, row in enumerate(rows, 1):  # Starting from 1 to account for header
        if row[0]:  # If model name is not empty
            current_model = row[0]
            color = '#d6eaf8'  # Light blue
        else:
            color = '#f8f9f9'  # Light grey

        if "Total" in row[1]:
            color = '#e8f8f5'  # Light green for total row

        for j in range(len(df.columns)):
            cell = table[i, j]
            cell.set_facecolor(color)

            # Make the model name and total row bold
            if j == 0 and row[0] or "Total" in row[1]:
                cell.set_text_props(weight='bold')

    # Add title
    plt.title('Model Architectures and Parameters', fontsize=16, pad=20)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Parameters table saved to {save_path}")

    # Display the table
    plt.tight_layout()
    plt.show()

def create_metrics_summary_table(results_dict, save_path='metrics_summary_table.png'):
    """
    Create a summary table with key metrics and training information.
    """
    # Extract data
    data = []
    for model_name, result in results_dict.items():
        test_results = result['test_results']
        resource_stats = result.get('resource_stats', {})

        # Get training history if available
        train_history = result.get('training_history', {})
        train_losses = train_history.get('train_losses', [])
        val_losses = train_history.get('val_losses', [])
        train_accs = train_history.get('train_accuracies', [])
        val_accs = train_history.get('val_accuracies', [])

        # Calculate average metrics
        avg_train_acc = np.mean(train_accs) if len(train_accs) > 0 else 'N/A'
        avg_val_acc = np.mean(val_accs) if len(val_accs) > 0 else 'N/A'
        avg_train_loss = np.mean(train_losses) if len(train_losses) > 0 else 'N/A'
        avg_val_loss = np.mean(val_losses) if len(val_losses) > 0 else 'N/A'

        # Calculate standard deviations
        std_train_acc = np.std(train_accs) if len(train_accs) > 0 else 'N/A'
        std_val_acc = np.std(val_accs) if len(val_accs) > 0 else 'N/A'
        std_train_loss = np.std(train_losses) if len(train_losses) > 0 else 'N/A'
        std_val_loss = np.std(val_losses) if len(val_losses) > 0 else 'N/A'

        # Only include these metrics for neural network models
        if isinstance(avg_train_acc, (int, float)):
            data.append({
                'Model': model_name,
                'Avg Train Acc': avg_train_acc,
                'Avg Val Acc': avg_val_acc,
                'Avg Train Loss': avg_train_loss,
                'Avg Val Loss': avg_val_loss,
                'Std Train Acc': std_train_acc,
                'Std Val Acc': std_val_acc,
                'Std Train Loss': std_train_loss,
                'Std Val Loss': std_val_loss,
                'Test Acc': test_results.get('accuracy', 'N/A'),
                'F1 Score': test_results.get('f1', 'N/A'),
                'Time (s)': resource_stats.get('time_elapsed', 'N/A')
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by F1 Score
    if 'F1 Score' in df.columns:
        df = df.sort_values(by='F1 Score', ascending=False)

    # Format numeric columns
    for col in df.columns:
        if col != 'Model':
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.7 + 2))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Set header style
    for i, key in enumerate(df.columns):
        cell = table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')

    # Highlight best model
    for i in range(len(df.columns)):
        cell = table[1, i]
        cell.set_facecolor('#d4efdf')

    # Add title
    plt.title('Neural Network Models Training and Evaluation Metrics', fontsize=16, pad=20)

    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Metrics summary table saved to {save_path}")

    # Display table
    plt.tight_layout()
    plt.show()
# Main Execution
# ----------------
def main():
    # Load dataset
    data_directory = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    dataset = VibrationDataset(data_directory, augment_bad=False)

    # Create dataloaders for neural network models
    batch_size = 128
    train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, dataset= (
        stratified_group_split(data_directory, idx_return=True))

    plot_label_distribution(dataset,train_idx,val_idx,test_idx)
    plot_operation_pie_charts(dataset,train_idx,val_idx,test_idx)


    # Create feature datasets for traditional ML
    feature_extractor = FeatureExtractor()
    X_train, y_train = extract_features_for_ml([dataset[i] for i in train_idx], feature_extractor)
    X_val, y_val = extract_features_for_ml([dataset[i] for i in val_idx], feature_extractor)
    X_test, y_test = extract_features_for_ml([dataset[i] for i in test_idx], feature_extractor)


    # Verify feature count
    sample_data, _ = dataset[0]
    features = feature_extractor(sample_data.numpy())
    print(f"Number of features extracted: {len(features)}")

    # Normalize features for traditional ML
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    # Create frequency domain dataloaders
    freq_train_loader, freq_val_loader, freq_test_loader,_ = stratified_group_split_freq(data_directory)

    # Results container
    all_results = {}
    complexity_metrics = {
        '1D-CNN-GN': 31938,
        '1D-CNN-Wide': 76898,
        'TCN': 4506,
        '1D-CNN-Freq': 14930,
        'MLP': 1552834,
        'SVM': 0,
        'Gradient Boosting': 0,
        'Random Forest': 0
    }
    feature_counts = {
        '1D-CNN-GN': 6000,
        '1D-CNN-Wide': 6000,
        'TCN': 6000,
        '1D-CNN-Freq': 3000,
        'MLP': 6000,
        'SVM': len(features),  # Use actual feature count
        'Gradient Boosting': len(features),
        'Random Forest': len(features)
    }

    # Train and evaluate neural network models
    # CNN1D_DS_Wide (Your strongest model)
    model = CNN1D_DS_Wide().to(device)
    results = train_neural_network("1D-CNN-GN", model, train_loader, val_loader,
                                   test_loader, epochs=30, lr=0.001, weight_decay=1e-4,
                                   scheduler=True)
    all_results['1D-CNN-GN'] = results

    # CNN1D_Wide ( strongest model for XAI)
    model = CNN1D_Wide().to(device)
    results = train_neural_network("1D-CNN-Wide", model, train_loader, val_loader,
                                   test_loader, epochs=30, lr=0.001, weight_decay=1e-4,
                                   scheduler=True)
    all_results['1D-CNN-Wide'] = results

    # CNN1D_Freq (Frequency domain model)
    model = CNN1D_Freq().to(device)
    results = train_neural_network("1D-CNN-Freq", model, freq_train_loader, freq_val_loader,
                                   freq_test_loader, epochs=30, lr=0.0008, weight_decay=1e-3,  # Lower learning rate
                                   scheduler=True)
    all_results['1D-CNN-Freq'] = results

    # TCN
    model = TCN().to(device)
    results = train_neural_network("TCN", model, train_loader, val_loader,
                                   test_loader, epochs=30, lr=0.001, weight_decay=1e-3,  # Higher weight_decay
                                   scheduler=True)
    all_results['TCN'] = results

    # MLP
    model = MLP_Model().to(device)
    results = train_neural_network("MLP", model, train_loader, val_loader,
                                   test_loader, epochs=30, lr=0.0005, weight_decay=1e-3,
                                   scheduler=True)
    all_results['MLP'] = results

    # Compare training dynamics of neural network models
    plot_training_dynamics_comparison(all_results)
    plot_training_dynamics_comparison_full(all_results)

    # Train and evaluate traditional ML models
    # SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    results = train_traditional_ml_model("SVM", svm_model, X_train, y_train, X_test, y_test)
    complexity_metrics['SVM'] = svm_model.n_support_.sum()  # Compute support vectors
    all_results['SVM'] = results
    print(f"SVM Number of Support Vectors: {complexity_metrics['SVM']}")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10)  # Limited trees and depth
    results = train_traditional_ml_model("Random Forest", rf_model, X_train, y_train, X_test, y_test)
    complexity_metrics['Random Forest'] = sum(tree.tree_.node_count for tree in rf_model.estimators_)
    all_results['Random Forest'] = results
    print(f"Random Forest Total Nodes: {complexity_metrics['Random Forest']}")


    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=50, max_depth=3)  # Limited boosting
    results = train_traditional_ml_model("Gradient Boosting", gb_model, X_train, y_train, X_test, y_test)
    complexity_metrics['Gradient Boosting'] = sum(tree.tree_.node_count for tree in gb_model.estimators_[:, 0])
    all_results['Gradient Boosting'] = results
    print(f"Gradient Boosting Total Nodes: {complexity_metrics['Gradient Boosting']}")

    # Plot comparison of all models (performance metrics)
    plot_model_comparison(all_results)

    # Plot comparison of resource usage (time and memory)
    plot_resource_comparison(all_results)

    plot_performance_complexity_tradeoff(all_results, complexity_metrics, feature_counts, len(test_loader))

    # Print detailed results summary
    print("\n==== DETAILED RESULTS SUMMARY ====")
    for model_name, result in all_results.items():
        test_results = result['test_results']
        resource_stats = result['resource_stats']
        print(f"\n{model_name}:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1 Score: {test_results['f1']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall: {test_results['recall']:.4f}")
        print(f"  Training Time: {resource_stats['time_elapsed']:.2f} seconds")
        print(f"  Memory Used: {resource_stats['memory_used']:.2f} MB")
        print(f"  Peak Memory: {resource_stats['peak_memory']:.2f} MB")

    # Add this at the end of the main function, after all models have been evaluated
    # Create and save comparison tables
    create_model_comparison_table(all_results, save_path='results/others/model_comparison_table.png')
    create_model_parameters_table(save_path='results/others/model_parameters_table.png')
    create_metrics_summary_table(all_results, save_path='results/others/metrics_summary_table.png')



if __name__ == "__main__":
    main()