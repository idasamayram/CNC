import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset, WeightedRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_recall_curve, auc
from utils.models import CNN1D_DS
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import time
from datetime import timedelta


# ------------------------
# 1️⃣ Custom Dataset Class
# ------------------------

class VibrationDataset(Dataset):
    '''
    This version includes the operation data so that it can be used for stratified
    sampling in the train/val/test split.
    '''

    def __init__(self, data_dir, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []  # Optional for operation-based stratification
        self.augment_bad = augment_bad
        self.file_groups = []  # e.g., 'M01_Feb_2019_OP02_000'

        for label, label_idx in zip(["good", "bad"], [0, 1]):  # 0=good, 1=bad
            folder = self.data_dir / label
            for file_name in folder.glob("*.h5"):
                self.file_paths.append(file_name)
                self.labels.append(label_idx)
                # Extract operation (e.g., 'OP02' from 'M01_Feb_2019_OP02_000_window_0.h5')
                operation = file_name.stem.split('_')[3]
                self.operations.append(operation)
                # Extract file group (e.g., 'M01_Feb_2019_OP02_000')
                file_group = file_name.stem.rsplit('_window_', 1)[0]
                self.file_groups.append(file_group)

        self.labels = np.array(self.labels)
        self.operations = np.array(self.operations)
        self.file_groups = np.array(self.file_groups)
        assert len(self.file_paths) == 7501, f"Expected 7501 files, found {len(self.file_paths)}"

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN

        # Pre-compute frequency domain features (more efficient)
        freq_data = np.abs(np.fft.rfft(data, axis=1))

        label = self.labels[idx]

        # Enhanced augmentation for bad samples
        if self.augment_bad and label == 1:
            # Add Gaussian noise with dynamic variance based on signal amplitude
            noise_level = 0.05 * np.std(data)
            data += np.random.normal(0, noise_level, data.shape)

            # Recalculate frequency features after augmentation
            freq_data = np.abs(np.fft.rfft(data, axis=1))

        return (torch.tensor(data, dtype=torch.float32),
                torch.tensor(freq_data, dtype=torch.float32)), torch.tensor(label, dtype=torch.long)


# ------------------------
# ------------------------

# 2️⃣ Define the CNN Model for downsampled data

class CNN1D_DS(nn.Module):
    def __init__(self):
        super(CNN1D_DS, self).__init__()
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=9, stride=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=7, stride=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool1d(1)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward_features(self, x):
        """Extract features only (before classification)"""
        x = self.feature_extractor(x)
        x = x.squeeze(-1)  # Remove the last dimension (from global avg pooling)
        return x

    def forward(self, x):
        """Full forward pass including classification"""
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


class FrequencyDomainCNN(nn.Module):
    def __init__(self):
        super(FrequencyDomainCNN, self).__init__()
        self.time_cnn = CNN1D_DS()  # Your existing time domain CNN

        # Frequency domain branch
        self.freq_conv1 = nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=2)
        self.freq_gn1 = nn.GroupNorm(4, 16)
        self.freq_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.freq_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.freq_gn2 = nn.GroupNorm(4, 32)
        self.freq_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.freq_global_pool = nn.AdaptiveAvgPool1d(1)

        # Fusion layer
        self.fusion = nn.Linear(64 + 32, 32)
        self.classifier = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Track intermediate activations for XAI methods like LRP
        self.intermediate_activations = {}

    def forward(self, x):
        # Check if x is a tuple or list
        if isinstance(x, (tuple, list)):
            # Handle the case of separate time and frequency inputs
            x_time, x_freq = x[0], x[1]
            
            # Time domain branch
            time_features = self.time_cnn.forward_features(x_time)
            
            # Frequency domain branch
            freq1 = self.freq_conv1(x_freq)
            self.intermediate_activations['freq_conv1'] = freq1
            
            freq1 = self.freq_gn1(freq1)
            freq1 = self.relu(freq1)
            freq1 = self.freq_pool1(freq1)
            self.intermediate_activations['freq_pool1'] = freq1
            
            freq2 = self.freq_conv2(freq1)
            self.intermediate_activations['freq_conv2'] = freq2
            
            freq2 = self.freq_gn2(freq2)
            freq2 = self.relu(freq2)
            freq2 = self.freq_pool2(freq2)
            self.intermediate_activations['freq_pool2'] = freq2
            
            x_freq = self.freq_global_pool(freq2).squeeze(-1)
            self.intermediate_activations['freq_features'] = x_freq
        else:
            # Handle the case of a single input tensor
            time_features = self.time_cnn.forward_features(x)
            
            # Compute FFT for the frequency domain branch
            x_freq = torch.abs(torch.fft.rfft(x, dim=2))
            x_freq = self.freq_pool1(self.relu(self.freq_gn1(self.freq_conv1(x_freq))))
            x_freq = self.freq_pool2(self.relu(self.freq_gn2(self.freq_conv2(x_freq))))
            x_freq = self.freq_global_pool(x_freq).squeeze(-1)

        # Fusion
        self.intermediate_activations['time_features'] = time_features
        combined = torch.cat([time_features, x_freq], dim=1)
        self.intermediate_activations['combined'] = combined
        
        x = self.relu(self.fusion(combined))
        self.intermediate_activations['fused'] = x
        
        x = self.dropout(x)
        x = self.classifier(x)
        self.intermediate_activations['output'] = x

        return x

    # Helper method for XAI compatibility
    def get_activations(self):
        return self.intermediate_activations


class CNN_1d(nn.Module):
    def __init__(self, dropout=0.3, n_out=2):
        super(CNN_1d, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=9, stride=1, padding=4),  # Reduce filters, increase kernel size
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),

            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout)
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),  # Add intermediate FC layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_out)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x


# ------------------------
# 3️⃣ Train & Evaluate Functions
# ------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    # Add timing
    start_time = time.time()
    
    for batch in train_loader:
        inputs, labels = batch
        
        # Fix: Handle both tuple inputs and potential list-type inputs
        if isinstance(inputs, (tuple, list)):
            if len(inputs) == 2:
                # Handle case when inputs is (time_domain, freq_domain)
                time_domain = inputs[0].to(device)
                freq_domain = inputs[1].to(device)
                inputs = (time_domain, freq_domain)
        else:
            inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    epoch_time = time.time() - start_time
    
    return total_loss / len(train_loader), accuracy, epoch_time


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    misclassified_indices = []
    all_preds, all_labels, all_probs = [], [], []  # For computing PR-AUC

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, labels = batch
            
            # Fix: Handle both tuple inputs and potential list-type inputs
            if isinstance(inputs, (tuple, list)):
                if len(inputs) == 2:
                    # Handle case when inputs is (time_domain, freq_domain)
                    time_domain = inputs[0].to(device)
                    freq_domain = inputs[1].to(device)
                    inputs = (time_domain, freq_domain)
            else:
                inputs = inputs.to(device)

            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions, true labels, and probabilities
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1

            # Collect indices of misclassified samples
            batch_indices = torch.where(predicted != labels)[0]
            for idx in batch_indices:
                global_idx = batch_idx * val_loader.batch_size + idx.item()
                if global_idx < len(val_loader.dataset):  # Avoid index errors
                    misclassified_indices.append(global_idx)

    # Calculate PR-AUC if possible (better for imbalanced datasets)
    pr_auc = 0.0
    if len(np.unique(all_labels)) > 1:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)

    val_loss /= len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc, misclassified_indices, pr_auc


# ------------------------
# 4️⃣ Test the Model
# ------------------------
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct = 0
    total = 0
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            
            # Fix: Handle both tuple inputs and potential list-type inputs
            if isinstance(inputs, (tuple, list)):
                if len(inputs) == 2:
                    # Handle case when inputs is (time_domain, freq_domain)
                    time_domain = inputs[0].to(device)
                    freq_domain = inputs[1].to(device)
                    inputs = (time_domain, freq_domain)
            else:
                inputs = inputs.to(device)

            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    # Compute metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = correct / total
    
    # Calculate PR-AUC for test set
    pr_auc = 0.0
    if len(np.unique(all_labels)) > 1:
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall, precision)
    
    print(f"Test PR-AUC: {pr_auc:.4f}")

    return f1, accuracy


# ------------------------
# 5️⃣ Full Training Pipeline
# ------------------------
def train_and_evaluate(train_loader, val_loader, test_loader, epochs=20, lr=0.001, weight_decay=1e-4,
                       EralyStopping=False, Schedule=False, patience=3, monitor_metric="loss",
                       early_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model setup
    model = FrequencyDomainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_pr_auc = -float('inf')
    best_model_weights = None
    patience_counter = 0
    early_stop_epoch = epochs
    total_training_time = 0

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    pr_aucs = []  # Store PR-AUC values
    epoch_times = []  # Track epoch times

    print(f"Starting training with monitoring metric: {monitor_metric}")
    print(f"Early stopping patience: {patience}, delta: {early_delta}")
    
    for epoch in range(epochs):
        # Training with timing
        train_loss, train_acc, epoch_time = train_epoch(model, train_loader, optimizer, criterion, device)
        total_training_time += epoch_time
        epoch_times.append(epoch_time)
        
        val_loss, val_acc, _, val_pr_auc = validate_epoch(model, val_loader, criterion, device)

        # Step the scheduler
        if Schedule:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
        else:
            current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        pr_aucs.append(val_pr_auc)

        # Format epoch time as minutes:seconds
        epoch_time_str = str(timedelta(seconds=int(epoch_time)))
        
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, PR-AUC: {val_pr_auc:.4f} - "
              f"LR: {current_lr:.6f} - Time: {epoch_time_str}")

        if EralyStopping:
            improved = False
            
            # Choose monitoring metric
            if monitor_metric == "pr_auc":
                # For PR-AUC, higher is better
                if val_pr_auc > best_val_pr_auc + early_delta:
                    best_val_pr_auc = val_pr_auc
                    improved = True
                    print(f"  → PR-AUC improved to {best_val_pr_auc:.4f}")
                    
            else:  # Default to loss monitoring
                # For loss, lower is better
                if val_loss < best_val_loss - early_delta:
                    best_val_loss = val_loss
                    improved = True
                    print(f"  → Loss improved to {best_val_loss:.4f}")
            
            if improved:
                best_model_weights = model.state_dict()  # Save the best model weights
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1  # Increment counter if no improvement
                print(f"  → No improvement for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    early_stop_epoch = epoch + 1
                    break

    # Restore the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        monitored_value = best_val_pr_auc if monitor_metric == "pr_auc" else best_val_loss
        print(f"Restored best model weights with {monitor_metric}: {monitored_value:.4f}")

    # Print total training time
    print(f"✅ Training complete! Total training time: {str(timedelta(seconds=int(total_training_time)))}")
    print(f"Average epoch time: {str(timedelta(seconds=int(np.mean(epoch_times))))}")

    # Evaluate on the test set
    f1, accuracy = test_model(model, test_loader, device)
    print(f"🔥 Test F1 Score: {f1:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot metrics and timing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    epochs_range = range(1, (early_stop_epoch if EralyStopping else epochs) + 1)
    
    # Loss plot
    ax1.plot(epochs_range, train_losses[:len(epochs_range)], label="Train Loss")
    ax1.plot(epochs_range, val_losses[:len(epochs_range)], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs_range, train_accuracies[:len(epochs_range)], label="Train Accuracy")
    ax2.plot(epochs_range, val_accuracies[:len(epochs_range)], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    
    # PR-AUC plot
    ax3.plot(epochs_range, pr_aucs[:len(epochs_range)], label="PR-AUC", color='purple')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("PR-AUC")
    ax3.set_title("Precision-Recall AUC")
    ax3.legend()
    
    # Epoch timing plot
    ax4.bar(epochs_range, epoch_times[:len(epochs_range)], color='green')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Time (s)")
    ax4.set_title("Epoch Training Time")

    plt.tight_layout()
    plt.show()

    return model

# Define a function to explain the model's prediction - useful for XAI methods later
def explain_model_prediction(model, input_data, device):
    """
    Demonstrates how the model can work with XAI methods like LRP
    This is just a skeleton that would need to be filled in with actual LRP code
    """
    model.eval()
    if isinstance(input_data, (tuple, list)):
        input_data = (input_data[0].to(device), input_data[1].to(device))
    else:
        input_data = input_data.to(device)

    # Forward pass with gradient tracking for XAI methods
    outputs = model(input_data)

    # Get activations for visualization
    activations = model.get_activations()
    return outputs, activations
# ------------------------
# 6️⃣ Run Training & Evaluation
# ------------------------

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch device:", device)
    
    # Print CUDA info if available
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")

    # Splitting the dataset - use augmentation for better handling of class imbalance
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data"
    dataset = VibrationDataset(data_directory, augment_bad=True)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=stratify_key
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[stratify_key[i] for i in temp_idx]
    )

    # Create Subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Use weighted sampler to address class imbalance (2.7:1 ratio)
    train_labels = np.array([dataset.labels[i] for i in train_idx])
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Verify split sizes and label distribution
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Train good: {sum(dataset.labels[train_idx] == 0)}, Train bad: {sum(dataset.labels[train_idx] == 1)}")
    print(f"Val good: {sum(dataset.labels[val_idx] == 0)}, Val bad: {sum(dataset.labels[val_idx] == 1)}")
    print(f"Test good: {sum(dataset.labels[test_idx] == 0)}, Test bad: {sum(dataset.labels[test_idx] == 1)}")

    # Class ratios
    train_ratio = sum(dataset.labels[train_idx] == 0) / sum(dataset.labels[train_idx] == 1)
    val_ratio = sum(dataset.labels[val_idx] == 0) / sum(dataset.labels[val_idx] == 1)
    test_ratio = sum(dataset.labels[test_idx] == 0) / sum(dataset.labels[test_idx] == 1)
    print(f"Class ratio (good/bad) - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    # Creating DataLoaders with efficient parallel loading
    batch_size = 128 
    
    # Try optimizing the number of workers based on your CPU cores
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count())
    print(f"Using {num_workers} workers for data loading")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=num_workers,  # Parallel data loading for efficiency
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Here you can choose what metric to monitor for early stopping
    # "loss" (default) or "pr_auc"
    monitor_metric = "pr_auc"  # This would stop when PR-AUC stops improving
    
    freq_model = train_and_evaluate(
        train_loader,
        val_loader,
        test_loader,
        epochs=30,  # Increased epochs
        lr=0.0005,  # Lower learning rate for better convergence
        weight_decay=2e-4,  # Increased weight decay for better regularization
        EralyStopping=True,
        Schedule=True,
        patience=3,  # Increased patience to avoid stopping too early
        monitor_metric=monitor_metric,
        early_delta=0.0001  # Smaller delta for more sensitive early stopping
    )

    # Save the best model
    torch.save(freq_model.state_dict(), "../efficient_cnn1d_freq_model.ckpt")
    print("✅ Model saved to efficient_cnn1d_freq_model.ckpt")
    freq_model.to(device)
    freq_model.eval()  # Switch to evaluation mode
    
    # Fix: Using get_activations() method instead of get_activation()
    activations = freq_model.get_activations()
    print(f"Available activation keys: {list(activations.keys())}")
    
    # Example of how to use model for explanation
    # Get a test sample
    test_sample_batch, test_sample_label = next(iter(test_loader))
    
    # Get model prediction and activations for this sample
    outputs, sample_activations = explain_model_prediction(freq_model, test_sample_batch, device)
    
    # Print prediction
    prediction = torch.argmax(outputs, dim=1)[0].item()
    prob = torch.softmax(outputs, dim=1)[0, prediction].item()
    label_name = "NOK" if prediction == 1 else "OK"
    print(f"Model prediction: {label_name} (class {prediction}) with confidence {prob:.4f}")
    
    print("\nTo use LRP and other XAI methods, import the ModelExplainer class:")
    print("from model_explainability import ModelExplainer")
    print("\nexplainer = ModelExplainer(freq_model, device)")
    print("lrp_result = explainer.explain_prediction_lrp(test_sample_batch)")
    print("explainer.visualize_attribution(lrp_result, 'LRP')")
