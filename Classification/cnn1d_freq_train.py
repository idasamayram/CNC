import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import os


class VibrationFreqDataset(Dataset):
    def __init__(self, data_dir, augment_bad=False, n_fft=2048, normalize=True):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []
        self.augment_bad = augment_bad
        self.file_groups = []
        self.n_fft = n_fft
        self.normalize = normalize

        for label, label_idx in zip(["good", "bad"], [0, 1]):
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

        # Transpose to have shape (3, 2000)
        data = np.transpose(data, (1, 0))

        # Normalize time domain data first - per channel normalization
        for ch in range(data.shape[0]):
            ch_std = np.std(data[ch])
            if ch_std > 0:  # Avoid division by zero
                data[ch] = (data[ch] - np.mean(data[ch])) / ch_std

        # Apply window function with proper scaling
        if self.window == 'hann':
            window_func = np.hanning(data.shape[1])
        elif self.window == 'hamming':
            window_func = np.hamming(data.shape[1])
        elif self.window == 'blackman':
            window_func = np.blackman(data.shape[1])
        else:
            window_func = np.ones(data.shape[1])

        # Scale window to preserve signal energy
        window_func = window_func / np.sqrt(np.mean(window_func ** 2))
        windowed_data = data * window_func[None, :]

        # Apply FFT to each channel with simplified, robust feature extraction
        freq_data = []
        for channel in range(data.shape[0]):
            # Compute FFT
            fft_result = np.fft.rfft(windowed_data[channel], n=self.n_fft)

            # Get log magnitude spectrum (dB scale)
            magnitude = np.abs(fft_result)
            # Use log10 with offset to avoid numerical issues
            magnitude_db = 10 * np.log10(magnitude + 1e-6)

            # Simple robust scaling to [0, 1] range
            magnitude_min = np.min(magnitude_db)
            magnitude_max = np.max(magnitude_db)
            if magnitude_max > magnitude_min:
                magnitude_norm = (magnitude_db - magnitude_min) / (magnitude_max - magnitude_min)
            else:
                magnitude_norm = np.zeros_like(magnitude_db)

            # Just use magnitude as the primary feature
            freq_data.append(magnitude_norm)

        # Stack channels: (3, n_fft//2+1)
        freq_data = np.array(freq_data)

        label = self.labels[idx]

        return torch.tensor(freq_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CNN1D_Freq(nn.Module):
    def __init__(self):
        super(CNN1D_Freq, self).__init__()
        # Frequency domain-specific architecture with wider kernels and fewer layers

        # First layer with large kernel to better capture frequency patterns
        self.conv1 = nn.Conv1d(3, 32, kernel_size=15, stride=1, padding=7)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout(0.1)

        # Second layer - slightly smaller kernel
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(0.15)

        # Third layer
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=3)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout3 = nn.Dropout(0.2)

        # Global pooling instead of flattening
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.fc = nn.Linear(64, 2)

        # Non-linearity
        self.relu = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # More conservative initialization
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Third block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.fc(x)

        return x

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, (all_preds, all_targets)


def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(all_targets, all_preds, average='weighted')
    accuracy = accuracy_score(all_targets, all_preds)

    return f1, accuracy


def train_and_evaluate_freq(train_loader, val_loader, test_loader, epochs=25, lr=0.0001, weight_decay=1e-5,
                            EarlyStopping=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create models directory if it doesn't exist
    os.makedirs("../models", exist_ok=True)

    # Model setup
    model = CNN1D_Freq().to(device)

    # Loss function with class balancing
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping with increased patience
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    patience = 7  # Increased patience
    early_stop_epoch = epochs

    # Learning rate scheduler with gentler decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3, verbose=True
    )

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # Training
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = validate_epoch(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Learning Rate: {current_lr:.6f}")

        # Early stopping with improved patience
        if EarlyStopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    early_stop_epoch = epoch + 1
                    break

    # Load best model weights
    if EarlyStopping and best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Restored best model with Val Loss: {best_val_loss:.4f}")

    # Test evaluation
    f1, accuracy = test_model(model, test_loader, device)
    print(f"Test F1 Score: {f1:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    if EarlyStopping:
        plot_range = range(1, early_stop_epoch + 1)
    else:
        plot_range = range(1, epochs + 1)

    ax1.plot(plot_range, train_losses[:len(plot_range)], label="Train Loss")
    ax1.plot(plot_range, val_losses[:len(plot_range)], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(plot_range, train_accuracies[:len(plot_range)], label="Train Accuracy")
    ax2.plot(plot_range, val_accuracies[:len(plot_range)], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("../models/freq_training_curves.png")

    # Save model
    model_path = "../models/cnn1d_freq_model.ckpt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    return model

def train_and_evaluate_with_cross_validation(dataset, n_splits=5, batch_size=128, epochs=20, lr=0.001,
                                             weight_decay=1e-4, early_stopping=True, scheduling=True,
                                             save_dir="../models"):
    from sklearn.model_selection import KFold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # Initialize KFold cross-validator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Metrics storage
    fold_val_accuracies = []
    fold_val_losses = []
    fold_test_f1_scores = []
    fold_test_accuracies = []
    best_models = []

    # For each fold
    for fold, (train_temp_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n{'=' * 20} Fold {fold + 1}/{n_splits} {'=' * 20}")

        # Further split train_temp into train and validation
        train_idx, val_idx = train_test_split(train_temp_idx, test_size=0.2,
                                              stratify=[stratify_key[i] for i in train_temp_idx])

        # Create Subset datasets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        # Creating DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize and train model
        model = CNN1D_Freq().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 3

        # Learning rate scheduler
        if scheduling:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        # Training and validation loop
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _ = validate_epoch(model, val_loader, criterion, device)

            # Step the scheduler
            if scheduling:
                scheduler.step()

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping check
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        # Load the best model if early stopping was used
        if early_stopping and best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save this fold's model
        fold_model_path = os.path.join(save_dir, f"fold_{fold + 1}_model.ckpt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Fold {fold + 1} model saved at: {fold_model_path}")

        # Test the model
        test_f1, test_acc = test_model(model, test_loader, device)
        print(f"Fold {fold + 1} Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")

        # Store metrics for this fold
        fold_val_accuracies.append(val_accuracies[-1])
        fold_val_losses.append(val_losses[-1])
        fold_test_f1_scores.append(test_f1)
        fold_test_accuracies.append(test_acc)
        best_models.append(model)

        # Plot metrics for this fold
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Fold {fold + 1} Training and Validation Loss")
        ax1.legend()

        ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Fold {fold + 1} Training and Validation Accuracy")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    # Print average metrics across folds
    print("\nCross-Validation Results:")
    print(f"Average Validation Accuracy: {np.mean(fold_val_accuracies):.4f} (±{np.std(fold_val_accuracies):.4f})")
    print(f"Average Validation Loss: {np.mean(fold_val_losses):.4f} (±{np.std(fold_val_losses):.4f})")
    print(f"Average Test F1 Score: {np.mean(fold_test_f1_scores):.4f} (±{np.std(fold_test_f1_scores):.4f})")
    print(f"Average Test Accuracy: {np.mean(fold_test_accuracies):.4f} (±{np.std(fold_test_accuracies):.4f})")

    # Return the model from the fold with the best test accuracy
    best_fold = np.argmax(fold_test_accuracies)
    best_model = best_models[best_fold]
    print(f"\nBest model from fold {best_fold + 1} with Test Accuracy: {fold_test_accuracies[best_fold]:.4f}")

    # Save the best model overall in the specified directory
    best_model_path = os.path.join(save_dir, "best_freq_model_overall.ckpt")
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved at: {best_model_path}")

    return best_model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create models directory
    os.makedirs("../models", exist_ok=True)

    # Load and prepare dataset
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data"
    freq_dataset = VibrationFreqDataset(data_directory, n_fft=512)
    print("Dataset created with frequency domain transformation")
    print(f"Dataset size: {len(freq_dataset)}")

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(freq_dataset.labels, freq_dataset.operations)]

    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(freq_dataset)), test_size=0.3, stratify=stratify_key, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[stratify_key[i] for i in temp_idx], random_state=42
    )

    # Create Subset datasets
    train_dataset = Subset(freq_dataset, train_idx)
    val_dataset = Subset(freq_dataset, val_idx)
    test_dataset = Subset(freq_dataset, test_idx)

    # Verify split sizes and label distribution
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(
        f"Train good: {sum(freq_dataset.labels[train_idx] == 0)}, Train bad: {sum(freq_dataset.labels[train_idx] == 1)}")
    print(f"Val good: {sum(freq_dataset.labels[val_idx] == 0)}, Val bad: {sum(freq_dataset.labels[val_idx] == 1)}")
    print(f"Test good: {sum(freq_dataset.labels[test_idx] == 0)}, Test bad: {sum(freq_dataset.labels[test_idx] == 1)}")

    # Class ratios
    train_ratio = sum(freq_dataset.labels[train_idx] == 0) / sum(freq_dataset.labels[train_idx] == 1)
    val_ratio = sum(freq_dataset.labels[val_idx] == 0) / sum(freq_dataset.labels[val_idx] == 1)
    test_ratio = sum(freq_dataset.labels[test_idx] == 0) / sum(freq_dataset.labels[test_idx] == 1)
    print(f"Class ratio (good/bad) - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")

    # Operation distribution
    train_ops = Counter(freq_dataset.operations[train_idx])
    val_ops = Counter(freq_dataset.operations[val_idx])
    test_ops = Counter(freq_dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train and evaluate model
    freq_model = train_and_evaluate_freq(train_loader, val_loader, test_loader, EarlyStopping=True)

    # Option 2: Cross-validation approach (uncomment to use)
    # best_model = train_and_evaluate_with_cross_validation(freq_dataset, n_splits=5)