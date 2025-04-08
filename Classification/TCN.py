import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import h5py
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset





# 1️⃣ Custom Dataset Class
# ------------------------
class VibrationDataset(Dataset):
    def __init__(self, data_dir, augment_bad=False):
        self.file_paths = []
        self.labels = []
        self.augment_bad = augment_bad

        for label, label_idx in zip(["good", "bad"], [0, 1]):  # 0=good, 1=bad
            folder = os.path.join(data_dir, label)
            for file_name in os.listdir(folder):
                if file_name.endswith(".h5"):
                    self.file_paths.append(os.path.join(folder, file_name))
                    self.labels.append(label_idx)

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
            data += np.random.normal(0, 0.01, data.shape)  # Add Gaussian noise

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# TCNBlock and TCN Model (Same as Before)
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.5):
        super(TCNBlock, self).__init__()
        # Causal padding for the entire block (two convolutions)
        self.padding = (kernel_size - 1) * dilation  # Padding needed per convolution
        self.total_padding = self.padding * 2  # Total padding for two convolutions

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.gn1 = nn.GroupNorm(4, out_channels)
        self.gn2 = nn.GroupNorm(4, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1x1 convolution for residual connection if in_channels != out_channels
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Save the input length for residual connection
        input_length = x.size(2)

        # First convolution
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Crop the right side to match the input length (after padding)
        out = out[:, :, :input_length]

        # Second convolution
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        # Crop the right side to match the input length (after padding)
        out = out[:, :, :input_length]

        # Residual connection
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, in_channels, num_classes, channels, kernel_size=7, dropout=0.5):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        num_levels = len(channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i-1]
            out_ch = channels[i]
            self.layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Data Augmentation (Same as Before)
class VibrationDataAugmentation:
    def __init__(self, noise_std=0.01, max_shift=50):
        self.noise_std = noise_std
        self.max_shift = max_shift

    def __call__(self, x):
        noise = torch.normal(mean=0.0, std=self.noise_std, size=x.shape)
        x = x + noise
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift > 0:
            x = torch.cat([torch.zeros(x.shape[0], shift), x[:, :-shift]], dim=1)
        elif shift < 0:
            x = torch.cat([x[:, -shift:], torch.zeros(x.shape[0], -shift)], dim=1)
        return x

# Helper Functions: train_epoch, validate_epoch, test_model
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    misclassified = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Track misclassified indices
            batch_misclassified = (predicted != labels).nonzero(as_tuple=True)[0]
            for i in batch_misclassified:
                global_idx = idx * val_loader.batch_size + i.item()
                misclassified.append(global_idx)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, misclassified

def test_model(model, test_loader, device):
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

    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, accuracy

# Train and Evaluate Function (Without K-Fold)
def train_and_evaluate(train_loader, val_loader, test_loader, epochs=20, lr=0.0005, weight_decay=5e-4, save_dir=None, early_stopping_patience=3, early_stopping_delta=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Class weights for imbalanced dataset (66:33 good:bad)
    num_good = 0.66 * (len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset))
    num_bad = 0.33 * (len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset))
    weight_good = 1 / num_good
    weight_bad = 1 / num_bad
    class_weights = torch.tensor([weight_good, weight_bad]).to(device)
    class_weights = class_weights / class_weights.sum() * 2

    # Data augmentation for training set
    augmentation = VibrationDataAugmentation(noise_std=0.01, max_shift=50)

    # Apply augmentation to training data
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            x, y = self.dataset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y

    augmented_train_dataset = AugmentedDataset(train_loader.dataset, transform=augmentation)
    train_loader = DataLoader(augmented_train_dataset, batch_size=train_loader.batch_size, shuffle=True)

    # Initialize the TCN model
    in_channels = 3
    channels = [16, 32, 64]
    num_classes = 2
    model = TCN(in_channels=in_channels, num_classes=num_classes, channels=channels, kernel_size=7, dropout=0.5).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    smoothed_val_loss = None
    alpha = 0.9  # Smoothing factor for EMA
    early_stopping_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        # Validation
        val_loss, val_acc, misclassified = validate_epoch(model, val_loader, criterion, device)

        # Smooth the validation loss for the scheduler
        if smoothed_val_loss is None:
            smoothed_val_loss = val_loss
        else:
            smoothed_val_loss = alpha * smoothed_val_loss + (1 - alpha) * val_loss
        scheduler.step(smoothed_val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Smoothed Val Loss: {smoothed_val_loss:.4f} - "
              f"Learning Rate: {current_lr:.6f}")
        print(f"Misclassified validation indices: {misclassified}")

        # Early stopping
        '''if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            early_stopping_counter = 0
            checkpoint_path = os.path.join(save_dir, "best_model.ckpt")
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))'''

    # Plot training and validation metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    ax1.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    ax2.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate on the test set
    test_f1, test_acc = test_model(model, test_loader, device)
    print(f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the final model
    final_model_path = os.path.join(save_dir, "final_model.ckpt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at: {final_model_path}")

    return model



if __name__ == "__main__":
    # Splitting the dataset
    data_directory = "../data/final/Selected_data_windowed_grouped_normalized_downsampled"


    dataset = VibrationDataset(data_directory)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Creating DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6️⃣ Run Training & Evaluation
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_and_evaluate(train_loader, val_loader, test_loader, epochs=15, lr=0.0005, weight_decay=5e-4, early_stopping_patience=3, early_stopping_delta=0.001)
    torch.save(model.state_dict(), "tcn_model_new.ckpt")
    model.to(device)
    model.eval()  # Switch to evaluation mode
    print("✅ Model loaded and ready for explanations")
