import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from utils.models import CNN1D_DS
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from visualization.CNN1D_visualization import *
from utils.dataloader import  stratified_group_split
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, accuracy_score

# ------------------------
# 1️⃣ Custom Dataset Class
# ------------------------

class VibrationDataset(Dataset):
    '''
    This version includes the operation data so that it can be used for stratified
    sampling in the train/val/test split
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
        assert len(self.file_paths) == 6383, f"Expected 7129 files, found {len(self.file_paths)}"  #it was 7501 with 80% overlap of  bad data windows, now it is 50% overlap, so less bad data

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

# ------------------------
# 2️⃣ Define the CNN Model for downsampled data
# ------------------------
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

class CNN1D_DS_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_DS_Wide, self).__init__()
        # Wider kernels with GroupNorm for better receptive field and stable training
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


# ------------------------
# 3️⃣ Train & Evaluate Functions
# ------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
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

    accuracy = correct / total
    return total_loss / len(train_loader), accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    misclassified_indices = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect indices of misclassified samples
            batch_indices = torch.where(predicted != labels)[0]
            for idx in batch_indices:
                global_idx = batch_idx * val_loader.batch_size + idx.item()
                misclassified_indices.append(global_idx)

    val_loss /= len(val_loader)
    val_acc = correct / total
    return val_loss, val_acc, misclassified_indices
# ------------------------
# 4️⃣ Test the Model
# ------------------------
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute F1-score
    f1 = f1_score(all_labels, all_preds, average="weighted")
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()



    return f1, accuracy, all_labels, all_preds
# ------------------------
# 5️⃣ Full Training Pipeline
# ------------------------
def train_and_evaluate(train_loader, val_loader, test_loader, model_class=CNN1D_Wide, epochs=30, lr=0.001, weight_decay=1e-4, EralyStopping=False, Schedule=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Model setup
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)) # Use weighted loss if provided

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    early_stop_epoch = epochs
    patience = 3


    train_recalls_bad, val_recalls_bad = [], []



    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # or use ReduceLROnPlateau scheduler
    # scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.5,  patience=2)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = validate_epoch(model, val_loader, criterion, device)



        # Step the scheduler
        if Schedule:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate

            # or Step the scheduler based on validation loss
            # scheduler.step(val_loss)
            # current_lr = scheduler.get_last_lr()[0]


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)



        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ")
              #f"Learning Rate: {current_lr:.6f}")

        if EralyStopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()  # Save the best model weights
                patience_counter = 0  # Reset counter
            else:
                patience_counter += 1  # Increment counter if no improvement
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    early_stop_epoch = epoch + 1
                    break

    # Restore the best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Restored best model weights from epoch with Val Loss: {best_val_loss:.4f}")


    print("✅ Training and validation complete!")


    # Evaluate on the test set
    # f1, accuracy, all_labels, all_preds = test_model(model, test_loader, device)
    f1, accuracy, y_true, y_pred = test_model(model, test_loader, device)

    print(f"🔥 Test F1 Score: {f1:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    # plot_confusion_matrix(all_labels, all_preds, class_names=["Good", "Bad"], normalize=False)
    plot_confmat_and_metrics(y_true, y_pred, class_names=["Good", "Bad"], title="Confusion Matrix & Key Metrics")

    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    if EralyStopping:
        ax1.plot(range(1, early_stop_epoch + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, early_stop_epoch + 1), val_losses, label="Val Loss")
    else:
        ax1.plot(range(1, epochs + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, epochs + 1), val_losses, label="Val Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    if EralyStopping:
        ax2.plot(range(1, early_stop_epoch + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, early_stop_epoch + 1), val_accuracies, label="Val Accuracy")
    else:
        ax2.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, epochs + 1), val_accuracies, label="Val Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return model

# ------------------------
# 6️⃣ Run Training & Evaluation
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Splitting the dataset with stratified sampling based on operations,labels, and groups
    data_directory = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    train_loader, val_loader, test_loader, _ = stratified_group_split(data_directory)

    best_model = train_and_evaluate(train_loader, val_loader, test_loader, model_class= CNN1D_Wide, EralyStopping=False, Schedule=True, epochs=30, lr=0.001, weight_decay=1e-4)

    # Save the best model
    # Save the trained model

    torch.save(best_model.state_dict(), "../cnn1d_model_new_test_2.ckpt")
    print("✅ Model saved to cnn1d_model.ckpt")
    best_model.to(device)
    best_model.eval()  # Switch to evaluation mode
