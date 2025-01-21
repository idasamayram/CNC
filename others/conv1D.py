import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# Define helper functions
'''def create_overlapping_windows(data, window_size, overlap):
    stride = int(window_size * (1 - overlap))
    num_windows = (len(data) - window_size) // stride + 1
    windows = []
    for i in range(0, num_windows * stride, stride):
        window = data[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)
    return np.array(windows)'''


def load_data_with_labels(base_folder, window_size, overlap):
    all_windows = []
    all_labels = []

    for root, dirs, files in os.walk(base_folder):
        for file_name in files:
            if file_name.endswith('.h5'):
                file_path = os.path.join(root, file_name)
                with h5py.File(file_path, 'r') as f:
                    dataset = np.array(f['vibration_data'])
                    windows = create_overlapping_windows(dataset, window_size, overlap)
                    all_windows.append(windows)

                if 'good' in root:
                    label = 0
                elif 'bad' in root:
                    label = 1
                else:
                    raise ValueError(f"Unknown label for path: {root}")

                labels = np.full(len(windows), label)
                all_labels.append(labels)

    all_windows = np.vstack(all_windows)
    all_labels = np.concatenate(all_labels)
    return all_windows, all_labels


# Define Conv1D model
class VibrationCNN(nn.Module):
    def __init__(self):
        super(VibrationCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (window_size // 4), 128)  # Adjust based on pooling
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x


# Load and prepare data
base_folder = 'data/windows_selected'
window_size = 6000
overlap = 0.2

data_windows, data_labels = load_data_with_labels(base_folder, window_size, overlap)

# Split data into training and validation sets
dataset = TensorDataset(torch.Tensor(data_windows), torch.Tensor(data_labels))
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = VibrationCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and evaluation loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        labels = labels.unsqueeze(1)  # Make sure labels are shaped correctly for BCELoss

        optimizer.zero_grad()
        outputs = model(inputs.transpose(1, 2))  # Conv1D expects (batch, channels, length)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.unsqueeze(1)
            outputs = model(inputs.transpose(1, 2))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    val_acc = correct_val / total_val

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

print("Training complete")
