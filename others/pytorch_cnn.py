import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from pathlib import Path
import random


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# Load data function
import torch.nn.functional as F


# Load data function
def load_data(file_paths):
    data = []
    labels = []
    max_length = 0
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            vibration_data = f['vibration_data'][:]
            label = 1 if 'good_data' in file_path.parts else 0
            data.append(vibration_data)
            labels.append(label)
            max_length = max(max_length, len(vibration_data))

    # Pad sequences to the maximum length
    data_padded = []
    for seq in data:
        pad_length = max_length - len(seq)
        padded_seq = np.pad(seq, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        data_padded.append(padded_seq)

    return np.array(data_padded), np.array(labels), max_length


# Load data
data_root = Path("../data/")
good_file_paths = list((data_root / 'balanced_data' / 'good_data').glob('*.h5'))
bad_file_paths = list((data_root / 'balanced_data' / 'bad_data').glob('*.h5'))
file_paths = good_file_paths + bad_file_paths
random.shuffle(file_paths)

X, y, max_length = load_data(file_paths)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Define model, loss function, and optimizer
input_shape = (X_train_tensor.shape[1], X_train_tensor.shape[2])
model = CNNModel(input_shape)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i + batch_size]
        targets = y_train_tensor[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 2, 1))  # PyTorch Conv1d expects channels first
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor.permute(0, 2, 1))
        val_loss = criterion(outputs.squeeze(), y_val_tensor)
        val_accuracy = ((outputs.squeeze() > 0.5).float() == y_val_tensor).float().mean()

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy.item():.4f}')
