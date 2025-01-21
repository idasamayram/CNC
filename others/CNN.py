import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import numpy as np
import random
import pandas as pd


class CNCTimeSeriesDataset(Dataset):
    def __init__(self, data_info, root_dir):
        self.data_info = data_info
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        machine, operation, label, file_name = self.data_info[idx]
        file_path = os.path.join(self.root_dir, machine, operation, label, file_name)

        with h5py.File(file_path, 'r') as file:
            vibration_data = file['vibration_data'][:]

        # Normalize data (optional)
        vibration_data = (vibration_data - np.mean(vibration_data, axis=0)) / np.std(vibration_data, axis=0)

        label = 0 if label == 'good' else 1
        return torch.tensor(vibration_data.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x


# Define a function to balance the dataset

data_root = "./data"
balanced_data = "./data/balanced_data"

dataset = CNCTimeSeriesDataset(balanced_data, data_root)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
