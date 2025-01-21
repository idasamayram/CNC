import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []

        # Collect all file paths and labels
        for machine in os.listdir(root_dir):
            machine_path = os.path.join(root_dir, machine)
            for operation in os.listdir(machine_path):
                operation_path = os.path.join(machine_path, operation)
                for label in ['good', 'bad']:
                    label_path = os.path.join(operation_path, label)
                    for file_name in os.listdir(label_path):
                        if file_name.endswith('.h5'):
                            self.file_paths.append(os.path.join(label_path, file_name))
                            self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        with h5py.File(file_path, 'r') as f:
            data = f['vibration_data'][:]

        # Convert data shape to (1, 3, 10000) where 1 is batch size, 3 is channels, 10000 is sequence length
        data = torch.tensor(data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        label = torch.tensor(1 if label == 'good' else 0, dtype=torch.long)

        return data, label


# Define the data directory
data_dir = "../data/other_ways/windows_selected/"

# Create dataset and dataloaders
dataset = TimeSeriesDataset(data_dir)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)



class TimeSeriesCNN(nn.Module):
    def __init__(self):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 2500, 128)  # Adjust according to your data dimensions
        self.fc2 = nn.Linear(128, 2)  # Output classes

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging line
        x = self.pool(F.relu(self.conv1(x)))
        print(f"After conv1 and pool: {x.shape}")  # Debugging line
        x = self.pool(F.relu(self.conv2(x)))
        print(f"After conv2 and pool: {x.shape}")  # Debugging line
        x = x.view(-1, 32 * 2500)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = TimeSeriesCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# Train the model
train(model, train_loader, criterion, optimizer)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

# Create test_loader (similarly to train_loader)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model
# evaluate(model, test_loader)
