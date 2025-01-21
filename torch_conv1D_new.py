import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# ------------------------
# 1️⃣ Define CNN1D Model
# ------------------------
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 2)  # Binary classification

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ------------------------
# 2️⃣ Load Data
# ------------------------
class VibrationDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.labels = [1 if "good" in str(fp) else 0 for fp in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (10000, 3)
        data = np.transpose(data, (1, 0))  # Convert to (3, 10000)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ------------------------
# 3️⃣ Training Function
# ------------------------
def train_model(model, train_loader, criterion, optimizer, epochs=20, device="cuda"):
    model.to(device)
    for epoch in range(epochs):
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
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f} - Accuracy: {correct/total:.4f}")

    print("✅ Training complete!")

# ------------------------
# 4️⃣ Evaluation Function
# ------------------------
def evaluate_model(model, test_loader, criterion, device="cuda"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"\n🔥 Test F1 Score: {f1:.4f} - Accuracy: {correct/total:.4f}")
    return f1

# ------------------------
# 5️⃣ Run Training & Evaluation
# ------------------------
# Load data paths
data_root = "./data/final/Selected_data_windowed_grouped_normalized"
good_files = list(Path(data_root, "good").glob("*.h5"))
bad_files = list(Path(data_root, "bad").glob("*.h5"))
file_paths = good_files + bad_files

# Split dataset into train & test
train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(VibrationDataset(train_files), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(VibrationDataset(test_files), batch_size=batch_size, shuffle=False)

# Define model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Train and evaluate
train_model(model, train_loader, criterion, optimizer, epochs=20, device=device)
evaluate_model(model, test_loader, criterion, device=device)
