import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split

# Updated CNN model with Conv2D
class CNN2DTimeSeries(nn.Module):
    def __init__(self):
        super(CNN2DTimeSeries, self).__init__()
        # Conv2D layer with kernel size 100x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(100, 3))
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * ((24000 - 100 + 1) // 2), 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Custom Dataset class with reshaping of data
class CustomDataset(Dataset):
    def __init__(self, file_paths, max_length):
        self.file_paths = file_paths
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as f:
            vibration_data = f['vibration_data'][:]
            label = 1 if 'good' in self.file_paths[idx].name else 0            # Pad sequences to max_length
            vibration_data = self.pad_sequence(vibration_data, self.max_length)
            # Reshape to (height, width, channels) -> (24000, 3) and add channel dimension
            vibration_data = vibration_data.reshape((self.max_length, 3))
        return torch.FloatTensor(vibration_data), torch.tensor(label)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        return sequence

# Data preparation
# Data preparation
data_root = Path("../data/")
good_file_paths = list((data_root / 'balanced_data_tr2' / 'good_data_tr2').glob('*.h5'))
bad_file_paths = list((data_root / 'balanced_data_tr2' / 'bad_data_tr2').glob('*.h5'))

file_paths = good_file_paths + bad_file_paths

np.random.shuffle(file_paths)
max_length = max([h5py.File(fp, 'r')['vibration_data'].shape[0] for fp in file_paths])

train_file_paths, val_file_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_file_paths, max_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_file_paths, max_length)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model initialization and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2DTimeSeries().to(device)
criterion = nn.BCELoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Reshape inputs to [batch_size, channels, height, width]
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            val_corrects += torch.sum(preds.squeeze() == labels.float()).item()
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects / len(val_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")



