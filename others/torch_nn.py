import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split


class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # Transpose input to [batch_size, in_channels, sequence_length]
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.global_pooling(x).squeeze()
        x = torch.sigmoid(self.fc(x))
        return x



# Example input tensor with shape (batch_size, channels, height, width)
# (e.g., batch_size=1, channels=1, height=20000, width=3)
example_input = torch.randn(1, 1, 20000, 3)

# Initialize model and move to GPU if available
model = CNNModel()

# Forward pass
output = model(example_input)
print(output)


class CustomDataset(Dataset):
    def __init__(self, file_paths, max_length):
        self.file_paths = file_paths
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as f:
            vibration_data = f['vibration_data'][:]
            label = 1 if 'good_data' in self.file_paths[idx].parts else 0
            # Pad sequences to max_length
            vibration_data = self.pad_sequence(vibration_data, self.max_length)
        return torch.FloatTensor(vibration_data), torch.tensor(label)

    def pad_sequence(self, sequence, max_length):
        if len(sequence) < max_length:
            padding = np.zeros((max_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        return sequence

data_root = Path("../data/")
good_file_paths = list((data_root / 'balanced_data' / 'good_data').glob('*.h5'))
bad_file_paths = list((data_root / 'balanced_data' / 'bad_data').glob('*.h5'))
file_paths = good_file_paths + bad_file_paths

# Shuffle file paths before splitting data
np.random.shuffle(file_paths)

# Calculate max length
max_length = max([h5py.File(fp, 'r')['vibration_data'].shape[0] for fp in file_paths])

# Split data into train and validation sets
train_file_paths, val_file_paths = train_test_split(file_paths, test_size=0.2, random_state=42)

train_dataset = CustomDataset(train_file_paths, max_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_file_paths, max_length)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel((1, train_dataset[0][0].shape[1])).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))  # Add channel dimension
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
            outputs = model(inputs.unsqueeze(1))  # Add channel dimension
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            val_corrects += torch.sum(preds.squeeze() == labels.float()).item()
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects / len(val_dataset)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
