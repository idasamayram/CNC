import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import random

class VibrationsDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.labels = []
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as f:
                vibration_data = f['vibration_data'][:]
                label = 1 if 'good_data' in file_path.parts else 0
                self.data.append(vibration_data)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.labels[idx]

class CNN(nn.Module):
    def __init__(self, input_shape):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_shape[1], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for data, labels in train_loader:
            data_lengths = [len(seq) for seq in data]
            data = pad_sequence(data, batch_first=True).to(device)
            labels = labels.to(device)
            data = nn.utils.rnn.pack_padded_sequence(data, data_lengths, batch_first=True, enforce_sorted=False)
            optimizer.zero_grad()
            outputs = model(data)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.batch_sizes.sum().item()
            train_acc += (outputs.squeeze() > 0).eq(labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data_lengths = [len(seq) for seq in data]
                data = pad_sequence(data, batch_first=True).to(device)
                labels = labels.to(device)
                data = nn.utils.rnn.pack_padded_sequence(data, data_lengths, batch_first=True, enforce_sorted=False)
                outputs = model(data)
                outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item() * data.batch_sizes.sum().item()
                val_acc += (outputs.squeeze() > 0).eq(labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
if __name__ == '__main__':
    data_root = Path("../data/")
    good_file_paths = list((data_root / 'balanced_data' / 'good_data').glob('*.h5'))
    bad_file_paths = list((data_root / 'balanced_data' / 'bad_data').glob('*.h5'))
    file_paths = good_file_paths + bad_file_paths
    random.shuffle(file_paths)

    dataset = VibrationsDataset(file_paths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = max(len(data) for data, _ in dataset)
    input_shape = (max_length, 3)  # (sequence_length, num_features)

    model = CNN(input_shape).to(device)
    num_epochs = 10
    train(model, train_loader, val_loader, num_epochs, device)