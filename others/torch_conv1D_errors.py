import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from pathlib import Path
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold



# PyTorch CNN model definition
class CNN1D(nn.Module):
    def __init__(self, input_shape):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=200)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=200)
        self.bn2 = nn.BatchNorm1d(128)  # Batch Normalization
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))  # BatchNorm + ReLU
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))  # BatchNorm + ReLU
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Function to load data from HDF5 files
def load_data(file_paths):
    data = []
    labels = []
    max_length = 0
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            vibration_data = f['vibration_data'][:]
            label = 1 if 'good' in file_path.parts else 0
            data.append(vibration_data)
            labels.append(label)
            max_length = max(max_length, len(vibration_data))

    data_padded = np.array([np.pad(d, ((0, max_length - len(d)), (0, 0)), 'constant') for d in data], dtype='float32')
    return np.array(data_padded), np.array(labels), max_length

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_labels = []
        train_predictions = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for F1-score calculation
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Compute F1-score for training
        train_f1 = f1_score(train_labels, train_predictions, average='weighted')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Collect labels and predictions for F1-score calculation
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total

        # Compute F1-score for validation
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')

        # Print training and validation metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, F1-Score: {train_f1:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc:.2f}%, Val F1-Score: {val_f1:.4f}')


# Function to evaluate the model on validation data
def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for F1-score calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Compute average loss and accuracy
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    # Compute F1-score
    f1 = f1_score(all_labels, all_predictions, average='weighted')  # Weighted for imbalanced datasets

    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1-Score: {f1:.4f}')
    return val_loss, val_acc, f1



# Load data
data_root = Path("../data/final")
good_file_paths = list((data_root / 'Selected_data_windowed_grouped' / 'good').glob('*.h5'))
bad_file_paths = list((data_root / 'Selected_data_windowed_grouped' / 'bad').glob('*.h5'))
file_paths = good_file_paths + bad_file_paths

random.shuffle(file_paths)

X, y, max_length = load_data(file_paths)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train).permute(0, 2, 1)  # PyTorch expects (batch_size, channels, sequence_length)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val).permute(0, 2, 1)
y_val = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader for batching
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model initialization
input_shape = (batch_size, X_train.shape[1], max_length)
print(input_shape[1])
model = CNN1D(input_shape).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Train the model
# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)

# Evaluate the model on validation data
evaluate_model(model, val_loader, criterion)
