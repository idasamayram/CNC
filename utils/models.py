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


# 1️⃣ CNN Model for 1D data old version
class CNN1D_ov(nn.Module):
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



# ------------------------
# 2️⃣  Define the CNN Model with flattened linear layer which is substituted by global average pooling in CNN1D_DS
# ------------------------
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 250, 256)  # Flattened size: (64, 250) down-sampled to 400Hz,
         # for original signal without down-sampling this would be:  64*1250
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Binary classification (good/bad)

        self.dropout = nn.Dropout(0.3)  # Reduce overfitting
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (we use CrossEntropyLoss)

        return x



# ------------------------
# 3️⃣ Define the CNN Model for down-sampled data with global average pooling instead of flattening
# ------------------------
class CNN1D_DS(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=9, stride=1)
        self.gn1 = nn.GroupNorm(4, 16)  # GroupNorm replaces BatchNorm
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1)
        self.gn2 = nn.GroupNorm(4, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1)
        self.gn3 = nn.GroupNorm(4, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        #changed this part compare to cnn1D_torch_update_2 which flattened the layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)  # Increased dropout to reduce overfitting
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

