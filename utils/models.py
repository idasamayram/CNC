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


