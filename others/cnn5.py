import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
import numpy as np
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pathlib import Path
import random


def create_model(input_shape):
    model = Sequential()
    # 1D Convolutional layer
    model.add(Conv1D(filters=32, kernel_size=100, activation='relu', input_shape=input_shape))
    # Max Pooling layer
    model.add(MaxPooling1D(pool_size=2))
    # Fully connected convolutional layer
    model.add(Conv1D(filters=64, kernel_size=100, activation='relu'))
    # Global average pooling to handle variable input length
    model.add(GlobalAveragePooling1D())
    # Dropout layer
    model.add(Dropout(0.5))
    # Output layer
    model.add(Dense(2, activation='softmax'))  # Assuming binary classification but with softmax

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


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

    data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')
    return np.array(data_padded), np.array(labels), max_length


data_root = Path("../data/")
good_file_paths = list((data_root / 'windows_selected_separate' / 'good').glob('*.h5'))
bad_file_paths = list((data_root / 'windows_selected_separate' / 'bad').glob('*.h5'))
file_paths = good_file_paths + bad_file_paths

random.shuffle(file_paths)

X, y, max_length = load_data(file_paths)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = (max_length, X.shape[2])

model = create_model(input_shape)
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))
