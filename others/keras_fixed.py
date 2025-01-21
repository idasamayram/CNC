import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense
import numpy as np
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path


def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_data(file_paths, max_length):
    data = []
    labels = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            vibration_data = f['vibration_data'][:]
            label = 1 if 'good_data_subset' in file_path.parts else 0
            data.append(vibration_data)
            labels.append(label)

    data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')
    return np.array(data_padded), np.array(labels)


data_root = Path("../data/")
good_file_paths = list((data_root / 'balanced_data_subset' / 'good_data_subset').glob('*.h5'))
bad_file_paths = list((data_root / 'balanced_data_subset' / 'bad_data_subset').glob('*.h5'))
file_paths = good_file_paths + bad_file_paths

max_length = 15000
X, y = load_data(file_paths, max_length)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

input_shape = (X.shape[1], X.shape[2])
model = create_model(input_shape)
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
