import os
import h5py
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Define the input shape
input_shape = (None, 3)  # (num_timesteps, num_features)

# Define the input layer
inputs = keras.Input(shape=input_shape)

# Define the model
x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)  # Added another convolutional layer

# Add global average pooling to handle varying sequence lengths
x = layers.GlobalAveragePooling1D()(x)

# Add a fully connected layer
x = layers.Dense(64, activation='relu')(x)

# Add the output layer
outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification

# Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

good_data_dir = Path("../data/balanced_data/good_data")
bad_data_dir = Path("../data/balanced_data/bad_data")

good_files = [file for file in good_data_dir.iterdir() if file.is_file() and file.suffix == '.h5']
bad_files = [file for file in bad_data_dir.iterdir() if file.is_file() and file.suffix == '.h5']

X_good = []
X_bad = []


for file in good_files:
    with h5py.File(file, 'r') as f:
        data = f['vibration_data'][:]  # Assuming the data is stored in a dataset named 'data'
        X_good.append(data)

for file in bad_files:
    with h5py.File(file, 'r') as f:
        data = f['vibration_data'][:]
        X_bad.append(data)


# Create labels
y_good = np.ones(len(X_good), dtype=int)  # Label 1 for good samples
y_bad = np.zeros(len(X_bad), dtype=int)   # Label 0 for bad samples

# Combine the data and labels
X = X_good + X_bad
y = np.concatenate((y_good, y_bad), axis=0)


# Shuffle the combined data
data = list(zip(X, y))
np.random.shuffle(data)
X, y = zip(*data)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)





# Pad the sequences to the maximum length
max_length = max(len(x) for x in X_train + X_val)
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
X_val_padded = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_length, padding='post', dtype='float32')

# Train the model
steps_per_epoch = len(X_train_padded) // 16
validation_steps = len(X_val_padded) // 16

history = model.fit(X_train_padded, np.array(y_train), epochs=10, batch_size=16,
                    validation_data=(X_val_padded, np.array(y_val)),
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    verbose=1)

# Print the training and validation accuracy/loss
print("Training Accuracy:", history.history['accuracy'][-1])
print("Training Loss:", history.history['loss'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])
print("Validation Loss:", history.history['val_loss'][-1])
