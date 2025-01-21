import os
import h5py
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tsfresh



# Load vibration data and labels
def load_data():
    X_data = []
    y_data = []
    machines = ["M01", "M02", "M03"]
    process_names = ["OP00", "OP01", "OP02", "OP03", "OP04", "OP05", "OP06", "OP07", "OP08", "OP09", "OP10", "OP11",
                     "OP12", "OP13", "OP14"]
    labels = ["good", "bad"]
    path_to_dataset = "./data/"
    for process_name in process_names:
        for machine in machines:
            for label in labels:
                data_path = os.path.join(path_to_dataset, machine, process_name, label)
                data_list, data_label = data_loader_utils.load_tool_research_data(data_path, label=label)
                # Assuming data_list and data_label are NumPy arrays or Pandas DataFrames
                X_data.extend(data_list)
                y_data.extend(data_label)
    return X_data, y_data


def apply_windowing(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size, step_size):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)



# Feature extraction
def extract_features(data, window_size):
    features = []

    # Basic statistical features
    mean_x = np.mean(data[:, 0])
    std_y = np.std(data[:, 1])
    # ... Calculate more features
    features.extend([mean_x, std_y])

    # Frequency features using FFT
    freqs, fft_values = np.fft.fft(data[:, 0])  # Calculate for each axis
    max_freq = np.argmax(fft_values)
    # ... Compute more frequency-based features
    features.append(max_freq)

    # Features with TSFRESH
    df = pd.DataFrame(data)  # Assuming data is already a NumPy array or DataFrame with 3 axes
    extracted_features = tsfresh.extract_features(df, column_id='id', column_sort='time')
    features.extend(extracted_features.values.flatten())  # Convert to 1D-array

    return np.array(features)





# Align all samples to a defined time window length
def align_samples(all_data, desired_length=2000, method='truncate'):
    processed_data = []
    for sample in all_data:
        if method == 'truncate':
            processed_data.append(sample[:desired_length])
        elif method == 'pad':
            padding_needed = desired_length - len(sample)
            padding = np.zeros((padding_needed, 3))  # Assuming 3 axes
            processed_data.append(np.concatenate((sample, padding)))
        else:
            raise ValueError("Invalid method. Choose 'truncate' or 'pad'")
    return processed_data





# Train-test split and machine learning models
def train_model(X, y):
    # Handle class imbalance using SMOTE
    oversampler = SMOTE()
    X, y = oversampler.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Standardize data with StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X


# --- MAIN EXECUTION ---
X_data, y_data = load_data()

# Align samples
X_data = align_samples(X_data)

# Apply windowing (if desired)
window_size = 500  # Example window size, adjust as needed
step_size = 250  # Example step size
X_data = apply_windowing(X_data, window_size, step_size)

# Feature extraction
features = []
for window in X_data:
    window_features = extract_features(window, window_size)
    features.append(window_features)
X = np.array(features)

# Machine learning pipeline
train_model(X, y)
