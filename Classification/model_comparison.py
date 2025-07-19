# model_comparison.py
import os
import numpy as np
import torch
import h5py
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import warnings
import joblib

# Import model training functions
from Classification.classifier.svm_classifier import train_svm_model
from Classification.classifier.random_forest_classifier import train_random_forest_model
from Classification.classifier.gradient_boosting_classifier import train_gradient_boosting_model
from Classification.classifier.mlp_classifier import train_mlp_sklearn, train_mlp_pytorch
from Classification.classifier.cnn1d_classifier import train_cnn1d_model
from Classification.classifier.cnn1d_freq_classifier import train_cnn1d_freq_model
from Classification.classifier.tcn_classifier import train_tcn_model

# Import visualization utilities
from visualization.visualization_utils import *

# Suppress warnings
warnings.filterwarnings('ignore')


class VibrationDataset(Dataset):
    '''
    Dataset class for vibration data with operation information for stratification
    '''

    def __init__(self, data_dir, augment_bad=False, use_fft=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []
        self.augment_bad = augment_bad
        self.file_groups = []  # e.g., 'M01_Feb_2019_OP02_000'
        self.use_fft = use_fft  # Whether to return FFT features instead of time domain signals

        for label, label_idx in zip(["good", "bad"], [0, 1]):  # 0=good, 1=bad
            folder = self.data_dir / label
            for file_name in folder.glob("*.h5"):
                self.file_paths.append(file_name)
                self.labels.append(label_idx)
                # Extract operation (e.g., 'OP02' from 'M01_Feb_2019_OP02_000_window_0.h5')
                operation = file_name.stem.split('_')[3]
                self.operations.append(operation)
                # Extract file group (e.g., 'M01_Feb_2019_OP02_000')
                file_group = file_name.stem.rsplit('_window_', 1)[0]
                self.file_groups.append(file_group)

        self.labels = np.array(self.labels)
        self.operations = np.array(self.operations)
        self.file_groups = np.array(self.file_groups)
        print(f"Dataset initialized with {len(self.file_paths)} files")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN
        label = self.labels[idx]

        # Augment bad samples by adding noise if required
        if self.augment_bad and label == 1:
            data += np.random.normal(0, 0.01, data.shape)  # Add Gaussian noise

        # Convert to frequency domain if needed
        if self.use_fft:
            # Compute magnitude spectrum using FFT for each axis
            fft_data = np.abs(np.fft.rfft(data, axis=1))
            # Log scale to reduce dynamic range
            fft_data = np.log1p(fft_data)  # log(1+x) to handle zeros
            data = fft_data  # Replace time domain signal with FFT

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Feature extraction function
def extract_features(X):
    """Extract time and frequency domain features from raw vibration signals"""
    features = []
    for sample in X:
        sample_features = []
        # For each axis (X, Y, Z)
        for axis in range(sample.shape[0]):
            signal = sample[axis]

            # Time-domain features
            mean = np.mean(signal)
            std = np.std(signal)
            rms = np.sqrt(np.mean(signal ** 2))
            peak = np.max(np.abs(signal))
            skewness = np.mean((signal - mean) ** 3) / (std ** 3) if std > 0 else 0
            kurtosis = np.mean((signal - mean) ** 4) / (std ** 4) if std > 0 else 0
            crest_factor = peak / rms if rms > 0 else 0

            # Add 5 percentiles
            percentiles = np.percentile(signal, [10, 25, 50, 75, 90])

            # Frequency-domain features
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freq = np.fft.rfftfreq(len(signal), d=1.0 / 400)  # Assuming 400Hz sampling rate

            # Mean frequency, spectral centroid
            spectral_centroid = np.sum(fft_freq * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0

            # Energy in specific frequency bands (e.g., 0-50Hz, 50-100Hz, 100-200Hz)
            bands = [(0, 50), (50, 100), (100, 200)]
            band_energies = []
            for low, high in bands:
                mask = (fft_freq >= low) & (fft_freq <= high)
                band_energies.append(np.sum(fft_vals[mask] ** 2))

            # Combine all features
            axis_features = [mean, std, rms, peak, skewness, kurtosis, crest_factor,
                             spectral_centroid] + list(percentiles) + band_energies

            sample_features.extend(axis_features)

        features.append(sample_features)

    return np.array(features)


# Helper function to convert dataset to numpy arrays with features
def process_dataset(dataset, batch_size=128):
    all_data = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for inputs, labels in loader:
        # Convert to numpy
        inputs = inputs.numpy()
        labels = labels.numpy()

        all_data.append(inputs)
        all_labels.append(labels)

    # Concatenate batches
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)

    return X, y


# Load data and extract features
def load_data_and_extract_features(train_dataset, val_dataset, test_dataset, batch_size=128):
    print("Loading datasets and extracting features...")

    # Process each dataset
    X_train, y_train = process_dataset(train_dataset, batch_size)
    X_val, y_val = process_dataset(val_dataset, batch_size)
    X_test, y_test = process_dataset(test_dataset, batch_size)

    # Extract features
    print(f"Extracting features from {X_train.shape[0]} training samples...")
    X_train_features = extract_features(X_train)
    print(f"Extracting features from {X_val.shape[0]} validation samples...")
    X_val_features = extract_features(X_val)
    print(f"Extracting features from {X_test.shape[0]} test samples...")
    X_test_features = extract_features(X_test)

    print(f"Feature extraction complete. Feature shape: {X_train_features.shape[1]} features")

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_features, X_val_features, X_test_features


def main():
    # Set paths and parameters
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data_lessBAD"
    batch_size = 128

    # Create single results directory
    results_dir = "./results/whole_dataset_comparison"
    os.makedirs(results_dir, exist_ok=True)
    ensure_dir(results_dir)


    # Load dataset
    print(f"Loading dataset from {data_directory}...")
    dataset = VibrationDataset(data_directory, augment_bad=False)

    # Load frequency domain dataset
    dataset_freq = VibrationDataset(data_directory, augment_bad=False, use_fft=True)

    # Create stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=stratify_key
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[stratify_key[i] for i in temp_idx]
    )

    # Create dataset subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # In model_comparison.py, after creating your train/val/test splits:

    # Extract labels
    train_labels = [label for _, label in train_dataset]
    val_labels = [label for _, label in val_dataset]
    test_labels = [label for _, label in test_dataset]

    # Plot label distribution
    label_stats = plot_label_distribution(train_labels, val_labels, test_labels, save_dir=results_dir)

    # You can also print a summary of the distribution
    print("\nLabel distribution summary:")
    print(
        f"Train set: {label_stats['train']['good']} good, {label_stats['train']['bad']} bad (ratio: {label_stats['train']['ratio']:.2f}:1)")
    print(
        f"Validation set: {label_stats['validation']['good']} good, {label_stats['validation']['bad']} bad (ratio: {label_stats['validation']['ratio']:.2f}:1)")
    print(
        f"Test set: {label_stats['test']['good']} good, {label_stats['test']['bad']} bad (ratio: {label_stats['test']['ratio']:.2f}:1)")
    print(
        f"Overall: {label_stats['total']['good']} good, {label_stats['total']['bad']} bad (ratio: {label_stats['total']['ratio']:.2f}:1)")


    # Create dataset subsets for frequency domain
    train_dataset_freq = Subset(dataset_freq, train_idx)
    val_dataset_freq = Subset(dataset_freq, val_idx)
    test_dataset_freq = Subset(dataset_freq, test_idx)

    # Verify split sizes and label distribution
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    print(f"Train good: {sum(dataset.labels[train_idx] == 0)}, Train bad: {sum(dataset.labels[train_idx] == 1)}")
    print(f"Val good: {sum(dataset.labels[val_idx] == 0)}, Val bad: {sum(dataset.labels[val_idx] == 1)}")
    print(f"Test good: {sum(dataset.labels[test_idx] == 0)}, Test bad: {sum(dataset.labels[test_idx] == 1)}")

    # Class ratios
    train_ratio = sum(dataset.labels[train_idx] == 0) / sum(dataset.labels[train_idx] == 1)
    val_ratio = sum(dataset.labels[val_idx] == 0) / sum(dataset.labels[val_idx] == 1)
    test_ratio = sum(dataset.labels[test_idx] == 0) / sum(dataset.labels[test_idx] == 1)
    print(f"Class ratio (good/bad) - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}")

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




    # Create DataLoaders for frequency domain
    train_loader_freq = DataLoader(train_dataset_freq, batch_size=batch_size, shuffle=True)
    val_loader_freq = DataLoader(val_dataset_freq, batch_size=batch_size, shuffle=False)
    test_loader_freq = DataLoader(test_dataset_freq, batch_size=batch_size, shuffle=False)

    # Plot operation distribution
    plot_operation_split_bar(train_ops, val_ops, test_ops, save_dir=results_dir, title="Stratified Distribution Across Operations")

    # Load data and extract features for traditional ML models
    X_train, y_train, X_val, y_val, X_test, y_test, X_train_features, X_val_features, X_test_features = load_data_and_extract_features(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )

    # Store results
    results = {}

    # Flag to control which models to run
    run_models = {
        "svm": True,
        "random_forest": True,
        "gradient_boosting": True,
        "mlp_sklearn": True,
        "mlp_pytorch": True,
        "cnn1d_time": True,
        "cnn1d_freq": True,
        "tcn": True
    }

    # Train SVM model
    if run_models["svm"]:
        print("\n" + "=" * 50)
        print("Training SVM model...")
        print("=" * 50)
        _, svm_metrics = train_svm_model(
            X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, save_dir=results_dir
        )
        results["SVM"] = svm_metrics

    # Train Random Forest model
    if run_models["random_forest"]:
        print("\n" + "=" * 50)
        print("Training Random Forest model...")
        print("=" * 50)
        _, rf_metrics = train_random_forest_model(
            X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, save_dir=results_dir
        )
        results["Random_Forest"] = rf_metrics

    # Train Gradient Boosting model
    if run_models["gradient_boosting"]:
        print("\n" + "=" * 50)
        print("Training Gradient Boosting model...")
        print("=" * 50)
        _, gb_metrics = train_gradient_boosting_model(
            X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, save_dir=results_dir
        )
        results["Gradient_Boosting"] = gb_metrics

    # Train MLP scikit-learn model
    if run_models["mlp_sklearn"]:
        print("\n" + "=" * 50)
        print("Training MLP scikit-learn model...")
        print("=" * 50)
        _, mlp_sklearn_metrics = train_mlp_sklearn(
            X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, save_dir=results_dir
        )
        results["MLP_Sklearn"] = mlp_sklearn_metrics

    # Train MLP PyTorch model
    if run_models["mlp_pytorch"]:
        print("\n" + "=" * 50)
        print("Training MLP PyTorch model...")
        print("=" * 50)
        _, mlp_pytorch_metrics = train_mlp_pytorch(
            X_train_features, y_train, X_val_features, y_val, X_test_features, y_test, save_dir=results_dir,
            hidden_sizes=[128, 64], epochs=100, lr=0.001
        )
        results["MLP_PyTorch"] = mlp_pytorch_metrics

    # Train CNN1D Time Domain model
    if run_models["cnn1d_time"]:
        print("\n" + "=" * 50)
        print("Training CNN1D Time-Domain model...")
        print("=" * 50)
        _, cnn_metrics = train_cnn1d_model(
            train_loader, val_loader, test_loader,
            epochs=30, lr=0.001, weight_decay=1e-4,
            model_type="Time",
            early_stopping=False,
            patience=5,
            use_scheduler=True,
            scheduler_type="cosine", save_dir=results_dir
        )
        results["CNN1D_Time"] = cnn_metrics

    # Train CNN1D Frequency Domain model
    if run_models["cnn1d_freq"]:
        print("\n" + "=" * 50)
        print("Training CNN1D Frequency-Domain model...")
        print("=" * 50)
        _, cnn_freq_metrics = train_cnn1d_freq_model(
            train_loader_freq, val_loader_freq, test_loader_freq,
            epochs=30, lr=0.001, weight_decay=1e-4,
            early_stopping=False,
            patience=5,
            use_scheduler=True,
            scheduler_type="onecycle",
            scheduler_params={"max_lr": 0.01}, save_dir=results_dir
        )
        results["CNN1D_Frequency"] = cnn_freq_metrics

    # Train TCN model
    if run_models["tcn"]:
        print("\n" + "=" * 50)
        print("Training TCN model...")
        print("=" * 50)
        _, tcn_metrics = train_tcn_model(
            train_loader, val_loader, test_loader,
            epochs=50, lr=0.001, weight_decay=1e-4,
            channels=[32, 64, 128, 128],
            kernel_size=5,
            dropout=0.3,
            early_stopping=False,
            patience=7,
            use_scheduler=True,
            scheduler_type="onecycle",
            scheduler_params={"max_lr": 0.005}, save_dir=results_dir
        )
        results["TCN"] = tcn_metrics

    # Create comparison tables
    print("\n" + "=" * 50)
    print("Creating Comparison Tables")
    print("=" * 50)

    # Table 1: Metrics comparison
    print("\nTable 1: Model Performance Metrics")
    create_metrics_table(results, save_dir=results_dir)

    # Table 2: Model parameters
    print("\nTable 2: Model Parameters")
    create_parameters_table(results, save_dir=results_dir)

    # Table 3: Performance metrics
    print("\nTable 3: Detailed Performance Metrics")
    create_performance_table(results, save_dir=results_dir)

    # Plot model comparison
    plot_model_comparison(results, save_dir=results_dir)

    # Generate HTML report
    save_results_as_report(results, os.path.join(results_dir, "model_comparison_report.html"))

    # Summary of best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

    # Save models if required
    save_models = False
    # In model_comparison.py, update the save_models section like this:

    # Save models if required
    save_models = True  # Set to True to save models
    if save_models:
        # Create directory for saved models
        models_path = os.path.join(results_dir, "saved_models")
        os.makedirs(models_path, exist_ok=True)

        # Save deep learning models
        if "CNN1D_Time" in results:
            # Get the model from the results
            if "model" in results["CNN1D_Time"]:
                cnn_model = results["CNN1D_Time"]["model"]
                torch.save(cnn_model.state_dict(), os.path.join(models_path, "cnn1d_time_model.pt"))

        if "CNN1D_Frequency" in results:
            if "model" in results["CNN1D_Frequency"]:
                cnn_freq_model = results["CNN1D_Frequency"]["model"]
                torch.save(cnn_freq_model.state_dict(), os.path.join(models_path, "cnn1d_freq_model.pt"))

        if "TCN" in results:
            if "model" in results["TCN"]:
                tcn_model = results["TCN"]["model"]
                torch.save(tcn_model.state_dict(), os.path.join(models_path, "tcn_model.pt"))

        if "MLP_PyTorch" in results:
            if "model" in results["MLP_PyTorch"]:
                mlp_pytorch_model = results["MLP_PyTorch"]["model"]
                torch.save(mlp_pytorch_model.state_dict(), os.path.join(models_path, "mlp_pytorch_model.pt"))

        # Save traditional ML models (sklearn models)
        for model_name in ["SVM", "Random_Forest", "Gradient_Boosting", "MLP_Sklearn"]:
            if model_name in results and "model" in results[model_name]:
                model_obj = results[model_name]["model"]
                joblib.dump(model_obj, os.path.join(models_path, f"{model_name.lower()}_model.pkl"))

        print(f"\nAll models have been saved to {models_path}")


if __name__ == "__main__":
    main()