import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
import time
import seaborn as sns


# Reuse  existing VibrationDataset class
class VibrationDataset(Dataset):
    '''
    This version includes the operation data so that it can be used for stratified
    sampling in the train/val/test split.
    '''
    def __init__(self, data_dir, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []  # Optional for operation-based stratification
        self.augment_bad = augment_bad
        self.file_groups = []  # e.g., 'M01_Feb_2019_OP02_000'

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

        # Augment bad samples by adding noise
        if self.augment_bad and label == 1:
            data += np.random.normal(0, 0.01, data.shape)  # Add Gaussian noise

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Feature extraction functions
def extract_features(X):
    """
    Extract features from raw vibration signals (3, 2000) -> feature vector
    Features: mean, std, min, max, rms, kurtosis, skewness, etc. for each axis
    """
    features = []
    for sample in X:
        sample_features = []
        # For each axis (X, Y, Z)
        for axis in range(sample.shape[0]):
            signal = sample[axis]
            
            # Time-domain features
            mean = np.mean(signal)
            std = np.std(signal)
            rms = np.sqrt(np.mean(signal**2))
            peak = np.max(np.abs(signal))
            skewness = np.mean((signal - mean)**3) / (std**3) if std > 0 else 0
            kurtosis = np.mean((signal - mean)**4) / (std**4) if std > 0 else 0
            crest_factor = peak / rms if rms > 0 else 0
            
            # Add 5 percentiles
            percentiles = np.percentile(signal, [10, 25, 50, 75, 90])
            
            # Frequency-domain features
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freq = np.fft.rfftfreq(len(signal), d=1.0/400)  # Assuming 400Hz sampling rate
            
            # Mean frequency, spectral centroid
            spectral_centroid = np.sum(fft_freq * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0
            
            # Energy in specific frequency bands (e.g., 0-50Hz, 50-100Hz, 100-200Hz)
            bands = [(0, 50), (50, 100), (100, 200)]
            band_energies = []
            for low, high in bands:
                mask = (fft_freq >= low) & (fft_freq <= high)
                band_energies.append(np.sum(fft_vals[mask]**2))
                
            # Combine all features
            axis_features = [mean, std, rms, peak, skewness, kurtosis, crest_factor, 
                            spectral_centroid] + list(percentiles) + band_energies
            
            sample_features.extend(axis_features)
        
        features.append(sample_features)
    
    return np.array(features)


def load_data_and_extract_features(train_dataset, val_dataset, test_dataset):
    """
    Load data from datasets and extract features
    """
    # Function to convert dataset to numpy arrays with features
    def process_dataset(dataset):
        all_data = []
        all_labels = []
        
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        for inputs, labels in loader:
            # Convert to numpy
            inputs = inputs.numpy()
            labels = labels.numpy()
            
            all_data.append(inputs)
            all_labels.append(labels)
        
        # Concatenate batches
        X = np.vstack(all_data)
        y = np.concatenate(all_labels)
        
        # Extract features
        print(f"Extracting features from {X.shape[0]} samples...")
        X_features = extract_features(X)
        print(f"Feature extraction complete. Feature shape: {X_features.shape}")
        
        return X_features, y
    
    # Process each dataset
    X_train, y_train = process_dataset(train_dataset)
    X_val, y_val = process_dataset(val_dataset)
    X_test, y_test = process_dataset(test_dataset)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_confmat_and_metrics(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("Only works for binary classification (2 classes).")
    
    TN, FP, FN, TP = cm.ravel()
    
    # Metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    metrics = [
        ["Accuracy", f"{accuracy:.3f}"],
        ["Precision", f"{precision:.3f}"],
        ["Recall (Sensitivity)", f"{recall:.3f}"],
        ["Specificity", f"{specificity:.3f}"],
        ["F1-score", f"{f1:.3f}"],
        ["TP", TP],
        ["FP", FP],
        ["TN", TN],
        ["FN", FN],
    ]

    # Plot
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.9])

    # Confusion matrix
    ax0 = fig.add_subplot(gs[0])
    if class_names is None:
        class_names = ["Good", "Bad"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax0,
                annot_kws={'size': 18})
    ax0.set_xlabel("Predicted label")
    ax0.set_ylabel("True label")
    ax0.set_title(title, fontsize=16)

    # Table
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    table = ax1.table(
        cellText=metrics,
        colLabels=["Metric", "Value"],
        loc='center',
        cellLoc='center',
        colLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(14)
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#cccccc")
        else:
            cell.set_facecolor("#f9f9f9" if row % 2 == 0 else "#e6e6e6")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model_name, train_sizes, train_scores, val_scores):
    """
    Plot learning curve from cross-validation results
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Validation score', color='red', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    plt.title(f'Learning Curve - {model_name}', fontsize=16)
    plt.xlabel('Training Set Size', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    plt.show()


def train_and_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate SVM model with parameter tuning
    """
    print("Training SVM model...")
    start_time = time.time()
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True))
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__kernel': ['rbf', 'linear']
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=3, n_jobs=-1, verbose=1, 
        scoring='f1'
    )
    
    # Combine train and validation for grid search
    X_grid = np.vstack((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    
    grid_search.fit(X_grid, y_grid)
    best_pipeline = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_val_pred = best_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], title="SVM Confusion Matrix")
    
    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid, 
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plot_learning_curve("SVM", train_sizes_abs, train_scores, val_scores)
    
    return best_pipeline, test_accuracy, test_f1


def train_and_evaluate_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate Random Forest model with parameter tuning
    """
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=3, n_jobs=-1, verbose=1, 
        scoring='f1'
    )
    
    # Combine train and validation for grid search
    X_grid = np.vstack((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    
    grid_search.fit(X_grid, y_grid)
    best_pipeline = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_val_pred = best_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], title="Random Forest Confusion Matrix")
    
    # Plot feature importances
    rf = best_pipeline.named_steps['rf']
    feature_importances = rf.feature_importances_
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title('Random Forest Feature Importances', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid, 
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plot_learning_curve("Random Forest", train_sizes_abs, train_scores, val_scores)
    
    return best_pipeline, test_accuracy, test_f1


def train_and_evaluate_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate Gradient Boosting model with parameter tuning
    """
    print("Training Gradient Boosting model...")
    start_time = time.time()
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2],
        'gb__max_depth': [3, 4, 5]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=3, n_jobs=-1, verbose=1, 
        scoring='f1'
    )
    
    # Combine train and validation for grid search
    X_grid = np.vstack((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    
    grid_search.fit(X_grid, y_grid)
    best_pipeline = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_val_pred = best_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], title="Gradient Boosting Confusion Matrix")
    
    # Plot feature importances
    gb = best_pipeline.named_steps['gb']
    feature_importances = gb.feature_importances_
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title('Gradient Boosting Feature Importances', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid, 
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plot_learning_curve("Gradient Boosting", train_sizes_abs, train_scores, val_scores)
    
    return best_pipeline, test_accuracy, test_f1


def train_and_evaluate_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate Neural Network (MLP) model with parameter tuning
    """
    print("Training Neural Network (MLP) model...")
    start_time = time.time()
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, max_iter=500))
    ])
    
    # Parameter grid for optimization
    param_grid = {
        'mlp__hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, 
        cv=3, n_jobs=-1, verbose=1, 
        scoring='f1'
    )
    
    # Combine train and validation for grid search
    X_grid = np.vstack((X_train, X_val))
    y_grid = np.concatenate((y_train, y_val))
    
    grid_search.fit(X_grid, y_grid)
    best_pipeline = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_val_pred = best_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Validation - Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], title="Neural Network (MLP) Confusion Matrix")
    
    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid, 
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plot_learning_curve("MLP", train_sizes_abs, train_scores, val_scores)
    
    return best_pipeline, test_accuracy, test_f1


def compare_all_models(results):
    """
    Compare all models using bar charts
    """
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    # Add labels and titles
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


# Import necessary function for learning curve plot
from sklearn.model_selection import learning_curve

if __name__ == "__main__":
    # Paths and configurations
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data_lessBAD"
    
    # Load the dataset
    dataset = VibrationDataset(data_directory)
    
    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]
    
    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=0.3, stratify=stratify_key
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[stratify_key[i] for i in temp_idx]
    )
    
    # Create Subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
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
    
    # Load data and extract features
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_and_extract_features(
        train_dataset, val_dataset, test_dataset
    )
    
    # Store results to compare later
    results = {}
    
    # Train and evaluate SVM
    svm_model, svm_accuracy, svm_f1 = train_and_evaluate_svm(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['SVM'] = {'model': svm_model, 'accuracy': svm_accuracy, 'f1': svm_f1}
    
    # Train and evaluate Random Forest
    rf_model, rf_accuracy, rf_f1 = train_and_evaluate_random_forest(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['Random Forest'] = {'model': rf_model, 'accuracy': rf_accuracy, 'f1': rf_f1}
    
    # Train and evaluate Gradient Boosting
    gb_model, gb_accuracy, gb_f1 = train_and_evaluate_gradient_boosting(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['Gradient Boosting'] = {'model': gb_model, 'accuracy': gb_accuracy, 'f1': gb_f1}
    
    # Train and evaluate Neural Network (MLP)
    mlp_model, mlp_accuracy, mlp_f1 = train_and_evaluate_mlp(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    results['Neural Network'] = {'model': mlp_model, 'accuracy': mlp_accuracy, 'f1': mlp_f1}
    
    # Compare all models
    compare_all_models(results)
    
    # Save the best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    print(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # You can save the best model here if needed
    import joblib
    joblib.dump(best_model, f"best_{best_model_name.lower().replace(' ', '_')}_model.pkl")
    print(f"Best model saved as best_{best_model_name.lower().replace(' ', '_')}_model.pkl")