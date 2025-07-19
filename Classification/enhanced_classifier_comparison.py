import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import time
import gc

# Define CNN1D model for comparison
class CNN1D_DS_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_DS_Wide, self).__init__()
        # Wider kernels with GroupNorm for better receptive field and stable training
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.gn1 = nn.GroupNorm(4, 16)  # GroupNorm for better generalization
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.gn2 = nn.GroupNorm(4, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.gn3 = nn.GroupNorm(4, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.3)
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

# Vibration Dataset Class
class VibrationDataset(Dataset):
    '''
    Dataset class for vibration data with operation information for stratification
    '''
    def __init__(self, data_dir, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.operations = []
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

# Plot confusion matrix and metrics
def plot_confmat_and_metrics(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """Plot confusion matrix with metrics table"""
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
    
    return accuracy, precision, recall, specificity, f1

# Plot learning curve
def plot_learning_curve(model_name, train_sizes, train_scores, val_scores):
    """Plot learning curve from cross-validation results"""
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

# Visualize filters from CNN1D model
def visualize_cnn_filters(model):
    """Visualize the filters from the first convolutional layer of a CNN1D model"""
    # Get the weights from the first convolutional layer
    weights = model.conv1.weight.data.cpu().numpy()
    
    # Determine the number of filters and their size
    n_filters, n_channels, filter_size = weights.shape
    
    # Create a figure
    fig, axs = plt.subplots(4, 4, figsize=(12, 10))
    axs = axs.flatten()
    
    # Plot each filter
    for i, ax in enumerate(axs):
        if i < n_filters:
            # Plot each channel of the filter with different colors
            for c in range(n_channels):
                color = ['red', 'green', 'blue'][c]
                ax.plot(weights[i, c], color=color, alpha=0.7, label=f'Channel {c+1}')
            
            ax.set_title(f'Filter {i+1}')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Only show legend on the first plot
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle('CNN1D First Layer Filter Visualization', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

# Feature importance visualization for traditional ML models
def visualize_feature_importance(model, feature_names=None):
    """Visualize feature importances for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances', fontsize=16)
        
        # Limit to top 20 features for clarity
        top_n = min(20, len(importances))
        plt.bar(range(top_n), importances[indices[:top_n]], align='center')
        
        if feature_names is not None:
            feature_names = [feature_names[i] for i in indices[:top_n]]
        else:
            feature_names = [f'Feature {i}' for i in indices[:top_n]]
            
        plt.xticks(range(top_n), feature_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.xlabel('Feature', fontsize=14)
        plt.ylabel('Importance', fontsize=14)
        plt.show()
    else:
        print("This model doesn't expose feature_importances_ attribute")

# CNN Training and Evaluation
def train_cnn1d_model(train_loader, val_loader, test_loader, epochs=30, lr=0.001):
    """Train and evaluate a CNN1D model with learning curve plotting"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN1D model on {device}...")
    
    start_time = time.time()
    
    # Initialize model
    model = CNN1D_DS_Wide().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # For early stopping
    best_val_loss = float('inf')
    best_model_weights = None
    patience = 5
    patience_counter = 0
    
    # For learning curve
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_weights)
    
    # Test evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    
    training_time = time.time() - start_time
    print(f"CNN1D Training Complete - Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(all_labels, all_preds, class_names=["Good", "Bad"], 
                             title="CNN1D Confusion Matrix")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN1D Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN1D Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize CNN filters
    visualize_cnn_filters(model)
    
    return model, test_accuracy, test_f1

# SVM Training and Evaluation
def train_svm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate SVM model with parameter tuning"""
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
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"SVM Training Complete - Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], 
                             title="SVM Confusion Matrix")
    
    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid, 
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    
    plot_learning_curve("SVM", train_sizes_abs, train_scores, val_scores)
    
    return best_pipeline, test_accuracy, test_f1

# Random Forest Training and Evaluation
def train_random_forest_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate Random Forest model with parameter tuning"""
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
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Random Forest Training Complete - Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], 
                             title="Random Forest Confusion Matrix")
    
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

# Gradient Boosting Training and Evaluation
def train_gradient_boosting_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate Gradient Boosting model with parameter tuning"""
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
    
    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    training_time = time.time() - start_time
    print(f"Gradient Boosting Training Complete - Time: {training_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrix
    plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"], 
                             title="Gradient Boosting Confusion Matrix")
    
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

# Model comparison visualization
def compare_models(results):
    """Compare all models using bar charts"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1'] for model in models]
    training_times = [results[model]['training_time'] for model in models] if 'training_time' in results[models[0]] else None
    
    # Plot accuracy and F1 scores
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='#e74c3c')
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Comparison (Test Scores)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
    # Plot training times if available
    if training_times:
        plt.figure(figsize=(10, 5))
        plt.bar(models, training_times, color='#2ecc71')
        plt.title('Training Time Comparison', fontsize=16)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.xticks(fontsize=12)
        
        # Add time labels
        for i, time in enumerate(training_times):
            plt.text(i, time + 0.5, f'{time:.1f}s', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def main():
    # Set paths
    data_directory = "../data/final/new_selection/normalized_windowed_downsampled_data_lessBAD"
    batch_size = 128
    
    # Load dataset
    print(f"Loading dataset from {data_directory}...")
    dataset = VibrationDataset(data_directory, augment_bad=False)
    
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
    
    # Load data and extract features for traditional ML models
    X_train, y_train, X_val, y_val, X_test, y_test, \
    X_train_features, X_val_features, X_test_features = load_data_and_extract_features(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    # Store results
    results = {}
    
    # Train CNN1D model
    print("\n" + "="*50)
    print("Training CNN1D model...")
    print("="*50)
    cnn_start_time = time.time()
    cnn_model, cnn_accuracy, cnn_f1 = train_cnn1d_model(
        train_loader, val_loader, test_loader, epochs=30, lr=0.001
    )
    cnn_training_time = time.time() - cnn_start_time
    results['CNN1D'] = {
        'model': cnn_model, 
        'accuracy': cnn_accuracy, 
        'f1': cnn_f1, 
        'training_time': cnn_training_time
    }
    
    # Train SVM model
    print("\n" + "="*50)
    print("Training SVM model...")
    print("="*50)
    svm_start_time = time.time()
    svm_model, svm_accuracy, svm_f1 = train_svm_model(
        X_train_features, y_train, X_val_features, y_val, X_test_features, y_test
    )
    svm_training_time = time.time() - svm_start_time
    results['SVM'] = {
        'model': svm_model, 
        'accuracy': svm_accuracy, 
        'f1': svm_f1, 
        'training_time': svm_training_time
    }
    
    # Train Random Forest model
    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    rf_start_time = time.time()
    rf_model, rf_accuracy, rf_f1 = train_random_forest_model(
        X_train_features, y_train, X_val_features, y_val, X_test_features, y_test
    )
    rf_training_time = time.time() - rf_start_time
    results['Random Forest'] = {
        'model': rf_model, 
        'accuracy': rf_accuracy, 
        'f1': rf_f1, 
        'training_time': rf_training_time
    }
    
    # Train Gradient Boosting model
    print("\n" + "="*50)
    print("Training Gradient Boosting model...")
    print("="*50)
    gb_start_time = time.time()
    gb_model, gb_accuracy, gb_f1 = train_gradient_boosting_model(
        X_train_features, y_train, X_val_features, y_val, X_test_features, y_test
    )
    gb_training_time = time.time() - gb_start_time
    results['Gradient Boosting'] = {
        'model': gb_model, 
        'accuracy': gb_accuracy, 
        'f1': gb_f1, 
        'training_time': gb_training_time
    }
    
    # Compare all models
    print("\n" + "="*50)
    print("Model Comparison Summary (Test Set Results)")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name:20} - Accuracy: {result['accuracy']:.4f}, F1: {result['f1']:.4f}, "
              f"Training time: {result['training_time']:.2f}s")
    
    # Visualize model comparison
    compare_models(results)
    
    # Find best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    # Save models
    import joblib
    for model_name, result in results.items():
        if model_name == "CNN1D":
            torch.save(result['model'].state_dict(), f"best_{model_name.lower()}_model.pt")
            print(f"Saved CNN1D model to best_{model_name.lower()}_model.pt")
        else:
            joblib.dump(result['model'], f"best_{model_name.lower().replace(' ', '_')}_model.pkl")
            print(f"Saved {model_name} model to best_{model_name.lower().replace(' ', '_')}_model.pkl")

if __name__ == "__main__":
    main()