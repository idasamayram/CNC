# mlp_classifier.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import gc
from Classification.classifier.result_visualization import track_time, plot_confmat_and_metrics, plot_learning_curve, get_memory_usage


class MLPModel(nn.Module):
    """PyTorch MLP implementation"""

    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=2, dropout=0.3):
        super(MLPModel, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@track_time
def train_mlp_sklearn(X_train, y_train, X_val, y_val, X_test, y_test, save_dir=None):
    """
    Train and evaluate an MLP classifier using scikit-learn.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data

    Returns:
        Tuple of (model, metrics_dict)
    """
    print("Training scikit-learn MLP model...")

    # Track memory usage before training
    memory_before = get_memory_usage()

    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(random_state=42, max_iter=500, early_stopping=True, validation_fraction=0.1))
    ])

    # Parameter grid for optimization
    param_grid = {
        'mlp__hidden_layer_sizes': [(100,), (100, 50), (100, 100)],
        'mlp__alpha': [0.0001, 0.001, 0.01],
        'mlp__learning_rate_init': [0.001, 0.01],
        'mlp__activation': ['relu', 'tanh']
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, return_train_score=True)

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

    # Track memory usage after training
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before

    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Memory used: {memory_used:.2f} MB")

    # Plot confusion matrix
    metrics_dict = plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"],
                                            title="MLP (scikit-learn) Confusion Matrix", save_dir=save_dir)

    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid,
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )

    plot_learning_curve("MLP (scikit-learn)", train_sizes_abs, train_scores, val_scores, save_dir=save_dir)

    # Clean up memory
    gc.collect()

    # Add metrics to dictionary
    metrics = {
        "model": best_pipeline,
        "model_type": "MLP_Sklearn",
        "accuracy": test_accuracy,
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "specificity": metrics_dict["specificity"],
        "f1": test_f1,
        "TP": metrics_dict["TP"],
        "FP": metrics_dict["FP"],
        "TN": metrics_dict["TN"],
        "FN": metrics_dict["FN"],
        "train_accuracy": grid_search.cv_results_['mean_train_score'][grid_search.best_index_],
        "val_accuracy": val_accuracy,
        "std_train_acc": np.std(train_scores[-1]),
        "std_val_acc": np.std(val_scores[-1]),
        "memory_usage": memory_used
    }

    return best_pipeline, metrics


@track_time
def train_mlp_pytorch(X_train, y_train, X_val, y_val, X_test, y_test, hidden_sizes=[128, 64], epochs=100, lr=0.001, save_dir=None):
    """
    Train and evaluate an MLP classifier using PyTorch.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        hidden_sizes: List of hidden layer sizes
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        Tuple of (model, metrics_dict)
    """
    print("Training PyTorch MLP model...")

    # Track memory usage before training
    memory_before = get_memory_usage()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # Convert data to torch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    input_size = X_train.shape[1]
    model = MLPModel(input_size, hidden_sizes=hidden_sizes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
        print(f"Epoch {epoch + 1}/{epochs} - "
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
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_model_weights)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    y_test_pred = predicted.cpu().numpy()

    # Track memory usage after training
    memory_after = get_memory_usage()
    memory_used = memory_after - memory_before

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Test - Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    print(f"Memory used: {memory_used:.2f} MB")

    # Plot confusion matrix
    metrics_dict = plot_confmat_and_metrics(y_test, y_test_pred, class_names=["Good", "Bad"],
                                            title="MLP (PyTorch) Confusion Matrix", save_dir=save_dir)

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP (PyTorch) Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('MLP (PyTorch) Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()

    #save the plots
    plt.savefig("./results/whole_dataset_comparison/mlp_pytorch_learning_curves.png")

    plt.show()

    # Clean up memory
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Add metrics to dictionary
    metrics = {
        "model": model,
        "model_type": "MLP_PyTorch",
        "accuracy": test_accuracy,
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "specificity": metrics_dict["specificity"],
        "f1": test_f1,
        "TP": metrics_dict["TP"],
        "FP": metrics_dict["FP"],
        "TN": metrics_dict["TN"],
        "FN": metrics_dict["FN"],
        "train_accuracy": train_accuracies[-1],
        "val_accuracy": val_accuracy,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "std_train_acc": np.std(train_accuracies[-5:]) if len(train_accuracies) >= 5 else np.std(train_accuracies),
        "std_val_acc": np.std(val_accuracies[-5:]) if len(val_accuracies) >= 5 else np.std(val_accuracies),
        "memory_usage": memory_used,
        "hyperparams": {
            "hidden_sizes": hidden_sizes,
            "learning_rate": lr,
            "epochs": epoch + 1  # Actual epochs trained
        }
    }

    return model, metrics