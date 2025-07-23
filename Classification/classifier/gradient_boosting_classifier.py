# gradient_boosting_classifier.py
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import gc
from Classification.classifier.result_visualization import track_time, plot_confmat_and_metrics, plot_learning_curve, get_memory_usage


@track_time
def train_gradient_boosting_model(X_train, y_train, X_val, y_val, X_test, y_test, save_dir=None):
    """
    Train and evaluate a Gradient Boosting classifier with parameter tuning.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data

    Returns:
        Tuple of (model, metrics_dict)
    """
    print("Training Gradient Boosting model...")

    # Track memory usage before training
    memory_before = get_memory_usage()

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
                                            title="Gradient Boosting Confusion Matrix", save_dir=save_dir)

    # Plot feature importances
    gb = best_pipeline.named_steps['gb']
    feature_importances = gb.feature_importances_

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title('Gradient Boosting Feature Importances', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()

    # Save or show the plot
    plt.savefig("./results/whole_dataset_comparison/gradient_boosting_feature_importances.png")
    plt.show()

    # Plot learning curve
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        best_pipeline, X_grid, y_grid,
        train_sizes=train_sizes, cv=5,
        scoring='accuracy', n_jobs=-1
    )

    plot_learning_curve("Gradient Boosting", train_sizes_abs, train_scores, val_scores, save_dir=save_dir)

    # Clean up memory
    gc.collect()

    # Add metrics to dictionary
    metrics = {
        "model": best_pipeline,
        "model_type": "Gradient_Boosting",
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