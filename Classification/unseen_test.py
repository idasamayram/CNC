
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
from visualization.CNN1D_visualization import *
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, accuracy_score
from Classification.cnn1D_model import CNN1D_Wide, CNN1D_DS_Wide
import seaborn as sns




# define a dataset class for the unseen data as they will be used for test and we don't need to split them based on their operations or labels
class UnseenVibrationDataset(Dataset):
    '''
    Dataset class for unseen vibration data.
    with the same structure as the original dataset, but without operations.
    '''
    def __init__(self, data_dir, transform=None, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.augment_bad = augment_bad
        self.transform = transform


        # Assuming the same folder structure as your original data
        for label, label_idx in zip(["good", "bad"], [0, 1]):
            folder = self.data_dir / label
            if folder.exists():
                for file_name in folder.glob("*.h5"):
                    self.file_paths.append(file_name)
                    self.labels.append(label_idx)

        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.file_paths)} files from unseen data")

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

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)


        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# unseen data evaluation
def evaluate_on_unseen_data(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    file_paths = dataloader.dataset.file_paths

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Get misclassified files
    misclassified_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
    misclassified_files = [file_paths[i] for i in misclassified_indices]

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'misclassified_files': misclassified_files
    }
# ------------------------
# Evaluate model on unseen data with balanced metrics
def evaluate_on_unseen_data_balanced(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate balanced metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    recall_good = recall_score(all_labels, all_preds, pos_label=0)
    recall_bad = recall_score(all_labels, all_preds, pos_label=1)
    precision_good = precision_score(all_labels, all_preds, pos_label=0)
    precision_bad = precision_score(all_labels, all_preds, pos_label=1)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    f1_balanced = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Standard Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Recall (Good): {recall_good:.4f}, Recall (Bad): {recall_bad:.4f}")
    print(f"Precision (Good): {precision_good:.4f}, Precision (Bad): {precision_bad:.4f}")
    print(f"Weighted F1: {f1:.4f}, Macro F1: {f1_balanced:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'recall_good': recall_good,
        'recall_bad': recall_bad,
        'precision_good': precision_good,
        'precision_bad': precision_bad,
        'f1_weighted': f1,
        'f1_macro': f1_balanced,
        'confusion_matrix': conf_matrix,
        'all_preds': all_preds,
        'all_labels': all_labels
    }
# ------------------------
# Evaluate model with weighting to favor minority class
def evaluate_with_weighting(model, dataloader, device):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_labels.append(labels)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Apply weighting to outputs before making prediction
    # This adjusts the decision boundary to favor the minority class
    class_weights = torch.tensor([1.0, 5.0]).to(device)  # Weight bad class 5x more
    weighted_outputs = all_outputs * class_weights

    _, preds = torch.max(weighted_outputs, 1)
    all_preds = preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    # Calculate metrics with adjusted predictions
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

# ------------------------
def find_optimal_threshold(model, dataloader, device):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)
            # Get probability for "bad" class (class 1)
            scores = probas[:, 1]

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Try different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    results = []

    for threshold in thresholds:
        preds = (all_scores >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(all_labels, preds)
        recall_bad = recall_score(all_labels, preds, pos_label=1)
        precision_bad = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        f1_bad = 2 * (precision_bad * recall_bad) / (precision_bad + recall_bad) if (precision_bad + recall_bad) > 0 else 0

        results.append({
            'threshold': threshold,
            'balanced_accuracy': balanced_acc,
            'recall_bad': recall_bad,
            'precision_bad': precision_bad,
            'f1_bad': f1_bad
        })

    # Convert results0 to DataFrame for easier analysis
    import pandas as pd
    results_df = pd.DataFrame(results)

    # Plot the results0
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['threshold'], results_df['balanced_accuracy'], label='Balanced Accuracy')
    plt.plot(results_df['threshold'], results_df['recall_bad'], label='Recall (Bad)')
    plt.plot(results_df['threshold'], results_df['precision_bad'], label='Precision (Bad)')
    plt.plot(results_df['threshold'], results_df['f1_bad'], label='F1 (Bad)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Performance Across Different Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('threshold_optimization.png', dpi=300)
    plt.show()

    # Find optimal threshold for balanced accuracy
    best_idx = np.argmax(results_df['balanced_accuracy'])
    optimal_threshold = results_df.loc[best_idx, 'threshold']

    print(f"Optimal threshold for balanced accuracy: {optimal_threshold:.2f}")
    print(f"Balanced accuracy at this threshold: {results_df.loc[best_idx, 'balanced_accuracy']:.4f}")
    print(f"Recall (Bad) at this threshold: {results_df.loc[best_idx, 'recall_bad']:.4f}")
    print(f"Precision (Bad) at this threshold: {results_df.loc[best_idx, 'precision_bad']:.4f}")

    return optimal_threshold, results_df

# 5. Main function to test on unseen data
def test_on_unseen_data(unseen_data_path, model_path, batch_size=32):
    print("Loading unseen dataset...")
    unseen_dataset = UnseenVibrationDataset(unseen_data_path)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    print("Loading trained model...")
    model, device = load_model(model_path)

    print("Evaluating on unseen data...")
    results = evaluate_with_weighting(model, unseen_loader, device)

    # print(f"Results on unseen data:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    # print(f"Recall (Good): {results0['recall_good']:.4f}, Recall (Bad): {results0['recall_bad']:.4f}")
    # print(f"Precision (Good): {results0['precision_good']:.4f}, Precision (Bad): {results0['precision_bad']:.4f}")
    # print(f"f1 Weighted: {results0['f1_weighted']:.4f}, f1 Macro: {results0['f1_macro']:.4f}")
    # print(f"f1_macro: {results0['f1_macro']:.4f}")

    # Visualize results0
    visualize_unseen_results(results)

    return results

def compare_models_on_unseen_data(unseen_data_path, model1_path, model2_path, batch_size=128):
    # Load dataset
    unseen_dataset = UnseenVibrationDataset(unseen_data_path)
    unseen_loader = DataLoader(unseen_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    model1 = CNN1D_Wide()
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device)
    model1.eval()

    model2 = CNN1D_Wide()
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device)
    model2.eval()

    # Evaluate models
    print("Evaluating CNN1D_Wide...")
    results1 = evaluate_with_weighting(model1, unseen_loader, device)

    print("\nEvaluating CNN1D_Wide...")
    results2 = evaluate_with_weighting(model2, unseen_loader, device)

    # Compare results0
    compare_model_results(results1, results2)

    return results1, results2

def compare_model_results(results1, results2):
    """Create visualization comparing model performance on unseen data"""
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score']

    # Extract metrics
    metrics1 = [results1[m] for m in metrics]
    metrics2 = [results2[m] for m in metrics]

    # Plot comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, metrics1, width, label='1D-CNN_Wide')
    plt.bar(x + width/2, metrics2, width, label='1D-CNN_GN')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison on Unseen Data')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add values on bars
    for i, v in enumerate(metrics1):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')

    for i, v in enumerate(metrics2):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig('model_comparison_unseen_data.png', dpi=300)
    plt.show()

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(results1['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    ax1.set_title('1D-CNN1D_Wide Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(results2['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
    ax2.set_title('1D-CNN-Wide weighted Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300)
    plt.show()

def analyze_data_distribution(train_loader, unseen_loader):
    """Compare signal statistics between training and unseen data"""
    import scipy.stats as stats

    # Extract batch of samples from each loader
    train_batch = next(iter(train_loader))
    unseen_batch = next(iter(unseen_loader))

    # Get just the data, not labels
    train_samples = train_batch[0].cpu().numpy()
    unseen_samples = unseen_batch[0].cpu().numpy()

    # If we don't have enough samples, collect more
    if train_samples.shape[0] < 100:
        train_data = []
        for i, (inputs, _) in enumerate(train_loader):
            train_data.append(inputs.cpu().numpy())
            if i >= 5:  # Collect up to 5 batches
                break
        train_samples = np.concatenate(train_data, axis=0)

    if unseen_samples.shape[0] < 100:
        unseen_data = []
        for i, (inputs, _) in enumerate(unseen_loader):
            unseen_data.append(inputs.cpu().numpy())
            if i >= 5:  # Collect up to 5 batches
                break
        unseen_samples = np.concatenate(unseen_data, axis=0)

    # Calculate statistics
    metrics = ['Mean', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis']
    axes = ['X-axis', 'Y-axis', 'Z-axis']

    fig, axs = plt.subplots(3, len(metrics), figsize=(18, 10))

    for i in range(3):  # For each axis
        # Calculate statistics
        train_mean = np.mean(train_samples[:, i, :], axis=1)
        unseen_mean = np.mean(unseen_samples[:, i, :], axis=1)

        train_std = np.std(train_samples[:, i, :], axis=1)
        unseen_std = np.std(unseen_samples[:, i, :], axis=1)

        train_min = np.min(train_samples[:, i, :], axis=1)
        unseen_min = np.min(unseen_samples[:, i, :], axis=1)

        train_max = np.max(train_samples[:, i, :], axis=1)
        unseen_max = np.max(unseen_samples[:, i, :], axis=1)

        train_skew = stats.skew(train_samples[:, i, :], axis=1)
        unseen_skew = stats.skew(unseen_samples[:, i, :], axis=1)

        train_kurt = stats.kurtosis(train_samples[:, i, :], axis=1)
        unseen_kurt = stats.kurtosis(unseen_samples[:, i, :], axis=1)

        # Plot histograms
        stats_list = [
            (train_mean, unseen_mean),
            (train_std, unseen_std),
            (train_min, unseen_min),
            (train_max, unseen_max),
            (train_skew, unseen_skew),
            (train_kurt, unseen_kurt)
        ]

        for j, (train_stat, unseen_stat) in enumerate(stats_list):
            axs[i, j].hist(train_stat, bins=20, alpha=0.5, label='Training')
            axs[i, j].hist(unseen_stat, bins=20, alpha=0.5, label='Unseen')

            if i == 0:
                axs[i, j].set_title(metrics[j])
            if j == 0:
                axs[i, j].set_ylabel(axes[i])

            # T-test to check if distributions are significantly different
            try:
                t_stat, p_val = stats.ttest_ind(train_stat, unseen_stat)
                axs[i, j].annotate(f'p={p_val:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
            except:
                axs[i, j].annotate(f'p=N/A', xy=(0.05, 0.95), xycoords='axes fraction')

            if j == len(metrics)-1 and i == 0:
                axs[i, j].legend()

    plt.tight_layout()
    plt.suptitle('Distribution Comparison: Training vs Unseen Data')
    plt.subplots_adjust(top=0.92)
    plt.savefig('data_distribution_comparison.png', dpi=300)
    plt.show()

# confidence of the model on unseen data, needs to be corrected for errors
def analyze_model_confidence(model, dataloader, device):
    """Analyze prediction confidence distribution"""
    model.eval()
    confidences_correct = []
    confidences_wrong = []
    confidences_good = []
    confidences_bad = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Get the confidence (highest probability)
            confidence, predictions = torch.max(probs, dim=1)

            # Separate confidences for correct and wrong predictions
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                if pred == label:
                    confidences_correct.append(confidence[i].item())
                else:
                    confidences_wrong.append(confidence[i].item())

                # Also track confidences by class
                if label == 0:  # Good samples
                    confidences_good.append((confidence[i].item(), pred == label))
                else:  # Bad samples
                    confidences_bad.append((confidence[i].item(), pred == label))

    # Plot confidence distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Overall confidence by correctness
    ax1.hist([confidences_correct, confidences_wrong], bins=20,
             label=[f'Correct Predictions ({len(confidences_correct)})',
                   f'Wrong Predictions ({len(confidences_wrong)})'],
             alpha=0.7)
    ax1.set_xlabel('Confidence (Probability)')
    ax1.set_ylabel('Count')
    ax1.set_title('Model Confidence by Prediction Correctness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confidence by class
    conf_good = [c[0] for c in confidences_good]
    correct_good = [c[1] for c in confidences_good]
    conf_bad = [c[0] for c in confidences_bad]
    correct_bad = [c[1] for c in confidences_bad]

    # Plot confidence histograms by class
    ax2.hist([conf_good, conf_bad], bins=20,
             label=[f'Good Samples (n={len(conf_good)})',
                   f'Bad Samples (n={len(conf_bad)})'],
             alpha=0.7)
    ax2.set_xlabel('Confidence (Probability)')
    ax2.set_ylabel('Count')
    ax2.set_title('Model Confidence by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_confidence_analysis.png', dpi=300)
    plt.show()

    # Calculate average confidences
    avg_conf_correct = np.mean(confidences_correct) if confidences_correct else 0
    avg_conf_wrong = np.mean(confidences_wrong) if confidences_wrong else 0
    avg_conf_good = np.mean(conf_good) if conf_good else 0
    avg_conf_bad = np.mean(conf_bad) if conf_bad else 0
    good_accuracy = sum(correct_good) / len(correct_good) if correct_good else 0
    bad_accuracy = sum(correct_bad) / len(correct_bad) if correct_bad else 0

    print(f"Average confidence for correct predictions: {avg_conf_correct:.4f}")
    print(f"Average confidence for wrong predictions: {avg_conf_wrong:.4f}")
    print(f"Average confidence for good samples: {avg_conf_good:.4f}")
    print(f"Average confidence for bad samples: {avg_conf_bad:.4f}")
    print(f"Accuracy on good samples: {good_accuracy:.4f}")
    print(f"Accuracy on bad samples: {bad_accuracy:.4f}")

    # Create another figure for confidence vs. correctness by class
    plt.figure(figsize=(12, 6))

    # Create confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate accuracy per confidence bin for each class
    accuracy_by_conf_good = []
    accuracy_by_conf_bad = []
    counts_good = []
    counts_bad = []

    for i in range(len(bins)-1):
        # Good samples
        in_bin = [(c, p) for c, p in zip(conf_good, correct_good) if bins[i] <= c < bins[i+1]]
        if in_bin:
            acc = sum(p for _, p in in_bin) / len(in_bin)
            count = len(in_bin)
        else:
            acc = 0
            count = 0
        accuracy_by_conf_good.append(acc)
        counts_good.append(count)

        # Bad samples
        in_bin = [(c, p) for c, p in zip(conf_bad, correct_bad) if bins[i] <= c < bins[i+1]]
        if in_bin:
            acc = sum(p for _, p in in_bin) / len(in_bin)
            count = len(in_bin)
        else:
            acc = 0
            count = 0
        accuracy_by_conf_bad.append(acc)
        counts_bad.append(count)

    # Plot accuracy by confidence
    plt.bar(bin_centers - 0.025, accuracy_by_conf_good, width=0.05, label='Good Samples', alpha=0.7)
    plt.bar(bin_centers + 0.025, accuracy_by_conf_bad, width=0.05, label='Bad Samples', alpha=0.7)

    # Annotate with counts
    for i, (x, y, count) in enumerate(zip(bin_centers - 0.025, accuracy_by_conf_good, counts_good)):
        if count > 0:
            plt.annotate(str(count), (x, y + 0.05), ha='center', va='bottom', fontsize=8)

    for i, (x, y, count) in enumerate(zip(bin_centers + 0.025, accuracy_by_conf_bad, counts_bad)):
        if count > 0:
            plt.annotate(str(count), (x, y + 0.05), ha='center', va='bottom', fontsize=8)

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Confidence by Class')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('confidence_calibration.png', dpi=300)
    plt.show()

    return {
        'avg_confidence_correct': avg_conf_correct,
        'avg_confidence_wrong': avg_conf_wrong,
        'avg_confidence_good': avg_conf_good,
        'avg_confidence_bad': avg_conf_bad,
        'accuracy_good': good_accuracy,
        'accuracy_bad': bad_accuracy
    }

def load_model(model_path, model_class=CNN1D_DS_Wide):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device


# Usage example
if __name__ == "__main__":
    # test two models on unseen data to compare their performance
    # Path to your unseen data
    unseen_data_path = "../data/final/unseen_data/normalized_windowed_downsampled_unseen"
    unseen_data_path = "../data/final/unseen_data/normalized_windowed_downsampled_unseen"
    model1_path = "../cnn1d_model_wide_new.ckpt"
    model2_path = "../cnn1d_model_ds_wide_new.ckpt"
    result1, result2 = compare_models_on_unseen_data(unseen_data_path, model1_path, model2_path)

    # test one model on unseen data
    # Path to your trained model
    # model_path = "../cnn1d_model_wide_new.ckpt"
    # Test on unseen data
    # results0 = test_on_unseen_data(unseen_data_path, model_path)


