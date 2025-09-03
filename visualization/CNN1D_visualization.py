import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import h5py





def plot_confmat_and_metrics(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("Only works for binary classification (2 classes).")
    # TN, FP, FN, TP = cm.ravel()
    TP, FN, FP, TN = cm.ravel()  # If you want to flip the positive/negative class designation
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
        class_names = ["Negative", "Positive"]
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


def plot_operation_split_bar(train_ops, val_ops, test_ops, title="Stratified Distribution of Train/Val/Test Sets per Operation"):
    """
    Plots a bar chart showing the distribution of operations in train/val/test sets.
    Args:
        train_ops, val_ops, test_ops: Counters with operation as key and count as value.
        title: Title for the plot.
    """
    all_ops = sorted(set(train_ops) | set(val_ops) | set(test_ops))
    n_groups = len(all_ops)
    train_counts = [train_ops.get(op, 0) for op in all_ops]
    val_counts = [val_ops.get(op, 0) for op in all_ops]
    test_counts = [test_ops.get(op, 0) for op in all_ops]

    ind = np.arange(n_groups)
    width = 0.7

    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = ax.bar(ind, train_counts, width, label='train', color="#FFCB57")
    p2 = ax.bar(ind, val_counts, width, bottom=train_counts, label='validate', color="#A348A6")
    bottom_stacked = np.array(train_counts) + np.array(val_counts)
    p3 = ax.bar(ind, test_counts, width, bottom=bottom_stacked, label='test', color="#36964A")

    ax.set_ylabel('Number of Samples')
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(all_ops, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

# this function visualizes the results0 of unseen data classification
def visualize_unseen_results(results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Good', 'Bad'],
        yticklabels=['Good', 'Bad']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix on Unseen Data\nAccuracy: {results["accuracy"]:.4f}, F1-Score: {results["f1_score"]:.4f}')
    plt.tight_layout()
    plt.savefig('unseen_data_confusion_matrix.png', dpi=300)
    plt.show()

    # If there are misclassifications, analyze one example
    if results['misclassified_files']:
        print(f"Found {len(results['misclassified_files'])} misclassified samples")
        # Load one misclassified sample to visualize
        misclass_file = results['misclassified_files'][0]
        print(f"Example misclassified file: {misclass_file}")

        with h5py.File(misclass_file, "r") as f:
            misclass_data = f["vibration_data"][:]

        misclass_data = np.transpose(misclass_data, (1, 0))

        # Plot the misclassified sample
        plt.figure(figsize=(10, 6))
        time = np.arange(misclass_data.shape[1]) / 400  # Convert to seconds
        axes_labels = ['X', 'Y', 'Z']

        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(time, misclass_data[i], 'k-')
            plt.ylabel(f'{axes_labels[i]}-axis')
            if i == 0:
                plt.title(f"Misclassified Sample: {misclass_file.name}")
            if i == 2:
                plt.xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig('misclassified_sample.png', dpi=300)
        plt.show()

