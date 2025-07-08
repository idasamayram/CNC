import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns




def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, title="Confusion Matrix"):
    """
    y_true: array-like of shape (n_samples,)
    y_pred: array-like of shape (n_samples,)
    class_names: list of string, default None
    normalize: bool, default False
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], ".2f" if normalize else "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix_pretty(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = ["Negative", "Positive"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=2, linecolor='white', cbar=False, annot_kws={'size': 18})
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.show()

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
