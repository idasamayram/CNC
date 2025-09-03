
import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping
from visualization.CNN1D_visualization import *
from utils.dataloader import  stratified_group_split
from sklearn.metrics import recall_score, classification_report
import copy

# 2️⃣ Define the CNN Model for downsampled data
# ------------------------
class CNN1D_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        # Wider kernels to increase receptive field
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)  # Increased kernel size
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout1 = nn.Dropout(0.2)  # Add dropout after first layer

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)  # Increased kernel size
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout2 = nn.Dropout(0.2)  # Add dropout after second layer

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)  # Increased kernel size
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)  # Increased pooling
        self.dropout3 = nn.Dropout(0.2)  # Add dropout after third layer

        # NEW: Add a fourth convolutional layer for deeper network
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)  # Changed input size to match conv4 output
        self.fc2 = nn.Linear(64, 2)  # Binary classification

        self.dropout = nn.Dropout(0.4)  # Increased dropout for final layer
        self.relu = nn.LeakyReLU(0.1)  # Using LeakyReLU for better gradient flow

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No activation (we use CrossEntropyLoss)

        return x

# ------------------------
# 3️⃣ Train & Evaluate Functions
# ------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recall_good = report["0"]["recall"]
    recall_bad = report["1"]["recall"]

    return train_loss, train_acc, recall_good, recall_bad

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    misclassified_indices = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Track misclassified indices if needed
            batch_misclassified = torch.where(predicted != labels)[0]
            for idx in batch_misclassified:
                global_idx = batch_idx * val_loader.batch_size + idx.item()
                misclassified_indices.append(global_idx)

    val_loss /= len(val_loader)
    val_acc = correct / total

    # Compute per-class recall using classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recall_good = report["0"]["recall"]  # good = class 0
    recall_bad = report["1"]["recall"]   # bad = class 1

    return val_loss, val_acc, misclassified_indices, recall_good, recall_bad
# ------------------------
# 4️⃣ Test the Model
# ------------------------
def test_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute detailed metrics
    report = classification_report(all_labels, all_preds, target_names=["Good", "Bad"], output_dict=True)
    f1 = report["weighted avg"]["f1-score"]
    accuracy = report["accuracy"]
    recall_good = report["Good"]["recall"]
    recall_bad = report["Bad"]["recall"]

    print(f"🔥 Test F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, "
          f"Recall (Good): {recall_good:.4f}, Recall (Bad): {recall_bad:.4f}")

    return f1, accuracy, all_labels, all_preds
# ------------------------
# 5️⃣ Full Training Pipeline
# ------------------------
def train_and_evaluate(train_loader, val_loader, test_loader, model_class=CNN1D_Wide, weights = None, epochs=30, lr=0.001, weight_decay=1e-4, EralyStopping=False, Schedule=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Model setup
    model = model_class().to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)) # Use weighted loss if provided

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping variables
    # best_val_loss = float('inf')
    best_model_weights = None
    # patience_counter = 0
    early_stop_epoch = epochs
    patience = 3

    # Early stopping variables - monitor recall_bad instead of loss
    best_val_recall_bad = 0
    patience = 10
    counter = 0
    best_model_weights = None

    # Training metrics tracking
    # train_losses, val_losses = [], []
    train_recalls_bad, val_recalls_bad = [], []


    # train_recalls_bad, val_recalls_bad = [], []



    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    # or use ReduceLROnPlateau scheduler
    # scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.5,  patience=2)

    # Training and validation loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        train_loss, train_acc, train_recall_good, train_recall_bad = train_epoch(model, train_loader, optimizer,
                                                                                 criterion, device)
        val_loss, val_acc, _, val_recall_good, val_recall_bad = validate_epoch(model, val_loader, criterion, device)

        train_recalls_bad.append(train_recall_bad)
        val_recalls_bad.append(val_recall_bad)

        ...
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Recall(Bad): {train_recall_bad:.4f} - "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Recall(Bad): {val_recall_bad:.4f}")

        # Step the scheduler
        if Schedule:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate

            # or Step the scheduler based on validation loss
            # scheduler.step(val_loss)
            # current_lr = scheduler.get_last_lr()[0]


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_recalls_bad.append(train_recall_bad)
        val_recalls_bad.append(val_recall_bad)



        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ")
              #f"Learning Rate: {current_lr:.6f}")

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Bad Recall: {train_recall_bad:.4f} - "
              f"Val Loss: {val_loss:.4f}, Bad Recall: {val_recall_bad:.4f}")

        # Save model if bad class recall improves
        if val_recall_bad > best_val_recall_bad:
            best_val_recall_bad = val_recall_bad
            best_model_weights = copy.deepcopy(model.state_dict())
            counter = 0
            print(f"✅ New best model with Bad Recall: {val_recall_bad:.4f}")
        else:
            counter += 1



        # Early stopping based on bad class recall
        if EarlyStopping == True:
            if counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                early_stop_epoch = epoch + 1
                break

    # Restore best model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Restored best model with Bad Recall: {best_val_recall_bad:.4f}")


    print("✅ Training and validation complete!")


    # Evaluate on the test set
    # f1, accuracy, all_labels, all_preds = test_model(model, test_loader, device)
    f1, accuracy, y_true, y_pred = test_model(model, test_loader, device)

    print(f"🔥 Test F1 Score: {f1:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    # plot_confusion_matrix(all_labels, all_preds, class_names=["Good", "Bad"], normalize=False)
    plot_confmat_and_metrics(y_true, y_pred, class_names=["Good", "Bad"], title="Confusion Matrix & Key Metrics")

    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    if EralyStopping:
        ax1.plot(range(1, early_stop_epoch + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, early_stop_epoch + 1), val_losses, label="Val Loss")
    else:
        ax1.plot(range(1, epochs + 1), train_losses, label="Train Loss")
        ax1.plot(range(1, epochs + 1), val_losses, label="Val Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    if EralyStopping:
        ax2.plot(range(1, early_stop_epoch + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, early_stop_epoch + 1), val_accuracies, label="Val Accuracy")
    else:
        ax2.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
        ax2.plot(range(1, epochs + 1), val_accuracies, label="Val Accuracy")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return model

# 6️⃣ Run Training & Evaluation
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Splitting the dataset with stratified sampling based on operations,labels, and groups
    data_directory = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    train_loader, val_loader, test_loader, dataset = stratified_group_split(data_directory, augment_bad=True)

    best_model = train_and_evaluate(train_loader, val_loader, test_loader, model_class= CNN1D_Wide, weights=dataset.weights, EralyStopping=False, Schedule=True, epochs=30, lr=0.001, weight_decay=1e-4)

    # Save the best model
    # Save the trained model

    torch.save(best_model.state_dict(), "../cnn1d_model_final.ckpt")
    print("✅ Model saved to cnn1d_model_test_newest_newest.ckpt")
    best_model.to(device)
    best_model.eval()  # Switch to evaluation mode


