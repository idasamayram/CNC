
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from visualization.CNN1D_visualization import *
from utils.dataloader import stratified_group_split

# ------------------------
# Model Definition
# ------------------------
class CNN1D_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_Wide, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.relu1 = nn.LeakyReLU(0.1)

        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.relu2 = nn.LeakyReLU(0.1)

        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.relu3 = nn.LeakyReLU(0.1)

        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.LeakyReLU(0.1)

        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.4)
        self.relu_fc = nn.LeakyReLU(0.1)

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
        x = self.dropout1(self.pool1(self.relu1(self.conv1(x))))
        x = self.dropout2(self.pool2(self.relu2(self.conv2(x))))
        x = self.dropout3(self.pool3(self.relu3(self.conv3(x))))
        x = self.dropout4(self.pool4(self.relu4(self.conv4(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNN1D_DS_Wide(nn.Module):
    def __init__(self):
        super(CNN1D_DS_Wide, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, kernel_size=25, stride=1, padding=12)
        self.gn1 = nn.GroupNorm(4, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.gn2 = nn.GroupNorm(4, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=4)
        self.gn3 = nn.GroupNorm(4, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.gn1(self.conv1(x))))
        x = self.pool2(self.relu(self.gn2(self.conv2(x))))
        x = self.pool3(self.relu(self.gn3(self.conv3(x))))

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ------------------------
# Simple Training Pipeline
# ------------------------
def simple_train(model, train_loader, val_loader, test_loader, epochs=30, lr=0.001, device='cpu'):
    """Simple training loop: no scheduling, no early stopping, no class weighting."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # --- Test ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n🔥 Test F1: {f1:.4f}, Test Acc: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Good", "Bad"]))

    plot_confmat_and_metrics(all_labels, all_preds, class_names=["Good", "Bad"],
                             title="Confusion Matrix & Key Metrics")
    return model


# ------------------------
# Run Training
# ------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_directory = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    train_loader, val_loader, test_loader, _ = stratified_group_split(
        data_directory, augment_bad=False  # no augmentation
    )

    model = CNN1D_Wide()
    trained_model = simple_train(model, train_loader, val_loader, test_loader,
                                  epochs=30, lr=0.001, device=device)

    torch.save(trained_model.state_dict(), "../cnn1d_model_new.ckpt")
    print("✅ Model saved to cnn1d_model_new_test.ckpt")
