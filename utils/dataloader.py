

import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from sklearn.model_selection import train_test_split
from collections import Counter
from visualization.CNN1D_visualization import plot_operation_split_bar
from pathlib import Path
import numpy as np
import h5py
from scipy.fft import fft



# 1️⃣ Custom Dataset Class
# ------------------------
class VibrationDataset(Dataset):
    '''
    This version includes the operation data so that it can be used for stratified
    sampling in the train/val/test split.
    '''
    def __init__(self, data_dir, transform=None, augment_bad=False):
        self.data_dir = Path(data_dir)
        self.file_paths = []
        self.labels = []
        self.weights = []  # Optional for inverse frequency weighting
        self.operations = []  # Optional for operation-based stratification
        self.augment_bad = augment_bad
        self.file_groups = []  # e.g., 'M01_Feb_2019_OP02_000'
        self.transform = transform


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

        self.labels = np.array(self.labels, dtype=np.int64)
        self.operations = np.array(self.operations)
        self.file_groups = np.array(self.file_groups)
        # assert len(self.file_paths) == 6383, f"Expected 6383 files, found {len(self.file_paths)}"  #it was 7501 with 80% overlap of  bad data windows, now it is 50% overlap, so less bad data


        # Calculate class weights once during initialization
        class_counts = np.bincount(self.labels)
        total = sum(class_counts)
        self.weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            data = f["vibration_data"][:]  # Shape (2000, 3)

        data = np.transpose(data, (1, 0))  # Change to (3, 2000) for CNN

        label = self.labels[idx]

        # Augment bad samples by adding noise
        if self.augment_bad and label == 0:
            data += np.random.normal(0, 0.25, data.shape)  # Add Gaussian noise

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)


        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
# ------------------------



class FrequencyTransform:
    """Transform time-domain signals to frequency domain using FFT"""

    def __init__(self, n_freq_bins=1000):
        self.n_freq_bins = n_freq_bins

    def __call__(self, data):
        # data has shape (3, 2000)
        freq_data = np.zeros((3, self.n_freq_bins), dtype=np.float32)
        for i in range(3):  # Process each axis
            # Compute FFT and take magnitude
            fft_vals = np.abs(fft(data[i]))[:self.n_freq_bins]
            # Normalize
            freq_data[i] = fft_vals / np.max(fft_vals) if np.max(fft_vals) > 0 else fft_vals
        return freq_data



# this function creates a stratified train/val/test split based on both operation and label
def stratified_data_loader(data_directory="../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD",
                        batch_size=128, augment_bad=False, test_size=0.3):
    """
    Create and return train, validation, and test data loaders with stratified split.

    Args:
        data_directory (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        augment_bad (bool): Whether to augment bad samples
        test_size (float): Proportion of the dataset to include in the test split

    Returns:
        tuple: train_loader, val_loader, test_loader, dataset
    """
    # Load dataset
    dataset = VibrationDataset(data_directory, augment_bad=augment_bad)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # Stratified split by both label and operation
    train_idx, temp_idx = train_test_split(
        range(len(dataset)), test_size=test_size, stratify=stratify_key
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

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, dataset


# this code is an extension of the previous one, ensuring that samples from the same file group are not present across train/test/val splits to prevent data leakage.
def stratified_group_split(data_directory="../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD",
                           batch_size=128, augment_bad=True, idx_return=False):
    """
    Create data loaders with stratified split while ensuring samples from the same file group
    are not present across train/test/val splits to prevent data leakage.

    Args:
        data_directory (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        augment_bad (bool): Whether to augment bad samples

    Returns:
        tuple: train_loader, val_loader, test_loader, dataset
    """
    # Import here to avoid circular imports
    from sklearn.model_selection import StratifiedGroupKFold

    # Load dataset
    dataset = VibrationDataset(data_directory, augment_bad=augment_bad)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # First use StratifiedGroupKFold to get a 70/30 split while respecting file groups
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=44)
    splits = list(sgkf.split(range(len(dataset)), stratify_key, groups=dataset.file_groups))

    # Take 7 splits for training (70%), and 3 splits for test+val (30%)
    train_idx = []
    temp_idx = []

    # First 7 splits go to training
    for i in range(7):
        train_idx.extend(splits[i][1])  # [1] contains the indices for the test fold

    # Last 3 splits go to test+val
    for i in range(7, 10):
        temp_idx.extend(splits[i][1])

    # Now split the 30% into equal parts for validation and test
    # Using StratifiedGroupKFold again for the val/test split
    temp_stratify = [stratify_key[i] for i in temp_idx]
    temp_groups = dataset.file_groups[temp_idx]

    # Use just 2 splits to get a 50/50 division of the temp dataset (15%/15% of total)
    sgkf_temp = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=44)
    temp_splits = list(sgkf_temp.split(range(len(temp_idx)), temp_stratify, groups=temp_groups))

    # Get val and test indices, mapping back to original dataset indices
    val_idx = [temp_idx[i] for i in temp_splits[0][1]]
    test_idx = [temp_idx[i] for i in temp_splits[1][1]]

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

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    plot_operation_split_bar(train_ops, val_ops, test_ops)


    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # File group distribution
    train_groups = set(dataset.file_groups[train_idx])
    val_groups = set(dataset.file_groups[val_idx])
    test_groups = set(dataset.file_groups[test_idx])

    print(f"\nUnique file groups in Train: {len(train_groups)}")
    print(f"Unique file groups in Val: {len(val_groups)}")
    print(f"Unique file groups in Test: {len(test_groups)}")

    # Check for overlap between splits
    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    print(f"\nFile group overlap between Train and Val: {len(train_val_overlap)}")
    print(f"File group overlap between Train and Test: {len(train_test_overlap)}")
    print(f"File group overlap between Val and Test: {len(val_test_overlap)}")
    if idx_return == True:
       return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, dataset

    return train_loader, val_loader, test_loader, dataset

# this is for frequency domain data, only difference is the transform applied to the data
def stratified_group_split_freq(data_directory="../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD",
                           batch_size=128, augment_bad=False):
    """
    Create data loaders with stratified split while ensuring samples from the same file group
    are not present across train/test/val splits to prevent data leakage.

    Args:
        data_directory (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        augment_bad (bool): Whether to augment bad samples

    Returns:
        tuple: train_loader, val_loader, test_loader, dataset
    """
    # Import here to avoid circular imports
    from sklearn.model_selection import StratifiedGroupKFold

    # Load dataset
    freq_transform = FrequencyTransform(n_freq_bins=1000)
    # Load dataset
    dataset = VibrationDataset(data_directory, transform=freq_transform)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}_{op}" for lbl, op in zip(dataset.labels, dataset.operations)]

    # First use StratifiedGroupKFold to get a 70/30 split while respecting file groups
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=44)
    splits = list(sgkf.split(range(len(dataset)), stratify_key, groups=dataset.file_groups))

    # Take 7 splits for training (70%), and 3 splits for test+val (30%)
    train_idx = []
    temp_idx = []

    # First 7 splits go to training
    for i in range(7):
        train_idx.extend(splits[i][1])  # [1] contains the indices for the test fold

    # Last 3 splits go to test+val
    for i in range(7, 10):
        temp_idx.extend(splits[i][1])

    # Now split the 30% into equal parts for validation and test
    # Using StratifiedGroupKFold again for the val/test split
    temp_stratify = [stratify_key[i] for i in temp_idx]
    temp_groups = dataset.file_groups[temp_idx]

    # Use just 2 splits to get a 50/50 division of the temp dataset (15%/15% of total)
    sgkf_temp = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=44)
    temp_splits = list(sgkf_temp.split(range(len(temp_idx)), temp_stratify, groups=temp_groups))

    # Get val and test indices, mapping back to original dataset indices
    val_idx = [temp_idx[i] for i in temp_splits[0][1]]
    test_idx = [temp_idx[i] for i in temp_splits[1][1]]

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

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    plot_operation_split_bar(train_ops, val_ops, test_ops)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # File group distribution
    train_groups = set(dataset.file_groups[train_idx])
    val_groups = set(dataset.file_groups[val_idx])
    test_groups = set(dataset.file_groups[test_idx])

    print(f"\nUnique file groups in Train: {len(train_groups)}")
    print(f"Unique file groups in Val: {len(val_groups)}")
    print(f"Unique file groups in Test: {len(test_groups)}")

    # Check for overlap between splits
    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    print(f"\nFile group overlap between Train and Val: {len(train_val_overlap)}")
    print(f"File group overlap between Train and Test: {len(train_test_overlap)}")
    print(f"File group overlap between Val and Test: {len(val_test_overlap)}")

    return train_loader, val_loader, test_loader, dataset

# this version only stratifies based on label, not operation
def stratified_group_label(data_directory="../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD",
                           batch_size=128, augment_bad=False, idx_return=False):
    """
    Create data loaders with stratified split while ensuring samples from the same file group
    are not present across train/test/val splits to prevent data leakage.

    Args:
        data_directory (str): Path to the data directory
        batch_size (int): Batch size for the data loaders
        augment_bad (bool): Whether to augment bad samples

    Returns:
        tuple: train_loader, val_loader, test_loader, dataset
    """
    # Import here to avoid circular imports
    from sklearn.model_selection import StratifiedGroupKFold

    # Load dataset
    dataset = VibrationDataset(data_directory, augment_bad=False)

    # Create a combined stratification key (label_operation)
    stratify_key = [f"{lbl}" for lbl in dataset.labels]

    # First use StratifiedGroupKFold to get a 70/30 split while respecting file groups
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=44)
    splits = list(sgkf.split(range(len(dataset)), stratify_key, groups=dataset.file_groups))

    # Take 7 splits for training (70%), and 3 splits for test+val (30%)
    train_idx = []
    temp_idx = []

    # First 7 splits go to training
    for i in range(7):
        train_idx.extend(splits[i][1])  # [1] contains the indices for the test fold

    # Last 3 splits go to test+val
    for i in range(7, 10):
        temp_idx.extend(splits[i][1])

    # Now split the 30% into equal parts for validation and test
    # Using StratifiedGroupKFold again for the val/test split
    temp_stratify = [stratify_key[i] for i in temp_idx]
    temp_groups = dataset.file_groups[temp_idx]

    # Use just 2 splits to get a 50/50 division of the temp dataset (15%/15% of total)
    sgkf_temp = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=44)
    temp_splits = list(sgkf_temp.split(range(len(temp_idx)), temp_stratify, groups=temp_groups))

    # Get val and test indices, mapping back to original dataset indices
    val_idx = [temp_idx[i] for i in temp_splits[0][1]]
    test_idx = [temp_idx[i] for i in temp_splits[1][1]]

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

    # Operation distribution
    train_ops = Counter(dataset.operations[train_idx])
    val_ops = Counter(dataset.operations[val_idx])
    test_ops = Counter(dataset.operations[test_idx])
    print(f"Train operations: {train_ops}")
    print(f"Val operations: {val_ops}")
    print(f"Test operations: {test_ops}")

    plot_operation_split_bar(train_ops, val_ops, test_ops)


    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # File group distribution
    train_groups = set(dataset.file_groups[train_idx])
    val_groups = set(dataset.file_groups[val_idx])
    test_groups = set(dataset.file_groups[test_idx])

    print(f"\nUnique file groups in Train: {len(train_groups)}")
    print(f"Unique file groups in Val: {len(val_groups)}")
    print(f"Unique file groups in Test: {len(test_groups)}")

    # Check for overlap between splits
    train_val_overlap = train_groups.intersection(val_groups)
    train_test_overlap = train_groups.intersection(test_groups)
    val_test_overlap = val_groups.intersection(test_groups)

    print(f"\nFile group overlap between Train and Val: {len(train_val_overlap)}")
    print(f"File group overlap between Train and Test: {len(train_test_overlap)}")
    print(f"File group overlap between Val and Test: {len(val_test_overlap)}")
    if idx_return == True:
       return train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, dataset

    return train_loader, val_loader, test_loader, dataset

# Regular train/val/test split
def load_data_normal(data_directory, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    # Load all data from directory
    dataset = VibrationDataset(data_directory)  # Assuming you have a CustomDataset class

    # Get dataset size
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Random split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, dataset


def main():
    data_directory = "../data/final/new_selection/less_bad/normalized_windowed_downsampled_data_lessBAD"
    train_loader, val_loader, test_loader, dataset = stratified_group_split(data_directory=data_directory) # this is for operation and label stratification with group split to avoid data leakage
    # train_loader, val_loader, test_loader, dataset = stratified_data_loader(data_directory=data_directory) # this is for operation and label stratification only

    # if you want to use the non-normalized data, uncomment the following lines:
    # data_directory_not_normalized = "../data/final/new_selection/Selected_data_grouped_windowed_downsampled_lessbad"
    # train_loader, val_loader, test_loader, dataset = stratified_group_split(data_directory=data_directory_not_normalized) # this is for operation and label stratification with group split to avoid data leakage, but without normalization
    return train_loader, val_loader, test_loader, dataset


if __name__ == "__main__":
    main()