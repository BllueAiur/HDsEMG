# util/data_utils.py
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader


def list_dataset_files(dataset_dir):
    """
    List all pickle files in dataset_dir/<gesture>/ and return file paths, labels array, and label_map.
    """
    gestures = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    label_map = {gesture: idx for idx, gesture in enumerate(gestures)}
    file_paths, labels = [], []
    for gesture in gestures:
        gesture_dir = os.path.join(dataset_dir, gesture)
        for fname in sorted(os.listdir(gesture_dir)):
            if not fname.endswith('.pkl'):
                continue
            file_paths.append(os.path.join(gesture_dir, fname))
            labels.append(label_map[gesture])
    return file_paths, np.array(labels, dtype=np.int64), label_map


def create_cv_splits(labels, n_splits=5, seed=42):
    """
    Create train/val/test indices for each fold in a 5-fold CV with a 3:1:1 ratio.
    Returns a list of (train_idx, val_idx, test_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(labels)), labels))
    test_folds = [test_idx for _, test_idx in folds]
    splits = []
    for i in range(n_splits):
        test_idx = test_folds[i]
        val_idx = test_folds[(i + 1) % n_splits]
        train_idx = np.setdiff1d(
            np.arange(len(labels)),
            np.concatenate((test_idx, val_idx))
        )
        splits.append((train_idx, val_idx, test_idx))
    return splits


class EMGDataset(Dataset):
    """
    PyTorch Dataset that loads EMG windows from pickle files on the fly.
    Each pickle file is expected to contain a dict with key 'data'.
    """
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with open(path, 'rb') as f:
            item = pickle.load(f)
        data = item['data']
        label = int(self.labels[idx])
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor, label


def get_data_loaders(dataset_dir, split_idx, batch_size=32, num_workers=4, n_splits=5, seed=42):
    """
    Prepare DataLoaders for a given CV split index.

    Args:
        dataset_dir: base directory with one subfolder per gesture containing .pkl windows
        split_idx: integer in [0, n_splits) to select which split to use
        batch_size, num_workers, n_splits, seed: see sklearn and DataLoader

    Returns:
        train_loader, val_loader, test_loader, label_map
    """
    # 1) list files and labels
    file_paths, labels, label_map = list_dataset_files(dataset_dir)
    # 2) build CV splits
    splits = create_cv_splits(labels, n_splits=n_splits, seed=seed)
    train_idx, val_idx, test_idx = splits[split_idx]
    # 3) subset
    train_files = [file_paths[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_files   = [file_paths[i] for i in val_idx]
    val_labels  = labels[val_idx]
    test_files  = [file_paths[i] for i in test_idx]
    test_labels = labels[test_idx]
    # 4) datasets
    train_ds = EMGDataset(train_files, train_labels)
    val_ds   = EMGDataset(val_files, val_labels)
    test_ds  = EMGDataset(test_files, test_labels)
    # 5) loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, label_map
