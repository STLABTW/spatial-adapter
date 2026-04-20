"""Synthetic binary classification data for adapter testing."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_synthetic_binary_dataloader_and_val(
    n_samples: int = 200,
    n_locations: int = 5,
    feature_dim: int = 64,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, np.ndarray]:
    """Build synthetic binary data and validation tensors.

    Returns (train_loader, val_cont, val_y, locs).
    Labels are sampled from Bernoulli(sigmoid(0.5 * mean_feat + noise)).
    """
    rng = np.random.default_rng(seed)
    T, N, p = n_samples, n_locations, feature_dim

    cont = rng.standard_normal((T, N, p)).astype(np.float32)
    mean_feat = cont.mean(axis=-1)
    noise = rng.standard_normal((T, N)).astype(np.float32) * 0.3
    logit = mean_feat * 0.5 + noise
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    targets = (rng.random((T, N)) < prob).astype(np.float32)

    n_train = int(T * train_ratio)
    train_cont = torch.from_numpy(cont[:n_train])
    train_y = torch.from_numpy(targets[:n_train])
    val_cont = torch.from_numpy(cont[n_train:])
    val_y = torch.from_numpy(targets[n_train:])

    train_cat = torch.zeros(n_train, N, 0, dtype=torch.long)
    dataset = TensorDataset(train_cat, train_cont, train_y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    locs = np.linspace(-1.0, 1.0, N).astype(np.float64)[:, np.newaxis]
    return train_loader, val_cont, val_y, locs
