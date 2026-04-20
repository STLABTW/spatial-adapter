"""Synthetic 2D patch-level binary data mimicking wheat-head detection."""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_2d_grid(grid_h: int, grid_w: int) -> np.ndarray:
    """Normalized 2D grid locations in [0, 1]^2, shape (H*W, 2)."""
    rows = np.linspace(0, 1, grid_h)
    cols = np.linspace(0, 1, grid_w)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.column_stack([rr.ravel(), cc.ravel()]).astype(np.float64)


def _gaussian_bump_2d(
    locs: np.ndarray,
    centre: Tuple[float, float] = (0.5, 0.5),
    sigma: float = 0.25,
) -> np.ndarray:
    """Unit-norm 2D Gaussian bump over spatial locations, shape (N,)."""
    diff = locs - np.array(centre)
    phi = np.exp(-0.5 * np.sum(diff**2, axis=1) / sigma**2)
    norm = np.linalg.norm(phi)
    if norm > 0:
        phi /= norm
    return phi.astype(np.float64)


def get_synthetic_patch_dataloader_and_val(
    n_images: int = 200,
    grid_h: int = 16,
    grid_w: int = 16,
    feature_dim: int = 64,
    n_basis: int = 2,
    ar_coeff: float = 0.8,
    signal_std: float = 2.0,
    noise_std: float = 0.5,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Build synthetic 2D patch-level binary data for the spatial adapter.

    Data model (logit scale)::

        logit(Y(t, s)) = x(t,s)^T beta + sum_k eta_k(t) phi_k(s) + eps

    Returns (train_loader, val_cont, val_y, locs, true_phi, train_prob, val_prob).
    """
    rng = np.random.default_rng(seed)
    T, N, p = n_images, grid_h * grid_w, feature_dim

    locs = _make_2d_grid(grid_h, grid_w)

    # True spatial basis (up to 3 Gaussian bumps)
    centres = [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]
    sigmas = [0.25, 0.20, 0.20]
    phi_list = [
        _gaussian_bump_2d(locs, centre=centres[k], sigma=sigmas[k])
        for k in range(min(n_basis, len(centres)))
    ]
    true_phi = np.column_stack(phi_list)
    K = true_phi.shape[1]

    # Temporal scores: AR(1)
    eta = np.zeros((T, K), dtype=np.float64)
    innov_std = signal_std * np.sqrt(1 - ar_coeff**2)
    for t in range(1, T):
        eta[t] = ar_coeff * eta[t - 1] + rng.normal(0, innov_std, K)

    # Features (weak trend) + strong spatial signal + noise → logit → Bernoulli
    cont = rng.standard_normal((T, N, p)).astype(np.float32)
    beta = np.zeros(p, dtype=np.float32)
    beta[:3] = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    trend = cont @ beta

    spatial_signal = eta @ true_phi.T
    noise = rng.normal(0, noise_std, (T, N)).astype(np.float64)
    logit = trend.astype(np.float64) + spatial_signal + noise

    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    targets = (rng.random((T, N)) < prob).astype(np.float32)

    # Train / val split
    n_train = int(T * train_ratio)
    train_cont = torch.from_numpy(cont[:n_train])
    train_y = torch.from_numpy(targets[:n_train])
    val_cont = torch.from_numpy(cont[n_train:])
    val_y = torch.from_numpy(targets[n_train:])

    train_cat = torch.zeros(n_train, N, 0, dtype=torch.long)
    dataset = TensorDataset(train_cat, train_cont, train_y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_prob = torch.from_numpy(prob[:n_train].astype(np.float32))
    val_prob = torch.from_numpy(prob[n_train:].astype(np.float32))

    return train_loader, val_cont, val_y, locs, true_phi, train_prob, val_prob
