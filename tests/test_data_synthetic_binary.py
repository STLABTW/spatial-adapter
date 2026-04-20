"""Unit tests for synthetic binary data generators."""

import numpy as np
import pytest
import torch

from spatial_adapter.data.synthetic_binary import (
    get_synthetic_binary_dataloader_and_val,
)
from spatial_adapter.data.synthetic_patch_binary import (
    _gaussian_bump_2d,
    _make_2d_grid,
    get_synthetic_patch_dataloader_and_val,
)

# ---------------------------------------------------------------------------
# get_synthetic_binary_dataloader_and_val
# ---------------------------------------------------------------------------


class TestSyntheticBinary:
    def test_shapes(self):
        loader, val_cont, val_y, locs = get_synthetic_binary_dataloader_and_val(
            n_samples=100,
            n_locations=8,
            feature_dim=32,
            train_ratio=0.8,
        )
        # Train loader
        cat, cont, y = next(iter(loader))
        assert cont.shape[-1] == 32  # feature_dim
        assert y.ndim == 2  # (B, N)
        # Val
        assert val_cont.shape == (20, 8, 32)
        assert val_y.shape == (20, 8)
        # Locs
        assert locs.shape == (8, 1)

    def test_binary_labels(self):
        _, _, val_y, _ = get_synthetic_binary_dataloader_and_val()
        unique = set(val_y.numpy().ravel().tolist())
        assert unique <= {0.0, 1.0}

    def test_deterministic(self):
        _, c1, y1, _ = get_synthetic_binary_dataloader_and_val(seed=0)
        _, c2, y2, _ = get_synthetic_binary_dataloader_and_val(seed=0)
        assert torch.equal(c1, c2)
        assert torch.equal(y1, y2)

    def test_different_seeds(self):
        _, _, y1, _ = get_synthetic_binary_dataloader_and_val(seed=0)
        _, _, y2, _ = get_synthetic_binary_dataloader_and_val(seed=1)
        assert not torch.equal(y1, y2)

    def test_train_val_disjoint(self):
        loader, val_cont, val_y, _ = get_synthetic_binary_dataloader_and_val(
            n_samples=50,
            train_ratio=0.6,
        )
        train_cont = list(loader.dataset.tensors)[1]
        # train has 30, val has 20 — together 50
        assert train_cont.shape[0] + val_cont.shape[0] == 50

    def test_loader_iterates(self):
        loader, _, _, _ = get_synthetic_binary_dataloader_and_val(batch_size=8)
        batches = list(loader)
        assert len(batches) >= 1
        cat, cont, y = batches[0]
        assert cat.shape[-1] == 0  # empty categorical


# ---------------------------------------------------------------------------
# _make_2d_grid / _gaussian_bump_2d  (helpers)
# ---------------------------------------------------------------------------


class TestPatchHelpers:
    def test_grid_shape(self):
        locs = _make_2d_grid(4, 5)
        assert locs.shape == (20, 2)

    def test_grid_range(self):
        locs = _make_2d_grid(10, 10)
        assert locs.min() >= 0.0
        assert locs.max() <= 1.0

    def test_bump_unit_norm(self):
        locs = _make_2d_grid(8, 8)
        phi = _gaussian_bump_2d(locs)
        assert np.linalg.norm(phi) == pytest.approx(1.0, abs=1e-10)

    def test_bump_peak_at_centre(self):
        locs = _make_2d_grid(16, 16)
        phi = _gaussian_bump_2d(locs, centre=(0.5, 0.5))
        # Centre patch (row~8, col~8) should have the highest value
        centre_idx = 8 * 16 + 8  # approximate
        assert phi[centre_idx] == phi.max() or abs(phi[centre_idx] - phi.max()) < 0.01


# ---------------------------------------------------------------------------
# get_synthetic_patch_dataloader_and_val
# ---------------------------------------------------------------------------


class TestSyntheticPatchBinary:
    def test_shapes(self):
        (
            loader,
            val_cont,
            val_y,
            locs,
            true_phi,
            tp,
            vp,
        ) = get_synthetic_patch_dataloader_and_val(
            n_images=50,
            grid_h=4,
            grid_w=4,
            feature_dim=16,
            n_basis=2,
            train_ratio=0.8,
        )
        N = 4 * 4
        assert val_cont.shape == (10, N, 16)
        assert val_y.shape == (10, N)
        assert locs.shape == (N, 2)
        assert true_phi.shape == (N, 2)
        assert tp.shape[0] == 40  # train_prob
        assert vp.shape[0] == 10  # val_prob

    def test_binary_labels(self):
        _, _, val_y, *_ = get_synthetic_patch_dataloader_and_val()
        unique = set(val_y.numpy().ravel().tolist())
        assert unique <= {0.0, 1.0}

    def test_true_phi_unit_columns(self):
        _, _, _, _, true_phi, *_ = get_synthetic_patch_dataloader_and_val(n_basis=2)
        for k in range(true_phi.shape[1]):
            assert np.linalg.norm(true_phi[:, k]) == pytest.approx(1.0, abs=1e-10)

    def test_deterministic(self):
        _, c1, y1, *_ = get_synthetic_patch_dataloader_and_val(seed=7)
        _, c2, y2, *_ = get_synthetic_patch_dataloader_and_val(seed=7)
        assert torch.equal(c1, c2)
        assert torch.equal(y1, y2)

    def test_different_seeds(self):
        _, _, y1, *_ = get_synthetic_patch_dataloader_and_val(seed=0)
        _, _, y2, *_ = get_synthetic_patch_dataloader_and_val(seed=1)
        assert not torch.equal(y1, y2)

    def test_single_basis(self):
        _, _, _, _, true_phi, *_ = get_synthetic_patch_dataloader_and_val(n_basis=1)
        assert true_phi.shape[1] == 1

    def test_prob_in_01(self):
        _, _, _, _, _, tp, vp = get_synthetic_patch_dataloader_and_val()
        assert tp.min() >= 0.0 and tp.max() <= 1.0
        assert vp.min() >= 0.0 and vp.max() <= 1.0
