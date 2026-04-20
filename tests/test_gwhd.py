"""
Unit tests for the GWHD data loader and the frozen-trend fix.

The GWHD loader requires torchvision, which is optional. Tests that
need it are skipped when torchvision is not installed.  The
frozen-trend test exercises the model fix directly and needs only
torch.
"""


import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spatial_adapter.data import get_gwhd_dataloader_and_val

# ---------------------------------------------------------------------------
# GWHD lazy import
# ---------------------------------------------------------------------------


class TestGwhdLazyImport:
    """Verify that importing spatial_adapter.data never crashes,
    even when torchvision is absent."""

    def test_import_succeeds(self):
        """The package-level import must always succeed."""
        import spatial_adapter.data  # should never raise

    def test_fallback_is_none_or_callable(self):
        """get_gwhd_dataloader_and_val is either callable (torchvision
        present) or None (torchvision absent)."""
        assert get_gwhd_dataloader_and_val is None or callable(
            get_gwhd_dataloader_and_val
        )


# ---------------------------------------------------------------------------
# Frozen-trend fix in SpatialNeuralAdapter
# ---------------------------------------------------------------------------


class TestFrozenTrendFix:
    """Verify that the adapter handles a fully-frozen trend gracefully."""

    @pytest.fixture
    def frozen_trend_setup(self):
        """Create a minimal adapter with a frozen (zero-param) trend."""
        from spatial_adapter.data import generate_combined_synthetic_data
        from spatial_adapter.models.spatial_basis_learner import (
            SpatialBasisLearner,
        )
        from spatial_adapter.models.spatial_adapter import (
            SpatialNeuralAdapterConfig,
        )
        from spatial_adapter.models.trend_model import TrendModel

        np.random.seed(0)
        torch.manual_seed(0)
        N, T, K = 10, 50, 2
        locs = np.linspace(-3, 3, N)

        cat, cont, targets = generate_combined_synthetic_data(
            location=locs,
            n_samples=T,
            noise_std=0.1,
            eigenvalue=4.0,
            global_mean=50.0,
            seed=0,
        )
        cont_t = torch.from_numpy(cont).float()
        targets_t = torch.from_numpy(targets).float()
        dataset = TensorDataset(
            torch.zeros(T, dtype=torch.long),
            cont_t,
            targets_t,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        p_dim = cont.shape[-1]
        # Trend with freeze_init=True AND no residual blocks → zero trainable params.
        # Must pass valid init_weight/init_bias for freeze_init to take effect.
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[],
            n_locations=N,
            init_weight=torch.zeros(p_dim),
            init_bias=0.0,
            freeze_init=True,
            dropout_rate=0.0,
        )
        basis = SpatialBasisLearner(num_locations=N, latent_dim=K)

        config = SpatialNeuralAdapterConfig.from_dict(
            {
                "task": "regression",
                "rho": 1.0,
                "max_iters": 5,
                "min_outer": 2,
                "batch_size": 16,
                "lr_mu": 1e-3,
                "pretrain_epochs": 0,
                "phi_every": 2,
                "phi_freeze": 3,
            }
        )

        return trend, basis, loader, cont_t, targets_t, locs, config

    def test_opt_mu_is_none(self, frozen_trend_setup):
        """When trend has no trainable params, opt_mu should be None."""
        from spatial_adapter.models.spatial_adapter import (
            SpatialNeuralAdapter,
        )

        trend, basis, loader, cont, targets, locs, config = frozen_trend_setup

        from unittest.mock import Mock

        from torch.utils.tensorboard import SummaryWriter

        adapter = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=loader,
            val_cont=cont[:10],
            val_y=targets[:10],
            locs=locs,
            config=config,
            device=torch.device("cpu"),
            writer=Mock(spec=SummaryWriter),
        )
        assert adapter.opt_mu is None

    def test_theta_step_no_crash(self, frozen_trend_setup):
        """_theta_step should return immediately without error."""
        from spatial_adapter.models.spatial_adapter import (
            SpatialNeuralAdapter,
        )

        trend, basis, loader, cont, targets, locs, config = frozen_trend_setup

        from unittest.mock import Mock

        from torch.utils.tensorboard import SummaryWriter

        adapter = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=loader,
            val_cont=cont[:10],
            val_y=targets[:10],
            locs=locs,
            config=config,
            device=torch.device("cpu"),
            writer=Mock(spec=SummaryWriter),
        )
        # This would crash before the fix (empty param list → AdamW error)
        adapter._theta_step()  # should be a no-op, no exception

    def test_run_completes(self, frozen_trend_setup):
        """Full adapter.run() should complete without error."""
        from spatial_adapter.models.spatial_adapter import (
            SpatialNeuralAdapter,
        )

        trend, basis, loader, cont, targets, locs, config = frozen_trend_setup

        from unittest.mock import Mock

        from torch.utils.tensorboard import SummaryWriter

        adapter = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=loader,
            val_cont=cont[:10],
            val_y=targets[:10],
            locs=locs,
            config=config,
            device=torch.device("cpu"),
            writer=Mock(spec=SummaryWriter),
        )
        result = adapter.run()
        assert isinstance(result, float)
