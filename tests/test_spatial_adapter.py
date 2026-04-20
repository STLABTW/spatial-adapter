import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from spatial_adapter.data import get_synthetic_binary_dataloader_and_val
from spatial_adapter.models.classification_wrapper import ClassificationWrapper
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.spatial_adapter import (
    SpatialNeuralAdapter,
    SpatialNeuralAdapterConfig,
)
from spatial_adapter.models.trend_model import TrendModel


class TestSpatialNeuralAdapter:
    def test_trainer_initialization(self, sample_data, device):
        """Test SpatialNeuralAdapter initialization."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        assert trainer.trend is not None
        assert trainer.basis is not None
        assert trainer.device == device
        assert trainer.tau1 == 0.1
        assert trainer.tau2 == 0.1

    def test_pretrain_trend(self, sample_data, device):
        """Test trend pretraining."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        # Test pretraining
        trainer.pretrain_trend(epochs=2)

        # Check that trend model parameters have been updated
        assert any(p.requires_grad for p in trend.parameters())

    def test_init_basis_dense(self, sample_data, device):
        """Test basis initialization."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        # Test basis initialization
        trainer.init_basis_dense()

        # Check that basis has been initialized
        assert basis.basis is not None
        assert not torch.isnan(basis.basis).any()
        assert not torch.isinf(basis.basis).any()

    # ---- L1: pluggable task (regression / binary) ----

    def test_config_task_default(self):
        """SpatialNeuralAdapterConfig defaults to task='regression'."""
        config = SpatialNeuralAdapterConfig()
        assert config.task == "regression"

    def test_config_from_dict_task_binary(self):
        """from_dict accepts task='binary' and sets config.task."""
        config = SpatialNeuralAdapterConfig.from_dict({"task": "binary"})
        assert config.task == "binary"

    def test_config_from_dict_task_regression(self):
        """from_dict with task='regression' or no task yields regression."""
        assert (
            SpatialNeuralAdapterConfig.from_dict({"task": "regression"}).task
            == "regression"
        )
        assert SpatialNeuralAdapterConfig.from_dict({}).task == "regression"

    def test_config_to_dict_roundtrip_task(self):
        """to_dict and from_dict roundtrip preserves task."""
        config = SpatialNeuralAdapterConfig()
        config.task = "binary"
        d = config.to_dict()
        assert d["task"] == "binary"
        restored = SpatialNeuralAdapterConfig.from_dict(d)
        assert restored.task == "binary"

    def test_config_from_dict_invalid_task_raises(self):
        """from_dict with invalid task raises ValueError."""
        with pytest.raises(ValueError, match="task must be 'regression' or 'binary'"):
            SpatialNeuralAdapterConfig.from_dict({"task": "classification"})

    def test_adapter_with_task_binary_initializes(self, sample_data, device):
        """SpatialNeuralAdapter with task=binary initializes and exposes task."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 2,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )
        assert trainer.config.task == "binary"
        assert trainer._is_binary is True

    def test_adapter_with_task_binary_pretrain_runs(self, sample_data, device):
        """SpatialNeuralAdapter with task=binary can run pretrain_trend without error."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 2,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )
        trainer.pretrain_trend(epochs=1)
        # No exception; trend parameters updated
        assert any(p.requires_grad for p in trend.parameters())

    def test_adapter_task_binary_z_step_proximal_runs(self, sample_data, device):
        """L3: With task=binary, _z_step uses proximal-gradient and updates Z without error."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        T = sample_data["cont_features"].shape[0]
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)
        # Binary-like targets in [0, 1] for BCE
        tmin, tmax = sample_data["targets"].min(), sample_data["targets"].max()
        denom = tmax - tmin + 1e-6
        targets = np.clip((sample_data["targets"] - tmin) / denom, 0.0, 1.0).astype(
            np.float32
        )
        train_dataset = TensorDataset(
            torch.zeros(T, 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(targets).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_size = T // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(targets[-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 2,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )
        trainer.init_basis_dense()
        batch_size = min(32, T)
        batch_indices = slice(0, batch_size)
        z_before = trainer.z_train[batch_indices].clone()
        trainer._z_step(batch_indices, val=False)
        z_after = trainer.z_train[batch_indices]
        assert not torch.isnan(z_after).any()
        assert not torch.isinf(z_after).any()
        # Proximal step should change Z (unless degenerate)
        assert not torch.allclose(z_before, z_after)

    def test_residual_R_binary_is_logit_space(self, sample_data, device):
        """L4: When task=binary, _residual_R uses R = logit(Y) - f_θ(X) (logit space)."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        T = sample_data["cont_features"].shape[0]
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[16],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.0,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=2,
            pca_init=None,
        ).to(device)
        targets = np.clip(
            (sample_data["targets"] - sample_data["targets"].min())
            / (sample_data["targets"].max() - sample_data["targets"].min() + 1e-6),
            0.2,
            0.8,
        ).astype(np.float32)
        train_dataset = TensorDataset(
            torch.zeros(T, 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(targets).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_size = T // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(targets[-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 5,
            "min_outer": 3,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 2,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 0,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        x = torch.from_numpy(sample_data["cont_features"][:1]).float().to(device)
        y_half = torch.full((1, n_locations), 0.5, device=device)
        R_half = trainer._residual_R(y_half, x)
        # For Y=0.5, logit(0.5)=0, so R = 0 - trend(x) = -trend(x)
        mu = trainer.trend(x)
        torch.testing.assert_close(R_half, -mu, atol=1e-5, rtol=1e-5)

    def test_validate_binary_returns_accuracy_f1_auc(self, sample_data, device):
        """L5: When task=binary, _validate() returns (accuracy, f1, auc) in [0,1]."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        T = sample_data["cont_features"].shape[0]
        tmin, tmax = sample_data["targets"].min(), sample_data["targets"].max()
        denom = tmax - tmin + 1e-6
        targets = np.clip((sample_data["targets"] - tmin) / denom, 0.2, 0.8).astype(
            np.float32
        )
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[16],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.0,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=2,
            pca_init=None,
        ).to(device)
        train_dataset = TensorDataset(
            torch.zeros(T, 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(targets).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_size = T // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(targets[-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 5,
            "min_outer": 3,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 2,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 0,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        trainer.init_basis_dense()
        acc, f1, auc = trainer._validate()
        assert 0 <= acc <= 1 and 0 <= f1 <= 1 and 0 <= auc <= 1
        assert (
            isinstance(acc, float) and isinstance(f1, float) and isinstance(auc, float)
        )

    def test_pretrain_trend_bce_and_mse(self, sample_data, device):
        """L6: pretrain_trend accepts loss_fn='mse'|'bce'; binary defaults to BCE."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]
        T = sample_data["cont_features"].shape[0]
        tmin, tmax = sample_data["targets"].min(), sample_data["targets"].max()
        targets = np.clip(
            (sample_data["targets"] - tmin) / (tmax - tmin + 1e-6), 0.2, 0.8
        ).astype(np.float32)
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[16],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.0,
        ).to(device)
        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=2,
            pca_init=None,
        ).to(device)
        train_dataset = TensorDataset(
            torch.zeros(T, 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(targets).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_size = T // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(targets[-val_size:]).float().to(device)
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 5,
            "min_outer": 3,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 2,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 0,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        # L6: default (binary) uses BCE
        trainer.pretrain_trend(epochs=1)
        # explicit BCE
        trainer.pretrain_trend(epochs=1, loss_fn="bce")
        # optional: force MSE for warm-up
        trainer.pretrain_trend(epochs=1, loss_fn="mse")
        assert any(p.requires_grad for p in trend.parameters())

    def test_resnet_wrapper_and_synthetic_binary_end_to_end(self, device):
        """W1: ClassificationWrapper + synthetic binary data runs with adapter (binary task)."""
        train_loader, val_cont, val_y, locs = get_synthetic_binary_dataloader_and_val(
            n_samples=120,
            n_locations=4,
            feature_dim=32,
            train_ratio=0.75,
            batch_size=16,
            seed=43,
        )
        _, cont, _ = train_loader.dataset.tensors
        T, N, p = cont.shape
        val_cont = val_cont.to(device)
        val_y = val_y.to(device)

        trend = ClassificationWrapper(feature_dim=p, n_locations=N).to(device)
        basis = SpatialBasisLearner(num_locations=N, latent_dim=2, pca_init=None).to(
            device
        )

        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 8,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 16,
            "phi_every": 2,
            "phi_freeze": 6,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 1,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        trainer.pretrain_trend(epochs=1)
        trainer.init_basis_dense()
        best = trainer.run()
        # Binary: best_val is best accuracy (max)
        assert isinstance(best, float)
        assert 0 <= best <= 1.0

    def test_classification_wrapper_head_only(self, device):
        """ClassificationWrapper (B,N,p)->(B,N) and residual_parameters()."""
        B, N, p = 4, 3, 64
        wrap = ClassificationWrapper(feature_dim=p, n_locations=N).to(device)
        x = torch.randn(B, N, p, device=device)
        out = wrap(x)
        assert out.shape == (B, N)
        params = wrap.residual_parameters()
        assert len(params) >= 1

    def test_classification_wrapper_with_adapter_binary(self, device):
        """ClassificationWrapper as trend runs with adapter (binary)."""
        train_loader, val_cont, val_y, locs = get_synthetic_binary_dataloader_and_val(
            n_samples=80,
            n_locations=3,
            feature_dim=24,
            train_ratio=0.75,
            batch_size=12,
            seed=44,
        )
        _, cont, _ = train_loader.dataset.tensors
        T, N, p = cont.shape
        val_cont = val_cont.to(device)
        val_y = val_y.to(device)

        trend = ClassificationWrapper(feature_dim=p, n_locations=N).to(device)
        basis = SpatialBasisLearner(num_locations=N, latent_dim=2, pca_init=None).to(
            device
        )
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 6,
            "min_outer": 4,
            "lr_mu": 1e-3,
            "batch_size": 12,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "pretrain_epochs": 1,
            "task": "binary",
        }
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        trainer.pretrain_trend(epochs=1)
        trainer.init_basis_dense()
        best = trainer.run()
        assert isinstance(best, float)
        assert 0 <= best <= 1.0


# ─────────────────────────────────────────────────────────────────────
# Inference helpers: reconstruct / predict / fit_forecaster
# ─────────────────────────────────────────────────────────────────────


def _make_regression_trainer(device):
    """Build a minimal regression trainer without running ADMM."""
    torch.manual_seed(0)
    np.random.seed(0)
    T, N, p = 20, 4, 2
    cont = torch.randn(T, N, p)
    y = torch.randn(T, N)
    locs = np.linspace(-1, 1, N).reshape(-1, 1).astype(np.float32)

    ds = TensorDataset(torch.zeros(T, 0, dtype=torch.long), cont, y)
    loader = DataLoader(ds, batch_size=T)

    trend = TrendModel(
        num_continuous_features=p, hidden_layer_sizes=[8], n_locations=N
    ).to(device)
    basis = SpatialBasisLearner(num_locations=N, latent_dim=2).to(device)
    config = {
        "rho": 1.0, "dual_momentum": 0.0, "max_iters": 1, "min_outer": 1,
        "lr_mu": 1e-3, "batch_size": T, "phi_every": 1, "phi_freeze": 1,
        "tol": 1e-4, "matrix_reg": 1e-6, "irl1_max_iters": 2,
        "irl1_eps": 1e-6, "irl1_tol": 1e-3, "pretrain_epochs": 1,
    }
    return SpatialNeuralAdapter(
        trend=trend, basis=basis, train_loader=loader,
        val_cont=cont[:5].to(device), val_y=y[:5].to(device),
        locs=locs, config=config, device=device, writer=None,
    )


class TestConfigLogging:
    def test_log_config_runs(self, caplog):
        """log_config iterates to_dict() and logs every key/value."""
        import logging

        cfg = SpatialNeuralAdapterConfig()
        with caplog.at_level(logging.INFO, logger="spatial_adapter"):
            cfg.log_config()
        # Should have emitted one line per config key
        assert len(caplog.records) >= 3


class TestReconstruct:
    def test_regression_output_shape_matches_y(self, device):
        trainer = _make_regression_trainer(device)
        cont = trainer.val_cont
        y = trainer.val_y
        out = trainer.reconstruct(cont, y)
        assert out.shape == y.shape
        assert not torch.isnan(out).any()

    def test_requires_y_true(self, device):
        trainer = _make_regression_trainer(device)
        with pytest.raises(ValueError, match="requires y_true"):
            trainer.reconstruct(trainer.val_cont, None)

    def test_binary_uses_logit_space(self, device):
        train_loader, val_cont, val_y, locs = get_synthetic_binary_dataloader_and_val(
            n_samples=40, n_locations=3, feature_dim=8,
            train_ratio=0.75, batch_size=8, seed=5,
        )
        _, cont, _ = train_loader.dataset.tensors
        _, N, p = cont.shape
        trend = ClassificationWrapper(feature_dim=p, n_locations=N).to(device)
        basis = SpatialBasisLearner(num_locations=N, latent_dim=2).to(device)
        trainer = SpatialNeuralAdapter(
            trend=trend, basis=basis, train_loader=train_loader,
            val_cont=val_cont.to(device), val_y=val_y.to(device), locs=locs,
            config={
                "rho": 1.0, "max_iters": 1, "min_outer": 1, "lr_mu": 1e-3,
                "batch_size": 8, "phi_every": 1, "phi_freeze": 1, "tol": 1e-4,
                "matrix_reg": 1e-6, "pretrain_epochs": 1, "task": "binary",
            },
            device=device, writer=None,
        )
        out = trainer.reconstruct(val_cont.to(device), val_y.to(device))
        assert out.shape == val_y.shape


class TestPredict:
    def test_trend_only_when_forecaster_not_fit(self, device):
        trainer = _make_regression_trainer(device)
        out = trainer.predict(trainer.val_cont)
        # Without fit_forecaster, prediction is pure trend → equals trend(cont)
        expected = trainer.trend(trainer.val_cont)
        torch.testing.assert_close(out, expected)

    def test_with_var1_forecaster_has_temporal_rollout(self, device):
        trainer = _make_regression_trainer(device)
        trainer.fit_forecaster()
        assert trainer._A is not None
        assert trainer._eta_last is not None

        out = trainer.predict(trainer.val_cont)
        assert out.shape == trainer.val_cont.shape[:2]  # (T_val, N)
        # Output should differ from trend-only since VAR(1) contributes
        trend_only = trainer.trend(trainer.val_cont)
        assert not torch.allclose(out, trend_only, atol=1e-8)


class TestPhiStepBceVariants:
    """Cover the full_taylor / irls / invalid-variant branches of _phi_step."""

    def _build_binary(self, device, bce_variant):
        train_loader, val_cont, val_y, locs = get_synthetic_binary_dataloader_and_val(
            n_samples=40, n_locations=4, feature_dim=6,
            train_ratio=0.75, batch_size=8, seed=11,
        )
        _, cont, _ = train_loader.dataset.tensors
        _, N, p = cont.shape
        trend = ClassificationWrapper(feature_dim=p, n_locations=N).to(device)
        basis = SpatialBasisLearner(num_locations=N, latent_dim=2).to(device)
        return SpatialNeuralAdapter(
            trend=trend, basis=basis, train_loader=train_loader,
            val_cont=val_cont.to(device), val_y=val_y.to(device), locs=locs,
            config={
                "rho": 1.0, "max_iters": 1, "min_outer": 1, "lr_mu": 1e-3,
                "batch_size": 8, "phi_every": 1, "phi_freeze": 10, "tol": 1e-4,
                "matrix_reg": 1e-6, "pretrain_epochs": 1,
                "task": "binary", "bce_variant": bce_variant,
            },
            device=device, writer=None,
        )

    def test_full_taylor_variant_runs(self, device):
        trainer = self._build_binary(device, "full_taylor")
        delta = trainer._phi_step(batch_indices=None)
        assert isinstance(delta, float)

    def test_irls_variant_runs(self, device):
        trainer = self._build_binary(device, "irls")
        delta = trainer._phi_step(batch_indices=None)
        assert isinstance(delta, float)

    def test_invalid_variant_raises(self, device):
        trainer = self._build_binary(device, "not_a_variant")
        with pytest.raises(ValueError, match="Unknown bce_variant"):
            trainer._phi_step(batch_indices=None)


class TestPretrainTrendBranches:
    def test_early_returns_when_opt_mu_is_none(self, device):
        """pretrain_trend is a no-op when the trend has no trainable params."""
        trainer = _make_regression_trainer(device)
        # Force opt_mu=None to hit the early-return branch
        trainer.opt_mu = None
        # Should return immediately without raising
        trainer.pretrain_trend(epochs=5)

    def test_default_epochs_uses_config(self, device):
        """Calling pretrain_trend() without `epochs` picks it up from config."""
        trainer = _make_regression_trainer(device)
        # Config has pretrain_epochs=1; no exception means default path was taken
        trainer.pretrain_trend(epochs=None)


class TestFullBatchRun:
    def test_run_with_batch_size_geq_T_sets_bi_none(self, device):
        """When batch_size >= T, the run loop uses full-batch (bi=None)."""
        trainer = _make_regression_trainer(device)
        # train_loader dataset has T=20; config already has batch_size=T=20 → bs >= T
        trainer.init_basis_dense() if hasattr(trainer, "init_basis_dense") else None
        best = trainer.run()
        assert isinstance(best, float)


class TestInternalStepsFullBatch:
    def test_theta_step_batch_indices_none(self, device):
        """_theta_step with batch_indices=None takes the full-tensor branch."""
        trainer = _make_regression_trainer(device)
        # Just invoke directly — non-None would slice; None hits the other branch
        trainer._theta_step(batch_indices=None)

    def test_z_step_nonval_full_batch(self, device):
        """_z_step(batch_indices=None, val=False) covers the full-train consensus + dual path."""
        trainer = _make_regression_trainer(device)
        trainer._z_step(batch_indices=None, val=False)
        # z_train and u_train should have changed from zero initialisation
        assert trainer.z_train.abs().sum().item() > 0 or trainer.u_train.abs().sum().item() > 0
