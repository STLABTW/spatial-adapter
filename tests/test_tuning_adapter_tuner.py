"""Unit tests for spatial_adapter.tuning.adapter_tuner.AdapterTuner."""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialAdapterConfig,
    TrainingConfig,
)
from spatial_adapter.tuning import AdapterTuner


class _SimpleTrend(nn.Module):
    """Minimal regression trend with a trainable residual MLP."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 8), nn.GELU(), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, p = x.shape
        return self.net(x.reshape(-1, p)).view(B, N)

    def residual_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def _make_synthetic_regression(T=40, N=16, p=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(T, N, p)
    W = torch.randn(p, 1)
    Y = (X @ W).squeeze(-1) + 0.1 * torch.randn(T, N)
    idx = torch.arange(T // 2)
    train_ds = TensorDataset(idx, X[: T // 2], Y[: T // 2])
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    xs, ys = np.meshgrid(np.arange(4), np.arange(4))
    locs = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    return {
        "train_loader": loader,
        "train_cont": X[: T // 2],
        "train_y": Y[: T // 2],
        "val_cont": X[T // 2 : 3 * T // 4],
        "val_y": Y[T // 2 : 3 * T // 4],
        "test_cont": X[3 * T // 4 :],
        "test_y": Y[3 * T // 4 :],
        "locs": locs,
        "p_dim": p,
        "n_locations": N,
    }


def _make_config(max_iters=4, pretrain_epochs=1) -> SpatialAdapterConfig:
    return SpatialAdapterConfig(
        admm=ADMMConfig(
            rho=5.0, dual_momentum=0.2, max_iters=max_iters, min_outer=1, tol=1e-4
        ),
        training=TrainingConfig(
            lr_mu=1e-3, batch_size=8, pretrain_epochs=pretrain_epochs
        ),
        basis=BasisConfig(phi_every=2, phi_freeze=3, matrix_reg=1e-6),
        task="regression",
    )


def _rmse_eval(trainer):
    with torch.no_grad():
        mu = trainer.trend(trainer.train_cont)
        R = trainer.train_y - mu
        spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
        rmse = ((trainer.train_y - (mu + spatial)) ** 2).mean().sqrt().item()
    return {"rmse": rmse}


class TestAdapterTuner:
    def test_fit_one_returns_metrics_and_trainer(self):
        data = _make_synthetic_regression()
        tuner = AdapterTuner(
            trend_template=_SimpleTrend(data["p_dim"]),
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(),
            evaluate_fn=_rmse_eval,
            n_trials=1,
        )
        trainer, metrics = tuner.fit_one(tau1=0.1, tau2=0.1)
        assert "rmse" in metrics
        assert metrics["tau1"] == 0.1
        assert metrics["tau2"] == 0.1
        assert "elapsed_s" in metrics
        assert trainer is not None

    def test_template_not_mutated(self):
        data = _make_synthetic_regression()
        template = _SimpleTrend(data["p_dim"])
        snapshot = copy.deepcopy(template.state_dict())

        tuner = AdapterTuner(
            trend_template=template,
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(),
            evaluate_fn=_rmse_eval,
        )
        tuner.fit_one(tau1=0.1, tau2=0.1)

        for k, v in template.state_dict().items():
            assert torch.equal(v, snapshot[k]), f"template param {k} was mutated"

    def test_warm_start_populates_cache(self):
        data = _make_synthetic_regression()
        tuner = AdapterTuner(
            trend_template=_SimpleTrend(data["p_dim"]),
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(),
            evaluate_fn=_rmse_eval,
        )
        assert tuner.cache.size() == 0
        tuner.fit_one(tau1=1.0, tau2=1.0, warm_start=True)
        assert tuner.cache.size() == 1
        tuner.fit_one(tau1=2.0, tau2=2.0, warm_start=True)
        assert tuner.cache.size() == 2

    def test_stage2_pretrain_false_skips_pretrain(self):
        """When stage2_pretrain=False, pretrain_trend must not be called."""
        data = _make_synthetic_regression()
        calls = {"n": 0}

        class _Trend(_SimpleTrend):
            pass

        tuner = AdapterTuner(
            trend_template=_Trend(data["p_dim"]),
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(pretrain_epochs=5),
            evaluate_fn=_rmse_eval,
            stage2_pretrain=False,
        )

        # Patch pretrain_trend via monkey-patching on the instance class
        import spatial_adapter.models.spatial_adapter as mod

        orig = mod.SpatialAdapter.pretrain_trend

        def _counted(self, *a, **kw):
            calls["n"] += 1
            return orig(self, *a, **kw)

        mod.SpatialAdapter.pretrain_trend = _counted
        try:
            tuner.fit_one(tau1=0.1, tau2=0.1)
        finally:
            mod.SpatialAdapter.pretrain_trend = orig

        assert (
            calls["n"] == 0
        ), "pretrain_trend must not be called when stage2_pretrain=False"

    def test_run_optuna_returns_best(self):
        data = _make_synthetic_regression()
        tuner = AdapterTuner(
            trend_template=_SimpleTrend(data["p_dim"]),
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(),
            evaluate_fn=_rmse_eval,
            n_trials=3,
            objective_key="rmse",
            direction="minimize",
            tau_range=(1e-3, 1e1),
        )
        best = tuner.run_optuna(seed=42)
        assert "rmse" in best
        assert "tau1" in best and "tau2" in best
        assert tuner.cache.size() == 3  # one entry per trial

    def test_run_optuna_clears_cache_on_start(self):
        data = _make_synthetic_regression()
        tuner = AdapterTuner(
            trend_template=_SimpleTrend(data["p_dim"]),
            train_loader=data["train_loader"],
            val_cont=data["val_cont"],
            val_y=data["val_y"],
            locs=data["locs"],
            n_locations=data["n_locations"],
            latent_dim=2,
            adapter_config=_make_config(),
            evaluate_fn=_rmse_eval,
            n_trials=2,
        )
        # Pre-populate cache
        tuner.fit_one(tau1=999.0, tau2=999.0, warm_start=True)
        assert tuner.cache.size() == 1

        tuner.run_optuna(seed=1)
        # After run_optuna: only the 2 Optuna trials, not the pre-seeded entry
        keys = tuner.cache.keys()
        assert (999.0, 999.0) not in keys
        assert len(keys) == 2
