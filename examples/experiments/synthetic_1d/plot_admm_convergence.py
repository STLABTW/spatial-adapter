#!/usr/bin/env python
"""
Plot ADMM primal/dual residual convergence traces for the synthetic 1D benchmark.

Produces the convergence diagnostic figure referenced in the paper's
Limitations discussion (Section 5, item (ii)) and Appendix.

Usage:
    python examples/experiments/synthetic_1d/plot_admm_convergence.py

Output:
    .local/admm_convergence_synthetic.pdf
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from spatial_adapter.data.generators import generate_time_synthetic_data
from spatial_adapter.data.preprocessing import prepare_all_with_scaling
from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialNeuralAdapter,
    SpatialNeuralAdapterConfig,
    TrainingConfig,
)
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel
from spatial_adapter.utils import compute_ols_coefficients

# Configuration
SEED = 42
N_LOCATIONS = 512
N_TIME_STEPS = 1024
NOISE_STD = 4.0
EIGENVALUE = 16.0
LATENT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path(".local")


@dataclass
class ResidualLogger:
    """Lightweight container that accumulates primal/dual residuals."""

    primal: list[float]
    dual: list[float]

    def __init__(self) -> None:
        self.primal = []
        self.dual = []


def _monkey_patch_run(adapter: SpatialNeuralAdapter, logger: ResidualLogger) -> float:
    """
    Run the ADMM loop with residual logging, mirroring SpatialNeuralAdapter.run()
    but appending primal/dual norms to *logger* at each iteration.
    """
    from tqdm import trange

    tol = float(adapter.config.admm.tol)
    min_outer = int(adapter.config.admm.min_outer)
    outer = trange(1, adapter.max_iters + 1, desc="ADMM", dynamic_ncols=True)

    for _ in outer:
        z_prev = adapter.z_train.clone()

        # Sample mini-batch
        batch_size = int(adapter.config.training.batch_size)
        T = adapter.train_y.size(0)
        if batch_size >= T:
            batch_indices = None
        else:
            max_start = T - batch_size
            t0 = int(
                torch.randint(0, max_start + 1, (1,), device=adapter.device).item()
            )
            batch_indices = slice(t0, t0 + batch_size)

        # ADMM steps
        delta_phi = adapter._phi_step(batch_indices)
        adapter._theta_step(batch_indices)
        adapter._z_step(batch_indices, val=False)
        adapter._z_step(batch_indices, val=True)

        # Compute residuals
        r_pri = adapter._residual_R(adapter.train_y, adapter.train_cont)
        pri = F.mse_loss(adapter.z_train, r_pri).sqrt().item()
        dua = (adapter.rho * (adapter.z_train - z_prev)).pow(2).mean().sqrt().item()

        logger.primal.append(pri)
        logger.dual.append(dua)

        res = max(delta_phi, pri)
        outer.set_postfix(pri=f"{pri:.2e}", dua=f"{dua:.2e}")

        # Validation (for early stopping bookkeeping)
        val_result = adapter._validate()
        primary = val_result[0]
        adapter.best_val = min(adapter.best_val, primary)

        if (adapter.global_iter + 1) >= min_outer and res < tol:
            outer.write(f"Converged (res={res:.2e} < ε={tol})")
            break
        adapter.global_iter += 1

    return adapter.best_val


def run_and_log(tau1: float, tau2: float, label: str) -> ResidualLogger:
    """Train one model and return its residual traces."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data
    locs = np.linspace(-3, 3, N_LOCATIONS)
    cat_features, cont_features, targets = generate_time_synthetic_data(
        locs=locs,
        n_time_steps=N_TIME_STEPS,
        noise_std=NOISE_STD,
        eigenvalue=EIGENVALUE,
        eta_rho=0.8,
        f_rho=0.6,
        global_mean=50.0,
        feature_noise_std=0.01,
        seed=SEED,
    )
    train_ds, val_ds, _, _ = prepare_all_with_scaling(
        cat_features=cat_features,
        cont_features=cont_features,
        targets=targets,
        train_ratio=0.7,
        val_ratio=0.15,
        feature_scaler_type="standard",
        target_scaler_type="standard",
        fit_on_train_only=True,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    _, train_X, train_y = train_ds.tensors
    _, val_X, val_y = val_ds.tensors
    p_dim = train_X.shape[-1]

    # OLS warm-start
    w_ols, b_ols = compute_ols_coefficients(train_X, train_y, device=DEVICE)

    # Models
    trend = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=[],
        n_locations=N_LOCATIONS,
        init_weight=w_ols,
        init_bias=b_ols,
        freeze_init=True,
        dropout_rate=0.0,
    ).to(DEVICE)
    basis = SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(DEVICE)

    config = SpatialNeuralAdapterConfig(
        admm=ADMMConfig(
            rho=1.0, dual_momentum=0.2, max_iters=3000, min_outer=20, tol=1e-4
        ),
        training=TrainingConfig(lr_mu=1e-2, batch_size=64, pretrain_epochs=5),
        basis=BasisConfig(phi_every=5, phi_freeze=200),
    )

    adapter = SpatialNeuralAdapter(
        trend,
        basis,
        train_loader,
        val_cont=val_X.to(DEVICE),
        val_y=val_y.to(DEVICE),
        locs=locs,
        config=config,
        device=DEVICE,
        writer=None,
        tau1=tau1,
        tau2=tau2,
    )
    adapter.pretrain_trend(epochs=5)
    adapter.init_basis_dense()

    # Run with logging
    res_logger = ResidualLogger()
    print(f"\n{'='*60}")
    print(f"Running: {label}  (tau1={tau1}, tau2={tau2})")
    print(f"{'='*60}")
    _monkey_patch_run(adapter, res_logger)

    return res_logger


def plot_convergence(
    traces: dict[str, ResidualLogger],
    output_path: Path,
    phi_freeze: int = 200,
) -> None:
    """Plot primal and dual residual traces with basis-freeze marker."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    colors = {"Adapter (unreg.)": "C0", "Adapter (reg.)": "C1"}

    for label, tr in traces.items():
        c = colors.get(label, None)
        iters = np.arange(1, len(tr.primal) + 1)
        axes[0].semilogy(iters, tr.primal, label=label, linewidth=1.2, color=c)
        # Dual: raw trace (faint) + rolling median (solid)
        axes[1].semilogy(iters, tr.dual, alpha=0.15, linewidth=0.5, color=c)
        win = min(50, len(tr.dual))
        dual_smooth = (
            pd.Series(tr.dual).rolling(win, center=True, min_periods=1).median()
        )
        axes[1].semilogy(iters, dual_smooth, label=label, linewidth=1.5, color=c)

    # Basis-freeze vertical line
    for ax in axes:
        ax.axvline(
            phi_freeze,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=rf"basis freeze ($N_{{\mathrm{{freeze}}}}={phi_freeze}$)",
        )

    axes[0].set_xlabel("ADMM sweep")
    axes[0].set_ylabel(r"Primal residual $\|Z - R(\psi)\|_F$")
    axes[0].set_title("(a) Primal residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("ADMM sweep")
    axes[1].set_ylabel(r"Dual residual $\rho\|Z^{(k)}-Z^{(k-1)}\|_F$")
    axes[1].set_title("(b) Dual residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    print(f"\nFigure saved to {output_path}")
    plt.close(fig)


def main() -> None:
    traces = {}

    # Unregularized
    traces["Adapter (unreg.)"] = run_and_log(
        tau1=0.0, tau2=0.0, label="Adapter (unreg.)"
    )

    # Regularized — use moderate regularization typical of the synthetic benchmark
    traces["Adapter (reg.)"] = run_and_log(tau1=1e2, tau2=1e2, label="Adapter (reg.)")

    output_path = OUTPUT_DIR / "admm_convergence_synthetic.pdf"
    plot_convergence(traces, output_path)


if __name__ == "__main__":
    main()
