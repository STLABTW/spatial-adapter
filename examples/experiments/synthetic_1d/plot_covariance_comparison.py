#!/usr/bin/env python
"""Emit a 4-panel covariance comparison figure for the main-body synthetic box.

Panels (shared colormap):
    True Σ  |  OLS residual sample Σ̂  |  Adapter (unreg.)  |  Adapter (reg.)

`True Σ` is the analytic population covariance implied by the rank-1 DGP.
`OLS residual sample Σ̂` is the sample covariance of residuals from the frozen
OLS first stage (no adapter, no regularization) — the "classical baseline"
the adapter must improve on.
`Adapter (unreg.)` uses the adapter with the smallest λ on the sweep grid
(basically λ≈0 — rank-K basis, no smoothing/sparsity).
`Adapter (reg.)` uses the tuned λ that maximises basis alignment.

Uses a single seed (fast); path / median statistics are in Appendix O.

Output:
    .local/figs/covariance_comparison.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

from spatial_adapter.data.generators import generate_time_synthetic_data
from spatial_adapter.data.preprocessing import prepare_all_with_scaling
from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialAdapter,
    SpatialAdapterConfig,
    TrainingConfig,
)
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel
from spatial_adapter.utils.experiment_helpers import compute_ols_coefficients

SEED = 50  # single seed; median-representative in the Appendix O panel anyway
N_LOCATIONS = 512
N_TIME_STEPS = 1024
NOISE_STD = 4.0
EIGENVALUE = 25.0
LATENT_DIM = 1
DEVICE = torch.device("cpu")
OUTPUT_DIR = Path(".local/figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_UNREG = 1e-3  # smallest grid point; effectively no smoothing
LAMBDA_REG = 0.05  # near optimum on the Appendix O sweep


def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    locs = np.linspace(-3, 3, N_LOCATIONS)
    cat, cont, targets = generate_time_synthetic_data(
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
    train_ds, val_ds, test_ds, prep = prepare_all_with_scaling(
        cat_features=cat,
        cont_features=cont,
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

    phi_true = np.exp(-(locs**2))[:, None]
    phi_true /= np.linalg.norm(phi_true)
    sigma_true = EIGENVALUE * (phi_true @ phi_true.T) + NOISE_STD**2 * np.eye(
        N_LOCATIONS
    )

    w_ols, b_ols = compute_ols_coefficients(train_X, train_y, device=DEVICE)

    return {
        "locs": locs,
        "train_loader": train_loader,
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "target_scaler": prep.target_scaler,
        "phi_true": phi_true,
        "sigma_true": sigma_true,
        "w_ols": w_ols,
        "b_ols": b_ols,
        "p_dim": train_X.shape[-1],
    }


def ols_residual_cov(data):
    """Sample covariance of OLS residuals on the raw scale."""
    trend = TrendModel(
        num_continuous_features=data["p_dim"],
        hidden_layer_sizes=[],
        n_locations=N_LOCATIONS,
        init_weight=data["w_ols"],
        init_bias=data["b_ols"],
        freeze_init=True,
        dropout_rate=0.0,
    ).to(DEVICE)
    trend.eval()
    with torch.no_grad():
        mu = trend(data["train_X"].to(DEVICE))
    R_std = (data["train_y"].to(DEVICE) - mu).cpu().numpy()
    S_std = (R_std.T @ R_std) / R_std.shape[0]
    scale = data["target_scaler"].scale_[0]
    return scale**2 * S_std


def adapter_cov(data, tau):
    """Train adapter at (tau1, tau2)=(tau,tau), return estimated Σ̂ on raw scale."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    trend = TrendModel(
        num_continuous_features=data["p_dim"],
        hidden_layer_sizes=[],
        n_locations=N_LOCATIONS,
        init_weight=data["w_ols"],
        init_bias=data["b_ols"],
        freeze_init=True,
        dropout_rate=0.0,
    ).to(DEVICE)
    basis = SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(DEVICE)
    config = SpatialAdapterConfig(
        admm=ADMMConfig(
            rho=1.0, dual_momentum=0.2, max_iters=3000, min_outer=100, tol=1e-5
        ),
        training=TrainingConfig(lr_mu=1e-2, batch_size=N_TIME_STEPS, pretrain_epochs=5),
        basis=BasisConfig(phi_every=5, phi_freeze=500),
    )
    adapter = SpatialAdapter(
        trend,
        basis,
        data["train_loader"],
        val_cont=data["val_X"].to(DEVICE),
        val_y=data["val_y"].to(DEVICE),
        locs=data["locs"],
        config=config,
        device=DEVICE,
        writer=None,
        tau1=tau,
        tau2=tau,
    )
    adapter.pretrain_trend(epochs=5)
    adapter.init_basis_dense()
    adapter.run()
    trend.eval()
    basis.eval()
    with torch.no_grad():
        mu_train = trend(data["train_X"].to(DEVICE))
        R_std = (data["train_y"].to(DEVICE) - mu_train).cpu().numpy()
    S_std = (R_std.T @ R_std) / R_std.shape[0]
    Phi = basis.basis.detach().cpu().numpy()
    PhiTS = Phi.T @ S_std @ Phi
    eigvals = np.linalg.eigvalsh(PhiTS)
    sigma2_hat = max(
        1e-6, (np.trace(S_std) - eigvals.sum()) / (N_LOCATIONS - LATENT_DIM)
    )
    Lambda_hat = np.maximum(eigvals - sigma2_hat, 0.0)
    scale = data["target_scaler"].scale_[0]
    return scale**2 * (
        Phi @ np.diag(Lambda_hat) @ Phi.T + sigma2_hat * np.eye(N_LOCATIONS)
    )


def offdiag_q(sigma, q):
    s = sigma.copy()
    np.fill_diagonal(s, np.nan)
    return float(np.nanpercentile(np.abs(s), q))


def plot_panel(ax, sigma, title, vmax):
    s = sigma.copy()
    np.fill_diagonal(s, np.nan)
    im = ax.imshow(
        s,
        cmap="YlOrRd",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_visible(True)
    return im


def main():
    print("[1/4] Setting up data ...")
    data = setup()
    print(f"  N={N_LOCATIONS}, T={N_TIME_STEPS}, seed={SEED}")

    print("[2/4] Computing OLS residual sample covariance ...")
    sigma_ols = ols_residual_cov(data)

    print(f"[3/4] Training adapter at λ={LAMBDA_UNREG} (unreg.) ...")
    sigma_unreg = adapter_cov(data, LAMBDA_UNREG)

    print(f"      Training adapter at λ={LAMBDA_REG} (reg.) ...")
    sigma_reg = adapter_cov(data, LAMBDA_REG)

    print("[4/4] Rendering comparison figure ...")
    vmax = offdiag_q(data["sigma_true"], 90)
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.85), constrained_layout=True)
    panels = [
        (data["sigma_true"], r"True $\Sigma$"),
        (sigma_ols, "OLS residual cov"),
        (sigma_unreg, "Adapter (unreg.)"),
        (sigma_reg, "Adapter (reg.)"),
    ]
    for ax, (S, title) in zip(axes, panels):
        im = plot_panel(ax, S, title, vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.015)
    cbar.ax.tick_params(labelsize=8)

    outpath = OUTPUT_DIR / "covariance_comparison.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    # Report CovFrob for each panel (sanity)
    def covfrob(S):
        return np.linalg.norm(S - data["sigma_true"]) / np.linalg.norm(
            data["sigma_true"]
        )

    print(f"\n  CovFrob (OLS residual): {covfrob(sigma_ols):.4f}")
    print(f"  CovFrob (unreg.)      : {covfrob(sigma_unreg):.4f}")
    print(f"  CovFrob (reg.)        : {covfrob(sigma_reg):.4f}")


if __name__ == "__main__":
    main()
