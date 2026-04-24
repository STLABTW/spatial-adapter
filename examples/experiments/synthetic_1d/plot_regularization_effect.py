#!/usr/bin/env python
"""
Plot the effect of spatial regularizers on the synthetic 1D benchmark.

Produces Figure 2 in the paper:
  (a) Regularization path: basis alignment + CovFrob vs λ
  (b) Estimated spatial covariance across λ (shared colorbar)

Appendix:
  (c) Learned basis φ̂ vs ground truth φ across λ

Usage:
    python examples/experiments/synthetic_1d/plot_regularization_effect.py

Output:
    .local/performance_path.png          (main, mean ± std over seeds)
    .local/covariance_evolution.png      (main, reference seed)
    .local/appendix_basis_evolution.png  (appendix, reference seed)
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Publication-quality defaults: serif labels, thin spines, computer-modern
# math, no frame around legends. Applied globally so every figure in this
# script renders with the same typography.
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
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

# Configuration
SEEDS = list(range(42, 72))  # 30 seeds — shaded band ± 1 std across seeds
REFERENCE_SEED = None  # picked dynamically: the seed whose mean CovFrob over
# the λ-grid is closest to the cross-seed median (i.e.
# a median-representative seed, not a cherry-picked one).
N_LOCATIONS = 512
N_TIME_STEPS = 1024
NOISE_STD = 4.0
EIGENVALUE = 25.0  # var(α_t); SNR = eigenvalue / noise_std² ≈ 1.56
LATENT_DIM = 1
DEVICE = torch.device("cpu")  # Deterministic tensor ops — path smoothness > GPU speed
OUTPUT_DIR = Path(".local")

# λ grid: sweep λ = λ₁ = λ₂ (both penalties together).
# Dense near the empirical optimum (~0.02) and coarse tails out to 10⁴
# so the full regularization regime (under → optimal → over) is visible
# without wasting points on the flat far tails.
LAMBDA_GRID = np.concatenate(
    [
        np.logspace(-3, -2, 3, endpoint=False),  # coarse under-reg tail
        np.logspace(-2, 0, 20, endpoint=False),  # dense around optimum (0.01–1)
        np.logspace(0, 4, 11),  # moderate → over-reg tail
    ]
)


def _denorm(y_std, scaler):
    """Inverse-transform standardized predictions to raw scale."""
    y_np = (
        y_std.detach().cpu().numpy()
        if hasattr(y_std, "detach")
        else y_std.cpu().numpy()
    )
    shape = y_np.shape
    y_raw = scaler.inverse_transform(y_np.reshape(-1, 1)).reshape(shape)
    return torch.tensor(y_raw, dtype=torch.float32)


def _metrics_paper(y_true, y_pred):
    """Compute RMSE, MAE, R² using paper's convention.

    R² uses sklearn's default multioutput='uniform_average' on 2-D (T, N)
    arrays — per-location R² averaged across locations — which is what
    ``compute_metrics`` in the production pipeline does. Evaluated in
    standardized space to match Table 3 numerics.
    """
    from sklearn.metrics import r2_score

    yt = (
        y_true.detach().cpu().numpy()
        if hasattr(y_true, "detach")
        else np.asarray(y_true)
    )
    yp = (
        y_pred.detach().cpu().numpy()
        if hasattr(y_pred, "detach")
        else np.asarray(y_pred)
    )
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    r2 = float(r2_score(yt, yp))  # 2-D → per-location, uniform-averaged
    return rmse, mae, r2


def setup_data(seed):
    """Generate synthetic data, standardize for training, keep scaler for inverse."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        seed=seed,
    )

    # Standardize for training (same as production pipeline)
    train_ds, val_ds, test_ds, preprocessor = prepare_all_with_scaling(
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
    _, test_X, test_y = test_ds.tensors

    # Target scaler for inverse transform
    target_scaler = preprocessor.target_scaler

    # Denormalize val/test targets to raw scale
    val_y_raw = _denorm(val_y, target_scaler)
    test_y_raw = _denorm(test_y, target_scaler)

    # Ground truth (raw scale)
    phi_true = np.exp(-(locs**2))[:, None]
    phi_true /= np.linalg.norm(phi_true)
    sigma_true = EIGENVALUE * (phi_true @ phi_true.T) + NOISE_STD**2 * np.eye(
        N_LOCATIONS
    )

    # OLS warm-start (on standardized data)
    w_ols, b_ols = compute_ols_coefficients(train_X, train_y, device=DEVICE)

    return {
        "locs": locs,
        "train_loader": train_loader,
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "val_y_raw": val_y_raw,
        "test_y_raw": test_y_raw,
        "target_scaler": target_scaler,
        "phi_true": phi_true,
        "sigma_true": sigma_true,
        "w_ols": w_ols,
        "b_ols": b_ols,
        "p_dim": train_X.shape[-1],
    }


def run_one_lambda(data, tau, seed, prev_state=None):
    """Train adapter with tau1 = tau2 = tau."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

    if prev_state is not None:
        trend.load_state_dict(prev_state[0], strict=False)
        basis.load_state_dict(prev_state[1], strict=False)

    # Full-batch ADMM + CPU + warm-start produces a smooth regularization
    # path. (Mini-batch stochasticity helps escape bad local minima when
    # SNR is very low, but at SNR ≈ 1.5+ the population signal already
    # dominates the residual spectrum so full-batch converges reliably.)
    max_iters = 1500 if prev_state is not None else 3000
    config = SpatialAdapterConfig(
        admm=ADMMConfig(
            rho=1.0, dual_momentum=0.2, max_iters=max_iters, min_outer=100, tol=1e-5
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
    if prev_state is None:
        adapter.pretrain_trend(epochs=5)
        adapter.init_basis_dense()
    adapter.run()

    scaler = data["target_scaler"]

    # Evaluate in STANDARDIZED space with 2-D r2_score (per-location avg) —
    # this matches the production pipeline's compute_metrics exactly.
    trend.eval()
    basis.eval()
    with torch.no_grad():
        # Validation
        mu_val = trend(data["val_X"].to(DEVICE))
        R_val = data["val_y"].to(DEVICE) - mu_val
        spatial_val = (R_val @ basis.basis) @ basis.basis.T
        y_pred_val_std = mu_val + spatial_val
        rmse_val, mae_val, r2_val = _metrics_paper(
            data["val_y"].to(DEVICE), y_pred_val_std
        )

        # Test
        mu_test = trend(data["test_X"].to(DEVICE))
        R_test = data["test_y"].to(DEVICE) - mu_test
        spatial_test = (R_test @ basis.basis) @ basis.basis.T
        y_pred_test_std = mu_test + spatial_test
        rmse_test, mae_test, r2_test = _metrics_paper(
            data["test_y"].to(DEVICE), y_pred_test_std
        )

        # Keep raw-scale quantities for covariance plots (unchanged)
        _denorm(y_pred_val_std, scaler)
        _denorm(y_pred_test_std, scaler)

        # Covariance: compute on RAW-scale residuals
        mu_train = trend(data["train_X"].to(DEVICE))
        y_pred_train_std = (
            mu_train
            + ((data["train_y"].to(DEVICE) - mu_train) @ basis.basis) @ basis.basis.T
        )
        y_pred_train_raw = _denorm(y_pred_train_std, scaler)
        train_y_raw_np = _denorm(data["train_y"], scaler).numpy()
        train_y_raw_np - y_pred_train_raw.numpy()

        # Estimated covariance from raw-scale predictions
        pred_raw_np = y_pred_train_raw.numpy()
        np.cov(pred_raw_np.T)
        np.cov(train_y_raw_np.T)

        # Basis-induced covariance from adapter
        Phi = basis.basis.cpu().numpy()
        R_std_np = (data["train_y"].to(DEVICE) - mu_train).cpu().numpy()
        S_std = (R_std_np.T @ R_std_np) / R_std_np.shape[0]
        PhiTS = Phi.T @ S_std @ Phi
        eigvals = np.linalg.eigvalsh(PhiTS)
        sigma2_hat = max(
            1e-6, (np.trace(S_std) - eigvals.sum()) / (N_LOCATIONS - LATENT_DIM)
        )
        Lambda_hat = np.maximum(eigvals - sigma2_hat, 0.0)
        # Scale back to raw: sigma_hat_raw = scale² * sigma_hat_std
        scale = scaler.scale_[0] if hasattr(scaler, "scale_") else 1.0
        sigma_hat_raw = scale**2 * (
            Phi @ np.diag(Lambda_hat) @ Phi.T + sigma2_hat * np.eye(N_LOCATIONS)
        )

        # CovFrob against ground truth
        covfrob = np.linalg.norm(sigma_hat_raw - data["sigma_true"]) / np.linalg.norm(
            data["sigma_true"]
        )

        # Basis alignment
        phi_learned = Phi[:, 0] / (np.linalg.norm(Phi[:, 0]) + 1e-12)
        phi_gt = data["phi_true"][:, 0]
        alignment = abs(np.dot(phi_learned, phi_gt))

    return {
        "tau": tau,
        "val_rmse": rmse_val,
        "test_mae": mae_test,
        "test_r2": r2_test,
        "covfrob": covfrob,
        "alignment": alignment,
        "sigma_hat": sigma_hat_raw,
        "phi_learned": phi_learned,
        "state": (trend.state_dict(), basis.state_dict()),
    }


def run_sweep(seed):
    """Run full λ sweep for a single seed. Returns list of per-λ result dicts."""
    data = setup_data(seed)
    print(f"\n── seed={seed} ──")
    results = []
    prev_state = None
    for i, tau in enumerate(LAMBDA_GRID):
        tag = "cold" if prev_state is None else "warm"
        print(f"  [{i+1}/{len(LAMBDA_GRID)}] λ={tau:.2e} ({tag})", end="", flush=True)
        r = run_one_lambda(data, tau, seed=seed, prev_state=prev_state)
        print(f"  → CovFrob={r['covfrob']:.4f}, align={r['alignment']:.4f}")
        results.append(r)
        prev_state = r["state"]
    return data, results


CACHE_PATH = Path(".local") / "sweep_cache.npz"


def _load_or_run_sweep():
    """Return (align_sn, covf_sn, sigma_hats_ref, phi_learned_ref, phi_true,
    sigma_true, locs, taus, alignments_ref, covfrobs_ref) — loading from the
    npz cache if available and matching SEEDS / LAMBDA_GRID, else running
    the full sweep and writing the cache.

    The cache holds the raw per-seed, per-λ arrays plus the reference-seed
    Σ̂/φ̂ needed to render the heatmap panels.  This lets us iterate on
    plot styling without paying the 15-minute sweep cost again.
    """
    if CACHE_PATH.exists():
        try:
            c = np.load(CACHE_PATH, allow_pickle=False)
            if np.array_equal(c["seeds"], np.array(SEEDS)) and np.allclose(
                c["taus"], LAMBDA_GRID
            ):
                print(f"Loaded cache: {CACHE_PATH}")
                return (
                    c["align_sn"],
                    c["covf_sn"],
                    c["sigma_hats_ref"],
                    c["phi_learned_ref"],
                    c["phi_true"],
                    c["sigma_true"],
                    c["locs"],
                    c["taus"],
                    c["alignments_ref"],
                    c["covfrobs_ref"],
                )
        except Exception as e:
            print(f"Cache load failed ({e!r}); re-running sweep")

    n_seeds = len(SEEDS)
    n_taus = len(LAMBDA_GRID)
    align_sn = np.zeros((n_seeds, n_taus))
    covf_sn = np.zeros((n_seeds, n_taus))

    # Hold per-seed (data, results) in memory so we can pick a
    # median-representative seed *after* the full sweep has completed
    # rather than hard-coding seed 42.
    all_data: dict = {}
    all_results: dict = {}
    for s_idx, seed in enumerate(SEEDS):
        data, results = run_sweep(seed)
        align_sn[s_idx] = [r["alignment"] for r in results]
        covf_sn[s_idx] = [r["covfrob"] for r in results]
        all_data[seed] = data
        all_results[seed] = results

    # Pick the seed whose mean CovFrob over the λ-grid is closest to the
    # cross-seed median.  Using the median (not seed 42) as the heatmap
    # source pre-empts the "cherry-picked seed" concern.
    seed_mean_covf = covf_sn.mean(axis=1)  # (n_seeds,)
    median_of_means = float(np.median(seed_mean_covf))
    rep_idx = int(np.argmin(np.abs(seed_mean_covf - median_of_means)))
    rep_seed = SEEDS[rep_idx]
    print(
        f"Representative seed (median mean-CovFrob across λ-grid): {rep_seed} "
        f"(mean CovFrob {seed_mean_covf[rep_idx]:.4f} vs. median {median_of_means:.4f})"
    )
    ref_data = all_data[rep_seed]
    ref_results = all_results[rep_seed]

    taus = np.array([r["tau"] for r in ref_results])
    sigma_hats_ref = np.stack([r["sigma_hat"] for r in ref_results])
    phi_learned_ref = np.stack([r["phi_learned"] for r in ref_results])
    alignments_ref = np.array([r["alignment"] for r in ref_results])
    covfrobs_ref = np.array([r["covfrob"] for r in ref_results])

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        seeds=np.array(SEEDS),
        taus=taus,
        align_sn=align_sn,
        covf_sn=covf_sn,
        sigma_hats_ref=sigma_hats_ref,
        phi_learned_ref=phi_learned_ref,
        phi_true=ref_data["phi_true"][:, 0],
        sigma_true=ref_data["sigma_true"],
        locs=ref_data["locs"],
        alignments_ref=alignments_ref,
        covfrobs_ref=covfrobs_ref,
    )
    print(f"Saved cache: {CACHE_PATH}")
    return (
        align_sn,
        covf_sn,
        sigma_hats_ref,
        phi_learned_ref,
        ref_data["phi_true"][:, 0],
        ref_data["sigma_true"],
        ref_data["locs"],
        taus,
        alignments_ref,
        covfrobs_ref,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        align_sn,
        covf_sn,
        sigma_hats_ref,
        phi_learned_ref,
        phi_true,
        sigma_true,
        locs,
        taus,
        alignments_ref,
        covfrobs_ref,
    ) = _load_or_run_sweep()

    align_sn.shape[0]

    # Median + IQR is more robust to the transition-region bifurcation than
    # mean ± std: near λ ≈ 1 the loss landscape flattens and a minority of
    # seeds collapse while the majority still recover φ; the median tracks
    # the typical solution while the IQR honestly shows the spread.
    align_median = np.median(align_sn, axis=0)
    align_q25 = np.percentile(align_sn, 25, axis=0)
    align_q75 = np.percentile(align_sn, 75, axis=0)
    covf_median = np.median(covf_sn, axis=0)
    covf_q25 = np.percentile(covf_sn, 25, axis=0)
    covf_q75 = np.percentile(covf_sn, 75, axis=0)

    best_idx = int(np.argmax(align_median))
    best_tau = float(taus[best_idx])
    best_align = float(align_median[best_idx])
    best_covfrob = float(covf_median[best_idx])

    print(
        f"\nBest: λ={best_tau:.2e}, align={best_align:.4f}, CovFrob={best_covfrob:.4f}"
    )

    # Okabe–Ito deep blue / vermillion — CVD-safe and grayscale-separable.
    # Vermillion (#D55E00) reads as a deep amber/burnt-orange, far enough
    # from bright yellow to stay stable under print/camera-ready scaling.
    C_ALIGN = "#0072B2"
    C_COVF = "#D55E00"

    # (a) Regularization path
    # Median [25–75% IQR] over n seeds. X-axis truncated to [1e-3, 10]:
    # beyond λ=10 the basis has fully collapsed and |⟨φ̂, φ⟩| behaves like
    # |⟨random unit vector, φ⟩| ~ 1/√N — uninformative noise rather than
    # a property of the method.
    fig, ax1 = plt.subplots(figsize=(6.5, 2.3), constrained_layout=True)
    ax1.semilogx(
        taus,
        align_median,
        "-",
        color=C_ALIGN,
        marker="o",
        markersize=4.2,
        markeredgewidth=0,
    )
    ax1.fill_between(taus, align_q25, align_q75, color=C_ALIGN, alpha=0.12, linewidth=0)
    ax1.axvline(best_tau, color="0.4", linestyle=":", linewidth=0.9)
    ax1.text(
        best_tau,
        0.04,
        rf" $\lambda^\star={best_tau:.2f}$",
        fontsize=9,
        color="0.3",
        va="bottom",
        ha="left",
    )
    ax1.set_xlabel(r"Regularization strength  $\lambda$")
    ax1.set_ylabel("Basis alignment", color=C_ALIGN)
    ax1.tick_params(axis="y", colors=C_ALIGN)
    ax1.set_xlim(1e-3, 1e1)
    ax1.set_ylim(0.0, 1.04)
    ax1.grid(True, which="both", axis="x", alpha=0.15, linewidth=0.5)
    ax1.grid(True, axis="y", alpha=0.15, linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.7)
    ax2.semilogx(
        taus,
        covf_median,
        "--",
        color=C_COVF,
        marker="s",
        markersize=4.2,
        markeredgewidth=0,
    )
    ax2.fill_between(taus, covf_q25, covf_q75, color=C_COVF, alpha=0.12, linewidth=0)
    ax2.set_ylabel("CovFrob", color=C_COVF)
    ax2.tick_params(axis="y", colors=C_COVF)

    # Legend omitted: coloured y-axis labels (blue "Basis alignment" left,
    # orange "CovFrob" right) already identify the two series redundantly
    # through colour + axis label.

    path_a = OUTPUT_DIR / "performance_path.png"
    fig.savefig(path_a)
    plt.close(fig)
    print(f"Saved: {path_a}")

    # (b) Covariance evolution
    # Three representative λ (under / optimal / over) + ground truth.
    # Shared colormap anchored on the ground-truth off-diagonal magnitude
    # so under-regularized panels saturate — that saturation *is* the
    # message.  The ground-truth basis is sign-definite (Gaussian bump),
    # so off-diagonal covariances are ≥ 0; we use the light-background
    # sequential map ``OrRd`` (white → orange → red, colorblind-safe,
    # print-safe) so that low-magnitude artifact structure remains
    # readable at paper-column size — contrast with a dark-background
    # map that compresses low values into near-black.
    # Pick λ one clear order of magnitude below and above the selected λ*
    # (100× on each side, snapping to the nearest grid point).  The left
    # panel lands near the grid floor (λ≈10⁻³), making the under-regularized
    # regime's spurious off-diagonal structure unambiguous.
    target_taus = [best_tau / 100.0, best_tau, best_tau * 50.0]
    pick_idx = sorted(
        {int(np.argmin(np.abs(np.log10(taus) - np.log10(t)))) for t in target_taus}
    )

    def _offdiag_q(sigma, q):
        s = sigma.copy()
        np.fill_diagonal(s, np.nan)
        return float(np.nanpercentile(np.abs(s), q))

    # Anchor vmax on the 90th percentile of the ground-truth off-diagonal
    # (rather than the 99th).  The top 10% gets clipped to the darkest
    # red in over-regularized panels, which is semantically fine; the
    # pay-off is that the mid-to-high range of well-fitted panels uses
    # more of the colormap (yellow → orange → red), making the optimal
    # and mild-under-reg panels easier to tell apart at paper-column size.
    vmax = _offdiag_q(sigma_true, 90)
    n_panels = len(pick_idx) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.6, 2.2), constrained_layout=True)

    def _plot_cov(ax, sigma, title):
        s = sigma.copy()
        np.fill_diagonal(s, np.nan)
        im = ax.imshow(
            s,
            # YlOrRd has four stops (yellow → orange → red → dark-red) vs.
            # OrRd's three — more perceptual range in the mid-high region.
            cmap="YlOrRd",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=11, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_visible(True)
        return im

    im = _plot_cov(axes[0], sigma_true, r"Ground truth")
    for j, idx in enumerate(pick_idx):
        _plot_cov(axes[j + 1], sigma_hats_ref[idx], rf"$\lambda={taus[idx]:.2g}$")

    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.015)
    cbar.ax.tick_params(labelsize=9, width=0.6)
    cbar.outline.set_linewidth(0.6)

    path_b = OUTPUT_DIR / "covariance_evolution.png"
    fig.savefig(path_b)
    plt.close(fig)
    print(f"Saved: {path_b}")

    # Appendix: basis evolution
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.8, 1.6), constrained_layout=True, sharey=True
    )

    axes[0].plot(locs, phi_true, color="black", linewidth=1.3)
    axes[0].set_title(r"Ground truth  $\phi$", fontsize=9, pad=3)
    axes[0].set_ylim(-0.15, 0.15)
    axes[0].grid(True, alpha=0.15, linewidth=0.5)

    for j, idx in enumerate(pick_idx):
        phi_l = phi_learned_ref[idx]
        if np.dot(phi_l, phi_true) < 0:
            phi_l = -phi_l
        axes[j + 1].plot(locs, phi_true, color="0.5", linestyle="--", linewidth=0.8)
        axes[j + 1].plot(locs, phi_l, color=C_ALIGN, linewidth=1.3)
        axes[j + 1].set_title(rf"$\lambda={taus[idx]:.2g}$", fontsize=9, pad=3)
        axes[j + 1].grid(True, alpha=0.15, linewidth=0.5)

    path_c = OUTPUT_DIR / "appendix_basis_evolution.png"
    fig.savefig(path_c)
    plt.close(fig)
    print(f"Saved: {path_c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
