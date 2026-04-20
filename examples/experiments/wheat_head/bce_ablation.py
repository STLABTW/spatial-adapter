#!/usr/bin/env python3
"""
Appendix D — BCE basis-update ablation (Tables 8 & 9).

Runs 3 variants × 3 seeds on ResNet-152 Wheat Head classification.
Uses the same Stage-1 checkpoint and Stage-2 schedule as the main text.

Usage:
    cd /path/to/geospatial-adapter
    conda run -n air-pollution python examples/experiments/wheat_head/bce_ablation.py \
        --cache_dir ./cache_gwhd --max_images 500
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import optuna
import torch
import torch.nn as nn

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from spatial_adapter.data.gwhd import get_gwhd_dataloader_and_val
from spatial_adapter.metrics import (
    compute_binary_metrics,
    expected_calibration_error,
)
from spatial_adapter.models import (
    ClassificationWrapper,
    SpatialBasisLearner,
    SpatialNeuralAdapter,
)
from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialNeuralAdapterConfig,
    TrainingConfig,
)

N_LOCATIONS = 256
LATENT_DIM = 16
VARIANTS = ["variance_only", "full_taylor", "irls"]
VARIANT_LABELS = {
    "variance_only": "(A) variance-only",
    "full_taylor": "(B) full Taylor C_BCE",
    "irls": "(C) IRLS working response",
}


# ── Two-stage trend wrapper (same as main experiment) ──────────────────────


class TwoStageTrend(nn.Module):
    def __init__(self, frozen_head: ClassificationWrapper, feature_dim: int):
        super().__init__()
        self.frozen_head = frozen_head
        self.mu_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.mu_net[-1].weight)
        nn.init.zeros_(self.mu_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base = self.frozen_head(x)
        B, N, p = x.shape
        mu = self.mu_net(x.reshape(-1, p)).view(B, N)
        return base + mu

    def residual_parameters(self):
        return list(self.mu_net.parameters())


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_config(
    bce_variant: str = "variance_only",
    max_iters: int = 200,
) -> SpatialNeuralAdapterConfig:
    return SpatialNeuralAdapterConfig(
        admm=ADMMConfig(
            rho=5.0, dual_momentum=0.2, max_iters=max_iters, min_outer=80, tol=1e-4
        ),
        training=TrainingConfig(lr_mu=1e-3, batch_size=64, pretrain_epochs=5),
        basis=BasisConfig(
            phi_every=5,
            phi_freeze=80,
            matrix_reg=1e-6,
            bce_variant=bce_variant,
        ),
        task="binary",
    )


def _build_trend_model(feature_dim: int, device: torch.device) -> ClassificationWrapper:
    return ClassificationWrapper(
        feature_dim=feature_dim,
        n_locations=N_LOCATIONS,
        hidden_dims=[256, 128],
    ).to(device)


@torch.no_grad()
def _evaluate_full(
    trainer: SpatialNeuralAdapter,
    cont: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Compute accuracy, AUC, F1, MPIW, ECE, CP for Table 8."""
    trainer.trend.eval()
    trainer.basis.eval()
    cont_d, y_d = cont.to(device), y.to(device)

    mu = trainer.trend(cont_d)
    R = torch.special.logit(y_d.clamp(1e-7, 1.0 - 1e-7)) - mu
    spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
    logits = mu + spatial  # (T, N)

    acc, f1, auc = compute_binary_metrics(y_d, logits)

    # --- UQ: per-location coverage (same fix as before) ---
    Phi = trainer.basis.basis  # (N, K)
    train_R = torch.special.logit(
        trainer.train_y.clamp(1e-7, 1.0 - 1e-7)
    ) - trainer.trend(trainer.train_cont)
    S = (train_R.T @ train_R) / train_R.shape[0]
    PhiTS = Phi.T @ S @ Phi
    eigvals = torch.linalg.eigvalsh(PhiTS)
    sigma2 = max(
        1e-6,
        (torch.trace(S).item() - eigvals.sum().item()) / (N_LOCATIONS - LATENT_DIM),
    )
    Lambda = torch.clamp(eigvals - sigma2, min=0.0)
    v_s = (Phi**2) @ Lambda + sigma2  # (N,)

    T_test = logits.shape[0]
    z_val = 1.96
    se = torch.sqrt(v_s / T_test)

    logit_bar = logits.mean(dim=0)
    p_emp = y_d.mean(dim=0)

    prob_lo = torch.sigmoid(logit_bar - z_val * se)
    prob_hi = torch.sigmoid(logit_bar + z_val * se)

    covered = ((p_emp >= prob_lo) & (p_emp <= prob_hi)).float()
    cp = covered.mean().item() * 100.0
    mpiw = (prob_hi - prob_lo).mean().item()

    # --- ECE ---
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    labels = y_d.cpu().numpy().ravel()
    ece = expected_calibration_error(labels, probs)

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "cp": cp,
        "mpiw": mpiw,
        "ece": ece,
    }


@torch.no_grad()
def _compute_offbasis_diagnostics(trainer: SpatialNeuralAdapter) -> Dict:
    """Compute Table 9 diagnostics: ||ZP⊥||∞, ||ZP⊥||F/||Z||F."""
    Z = trainer.z_train  # (T, N)
    Phi = trainer.basis.basis  # (N, K)
    # P = Φ Φ^T (projection onto basis), P⊥ = I - P
    # ZP⊥ = Z - Z Φ Φ^T = Z (I - Φ Φ^T)
    Z_proj = (Z @ Phi) @ Phi.T  # Z P
    Z_perp = Z - Z_proj  # Z P⊥

    zp_inf = Z_perp.abs().max().item()
    z_fro = Z.norm("fro").item()
    zp_fro = Z_perp.norm("fro").item()
    zp_ratio = zp_fro / z_fro if z_fro > 0 else 0.0

    return {
        "zp_perp_inf": zp_inf,
        "zp_perp_ratio": zp_ratio,
    }


def _count_basis_steps(config: SpatialNeuralAdapterConfig, total_iters: int) -> int:
    """Count how many basis-step calls were executed before freezing."""
    phi_every = int(config.basis.phi_every)
    phi_freeze = int(config.basis.phi_freeze)
    count = 0
    for i in range(total_iters):
        if (i % phi_every == 0) and (i < phi_freeze):
            count += 1
    return count


# ── Single-seed run ────────────────────────────────────────────────────────


def run_single_seed(
    variant: str,
    csv_path: str,
    image_dir: str,
    cache_dir: str,
    seed: int,
    device: torch.device,
    frozen_state: dict,
    train_loader,
    val_cont: torch.Tensor,
    val_y: torch.Tensor,
    test_cont: torch.Tensor,
    test_y: torch.Tensor,
    locs,
    feature_dim: int,
    best_tau1: float,
    best_tau2: float,
    admm_iters: int = 200,
) -> Dict:
    """Run one variant × one seed. Returns results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build adapter with the specified bce_variant
    frozen_head = _build_trend_model(feature_dim, device)
    frozen_head.load_state_dict(frozen_state)
    for p in frozen_head.parameters():
        p.requires_grad = False
    trend = TwoStageTrend(frozen_head, feature_dim).to(device)
    basis = SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(device)
    config = _make_config(bce_variant=variant, max_iters=admm_iters)
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
        tau1=best_tau1,
        tau2=best_tau2,
    )
    trainer.init_basis_dense()

    # Time the basis steps
    t0 = time.perf_counter()
    trainer.run()
    t_total = time.perf_counter() - t0

    total_iters = trainer.global_iter
    n_basis_calls = _count_basis_steps(config, total_iters)
    basis_ms_per_sweep = (t_total / total_iters * 1000) if total_iters > 0 else 0.0

    # Table 8 metrics
    metrics = _evaluate_full(trainer, test_cont, test_y, device)
    metrics["basis_ms_sweep"] = basis_ms_per_sweep

    # Table 9 diagnostics
    diag = _compute_offbasis_diagnostics(trainer)
    metrics["zp_perp_inf"] = diag["zp_perp_inf"]
    metrics["zp_perp_ratio"] = diag["zp_perp_ratio"]
    metrics["n_basis_calls"] = n_basis_calls

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BCE basis-update ablation (Tables 8 & 9)"
    )
    parser.add_argument("--csv_path", type=str, default="dummy")
    parser.add_argument("--image_dir", type=str, default="dummy")
    parser.add_argument("--cache_dir", type=str, default="./cache_gwhd")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=42, help="First seed value")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--optuna_trials", type=int, default=30)
    parser.add_argument("--admm_iters", type=int, default=200)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results: Dict[str, List[Dict]] = {v: [] for v in VARIANTS}

    for s_idx in range(args.seeds):
        seed = args.seed_start + s_idx
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load data (shared across variants for this seed)
        (
            train_loader,
            val_cont,
            val_y,
            test_cont,
            test_y,
            locs,
        ) = get_gwhd_dataloader_and_val(
            csv_path=args.csv_path,
            image_dir=args.image_dir,
            backbone_name="resnet152",
            device=device,
            train_ratio=0.7,
            val_ratio=0.15,
            batch_size=32,
            seed=seed,
            max_images=args.max_images,
            cache_dir=args.cache_dir,
        )
        feature_dim = val_cont.shape[-1]

        # Stage 1: shared backbone pretraining
        trend_stage1 = _build_trend_model(feature_dim, device)
        config_s1 = _make_config(max_iters=0)
        trainer_s1 = SpatialNeuralAdapter(
            trend=trend_stage1,
            basis=SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(device),
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config_s1,
            device=device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        trainer_s1.pretrain_trend(epochs=10, loss_fn="bce")
        frozen_state = copy.deepcopy(trend_stage1.state_dict())

        # Optuna: tune (τ₁, τ₂) using variance_only (default), shared across variants
        def _make_adapter_for_optuna(tau1, tau2):
            frozen_head = _build_trend_model(feature_dim, device)
            frozen_head.load_state_dict(frozen_state)
            for p in frozen_head.parameters():
                p.requires_grad = False
            trend = TwoStageTrend(frozen_head, feature_dim).to(device)
            basis = SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(device)
            config = _make_config(
                bce_variant="variance_only", max_iters=args.admm_iters
            )
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
                tau1=tau1,
                tau2=tau2,
            )
            trainer.init_basis_dense()
            return trainer

        def objective(trial):
            tau1 = trial.suggest_float("tau1", 1e-4, 1e2, log=True)
            tau2 = trial.suggest_float("tau2", 1e-6, 1e0, log=True)
            trainer_t = _make_adapter_for_optuna(tau1, tau2)
            return trainer_t.run()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials, show_progress_bar=False)
        best_tau1 = study.best_params["tau1"]
        best_tau2 = study.best_params["tau2"]
        print(f"  Best tau1={best_tau1:.4e} tau2={best_tau2:.4e}")

        # Run each variant with the same (τ₁, τ₂)
        for variant in VARIANTS:
            print(f"\n  --- {VARIANT_LABELS[variant]} ---")
            res = run_single_seed(
                variant=variant,
                csv_path=args.csv_path,
                image_dir=args.image_dir,
                cache_dir=args.cache_dir,
                seed=seed,
                device=device,
                frozen_state=frozen_state,
                train_loader=train_loader,
                val_cont=val_cont,
                val_y=val_y,
                test_cont=test_cont,
                test_y=test_y,
                locs=locs,
                feature_dim=feature_dim,
                best_tau1=best_tau1,
                best_tau2=best_tau2,
                admm_iters=args.admm_iters,
            )
            all_results[variant].append(res)
            print(
                f"    acc={res['accuracy']:.4f} MPIW={res['mpiw']:.4f} "
                f"ECE={res['ece']:.4f} CP={res['cp']:.2f}% "
                f"basis_ms={res['basis_ms_sweep']:.1f} "
                f"||ZP⊥||∞={res['zp_perp_inf']:.4e} "
                f"ratio={res['zp_perp_ratio']:.4e} "
                f"#basis={res['n_basis_calls']}"
            )

    # ── Print Tables ──────────────────────────────────────────────────────
    se = lambda arr: np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    print("\n" + "=" * 90)
    print("TABLE 8: Bernoulli basis-update ablation (ResNet-152, 3 seeds)")
    print("=" * 90)
    print(
        f"{'Variant':<30} {'Accuracy (%)':<16} {'MPIW':<16} {'ECE':<16} {'CP (%)':<16} {'Basis ms/sweep':<16}"
    )
    print("-" * 90)
    for v in VARIANTS:
        seeds = all_results[v]
        accs = [s["accuracy"] * 100 for s in seeds]
        mpiws = [s["mpiw"] for s in seeds]
        eces = [s["ece"] for s in seeds]
        cps = [s["cp"] for s in seeds]
        bms = [s["basis_ms_sweep"] for s in seeds]
        print(
            f"{VARIANT_LABELS[v]:<30} "
            f"{np.mean(accs):>6.2f} ({se(accs):.2f})   "
            f"{np.mean(mpiws):>6.4f} ({se(mpiws):.4f}) "
            f"{np.mean(eces):>6.4f} ({se(eces):.4f}) "
            f"{np.mean(cps):>6.2f} ({se(cps):.2f})   "
            f"{np.mean(bms):>6.1f} ({se(bms):.1f})"
        )

    print("\n" + "=" * 90)
    print("TABLE 9: Diagnostic measurements (ResNet-152, 3 seeds)")
    print("=" * 90)
    print(
        f"{'Variant':<30} {'||ZP⊥||∞':<16} {'||ZP⊥||F/||Z||F':<16} {'# basis-step calls':<20}"
    )
    print("-" * 90)
    for v in VARIANTS:
        seeds = all_results[v]
        infs = [s["zp_perp_inf"] for s in seeds]
        rats = [s["zp_perp_ratio"] for s in seeds]
        calls = [s["n_basis_calls"] for s in seeds]
        print(
            f"{VARIANT_LABELS[v]:<30} "
            f"{np.mean(infs):>8.4f} ({se(infs):.4f}) "
            f"{np.mean(rats):>8.4f} ({se(rats):.4f}) "
            f"{np.mean(calls):>6.1f} ({se(calls):.1f})"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
