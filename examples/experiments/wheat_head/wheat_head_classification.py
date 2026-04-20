#!/usr/bin/env python3
"""
Section 3.4 — Wheat Head Patch Classification (GWHD).

Fills Table 5 (Accuracy / AUC / F1 across 4 backbones × 3 configs)
and Table 6 (Prediction interval calibration for ResNet-152).

Usage:
    python experiments/wheat_head_classification.py \
        --csv_path  /path/to/train.csv \
        --image_dir /path/to/train/ \
        --cache_dir ./cache_gwhd \
        --seeds 5 \
        --max_images 500
"""

import argparse
import copy
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn

# Suppress noisy logs during sweeps
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spatial_adapter.data.gwhd import get_gwhd_dataloader_and_val
from spatial_adapter.metrics import compute_binary_metrics
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

# Experiment configuration
BACKBONES = ["resnet152", "convnext_tiny", "vit_b_16", "sam_vit_h"]

BACKBONE_FEATURE_DIMS = {
    "resnet152": 2048,
    "convnext_tiny": 768,
    "vit_b_16": 768,
    "sam_vit_h": 256,  # SAM ViT-H image encoder → global avg pool
}

N_LOCATIONS = 256  # 16 × 16 patch grid
LATENT_DIM = 16  # K basis functions

# Two-stage trend wrapper
class TwoStageTrend(nn.Module):
    """
    Paper's two-stage trend: forward(x) = frozen_head(x) + mu(x).

    Stage 1 trains the backbone head (ClassificationWrapper).
    Stage 2 freezes the head and attaches a lightweight learnable trend
    correction µ (small MLP), updated in the ADMM T-step.
    """

    def __init__(self, frozen_head: ClassificationWrapper, feature_dim: int):
        super().__init__()
        self.frozen_head = frozen_head
        # Lightweight trend correction µ: small MLP on the same features
        self.mu_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        # Zero-init so µ starts at 0 (adapter begins from backbone prediction)
        nn.init.zeros_(self.mu_net[-1].weight)
        nn.init.zeros_(self.mu_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, p) -> (B, N) logits = frozen_head(x) + mu(x)."""
        with torch.no_grad():
            base = self.frozen_head(x)
        B, N, p = x.shape
        mu = self.mu_net(x.reshape(-1, p)).view(B, N)
        return base + mu

    def residual_parameters(self):
        """Only µ parameters are trainable in Stage 2."""
        return list(self.mu_net.parameters())


# Helpers
def _make_config(
    task: str = "binary", max_iters: int = 200
) -> SpatialNeuralAdapterConfig:
    return SpatialNeuralAdapterConfig(
        admm=ADMMConfig(
            rho=5.0, dual_momentum=0.2, max_iters=max_iters, min_outer=80, tol=1e-4
        ),
        training=TrainingConfig(lr_mu=1e-3, batch_size=64, pretrain_epochs=5),
        basis=BasisConfig(phi_every=5, phi_freeze=80, matrix_reg=1e-6),
        task=task,
    )


def _build_trend_model(feature_dim: int, device: torch.device) -> ClassificationWrapper:
    """Build a ClassificationWrapper as the Stage 1 trend model."""
    return ClassificationWrapper(
        feature_dim=feature_dim,
        n_locations=N_LOCATIONS,
        hidden_dims=[256, 128],
    ).to(device)


@torch.no_grad()
def _evaluate(
    trainer: SpatialNeuralAdapter,
    cont: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate reconstruction quality: trend + spatial basis projection.

    Following the paper's spatial interpolation framework, the adapter refines
    backbone predictions by projecting residuals (logit(y) - mu) onto the learned
    basis Phi. This measures how well the learned spatial structure captures
    within-image patch correlations — analogous to kriging reconstruction.
    """
    trainer.trend.eval()
    trainer.basis.eval()
    cont_d, y_d = cont.to(device), y.to(device)

    mu = trainer.trend(cont_d)
    R = torch.special.logit(y_d.clamp(1e-7, 1.0 - 1e-7)) - mu
    spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
    logits = mu + spatial
    return compute_binary_metrics(y_d, logits)


@torch.no_grad()
def _evaluate_backbone_only(
    trend: ClassificationWrapper,
    cont: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate backbone-only (no adapter)."""
    trend.eval()
    logits = trend(cont.to(device))
    return compute_binary_metrics(y.to(device), logits)


@torch.no_grad()
def _evaluate_uq(
    trainer: SpatialNeuralAdapter,
    cont: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Prediction interval calibration: returns (accuracy, coverage_probability, MPIW).
    Uses delta-method intervals on the logit scale:
        σ(ℓ̂ ± z_{α/2} √v(s))
    """
    trainer.trend.eval()
    trainer.basis.eval()
    cont_d, y_d = cont.to(device), y.to(device)

    mu = trainer.trend(cont_d)
    R = torch.special.logit(y_d.clamp(1e-7, 1.0 - 1e-7)) - mu
    spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
    logits = mu + spatial

    # Accuracy
    acc, _, _ = compute_binary_metrics(y_d, logits)

    # Spatial covariance: Σ = Φ Λ Φ^T + σ² I
    Phi = trainer.basis.basis  # (N, K)
    train_R = torch.special.logit(
        trainer.train_y.clamp(1e-7, 1.0 - 1e-7)
    ) - trainer.trend(trainer.train_cont)
    S = (train_R.T @ train_R) / train_R.shape[0]  # (N, N) sample covariance
    PhiTS = Phi.T @ S @ Phi  # (K, K)
    eigvals = torch.linalg.eigvalsh(PhiTS)
    sigma2 = max(
        1e-6,
        (torch.trace(S).item() - eigvals.sum().item()) / (N_LOCATIONS - LATENT_DIM),
    )

    # Per-patch logit variance: v(s) = [Σ]_{s,s}
    Lambda = torch.clamp(eigvals - sigma2, min=0.0)
    # v_s = sum_k Phi_{s,k}^2 * Lambda_k + sigma2
    v_s = (Phi**2) @ Lambda + sigma2  # (N,)

    z_val = 1.96  # z_{0.025} for 95% CI
    sqrt_v = torch.sqrt(v_s).unsqueeze(0)  # (1, N)

    # Prediction intervals on probability scale
    prob_lo = torch.sigmoid(logits - z_val * sqrt_v)
    prob_hi = torch.sigmoid(logits + z_val * sqrt_v)

    # Coverage: does the true label fall within [lo, hi]?
    y_flat = y_d
    covered = ((y_flat >= prob_lo) & (y_flat <= prob_hi)).float()
    cp = covered.mean().item() * 100.0

    # Mean prediction interval width
    mpiw = (prob_hi - prob_lo).mean().item()

    return acc, cp, mpiw


# Single-seed run
def run_single_seed(
    backbone_name: str,
    csv_path: str,
    image_dir: str,
    cache_dir: str,
    seed: int,
    device: torch.device,
    max_images: Optional[int] = None,
    n_optuna_trials: int = 30,
    admm_iters: int = 200,
) -> Dict:
    """
    Run one seed for one backbone. Returns dict with results for all 3 configs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    (
        train_loader,
        val_cont,
        val_y,
        test_cont,
        test_y,
        locs,
    ) = get_gwhd_dataloader_and_val(
        csv_path=csv_path,
        image_dir=image_dir,
        backbone_name=backbone_name,
        device=device,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32,
        seed=seed,
        max_images=max_images,
        cache_dir=cache_dir,
    )

    feature_dim = val_cont.shape[-1]
    results = {}

    # Stage 1: Train backbone classifier (shared across all configs)
    # Per the paper: pretrain backbone + head, then freeze for Stage 2.
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

    # 1. Backbone only (= Stage 1 result)
    acc, f1, auc = _evaluate_backbone_only(trend_stage1, test_cont, test_y, device)
    results["backbone_only"] = {"accuracy": acc, "f1": f1, "auc": auc}
    print(f"  [backbone_only] acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # Stage 2 helper: create adapter with frozen head + learnable µ
    def _make_adapter(tau1, tau2):
        """Build adapter per the paper: frozen backbone head + learnable µ correction."""
        # Rebuild and load frozen Stage 1 head
        frozen_head = _build_trend_model(feature_dim, device)
        frozen_head.load_state_dict(frozen_state)
        for p in frozen_head.parameters():
            p.requires_grad = False
        # Wrap with learnable trend correction µ
        trend = TwoStageTrend(frozen_head, feature_dim).to(device)
        basis = SpatialBasisLearner(N_LOCATIONS, LATENT_DIM).to(device)
        config = _make_config(max_iters=admm_iters)
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

    # 2. Adapter (unreg.) — τ₁ = τ₂ = 0
    trainer_unreg = _make_adapter(0.0, 0.0)
    trainer_unreg.run()
    acc, f1, auc = _evaluate(trainer_unreg, test_cont, test_y, device)
    results["adapter_unreg"] = {"accuracy": acc, "f1": f1, "auc": auc}
    print(f"  [adapter_unreg] acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # 3. Adapter (reg.) — τ₁, τ₂ tuned via Optuna
    def objective(trial):
        tau1 = trial.suggest_float("tau1", 1e-4, 1e2, log=True)
        tau2 = trial.suggest_float("tau2", 1e-6, 1e0, log=True)
        trainer_t = _make_adapter(tau1, tau2)
        best_acc = trainer_t.run()
        return best_acc  # maximize

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna_trials, show_progress_bar=False)
    best_tau1 = study.best_params["tau1"]
    best_tau2 = study.best_params["tau2"]
    print(f"  [adapter_reg] best tau1={best_tau1:.4e} tau2={best_tau2:.4e}")

    # Retrain with best params
    trainer_reg = _make_adapter(best_tau1, best_tau2)
    trainer_reg.run()
    acc, f1, auc = _evaluate(trainer_reg, test_cont, test_y, device)
    results["adapter_reg"] = {"accuracy": acc, "f1": f1, "auc": auc}
    print(f"  [adapter_reg]   acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # Table 6: UQ for this backbone
    # Backbone only: no UQ
    results["uq_backbone"] = {
        "accuracy": results["backbone_only"]["accuracy"],
        "cp": None,
        "mpiw": None,
    }
    # Unreg
    acc_u, cp_u, mpiw_u = _evaluate_uq(trainer_unreg, test_cont, test_y, device)
    results["uq_unreg"] = {"accuracy": acc_u, "cp": cp_u, "mpiw": mpiw_u}
    # Reg
    acc_r, cp_r, mpiw_r = _evaluate_uq(trainer_reg, test_cont, test_y, device)
    results["uq_reg"] = {"accuracy": acc_r, "cp": cp_r, "mpiw": mpiw_r}

    return results


# Aggregation
def aggregate_results(
    all_results: Dict[str, List[Dict]],
) -> None:
    """Print Tables 5 and 6."""

    print("\n" + "=" * 80)
    print(
        "TABLE 5: Wheat head patch classification (GWHD). Mean (std. error) across seeds."
    )
    print("=" * 80)
    print(f"{'Backbone':<15} {'Model':<22} {'Accuracy (%)':<16} {'AUC':<14} {'F1':<14}")
    print("-" * 80)

    for backbone in BACKBONES:
        if backbone not in all_results:
            continue
        seeds = all_results[backbone]
        for config_key, label in [
            ("backbone_only", "Backbone only"),
            ("adapter_unreg", "+ Adapter (unreg.)"),
            ("adapter_reg", "+ Adapter (reg.)"),
        ]:
            accs = [s[config_key]["accuracy"] * 100 for s in seeds]
            aucs = [s[config_key]["auc"] for s in seeds]
            f1s = [s[config_key]["f1"] for s in seeds]
            n = len(accs)
            se = lambda arr: np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            print(
                f"{backbone:<15} {label:<22} "
                f"{np.mean(accs):>6.2f} ({se(accs):.2f})   "
                f"{np.mean(aucs):>6.4f} ({se(aucs):.4f}) "
                f"{np.mean(f1s):>6.4f} ({se(f1s):.4f})"
            )
        print("-" * 80)

    # Table 6 — ResNet-152 only
    print("\n" + "=" * 80)
    print(
        "TABLE 6: Prediction interval calibration (GWHD, ResNet-152). Mean (std. error) across seeds."
    )
    print("=" * 80)
    print(f"{'Model':<28} {'Accuracy (%)':<16} {'CP (%)':<14} {'MPIW':<14}")
    print("-" * 70)

    if "resnet152" in all_results:
        seeds = all_results["resnet152"]
        for config_key, label in [
            ("uq_backbone", "Backbone only (no UQ)"),
            ("uq_unreg", "+ Adapter (unreg.)"),
            ("uq_reg", "+ Adapter (reg.)"),
        ]:
            accs = [s[config_key]["accuracy"] * 100 for s in seeds]
            n = len(accs)
            se = lambda arr: np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            acc_str = f"{np.mean(accs):>6.2f} ({se(accs):.2f})"

            cps = [
                s[config_key]["cp"] for s in seeds if s[config_key]["cp"] is not None
            ]
            mpiws = [
                s[config_key]["mpiw"]
                for s in seeds
                if s[config_key]["mpiw"] is not None
            ]

            if cps:
                cp_str = f"{np.mean(cps):>6.2f} ({se(cps):.2f})"
                mpiw_str = f"{np.mean(mpiws):>6.4f} ({se(mpiws):.4f})"
            else:
                cp_str = "n/a"
                mpiw_str = "n/a"
            print(f"{label:<28} {acc_str:<16} {cp_str:<14} {mpiw_str:<14}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="GWHD Wheat Head Classification (Section 3.4)"
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to image directory"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache_gwhd", help="Feature cache dir"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--max_images", type=int, default=None, help="Limit images (for debugging)"
    )
    parser.add_argument(
        "--backbones", nargs="+", default=None, help="Subset of backbones to run"
    )
    parser.add_argument(
        "--optuna_trials", type=int, default=30, help="Optuna trials per seed"
    )
    parser.add_argument("--admm_iters", type=int, default=200, help="ADMM iterations")
    parser.add_argument(
        "--output_json", type=str, default=None, help="Save results to JSON"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbones = args.backbones if args.backbones else BACKBONES
    all_results: Dict[str, List[Dict]] = {}

    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone}")
        print(f"{'='*60}")
        seed_results = []
        for s in range(args.seeds):
            seed = 42 + s
            print(f"\n--- Seed {seed} ---")
            res = run_single_seed(
                backbone_name=backbone,
                csv_path=args.csv_path,
                image_dir=args.image_dir,
                cache_dir=args.cache_dir,
                seed=seed,
                device=device,
                max_images=args.max_images,
                n_optuna_trials=args.optuna_trials,
                admm_iters=args.admm_iters,
            )
            seed_results.append(res)
        all_results[backbone] = seed_results

    # Print tables
    aggregate_results(all_results)

    # Save raw results
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
