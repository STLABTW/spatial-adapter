"""
Wheat Head patch classification experiment — subclass of BaseExperiment.

Stage 1: Frozen pretrained backbone + trained classification head.
Stage 2: TwoStageTrend (frozen head + learnable µ correction) + Spatial Adapter.
Evaluation: reconstruction Accuracy, AUC, F1, + UQ diagnostics (MPIW, CP, ECE).
"""

import numpy as np
import torch
import torch.nn as nn

from examples.experiments.base import BaseExperiment, DataSplit, _record
from spatial_adapter.data.gwhd import get_gwhd_dataloader_and_val
from spatial_adapter.metrics import compute_binary_metrics, expected_calibration_error
from spatial_adapter.models.classification_wrapper import ClassificationWrapper
from spatial_adapter.models.spatial_adapter import SpatialAdapter
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner


class TwoStageTrend(nn.Module):
    """Paper §3.4 two-stage trend: forward(x) = frozen_head(x) + mu(x).

    Stage 1 trains the backbone head (ClassificationWrapper), which is then frozen.
    Stage 2 attaches a zero-init learnable µ correction updated in the ADMM T-step.
    """

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


class WheatHeadExperiment(BaseExperiment):
    """Wheat Head patch classification with frozen vision backbones.

    Paper §3.4 two-stage trend: Stage 2 learns µ only through ADMM, so we
    skip the Stage-2 trend pretrain.
    """

    stage2_pretrain = False

    def __init__(self, config: dict):
        super().__init__(config)
        self.backbones = self.data_cfg.get("backbones", ["resnet152"])
        self.adapter_config.task = "binary"

    def run(self):
        """Override to iterate over backbones × seeds."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}")
        print(f"Backbones: {self.backbones}")
        print(f"Seeds: {self.seeds}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        for backbone_name in self.backbones:
            print(f"\n=== Backbone: {backbone_name} ===")
            for seed in self.seeds:
                print(f"\n--- Seed {seed} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)

                data = self.load_data(seed, backbone_name=backbone_name)
                trend = self.build_first_stage(data, seed)

                baseline_metrics = self._evaluate_baseline_binary(trend, data)
                self.results.extend(
                    _record(
                        self.exp_name,
                        seed,
                        f"{backbone_name}_baseline",
                        baseline_metrics,
                    )
                )
                print(f"  Baseline: acc={baseline_metrics.get('accuracy', 0):.4f}")

                unreg_metrics = self._run_adapter(
                    trend,
                    data,
                    seed,
                    tau1=0.0,
                    tau2=0.0,
                    model_name=f"{backbone_name}_unreg",
                )
                self.results.extend(
                    _record(
                        self.exp_name,
                        seed,
                        f"{backbone_name}_unreg",
                        unreg_metrics,
                    )
                )
                print(f"  Unreg:    acc={unreg_metrics.get('accuracy', 0):.4f}")

                reg_metrics = self._run_optuna(trend, data, seed)
                self.results.extend(
                    _record(
                        self.exp_name,
                        seed,
                        f"{backbone_name}_reg",
                        reg_metrics,
                    )
                )
                print(f"  Reg:      acc={reg_metrics.get('accuracy', 0):.4f}")

        self._save_results()
        self.timer_log.save()
        self.timer_log.summary()
        print(f"\nResults saved to {self.output_dir}")

    def load_data(self, seed: int, backbone_name: str = "resnet152") -> DataSplit:
        cfg = self.data_cfg
        (
            train_loader,
            val_cont,
            val_y,
            test_cont,
            test_y,
            locs,
        ) = get_gwhd_dataloader_and_val(
            csv_path=cfg["csv_path"],
            image_dir=cfg["image_dir"],
            backbone_name=backbone_name,
            device=self.device,
            train_ratio=cfg.get("train_ratio", 0.7),
            val_ratio=cfg.get("val_ratio", 0.15),
            batch_size=self.adapter_config.training.batch_size,
            seed=seed,
            max_images=cfg.get("max_images"),
            cache_dir=cfg.get("cache_dir"),
        )

        _, train_cont, train_y = train_loader.dataset.tensors

        return DataSplit(
            train_loader=train_loader,
            train_cont=train_cont,
            train_y=train_y,
            val_cont=val_cont,
            val_y=val_y,
            test_cont=test_cont,
            test_y=test_y,
            locs=locs,
            n_locations=cfg.get("n_locations", 256),
            p_dim=train_cont.shape[-1],
            metadata={"backbone_name": backbone_name},
        )

    def build_first_stage(self, data: DataSplit, seed: int) -> nn.Module:
        """Stage 1: train ClassificationWrapper, then wrap frozen head with TwoStageTrend.

        Returns a TwoStageTrend template: each Stage-2 trial will deepcopy this
        template, giving a fresh zero-init µ on top of the shared frozen head.
        """
        head = ClassificationWrapper(
            feature_dim=data.p_dim,
            n_locations=data.n_locations,
            hidden_dims=[256, 128],
        ).to(self.device)

        basis_dummy = SpatialBasisLearner(data.n_locations, self.latent_dim).to(
            self.device
        )
        trainer_s1 = SpatialAdapter(
            head,
            basis_dummy,
            data.train_loader,
            val_cont=data.val_cont.to(self.device),
            val_y=data.val_y.to(self.device),
            locs=data.locs,
            config=self.adapter_config,
            device=self.device,
            writer=None,
            tau1=0.0,
            tau2=0.0,
        )
        trainer_s1.pretrain_trend(
            epochs=self.adapter_config.training.pretrain_epochs,
            loss_fn="bce",
        )

        for p in head.parameters():
            p.requires_grad = False

        del basis_dummy, trainer_s1

        trend = TwoStageTrend(head, feature_dim=data.p_dim).to(self.device)
        return trend

    def _evaluate_baseline_binary(self, trend, data):
        """Evaluate backbone-only: TwoStageTrend with zero-init µ = frozen_head(x)."""
        trend.eval()
        with torch.no_grad():
            logits = trend(data.test_cont.to(self.device))
        test_y = data.test_y.to(self.device)
        acc, f1, auc = compute_binary_metrics(test_y, logits)
        return {"accuracy": round(acc, 6), "f1": round(f1, 6), "auc": round(auc, 6)}

    def evaluate(
        self,
        trainer: SpatialAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate reconstruction: accuracy, F1, AUC, + UQ diagnostics."""
        trainer.trend.eval()
        trainer.basis.eval()

        with torch.no_grad():
            test_cont = data.test_cont.to(self.device)
            test_y = data.test_y.to(self.device)

            mu = trainer.trend(test_cont)
            R = torch.special.logit(test_y.clamp(1e-7, 1.0 - 1e-7)) - mu
            spatial = (R @ trainer.basis.basis) @ trainer.basis.basis.T
            logits = mu + spatial

        acc, f1, auc = compute_binary_metrics(test_y, logits)
        metrics = {
            "accuracy": round(acc, 6),
            "f1": round(f1, 6),
            "auc": round(auc, 6),
        }

        try:
            Phi = trainer.basis.basis
            train_R = torch.special.logit(
                trainer.train_y.clamp(1e-7, 1.0 - 1e-7)
            ) - trainer.trend(trainer.train_cont)
            S = (train_R.T @ train_R) / train_R.shape[0]
            PhiTS = Phi.T @ S @ Phi
            eigvals = torch.linalg.eigvalsh(PhiTS)
            sigma2 = max(
                1e-6,
                (torch.trace(S).item() - eigvals.sum().item())
                / (data.n_locations - self.latent_dim),
            )

            Lambda = torch.clamp(eigvals - sigma2, min=0.0)
            v_s = (Phi**2) @ Lambda + sigma2  # (N,) — per-location variance

            # Location-level CP: compare the interval on the *mean* logit at
            # each patch location against the empirical occurrence rate
            # aggregated across test images.  Sample-level coverage against
            # y ∈ {0, 1} is a type mismatch (the interval targets the latent
            # probability p(s), not the Bernoulli outcome) and is not used.
            T_test = logits.shape[0]
            se = torch.sqrt(v_s / T_test)  # SE of the mean across T
            logit_bar = logits.mean(dim=0)  # (N,)
            p_emp = test_y.mean(dim=0)  # (N,) empirical rate / loc

            prob_lo = torch.sigmoid(logit_bar - 1.96 * se)
            prob_hi = torch.sigmoid(logit_bar + 1.96 * se)

            mpiw = (prob_hi - prob_lo).mean().item()
            metrics["mpiw"] = round(mpiw, 6)

            covered = ((p_emp >= prob_lo) & (p_emp <= prob_hi)).float()
            cp = covered.mean().item() * 100.0
            metrics["cp"] = round(cp, 4)

            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            labels = test_y.cpu().numpy().ravel()
            ece = expected_calibration_error(labels, probs)
            metrics["ece"] = round(ece, 6)

        except Exception as e:
            print(f"  [warning] UQ diagnostics failed: {e}")

        return metrics
