"""
BaseExperiment — shared skeleton for all Spatial Adapter experiments.

Each experiment subclass implements:
  - load_data()        → DataSplit
  - build_first_stage() → nn.Module (trend model)
  - evaluate()          → dict of metrics

The base class provides:
  - run()              → full pipeline: data → Stage 1 → Stage 2 (unreg + Optuna reg) → eval → save
  - Unified CSV output  (long format)
  - Wall-time logging   (via TimerLog)
  - Seed iteration
"""

from __future__ import annotations

import csv
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spatial_adapter.metrics import compute_metrics, cov_frob_observed
from spatial_adapter.models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialAdapter,
    SpatialAdapterConfig,
    TrainingConfig,
)
from spatial_adapter.tuning import AdapterTuner
from spatial_adapter.utils.timer import TimerLog


# Data container
@dataclass
class DataSplit:
    """Holds everything an experiment needs after data loading."""

    train_loader: DataLoader
    train_cont: torch.Tensor  # (T_train, N, p) or (T_train, N, p)
    train_y: torch.Tensor  # (T_train, N)
    val_cont: torch.Tensor
    val_y: torch.Tensor
    test_cont: torch.Tensor
    test_y: torch.Tensor
    locs: np.ndarray  # (N,) or (N, 2)
    n_locations: int
    p_dim: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# Result record
RESULT_FIELDS = [
    "experiment",
    "seed",
    "model",
    "split",
    "metric",
    "value",
]


def _record(
    experiment: str, seed: int, model: str, metrics: dict, split: str = "test"
) -> List[dict]:
    """Convert a metrics dict into long-format result records."""
    rows = []
    for metric, value in metrics.items():
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "model": model,
                "split": split,
                "metric": metric,
                "value": value,
            }
        )
    return rows


# Adapter config from YAML dict
def build_adapter_config(cfg: dict) -> SpatialAdapterConfig:
    """Build SpatialAdapterConfig from a flat YAML dict."""
    admm = cfg.get("admm", {})
    training = cfg.get("training", {})
    basis = cfg.get("basis", {})
    return SpatialAdapterConfig(
        admm=ADMMConfig(
            **{k: v for k, v in admm.items() if k in ADMMConfig.__dataclass_fields__}
        ),
        training=TrainingConfig(
            **{
                k: v
                for k, v in training.items()
                if k in TrainingConfig.__dataclass_fields__
            }
        ),
        basis=BasisConfig(
            **{k: v for k, v in basis.items() if k in BasisConfig.__dataclass_fields__}
        ),
    )


# Base class
class BaseExperiment(ABC):
    """Abstract base for all Spatial Adapter experiments."""

    # Override in subclasses where the Stage-2 trend's learnable correction is
    # zero-initialised and should be learned only through ADMM (paper §3.4).
    stage2_pretrain: bool = True

    def __init__(self, config: dict):
        self.config = config
        self.exp_name: str = config["experiment"]["name"]
        seed_start: int = config["experiment"].get("seed_start", 1)
        n_replications: int = config["experiment"].get("n_replications", 10)
        self.seeds: List[int] = list(range(seed_start, seed_start + n_replications))
        self.n_optuna_trials: int = config["experiment"].get("n_optuna_trials", 30)
        self.output_dir = Path(
            config["experiment"].get("output_dir", f"results/{self.exp_name}")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_config = build_adapter_config(config.get("adapter", {}))

        self.data_cfg = config.get("data", {})
        self.latent_dim: int = self.data_cfg.get("latent_dim", 1)

        self.optuna_objective_key: str = config.get("experiment", {}).get(
            "optuna_objective", "rmse"
        )
        self.optuna_direction: str = config.get("experiment", {}).get(
            "optuna_direction", "minimize"
        )

        tau_range = config.get("adapter", {}).get("tau_range", [1e-4, 1e8])
        self.tau_min = tau_range[0]
        self.tau_max = tau_range[1]

        self.timer_log = TimerLog(str(self.output_dir / "wall_times.csv"))

        self.results: List[dict] = []

    # Abstract methods (subclass implements)
    @abstractmethod
    def load_data(self, seed: int) -> DataSplit:
        """Load and split data for one seed."""
        ...

    @abstractmethod
    def build_first_stage(self, data: DataSplit, seed: int) -> nn.Module:
        """Build and return the Stage-1 trend model (already trained/frozen)."""
        ...

    @abstractmethod
    def evaluate(
        self,
        trainer: SpatialAdapter,
        data: DataSplit,
        model_name: str,
    ) -> dict:
        """Evaluate a trained adapter and return a metrics dict."""
        ...

    # Shared pipeline
    def run(self):
        """Run the full experiment across all seeds."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.exp_name}")
        print(f"Seeds: {self.seeds}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for seed in self.seeds:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 1. Load data
            data = self.load_data(seed)

            # 2. Build first stage (timed)
            t0_s1 = time.perf_counter()
            trend = self.build_first_stage(data, seed)
            elapsed_s1 = time.perf_counter() - t0_s1
            self.timer_log.records.append(
                {
                    "function": "stage1",
                    "wall_time_s": round(elapsed_s1, 3),
                    "seed": seed,
                    "experiment": self.exp_name,
                }
            )
            if elapsed_s1 < 60:
                print(f"  [timer] stage1: {elapsed_s1:.2f}s")
            else:
                m, s = divmod(elapsed_s1, 60)
                print(f"  [timer] stage1: {int(m)}m {s:.1f}s")

            # 3. Evaluate first-stage only (baseline)
            baseline_metrics = self._evaluate_baseline(trend, data)
            self.results.extend(
                _record(self.exp_name, seed, "baseline", baseline_metrics)
            )
            print(f"  Baseline: {baseline_metrics}")

            # 4. Stage 2 — unregularized
            unreg_metrics = self._run_adapter(
                trend,
                data,
                seed,
                tau1=0.0,
                tau2=0.0,
                model_name="adapter_unreg",
            )
            self.results.extend(
                _record(self.exp_name, seed, "adapter_unreg", unreg_metrics)
            )
            print(f"  Unreg:    {unreg_metrics}")

            # 5. Stage 2 — Optuna-tuned regularized
            reg_metrics = self._run_optuna(trend, data, seed)
            self.results.extend(
                _record(self.exp_name, seed, "adapter_reg", reg_metrics)
            )
            print(f"  Reg:      {reg_metrics}")

        # Save
        self._save_results()
        self.timer_log.save()
        self.timer_log.summary()

        print(f"\nResults saved to {self.output_dir}")

    def _evaluate_baseline(self, trend: nn.Module, data: DataSplit) -> dict:
        """Evaluate the first-stage predictor without any adapter."""
        trend.eval()
        with torch.no_grad():
            y_pred = trend(data.test_cont.to(self.device))
        return self._compute_metrics(data.test_y, y_pred, data)

    def _compute_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, data: DataSplit
    ) -> dict:
        """Default regression metrics. Override for classification."""
        y_true_d = y_true.to(self.device)
        y_pred_d = y_pred.to(self.device)
        rmse, mae, r2 = compute_metrics(y_true_d, y_pred_d)
        metrics = {"rmse": round(rmse, 6), "mae": round(mae, 6), "r2": round(r2, 6)}

        # CovFrob if enough samples
        if y_true.shape[0] >= 3:
            try:
                cf = cov_frob_observed(
                    y_true.cpu().numpy(),
                    y_pred.detach().cpu().numpy(),
                )
                metrics["covfrob"] = round(cf, 6)
            except Exception:
                pass

        return metrics

    def _make_tuner(
        self, trend_template: nn.Module, data: DataSplit, seed: int
    ) -> AdapterTuner:
        """Build an AdapterTuner bound to this seed's data and evaluator."""

        def _eval(trainer: SpatialAdapter) -> dict:
            return self.evaluate(trainer, data, "")

        return AdapterTuner(
            trend_template=trend_template,
            train_loader=data.train_loader,
            val_cont=data.val_cont,
            val_y=data.val_y,
            locs=data.locs,
            n_locations=data.n_locations,
            latent_dim=self.latent_dim,
            adapter_config=self.adapter_config,
            evaluate_fn=_eval,
            tau_range=(self.tau_min, self.tau_max),
            n_trials=self.n_optuna_trials,
            objective_key=self.optuna_objective_key,
            direction=self.optuna_direction,
            stage2_pretrain=self.stage2_pretrain,
            device=self.device,
            study_name=f"{self.exp_name}_seed{seed}",
        )

    def _run_adapter(
        self,
        trend_template: nn.Module,
        data: DataSplit,
        seed: int,
        tau1: float,
        tau2: float,
        model_name: str,
        warm_start: bool = False,
    ) -> dict:
        """Train one adapter configuration via AdapterTuner and return metrics."""
        tuner = self._make_tuner(trend_template, data, seed)
        trainer, metrics = tuner.fit_one(tau1, tau2, warm_start=warm_start)
        elapsed = metrics["elapsed_s"]

        self._log_timer(model_name, elapsed, seed)

        del trainer, tuner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def _run_optuna(
        self, trend_template: nn.Module, data: DataSplit, seed: int
    ) -> dict:
        """Run Optuna hyperparameter search for (tau1, tau2) via AdapterTuner."""
        tuner = self._make_tuner(trend_template, data, seed)

        t0 = time.perf_counter()
        best_metrics = tuner.run_optuna(seed)
        elapsed = time.perf_counter() - t0
        per_trial = elapsed / max(self.n_optuna_trials, 1)

        self.timer_log.records.append(
            {
                "function": "adapter_reg_total",
                "wall_time_s": round(elapsed, 3),
                "seed": seed,
                "experiment": self.exp_name,
                "n_trials": self.n_optuna_trials,
            }
        )
        self.timer_log.records.append(
            {
                "function": "adapter_reg_per_trial",
                "wall_time_s": round(per_trial, 3),
                "seed": seed,
                "experiment": self.exp_name,
                "n_trials": self.n_optuna_trials,
            }
        )
        if elapsed < 60:
            print(
                f"  [timer] adapter_reg: {elapsed:.2f}s total, {per_trial:.2f}s/trial ({self.n_optuna_trials} trials)"
            )
        else:
            m, s = divmod(elapsed, 60)
            print(
                f"  [timer] adapter_reg: {int(m)}m {s:.1f}s total, {per_trial:.1f}s/trial ({self.n_optuna_trials} trials)"
            )

        del tuner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_metrics

    def _log_timer(self, name: str, elapsed_s: float, seed: int) -> None:
        """Append a wall-time record; skip Optuna trials (noisy)."""
        if name.startswith("trial_"):
            return
        self.timer_log.records.append(
            {
                "function": name,
                "wall_time_s": round(elapsed_s, 3),
                "seed": seed,
                "experiment": self.exp_name,
            }
        )
        if elapsed_s < 60:
            print(f"  [timer] {name}: {elapsed_s:.2f}s")
        else:
            m, s = divmod(elapsed_s, 60)
            print(f"  [timer] {name}: {int(m)}m {s:.1f}s")

    def _save_results(self):
        """Save all results to a long-format CSV."""
        csv_path = self.output_dir / "results.csv"
        if not self.results:
            print("[warning] No results to save.")
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(self.results)

        print(f"Saved {len(self.results)} records to {csv_path}")
