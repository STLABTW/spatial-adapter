"""Optuna-based tuning of (tau1, tau2) with warm-start cache.

Each trial deep-copies a frozen trend template, optionally warm-starts trend
and basis from the nearest cached (tau1, tau2) in log-space, trains the
adapter via ADMM, and evaluates with a user-supplied callable. The cache
persists across trials within a single ``run_optuna`` call.
"""


import copy
import time
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from spatial_adapter.models.spatial_adapter import (
    SpatialNeuralAdapter,
    SpatialNeuralAdapterConfig,
)
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.tuning.model_cache import ModelCache

EvaluateFn = Callable[[SpatialNeuralAdapter], Dict[str, float]]


class AdapterTuner:
    """Tune (tau1, tau2) for SpatialNeuralAdapter via Optuna with warm-start cache.

    Parameters
    ----------
    trend_template : nn.Module
        Frozen trend model to deep-copy for each trial. The template itself is
        never mutated.
    train_loader, val_cont, val_y, locs, n_locations, latent_dim :
        Data plumbing forwarded to :class:`SpatialNeuralAdapter`.
    adapter_config : SpatialNeuralAdapterConfig
        ADMM / training / basis configuration, shared across trials.
    evaluate_fn : Callable[[SpatialNeuralAdapter], Dict[str, float]]
        Called after ``trainer.run()`` to produce the metrics dict. The
        ``objective_key`` must appear in the returned dict.
    tau_range : (float, float), default=(1e-4, 1e2)
        Log-scale search bounds for both tau1 and tau2.
    n_trials : int, default=30
    objective_key : str, default="rmse"
        Key in the metrics dict used as the Optuna objective.
    direction : {"minimize", "maximize"}, default="minimize"
    stage2_pretrain : bool, default=True
        If False, skip ``trainer.pretrain_trend`` in Stage 2. Use when the
        trend's learnable correction is zero-initialised and should be learned
        only through ADMM (paper §3.4 two-stage trend).
    device : torch.device, optional
    study_name : str, optional
        Forwarded to ``optuna.create_study``.
    """

    def __init__(
        self,
        trend_template: nn.Module,
        train_loader: DataLoader,
        val_cont: torch.Tensor,
        val_y: torch.Tensor,
        locs: np.ndarray,
        n_locations: int,
        latent_dim: int,
        adapter_config: SpatialNeuralAdapterConfig,
        *,
        evaluate_fn: EvaluateFn,
        tau_range: Tuple[float, float] = (1e-4, 1e2),
        n_trials: int = 30,
        objective_key: str = "rmse",
        direction: Literal["minimize", "maximize"] = "minimize",
        stage2_pretrain: bool = True,
        device: Optional[torch.device] = None,
        study_name: Optional[str] = None,
    ):
        self.trend_template = trend_template
        self.train_loader = train_loader
        self.val_cont = val_cont
        self.val_y = val_y
        self.locs = locs
        self.n_locations = n_locations
        self.latent_dim = latent_dim
        self.adapter_config = adapter_config
        self.evaluate_fn = evaluate_fn
        self.tau_min, self.tau_max = tau_range
        self.n_trials = n_trials
        self.objective_key = objective_key
        self.direction = direction
        self.stage2_pretrain = stage2_pretrain
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.study_name = study_name

        self.cache = ModelCache()

    def fit_one(
        self,
        tau1: float,
        tau2: float,
        *,
        warm_start: bool = False,
    ) -> Tuple[SpatialNeuralAdapter, Dict[str, float]]:
        """Train one adapter configuration.

        Returns
        -------
        trainer : SpatialNeuralAdapter
            The trained trainer. Caller is responsible for freeing it.
        metrics : dict
            ``evaluate_fn`` output plus ``tau1``, ``tau2``, ``elapsed_s``.
        """
        trend = copy.deepcopy(self.trend_template).to(self.device)
        basis = SpatialBasisLearner(self.n_locations, self.latent_dim).to(self.device)

        warm_started = False
        if warm_start and self.cache.size() > 0:
            self.cache.load_nearest(trend, basis, tau1, tau2)
            warm_started = True

        trainer = SpatialNeuralAdapter(
            trend,
            basis,
            self.train_loader,
            val_cont=self.val_cont.to(self.device),
            val_y=self.val_y.to(self.device),
            locs=self.locs,
            config=self.adapter_config,
            device=self.device,
            writer=None,
            tau1=tau1,
            tau2=tau2,
        )

        t0 = time.perf_counter()
        if not warm_started:
            if self.stage2_pretrain:
                trainer.pretrain_trend(
                    epochs=self.adapter_config.training.pretrain_epochs
                )
            trainer.init_basis_dense()
        trainer.run()
        elapsed = time.perf_counter() - t0

        if warm_start:
            self.cache.store(
                tau1,
                tau2,
                copy.deepcopy(trend.state_dict()),
                copy.deepcopy(basis.state_dict()),
            )

        metrics = dict(self.evaluate_fn(trainer))
        metrics["tau1"] = tau1
        metrics["tau2"] = tau2
        metrics["elapsed_s"] = round(elapsed, 3)
        return trainer, metrics

    def run_optuna(self, seed: int) -> Dict[str, float]:
        """Run Optuna sweep over (tau1, tau2).

        The cache is cleared at the start so different seeds do not share
        warm-start state. Returns the metrics dict of the best trial.
        """
        self.cache.clear()

        best_metrics: Dict[str, float] = {}
        worst = float("inf") if self.direction == "minimize" else float("-inf")
        best_val = worst

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_metrics, best_val
            tau1 = trial.suggest_float("tau1", self.tau_min, self.tau_max, log=True)
            tau2 = trial.suggest_float("tau2", self.tau_min, self.tau_max, log=True)

            trainer, metrics = self.fit_one(tau1, tau2, warm_start=True)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            val = metrics.get(self.objective_key, worst)
            improved = (
                val < best_val if self.direction == "minimize" else val > best_val
            )
            if improved:
                best_val = val
                best_metrics = metrics.copy()
            return val

        study = optuna.create_study(
            study_name=self.study_name or f"adapter_tuner_seed{seed}",
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=MedianPruner(n_warmup_steps=3),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials)

        return best_metrics

    def clear_cache(self) -> None:
        self.cache.clear()
