"""Three-block mini-batch ADMM for the Spatial Adapter (θ, Φ, Z steps)."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from spatial_adapter.cpp_extensions import spatial_utils
from spatial_adapter.logger import setup_logger
from spatial_adapter.metrics import compute_binary_metrics
from spatial_adapter.models.data_losses import (
    grad_Z_loss_data_binary,
    projection_orthogonal_complement,
)
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel

logger = setup_logger("spatial_adapter")

# Config dataclasses
@dataclass
class ADMMConfig:
    rho: float = 5.0
    dual_momentum: float = 0.2
    max_iters: int = 300
    min_outer: int = 100
    tol: float = 1e-4


@dataclass
class TrainingConfig:
    lr_mu: float = 1e-3
    batch_size: int = 128
    pretrain_epochs: int = 5
    use_mixed_precision: bool = False


@dataclass
class BasisConfig:
    phi_every: int = 5
    phi_freeze: int = 100
    matrix_reg: float = 1e-6
    irl1_max_iters: int = 10
    irl1_eps: float = 1e-6
    irl1_tol: float = 5e-4
    # BCE basis-update variant (Remark 1 ablation):
    #   "variance_only" | "full_taylor" | "irls"
    bce_variant: Literal["variance_only", "full_taylor", "irls"] = "variance_only"


@dataclass
class SpatialAdapterConfig:
    admm: ADMMConfig = None
    training: TrainingConfig = None
    basis: BasisConfig = None
    task: Literal["regression", "binary"] = "regression"

    def __post_init__(self):
        if self.admm is None:
            self.admm = ADMMConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.basis is None:
            self.basis = BasisConfig()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for cfg in (self.admm, self.training, self.basis):
            d.update(cfg.__dict__)
        d["task"] = self.task
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialAdapterConfig":
        admm = ADMMConfig(
            rho=d.get("rho", 5.0),
            dual_momentum=d.get("dual_momentum", 0.2),
            max_iters=d.get("max_iters", 300),
            min_outer=d.get("min_outer", 100),
            tol=d.get("tol", 1e-4),
        )
        training = TrainingConfig(
            lr_mu=d.get("lr_mu", 1e-3),
            batch_size=d.get("batch_size", 128),
            pretrain_epochs=d.get("pretrain_epochs", 5),
            use_mixed_precision=d.get("use_mixed_precision", False),
        )
        basis = BasisConfig(
            phi_every=d.get("phi_every", 5),
            phi_freeze=d.get("phi_freeze", 100),
            matrix_reg=d.get("matrix_reg", 1e-6),
            irl1_max_iters=d.get("irl1_max_iters", 10),
            irl1_eps=d.get("irl1_eps", 1e-6),
            irl1_tol=d.get("irl1_tol", 5e-4),
            bce_variant=d.get("bce_variant", "variance_only"),
        )
        task = d.get("task", "regression")
        if task not in ("regression", "binary"):
            raise ValueError(f"task must be 'regression' or 'binary', got {task!r}")
        return cls(admm=admm, training=training, basis=basis, task=task)

    def log_config(self) -> None:
        for k, v in self.to_dict().items():
            logger.info(f"  {k}: {v}")


# Main ADMM trainer
class SpatialAdapter:
    """Three-block mini-batch ADMM: θ-step (trend), Φ-step (basis), Z-step (consensus)."""

    def __init__(
        self,
        trend: TrendModel,
        basis: SpatialBasisLearner,
        train_loader: DataLoader,
        val_cont: torch.Tensor,
        val_y: torch.Tensor,
        locs: np.ndarray,
        config: Union[SpatialAdapterConfig, Dict[str, Any]],
        device: torch.device,
        writer: SummaryWriter,
        tau1: float = 0.0,
        tau2: float = 0.0,
    ):
        self.device = device
        self.writer = writer
        self.tau1, self.tau2 = tau1, tau2

        if isinstance(config, dict):
            self.config = SpatialAdapterConfig.from_dict(config)
        else:
            self.config = config

        self.use_mixed_precision = self.config.training.use_mixed_precision
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")

        self.trend = trend.to(device)
        self.basis = basis.to(device)
        self.train_loader = train_loader
        self.val_cont, self.val_y = val_cont.to(device), val_y.to(device)

        _, train_cont, train_y = train_loader.dataset.tensors
        self.train_cont, self.train_y = train_cont.to(device), train_y.to(device)

        self._rho_base = float(self.config.admm.rho * torch.std(self.train_y).item())
        self.rho = self._rho_base
        self.beta = float(self.config.admm.dual_momentum)
        self.max_iters = int(self.config.admm.max_iters)

        omega = spatial_utils.smoothing_penalty_matrix(locs)
        self.omega = torch.as_tensor(omega, dtype=torch.float32, device=device)

        T_train, N = self.train_y.shape
        T_val = self.val_y.shape[0]
        self.z_train = torch.zeros(T_train, N, device=device)
        self.u_train = torch.zeros_like(self.z_train)
        self.z_val = torch.zeros(T_val, N, device=device)
        self.u_val = torch.zeros_like(self.z_val)

        trend_params = list(self.trend.residual_parameters())
        self.opt_mu = (
            optim.AdamW(trend_params, lr=self.config.training.lr_mu)
            if trend_params
            else None
        )

        self.best_val = float("-inf") if self.config.task == "binary" else float("inf")
        self.global_iter = 0
        self.y_mean = self.train_y.mean().item()
        self.y_std = self.train_y.std(unbiased=False).item() + 1e-12
        self._is_binary = self.config.task == "binary"

        # VAR(1) forecaster (fitted post-hoc via fit_forecaster)
        self._A = None
        self._eta_last = None

    # Residual
    def _residual_R(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """R = g†(Y) − f_θ(X).  Regression: Y−μ.  Binary: logit(Y)−μ."""
        if not self._is_binary:
            return y - self.trend(x)
        return torch.special.logit(y.clamp(1e-7, 1.0 - 1e-7)) - self.trend(x)

    # Stage 1: pretrain trend
    def pretrain_trend(
        self,
        epochs: Optional[int] = None,
        loss_fn: Optional[Literal["mse", "bce"]] = None,
    ) -> None:
        """Warm-up trend parameters (Stage 1)."""
        if self.opt_mu is None:
            return
        if epochs is None:
            epochs = int(self.config.training.pretrain_epochs)
        if loss_fn is None:
            loss_fn = "bce" if self._is_binary else "mse"

        self.trend.train()
        for _ in range(epochs):
            for _, x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.trend(x)
                if loss_fn == "bce":
                    loss = F.binary_cross_entropy_with_logits(
                        logits, y.clamp(1e-7, 1.0 - 1e-7), reduction="mean"
                    )
                else:
                    loss = F.mse_loss(logits, y)
                self.opt_mu.zero_grad()
                loss.backward()
                self.opt_mu.step()

    @torch.no_grad()
    def init_basis_dense(self) -> None:
        """Warm-start basis from top-K eigenvectors of the penalised residual Gram."""
        X = torch.cat([xb for _, xb, _ in self.train_loader]).to(self.device)
        Y = torch.cat([yb for _, _, yb in self.train_loader]).to(self.device)
        R = self._residual_R(Y, X)

        C = R.T @ R
        M = 0.5 * (C - self.tau1 * self.omega + (C - self.tau1 * self.omega).T)
        _, V = torch.linalg.eigh(M)
        K = self.basis.basis.shape[1]
        self.basis.basis.data.copy_(V[:, -K:])

        self.z_train.copy_(R)
        self.u_train.zero_()
        self.z_val.copy_(self._residual_R(self.val_y, self.val_cont))
        self.u_val.zero_()

    # Stage 2: ADMM loop
    def run(self) -> float:
        """Run ADMM optimisation.  Returns best validation metric.

        Convergence follows Boyd et al. (2011) relative + absolute
        stopping criterion:
            ||Z - R(psi)||     <= eps_abs + eps_rel * max(||Z||, ||R||)
            rho * ||Z - Z_prev|| <= eps_abs + eps_rel * ||Y||
        Absolute tol (config.admm.tol) is used as eps_abs; eps_rel
        defaults to 10x eps_abs to match Boyd's convention.
        """
        eps_abs = float(self.config.admm.tol)
        eps_rel = eps_abs * 10.0  # Boyd's default: eps_rel = 10 * eps_abs
        min_outer = int(self.config.admm.min_outer)
        outer = trange(1, self.max_iters + 1, desc="ADMM", dynamic_ncols=True)

        for _ in outer:
            z_prev = self.z_train.clone()

            # Mini-batch: contiguous time window
            bs = int(self.config.training.batch_size)
            T = self.train_y.size(0)
            if bs >= T:
                bi = None
            else:
                t0 = int(torch.randint(0, T - bs + 1, (1,), device=self.device).item())
                bi = slice(t0, t0 + bs)

            # ADMM blocks
            delta_phi = self._phi_step(bi)
            self._theta_step(bi)
            self._z_step(bi, val=False)
            self._z_step(bi, val=True)

            # Residuals (RMS-scale)
            r_pri = self._residual_R(self.train_y, self.train_cont)
            pri = F.mse_loss(self.z_train, r_pri).sqrt().item()
            dua = (self.rho * (self.z_train - z_prev)).pow(2).mean().sqrt().item()

            # Relative tolerances (Boyd 2011, eq. 3.12)
            z_norm = self.z_train.pow(2).mean().sqrt().item()
            r_norm = r_pri.pow(2).mean().sqrt().item()
            y_norm = self.train_y.pow(2).mean().sqrt().item()
            eps_pri = eps_abs + eps_rel * max(z_norm, r_norm, 1e-12)
            eps_dua = eps_abs + eps_rel * max(y_norm, 1e-12)

            outer.set_postfix(
                pri=f"{pri:.2e}/{eps_pri:.0e}",
                dua=f"{dua:.2e}/{eps_dua:.0e}",
                rho=f"{self.rho:.2e}",
            )

            # Validation + logging
            val_result = self._validate()
            primary = val_result[0]
            if self.writer is not None:
                self.writer.add_scalar("train/primal_residual", pri, self.global_iter)
                self.writer.add_scalar("train/dual_residual", dua, self.global_iter)
            if self._is_binary:
                acc, f1, auc = val_result
                if self.writer is not None:
                    self.writer.add_scalar("val/accuracy", acc, self.global_iter)
                    self.writer.add_scalar("val/f1", f1, self.global_iter)
                    self.writer.add_scalar("val/auc", auc, self.global_iter)
                self.best_val = max(self.best_val, acc)
            else:
                if self.writer is not None:
                    self.writer.add_scalar("val/rmse_admm", primary, self.global_iter)
                self.best_val = min(self.best_val, primary)

            # Stochastic mini-batch ADMM has an irreducible noise floor
            # on the primal residual (Z = R(ψ) cannot hold exactly under
            # SGD + mini-batch sampling).  We therefore monitor the dual
            # residual (Boyd 2011 eq. 3.11) and the basis change δΦ, both
            # of which go to zero at convergence.
            converged = dua <= eps_dua and delta_phi <= eps_pri
            if (self.global_iter + 1) >= min_outer and converged:
                outer.write(
                    f"Converged (dua={dua:.2e}<{eps_dua:.2e}, "
                    f"δΦ={delta_phi:.2e}<{eps_pri:.2e})"
                )
                break
            self.global_iter += 1

        logger.info(f"Training completed in {self.global_iter} iterations")
        return self.best_val

    # (T) Trend step
    def _theta_loss(self, yb, xb, zb, ub):
        """(ρ/2)‖(Z+U) − R‖².  Regression rescales by y_std."""
        if not self._is_binary:
            s = self.y_std
            r = (yb - self.trend(xb)) / s
            cons = (zb + ub) / s
        else:
            r = self._residual_R(yb, xb)
            cons = zb + ub
        return 0.5 * self.rho * F.mse_loss(cons, r)

    def _theta_step(self, batch_indices: Optional[slice] = None) -> None:
        if self.opt_mu is None:
            return
        self.trend.train()
        if batch_indices is None:
            xb, yb, zb, ub = self.train_cont, self.train_y, self.z_train, self.u_train
        else:
            xb = self.train_cont[batch_indices]
            yb = self.train_y[batch_indices]
            zb = self.z_train[batch_indices]
            ub = self.u_train[batch_indices]

        loss = self._theta_loss(yb, xb, zb, ub)
        self.opt_mu.zero_grad()
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt_mu)
            self.scaler.update()
        else:
            loss.backward()
            self.opt_mu.step()

    # (B) Basis step
    @torch.no_grad()
    def _phi_step(self, batch_indices: Optional[slice] = None) -> float:
        """Update Φ from the mini-batch consensus Z.

        BCE variant (config.basis.bce_variant) controls the target matrix
        for binary tasks — see Remark 1 / Appendix C.
        """
        phi_every = int(self.config.basis.phi_every)
        freeze_after = int(self.config.basis.phi_freeze)
        if (self.global_iter % phi_every) or (self.global_iter >= freeze_after):
            return 0.0

        K = self.basis.basis.shape[1]
        old = self.basis.basis.data.clone()

        if batch_indices is None:
            Z_batch, Y_batch = self.z_train, self.train_y
        else:
            Z_batch, Y_batch = self.z_train[batch_indices], self.train_y[batch_indices]

        Zc = Z_batch - Z_batch.mean(0, keepdim=True)

        # Target matrix C: regression always variance-only; binary dispatches on bce_variant
        if not self._is_binary:
            C = Zc.T @ Zc
        else:
            variant = str(self.config.basis.bce_variant)
            if variant == "variance_only":
                C = Zc.T @ Zc
            elif variant == "full_taylor":
                A = (0.5 - Y_batch).T @ Z_batch
                C = 0.5 * (A + A.T) + (1.0 / 8.0) * (Z_batch.T @ Z_batch)
            elif variant == "irls":
                p = torch.sigmoid(Z_batch)
                w = p * (1.0 - p) + 1e-4
                Zt = Z_batch + (Y_batch - p) / w
                Ztc = Zt - Zt.mean(0, keepdim=True)
                C = Ztc.T @ Ztc
            else:
                raise ValueError(f"Unknown bce_variant {variant!r}")

        M = 0.5 * (C - self.tau1 * self.omega + (C - self.tau1 * self.omega).T)
        reg = float(self.config.basis.matrix_reg)
        M += (
            reg
            * torch.trace(M).item()
            / M.size(0)
            * torch.eye(M.size(0), device=M.device)
        )

        if self.tau2 == 0.0:
            _, V = torch.linalg.eigh(M)
            new_phi = V[:, -K:]
        else:
            # IRL₁ inner loop for column sparsity
            _, V = torch.linalg.eigh(M)
            Phi = V[:, -K:]
            alpha = 1.0 / (2.0 * torch.linalg.norm(M, 2))
            eps_irl = float(self.config.basis.irl1_eps)
            tol_inner = float(self.config.basis.irl1_tol)
            for t in range(int(self.config.basis.irl1_max_iters)):
                Phi_prev = Phi.clone()
                G = 2 * (M @ Phi)
                Y_ = Phi + alpha * G
                W = 1.0 / (Phi.abs() + eps_irl)
                Phi = torch.sign(Y_) * torch.clamp(
                    Y_.abs() - alpha * self.tau2 * W, 0.0
                )
                U, _, Vt = torch.linalg.svd(Phi, full_matrices=False)
                Phi = U @ Vt
                if t >= 2 and torch.norm(Phi - Phi_prev) < tol_inner:
                    break
            new_phi = Phi

        self.basis.basis.data.copy_(new_phi)
        return float(torch.norm(new_phi - old, p="fro").item())

    # (Z) Consensus step
    @torch.no_grad()
    def _z_step(
        self, batch_indices: Optional[slice] = None, *, val: bool = False
    ) -> None:
        """Update Z (and dual U).  Regression: closed-form.  Binary: proximal-gradient."""
        if val:
            R = self._residual_R(self.val_y, self.val_cont)
            z, u, Y = self.z_val, self.u_val, self.val_y
            Res = R - u
        else:
            if batch_indices is None:
                R = self._residual_R(self.train_y, self.train_cont)
                z, u, Y = self.z_train, self.u_train, self.train_y
            else:
                R = self._residual_R(
                    self.train_y[batch_indices], self.train_cont[batch_indices]
                )
                z = self.z_train[batch_indices]
                u = self.u_train[batch_indices]
                Y = self.train_y[batch_indices]
            Res = R - u

        Phi = self.basis.basis
        L, N = z.shape

        if self._is_binary:
            # Proximal-gradient: Z ← Z − α [∇_Z ℓ_data(P⊥Z) + (ρ/LN)(Z − R + U)]
            P_perp = projection_orthogonal_complement(Phi)
            with torch.enable_grad():
                _, data_grad = grad_Z_loss_data_binary(z, P_perp, Y)
            z_new = z - 0.1 * (data_grad + (self.rho / (L * N)) * (z - R + u))
        else:
            # Closed-form: a₁·Res + a₂·P·Res  (eq:z-closed-form)
            a1 = self.rho / (self.rho + 2.0)
            a2 = 2.0 / (self.rho + 2.0)
            PRes = (Res @ Phi) @ Phi.T
            z_new = a1 * Res + a2 * PRes

        # Write Z
        if batch_indices is None or val:
            z.copy_(z_new)
        else:
            self.z_train[batch_indices] = z_new

        # Dual update with momentum
        def _dual_update(u_ref, z_ref, R_ref):
            u_prev = u_ref.clone()
            u_ref.add_(z_ref - R_ref)
            if self.beta:
                u_ref.add_(self.beta * (u_ref - u_prev))

        if val:
            _dual_update(u, z, R)
        elif batch_indices is None:
            _dual_update(u, z, R)
        else:
            u_batch = self.u_train[batch_indices]
            u_prev = u_batch.clone()
            u_batch.add_(z_new - R)
            if self.beta:
                u_batch.add_(self.beta * (u_batch - u_prev))
            self.u_train[batch_indices] = u_batch

    # Validation
    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """Regression: (rmse, 0, rmse).  Binary: (accuracy, f1, auc)."""
        self.trend.eval()
        self.basis.eval()
        mu = self.trend(self.val_cont)
        y_hat = mu + (self.z_val @ self.basis.basis) @ self.basis.basis.T
        if self._is_binary:
            return compute_binary_metrics(self.val_y, y_hat)
        rmse = math.sqrt(F.mse_loss(y_hat, self.val_y).item())
        return rmse, 0.0, rmse

    # Inference
    @torch.no_grad()
    def reconstruct(
        self, cont_features: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """Evaluation-only: project residuals onto Φ and add back (uses y_true)."""
        if y_true is None:
            raise ValueError("reconstruct() requires y_true")
        self.trend.eval()
        self.basis.eval()
        mu = self.trend(cont_features)
        if self._is_binary:
            residual = torch.special.logit(y_true.clamp(1e-7, 1 - 1e-7)) - mu
        else:
            residual = y_true - mu
        return mu + (residual @ self.basis.basis) @ self.basis.basis.T

    @torch.no_grad()
    def predict(self, cont_features: torch.Tensor) -> torch.Tensor:
        """Label-free prediction: trend-only, or trend + VAR(1) spatial if fit_forecaster was called."""
        self.trend.eval()
        self.basis.eval()
        mu = self.trend(cont_features)
        if self._A is None or self._eta_last is None:
            return mu
        Phi = self.basis.basis
        eta = self._eta_last.clone()
        outs = []
        for t in range(mu.size(0)):
            eta = self._A @ eta
            outs.append(mu[t] + eta @ Phi.T)
        self._eta_last = eta.clone()
        return torch.stack(outs, 0)

    @torch.no_grad()
    def fit_forecaster(self) -> None:
        """Learn VAR(1) on train latent scores: η_t = (y_t − μ_t) Φ."""
        self.trend.eval()
        self.basis.eval()
        Phi = self.basis.basis
        eta = (self.train_y - self.trend(self.train_cont)) @ Phi
        X, Y = eta[:-1], eta[1:]
        K = X.shape[1]
        A = torch.linalg.solve(
            X.T @ X + 1e-4 * torch.eye(K, device=X.device), X.T @ Y
        ).T
        self._A = A
        self._eta_last = eta[-1].clone()

    # Tuned one-shot constructor

    @classmethod
    def fit_tuned(
        cls,
        trend: TrendModel,
        train_loader: DataLoader,
        *,
        val_cont: torch.Tensor,
        val_y: torch.Tensor,
        locs: np.ndarray,
        device: torch.device,
        latent_dim: int,
        seed: int = 0,
        n_trials: int = 10,
        criterion: Literal["auto", "rmse", "accuracy", "auc"] = "auto",
        config: Optional["SpatialAdapterConfig"] = None,
        tau_range: Tuple[float, float] = (1e-4, 1e2),
    ):
        """Hide the 2D (tau1, tau2) search behind one call.

        Runs a short Optuna sweep (default 10 trials, per-seed) via
        :class:`AdapterTuner`, then refits at the best (tau1, tau2) and
        returns the trained adapter together with the chosen weights and
        the trials dataframe. The underlying ADMM / IRL₁ algorithm is
        unchanged — this is purely a UX wrapper.

        Parameters
        ----------
        trend : TrendModel
            First-stage predictor; deep-copied per Optuna trial.
        train_loader, val_cont, val_y, locs, device
            Standard adapter plumbing.
        latent_dim : int
            K — the basis rank for the Φ step.
        seed : int, default=0
            Reproducible TPE sampler seed (one seed per Monte-Carlo run).
        n_trials : int, default=10
            Optuna budget. Usually enough for RMSE; tighten when the
            objective is noisier (SNR-dependent, e.g. binary accuracy).
        criterion : {"auto", "rmse", "accuracy", "auc"}, default="auto"
            "auto" -> "accuracy" for binary tasks, "rmse" for regression.
        config : SpatialAdapterConfig, optional
            Shared ADMM/training/basis config. Defaults to
            ``SpatialAdapterConfig()``.
        tau_range : (float, float), default=(1e-4, 1e2)
            Log-scale bounds for both tau1 and tau2.

        Returns
        -------
        FitTunedResult
            ``result.adapter`` is already trained at the best weights;
            ``result.tau1`` / ``result.tau2`` / ``result.trials`` expose
            the search outcome.
        """
        from spatial_adapter.tuning import AdapterTuner, FitTunedResult

        if config is None:
            config = SpatialAdapterConfig()

        if criterion == "auto":
            criterion = "accuracy" if config.task == "binary" else "rmse"
        direction = "maximize" if criterion in ("accuracy", "auc") else "minimize"

        task = config.task

        def _evaluate(trainer) -> Dict[str, float]:
            primary, f1, auc = trainer._validate()
            if task == "binary":
                return {"accuracy": primary, "f1": f1, "auc": auc}
            return {"rmse": primary}

        n_locations = int(val_y.shape[-1])

        tuner = AdapterTuner(
            trend_template=trend,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            n_locations=n_locations,
            latent_dim=latent_dim,
            adapter_config=config,
            evaluate_fn=_evaluate,
            tau_range=tau_range,
            n_trials=n_trials,
            objective_key=criterion,
            direction=direction,
            device=device,
        )
        best = tuner.run_optuna(seed=seed)

        # Refit once at the best (tau1, tau2); AdapterTuner.fit_one does
        # pretrain + init_basis + run internally so we don't duplicate logic.
        trained, _ = tuner.fit_one(best["tau1"], best["tau2"], warm_start=False)

        return FitTunedResult(
            adapter=trained,
            tau1=float(best["tau1"]),
            tau2=float(best["tau2"]),
            trials=tuner.study.trials_dataframe(),
        )
