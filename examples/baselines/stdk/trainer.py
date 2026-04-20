"""
Trainer and evaluate_model for spatio-temporal interpolation.
Encapsulates training loop, validation, early stopping, EMA, and evaluation metrics.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import (
    check_loss_numpy,
    compute_crps_multi_quantile,
    compute_p_nc_delta_penalty,
    compute_p_nc_delta_penalty_conditional,
    compute_picp,
    compute_qice,
    non_crossing_penalty,
    quantile_loss,
)
from .utils.ema import ModelEMA


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Returns:
        metrics: dict with 'mse', 'mae', 'rmse', and optionally 'check_loss', 'crps'
        for quantile/multi-quantile.
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []

    if len(data_loader) == 0 or len(data_loader.dataset) == 0:
        regression_type = (
            config.get("regression_type", "mean") if config is not None else "mean"
        )
        if regression_type == "multi-quantile":
            return {
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "crps": 0.0,
                "mean_check_loss": 0.0,
                "check_loss": 0.0,
                "coverage_90": 0.0,
            }
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "check_loss": 0.0}

    with torch.no_grad():
        for batch in data_loader:
            X = batch["X"].to(device)
            coords = batch["coords"].to(device)
            t = batch["t"].to(device)
            y = batch["y"].to(device)
            y_pred = model(X, coords, t)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    if len(all_preds) == 0:
        regression_type = (
            config.get("regression_type", "mean") if config is not None else "mean"
        )
        if regression_type == "multi-quantile":
            return {
                "mse": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "crps": 0.0,
                "mean_check_loss": 0.0,
                "check_loss": 0.0,
                "coverage_90": 0.0,
            }
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "check_loss": 0.0}

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    regression_type = (
        config.get("regression_type", "mean") if config is not None else "mean"
    )

    if regression_type == "multi-quantile":
        quantile_levels = config.get("quantile_levels", [0.05, 0.25, 0.5, 0.75, 0.95])
        median_idx = len(quantile_levels) // 2
        preds_for_metrics = preds[:, median_idx : median_idx + 1]
    else:
        preds_for_metrics = preds

    mse = np.mean((preds_for_metrics - trues) ** 2)
    mae = np.mean(np.abs(preds_for_metrics - trues))
    metrics: Dict[str, float] = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
    }

    if (
        config is not None
        and config.get("regression_type") == "quantile"
        and config.get("current_quantile") is not None
    ):
        q = config["current_quantile"]
        check_loss = quantile_loss(
            torch.tensor(preds, dtype=torch.float32),
            torch.tensor(trues, dtype=torch.float32),
            q,
        ).item()
        metrics["check_loss"] = float(check_loss)

    if config is not None and config.get("regression_type") == "multi-quantile":
        quantile_levels = config.get("quantile_levels", [0.05, 0.25, 0.5, 0.75, 0.95])
        crps_weighting = config.get("crps_weighting", "trapezoidal")
        metrics["crps"] = float(
            compute_crps_multi_quantile(
                preds, trues, quantile_levels, crps_weighting=crps_weighting
            )
        )
        metrics["crps_weighting"] = crps_weighting
        check_losses = []
        for q_idx, q_level in enumerate(quantile_levels):
            q_pred = preds[:, q_idx : q_idx + 1]
            check_loss_q = quantile_loss(
                torch.tensor(q_pred, dtype=torch.float32),
                torch.tensor(trues, dtype=torch.float32),
                q_level,
            ).item()
            check_losses.append(check_loss_q)
        metrics["mean_check_loss"] = float(np.mean(check_losses))
        metrics["check_loss"] = float(np.mean(check_losses))
        metrics["picp_90"] = float(
            compute_picp(preds, trues, quantile_levels, alpha=0.1)
        )
        metrics["coverage_90"] = metrics["picp_90"]
        metrics["qice"] = float(compute_qice(preds, trues, quantile_levels, M=4))

    return metrics


class Trainer:
    """
    Trains a spatio-temporal interpolation model with optional EMA, progressive
    unfreezing, warmup, and early stopping. Use .fit() to run the full loop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Any,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)

        basis_unfreeze_epoch = config.get("basis_unfreeze_epoch", 0)
        config.get("basis_lr_rampup_epochs", 0)
        spatial_learnable = config.get("spatial_learnable", False)
        lr = float(config.get("lr", 1e-3))
        basis_lr_ratio = config.get("basis_lr_ratio", 0.05)

        if spatial_learnable:
            basis_params = list(model.spatial_basis.parameters())
            mlp_params = [
                p for p in model.parameters() if not any(p is bp for bp in basis_params)
            ]
            initial_basis_lr = 0.0 if basis_unfreeze_epoch > 0 else lr * basis_lr_ratio
            self.optimizer = optim.AdamW(
                [
                    {"params": mlp_params, "lr": lr, "name": "mlp"},
                    {"params": basis_params, "lr": initial_basis_lr, "name": "basis"},
                ],
                weight_decay=float(config.get("weight_decay", 1e-5)),
            )
            for pg in self.optimizer.param_groups:
                pg["initial_lr"] = pg["lr"]
                if pg.get("name") == "basis":
                    pg["target_lr"] = lr * basis_lr_ratio
        else:
            mlp_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(
                mlp_params,
                lr=lr,
                weight_decay=float(config.get("weight_decay", 1e-5)),
            )
            for pg in self.optimizer.param_groups:
                pg["initial_lr"] = pg["lr"]

        warmup_epochs = config.get("warmup_epochs", 0)
        self.warmup_steps = (
            warmup_epochs * len(train_loader) if warmup_epochs > 0 else 0
        )
        self.epochs = config.get("epochs", 100)
        self.scheduler = None
        if config.get("scheduler") == "cosine":
            eta_min = lr * 0.5
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=eta_min,
            )
        self.warmup_epochs = warmup_epochs

        use_ema = config.get("use_ema", True)
        self.ema = None
        if use_ema:
            batches_per_epoch = len(train_loader)
            ema_decay = 1.0 - 1.0 / (10.0 * batches_per_epoch)
            self.ema = ModelEMA(model, decay=ema_decay)

        self.regression_type = config.get("regression_type", "mean")
        self.current_quantile = config.get("current_quantile", None)
        self.quantile_levels = config.get(
            "quantile_levels", [0.05, 0.25, 0.5, 0.75, 0.95]
        )
        if self.regression_type == "mean":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None

        self.history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "lr": [],
        }
        self.basis_centers_history: List[Tuple[int, np.ndarray]] = []
        self.center_record_interval = 100
        self.best_model_path = self.output_dir / "model_best.pt"
        self.global_step = 0

    def fit(self) -> Tuple[nn.Module, Dict[str, List], List[Tuple[int, np.ndarray]]]:
        """
        Run full training loop with early stopping. Returns (model, history, basis_centers_history).
        """
        config = self.config
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        device = self.device
        regression_type = self.regression_type
        current_quantile = self.current_quantile
        quantile_levels = self.quantile_levels
        basis_unfreeze_epoch = config.get("basis_unfreeze_epoch", 0)
        basis_lr_rampup_epochs = config.get("basis_lr_rampup_epochs", 0)
        patience = config.get("patience", 15)
        use_check_loss_early_stop = config.get("use_check_loss_early_stop", False)

        best_val_loss = float("inf")
        patience_counter = 0
        best_val_check_loss = float("inf")  # used when use_check_loss_early_stop

        for epoch in range(self.epochs):
            # Progressive unfreezing
            if config.get("spatial_learnable", False) and basis_unfreeze_epoch > 0:
                if epoch == basis_unfreeze_epoch:
                    for param_group in self.optimizer.param_groups:
                        if param_group.get("name") == "basis":
                            if basis_lr_rampup_epochs > 0:
                                param_group["lr"] = param_group["target_lr"] * 0.1
                            else:
                                param_group["lr"] = param_group["target_lr"]
                elif (
                    basis_unfreeze_epoch
                    < epoch
                    < basis_unfreeze_epoch + basis_lr_rampup_epochs
                ):
                    rampup_progress = (
                        epoch - basis_unfreeze_epoch
                    ) / basis_lr_rampup_epochs
                    for param_group in self.optimizer.param_groups:
                        if param_group.get("name") == "basis":
                            param_group["lr"] = param_group["target_lr"] * (
                                0.1 + 0.9 * rampup_progress
                            )

            model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                X = batch["X"].to(device)
                coords = batch["coords"].to(device)
                t = batch["t"].to(device)
                y = batch["y"].to(device)
                self.optimizer.zero_grad()
                y_pred = model(X, coords, t)

                if regression_type == "mean":
                    loss = self.criterion(y_pred, y)
                elif regression_type == "quantile":
                    loss = quantile_loss(y_pred, y, current_quantile)
                else:
                    losses = []
                    for q_idx, q_level in enumerate(quantile_levels):
                        q_pred = y_pred[:, q_idx : q_idx + 1]
                        losses.append(quantile_loss(q_pred, y, q_level))
                    loss = torch.mean(torch.stack(losses))
                    use_delta_reparam = config.get(
                        "use_delta_reparameterization", False
                    )
                    if use_delta_reparam:
                        non_crossing_lambda = config.get("non_crossing_lambda", 0.0)
                        if non_crossing_lambda > 0:
                            delta_params = model.get_delta_parameters()
                            if delta_params is not None:
                                l1_penalization = config.get(
                                    "non_crossing_l1_penalization", False
                                )
                                if l1_penalization:
                                    p_nc = compute_p_nc_delta_penalty_conditional(
                                        delta_params, use_positive_penalty=True
                                    )
                                else:
                                    p_nc = compute_p_nc_delta_penalty(
                                        delta_params,
                                        use_positive_penalty=config.get(
                                            "use_positive_p_nc_penalty", False
                                        ),
                                    )
                                    loss = loss + non_crossing_lambda * p_nc
                    else:
                        non_crossing_weight = config.get("non_crossing_weight", 0.0)
                        if non_crossing_weight > 0:
                            nc_power = int(config.get("non_crossing_power", 1))
                            loss = loss + non_crossing_weight * non_crossing_penalty(
                                y_pred, reduction="mean", power=nc_power
                            )

                if config.get("spatial_learnable", False):
                    domain_penalty_weight = config.get("domain_penalty_weight", 0.0)
                    if domain_penalty_weight > 0:
                        loss = (
                            loss
                            + domain_penalty_weight * model.compute_domain_penalty()
                        )
                    movement_penalty_weight = config.get("movement_penalty_weight", 0.0)
                    if movement_penalty_weight > 0:
                        loss = (
                            loss
                            + movement_penalty_weight * model.compute_movement_penalty()
                        )

                sparsity_type = config.get("sparsity_penalty_type", "none")
                if sparsity_type != "none":
                    penalties = model.compute_sparsity_penalty(
                        penalty_type=sparsity_type,
                        lambda_l1=config.get("sparsity_lambda_l1", 0.001),
                        lambda_group=config.get("sparsity_lambda_group", 0.01),
                    )
                    if config.get("sparsity_apply_to_spatial", True):
                        loss = loss + penalties["spatial_penalty"]
                    if config.get("sparsity_apply_to_temporal", True):
                        loss = loss + penalties["temporal_penalty"]

                loss.backward()
                if config.get("grad_clip", 0) > 0:
                    if config.get("spatial_learnable", False):
                        basis_params = list(model.spatial_basis.parameters())
                        mlp_params = [
                            p
                            for p in model.parameters()
                            if not any(p is bp for bp in basis_params)
                        ]
                        torch.nn.utils.clip_grad_norm_(
                            basis_params, config["grad_clip"] * 0.1
                        )
                        torch.nn.utils.clip_grad_norm_(mlp_params, config["grad_clip"])
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["grad_clip"]
                        )
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(model)
                if self.global_step < self.warmup_steps:
                    warmup_factor = (self.global_step + 1) / self.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = param_group["initial_lr"] * warmup_factor
                self.global_step += 1
                train_loss += loss.item()
                if np.isnan(loss.item()):
                    break

            train_loss /= len(train_loader)
            model.eval()
            if self.ema is not None:
                self.ema.apply_shadow()
            val_loss = 0.0
            val_preds: List[np.ndarray] = []
            val_trues: List[np.ndarray] = []
            has_validation = len(val_loader) > 0
            with torch.no_grad():
                if has_validation:
                    for batch in val_loader:
                        X = batch["X"].to(device)
                        coords = batch["coords"].to(device)
                        t = batch["t"].to(device)
                        y = batch["y"].to(device)
                        y_pred = model(X, coords, t)
                        if regression_type == "mean":
                            loss = self.criterion(y_pred, y)
                        elif regression_type == "quantile":
                            loss = quantile_loss(y_pred, y, current_quantile)
                        else:
                            losses = []
                            for q_idx, q_level in enumerate(quantile_levels):
                                losses.append(
                                    quantile_loss(
                                        y_pred[:, q_idx : q_idx + 1], y, q_level
                                    )
                                )
                            loss = torch.mean(torch.stack(losses))
                            use_delta_reparam = config.get(
                                "use_delta_reparameterization", False
                            )
                            if (
                                use_delta_reparam
                                and config.get("non_crossing_lambda", 0.0) > 0
                            ):
                                delta_params = model.get_delta_parameters()
                                if delta_params is not None:
                                    if config.get(
                                        "non_crossing_l1_penalization", False
                                    ):
                                        loss = loss + config.get(
                                            "non_crossing_lambda", 0.0
                                        ) * compute_p_nc_delta_penalty_conditional(
                                            delta_params, use_positive_penalty=True
                                        )
                                    else:
                                        loss = loss + config.get(
                                            "non_crossing_lambda", 0.0
                                        ) * compute_p_nc_delta_penalty(
                                            delta_params,
                                            use_positive_penalty=config.get(
                                                "use_positive_p_nc_penalty", False
                                            ),
                                        )
                            elif config.get("non_crossing_weight", 0.0) > 0:
                                loss = loss + config.get(
                                    "non_crossing_weight", 0.0
                                ) * non_crossing_penalty(
                                    y_pred,
                                    reduction="mean",
                                    power=int(config.get("non_crossing_power", 1)),
                                )
                        val_loss += loss.item()
                        val_preds.append(y_pred.cpu().numpy())
                        val_trues.append(y.cpu().numpy())
                else:
                    val_loss = train_loss
            if self.ema is not None:
                self.ema.restore()

            if has_validation:
                val_loss /= len(val_loader)
            else:
                val_loss = train_loss
            if has_validation and len(val_preds) > 0:
                val_preds_c = np.concatenate(val_preds, axis=0)
                val_trues_c = np.concatenate(val_trues, axis=0)
                if regression_type == "multi-quantile":
                    median_idx = len(quantile_levels) // 2
                    val_preds_for_rmse = val_preds_c[:, median_idx : median_idx + 1]
                else:
                    val_preds_for_rmse = val_preds_c
                val_rmse = float(
                    np.sqrt(np.mean((val_preds_for_rmse - val_trues_c) ** 2))
                )
                val_check_loss = None
                if use_check_loss_early_stop and regression_type == "multi-quantile":
                    check_losses = []
                    for q_idx, q_level in enumerate(quantile_levels):
                        check_losses.append(
                            check_loss_numpy(
                                val_preds_c[:, q_idx : q_idx + 1].flatten(),
                                val_trues_c.flatten(),
                                q_level,
                            )
                        )
                    val_check_loss = np.mean(check_losses)
                elif use_check_loss_early_stop and regression_type == "quantile":
                    val_check_loss = check_loss_numpy(
                        val_preds_c.flatten(),
                        val_trues_c.flatten(),
                        config.get("current_quantile", 0.5),
                    )
            else:
                val_rmse = 0.0
                val_check_loss = None

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_rmse"].append(val_rmse)
            if val_check_loss is not None:
                if "val_check_loss" not in self.history:
                    self.history["val_check_loss"] = []
                self.history["val_check_loss"].append(val_check_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            if not has_validation:
                patience_counter = 0
            elif use_check_loss_early_stop and val_check_loss is not None:
                if (
                    not np.isnan(val_check_loss)
                    and val_check_loss < best_val_check_loss
                ):
                    best_val_check_loss = val_check_loss
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    torch.save(model.state_dict(), self.best_model_path)
                    if self.ema is not None:
                        self.ema.restore()
                else:
                    patience_counter += 1
            else:
                if not np.isnan(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    torch.save(model.state_dict(), self.best_model_path)
                    if self.ema is not None:
                        self.ema.restore()
                else:
                    patience_counter += 1

            if self.scheduler is not None and epoch >= self.warmup_epochs:
                self.scheduler.step()

            if (
                config.get("spatial_learnable", False)
                and (epoch + 1) % self.center_record_interval == 0
            ):
                with torch.no_grad():
                    centers = model.spatial_basis.centers.cpu().numpy().copy()
                    self.basis_centers_history.append((epoch + 1, centers))

            if has_validation and patience_counter >= patience:
                break

        if self.best_model_path.exists():
            model.load_state_dict(torch.load(self.best_model_path))
        elif self.ema is not None:
            self.ema.apply_shadow()

        return model, self.history, self.basis_centers_history
