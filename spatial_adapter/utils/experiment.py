"""TensorBoard logging helpers for covariance and basis diagnostics."""

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel


def log_covariance_and_basis(
    writer: SummaryWriter,
    tag: str,
    step: int,
    trend_best: TrendModel,
    basis_best: SpatialBasisLearner,
    val_cont: torch.Tensor,
    val_y: torch.Tensor,
    locs: np.ndarray,
    config: dict,
    tau1: float,
    tau2: float,
    best_val: float,
) -> None:
    """Log basis histogram, norms, and empirical covariance stats to TensorBoard."""
    basis_np = basis_best.basis.detach().cpu().numpy()
    writer.add_histogram(f"{tag}/basis_hist", basis_np, step)
    writer.add_scalar(f"{tag}/basis_norm", np.linalg.norm(basis_np), step)
    writer.add_scalar(f"{tag}/best_val", best_val, step)
    writer.add_scalar(f"{tag}/tau1", tau1, step)
    writer.add_scalar(f"{tag}/tau2", tau2, step)

    try:
        with torch.no_grad():
            residuals = (val_y - trend_best(val_cont)).cpu().numpy()
            emp_cov = residuals.T @ residuals / residuals.shape[0]
            writer.add_scalar(f"{tag}/emp_cov_trace", np.trace(emp_cov), step)
            writer.add_scalar(f"{tag}/emp_cov_norm", np.linalg.norm(emp_cov), step)
    except Exception as e:
        print(f"Warning: Could not compute covariance for logging: {e}")
