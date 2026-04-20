"""
Evaluation metrics for geospatial neural adapter.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, r2_score, roc_auc_score


def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[float, float, float]:
    """Compute RMSE, MAE, and R² metrics."""
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    mae = torch.abs(y_true - y_pred).mean().item()
    r2 = r2_score(y_true_np, y_pred_np)

    return rmse, mae, r2


def compute_binary_metrics(
    y_true: torch.Tensor, y_logits: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute Accuracy, F1, and AUC for binary classification.
    y_true: (T, N) in [0, 1] (soft or hard). y_logits: (T, N) logits (e.g. μ + Φη in logit space).
    Labels are binarized with threshold 0.5 for F1/AUC.
    """
    y_true_np = y_true.cpu().numpy().ravel()
    y_bin = (y_true_np >= 0.5).astype(np.int64)
    prob = F.sigmoid(y_logits).cpu().numpy().ravel()
    pred = (prob >= 0.5).astype(np.int64)

    accuracy = float((pred == y_bin).mean())
    f1 = float(f1_score(y_bin, pred, zero_division=0.0))
    try:
        auc = float(roc_auc_score(y_bin, prob))
    except ValueError:
        auc = 0.5
    return accuracy, f1, auc


def fusion_score(rmse: float, proj_gap: Optional[float], p: Optional[int]) -> float:
    """
    Combine RMSE with an average projection gap.
    Returns a single scalar (lower = better).
    """
    if proj_gap is None or p is None or p == 0:
        return rmse
    avg_gap = proj_gap / p
    return rmse + avg_gap


def frobenius_norm(A, B):
    """
    Frobenius norm between two matrices.
    """
    return float(np.linalg.norm(A - B, ord="fro"))


# ---------------------------------------------------------------------------
# Pooled metrics (NaN-aware, mask-aware)
#
# These are the metrics reported in the paper tables (tab:temporal_results,
# tab:ablation_val, tab:weather2k, tab:weather2k_holdout).  They supersede
# the basic RMSE/MAE/R² in compute_metrics() above for any evaluation
# that needs NaN safety or a boolean mask (e.g. held-out stations).
# ---------------------------------------------------------------------------


def rmse_pooled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    RMSE over masked entries, ignoring NaN.

    Args:
        y_true: ground truth array (any shape)
        y_pred: prediction array (same shape as y_true)
        mask:   boolean array (same shape); True = include

    Returns:
        Scalar RMSE (float), or nan if no valid entries.
    """
    valid = mask & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def mae_pooled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    MAE over masked entries, ignoring NaN.

    Args:
        y_true: ground truth array
        y_pred: prediction array
        mask:   optional boolean mask (default: finite entries)

    Returns:
        Scalar MAE (float), or nan if no valid entries.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2_pooled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    R² over masked entries, ignoring NaN.  Computed directly from
    SS_res / SS_tot without sklearn dependency.

    Args:
        y_true: ground truth array
        y_pred: prediction array
        mask:   optional boolean mask (default: finite entries)

    Returns:
        Scalar R² (float), or nan if no valid entries or zero variance.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) == 0:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def empirical_cov(field: np.ndarray) -> np.ndarray:
    """
    Sample covariance matrix from a (n_samples, n_locations) field matrix.
    Centers columns, divides by n-1 (unbiased), and symmetrizes.

    Args:
        field: (n_samples, n_locations) array

    Returns:
        (n_locations, n_locations) symmetric covariance matrix.

    Raises:
        ValueError: if field is not 2D or has fewer than 2 rows.
    """
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape={field.shape}")
    if field.shape[0] <= 1:
        raise ValueError("field must have at least 2 rows to compute sample covariance")
    field_centered = field - np.mean(field, axis=0, keepdims=True)
    cov = (field_centered.T @ field_centered) / (field.shape[0] - 1)
    return 0.5 * (cov + cov.T)


def cov_frob_observed(
    y_true_field: np.ndarray,
    y_pred_field: np.ndarray,
) -> float:
    """
    Relative Frobenius covariance error between observed and predicted
    spatial fields.  This is the CovFrob metric reported in the paper:

        CovFrob = ||Σ_pred − Σ_obs||_F / ||Σ_obs||_F

    where Σ = empirical_cov(field).

    Args:
        y_true_field: (n_samples, n_locations) ground-truth field
        y_pred_field: (n_samples, n_locations) predicted field

    Returns:
        Scalar relative Frobenius error (float).
    """
    sigma_obs = empirical_cov(y_true_field)
    sigma_pred = empirical_cov(y_pred_field)
    return float(
        np.linalg.norm(sigma_pred - sigma_obs, ord="fro")
        / (np.linalg.norm(sigma_obs, ord="fro") + 1e-12)
    )


# ---------------------------------------------------------------------------
# Prediction-interval calibration metrics (eq:cp-mpiw)
# ---------------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
    mask: Optional[np.ndarray] = None,
) -> float:
    r"""
    Expected Calibration Error (ECE).

    Partitions predicted probabilities into ``n_bins`` equal-width bins and
    measures the weighted average absolute gap between the bin-average
    predicted probability and the bin-average observed frequency:

    .. math::

        \mathrm{ECE}
        = \sum_{b=1}^{B} \frac{|B_b|}{n}
          \bigl|\bar p_b - \bar y_b\bigr|

    where :math:`\bar p_b` and :math:`\bar y_b` are the mean predicted
    probability and mean observed label in bin :math:`b`.

    Parameters
    ----------
    y_true : array
        Binary ground-truth labels in {0, 1}.
    y_prob : array (same shape)
        Predicted probabilities in [0, 1].
    n_bins : int, optional
        Number of equal-width bins (default: 15).
    mask : array (same shape), optional
        Boolean mask; True = include.  Default: all finite entries.

    Returns
    -------
    ece : float
        Expected calibration error in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if mask is not None:
        mask = np.asarray(mask).ravel()
        y_true = y_true[mask]
        y_prob = y_prob[mask]

    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_prob > lo) & (y_prob <= hi)
        # include the left boundary for the first bin
        if lo == 0.0:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_prob = y_prob[in_bin].mean()
        avg_true = y_true[in_bin].mean()
        ece += count * abs(avg_prob - avg_true)

    return float(ece / len(y_true))


def coverage_probability(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    r"""
    Empirical coverage probability (CP) from eq:cp-mpiw.

    .. math::

        \mathrm{CP}
        = \frac{1}{|\mathcal I|}
          \sum_{(j,i)\in\mathcal I}
          \mathbf{1}\{Y_{j,i} \in [\widehat L_{j,i},\widehat U_{j,i}]\}

    Parameters
    ----------
    y_true : array
        Ground-truth values.
    lower : array (same shape)
        Lower bounds of the prediction intervals.
    upper : array (same shape)
        Upper bounds of the prediction intervals.
    mask : array (same shape), optional
        Boolean mask; True = include.  Default: all finite entries.

    Returns
    -------
    cp : float
        Fraction of entries covered, in [0, 1].
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if mask is None:
        mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if np.sum(mask) == 0:
        return float("nan")
    covered = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    return float(np.mean(covered))


def mpiw(
    lower: np.ndarray,
    upper: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    r"""
    Mean Prediction Interval Width (MPIW) from eq:cp-mpiw.

    .. math::

        \mathrm{MPIW}
        = \frac{1}{|\mathcal I|}
          \sum_{(j,i)\in\mathcal I}
          (\widehat U_{j,i} - \widehat L_{j,i})

    Parameters
    ----------
    lower : array
        Lower bounds of the prediction intervals.
    upper : array
        Upper bounds of the prediction intervals.
    mask : array (same shape), optional
        Boolean mask; True = include.  Default: all finite entries.

    Returns
    -------
    mean_width : float
        Average interval width over the masked entries.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    width = upper - lower
    if mask is None:
        mask = np.isfinite(width)
    if np.sum(mask) == 0:
        return float("nan")
    return float(np.mean(width[mask]))
