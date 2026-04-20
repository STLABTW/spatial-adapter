"""
Parameter-level non-crossing penalty P_nc(δ) (Section 3.2, Equation 3.10).

Used when use_delta_reparameterization=True. Violation set A(δ) and
l₁-penalization (conditional penalty) per Moon et al. (2021).
"""
import torch


def get_crossing_violation_mask(delta_params: list) -> list:
    """
    Violation set A(δ): k ∈ A(δ) iff δ_{k,0} < Σ_{j=1}^d max(0, -δ_{k,j}).

    Returns list of length Q-1 (for k=2..Q): each element is a scalar bool tensor.
    """
    if delta_params is None or len(delta_params) < 2:
        return []
    Q = len(delta_params)
    mask = []
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        mask.append(delta_k_0 < sum_negative)
    return mask


def compute_p_nc_delta_penalty(
    delta_params: list, use_positive_penalty: bool = False
) -> torch.Tensor:
    """
    P_nc(δ) = Σ_{k=2}^Q J(δ_k).

    Original (use_positive_penalty=False):
        J(δ_k) = δ_k,0 - max(δ_k,0, Σ_{j=1}^d max(0, -δ_k,j))
    Corrected (use_positive_penalty=True):
        J(δ_k) = max(0, Σ_{j=1}^d max(0, -δ_k,j) - δ_k,0)

    Args:
        delta_params: list of δ_k Parameter tensors, each (d+1,); δ_k[0]=intercept, δ_k[1:]=features
        use_positive_penalty: if True, use corrected positive penalty (recommended)

    Returns:
        Scalar penalty tensor
    """
    if delta_params is None or len(delta_params) < 2:
        device = delta_params[0].device if delta_params else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    Q = len(delta_params)
    penalty = torch.tensor(0.0, device=delta_params[0].device)
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        if use_positive_penalty:
            J_delta_k = torch.clamp(sum_negative - delta_k_0, min=0.0)
        else:
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
        penalty = penalty + J_delta_k
    return penalty


def compute_p_nc_delta_penalty_conditional(
    delta_params: list, use_positive_penalty: bool = True
) -> torch.Tensor:
    """
    P_nc(δ) only over violation set A(δ): Σ_{k ∈ A(δ)} J(δ_k).
    l₁-penalization: gradient applied only to heads in A(δ).
    """
    if delta_params is None or len(delta_params) < 2:
        device = delta_params[0].device if delta_params else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    Q = len(delta_params)
    device = delta_params[0].device
    penalty = torch.tensor(0.0, device=device)
    violation_mask = get_crossing_violation_mask(delta_params)
    for k in range(1, Q):
        delta_k = delta_params[k]
        delta_k_0 = delta_k[0]
        delta_k_features = delta_k[1:]
        sum_negative = torch.clamp(-delta_k_features, min=0.0).sum()
        if use_positive_penalty:
            J_delta_k = torch.clamp(sum_negative - delta_k_0, min=0.0)
        else:
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
        m = violation_mask[k - 1].detach().float()
        penalty = penalty + m * J_delta_k
    return penalty


def non_crossing_penalty(
    y_pred_multi_q: torch.Tensor, reduction: str = "mean", power: int = 1
) -> torch.Tensor:
    r"""
    Penalize quantile crossing for multi-quantile outputs (prediction-level penalty).

    Given predicted quantiles \hat{q}_1, ..., \hat{q}_Q (in increasing tau order),
    crossing happens when \hat{q}_k > \hat{q}_{k+1}. We penalize positive violations:

        P_nc = sum_{k=1}^{Q-1} ReLU(\hat{q}_k - \hat{q}_{k+1})^power

    Args:
        y_pred_multi_q: (B, Q) predicted quantiles in increasing tau order
        reduction: "mean" or "sum" over batch
        power: 1 (hinge) or 2 (squared hinge)

    Returns:
        Scalar penalty tensor.
    """
    if y_pred_multi_q.dim() != 2 or y_pred_multi_q.shape[1] < 2:
        return torch.tensor(0.0, device=y_pred_multi_q.device)
    diffs = y_pred_multi_q[:, :-1] - y_pred_multi_q[:, 1:]
    violations = torch.relu(diffs)
    if power == 2:
        violations = violations**2
    elif power != 1:
        raise ValueError(f"Unsupported power={power}; use 1 or 2.")
    per_sample = violations.sum(dim=1)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction='{reduction}'; use 'mean' or 'sum'.")


__all__ = [
    "get_crossing_violation_mask",
    "compute_p_nc_delta_penalty",
    "compute_p_nc_delta_penalty_conditional",
    "non_crossing_penalty",
]
