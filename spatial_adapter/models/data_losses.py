"""ADMM data-term losses: ℓ_data(P^⊥ Z) and gradients for non-identity links."""

from typing import Tuple

import torch
import torch.nn.functional as F


def loss_data_binary(P_perp_Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """BCE data term: ℓ_data = Σ BCE(σ(P^⊥ Z), Y).  Returns scalar (sum)."""
    if P_perp_Z.shape != Y.shape:
        raise ValueError(f"P_perp_Z shape {P_perp_Z.shape} != Y shape {Y.shape}")
    Y = Y.clamp(min=1e-7, max=1.0 - 1e-7)
    return F.binary_cross_entropy_with_logits(P_perp_Z, Y, reduction="sum")


def grad_Z_loss_data_binary(
    Z: torch.Tensor,
    P_perp: torch.Tensor,
    Y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ℓ_data(P^⊥ Z) and ∇_Z ℓ_data via autograd.

    Returns (loss_detached, Z_grad) for the Z-step proximal-gradient.
    """
    if Z.dim() != 2 or P_perp.dim() != 2 or Y.dim() != 2:
        raise ValueError("Z, P_perp, Y must be 2D")
    T, N = Z.shape
    if P_perp.shape != (N, N) or Y.shape != (T, N):
        raise ValueError(f"P_perp must be ({N},{N}), Y must be ({T},{N})")

    Z_in = Z.detach().clone().requires_grad_(True)
    loss = loss_data_binary(Z_in @ P_perp, Y)
    loss.backward()
    return loss.detach(), Z_in.grad.clone()


def projection_orthogonal_complement(Phi: torch.Tensor) -> torch.Tensor:
    """P^⊥ = I − ΦΦ^T from basis Φ (N, K)."""
    N = Phi.shape[0]
    return torch.eye(N, device=Phi.device, dtype=Phi.dtype) - Phi @ Phi.T
