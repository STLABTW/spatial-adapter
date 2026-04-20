"""Spatial basis learner: orthonormal Φ ∈ ℝ^{N×K} via SVD retraction."""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class SpatialBasisLearner(nn.Module):
    """Learns an orthonormal spatial basis Φ ∈ ℝ^{N×K}."""

    def __init__(
        self,
        num_locations: int,
        latent_dim: int,
        pca_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if latent_dim > num_locations:
            raise ValueError("latent_dim must be ≤ num_locations")

        if pca_init is not None:
            if pca_init.shape != (num_locations, latent_dim):
                raise ValueError("pca_init must have shape (N, K)")
            B = torch.tensor(pca_init, dtype=torch.float32)
        else:
            B = torch.empty(num_locations, latent_dim, dtype=torch.float32)
            nn.init.orthogonal_(B)

        self.basis = nn.Parameter(B)

    @torch.no_grad()
    def _retract(self):
        """Re-project onto the Stiefel manifold (orthonormal columns) via SVD."""
        U, _, _ = torch.linalg.svd(self.basis, full_matrices=False)
        self.basis.copy_(U)

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """Project residuals onto the basis: (B, N) → (B, N)."""
        self._retract()
        P = self.basis @ self.basis.T
        return residuals @ P

    def get_basis(self) -> torch.Tensor:
        return self.basis

    def project(self, data: torch.Tensor) -> torch.Tensor:
        """(B, N) → (B, K) coefficients."""
        return data @ self.basis

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """(B, K) → (B, N) reconstructed."""
        return coefficients @ self.basis.T

    @torch.no_grad()
    def reinit_from_pca(self, pca_basis: Union[np.ndarray, torch.Tensor]):
        """Overwrite with a new basis and re-orthonormalize."""
        if isinstance(pca_basis, np.ndarray):
            pca_basis = torch.from_numpy(pca_basis).to(self.basis.device).float()
        else:
            pca_basis = pca_basis.to(self.basis.device).float()
        if pca_basis.shape != self.basis.shape:
            raise ValueError("Shape mismatch when re-initialising basis")
        self.basis.copy_(pca_basis)
        self._retract()
