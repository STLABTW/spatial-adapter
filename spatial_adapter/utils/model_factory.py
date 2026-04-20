"""Factory for creating OLS-initialized trend + basis model pairs."""

from typing import List, Tuple

import torch

from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel


def create_fresh_models(
    device: torch.device,
    p_dim: int,
    n_locations: int,
    latent_dim: int,
    w_ols: torch.Tensor,
    b_ols: float,
    hidden_layer_sizes: List[int] = None,
    dropout_rate: float = 0.1,
) -> Tuple[TrendModel, SpatialBasisLearner]:
    """Create a fresh (trend, basis) pair with OLS-initialized frozen linear layer."""
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [256, 64]
    trend = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        n_locations=n_locations,
        init_weight=w_ols,
        init_bias=b_ols,
        freeze_init=True,
        dropout_rate=dropout_rate,
    ).to(device)
    basis = SpatialBasisLearner(num_locations=n_locations, latent_dim=latent_dim).to(
        device
    )
    return trend, basis
