"""
Losses and metrics: CRPS (Eq. 4.6), check loss, P_nc(δ), PICP, QICE.
"""
from .crps import (
    check_loss_numpy,
    compute_coverage,
    compute_crps,
    compute_crps_multi_quantile,
    compute_picp,
    compute_qice,
    quantile_loss,
    trapezoidal_weights_for_quantiles,
)
from .non_crossing import (
    compute_p_nc_delta_penalty,
    compute_p_nc_delta_penalty_conditional,
    get_crossing_violation_mask,
    non_crossing_penalty,
)

__all__ = [
    "check_loss_numpy",
    "quantile_loss",
    "trapezoidal_weights_for_quantiles",
    "compute_crps",
    "compute_crps_multi_quantile",
    "compute_picp",
    "compute_coverage",
    "compute_qice",
    "get_crossing_violation_mask",
    "compute_p_nc_delta_penalty",
    "compute_p_nc_delta_penalty_conditional",
    "non_crossing_penalty",
]
