"""
Spatial Adapter

A Python package for neural spatial modeling with low-rank approximations.
"""

__version__ = "0.6.0"
__author__ = "Wen-Ting Wang"
__email__ = "egpivo@gmail.com"

# Data generation
from .data.generators import (
    generate_combined_synthetic_data,
    generate_time_synthetic_data,
)
from .data.preprocessing import prepare_all

# Metrics
from .metrics import (
    compute_metrics,
    expected_calibration_error,
    frobenius_norm,
    fusion_score,
)

# Models + configs (so users can `from spatial_adapter import SpatialAdapter, ...`)
from .models.spatial_adapter import (
    ADMMConfig,
    BasisConfig,
    SpatialAdapter,
    SpatialAdapterConfig,
    TrainingConfig,
)
from .models.spatial_basis_learner import SpatialBasisLearner
from .models.trend_model import TrendModel

# Tuning
from .tuning import AdapterTuner, FitTunedResult

__all__ = [
    # Models
    "SpatialAdapter",
    "SpatialBasisLearner",
    "TrendModel",
    # Configs
    "SpatialAdapterConfig",
    "ADMMConfig",
    "TrainingConfig",
    "BasisConfig",
    # Tuning
    "AdapterTuner",
    "FitTunedResult",
    # Data
    "generate_combined_synthetic_data",
    "generate_time_synthetic_data",
    "prepare_all",
    # Metrics
    "compute_metrics",
    "expected_calibration_error",
    "frobenius_norm",
    "fusion_score",
]
