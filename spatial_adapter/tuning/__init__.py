"""Hyperparameter tuning for Spatial Adapter.

Provides Optuna-based sweeps over (tau1, tau2) with warm-start caching of
model state across trials.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .adapter_tuner import AdapterTuner
from .model_cache import ModelCache

if TYPE_CHECKING:
    import pandas as pd

    from spatial_adapter.models.spatial_adapter import SpatialAdapter


@dataclass
class FitTunedResult:
    """Return value of :meth:`SpatialAdapter.fit_tuned`.

    Attributes
    ----------
    adapter : SpatialAdapter
        Adapter trained at the best (tau1, tau2) found by Optuna.
    tau1, tau2 : float
        Best regularisation weights.
    trials : pandas.DataFrame
        One row per Optuna trial (from ``study.trials_dataframe()``).
    """

    adapter: "SpatialAdapter"
    tau1: float
    tau2: float
    trials: "pd.DataFrame"


__all__ = ["AdapterTuner", "FitTunedResult", "ModelCache"]
