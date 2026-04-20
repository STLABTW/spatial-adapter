"""Hyperparameter tuning for Spatial Adapter.

Provides Optuna-based sweeps over (tau1, tau2) with warm-start caching of
model state across trials.
"""

from .adapter_tuner import AdapterTuner
from .model_cache import ModelCache

__all__ = ["AdapterTuner", "ModelCache"]
