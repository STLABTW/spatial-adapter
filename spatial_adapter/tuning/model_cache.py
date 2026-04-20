"""Model weight cache for warm-starting across Optuna trials."""

import math
from typing import Dict, Tuple

from torch import nn


class ModelCache:
    """Stores (trend, basis) state dicts keyed by (tau1, tau2).

    ``load_nearest`` finds the closest cached key in log-space and
    warm-starts the models, which speeds up Optuna sweeps.
    """

    def __init__(self):
        self.cache: Dict[Tuple[float, float], Tuple[Dict, Dict]] = {}

    def store(self, tau1: float, tau2: float, trend_state: Dict, basis_state: Dict):
        self.cache[(tau1, tau2)] = (trend_state, basis_state)

    def load_nearest(
        self, trend: nn.Module, basis: nn.Module, tau1: float, tau2: float
    ):
        """Load the cached weights closest to (tau1, tau2) in log-space."""
        if not self.cache:
            return
        lt1, lt2 = math.log10(tau1 + 1e-12), math.log10(tau2 + 1e-12)
        best = min(
            self.cache,
            key=lambda k: (math.log10(k[0] + 1e-12) - lt1) ** 2
            + (math.log10(k[1] + 1e-12) - lt2) ** 2,
        )
        trend.load_state_dict(self.cache[best][0], strict=False)
        basis.load_state_dict(self.cache[best][1], strict=False)

    def clear(self):
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)

    def keys(self) -> list:
        return list(self.cache.keys())
