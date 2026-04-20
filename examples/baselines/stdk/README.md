# STDK baseline

Reference implementation of **Spatio-Temporal Deep Kriging** used as
one of the benchmark comparison models in the paper's time-split
simulation experiments.  Originally imported from the
`experiment/stdk-gna-simulation` branch.

## Layout

```
stdk/
├── __init__.py           package marker + high-level docstring
├── st_interp.py          STDK spatio-temporal interpolation model (~930 LOC)
├── trainer.py            training loop, evaluation, early stopping, EMA
├── losses/
│   ├── __init__.py       re-exports of loss / metric helpers
│   ├── crps.py           CRPS, check loss, PICP, QICE
│   └── non_crossing.py   non-crossing penalty for quantile outputs
└── utils/
    ├── __init__.py
    └── ema.py            ModelEMA helper
```

## Import style

The package uses **relative imports** internally
(`from .losses import ...`, `from .utils.ema import ModelEMA`) so
that the whole directory can be moved or renamed without rewriting
any source file.  It does **not** import anything from the
`spatial_adapter` library — STDK and the Spatial Adapter
live in separate namespaces and never touch each other at import
time.

## How to use

Intended to be imported from experiment notebooks and comparison
scripts, for example:

```python
from examples.baselines.stdk.st_interp import STInterp
from examples.baselines.stdk.trainer import evaluate_model
```

The experiment notebooks from the `experiment/stdk-gna-simulation`
branch currently import STDK via
`from spatial_adapter.models.stdk...` — those imports will
be rewritten to match the new path in Phase 7 of the migration.

## Not part of the Spatial Adapter library

STDK is a **benchmark**, not the method proposed in the paper.  It
lives here under `examples/baselines/` precisely so that importing
`spatial_adapter` never transitively imports STDK or its
dependencies (torch, sklearn, etc.).
