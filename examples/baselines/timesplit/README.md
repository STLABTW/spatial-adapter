# Time-split experiment utilities

Infrastructure used by the STDK-comparison experiment runners at
[`source_code_test/`](../../../source_code_test/) to set up time-split
train/val/test partitions, train STDK + Spatial Adapter, and compute
the metrics reported in the paper tables
[`tab:weather2k`](../../../.local/neurips_2026.tex) and
`tab:weather2k_holdout`.

## Layout

```
timesplit/
├── __init__.py          re-exports everything from experiment_core (``from .experiment_core import *``)
└── experiment_core.py   ~660 LOC: dataset loaders, time-split helpers,
                         metric helpers (rmse_pooled, cov_frob_observed,
                         semivariogram_match_loss_*), STDK training
                         loop, adapter fit helpers
```

## Dependencies

Imports from the Spatial Adapter library (legitimate — this is the
experiment scaffolding that runs our method):

- `spatial_adapter.models.spatial_basis_learner.SpatialBasisLearner`
- `spatial_adapter.models.spatial_adapter.SpatialAdapter`
- `spatial_adapter.models.trend_model.TrendModel`

Imports from the STDK baseline (rewritten to a relative sibling import):

- `..stdk.losses.quantile_loss`

Third-party: `torch`, `numpy`, `pandas`, `matplotlib`, `gstools`, `scipy`.

## Upstream provenance and local divergence

This file originates from the `experiment/stdk-gna-simulation` branch
where it lives at
`spatial_adapter/timesplit_functions/experiment_core.py`
(inside the library package).  On our branch it is relocated under
`examples/baselines/timesplit/` so the core library has no
dependency on baseline experiment code.  The only source-level
difference from upstream is a single import line rewrite:

```
# upstream:
from spatial_adapter.models.stdk.losses import quantile_loss

# ours:
from ..stdk.losses import quantile_loss
```

The earlier `core.py` that we mirrored in Phase 5 of the migration
has been deleted; `experiment_core.py` is a superset and is now the
sole source of truth on our side.

## How to use

Intended to be imported from the runners in `source_code_test/`:

```python
from examples.baselines.timesplit.experiment_core import (
    build_contiguous_time_splits,
    build_stdk_model_config,
    train_simple_loop,
    predict_all_simple,
    new_trend_basis,
    fit_adapter_reconstruct_all_times,
    load_weather2k_as_long_df,
    ...
)
```

When running a script from a nested directory such as
`source_code_test/Weather2K/`, the script should add the repo root to
`sys.path` before importing (which the runners already do via
`REPO_ROOT = THIS_FILE.parents[2]; sys.path.insert(0, str(REPO_ROOT))`).

## Not part of the Spatial Adapter library

This directory holds experiment scaffolding, not the method itself.
It lives under `examples/baselines/` for the same reason as STDK: to
keep the `spatial_adapter` package import surface clean.
