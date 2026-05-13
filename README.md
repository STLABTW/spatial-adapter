# Spatial Adapter

[![Tests](https://github.com/STLABTW/spatial-adapter/workflows/Test/badge.svg)](https://github.com/STLABTW/spatial-adapter/actions)
[![codecov](https://codecov.io/github/STLABTW/spatial-adapter/graph/badge.svg?token=CODECOV_TOKEN)](https://codecov.io/github/STLABTW/spatial-adapter)
[![arXiv](https://img.shields.io/badge/arXiv-2605.11394-b31b1b.svg)](https://arxiv.org/abs/2605.11394)
<a href="https://pypi.org/project/spatial-adapter/"><img src="https://img.shields.io/pypi/v/spatial-adapter.svg?logo=pypi&label=PyPI&logoColor=silver" alt="PyPI version"/></a>

A post-hoc cascade adapter that extracts explicit low-rank spatial representations from the residuals of any frozen first-stage predictor, yielding closed-form covariance estimation and uncertainty quantification.


## Installation

Requires Python 3.10+, Conda, and (optionally) CUDA 12.8+ for GPU.

```bash
git clone https://github.com/STLABTW/spatial-adapter.git
cd spatial-adapter

make conda-env                 # recommended: conda env + C++ extensions
# or
pip install -e ".[all]"        # pip-only
```

For Blackwell / sm_120 GPUs (e.g. RTX 5070 Ti):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

## Quick Start

### Option A — one-shot tuned fit (recommended)

Hides the (τ₁, τ₂) grid search behind a single call: runs a short Optuna
sweep, refits at the best weights, returns the trained adapter plus the
chosen τ's and the trials dataframe.

```python
import torch
from spatial_adapter import SpatialAdapter, TrendModel

trend = TrendModel(num_continuous_features=5, hidden_layer_sizes=[], n_locations=100)

result = SpatialAdapter.fit_tuned(
    trend, train_loader,
    val_cont=val_X, val_y=val_Y, locs=station_coords,
    device=torch.device("cpu"),
    latent_dim=10,
    n_trials=10,              # per-seed Optuna budget (defaults to 10)
    seed=42,
    criterion="auto",         # see table below
)
result.adapter.reconstruct(val_X, val_Y)   # trained; ready to use
result.tau1, result.tau2                    # best weights chosen by Optuna
result.trials                               # pandas DataFrame of all trials
```

**Validation criterion** — what Optuna minimizes/maximizes over (τ₁, τ₂):

| `criterion` | Task | Direction | What it measures |
|---|---|---|---|
| `"auto"` *(default)* | any | — | regression → `rmse`, binary → `accuracy` |
| `"rmse"` | regression | min | point-prediction RMSE on val |
| `"accuracy"` | binary | max | classification accuracy on val |
| `"auc"` | binary | max | ROC AUC on val |
| `"cov_frob"` | regression | min | relative Frobenius covariance error, \|\|Σ̂_pred − Σ̂_obs\|\|_F / \|\|Σ̂_obs\|\|_F |
| `"sv_score"` | regression | min | weighted L² distance between empirical semivariograms of val and prediction (gstools) |

For the paper's **KAUST Table 4 ablation** (RMSE vs CovFrob vs SV_score
selection), use these criteria directly. For domain-specific ground-truth
covariance (e.g. a known Matérn model), fall back to `AdapterTuner` with a
custom `evaluate_fn`.

### Option B — manual control (power users)

Bypasses tuning when you already have (τ₁, τ₂) or want direct control of
the ADMM loop.

```python
import torch
from spatial_adapter import (
    SpatialAdapter, SpatialAdapterConfig,
    SpatialBasisLearner, TrendModel,
)

trend = TrendModel(num_continuous_features=5, hidden_layer_sizes=[], n_locations=100)
basis = SpatialBasisLearner(num_locations=100, latent_dim=10)

adapter = SpatialAdapter(
    trend, basis, train_loader,
    val_cont=val_X, val_y=val_Y, locs=station_coords,
    config=SpatialAdapterConfig(),
    device=torch.device("cpu"),
    tau1=1.0, tau2=1.0,
)
adapter.pretrain_trend(epochs=5)
adapter.init_basis_dense()
adapter.run()  # ADMM optimization
```

## Examples & paper experiments

End-to-end scripts, configs, and notebooks live under [`examples/`](examples/).
To reproduce the paper tables (data download, config → table mapping,
expected output) see [`examples/experiments/README.md`](examples/experiments/README.md).

## Project layout

- [`spatial_adapter/`](spatial_adapter/) — core library: `models/`, `data/`,
  `metrics.py`, `prediction.py`, `tuning/`, `utils/`, and `cpp_extensions/`
  (pybind11 C++ kernels).
- [`examples/`](examples/) — runnable experiments ([`experiments/`](examples/experiments/))
  and baselines ([`baselines/stdk/`](examples/baselines/stdk/)).
- [`tests/`](tests/) — pytest suite; see `make test`.
- [`data/`](data/) — external datasets (git-ignored; see [`data/README.md`](data/README.md)).

## Development

```bash
make conda-env       # set up environment
make build-cpp       # build C++ extensions
make test            # run tests
make test-cov        # tests with HTML coverage
```

## Citation

```bibtex
@misc{wang2026spatialadapterstructuredspatial,
      title={Spatial Adapter: Structured Spatial Decomposition and Closed-Form Covariance for Frozen Predictors},
      author={Wen-Ting Wang and Wei-Ying Wu and Hao-Yun Huang and Xuan-Chun Wang},
      year={2026},
      eprint={2605.11394},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2605.11394},
}
```

## License

MIT — see [LICENSE](LICENSE).
