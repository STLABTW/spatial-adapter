# Spatial Adapter

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
    # criterion="auto"        # regression -> rmse, binary -> accuracy
)
result.adapter.reconstruct(val_X, val_Y)   # trained; ready to use
result.tau1, result.tau2                    # best weights chosen by Optuna
result.trials                               # pandas DataFrame of all trials
```

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
@inproceedings{spatial-adapter-2026,
  title   = {Spatial Adapter: Closed-Form Covariance and Uncertainty
             from Residual Representations},
  author  = {Anonymous},
  year    = {2026},
  note    = {NeurIPS 2026 submission}
}
```

## License

MIT — see [LICENSE](LICENSE).
