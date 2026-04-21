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

```python
import torch
from spatial_adapter.models.spatial_adapter import (
    SpatialAdapter, SpatialAdapterConfig,
)
from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel

# Stage 1: any first-stage predictor (OLS, STDK, deep backbone, ...)
trend = TrendModel(num_continuous_features=5, hidden_layer_sizes=[], n_locations=100)

# Stage 2: attach spatial adapter
basis = SpatialBasisLearner(num_locations=100, latent_dim=10)
config = SpatialAdapterConfig()

adapter = SpatialAdapter(
    trend, basis, train_loader,
    val_cont=val_X, val_y=val_Y,
    locs=station_coords,
    config=config,
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
