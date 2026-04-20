# Paper Experiments

Scripts and configs to reproduce every table in the Spatial Adapter paper.

## 1. Download data

See [`data/README.md`](../../data/README.md) for details. Short version:

```bash
bash data/download_data.sh
# KAUST: auto-download
# Weather2K: copy weather2k.npy to data/weather2k/
# GWHD: requires a Kaggle API key
```

## 2. Run

```bash
conda activate spatial-adapter

# Single experiment
python -m examples.experiments.runner \
    --config examples/experiments/configs/production/synthetic_1d.yaml

# All paper experiments
for cfg in examples/experiments/configs/production/*.yaml; do
    python -m examples.experiments.runner --config "$cfg"
done
```

## 3. Config → paper table

| Config | Paper Table | Description |
|---|---|---|
| `synthetic_1d.yaml` | Table 3 | Synthetic 1D reconstruction (10 seeds) |
| `kaust_rmse.yaml` | Table 4 | KAUST ablation — RMSE (100 seeds) |
| `kaust_covfrob.yaml` | Table 4 | KAUST ablation — CovFrob (100 seeds) |
| `kaust_svscore.yaml` | Table 4 | KAUST ablation — SV_score (100 seeds) |
| `weather2k_80_10_10.yaml` | Table 5 | Weather2K reconstruction — 80/10/10 (100 seeds) |
| `weather2k_10_10_80.yaml` | Table 5 | Weather2K reconstruction — 10/10/80 (100 seeds) |
| `weather2k_holdout.yaml` | Table 6 | Weather2K held-out stations (10 seeds) |
| `wheat_head.yaml` | Tables 7 & 8 | GWHD classification + UQ (3 seeds × 4 backbones) |

## 4. Output

Results land in `results/production/<experiment>/`:

- `results.csv` — long-format metrics
- `wall_times.csv` — per-stage wall times

## Layout

```
examples/experiments/
├── runner.py                # Unified entry point
├── base.py                  # BaseExperiment ABC
├── configs/production/      # Paper experiment configs (YAML)
├── synthetic_1d/experiment.py
├── kaust_2b8/experiment.py
├── weather2k/experiment.py
└── wheat_head/experiment.py

examples/baselines/stdk/     # STDK baseline model
```
