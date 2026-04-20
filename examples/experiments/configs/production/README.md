# Production Configs
These configs reproduce all tables in the paper.

| Config | Paper Table | Seeds | Est. Time (1× V100) |
|--------|-------------|-------|---------------------|
| `synthetic_1d.yaml` | Table 3 | 10 | ~30 min |
| `kaust_rmse.yaml` | Table 4 (RMSE row) | 100 | ~8 hrs |
| `kaust_covfrob.yaml` | Table 4 (CovFrob row) | 100 | ~8 hrs |
| `kaust_svscore.yaml` | Table 4 (SV row) | 100 | ~8 hrs |
| `weather2k_80_10_10.yaml` | Table 5 (80/10/10) | 100 | ~5 hrs |
| `weather2k_10_10_80.yaml` | Table 5 (10/10/80) | 100 | ~5 hrs |
| `weather2k_holdout.yaml` | Table 6 | 10 | ~2 hrs |
| `wheat_head.yaml` | Tables 7 & 8 | 3 | ~6 hrs |

## Run all

```bash
# Sequential (safest)
for cfg in examples/experiments/configs/production/*.yaml; do
    python -m examples.experiments.runner --config "$cfg"
done

# Or one at a time
python -m examples.experiments.runner --config examples/experiments/configs/production/synthetic_1d.yaml
```

## Output

All results saved to `results/production/<experiment>/`:
- `results.csv` — long-format metrics (experiment, seed, model, split, metric, value)
- `wall_times.csv` — per-stage wall times

## Hardware

Experiments were conducted on a single NVIDIA Tesla V100 GPU (32 GB).

## First-stage hyperparameters

STDK hyperparameters follow the [official implementation](https://github.com/pratiknag/Space-Time.DeepKriging/blob/main/50kSimulation-space-time_DeepKriging.ipynb):
epochs=350, lr=0.001 (Adam), patience=30, batch_size=512.
