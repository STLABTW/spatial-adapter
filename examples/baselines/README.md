# `examples/baselines/`

Comparison baselines used in the paper's experiments.  **These modules
are not part of the Spatial Adapter library** (`spatial_adapter/`)
and are intentionally kept outside the installed package so that
importing the library never pulls in baseline-only dependencies.

## Contents

| Path | Purpose |
|---|---|
| [`stdk/`](stdk/) | Spatio-Temporal Deep Kriging reference implementation used as a benchmark model in the time-split simulation studies. |
| [`timesplit/`](timesplit/) | Time-split experiment utilities shared by the STDK simulation notebooks. |

## Status

Imported from the `experiment/stdk-gna-simulation` branch into a
staging folder (`_stdk_import/`, git-excluded) and migrated piecewise
into this tree.  Each subdirectory below has its own README with the
specific migration notes.

## Why not under `spatial_adapter/models/`?

`spatial_adapter/` contains only the Spatial Adapter method
proposed in the paper.  Keeping baselines outside the package enforces
that distinction at the import-boundary level.
