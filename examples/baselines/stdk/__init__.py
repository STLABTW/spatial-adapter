"""
STDK (Spatio-Temporal Deep Kriging) baseline.

Reference implementation used as a benchmark comparison model in the
paper's time-split simulation experiments.  Not part of the Spatial
Adapter library: this package lives under ``examples/baselines/`` so
that importing ``spatial_adapter`` never transitively pulls
in STDK or its dependencies.

Top-level submodules:
    st_interp  - the STDK spatio-temporal interpolation model
    trainer    - training loop, evaluation, early stopping, EMA
    losses     - CRPS, check loss, non-crossing penalties, PICP, QICE
    utils      - utility helpers (EMA, etc.)
"""
