#!/usr/bin/env python
"""
Plot cumulative explained variance for rank selection (eq:rank-selection).

Reads residual covariance eigenvalue CSVs and produces a two-panel figure
showing the scree plot and cumulative explained variance with the chosen
K marked for each dataset.

Usage:
    python examples/experiments/plot_rank_selection.py

Output:
    .local/rank_selection.pdf
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
DATA_DIR = Path("experiment_report/residual_spectrum_eigenvalues")
OUTPUT_DIR = Path(".local")

DATASETS = [
    {
        "csv": DATA_DIR / "2b_8_seed123_residual_cov_eigenvalues.csv",
        "label": "KAUST ($N=1000$)",
        "K": 80,
        "tau": 0.9,
    },
    {
        "csv": DATA_DIR / "weather2k_seed123_t1000_residual_cov_eigenvalues.csv",
        "label": r"Weather2K ($N=187$)",
        "K": 40,
        "tau": 0.8,
    },
]


def load_eigenvalues(csv_path: Path):
    """Load eigenvalues and cumulative explained variance from CSV."""
    components = []
    eigenvalues = []
    cumulative = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            components.append(int(row["component"]))
            eigenvalues.append(float(row["eigenvalue"]))
            cumulative.append(float(row["cumulative_explained_variance"]))
    return np.array(components), np.array(eigenvalues), np.array(cumulative)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ["C0", "C1"]

    # Pre-load all datasets to find common x-axis range
    loaded = []
    for ds in DATASETS:
        comp, eig, cum = load_eigenvalues(ds["csv"])
        # Effective rank: last component with eigenvalue > 1e-10
        eff_rank = int(np.searchsorted(-eig, -1e-10))
        loaded.append((comp, eig, cum, eff_rank))

    # Use max effective rank + small margin as common x limit
    x_max = max(er for _, _, _, er in loaded) + 5

    for i, (ds, (comp, eig, cum, eff_rank)) in enumerate(zip(DATASETS, loaded)):
        c = colors[i]
        K = ds["K"]
        tau = ds["tau"]
        n_show = min(x_max, len(comp))

        # Only plot eigenvalues above numerical noise for scree
        mask = eig[:n_show] > 1e-14
        axes[0].semilogy(
            comp[:n_show][mask],
            eig[:n_show][mask],
            label=ds["label"],
            color=c,
            linewidth=1.2,
        )
        # Mark chosen K
        axes[0].axvline(
            K,
            color=c,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        axes[0].annotate(
            f"$K={K}$",
            xy=(K, eig[K - 1]),
            xytext=(K + 3, eig[K - 1] * 2),
            fontsize=8,
            color=c,
            arrowprops=dict(arrowstyle="-", color=c, linewidth=0.6),
        )

        # (b) Cumulative explained variance — plot up to effective rank
        axes[1].plot(
            comp[:eff_rank],
            cum[:eff_rank],
            label=ds["label"],
            color=c,
            linewidth=1.2,
        )
        # Threshold line
        axes[1].axhline(
            tau,
            color=c,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )
        # Mark chosen K
        axes[1].plot(K, cum[K - 1], "o", color=c, markersize=5, zorder=5)
        axes[1].annotate(
            f"$K={K}$, $\\tau={tau}$",
            xy=(K, cum[K - 1]),
            xytext=(K + 5, cum[K - 1] - 0.06),
            fontsize=8,
            color=c,
            arrowprops=dict(arrowstyle="-", color=c, linewidth=0.6),
        )

    axes[0].set_xlabel("Component $k$")
    axes[0].set_ylabel("Eigenvalue $\\hat\\lambda_k$")
    axes[0].set_title("(a) Scree plot")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Component $k$")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("(b) Cumulative variance")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "rank_selection.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Figure saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
