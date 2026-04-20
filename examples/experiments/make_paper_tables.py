#!/usr/bin/env python
"""Aggregate results.csv from each experiment into paper-ready tables.

Reads each experiment's long-format results.csv (schema: experiment, seed,
model, split, metric, value) and prints mean (std. error) aggregations
grouped by (model, metric). Outputs both a plain-text preview and a
LaTeX row fragment that can be pasted into neurips_2026_v1.tex.

Usage
-----
    # All experiments from the default output_dirs:
    python -m examples.experiments.make_paper_tables

    # Point at smoke-test outputs:
    python -m examples.experiments.make_paper_tables \
        --results-root /tmp \
        --prefix smoke_
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Table specs ───────────────────────────────────────────────────────

# Each spec: list of (model_key, label_in_table, metrics_to_show)
TABLE_SPECS: Dict[str, dict] = {
    "synthetic_1d": {
        "caption": "Table 1 — Reconstruction on 1D synthetic benchmark",
        "rows": [
            ("baseline", "OLS"),
            ("adapter_unreg", "Neural Spatial (Unreg.)"),
            ("adapter_reg", "Neural Spatial (Reg.)"),
        ],
        "metrics": [
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("r2", "R^2 (%)"),
            ("covfrob", "CovFrob"),
        ],
        "r2_percent": True,
    },
    "kaust": {
        "caption": "Table 2 — KAUST ablation on validation criterion",
        "rows": [
            ("baseline", "STDK"),
            ("adapter_unreg", "+ Adapter (unreg.)"),
            ("adapter_reg", "+ Adapter (reg.)"),
        ],
        "metrics": [
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("r2", "R^2 (%)"),
            ("covfrob", "CovFrob"),
        ],
        "r2_percent": True,
    },
    "weather2k_timesplit": {
        "caption": "Table 3 — Weather2K time-split reconstruction",
        "rows": [
            ("baseline", "STDK"),
            ("adapter_unreg", "Adapter (unreg.)"),
            ("adapter_reg", "Adapter (reg.)"),
        ],
        "metrics": [
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("r2", "R^2 (%)"),
            ("covfrob", "CovFrob"),
        ],
        "r2_percent": True,
    },
    "weather2k_holdout": {
        "caption": "Table 4 — Weather2K spatial prediction at 20% held-out stations",
        "rows": [
            ("baseline", "STDK"),
            ("adapter_unreg", "Adapter (unreg.)"),
            ("adapter_reg", "Adapter (reg.)"),
        ],
        "metrics": [
            ("rmse", "RMSE"),
            ("rmse_heldout", "RMSE (held-out)"),
            ("r2", "R^2 (%)"),
            ("covfrob", "CovFrob"),
        ],
        "r2_percent": True,
    },
    "wheat_head_table5": {
        "source": "wheat_head",
        "caption": "Table 5 — Wheat Head patch classification (GWHD)",
        "rows_by_backbone": True,  # grouped by backbone prefix
        "row_labels": {
            "baseline": "Backbone only",
            "unreg": "+ Adapter (unreg.)",
            "reg": "+ Adapter (reg.)",
        },
        "metrics": [("accuracy", "Accuracy (%)"), ("auc", "AUC"), ("f1", "F1")],
        "accuracy_percent": True,
    },
    "wheat_head_table6": {
        "source": "wheat_head",
        "caption": "Table 6 — UQ diagnostics (GWHD, ResNet-152)",
        "backbone_filter": "resnet152",
        "row_labels": {
            "baseline": "Backbone only",
            "unreg": "+ Adapter (unreg.)",
            "reg": "+ Adapter (reg.)",
        },
        "metrics": [
            ("accuracy", "Accuracy (%)"),
            ("mpiw", "MPIW"),
            ("ece", "ECE"),
            ("cp", "CP (%)"),
        ],
        "accuracy_percent": True,
    },
}


# ── Aggregation helpers ───────────────────────────────────────────────


def mean_se(values: np.ndarray) -> Tuple[float, float]:
    """Mean and standard error across seeds."""
    n = len(values)
    m = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return m, se


def format_cell(mean: float, se: float, scale: float = 1.0, digits: int = 4) -> str:
    return f"{mean * scale:.{digits}f} ({se * scale:.{digits}f})"


def aggregate(
    df: pd.DataFrame, model: str, metric: str
) -> Optional[Tuple[float, float, int]]:
    """Return (mean, se, n_seeds) for a (model, metric) pair, or None if empty."""
    sub = df[(df["model"] == model) & (df["metric"] == metric)]
    if sub.empty:
        return None
    vals = sub["value"].to_numpy()
    m, se = mean_se(vals)
    return m, se, len(vals)


# ── Table renderers ───────────────────────────────────────────────────


def render_simple_table(df: pd.DataFrame, spec: dict) -> List[str]:
    """Flat (model × metric) table, for Tables 1–4."""
    rows = []
    header_cols = [label for _, label in spec["metrics"]]
    rows.append(f"\n{spec['caption']}")
    rows.append("-" * 78)
    rows.append(f"{'Model':<28} " + "  ".join(f"{c:>16}" for c in header_cols))
    for model_key, label in spec["rows"]:
        cells = []
        for metric_key, _ in spec["metrics"]:
            agg = aggregate(df, model_key, metric_key)
            if agg is None:
                cells.append("--".rjust(16))
                continue
            m, se, _ = agg
            scale = 100.0 if (metric_key == "r2" and spec.get("r2_percent")) else 1.0
            digits = 2 if scale == 100.0 else 4
            cells.append(format_cell(m, se, scale=scale, digits=digits).rjust(16))
        rows.append(f"{label:<28} " + "  ".join(cells))
    rows.append(f"(seeds: {df['seed'].nunique()})")
    return rows


def render_wheat_head_table5(df: pd.DataFrame, spec: dict) -> List[str]:
    """Backbone × {baseline, unreg, reg} × metrics table."""
    rows = [f"\n{spec['caption']}", "-" * 78]
    header = [label for _, label in spec["metrics"]]
    rows.append(
        f"{'Backbone':<15} {'Model':<22} " + "  ".join(f"{h:>16}" for h in header)
    )

    # Find distinct backbones by looking at model names like "<backbone>_<stage>"
    model_names = df["model"].unique().tolist()
    backbones = sorted({m.rsplit("_", 1)[0] for m in model_names if "_" in m})

    for backbone in backbones:
        for stage in ["baseline", "unreg", "reg"]:
            model_key = f"{backbone}_{stage}"
            label = spec["row_labels"][stage]
            cells = []
            for metric_key, _ in spec["metrics"]:
                agg = aggregate(df, model_key, metric_key)
                if agg is None:
                    cells.append("--".rjust(16))
                    continue
                m, se, _ = agg
                scale = (
                    100.0
                    if (metric_key == "accuracy" and spec.get("accuracy_percent"))
                    else 1.0
                )
                digits = 2 if scale == 100.0 else 4
                cells.append(format_cell(m, se, scale=scale, digits=digits).rjust(16))
            rows.append(f"{backbone:<15} {label:<22} " + "  ".join(cells))
        rows.append("-" * 78)
    rows.append(f"(seeds: {df['seed'].nunique()})")
    return rows


def render_wheat_head_table6(df: pd.DataFrame, spec: dict) -> List[str]:
    """UQ table for a single backbone."""
    backbone = spec["backbone_filter"]
    rows = [f"\n{spec['caption']}", "-" * 78]
    header = [label for _, label in spec["metrics"]]
    rows.append(f"{'Model':<22} " + "  ".join(f"{h:>16}" for h in header))
    for stage in ["baseline", "unreg", "reg"]:
        model_key = f"{backbone}_{stage}"
        label = spec["row_labels"][stage]
        cells = []
        for metric_key, _ in spec["metrics"]:
            agg = aggregate(df, model_key, metric_key)
            if agg is None:
                cells.append("--".rjust(16))
                continue
            m, se, _ = agg
            scale = (
                100.0
                if (metric_key == "accuracy" and spec.get("accuracy_percent"))
                else 1.0
            )
            digits = 2 if scale == 100.0 or metric_key == "cp" else 4
            cells.append(format_cell(m, se, scale=scale, digits=digits).rjust(16))
        rows.append(f"{label:<22} " + "  ".join(cells))
    rows.append(f"(seeds: {df['seed'].nunique()})")
    return rows


# ── Main ──────────────────────────────────────────────────────────────


def _load_results(
    results_root: Path, exp_name: str, prefix: str
) -> Optional[pd.DataFrame]:
    # Try <root>/<prefix><exp>/results.csv first, then <root>/<exp>/results.csv
    for cand in (
        results_root / f"{prefix}{exp_name}" / "results.csv",
        results_root / exp_name / "results.csv",
    ):
        if cand.exists():
            return pd.read_csv(cand)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment CSVs into paper tables"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--prefix", default="", help="Optional dir prefix (e.g. smoke_)"
    )
    args = parser.parse_args()

    missing: List[str] = []
    for table_key, spec in TABLE_SPECS.items():
        source = spec.get("source", table_key)
        df = _load_results(args.results_root, source, args.prefix)
        if df is None:
            missing.append(f"{table_key} (looking for {source})")
            continue

        # Filter to test split if present
        if "split" in df.columns:
            df = df[df["split"] == "test"]

        if table_key == "wheat_head_table5":
            lines = render_wheat_head_table5(df, spec)
        elif table_key == "wheat_head_table6":
            lines = render_wheat_head_table6(df, spec)
        else:
            lines = render_simple_table(df, spec)

        print("\n".join(lines))

    if missing:
        print("\n[warning] No results.csv found for:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
