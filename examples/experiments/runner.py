#!/usr/bin/env python
"""
Unified experiment runner for Spatial Adapter experiments.

Usage:
    # Run a single experiment
    python -m examples.experiments.runner --config examples/experiments/configs/synthetic_1d.yaml

    # Run multiple experiments
    python -m examples.experiments.runner \
        --config examples/experiments/configs/synthetic_1d.yaml \
                 examples/experiments/configs/kaust.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

EXPERIMENT_REGISTRY = {
    "synthetic_1d": "examples.experiments.synthetic_1d.experiment.Synthetic1DExperiment",
    "kaust": "examples.experiments.kaust_2b8.experiment.KAUSTExperiment",
    "weather2k_timesplit": "examples.experiments.weather2k.experiment.Weather2KTimesplitExperiment",
    "weather2k_holdout": "examples.experiments.weather2k.experiment.Weather2KHoldoutExperiment",
    "wheat_head": "examples.experiments.wheat_head.experiment.WheatHeadExperiment",
}


def _import_class(dotted_path: str):
    """Import a class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_config(config_path: str):
    """Load a YAML config and run the corresponding experiment."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["experiment"]["name"]
    if exp_name not in EXPERIMENT_REGISTRY:
        print(f"Unknown experiment: {exp_name}")
        print(f"Available: {list(EXPERIMENT_REGISTRY.keys())}")
        sys.exit(1)

    cls = _import_class(EXPERIMENT_REGISTRY[exp_name])
    experiment = cls(config)
    experiment.run()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Spatial Adapter experiment runner",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="Path(s) to YAML config file(s)",
    )
    args = parser.parse_args()

    for cfg_path in args.config:
        run_config(cfg_path)


if __name__ == "__main__":
    main()
