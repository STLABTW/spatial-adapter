"""Experiment helpers: OLS computation, GPU memory, device info."""

from typing import Dict, Tuple

import torch


def predict_ols(X_val: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
    """OLS prediction: (B, N, F) @ w + b → (B, N)."""
    device = X_val.device
    w = w.to(device)
    flat = X_val.reshape(-1, X_val.shape[-1]).float()
    return (flat @ w + torch.tensor(b, device=device, dtype=torch.float32)).reshape(
        X_val.shape[0], X_val.shape[1]
    )


def clear_gpu_memory() -> None:
    """Empty CUDA cache and synchronize."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_ols_coefficients(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device = None,
) -> Tuple[torch.Tensor, float]:
    """Solve OLS via least squares. Returns (w, b) as (float32 Tensor, float)."""
    if device is None:
        device = X_train.device
    X_flat = X_train.reshape(-1, X_train.shape[-1]).double().to(device)
    y_flat = y_train.reshape(-1, 1).double().to(device)
    driver = "gels" if device.type == "cuda" else "gelsy"
    coef = torch.linalg.lstsq(
        torch.cat([X_flat, torch.ones_like(y_flat)], 1),
        y_flat,
        driver=driver,
    ).solution.squeeze()
    return coef[:-1].float(), float(coef[-1])


def create_experiment_config(
    n_trials_per_seed: int = None,
    n_dataset_seeds: int = 10,
    seed_range_start: int = 1,
    seed_range_end: int = 11,
) -> Dict:
    """Default experiment config dict (Optuna trials × seeds)."""
    if n_trials_per_seed is None:
        n_trials_per_seed = 20 if torch.cuda.is_available() else 50
    return {
        "n_trials_per_seed": n_trials_per_seed,
        "n_dataset_seeds": n_dataset_seeds,
        "seed_range_start": seed_range_start,
        "seed_range_end": seed_range_end,
    }


def print_experiment_summary(config: Dict) -> None:
    """Print experiment config to stdout."""
    total = config["n_trials_per_seed"] * (
        config["seed_range_end"] - config["seed_range_start"]
    )
    print(
        f"Experiment: {config['n_trials_per_seed']} trials/seed, "
        f"seeds {config['seed_range_start']}–{config['seed_range_end']-1}, "
        f"total {total}, device={'GPU' if torch.cuda.is_available() else 'CPU'}"
    )


def get_device_info() -> Dict[str, str]:
    """Return a dict describing the current CUDA / CPU setup."""
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "memory_gb": f"{mem:.1f}",
            "device_count": torch.cuda.device_count(),
        }
    return {
        "device": "cpu",
        "device_name": "CPU",
        "memory_gb": "N/A",
        "device_count": 0,
    }
