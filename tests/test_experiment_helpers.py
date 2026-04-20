"""Unit tests for spatial_adapter.utils.experiment_helpers."""

import pytest
import torch

from spatial_adapter.utils.experiment_helpers import (
    clear_gpu_memory,
    compute_ols_coefficients,
    create_experiment_config,
    get_device_info,
    predict_ols,
    print_experiment_summary,
)


class TestPredictOls:
    def test_shape(self):
        X = torch.randn(4, 10, 5)
        w = torch.randn(5)
        out = predict_ols(X, w, b=1.0)
        assert out.shape == (4, 10)

    def test_zero_weights_returns_bias(self):
        X = torch.randn(2, 3, 4)
        w = torch.zeros(4)
        out = predict_ols(X, w, b=5.0)
        assert out.mean().item() == pytest.approx(5.0, abs=1e-5)

    def test_deterministic(self):
        X = torch.randn(3, 8, 6)
        w = torch.randn(6)
        o1 = predict_ols(X, w, 0.5)
        o2 = predict_ols(X, w, 0.5)
        assert torch.equal(o1, o2)


class TestComputeOlsCoefficients:
    def test_returns_w_and_b(self):
        X = torch.randn(50, 10, 3)
        y = torch.randn(50, 10)
        w, b = compute_ols_coefficients(X, y)
        assert w.shape == (3,)
        assert isinstance(b, float)

    def test_perfect_linear(self):
        """y = 2*x1 + 3*x2 + 1 → should recover w=[2,3], b=1."""
        torch.manual_seed(0)
        X = torch.randn(200, 5, 2)
        w_true = torch.tensor([2.0, 3.0])
        b_true = 1.0
        y = (X.reshape(-1, 2) @ w_true + b_true).reshape(200, 5)
        w, b = compute_ols_coefficients(X, y)
        assert w[0].item() == pytest.approx(2.0, abs=0.01)
        assert w[1].item() == pytest.approx(3.0, abs=0.01)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_dtype_is_float32(self):
        X = torch.randn(20, 5, 3)
        y = torch.randn(20, 5)
        w, _ = compute_ols_coefficients(X, y)
        assert w.dtype == torch.float32


class TestCreateExperimentConfig:
    def test_default_keys(self):
        cfg = create_experiment_config()
        assert "n_trials_per_seed" in cfg
        assert "n_dataset_seeds" in cfg
        assert "seed_range_start" in cfg
        assert "seed_range_end" in cfg

    def test_custom_values(self):
        cfg = create_experiment_config(n_trials_per_seed=10, n_dataset_seeds=5)
        assert cfg["n_trials_per_seed"] == 10
        assert cfg["n_dataset_seeds"] == 5


class TestPrintExperimentSummary:
    def test_no_crash(self, capsys):
        cfg = create_experiment_config(n_trials_per_seed=5)
        print_experiment_summary(cfg)
        captured = capsys.readouterr()
        assert "5 trials/seed" in captured.out


class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert info["device"] in ("cpu", "cuda")

    def test_has_device_name(self):
        info = get_device_info()
        assert "device_name" in info


class TestClearGpuMemory:
    def test_no_crash_on_cpu(self):
        clear_gpu_memory()  # should be a no-op on CPU
