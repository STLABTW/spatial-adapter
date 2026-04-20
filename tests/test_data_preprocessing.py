"""
Unit tests for spatial_adapter.data.preprocessing module.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from spatial_adapter.data.preprocessing import (
    DataPreprocessor,
    denormalize_predictions,
    prepare_all,
    prepare_all_with_scaling,
)


class TestPrepareAll:
    """Test prepare_all function."""

    def test_data_splitting(self):
        """Test that data is correctly split into train/val/test."""
        # Create dummy data
        T, N = 100, 10
        cat = np.zeros((T, N, 1), dtype=np.int64)
        cont = np.random.randn(T, N, 5).astype(np.float64)
        y = np.random.randn(T, N).astype(np.float64)

        train, val, test = prepare_all(cat, cont, y)

        # Check that we get TensorDatasets
        assert isinstance(train, TensorDataset)
        assert isinstance(val, TensorDataset)
        assert isinstance(test, TensorDataset)

        # Check shapes
        train_cat, train_cont, train_y = train.tensors
        val_cat, val_cont, val_y = val.tensors
        test_cat, test_cont, test_y = test.tensors

        # Default ratios: 0.7, 0.15, 0.15
        expected_train = int(T * 0.7)
        expected_val = int(T * 0.15)

        assert train_cat.shape[0] == expected_train
        assert val_cat.shape[0] == expected_val
        assert test_cat.shape[0] == T - expected_train - expected_val

        # Check data types
        assert train_cat.dtype == torch.int64
        assert train_cont.dtype == torch.float32
        assert train_y.dtype == torch.float32

    def test_custom_ratios(self):
        """Test data splitting with custom ratios."""
        T, N = 100, 10
        cat = np.zeros((T, N, 1), dtype=np.int64)
        cont = np.random.randn(T, N, 5).astype(np.float64)
        y = np.random.randn(T, N).astype(np.float64)

        train, val, test = prepare_all(cat, cont, y, train_ratio=0.6, val_ratio=0.2)

        train_cat, _, _ = train.tensors
        val_cat, _, _ = val.tensors
        test_cat, _, _ = test.tensors

        expected_train = int(T * 0.6)
        expected_val = int(T * 0.2)

        assert train_cat.shape[0] == expected_train
        assert val_cat.shape[0] == expected_val
        assert test_cat.shape[0] == T - expected_train - expected_val


class TestDataPreprocessor:
    """Unit tests for the DataPreprocessor scaler wrapper."""

    @staticmethod
    def _dummy(T=30, N=4, F=3, seed=0):
        rng = np.random.default_rng(seed)
        cont = rng.normal(loc=5.0, scale=2.0, size=(T, N, F)).astype(np.float64)
        targets = rng.normal(loc=10.0, scale=3.0, size=(T, N)).astype(np.float64)
        return cont, targets

    def test_standard_scaler_fit_transform_roundtrip(self):
        cont, targets = self._dummy()
        pp = DataPreprocessor()  # defaults: standard/standard
        pp.fit_scalers(cont, targets)
        assert pp.is_fitted

        cont_s, targ_s = pp.transform_data(cont, targets)
        assert cont_s.shape == cont.shape
        assert targ_s.shape == targets.shape

        # Standard-scaled features are approximately zero-mean, unit-variance
        flat = cont_s.reshape(-1, cont.shape[-1])
        np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(flat.std(axis=0), 1.0, atol=1e-6)

        # inverse_transform_targets recovers the original targets
        recovered = pp.inverse_transform_targets(targ_s)
        np.testing.assert_allclose(recovered, targets, atol=1e-6)

    def test_minmax_scaler_bounds(self):
        cont, targets = self._dummy()
        pp = DataPreprocessor(feature_scaler_type="minmax", target_scaler_type="minmax")
        pp.fit_scalers(cont, targets)
        cont_s, targ_s = pp.transform_data(cont, targets)
        # MinMax scales into [0, 1] per feature column
        assert cont_s.min() >= -1e-9
        assert cont_s.max() <= 1.0 + 1e-9
        assert targ_s.min() >= -1e-9
        assert targ_s.max() <= 1.0 + 1e-9

    def test_unknown_scaler_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scaler type"):
            DataPreprocessor(feature_scaler_type="notreal")

    def test_transform_before_fit_raises(self):
        cont, targets = self._dummy()
        pp = DataPreprocessor()
        with pytest.raises(ValueError, match="must be fitted"):
            pp.transform_data(cont, targets)

    def test_inverse_transform_before_fit_raises(self):
        pp = DataPreprocessor()
        with pytest.raises(ValueError, match="must be fitted"):
            pp.inverse_transform_targets(np.zeros((2, 3)))

    def test_get_scaler_info_not_fitted(self):
        pp = DataPreprocessor()
        info = pp.get_scaler_info()
        assert "error" in info

    def test_get_scaler_info_after_fit(self):
        cont, targets = self._dummy()
        pp = DataPreprocessor()
        pp.fit_scalers(cont, targets)
        info = pp.get_scaler_info()
        # Standard scaler populates mean_/scale_ for both feature and target
        assert "mean_" in info and info["mean_"] is not None
        assert "target_mean_" in info and info["target_mean_"] is not None


class TestPrepareAllWithScaling:
    def test_returns_scaled_datasets_and_preprocessor(self):
        T, N, F = 40, 6, 3
        rng = np.random.default_rng(42)
        cat = np.zeros((T, N, 1), dtype=np.int64)
        cont = rng.normal(size=(T, N, F)).astype(np.float64)
        y = rng.normal(size=(T, N)).astype(np.float64)

        train, val, test, pp = prepare_all_with_scaling(
            cat, cont, y, train_ratio=0.6, val_ratio=0.2
        )
        assert isinstance(pp, DataPreprocessor)
        assert pp.is_fitted

        # Split ratios preserved
        assert train.tensors[0].shape[0] == int(T * 0.6)
        assert val.tensors[0].shape[0] == int(T * 0.2)
        assert test.tensors[0].shape[0] == T - int(T * 0.6) - int(T * 0.2)

        # Scalers fit only on training window: train segment has ~zero mean
        train_cont = train.tensors[1].numpy()
        np.testing.assert_allclose(
            train_cont.reshape(-1, F).mean(axis=0), 0.0, atol=1e-6
        )


class TestDenormalizePredictions:
    def test_round_trip_matches_original_targets(self):
        T, N, F = 20, 5, 2
        rng = np.random.default_rng(7)
        cont = rng.normal(loc=3.0, size=(T, N, F)).astype(np.float64)
        targets = rng.normal(loc=100.0, scale=10.0, size=(T, N)).astype(np.float64)

        pp = DataPreprocessor()
        pp.fit_scalers(cont, targets)
        _, targets_scaled = pp.transform_data(cont, targets)

        recovered = denormalize_predictions(targets_scaled, pp)
        np.testing.assert_allclose(recovered, targets, atol=1e-6)
