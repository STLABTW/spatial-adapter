"""
Unit tests for spatial_adapter.utils.experiment module.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel
from spatial_adapter.utils.experiment import log_covariance_and_basis


class TestLogCovarianceAndBasis:
    """Test log_covariance_and_basis function."""

    @pytest.fixture
    def mock_writer(self):
        """Create a mock TensorBoard writer."""
        return Mock(spec=SummaryWriter)

    @pytest.fixture
    def sample_models(self):
        """Create sample trend and basis models."""
        trend = TrendModel(
            num_continuous_features=3,
            hidden_layer_sizes=[16],
            n_locations=5,
            init_weight=torch.randn(3),
            init_bias=1.0,
            dropout_rate=0.1,
        )
        basis = SpatialBasisLearner(num_locations=5, latent_dim=1)
        return trend, basis

    @pytest.fixture
    def sample_data(self):
        """Create sample validation data."""
        val_cont = torch.randn(10, 5, 3)  # (T, N, features)
        val_y = torch.randn(10, 5)  # (T, N)
        locs = np.linspace(-3, 3, 5)
        config = {
            "eigenvalue": 4.0,
            "noise_std": 0.1,
        }
        return val_cont, val_y, locs, config

    def test_log_covariance_and_basis_basic(
        self, mock_writer, sample_models, sample_data
    ):
        """Test basic functionality of log_covariance_and_basis."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        # Call the function
        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.1,
        )

        # Check that writer methods were called (histogram + scalars)
        assert mock_writer.add_histogram.called
        assert mock_writer.add_scalar.called

    def test_log_covariance_and_basis_different_parameters(
        self, mock_writer, sample_models, sample_data
    ):
        """Test with different tau parameters."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=1,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=0.0,
            tau2=0.0,
            best_val=0.05,
        )

        assert mock_writer.add_histogram.called

    def test_log_covariance_and_basis_device_handling(
        self, mock_writer, sample_models, sample_data
    ):
        """Test that function works with different devices."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        # Move models to CPU explicitly
        trend = trend.cpu()
        basis = basis.cpu()
        val_cont = val_cont.cpu()
        val_y = val_y.cpu()

        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.1,
        )

        assert mock_writer.add_histogram.called


class TestExperimentIntegration:
    """Integration tests for experiment functions."""

    def test_log_covariance_and_basis_with_real_models(self):
        """Test with real model instances."""
        trend = TrendModel(
            num_continuous_features=2,
            hidden_layer_sizes=[8],
            n_locations=4,
            init_weight=torch.randn(2),
            init_bias=0.5,
            dropout_rate=0.1,
        )
        basis = SpatialBasisLearner(num_locations=4, latent_dim=1)

        # Create real data
        val_cont = torch.randn(5, 4, 2)
        val_y = torch.randn(5, 4)
        locs = np.linspace(-2, 2, 4)
        config = {"eigenvalue": 2.0, "noise_std": 0.1}

        # Create mock writer
        mock_writer = Mock(spec=SummaryWriter)

        # Test the function
        log_covariance_and_basis(
            writer=mock_writer,
            tag="integration_test",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.15,
        )

        # Verify it completed without errors
        assert mock_writer.add_histogram.called
