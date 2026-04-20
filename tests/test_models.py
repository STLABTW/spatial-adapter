"""
Unit tests for spatial_adapter.models module.
"""

import numpy as np
import pytest
import torch

from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel


class TestTrendModel:
    def test_trend_model_initialization(self):
        """Test TrendModel initialization with OLS weights."""
        p_dim = 5
        n_locations = 10
        hidden_sizes = [64, 32]

        # Create OLS weights
        w_ols = torch.randn(p_dim)
        b_ols = 1.5

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=hidden_sizes,
            n_locations=n_locations,
            init_weight=w_ols,
            init_bias=b_ols,
            freeze_init=True,
            dropout_rate=0.1,
        )

        assert trend.init_lin.weight.shape == (1, p_dim)
        assert trend.init_lin.bias.shape == (1,)
        assert not trend.init_lin.weight.requires_grad  # Should be frozen

        # Check that weights are set correctly
        torch.testing.assert_close(trend.init_lin.weight.squeeze(), w_ols)
        torch.testing.assert_close(trend.init_lin.bias.squeeze(), torch.tensor(b_ols))

    def test_trend_model_forward(self):
        """Test TrendModel forward pass."""
        p_dim = 3
        n_locations = 5
        batch_size = 4

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        )

        x = torch.randn(batch_size, n_locations, p_dim)
        output = trend(x)

        assert output.shape == (batch_size, n_locations)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_trend_model_no_hidden_layers(self):
        """Test TrendModel with no hidden layers (linear only)."""
        p_dim = 3
        n_locations = 5
        batch_size = 4

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.0,
        )

        x = torch.randn(batch_size, n_locations, p_dim)
        output = trend(x)

        assert output.shape == (batch_size, n_locations)
        assert not torch.isnan(output).any()

    def test_trend_model_residual_parameters(self):
        """Test that residual_parameters returns only trainable parameters."""
        p_dim = 3
        n_locations = 5

        # Create model with frozen initialization
        w_ols = torch.randn(p_dim)
        b_ols = 1.0

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64],
            n_locations=n_locations,
            init_weight=w_ols,
            init_bias=b_ols,
            freeze_init=True,
            dropout_rate=0.1,
        )

        residual_params = trend.residual_parameters()

        # Check that init_lin parameters are not in residual_params
        init_params = set(trend.init_lin.parameters())
        residual_param_set = set(residual_params)

        assert len(init_params.intersection(residual_param_set)) == 0

        # Check that other parameters are in residual_params
        assert len(residual_params) > 0


class TestSpatialBasisLearner:
    def test_spatial_basis_learner_initialization(self):
        """Test SpatialBasisLearner initialization."""
        num_locations = 10
        latent_dim = 3

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )

        assert basis.basis.shape == (num_locations, latent_dim)

    def test_spatial_basis_learner_forward(self):
        """Test SpatialBasisLearner forward pass (project residuals onto basis)."""
        num_locations = 10
        latent_dim = 3
        batch_size = 4

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )
        residuals = torch.randn(batch_size, num_locations)
        output = basis(residuals)

        assert output.shape == (batch_size, num_locations)
        assert not torch.isnan(output).any() and not torch.isinf(output).any()

    def test_spatial_basis_learner_pca_init(self):
        """Test SpatialBasisLearner with PCA initialization."""
        num_locations = 10
        latent_dim = 3

        # Create PCA initialization
        pca_init = np.random.randn(num_locations, latent_dim)

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=pca_init,
        )

        # Check that basis is close to PCA init (after orthogonalization)
        basis_np = basis.basis.detach().numpy()
        assert basis_np.shape == (num_locations, latent_dim)

    def test_spatial_basis_learner_project_reconstruct(self):
        """Test projection and reconstruction methods."""
        num_locations = 10
        latent_dim = 3
        batch_size = 4

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )

        # Test data
        data = torch.randn(batch_size, num_locations)

        # Project
        coefficients = basis.project(data)
        assert coefficients.shape == (batch_size, latent_dim)

        # Reconstruct
        reconstructed = basis.reconstruct(coefficients)
        assert reconstructed.shape == (batch_size, num_locations)

        # Check that projection + reconstruction gives reasonable result
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()

    def test_latent_dim_larger_than_locations_raises(self):
        with pytest.raises(ValueError, match="latent_dim must be"):
            SpatialBasisLearner(num_locations=5, latent_dim=10)

    def test_pca_init_shape_mismatch_raises(self):
        bad = np.random.randn(5, 4)  # N=5, K=4 doesn't match K=3
        with pytest.raises(ValueError, match="pca_init must have shape"):
            SpatialBasisLearner(num_locations=5, latent_dim=3, pca_init=bad)

    def test_get_basis_returns_parameter(self):
        basis = SpatialBasisLearner(num_locations=8, latent_dim=2)
        out = basis.get_basis()
        assert out.shape == (8, 2)
        assert out is basis.basis  # same tensor, not a copy

    def test_forward_output_is_orthogonal_projection(self):
        """After _retract(), basis columns should be orthonormal → P = Φ Φᵀ is idempotent."""
        torch.manual_seed(0)
        basis = SpatialBasisLearner(num_locations=12, latent_dim=4)
        residuals = torch.randn(3, 12)
        projected = basis(residuals)
        # Applying the projection again should be a fixed point
        projected_twice = basis(projected)
        torch.testing.assert_close(projected, projected_twice, atol=1e-5, rtol=1e-5)

    def test_reinit_from_pca_numpy(self):
        basis = SpatialBasisLearner(num_locations=6, latent_dim=2)
        new_basis = np.random.default_rng(1).normal(size=(6, 2))
        basis.reinit_from_pca(new_basis)
        # After retraction, columns are orthonormal
        B = basis.basis.detach().numpy()
        gram = B.T @ B
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-5)

    def test_reinit_from_pca_torch_tensor(self):
        basis = SpatialBasisLearner(num_locations=6, latent_dim=2)
        new_basis = torch.randn(6, 2, dtype=torch.float64)
        basis.reinit_from_pca(new_basis)
        B = basis.basis.detach()
        gram = B.T @ B
        torch.testing.assert_close(gram, torch.eye(2), atol=1e-5, rtol=1e-5)

    def test_reinit_from_pca_shape_mismatch_raises(self):
        basis = SpatialBasisLearner(num_locations=6, latent_dim=2)
        with pytest.raises(ValueError, match="Shape mismatch"):
            basis.reinit_from_pca(np.zeros((5, 2)))
