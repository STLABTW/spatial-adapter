"""Unit tests for spatial_adapter.utils.model_factory."""

import torch

from spatial_adapter.models.spatial_basis_learner import SpatialBasisLearner
from spatial_adapter.models.trend_model import TrendModel
from spatial_adapter.utils.model_factory import create_fresh_models


class TestCreateFreshModels:
    def test_returns_tuple(self):
        w = torch.randn(5)
        trend, basis = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=5,
            n_locations=10,
            latent_dim=3,
            w_ols=w,
            b_ols=1.0,
        )
        assert isinstance(trend, TrendModel)
        assert isinstance(basis, SpatialBasisLearner)

    def test_trend_on_correct_device(self):
        w = torch.randn(4)
        trend, _ = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=4,
            n_locations=8,
            latent_dim=2,
            w_ols=w,
            b_ols=0.5,
        )
        assert next(trend.parameters()).device == torch.device("cpu")

    def test_basis_shape(self):
        w = torch.randn(3)
        _, basis = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=3,
            n_locations=20,
            latent_dim=5,
            w_ols=w,
            b_ols=0.0,
        )
        assert basis.basis.shape == (20, 5)

    def test_default_hidden_layers(self):
        w = torch.randn(3)
        trend, _ = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=3,
            n_locations=10,
            latent_dim=2,
            w_ols=w,
            b_ols=0.0,
        )
        assert trend.res_blocks is not None  # default [256, 64]

    def test_custom_hidden_layers(self):
        w = torch.randn(3)
        trend, _ = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=3,
            n_locations=10,
            latent_dim=2,
            w_ols=w,
            b_ols=0.0,
            hidden_layer_sizes=[16],
        )
        assert trend.res_blocks is not None

    def test_no_hidden_layers(self):
        w = torch.randn(3)
        trend, _ = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=3,
            n_locations=10,
            latent_dim=2,
            w_ols=w,
            b_ols=0.0,
            hidden_layer_sizes=[],
        )
        assert trend.res_blocks is None

    def test_init_lin_frozen(self):
        w = torch.randn(3)
        trend, _ = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=3,
            n_locations=10,
            latent_dim=2,
            w_ols=w,
            b_ols=1.0,
        )
        assert not trend.init_lin.weight.requires_grad

    def test_forward_works(self):
        w = torch.randn(5)
        trend, basis = create_fresh_models(
            device=torch.device("cpu"),
            p_dim=5,
            n_locations=12,
            latent_dim=3,
            w_ols=w,
            b_ols=0.0,
        )
        x = torch.randn(4, 12, 5)
        out = trend(x)
        assert out.shape == (4, 12)
