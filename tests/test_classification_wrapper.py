"""Unit tests for ClassificationWrapper."""

import torch

from spatial_adapter.models.classification_wrapper import ClassificationWrapper


class TestClassificationWrapper:
    def test_forward_shape(self):
        """(B, N, p) → (B, N) logits."""
        model = ClassificationWrapper(feature_dim=32, n_locations=16)
        x = torch.randn(4, 16, 32)
        out = model(x)
        assert out.shape == (4, 16)

    def test_forward_with_hidden(self):
        model = ClassificationWrapper(feature_dim=32, hidden_dims=[16, 8])
        x = torch.randn(2, 10, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_single_location(self):
        model = ClassificationWrapper(feature_dim=64, n_locations=1)
        x = torch.randn(8, 1, 64)
        out = model(x)
        assert out.shape == (8, 1)

    def test_residual_parameters_all_trainable(self):
        model = ClassificationWrapper(feature_dim=16, hidden_dims=[8])
        params = model.residual_parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_no_hidden_is_linear(self):
        model = ClassificationWrapper(feature_dim=4)
        assert isinstance(model.head, torch.nn.Linear)
        assert model.head.in_features == 4
        assert model.head.out_features == 1

    def test_output_is_differentiable(self):
        model = ClassificationWrapper(feature_dim=8, hidden_dims=[4])
        x = torch.randn(2, 3, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_backbone_integration(self):
        """With a backbone, accepts (B, N, C, H, W)."""
        backbone = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        # backbone: (B*N, 3, 4, 4) → (B*N, 3)
        model = ClassificationWrapper(feature_dim=3, backbone=backbone)
        x = torch.randn(2, 5, 3, 4, 4)
        out = model(x)
        assert out.shape == (2, 5)

    def test_no_backbone_5d_ignored(self):
        """Without backbone, 5D input should NOT trigger backbone path."""
        model = ClassificationWrapper(feature_dim=8)
        x = torch.randn(2, 3, 8)  # normal 3D input
        out = model(x)
        assert out.shape == (2, 3)
