"""
Unit tests for L2: BCE ℓ_data and ∇_Z (data_losses module).
"""

import pytest
import torch

from spatial_adapter.models.data_losses import (
    grad_Z_loss_data_binary,
    loss_data_binary,
    projection_orthogonal_complement,
)


class TestLossDataBinary:
    """Test loss_data_binary(P_perp_Z, Y)."""

    def test_loss_scalar_return(self):
        """Return value is a scalar tensor."""
        T, N = 4, 6
        P_perp_Z = torch.randn(T, N)
        Y = torch.empty(T, N).uniform_(0, 1)
        loss = loss_data_binary(P_perp_Z, Y)
        assert loss.dim() == 0
        assert loss.dtype in (torch.float32, torch.float64)
        assert loss.item() >= 0

    def test_loss_shape_mismatch_raises(self):
        """Raises when P_perp_Z and Y shapes differ."""
        P_perp_Z = torch.randn(3, 5)
        Y = torch.rand(3, 4)
        with pytest.raises(ValueError, match="shape"):
            loss_data_binary(P_perp_Z, Y)

    def test_loss_perfect_prediction_low(self):
        """When logits -> 0 and Y=0, loss is small."""
        P_perp_Z = torch.tensor([[-10.0, -10.0], [-10.0, -10.0]])
        Y = torch.zeros(2, 2)
        loss = loss_data_binary(P_perp_Z, Y)
        assert loss.item() < 0.01

    def test_loss_perfect_prediction_high(self):
        """When logits -> +inf and Y=1, loss is small."""
        P_perp_Z = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
        Y = torch.ones(2, 2)
        loss = loss_data_binary(P_perp_Z, Y)
        assert loss.item() < 0.01

    def test_loss_requires_grad(self):
        """Loss backward gives gradient on P_perp_Z."""
        P_perp_Z = torch.randn(2, 3, requires_grad=True)
        Y = torch.empty(2, 3).uniform_(0, 1)
        loss = loss_data_binary(P_perp_Z, Y)
        loss.backward()
        assert P_perp_Z.grad is not None
        assert P_perp_Z.grad.shape == P_perp_Z.shape


class TestGradZLossDataBinary:
    """Test grad_Z_loss_data_binary(Z, P_perp, Y)."""

    def test_returns_loss_and_grad(self):
        """Returns (loss, Z_grad) with correct shapes."""
        T, N = 5, 4
        Z = torch.randn(T, N)
        Phi = torch.randn(N, 2)
        P_perp = projection_orthogonal_complement(Phi)
        Y = torch.empty(T, N).uniform_(0, 1)

        loss, Z_grad = grad_Z_loss_data_binary(Z, P_perp, Y)

        assert loss.dim() == 0
        assert Z_grad.shape == Z.shape
        assert Z_grad.dtype == Z.dtype

    def test_grad_nonzero_when_wrong(self):
        """Gradient is non-zero when prediction is wrong."""
        T, N = 3, 4
        Z = torch.randn(T, N)
        Phi = torch.eye(N, 2)
        P_perp = projection_orthogonal_complement(Phi)
        Y = torch.ones(T, N)

        loss, Z_grad = grad_Z_loss_data_binary(Z, P_perp, Y)

        assert Z_grad.abs().sum().item() > 1e-6

    def test_shape_mismatch_raises(self):
        """Raises on shape mismatch."""
        Z = torch.randn(3, 4)
        P_perp = torch.eye(5, 5)
        Y = torch.rand(3, 4)
        with pytest.raises(ValueError):
            grad_Z_loss_data_binary(Z, P_perp, Y)

    def test_non_2d_input_raises(self):
        """All inputs must be 2D — a 1D Z hits the dim check before the shape check."""
        Z = torch.randn(5)  # 1D — triggers the dim != 2 branch
        P_perp = torch.eye(4)
        Y = torch.rand(2, 4)
        with pytest.raises(ValueError, match="must be 2D"):
            grad_Z_loss_data_binary(Z, P_perp, Y)


class TestProjectionOrthogonalComplement:
    """Test projection_orthogonal_complement(Phi)."""

    def test_shape(self):
        """P_perp is (N, N) when Phi is (N, K)."""
        N, K = 6, 3
        Phi = torch.randn(N, K)
        P_perp = projection_orthogonal_complement(Phi)
        assert P_perp.shape == (N, N)

    def test_symmetric(self):
        """P^⊥ is symmetric."""
        Phi = torch.randn(5, 2)
        P_perp = projection_orthogonal_complement(Phi)
        torch.testing.assert_close(P_perp, P_perp.T)

    def test_idempotent(self):
        """P^⊥ P^⊥ = P^⊥ when Φ has orthonormal columns (as in adapter)."""
        Phi = torch.randn(4, 2)
        Phi, _ = torch.linalg.qr(Phi)
        P_perp = projection_orthogonal_complement(Phi)
        torch.testing.assert_close(P_perp @ P_perp, P_perp, atol=1e-5, rtol=1e-5)


class TestGradZNumericalValidation:
    """Validate ∇_Z ℓ_data and proximal step (no closed form for binary → gradient descent)."""

    def test_grad_Z_matches_finite_difference(self):
        """∇_Z ℓ_data from autograd matches finite-difference in a random direction."""
        T, N = 3, 4
        Z = torch.randn(T, N, dtype=torch.float64)
        Phi = torch.randn(N, 2, dtype=torch.float64)
        Phi, _ = torch.linalg.qr(Phi)
        P_perp = projection_orthogonal_complement(Phi)
        Y = torch.empty(T, N, dtype=torch.float64).uniform_(0.2, 0.8)

        _, Z_grad = grad_Z_loss_data_binary(Z, P_perp, Y)
        Z_grad = Z_grad.double()

        # Directional derivative: (f(Z + eps*d) - f(Z - eps*d)) / (2*eps) ≈ (∇f)·d
        d = torch.randn(T, N, dtype=torch.float64)
        d = d / (d.norm() + 1e-10)
        eps = 1e-6
        Z_plus = (Z + eps * d).detach().requires_grad_(True)
        Z_minus = (Z - eps * d).detach().requires_grad_(True)
        f_plus = loss_data_binary(Z_plus @ P_perp, Y)
        f_minus = loss_data_binary(Z_minus @ P_perp, Y)
        numerical_dir_deriv = (f_plus.item() - f_minus.item()) / (2 * eps)
        analytical_dir_deriv = (Z_grad * d).sum().item()
        assert abs(numerical_dir_deriv - analytical_dir_deriv) < 1e-4 * (
            abs(analytical_dir_deriv) + 1e-8
        )

    def test_proximal_step_decreases_objective(self):
        """One proximal step Z_new = Z - α(∇_Z ℓ_data + (ρ/LN)(Z−R+U)) decreases the full objective."""
        T, N = 4, 5
        Z = torch.randn(T, N) * 0.5
        R = torch.randn(T, N) * 0.5
        U = torch.randn(T, N) * 0.1
        Phi = torch.randn(N, 2)
        Phi, _ = torch.linalg.qr(Phi)
        P_perp = projection_orthogonal_complement(Phi)
        Y = torch.empty(T, N).uniform_(0.2, 0.8)
        rho = 1.0
        alpha = 0.1
        L, N_dim = T, N

        def objective(z):
            data = loss_data_binary(z @ P_perp, Y)
            penalty = 0.5 * rho * ((z - R + U) ** 2).sum()
            return data + penalty

        obj_old = objective(Z).item()
        _, data_grad = grad_Z_loss_data_binary(Z, P_perp, Y)
        penalty_grad = (rho / (L * N_dim)) * (Z - R + U)
        Z_new = Z - alpha * (data_grad + penalty_grad)
        obj_new = objective(Z_new).item()

        assert (
            obj_new < obj_old
        ), f"Proximal step should decrease objective: {obj_old} -> {obj_new}"
