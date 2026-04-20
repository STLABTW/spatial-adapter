"""
Integration test: prediction intervals on the synthetic 1D benchmark.

Uses the paper's Section 3.1 data-generating process with **known**
(oracle) parameters to verify that the prediction interval machinery
in ``spatial_adapter.prediction`` achieves the correct nominal
coverage.  No adapter training is required — the test verifies the
interval code itself, not the ADMM estimation.

The DGP is:
    Y(s_i, t) = mu_t + alpha_t * phi(s_i) + eps_{i,t}

with:
    phi(s) = exp(-s^2) / ||exp(-s^2)||_2   (rank K=1 basis)
    alpha_t ~ AR(1) with marginal variance lambda_true
    eps_{i,t} ~ N(0, sigma_true^2)

Under the Gaussian working model, the conditional predictive
distribution at a held-out location s* given the observed residuals
at all other locations is:

    eta(s*, t) | r_{t, O} ~ N(eta_hat, v_hat)

where eta_hat and v_hat are computed by the prediction module.
The test checks that the empirical coverage of the resulting
intervals matches the nominal level.
"""


import numpy as np
import pytest

from spatial_adapter.prediction import (
    conditional_covariance,
    conditional_gaussian_interval,
    conditional_score,
    prediction_interval,
    predictive_variance_batch,
)


def _generate_oracle_synthetic(
    N: int = 128,
    T: int = 500,
    lambda_true: float = 16.0,
    sigma_true: float = 4.0,
    seed: int = 42,
):
    """
    Generate synthetic data with known spatial structure and return
    both the data and the oracle parameters.

    Returns
    -------
    y : (T, N) array — observations
    phi : (N, 1) array — true basis (normalised Gaussian kernel)
    Lambda : (1, 1) array — true score covariance
    sigma2 : float — true noise variance
    locs : (N,) array — 1D spatial locations
    """
    rng = np.random.default_rng(seed)
    locs = np.linspace(-3, 3, N)

    # Basis: phi(s) = exp(-s^2) / ||...||
    phi_raw = np.exp(-(locs**2))
    phi = (phi_raw / np.linalg.norm(phi_raw)).reshape(N, 1)

    # Score covariance (K=1 → scalar)
    Lambda = np.array([[lambda_true]])
    sigma2 = sigma_true**2

    # Generate scores alpha_t ~ N(0, lambda_true)
    alpha = rng.standard_normal((T, 1)) * np.sqrt(lambda_true)

    # Generate observations: Y = alpha @ phi^T + noise
    noise = rng.standard_normal((T, N)) * sigma_true
    y = alpha @ phi.T + noise

    return y, phi, Lambda, sigma2, locs


class TestSyntheticPredictionInterval:
    """End-to-end coverage tests on the synthetic 1D benchmark."""

    def test_marginal_coverage_95(self):
        """
        O_j = ∅ (no residual observed) → marginal interval.

        With oracle parameters, nominal 95% coverage should be
        achieved within sampling noise (~1-2% tolerance for T=2000).
        """
        y, phi, Lambda, sigma2, locs = _generate_oracle_synthetic(N=64, T=2000, seed=0)

        # Marginal: O_j = ∅ → Lambda_cond = Lambda
        Phi_O = np.zeros((0, 1))
        Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2)

        # Predictive variance at every location (marginal)
        pred_var = predictive_variance_batch(phi, Lambda_cond, sigma2)

        # Point prediction with no conditioning: alpha_hat = 0
        # so eta_hat = 0 (no trend in this simplified test)
        eta_hat = np.zeros_like(y)  # (T, N)
        pred_var_mat = np.broadcast_to(pred_var, y.shape)  # same var every t

        # 95% interval
        lo, hi = conditional_gaussian_interval(eta_hat, pred_var_mat, alpha=0.05)

        # Empirical coverage
        inside = (y >= lo) & (y <= hi)
        coverage = np.mean(inside)

        assert coverage == pytest.approx(
            0.95, abs=0.02
        ), f"marginal coverage = {coverage:.4f}, expected ~0.95"

    def test_conditional_coverage_95(self):
        """
        O_j = all locations except s* → conditional interval (full kriging).

        Hold out one location at a time, condition on the rest,
        and check coverage.
        """
        N, T = 32, 3000
        y, phi, Lambda, sigma2, locs = _generate_oracle_synthetic(N=N, T=T, seed=1)

        # Pick a few held-out locations to test
        test_locs = [0, N // 4, N // 2, 3 * N // 4, N - 1]
        coverages = []

        for held_out in test_locs:
            obs_idx = [i for i in range(N) if i != held_out]
            Phi_O = phi[obs_idx]  # (N-1, K)
            phi_star = phi[held_out]  # (K,)

            Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2)
            v_hat = float(sigma2 + phi_star @ Lambda_cond @ phi_star)

            # For each time step, compute conditional prediction
            inside_count = 0
            for t in range(T):
                r_O = y[t, obs_idx]  # residuals at observed locations
                alpha_hat = conditional_score(Lambda_cond, Phi_O, r_O, sigma2)
                eta_hat_t = float(phi_star @ alpha_hat)

                lo, hi = conditional_gaussian_interval(
                    np.array([eta_hat_t]), np.array([v_hat]), alpha=0.05
                )
                if lo[0] <= y[t, held_out] <= hi[0]:
                    inside_count += 1

            coverage = inside_count / T
            coverages.append(coverage)

        mean_coverage = np.mean(coverages)
        assert mean_coverage == pytest.approx(0.95, abs=0.025), (
            f"conditional coverage = {mean_coverage:.4f} "
            f"(per-location: {[f'{c:.3f}' for c in coverages]}), expected ~0.95"
        )

    def test_conditional_tighter_than_marginal(self):
        """
        Conditioning on observed residuals should tighten the interval
        (v_hat_cond < v_hat_marginal) at every location.
        """
        y, phi, Lambda, sigma2, locs = _generate_oracle_synthetic(N=32, T=100, seed=2)

        # Marginal variance
        Lambda_cond_marginal = conditional_covariance(Lambda, np.zeros((0, 1)), sigma2)
        v_marginal = predictive_variance_batch(phi, Lambda_cond_marginal, sigma2)

        # Conditional variance (observe all but location 0)
        obs_idx = list(range(1, 32))
        Lambda_cond = conditional_covariance(Lambda, phi[obs_idx], sigma2)
        v_cond = predictive_variance_batch(phi[:1], Lambda_cond, sigma2)

        assert v_cond[0] < v_marginal[0], (
            f"conditional variance {v_cond[0]:.4f} should be < "
            f"marginal {v_marginal[0]:.4f}"
        )

    def test_binary_interval_bounds(self):
        """
        Binary (logistic-normal) intervals should always be in [0, 1]
        and lower ≤ upper.
        """
        y, phi, Lambda, sigma2, locs = _generate_oracle_synthetic(N=32, T=100, seed=3)

        pred_var = predictive_variance_batch(
            phi,
            conditional_covariance(Lambda, np.zeros((0, 1)), sigma2),
            sigma2,
        )
        # Use y as logit-scale predictions (arbitrary but valid)
        eta_hat = y[:10]  # (10, 32)
        pred_var_mat = np.broadcast_to(pred_var, eta_hat.shape)

        lo, hi = prediction_interval(eta_hat, pred_var_mat, task="binary")

        assert np.all(lo >= 0) and np.all(lo <= 1), "lower bound outside [0,1]"
        assert np.all(hi >= 0) and np.all(hi <= 1), "upper bound outside [0,1]"
        assert np.all(lo <= hi + 1e-12), "lower > upper"

    def test_narrower_alpha_gives_narrower_interval(self):
        """
        A 90% interval should be strictly narrower than a 95% interval
        at every location.
        """
        y, phi, Lambda, sigma2, locs = _generate_oracle_synthetic(N=16, T=10, seed=4)

        pred_var = predictive_variance_batch(
            phi,
            conditional_covariance(Lambda, np.zeros((0, 1)), sigma2),
            sigma2,
        )
        eta_hat = np.zeros(16)

        _, hi_90 = conditional_gaussian_interval(eta_hat, pred_var, alpha=0.10)
        _, hi_95 = conditional_gaussian_interval(eta_hat, pred_var, alpha=0.05)
        _, hi_99 = conditional_gaussian_interval(eta_hat, pred_var, alpha=0.01)

        assert np.all(hi_90 < hi_95), "90% should be narrower than 95%"
        assert np.all(hi_95 < hi_99), "95% should be narrower than 99%"
