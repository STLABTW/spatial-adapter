"""
Unit tests for spatial_adapter.prediction module.

Tests cover the building blocks (conditional covariance, conditional
score, predictive variance) and the two interval types (conditional
Gaussian for regression, logistic-normal for binary), cross-checked
against the paper's equations eq:lambda-cond, eq:point-pred,
eq:kriging-se, eq:cgi, eq:lnui.
"""


import numpy as np
import pytest
from scipy import stats

from spatial_adapter.prediction import (
    calibrate_q,
    calibrated_interval,
    conditional_covariance,
    conditional_gaussian_interval,
    conditional_score,
    logistic_normal_interval,
    prediction_interval,
    predictive_variance,
    predictive_variance_batch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_setup():
    """A small K=2 setup with known Lambda, Phi_O, sigma2."""
    Lambda = np.diag([4.0, 1.0])
    Phi_O = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])  # 3 observed locations
    sigma2 = 0.5
    return Lambda, Phi_O, sigma2


# ---------------------------------------------------------------------------
# conditional_covariance (eq:lambda-cond)
# ---------------------------------------------------------------------------


class TestConditionalCovariance:
    def test_empty_observation_set(self):
        """O_j = ∅ → Lambda_cond = Lambda (marginal)."""
        Lambda = np.diag([4.0, 2.0])
        Phi_O = np.zeros((0, 2))
        Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2=1.0)
        np.testing.assert_allclose(Lambda_cond, Lambda, atol=1e-8)

    def test_shrinks_below_marginal(self, simple_setup):
        """With observations, Lambda_cond ≤ Lambda in Loewner order."""
        Lambda, Phi_O, sigma2 = simple_setup
        Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2)
        diff = Lambda - Lambda_cond
        eigvals = np.linalg.eigvalsh(diff)
        assert eigvals.min() >= -1e-10, "Lambda_cond should be ≤ Lambda"

    def test_symmetry(self, simple_setup):
        Lambda, Phi_O, sigma2 = simple_setup
        Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2)
        np.testing.assert_allclose(Lambda_cond, Lambda_cond.T, atol=1e-12)

    def test_psd(self, simple_setup):
        Lambda, Phi_O, sigma2 = simple_setup
        Lambda_cond = conditional_covariance(Lambda, Phi_O, sigma2)
        assert np.linalg.eigvalsh(Lambda_cond).min() >= -1e-10

    def test_matches_manual_formula(self):
        """Cross-check: (Lambda^{-1} + sigma^{-2} Phi^T Phi)^{-1}."""
        Lambda = np.array([[3.0, 0.5], [0.5, 2.0]])
        Phi_O = np.array([[1.0, 0.0], [0.0, 1.0]])
        sigma2 = 0.25
        expected = np.linalg.inv(np.linalg.inv(Lambda) + Phi_O.T @ Phi_O / sigma2)
        result = conditional_covariance(Lambda, Phi_O, sigma2)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# conditional_score (eq:point-pred)
# ---------------------------------------------------------------------------


class TestConditionalScore:
    def test_empty_observation_returns_zero(self):
        """alpha_hat(∅) = 0 by convention."""
        Lambda_cond = np.eye(3)
        Phi_O = np.zeros((0, 3))
        r_O = np.array([])
        alpha = conditional_score(Lambda_cond, Phi_O, r_O, sigma2=1.0)
        np.testing.assert_array_equal(alpha, np.zeros(3))

    def test_known_value(self):
        """Simple 1D case: Lambda_cond=[[a]], Phi_O=[[1]], r_O=[r], sigma2=s."""
        a, r, s = 2.0, 3.0, 0.5
        Lambda_cond = np.array([[a]])
        Phi_O = np.array([[1.0]])
        r_O = np.array([r])
        expected = a * 1.0 * r / s  # = 2 * 1 * 3 / 0.5 = 12
        result = conditional_score(Lambda_cond, Phi_O, r_O, s)
        assert result[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# predictive_variance (eq:kriging-se)
# ---------------------------------------------------------------------------


class TestPredictiveVariance:
    def test_minimum_is_sigma2(self):
        """v_hat >= sigma2 always (Lambda_cond is PSD)."""
        sigma2 = 0.5
        Lambda_cond = np.diag([1.0, 2.0])
        phi_star = np.array([0.0, 0.0])  # zero basis → v = sigma2
        assert predictive_variance(phi_star, Lambda_cond, sigma2) == pytest.approx(
            sigma2
        )

    def test_known_value(self):
        sigma2 = 0.3
        Lambda_cond = np.diag([2.0, 1.0])
        phi_star = np.array([1.0, 1.0])
        # v = 0.3 + [1,1] @ diag(2,1) @ [1,1]^T = 0.3 + 2 + 1 = 3.3
        assert predictive_variance(phi_star, Lambda_cond, sigma2) == pytest.approx(3.3)

    def test_batch_matches_loop(self):
        rng = np.random.default_rng(0)
        K = 3
        Lambda_cond = rng.standard_normal((K, K))
        Lambda_cond = Lambda_cond.T @ Lambda_cond  # PSD
        sigma2 = 0.5
        Phi_star = rng.standard_normal((10, K))

        batch_result = predictive_variance_batch(Phi_star, Lambda_cond, sigma2)
        loop_result = np.array(
            [predictive_variance(Phi_star[i], Lambda_cond, sigma2) for i in range(10)]
        )
        np.testing.assert_allclose(batch_result, loop_result, atol=1e-12)


# ---------------------------------------------------------------------------
# conditional_gaussian_interval (eq:cgi)
# ---------------------------------------------------------------------------


class TestConditionalGaussianInterval:
    def test_95_percent_width(self):
        """95% interval width = 2 * 1.96 * sqrt(var)."""
        eta = np.array([0.0])
        var = np.array([1.0])
        lo, hi = conditional_gaussian_interval(eta, var, alpha=0.05)
        expected_half = stats.norm.ppf(0.975)  # ≈ 1.96
        assert hi[0] - lo[0] == pytest.approx(2 * expected_half, rel=1e-6)

    def test_centered_on_eta(self):
        eta = np.array([5.0, -3.0])
        var = np.array([2.0, 0.5])
        lo, hi = conditional_gaussian_interval(eta, var)
        midpoints = (lo + hi) / 2
        np.testing.assert_allclose(midpoints, eta, atol=1e-12)

    def test_zero_variance(self):
        eta = np.array([1.0])
        var = np.array([0.0])
        lo, hi = conditional_gaussian_interval(eta, var)
        assert lo[0] == pytest.approx(1.0)
        assert hi[0] == pytest.approx(1.0)

    def test_wider_at_higher_confidence(self):
        eta = np.array([0.0])
        var = np.array([1.0])
        _, hi_90 = conditional_gaussian_interval(eta, var, alpha=0.10)
        _, hi_95 = conditional_gaussian_interval(eta, var, alpha=0.05)
        _, hi_99 = conditional_gaussian_interval(eta, var, alpha=0.01)
        assert hi_90[0] < hi_95[0] < hi_99[0]


# ---------------------------------------------------------------------------
# logistic_normal_interval (eq:lnui)
# ---------------------------------------------------------------------------


class TestLogisticNormalInterval:
    def test_bounds_in_01(self):
        """Sigmoid pushes everything to [0, 1]."""
        rng = np.random.default_rng(1)
        eta = rng.standard_normal(20) * 3
        var = np.abs(rng.standard_normal(20)) + 0.1
        lo, hi = logistic_normal_interval(eta, var)
        assert np.all(lo >= 0) and np.all(lo <= 1)
        assert np.all(hi >= 0) and np.all(hi <= 1)

    def test_lo_le_hi(self):
        """Lower bound ≤ upper bound everywhere."""
        rng = np.random.default_rng(2)
        eta = rng.standard_normal(50)
        var = np.abs(rng.standard_normal(50)) + 0.01
        lo, hi = logistic_normal_interval(eta, var)
        assert np.all(lo <= hi + 1e-12)

    def test_zero_variance_collapses_to_point(self):
        eta = np.array([0.0, 2.0, -2.0])
        var = np.zeros(3)
        lo, hi = logistic_normal_interval(eta, var)
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        expected = sigmoid(eta)
        np.testing.assert_allclose(lo, expected, atol=1e-12)
        np.testing.assert_allclose(hi, expected, atol=1e-12)

    def test_symmetric_around_logit_zero(self):
        """At eta=0, the interval should be symmetric around 0.5."""
        var = np.array([1.0])
        lo, hi = logistic_normal_interval(np.array([0.0]), var)
        assert lo[0] == pytest.approx(1 - hi[0], abs=1e-12)

    def test_asymmetric_near_boundary(self):
        """At eta=5 (prob≈0.993), interval width on probability scale
        should be much narrower than at eta=0 (prob=0.5)."""
        var = np.array([1.0])
        _, hi_center = logistic_normal_interval(np.array([0.0]), var)
        lo_center, _ = logistic_normal_interval(np.array([0.0]), var)
        width_center = hi_center[0] - lo_center[0]

        lo_edge, hi_edge = logistic_normal_interval(np.array([5.0]), var)
        width_edge = hi_edge[0] - lo_edge[0]
        assert width_edge < width_center


# ---------------------------------------------------------------------------
# prediction_interval (unified dispatcher)
# ---------------------------------------------------------------------------


class TestPredictionInterval:
    def test_regression_dispatches(self):
        eta = np.array([1.0, 2.0])
        var = np.array([0.5, 0.5])
        lo, hi = prediction_interval(eta, var, task="regression")
        lo2, hi2 = conditional_gaussian_interval(eta, var)
        np.testing.assert_array_equal(lo, lo2)
        np.testing.assert_array_equal(hi, hi2)

    def test_binary_dispatches(self):
        eta = np.array([1.0, 2.0])
        var = np.array([0.5, 0.5])
        lo, hi = prediction_interval(eta, var, task="binary")
        lo2, hi2 = logistic_normal_interval(eta, var)
        np.testing.assert_array_equal(lo, lo2)
        np.testing.assert_array_equal(hi, hi2)

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="regression.*binary"):
            prediction_interval(np.array([0.0]), np.array([1.0]), task="multiclass")

    def test_q_hat_overrides_gaussian(self):
        """When q_hat is provided, alpha is ignored and q_hat is used."""
        eta = np.array([0.0])
        var = np.array([1.0])
        q_hat = 3.0  # much wider than 1.96
        lo, hi = prediction_interval(eta, var, q_hat=q_hat)
        assert hi[0] == pytest.approx(3.0)
        assert lo[0] == pytest.approx(-3.0)

    def test_q_hat_none_falls_back_to_gaussian(self):
        eta = np.array([0.0])
        var = np.array([1.0])
        lo, hi = prediction_interval(eta, var, alpha=0.05, q_hat=None)
        z = stats.norm.ppf(0.975)
        assert hi[0] == pytest.approx(z)


# ---------------------------------------------------------------------------
# calibrate_q (eq:calibrated-q)
# ---------------------------------------------------------------------------


class TestCalibrateQ:
    def test_perfect_model_q_is_small(self):
        """If predictions are exact, conformity scores are 0 → q ≈ 0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        eta = y.copy()
        var = np.ones(5)
        q = calibrate_q(y, eta, var, alpha=0.05)
        assert q == pytest.approx(0.0, abs=1e-10)

    def test_gaussian_data_q_near_z(self):
        """For data from the working model, q_hat ≈ z_{α/2}."""
        rng = np.random.default_rng(42)
        n = 10000
        eta = np.zeros(n)
        var = np.ones(n)
        y = rng.standard_normal(n)  # Y ~ N(eta, var) = N(0, 1)
        q = calibrate_q(y, eta, var, alpha=0.05)
        z = stats.norm.ppf(0.975)  # ≈ 1.96
        assert q == pytest.approx(z, rel=0.05)

    def test_heavy_tailed_q_larger_than_z(self):
        """For heavy-tailed data, q_hat should exceed z_{α/2}."""
        rng = np.random.default_rng(0)
        n = 5000
        eta = np.zeros(n)
        var = np.ones(n)
        y = rng.standard_t(df=3, size=n)  # t(3) is heavy-tailed
        q = calibrate_q(y, eta, var, alpha=0.05)
        z = stats.norm.ppf(0.975)
        assert q > z, f"expected q > z={z:.3f}, got q={q:.3f}"

    def test_returns_scalar(self):
        y = np.array([1.0, 2.0, 3.0])
        q = calibrate_q(y, y, np.ones(3))
        assert isinstance(q, float)


# ---------------------------------------------------------------------------
# calibrated_interval (eq:cgi with q_hat)
# ---------------------------------------------------------------------------


class TestCalibratedInterval:
    def test_regression_matches_manual(self):
        eta = np.array([5.0])
        var = np.array([4.0])
        q = 2.5
        lo, hi = calibrated_interval(eta, var, q, task="regression")
        assert lo[0] == pytest.approx(5.0 - 2.5 * 2.0)
        assert hi[0] == pytest.approx(5.0 + 2.5 * 2.0)

    def test_binary_bounds_in_01(self):
        rng = np.random.default_rng(1)
        eta = rng.standard_normal(20)
        var = np.abs(rng.standard_normal(20)) + 0.1
        lo, hi = calibrated_interval(eta, var, q_hat=2.0, task="binary")
        assert np.all(lo >= 0) and np.all(lo <= 1)
        assert np.all(hi >= 0) and np.all(hi <= 1)
        assert np.all(lo <= hi + 1e-12)

    def test_wider_q_gives_wider_interval(self):
        eta = np.array([0.0])
        var = np.array([1.0])
        _, hi_narrow = calibrated_interval(eta, var, q_hat=1.0)
        _, hi_wide = calibrated_interval(eta, var, q_hat=3.0)
        assert hi_narrow[0] < hi_wide[0]

    def test_gaussian_q_matches_uncalibrated(self):
        """calibrated_interval with q=z_{α/2} == conditional_gaussian_interval."""
        eta = np.array([1.0, -2.0, 3.0])
        var = np.array([0.5, 1.0, 2.0])
        z = stats.norm.ppf(0.975)
        lo_cal, hi_cal = calibrated_interval(eta, var, q_hat=z, task="regression")
        lo_gauss, hi_gauss = conditional_gaussian_interval(eta, var, alpha=0.05)
        np.testing.assert_allclose(lo_cal, lo_gauss, atol=1e-12)
        np.testing.assert_allclose(hi_cal, hi_gauss, atol=1e-12)

    def test_end_to_end_coverage(self):
        """calibrate_q → calibrated_interval should hit ~95% coverage."""
        rng = np.random.default_rng(99)
        n_cal, n_test = 2000, 5000
        # Generate from a slightly misspecified model (t-distributed noise)
        eta_true = np.zeros(n_cal + n_test)
        var_true = np.ones(n_cal + n_test)
        y = rng.standard_t(df=5, size=n_cal + n_test)

        y_cal, y_test = y[:n_cal], y[n_cal:]
        eta_cal, eta_test = eta_true[:n_cal], eta_true[n_cal:]
        var_cal, var_test = var_true[:n_cal], var_true[n_cal:]

        q = calibrate_q(y_cal, eta_cal, var_cal, alpha=0.05)
        lo, hi = calibrated_interval(eta_test, var_test, q, task="regression")

        coverage = np.mean((y_test >= lo) & (y_test <= hi))
        assert coverage == pytest.approx(
            0.95, abs=0.025
        ), f"coverage = {coverage:.4f}, expected ~0.95"
