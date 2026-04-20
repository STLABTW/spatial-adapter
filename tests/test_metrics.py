"""
Unit tests for spatial_adapter.metrics module.
"""

import numpy as np
import pytest
import torch

from spatial_adapter.metrics import (
    compute_binary_metrics,
    compute_metrics,
    cov_frob_observed,
    coverage_probability,
    empirical_cov,
    expected_calibration_error,
    frobenius_norm,
    fusion_score,
    mae_pooled,
    mpiw,
    r2_pooled,
    rmse_pooled,
)


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_compute_metrics_perfect_prediction(self):
        """Test compute_metrics with perfect predictions."""
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert rmse == 0.0
        assert mae == 0.0
        assert r2 == 1.0

    def test_compute_metrics_with_error(self):
        """Test compute_metrics with prediction errors."""
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert rmse > 0.0
        assert mae > 0.0
        assert r2 < 1.0

    def test_compute_metrics_different_shapes(self):
        """Test compute_metrics with different tensor shapes."""
        y_true = torch.randn(10, 5)
        y_pred = torch.randn(10, 5)

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert isinstance(rmse, float)
        assert isinstance(mae, float)
        assert isinstance(r2, float)
        assert rmse >= 0.0
        assert mae >= 0.0


class TestFusionScore:
    """Test fusion_score function."""

    def test_fusion_score_rmse_only(self):
        """Test fusion_score when only RMSE is provided."""
        rmse = 0.5
        result = fusion_score(rmse, None, None)
        assert result == rmse

    def test_fusion_score_with_projection_gap(self):
        """Test fusion_score with projection gap."""
        rmse = 0.5
        proj_gap = 0.1
        p = 10
        expected = rmse + (proj_gap / p)
        result = fusion_score(rmse, proj_gap, p)
        assert result == expected

    def test_fusion_score_zero_p(self):
        """Test fusion_score when p is zero."""
        rmse = 0.5
        proj_gap = 0.1
        p = 0
        result = fusion_score(rmse, proj_gap, p)
        assert result == rmse


class TestFrobeniusNorm:
    """Test frobenius_norm function."""

    def test_frobenius_norm_identical_matrices(self):
        """Test frobenius_norm with identical matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4]])
        result = frobenius_norm(A, B)
        assert result == 0.0

    def test_frobenius_norm_different_matrices(self):
        """Test frobenius_norm with different matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 3], [4, 5]])
        result = frobenius_norm(A, B)
        assert result > 0.0

    def test_frobenius_norm_large_matrices(self):
        """Test frobenius_norm with larger matrices."""
        A = np.random.randn(10, 10)
        B = np.random.randn(10, 10)
        result = frobenius_norm(A, B)
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Pooled metrics (paper eq:pointwise_def, eq:covfrob_def)
# ---------------------------------------------------------------------------


class TestRmsePooled:
    """Test rmse_pooled — eq:pointwise_def RMSE."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        mask = np.ones(3, dtype=bool)
        assert rmse_pooled(y, y, mask) == 0.0

    def test_known_error(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        mask = np.ones(3, dtype=bool)
        assert rmse_pooled(y_true, y_pred, mask) == pytest.approx(1.0)

    def test_mask_excludes_entries(self):
        y_true = np.array([0.0, 0.0, 100.0])
        y_pred = np.array([1.0, 1.0, 0.0])
        mask = np.array([True, True, False])
        assert rmse_pooled(y_true, y_pred, mask) == pytest.approx(1.0)

    def test_nan_in_input(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mask = np.ones(3, dtype=bool)
        assert rmse_pooled(y_true, y_pred, mask) == pytest.approx(0.0)

    def test_all_masked_returns_nan(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        mask = np.array([False, False])
        assert np.isnan(rmse_pooled(y_true, y_pred, mask))

    def test_2d_input(self):
        """Paper uses (T, N) field matrices; RMSE should flatten correctly."""
        y_true = np.zeros((5, 10))
        y_pred = np.ones((5, 10))
        mask = np.ones((5, 10), dtype=bool)
        assert rmse_pooled(y_true, y_pred, mask) == pytest.approx(1.0)


class TestMaePooled:
    """Test mae_pooled — eq:pointwise_def MAE."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae_pooled(y, y) == 0.0

    def test_known_error(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        assert mae_pooled(y_true, y_pred) == pytest.approx(1.0)

    def test_mask(self):
        y_true = np.array([0.0, 0.0, 999.0])
        y_pred = np.array([1.0, 1.0, 0.0])
        mask = np.array([True, True, False])
        assert mae_pooled(y_true, y_pred, mask) == pytest.approx(1.0)

    def test_default_mask_skips_nan(self):
        y_true = np.array([0.0, np.nan])
        y_pred = np.array([1.0, 2.0])
        assert mae_pooled(y_true, y_pred) == pytest.approx(1.0)


class TestR2Pooled:
    """Test r2_pooled — eq:pointwise_def R²."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_pooled(y, y) == pytest.approx(1.0)

    def test_mean_prediction(self):
        """Predicting the mean should give R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, y_true.mean())
        assert r2_pooled(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)

    def test_worse_than_mean(self):
        """Bad prediction should give R² < 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert r2_pooled(y_true, y_pred) < 0.0

    def test_mask(self):
        y_true = np.array([1.0, 2.0, 3.0, 999.0])
        y_pred = np.array([1.0, 2.0, 3.0, 0.0])
        mask = np.array([True, True, True, False])
        assert r2_pooled(y_true, y_pred, mask) == pytest.approx(1.0)

    def test_zero_variance_returns_nan(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert np.isnan(r2_pooled(y_true, y_pred))

    def test_matches_sklearn(self):
        """Cross-check against sklearn's r2_score on the same data."""
        from sklearn.metrics import r2_score

        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = y_true + 0.1 * rng.standard_normal(100)
        expected = r2_score(y_true, y_pred)
        assert r2_pooled(y_true, y_pred) == pytest.approx(expected, rel=1e-10)


class TestEmpiricalCov:
    """Test empirical_cov — sample covariance for eq:covfrob_def."""

    def test_identity_covariance(self):
        """Large iid N(0,I) sample should recover ~I."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10000, 4))
        cov = empirical_cov(X)
        np.testing.assert_allclose(cov, np.eye(4), atol=0.05)

    def test_known_covariance(self):
        """Sample from known Sigma, recover it."""
        rng = np.random.default_rng(1)
        Sigma = np.array([[4.0, 1.0], [1.0, 2.0]])
        L = np.linalg.cholesky(Sigma)
        X = rng.standard_normal((50000, 2)) @ L.T
        cov = empirical_cov(X)
        np.testing.assert_allclose(cov, Sigma, atol=0.05)

    def test_symmetry(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 5))
        cov = empirical_cov(X)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)

    def test_psd(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((30, 6))
        cov = empirical_cov(X)
        eigvals = np.linalg.eigvalsh(cov)
        assert eigvals.min() >= -1e-10

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            empirical_cov(np.array([1.0, 2.0, 3.0]))

    def test_single_row_raises(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            empirical_cov(np.array([[1.0, 2.0]]))


class TestCovFrobObserved:
    """Test cov_frob_observed — eq:covfrob_def relative Frobenius error."""

    def test_identical_fields(self):
        """Same field → CovFrob = 0."""
        rng = np.random.default_rng(10)
        field = rng.standard_normal((50, 8))
        assert cov_frob_observed(field, field) == pytest.approx(0.0, abs=1e-10)

    def test_known_frob_error(self):
        """
        If pred = 2*true (same structure, doubled scale), empirical
        covariance is 4x, so CovFrob = ||4Σ - Σ||/||Σ|| = 3.
        """
        rng = np.random.default_rng(11)
        field_true = rng.standard_normal((10000, 3))
        field_pred = 2.0 * field_true
        result = cov_frob_observed(field_true, field_pred)
        assert result == pytest.approx(3.0, rel=0.02)

    def test_nonnegative(self):
        rng = np.random.default_rng(12)
        a = rng.standard_normal((40, 5))
        b = rng.standard_normal((40, 5))
        assert cov_frob_observed(a, b) >= 0.0

    def test_symmetric_in_error(self):
        """CovFrob(a, b) is NOT symmetric in a, b because the denominator
        uses ||Σ_obs||_F = ||Σ_a||_F.  This test documents that."""
        rng = np.random.default_rng(13)
        a = rng.standard_normal((100, 4))
        b = rng.standard_normal((100, 4)) * 3
        ab = cov_frob_observed(a, b)
        ba = cov_frob_observed(b, a)
        # Denominator differs → result differs.
        assert ab != pytest.approx(ba, rel=0.1)


# ---------------------------------------------------------------------------
# Prediction-interval calibration metrics (eq:cp-mpiw)
# ---------------------------------------------------------------------------


class TestCoverageProbability:
    """Test coverage_probability — eq:cp-mpiw CP."""

    def test_perfect_coverage(self):
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([2.0, 3.0, 4.0])
        assert coverage_probability(y, lo, hi) == pytest.approx(1.0)

    def test_zero_coverage(self):
        y = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage_probability(y, lo, hi) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([0.5, 5.0, 0.5, 5.0])
        lo = np.zeros(4)
        hi = np.ones(4)
        # y[0]=0.5 ∈ [0,1], y[1]=5 ∉, y[2]=0.5 ∈, y[3]=5 ∉
        assert coverage_probability(y, lo, hi) == pytest.approx(0.5)

    def test_mask(self):
        y = np.array([0.5, 100.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        mask = np.array([True, False])
        assert coverage_probability(y, lo, hi, mask) == pytest.approx(1.0)

    def test_boundary_included(self):
        """Boundary points (y == lo or y == hi) count as covered."""
        y = np.array([0.0, 1.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage_probability(y, lo, hi) == pytest.approx(1.0)

    def test_nan_default_mask(self):
        y = np.array([0.5, np.nan])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage_probability(y, lo, hi) == pytest.approx(1.0)

    def test_all_masked_returns_nan(self):
        y = np.array([1.0])
        lo = np.array([0.0])
        hi = np.array([2.0])
        mask = np.array([False])
        assert np.isnan(coverage_probability(y, lo, hi, mask))


class TestMPIW:
    """Test mpiw — eq:cp-mpiw MPIW."""

    def test_uniform_width(self):
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([1.0, 2.0, 3.0])
        assert mpiw(lo, hi) == pytest.approx(1.0)

    def test_varying_width(self):
        lo = np.array([0.0, 0.0])
        hi = np.array([2.0, 4.0])
        assert mpiw(lo, hi) == pytest.approx(3.0)

    def test_mask(self):
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([1.0, 1.0, 100.0])
        mask = np.array([True, True, False])
        assert mpiw(lo, hi, mask) == pytest.approx(1.0)

    def test_zero_width(self):
        x = np.array([1.0, 2.0, 3.0])
        assert mpiw(x, x) == pytest.approx(0.0)

    def test_2d_input(self):
        """Paper uses (T, N) shaped intervals."""
        lo = np.zeros((5, 10))
        hi = np.ones((5, 10)) * 2.0
        assert mpiw(lo, hi) == pytest.approx(2.0)

    def test_all_masked_returns_nan(self):
        lo = np.array([0.0])
        hi = np.array([1.0])
        mask = np.array([False])
        assert np.isnan(mpiw(lo, hi, mask))


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    """Test expected_calibration_error — ECE for binary classification."""

    def test_perfect_calibration(self):
        """Perfectly calibrated model → ECE = 0."""
        rng = np.random.default_rng(0)
        n = 10000
        # Generate probabilities and sample labels from them
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(float)
        ece = expected_calibration_error(y_true, y_prob, n_bins=20)
        assert ece == pytest.approx(0.0, abs=0.02)

    def test_constant_prediction(self):
        """Predict 0.5 for everything; 50% positive labels → ECE ≈ 0."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        y_prob = np.full(10, 0.5)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_maximally_miscalibrated(self):
        """Predict 1.0 for all-negative labels → ECE = 1.0."""
        y_true = np.zeros(100)
        y_prob = np.ones(100)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece == pytest.approx(1.0)

    def test_predict_zero_all_negative(self):
        """Predict 0.0 for all-negative labels → ECE = 0."""
        y_true = np.zeros(100)
        y_prob = np.zeros(100)
        ece = expected_calibration_error(y_true, y_prob)
        assert ece == pytest.approx(0.0)

    def test_ece_in_unit_interval(self):
        """ECE should always be in [0, 1]."""
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_prob = rng.uniform(0, 1, 200)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_mask(self):
        """Masked entries should be excluded."""
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        # Without mask: perfect calibration (all 0s predict 0, all 1s predict 1)
        assert expected_calibration_error(y_true, y_prob) == pytest.approx(
            0.0, abs=1e-10
        )
        # Mask out the positive labels: all negative, predict 0 → still 0
        mask = np.array([True, True, True, False, False, False])
        ece_masked = expected_calibration_error(y_true, y_prob, mask=mask)
        assert ece_masked == pytest.approx(0.0, abs=1e-10)

    def test_nan_handling(self):
        """NaN in input should be excluded automatically."""
        y_true = np.array([0, 1, np.nan, 0, 1], dtype=float)
        y_prob = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
        ece = expected_calibration_error(y_true, y_prob)
        assert not np.isnan(ece)

    def test_empty_returns_nan(self):
        y_true = np.array([], dtype=float)
        y_prob = np.array([], dtype=float)
        assert np.isnan(expected_calibration_error(y_true, y_prob))

    def test_all_masked_returns_nan(self):
        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.3, 0.7])
        mask = np.array([False, False])
        assert np.isnan(expected_calibration_error(y_true, y_prob, mask=mask))

    def test_more_bins_than_samples(self):
        """Should not crash when n_bins > n_samples."""
        y_true = np.array([0, 1], dtype=float)
        y_prob = np.array([0.2, 0.8])
        ece = expected_calibration_error(y_true, y_prob, n_bins=100)
        assert 0.0 <= ece <= 1.0

    def test_sklearn_crosscheck(self):
        """Cross-check against a manual bin computation."""
        # 4 samples, 2 bins: [0, 0.5] and (0.5, 1.0]
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_prob = np.array([0.2, 0.4, 0.7, 0.9])
        # Bin 1 [0, 0.5]: samples {0.2, 0.4}, avg_prob=0.3, avg_true=0.0
        #   contrib = 2/4 * |0.3 - 0.0| = 0.15
        # Bin 2 (0.5, 1.0]: samples {0.7, 0.9}, avg_prob=0.8, avg_true=1.0
        #   contrib = 2/4 * |0.8 - 1.0| = 0.10
        # ECE = 0.15 + 0.10 = 0.25
        ece = expected_calibration_error(y_true, y_prob, n_bins=2)
        assert ece == pytest.approx(0.25)


class TestEdgeCaseFallbacks:
    """Cover single-class AUC fallback and empty-mask branches."""

    def test_compute_binary_metrics_auc_fallback_on_value_error(self, monkeypatch):
        """When roc_auc_score raises ValueError (e.g. single-class y_true on older
        sklearn), compute_binary_metrics falls back to auc=0.5."""
        from spatial_adapter import metrics as metrics_mod

        def _raise(*a, **kw):
            raise ValueError("single-class")

        monkeypatch.setattr(metrics_mod, "roc_auc_score", _raise)

        y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
        y_logits = torch.randn(4)
        _, _, auc = compute_binary_metrics(y_true, y_logits)
        assert auc == 0.5

    def test_mae_pooled_empty_mask_returns_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mask = np.zeros_like(y_true, dtype=bool)  # empty mask
        assert np.isnan(mae_pooled(y_true, y_pred, mask))

    def test_r2_pooled_empty_mask_returns_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        mask = np.zeros_like(y_true, dtype=bool)
        assert np.isnan(r2_pooled(y_true, y_pred, mask))
