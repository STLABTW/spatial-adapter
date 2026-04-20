"""
Regression tests for the pybind11 C++ extension ``spatial_utils``.

Covers ``estimate_covariance`` (Wang 2017 rank selection + noise-subtracted
plug-in covariance) and ``fixed_rank_kriging`` (BLUP with explicit
conditional variance).  The test module is skipped at collection time if
the compiled ``spatial_utils.so`` cannot be imported, so these tests only
run in environments where ``make build-cpp`` has succeeded.
"""


import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module if the C++ extension is not available in the env.
# ---------------------------------------------------------------------------
spatial_utils = pytest.importorskip(
    "spatial_adapter.cpp_extensions",
    reason="compiled C++ extension not available; run `make build-cpp`",
)

estimate_covariance = spatial_utils.estimate_covariance
fixed_rank_kriging = spatial_utils.fixed_rank_kriging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _orthonormal_basis(p: int, K: int, *, seed: int) -> np.ndarray:
    """Return a ``(p, K)`` column-orthonormal matrix drawn from Haar-ish prior."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, K))
    Q, _ = np.linalg.qr(A)
    return np.ascontiguousarray(Q[:, :K], dtype=np.float64)


def _synthetic_low_rank_data(
    phi: np.ndarray,
    lambdas: np.ndarray,
    sigma: float,
    n: int,
    *,
    seed: int,
) -> np.ndarray:
    """
    Generate ``n`` samples of ``Y_i = alpha_i phi^T + noise_i`` where
    ``alpha_i ~ N(0, diag(lambdas))`` and ``noise_i ~ N(0, sigma^2 I_p)``.

    Returns a ``(n, p)`` float64 matrix that estimate_covariance expects.
    """
    rng = np.random.default_rng(seed)
    K = phi.shape[1]
    p = phi.shape[0]
    alphas = rng.standard_normal((n, K)) * np.sqrt(lambdas)
    noise = rng.standard_normal((n, p)) * sigma
    Y = alphas @ phi.T + noise
    return np.ascontiguousarray(Y, dtype=np.float64)


def _orthogonal_complement_data(
    phi: np.ndarray,
    sigma: float,
    n: int,
    *,
    seed: int,
) -> np.ndarray:
    """
    Generate ``n`` samples of ``Y_i`` whose row vectors are deterministically
    orthogonal to every column of ``phi``.  Concretely, draw iid Gaussian
    noise ``N(0, sigma^2 I_p)`` and project it onto ``phi``'s orthogonal
    complement via ``I - phi phi^T``.  This guarantees
    ``phi^T Y^T Y phi == 0`` exactly, so ``estimate_covariance`` sees zero
    projected eigenvalues and must take the Appendix B empty-branch path.
    """
    rng = np.random.default_rng(seed)
    p = phi.shape[0]
    noise = rng.standard_normal((n, p)) * sigma
    # Remove the component of each row that lies in the column span of phi.
    Y = noise - (noise @ phi) @ phi.T
    return np.ascontiguousarray(Y, dtype=np.float64)


# ---------------------------------------------------------------------------
# estimate_covariance
# ---------------------------------------------------------------------------
class TestEstimateCovariance:
    """Smoke + correctness tests for ``estimate_covariance``."""

    def test_smoke_return_shape(self):
        """The returned dict must expose the documented keys."""
        phi = _orthonormal_basis(p=20, K=3, seed=0)
        Y = _synthetic_low_rank_data(
            phi, lambdas=np.array([4.0, 2.0, 1.0]), sigma=0.3, n=200, seed=1
        )
        result = estimate_covariance(phi, Y)

        expected_keys = {
            "eigenvalues",
            "V",
            "noise_var",
            "estimated_covariance",
            "effective_rank",
        }
        assert (
            set(result.keys()) == expected_keys
        ), f"estimate_covariance returned unexpected keys: {set(result.keys())}"
        assert result["eigenvalues"].shape == (3,)
        assert result["V"].shape == (3, 3)
        assert result["estimated_covariance"].shape == (20, 20)
        assert isinstance(result["noise_var"], float)
        assert isinstance(result["effective_rank"], int)

    def test_strong_signal_recovers_full_rank(self):
        """When all lambdas dominate the noise floor, effective_rank == K."""
        phi = _orthonormal_basis(p=30, K=4, seed=2)
        lambdas_true = np.array([6.0, 4.0, 2.5, 1.5])
        sigma_true = 0.2
        Y = _synthetic_low_rank_data(
            phi, lambdas=lambdas_true, sigma=sigma_true, n=500, seed=3
        )
        result = estimate_covariance(phi, Y)

        assert (
            result["effective_rank"] == 4
        ), f"expected full rank 4, got {result['effective_rank']}"
        # Noise variance should be in the right ballpark (finite-sample noisy).
        assert result["noise_var"] == pytest.approx(sigma_true**2, rel=0.5)
        # Non-zero eigenvalues should be noise-subtracted estimates of lambdas_true.
        recovered = np.sort(result["eigenvalues"])[::-1]
        assert np.all(recovered > 0)
        np.testing.assert_allclose(recovered, lambdas_true, rtol=0.35)

    def test_empty_branch_isotropic_fallback(self):
        """
        Empty-branch path from Appendix B: when the projected covariance
        ``phi^T cov phi`` is exactly zero, Wang's rule yields an empty
        defining set and the estimator falls back to
            L_hat = 0,
            sigma^2 = tr(S) / p,
            est_cov = sigma^2 * I.
        We force this path deterministically by constructing ``Y`` whose
        rows live entirely in the orthogonal complement of ``phi``; no
        finite-sample luck is needed.  Note: because ``estimate_covariance``
        does not receive the smoothness penalty ``tau`` (applied upstream
        in the Python basis step), the empty branch can only fire when
        the projected eigenvalues are exactly zero.  This test exercises
        that path as a regression guard for the Appendix B fix.
        """
        phi = _orthonormal_basis(p=25, K=5, seed=4)
        sigma_true = 1.0
        Y = _orthogonal_complement_data(phi, sigma=sigma_true, n=400, seed=5)
        result = estimate_covariance(phi, Y)

        # effective_rank must be zero — the defining set is empty by
        # construction.
        assert (
            result["effective_rank"] == 0
        ), f"expected empty branch (rank 0), got {result['effective_rank']}"
        # All eigenvalues must be exactly zero in the empty branch.
        assert np.all(
            result["eigenvalues"] == 0.0
        ), "empty branch should zero out every entry of lambda_adj"
        # est_cov should be strictly isotropic: sigma_hat * I.  Because Y
        # lives in a (p-K)-dimensional subspace of R^p, sigma_hat =
        # tr(S)/p ~ (p-K)/p * sigma^2 in expectation, not sigma^2.
        est_cov = result["estimated_covariance"]
        off_diag = est_cov - np.diag(np.diag(est_cov))
        assert (
            np.max(np.abs(off_diag)) < 1e-10
        ), "empty branch must produce an exactly diagonal covariance"
        diag = np.diag(est_cov)
        np.testing.assert_allclose(diag, result["noise_var"], rtol=1e-10)
        # Sanity: noise_var is in the right ballpark for Y restricted to
        # the (p-K)-dimensional complement of phi.
        p = phi.shape[0]
        K = phi.shape[1]
        expected_noise_var = (p - K) / p * sigma_true**2
        assert result["noise_var"] == pytest.approx(expected_noise_var, rel=0.25)

    def test_mixed_signal_shrinks_weak_components(self):
        """
        Under mixed-strength signal, the paper's covariance estimator
        noise-subtracts the empirical eigenvalues so that the shrunken
        lambda_adj values are close to the true lambdas.  Components with
        true lambda << noise_var should end up with lambda_adj that is
        orders of magnitude smaller than the strong components, even if
        they are not pruned to exactly zero (Wang's rule without a
        smoothness penalty tends to accept every positive signal when
        p >> K, so we do not assert on ``effective_rank`` here).
        """
        phi = _orthonormal_basis(p=40, K=4, seed=6)
        lambdas_true = np.array([10.0, 5.0, 0.01, 0.005])
        sigma_true = 0.5
        Y = _synthetic_low_rank_data(
            phi, lambdas=lambdas_true, sigma=sigma_true, n=800, seed=7
        )
        result = estimate_covariance(phi, Y)

        eigs = result["eigenvalues"]
        # Eigenvalues must be in non-increasing order.
        assert np.all(np.diff(eigs) <= 1e-8), "eigenvalues must be sorted descending"

        # Strong components: recovered lambdas well above 1.
        assert eigs[0] > 5.0, f"expected strong first component, got {eigs[0]}"
        assert eigs[1] > 2.5, f"expected strong second component, got {eigs[1]}"

        # Weak components: shrunken to at most a small fraction of the
        # strongest component, even if Wang's rule didn't prune them to
        # exactly zero.  Factor of 50 gap is loose enough to survive
        # finite-sample fluctuation yet tight enough to be meaningful.
        assert eigs[0] / max(eigs[2], 1e-12) > 50.0, (
            f"weak component 3 not sufficiently shrunk: "
            f"eigs[0]={eigs[0]}, eigs[2]={eigs[2]}"
        )
        assert eigs[0] / max(eigs[3], 1e-12) > 50.0, (
            f"weak component 4 not sufficiently shrunk: "
            f"eigs[0]={eigs[0]}, eigs[3]={eigs[3]}"
        )
        # Eigenvalues past the reported effective_rank must be zero.
        L_hat = result["effective_rank"]
        assert np.all(
            eigs[L_hat:] == 0.0
        ), "eigenvalues beyond effective_rank must be zero-padded"

    def test_estimated_covariance_identity(self):
        """
        The returned estimated_covariance must satisfy
            est_cov = (phi V) diag(eigenvalues) (phi V)^T + noise_var * I.
        This is an algebraic self-consistency check that catches any future
        drift in the reconstruction formula.
        """
        phi = _orthonormal_basis(p=15, K=3, seed=8)
        Y = _synthetic_low_rank_data(
            phi, lambdas=np.array([3.0, 1.5, 0.8]), sigma=0.25, n=300, seed=9
        )
        result = estimate_covariance(phi, Y)

        V = result["V"]
        lam = result["eigenvalues"]
        sigma2 = result["noise_var"]
        proj = phi @ V
        expected = proj @ np.diag(lam) @ proj.T + sigma2 * np.eye(phi.shape[0])

        np.testing.assert_allclose(
            result["estimated_covariance"], expected, rtol=1e-10, atol=1e-10
        )

    def test_output_psd(self):
        """The estimated covariance must be symmetric positive semidefinite."""
        phi = _orthonormal_basis(p=18, K=4, seed=10)
        Y = _synthetic_low_rank_data(
            phi, lambdas=np.array([5.0, 2.0, 1.0, 0.3]), sigma=0.4, n=400, seed=11
        )
        result = estimate_covariance(phi, Y)
        est_cov = result["estimated_covariance"]

        # Symmetry.
        np.testing.assert_allclose(est_cov, est_cov.T, atol=1e-10)
        # PSD: smallest eigenvalue >= 0 up to numerical floor.
        min_eig = float(np.linalg.eigvalsh(est_cov).min())
        assert (
            min_eig >= -1e-8
        ), f"estimated_covariance is not PSD: min eigenvalue = {min_eig}"


# ---------------------------------------------------------------------------
# fixed_rank_kriging
# ---------------------------------------------------------------------------
class TestFixedRankKriging:
    """Smoke + correctness tests for ``fixed_rank_kriging``."""

    def _fit_then_predict(self, *, seed: int = 42):
        """Shared fixture: fit est_cov on phi_train, predict on phi_pred."""
        p_train, p_pred, K, n = 24, 6, 3, 300
        phi_full = _orthonormal_basis(p=p_train + p_pred, K=K, seed=seed)
        phi_train = np.ascontiguousarray(phi_full[:p_train], dtype=np.float64)
        phi_pred = np.ascontiguousarray(phi_full[p_train:], dtype=np.float64)
        lambdas_true = np.array([4.0, 2.0, 1.0])
        sigma_true = 0.3
        Y_train = _synthetic_low_rank_data(
            phi_train, lambdas=lambdas_true, sigma=sigma_true, n=n, seed=seed + 1
        )
        fit = estimate_covariance(phi_train, Y_train)
        # NOTE: fixed_rank_kriging exposes a positional parameter named
        # ``lambda`` which is a Python reserved keyword, so we must call
        # it positionally (or via ``**dict``).  Order matches the pybind11
        # signature: phi_train, V, lambda, noise_var, Y_new, phi_pred.
        krig = fixed_rank_kriging(
            phi_train,
            fit["V"],
            fit["eigenvalues"],
            fit["noise_var"],
            Y_train,
            phi_pred,
        )
        return fit, krig, phi_train, phi_pred, Y_train

    def test_smoke_return_shape(self):
        """The dict must expose both spatial_predictions and predictive_variance."""
        fit, krig, phi_train, phi_pred, Y_train = self._fit_then_predict(seed=100)

        assert {"spatial_predictions", "predictive_variance"} <= set(
            krig.keys()
        ), f"fixed_rank_kriging missing expected keys, got {set(krig.keys())}"
        assert krig["spatial_predictions"].shape == (
            Y_train.shape[0],
            phi_pred.shape[0],
        )
        assert krig["predictive_variance"].shape == (phi_pred.shape[0],)

    def test_predictive_variance_nonnegative(self):
        """Predictive variance must be entrywise non-negative."""
        _, krig, _, _, _ = self._fit_then_predict(seed=101)
        assert np.all(krig["predictive_variance"] >= 0)

    def test_predictive_variance_matches_python_reference(self):
        """
        Cross-check the C++ kriging variance against a pure-NumPy implementation
        of the same Gaussian conditioning formulas.
        """
        fit, krig, phi_train, phi_pred, _ = self._fit_then_predict(seed=102)
        V = fit["V"]
        lam = fit["eigenvalues"]
        sigma2 = fit["noise_var"]

        B_obs = phi_train @ V
        B_pred = phi_pred @ V
        Lambda = np.diag(lam)

        Sigma_oo = B_obs @ Lambda @ B_obs.T + sigma2 * np.eye(phi_train.shape[0])
        Sigma_op = B_obs @ Lambda @ B_pred.T
        Sigma_pp = B_pred @ Lambda @ B_pred.T + sigma2 * np.eye(phi_pred.shape[0])
        K_op = np.linalg.solve(Sigma_oo, Sigma_op)
        cond_cov = Sigma_pp - Sigma_op.T @ K_op
        expected_var = np.clip(np.diag(cond_cov), 0.0, np.inf)

        np.testing.assert_allclose(
            krig["predictive_variance"], expected_var, rtol=1e-8, atol=1e-10
        )

    def test_spatial_predictions_match_python_reference(self):
        """
        Cross-check spatial_predictions against the NumPy BLUP mean
            spatial_pred = R_train @ Sigma_oo^{-1} @ Sigma_op.
        """
        fit, krig, phi_train, phi_pred, Y_train = self._fit_then_predict(seed=103)
        V = fit["V"]
        lam = fit["eigenvalues"]
        sigma2 = fit["noise_var"]

        B_obs = phi_train @ V
        B_pred = phi_pred @ V
        Lambda = np.diag(lam)

        Sigma_oo = B_obs @ Lambda @ B_obs.T + sigma2 * np.eye(phi_train.shape[0])
        Sigma_op = B_obs @ Lambda @ B_pred.T
        K_op = np.linalg.solve(Sigma_oo, Sigma_op)
        expected_pred = Y_train @ K_op

        np.testing.assert_allclose(
            krig["spatial_predictions"], expected_pred, rtol=1e-8, atol=1e-10
        )
