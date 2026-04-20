"""
Prediction intervals and conditional inference for the Spatial Adapter.

Implements the plug-in conditional Gaussian predictive law
(Proposition ``prop:gaussian-predictive`` in the paper) and the
two interval types derived from it:

  * ``conditional_gaussian_interval`` — regression (eq:cgi)
  * ``logistic_normal_interval``      — binary classification (eq:lnui)

Also provides the underlying building blocks (conditional covariance,
conditional score, predictive variance) as pure-NumPy functions that
parallel the C++ ``fixed_rank_kriging`` for verification and for
lightweight use cases (e.g. the Wheat Head experiment where
:math:`\\mathcal O_j = \\emptyset` and kriging is trivial).
"""


from typing import Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Building blocks (eq:lambda-cond, eq:point-pred, eq:kriging-se)
# ---------------------------------------------------------------------------


def conditional_covariance(
    Lambda: np.ndarray,
    Phi_O: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    r"""
    Plug-in conditional covariance of the scores (eq:lambda-cond).

    .. math::

        \widehat\Lambda_{\mathrm{cond}}(\mathcal O_j)
        = \bigl(\widehat\Lambda^{-1}
          + \widehat\sigma^{-2}\,\Phi_{\mathcal O_j}^\top
            \Phi_{\mathcal O_j}\bigr)^{-1}

    When :math:`\mathcal O_j = \emptyset` (no residual observed),
    pass ``Phi_O`` as a ``(0, K)`` array and the result reduces to
    the marginal :math:`\widehat\Lambda`.

    Parameters
    ----------
    Lambda : (K, K) array
        Plug-in score covariance (diagonal or full).
    Phi_O : (|O_j|, K) array
        Basis rows at the observed locations. Shape ``(0, K)`` for the
        empty observation set.
    sigma2 : float
        Plug-in noise variance :math:`\widehat\sigma^2`.

    Returns
    -------
    Lambda_cond : (K, K) array
        Conditional covariance of the scores.
    """
    K = Lambda.shape[0]
    Lambda_inv = np.linalg.inv(Lambda + 1e-12 * np.eye(K))
    info = Lambda_inv + (Phi_O.T @ Phi_O) / sigma2
    return np.linalg.inv(info)


def conditional_score(
    Lambda_cond: np.ndarray,
    Phi_O: np.ndarray,
    r_O: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    r"""
    Plug-in conditional-mean score estimate (eq:point-pred).

    .. math::

        \widehat{\bm\alpha}_j(\mathcal O_j)
        = \widehat\sigma^{-2}\,
          \widehat\Lambda_{\mathrm{cond}}(\mathcal O_j)\,
          \Phi_{\mathcal O_j}^\top\,\mathbf r_{j,\mathcal O_j}

    With the convention :math:`\widehat{\bm\alpha}_j(\emptyset) = \mathbf 0`.

    Parameters
    ----------
    Lambda_cond : (K, K) array
        From :func:`conditional_covariance`.
    Phi_O : (|O_j|, K) array
        Basis rows at the observed locations.
    r_O : (|O_j|,) array
        Residuals at the observed locations for sample *j*.
    sigma2 : float
        Plug-in noise variance.

    Returns
    -------
    alpha_hat : (K,) array
        Conditional-mean score vector.
    """
    if Phi_O.shape[0] == 0:
        return np.zeros(Lambda_cond.shape[0])
    return (Lambda_cond @ Phi_O.T @ r_O) / sigma2


def predictive_variance(
    phi_star: np.ndarray,
    Lambda_cond: np.ndarray,
    sigma2: float,
) -> float:
    r"""
    Conditional predictive variance at a query location (eq:kriging-se).

    .. math::

        \hat v(\mathbf s^*,j)
        = \widehat\sigma^2
          + \widehat{\bm\phi}(\mathbf s^*)^\top\,
            \widehat\Lambda_{\mathrm{cond}}(\mathcal O_j)\,
            \widehat{\bm\phi}(\mathbf s^*)

    Parameters
    ----------
    phi_star : (K,) array
        Basis row at the query location :math:`\mathbf s^*`.
    Lambda_cond : (K, K) array
        From :func:`conditional_covariance`.
    sigma2 : float
        Plug-in noise variance.

    Returns
    -------
    v_hat : float
        Conditional predictive variance (always ≥ σ²).
    """
    return float(sigma2 + phi_star @ Lambda_cond @ phi_star)


def predictive_variance_batch(
    Phi_star: np.ndarray,
    Lambda_cond: np.ndarray,
    sigma2: float,
) -> np.ndarray:
    r"""
    Vectorised version of :func:`predictive_variance` for multiple
    query locations.

    Parameters
    ----------
    Phi_star : (M, K) array
        Basis rows at *M* query locations.
    Lambda_cond : (K, K) array
        From :func:`conditional_covariance`.
    sigma2 : float
        Plug-in noise variance.

    Returns
    -------
    v_hat : (M,) array
        Per-location conditional predictive variances.
    """
    # diag(Phi @ Lambda_cond @ Phi^T) via row-wise dot product
    return sigma2 + np.sum((Phi_star @ Lambda_cond) * Phi_star, axis=1)


# ---------------------------------------------------------------------------
# Prediction intervals (eq:cgi, eq:lnui)
# ---------------------------------------------------------------------------


def conditional_gaussian_interval(
    eta_hat: np.ndarray,
    pred_var: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Plug-in conditional Gaussian predictive interval on the
    natural-parameter scale (eq:cgi).

    .. math::

        \hat\eta(\mathbf s^*,j)
        \;\pm\; z_{\alpha/2}\,\sqrt{\hat v(\mathbf s^*,j)}

    Parameters
    ----------
    eta_hat : array
        Point prediction on the natural-parameter scale
        (trend + spatial correction).
    pred_var : array (same shape as *eta_hat*)
        Conditional predictive variance from
        :func:`predictive_variance_batch` or from the C++ extension's
        ``fixed_rank_kriging["predictive_variance"]``.
    alpha : float
        Significance level; default 0.05 → 95 % interval.

    Returns
    -------
    lower, upper : arrays (same shape as *eta_hat*)
        Lower and upper bounds of the interval.
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    se = np.sqrt(np.maximum(pred_var, 0.0))
    return eta_hat - z * se, eta_hat + z * se


def logistic_normal_interval(
    eta_hat: np.ndarray,
    pred_var: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Logistic-normal uncertainty interval for binary tasks (eq:lnui).

    Constructs the conditional Gaussian interval on the logit scale
    and pushes both endpoints through the sigmoid:

    .. math::

        \Bigl[\,\sigma\!\bigl(\hat\eta - z_{\alpha/2}\sqrt{\hat v}\bigr),\;
               \sigma\!\bigl(\hat\eta + z_{\alpha/2}\sqrt{\hat v}\bigr)\,\Bigr]

    The result quantifies uncertainty about the *predicted probability
    induced by the latent Gaussian uncertainty*, not a coverage
    statement for the Bernoulli outcome :math:`Y \in \{0, 1\}`.

    Parameters
    ----------
    eta_hat : array
        Point prediction on the logit scale.
    pred_var : array (same shape as *eta_hat*)
        Conditional predictive variance on the logit scale.
    alpha : float
        Significance level; default 0.05 → 95 % interval.

    Returns
    -------
    lower, upper : arrays (same shape as *eta_hat*)
        Lower and upper probability bounds in [0, 1].
    """
    lower_logit, upper_logit = conditional_gaussian_interval(eta_hat, pred_var, alpha)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    return sigmoid(lower_logit), sigmoid(upper_logit)


def calibrate_q(
    y_cal: np.ndarray,
    eta_hat_cal: np.ndarray,
    pred_var_cal: np.ndarray,
    alpha: float = 0.05,
) -> float:
    r"""
    Estimate the post-hoc calibrated critical value :math:`\hat q_\alpha`
    from a calibration set (eq:calibrated-q).

    Computes normalised conformity scores

    .. math::

        s_i = \frac{|Y_i^{\mathrm{cal}} - \hat\eta_i|}{\sqrt{\hat v_i}}

    and returns the :math:`\lceil(1-\alpha)(1+1/n)\rceil`-th order
    statistic.  When the calibration set is disjoint from training and
    validation data, using :math:`\hat q_\alpha` in place of
    :math:`z_{\alpha/2}` gives a split-conformal prediction interval
    with finite-sample marginal coverage :math:`\geq 1-\alpha`.

    When the calibration set overlaps with the validation data used
    for hyperparameter selection, the coverage guarantee does not hold
    and the result should be described as *post-hoc calibration*.

    Parameters
    ----------
    y_cal : array
        Ground-truth values on the calibration set.
    eta_hat_cal : array (same shape)
        Point predictions on the calibration set
        (natural-parameter / logit scale).
    pred_var_cal : array (same shape)
        Conditional predictive variances on the calibration set.
    alpha : float
        Target miscoverage rate (default 0.05 → 95 % interval).

    Returns
    -------
    q_hat : float
        Calibrated critical value to use in place of
        :math:`z_{\alpha/2}` in eq:cgi / eq:lnui.
    """
    y_cal = np.asarray(y_cal).ravel()
    eta_hat_cal = np.asarray(eta_hat_cal).ravel()
    pred_var_cal = np.asarray(pred_var_cal).ravel()

    se = np.sqrt(np.maximum(pred_var_cal, 0.0))
    # Avoid division by zero at locations with zero variance.
    se = np.where(se < 1e-12, 1e-12, se)
    scores = np.abs(y_cal - eta_hat_cal) / se

    n = len(scores)
    # Finite-sample conformal quantile level (Vovk et al. 2005).
    level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
    return float(np.quantile(scores, level))


def calibrated_interval(
    eta_hat: np.ndarray,
    pred_var: np.ndarray,
    q_hat: float,
    task: str = "regression",
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Prediction interval using a calibrated critical value
    :math:`\hat q_\alpha` (eq:cgi with post-hoc calibration).

    For regression:

    .. math::

        \hat\eta(\mathbf s^*,j)
        \;\pm\;\hat q_\alpha\,\sqrt{\hat v(\mathbf s^*,j)}

    For binary classification, the logit-scale interval is pushed
    through the sigmoid (eq:lnui):

    .. math::

        \bigl[\sigma(\hat\eta - \hat q_\alpha\sqrt{\hat v}),\;
               \sigma(\hat\eta + \hat q_\alpha\sqrt{\hat v})\bigr]

    Parameters
    ----------
    eta_hat : array
        Point prediction on the natural-parameter / logit scale.
    pred_var : array (same shape)
        Conditional predictive variance.
    q_hat : float
        Calibrated critical value from :func:`calibrate_q`.
        Set ``q_hat = scipy.stats.norm.ppf(1 - alpha/2)`` to recover
        the uncalibrated Gaussian plug-in interval.
    task : ``"regression"`` or ``"binary"``

    Returns
    -------
    lower, upper : arrays
        For regression: bounds on the natural-parameter scale.
        For binary: probability bounds in [0, 1].
    """
    se = np.sqrt(np.maximum(pred_var, 0.0))
    lower_eta = eta_hat - q_hat * se
    upper_eta = eta_hat + q_hat * se

    if task == "regression":
        return lower_eta, upper_eta
    elif task == "binary":
        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        return sigmoid(lower_eta), sigmoid(upper_eta)
    else:
        raise ValueError(f"task must be 'regression' or 'binary', got {task!r}")


def prediction_interval(
    eta_hat: np.ndarray,
    pred_var: np.ndarray,
    alpha: float = 0.05,
    task: str = "regression",
    q_hat: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified entry point for prediction intervals.

    If ``q_hat`` is provided, uses the calibrated critical value
    (post-hoc calibration via :func:`calibrate_q`).
    Otherwise falls back to the Gaussian quantile
    :math:`z_{\\alpha/2}` (uncalibrated plug-in interval).

    Parameters
    ----------
    eta_hat : array
        Point prediction on the natural-parameter / logit scale.
    pred_var : array
        Conditional predictive variance (same shape as *eta_hat*).
    alpha : float
        Significance level (default 0.05 → 95 % interval).
        Ignored when *q_hat* is provided.
    task : ``"regression"`` or ``"binary"``
        Determines which interval type to compute.
    q_hat : float, optional
        Calibrated critical value.  When ``None`` (default), the
        Gaussian quantile ``z_{alpha/2}`` is used.

    Returns
    -------
    lower, upper : arrays
        For regression: bounds on the natural-parameter scale.
        For binary: probability bounds in [0, 1].
    """
    if q_hat is None:
        q_hat = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return calibrated_interval(eta_hat, pred_var, q_hat, task=task)
