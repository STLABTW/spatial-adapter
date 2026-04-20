"""Synthetic data generators for regression benchmarks."""

from typing import Optional, Tuple

import numpy as np


def _generate_spatial_basis(locations: np.ndarray) -> np.ndarray:
    """Normalized Gaussian-kernel basis: phi(s) = exp(-s^2) / ||...||."""
    phi = np.exp(-(locations**2))[:, None]
    phi /= np.linalg.norm(phi)
    return phi


def generate_combined_synthetic_data(
    location: np.ndarray,
    n_samples: int,
    noise_std: float,
    eigenvalue: float,
    global_mean: float = 50.0,
    feature_noise_std: float = 0.0,
    non_linear_strength: float = 0.0,
    correlation_strength: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate i.i.d. spatial data with trend + low-rank + noise.

    Returns (cat, cont, targets) with shapes
    (T, N, 0), (T, N, 3), (T, N).
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(location)
    p = 3

    cont_features = np.random.randn(n_samples, N, p)

    spatial_pattern = np.sin(location * np.pi / 5)
    temporal_trend = np.linspace(0, 2 * np.pi, n_samples)

    cont_features[:, :, 0] += (
        np.outer(temporal_trend, spatial_pattern) * 0.5 * correlation_strength
    )
    cont_features[:, :, 1] += (
        np.outer(temporal_trend, np.ones(N)) * 0.3 * correlation_strength
    )
    cont_features[:, :, 2] += (
        cont_features[:, :, 0] * cont_features[:, :, 1] * 0.2 * correlation_strength
    )

    if feature_noise_std > 0:
        cont_features += np.random.randn(*cont_features.shape) * feature_noise_std

    spatial_basis = _generate_spatial_basis(location)
    spatial_weights = np.random.randn(n_samples, 1) * np.sqrt(eigenvalue)

    targets = np.zeros((n_samples, N))
    for t in range(n_samples):
        trend = global_mean + np.random.randn(N) * 0.1
        spatial = spatial_weights[t] * spatial_basis.flatten()

        feature_component = (
            0.3 * cont_features[t, :, 0]
            + 0.4 * cont_features[t, :, 1]
            + 0.2 * cont_features[t, :, 2]
        ) * correlation_strength

        if non_linear_strength > 0:
            feature_component += (
                non_linear_strength * cont_features[t, :, 0] ** 2
                + non_linear_strength
                * 0.5
                * cont_features[t, :, 0]
                * cont_features[t, :, 1]
                + non_linear_strength * 0.3 * np.sin(cont_features[t, :, 2])
            )

        noise = np.random.randn(N) * noise_std
        targets[t] = trend + spatial + feature_component + noise

    cat_features = np.zeros((n_samples, N, 0), dtype=np.int64)
    return cat_features, cont_features, targets


def generate_time_synthetic_data(
    locs: np.ndarray,
    n_time_steps: int,
    noise_std: float,
    eigenvalue: float,
    eta_rho: float = 0.8,
    f_rho: float = 0.6,
    global_mean: float = 50.0,
    feature_noise_std: float = 0.01,
    include_gamma: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the 1D synthetic benchmark of NeurIPS §4.1.

    The five time-varying covariates (temp, wind, hum, pres, poll) are
    defined on :math:`t\\in[0, 2\\pi]` and replicated to every location
    with i.i.d. :math:`\\mathcal N(0, \\sigma_x^2)` perturbations, where
    :math:`\\sigma_x` = ``feature_noise_std``.  Two AR(1) drifts
    (:math:`\\gamma_t` with stationary var 1; :math:`\\alpha_t` with
    stationary var ``eigenvalue``) and a rank-1 spatial basis
    :math:`\\phi(s)=e^{-s^2}/\\|e^{-s^2}\\|_2` combine with i.i.d.\\
    observation noise :math:`\\mathcal N(0, \\text{noise\\_std}^2)` to
    form

        Y(s_i, t) = global_mean + x_{i,t}^T β + γ_t + α_t φ(s_i) + ε_{i,t},

    with β = (10, −9.5, 2.6, −1.3, 1.2).  Returns (cat, cont, y) with
    shapes (T, N, 0), (T, N, 5), (T, N).
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(locs)
    T = n_time_steps

    # (i) Base time-varying covariates on t ∈ [0, 2π]
    t_grid = np.linspace(0.0, 2.0 * np.pi, T)
    eps_small = 0.1  # paper: all base-covariate measurement stds

    temp = 15.0 + 10.0 * np.sin(t_grid) + np.random.randn(T) * eps_small
    wind = np.abs(3.0 + np.random.randn(T) * eps_small)
    hum = 70.0 - 0.5 * temp + np.random.randn(T) * eps_small
    # Gaussian-pdf pressure bump centered at π with unit variance
    pres = 1000.0 + 20.0 * np.exp(-0.5 * (t_grid - np.pi) ** 2) / np.sqrt(2.0 * np.pi)
    poll = 50.0 + 2.0 * temp - 3.0 * wind + np.random.randn(T) * eps_small

    base = np.stack([temp, wind, hum, pres, poll], axis=1)  # (T, 5)
    # Replicate to every location with i.i.d. small perturbations
    cont = base[:, None, :] + np.random.randn(T, N, 5) * feature_noise_std

    # (ii) AR(1) spatial score α_t (stationary variance = ``eigenvalue``).
    # γ_t is a global drift the paper marks as *optional*; we disable it
    # by default because its rank-1 contribution N·var(γ) would dominate
    # the residual sample covariance and prevent the basis learner from
    # recovering φ(s).  Enable via ``include_gamma=True`` to reproduce
    # the full model.
    sigma_alpha = np.sqrt(max(eigenvalue * (1.0 - eta_rho**2), 0.0))
    alpha = np.zeros(T)
    alpha[0] = np.random.randn() * np.sqrt(eigenvalue)
    for t in range(1, T):
        alpha[t] = eta_rho * alpha[t - 1] + np.random.randn() * sigma_alpha

    if include_gamma:
        sigma_gamma = np.sqrt(max(1.0 - f_rho**2, 0.0))
        gamma = np.zeros(T)
        gamma[0] = np.random.randn()
        for t in range(1, T):
            gamma[t] = f_rho * gamma[t - 1] + np.random.randn() * sigma_gamma
    else:
        gamma = np.zeros(T)

    # (iii) Observation model
    phi = _generate_spatial_basis(locs).flatten()  # (N,) unit-norm
    beta = np.array([10.0, -9.5, 2.6, -1.3, 1.2])  # (p,)

    feat_term = cont @ beta  # (T, N)
    spatial_term = alpha[:, None] * phi[None, :]  # (T, N)
    noise = np.random.randn(T, N) * noise_std

    y = global_mean + feat_term + gamma[:, None] + spatial_term + noise

    cat = np.zeros((T, N, 0), dtype=np.int64)
    return cat, cont, y
