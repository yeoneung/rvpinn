"""Exact OU transition moments vs Monte Carlo and vs theory."""
import numpy as np

from src.dynamics import correlated_normals, ou_step_exact


def test_conditional_moments():
    kappa, sigma, dt = 1.3, 40.0, 0.25
    z0 = 55.0
    rng = np.random.default_rng(11)
    eps = rng.standard_normal(2_000_000)
    z1 = ou_step_exact(np.full_like(eps, z0), kappa, sigma, dt, eps)
    mean_th = np.exp(-kappa * dt) * z0
    var_th = sigma ** 2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    assert np.isclose(z1.mean(), mean_th, rtol=1e-3)
    assert np.isclose(z1.var(), var_th, rtol=5e-3)


def test_stationary_variance_convergence():
    kappa, sigma = 2.0, 10.0
    rng = np.random.default_rng(3)
    z = np.zeros(500_000)
    for _ in range(200):
        z = ou_step_exact(z, kappa, sigma, 0.1, rng.standard_normal(z.shape))
    assert np.isclose(z.var(), sigma ** 2 / (2 * kappa), rtol=1e-2)


def test_innovation_correlation():
    rng = np.random.default_rng(9)
    e1, e2 = correlated_normals(rng, 0.6, 1_000_000)
    assert np.isclose(np.corrcoef(e1, e2)[0, 1], 0.6, atol=5e-3)
    assert np.isclose(e2.std(), 1.0, atol=5e-3)
