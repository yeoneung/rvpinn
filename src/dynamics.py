"""Battery and exogenous-state dynamics.

Sign convention (main.tex eq. 3): A > 0 discharges the battery, A < 0
charges it. All functions are written with operations valid for both numpy
float64 arrays and torch float64 tensors: max(a,0) = (a+|a|)/2 and
min(x,y) = (x+y-|x-y|)/2.
"""
from __future__ import annotations

import numpy as np

from .config import ModelParams


def pos(a):
    """max(a, 0), numpy/torch generic."""
    return 0.5 * (a + abs(a))


def neg(a):
    """max(-a, 0), numpy/torch generic."""
    return 0.5 * (abs(a) - a)


def gmin(x, y):
    return 0.5 * (x + y - abs(x - y))


def gmax(x, y):
    return 0.5 * (x + y + abs(x - y))


def b_S(a, p: ModelParams):
    """SoC drift (1/h): eta_c*a^-/E_max - a^+/(eta_d*E_max)."""
    return p.eta_c * neg(a) / p.E_max - pos(a) / (p.eta_d * p.E_max)


def soc_step(s, a, dt: float, p: ModelParams):
    """Explicit SoC update used in simulation and deployment (eq. 34)."""
    return s + dt * b_S(a, p)


# ---------------------------------------------------------------------------
# Reflected OU dynamics (numpy only; simulation path space)
# ---------------------------------------------------------------------------

def reflect(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mirror-reflect z into [lo, hi], correct for arbitrarily large
    overshoots (repeated reflection via the sawtooth fold)."""
    width = hi - lo
    if width <= 0:
        raise ValueError("empty interval")
    values = np.asarray(z, dtype=np.float64)
    w = np.mod(values - lo, 2.0 * width)
    folded = lo + np.minimum(w, 2.0 * width - w)
    # Reflection is exactly the identity in the box, including in floating
    # point. Avoid introducing observation noise at a hard tariff crossing.
    return np.where((values >= lo) & (values <= hi), values, folded)


def ou_step_exact(z: np.ndarray, kappa: float, sigma: float, dt: float,
                  eps: np.ndarray) -> np.ndarray:
    """Exact Gaussian OU transition (unreflected)."""
    phi = np.exp(-kappa * dt)
    var = sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
    return phi * z + np.sqrt(var) * eps


def correlated_normals(rng: np.random.Generator, rho: float, size) -> tuple:
    """Standard-normal innovations (e1, e2) with corr(e1, e2) = rho."""
    e1 = rng.standard_normal(size)
    e2i = rng.standard_normal(size)
    e2 = rho * e1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * e2i
    return e1, e2


def ou_pair_step(y: np.ndarray, pz: np.ndarray, p: ModelParams, dt: float,
                 rng: np.random.Generator, m_sigma: float = 1.0) -> tuple:
    """One reflected step of the correlated (Y, P) pair."""
    e1, e2 = correlated_normals(rng, p.rho, np.shape(y))
    y_next = ou_step_exact(y, p.kappa_y, p.sigma_y * m_sigma, dt, e1)
    p_next = ou_step_exact(pz, p.kappa_p, p.sigma_p * m_sigma, dt, e2)
    return (reflect(y_next, p.y_min, p.y_max),
            reflect(p_next, p.p_min, p.p_max))


def simulate_exogenous(p: ModelParams, n_paths: int, n_steps: int, dt: float,
                       rng: np.random.Generator, y0=0.0, p0=0.0,
                       m_sigma: float = 1.0) -> tuple:
    """Simulate reflected correlated OU paths; returns (Y, P) with shape
    (n_paths, n_steps + 1)."""
    Y = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    P = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    Y[:, 0] = y0
    P[:, 0] = p0
    for k in range(n_steps):
        Y[:, k + 1], P[:, k + 1] = ou_pair_step(
            Y[:, k], P[:, k], p, dt, rng, m_sigma)
    return Y, P
