"""OU calibration from hourly residual series (main.tex eq. 8).

AR(1) representation: z_{k+1} = phi z_k + e_k. phi is fitted by least
squares (equal to conditional MLE for Gaussian AR(1)), then
kappa = -log(phi)/dt and sigma^2 = q * 2 kappa / (1 - exp(-2 kappa dt)),
with q the innovation variance. rho is the correlation of standardized
innovations of the two series, never of the raw levels.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def fit_ar1(z: np.ndarray, dt: float) -> Tuple[float, float, np.ndarray]:
    """Return (kappa, sigma, innovations). Requires len(z) >= 10."""
    z = np.asarray(z, dtype=np.float64)
    z0, z1 = z[:-1], z[1:]
    denom = float(np.dot(z0, z0))
    if denom <= 0:
        raise ValueError("degenerate series")
    phi = float(np.dot(z0, z1) / denom)
    # keep phi in (0, 1) for a well-defined mean-reverting OU
    phi = min(max(phi, 1e-6), 1.0 - 1e-6)
    innov = z1 - phi * z0
    q = float(np.var(innov, ddof=1))
    kappa = -np.log(phi) / dt
    sigma2 = q * 2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * dt))
    return float(kappa), float(np.sqrt(max(sigma2, 0.0))), innov


def fit_ar1_pairs(z0: np.ndarray, z1: np.ndarray, dt: float
                  ) -> Tuple[float, float, np.ndarray]:
    """AR(1) fit from explicit consecutive pairs (handles non-contiguous
    days: only within-day transitions are supplied)."""
    z0 = np.asarray(z0, dtype=np.float64)
    z1 = np.asarray(z1, dtype=np.float64)
    denom = float(np.dot(z0, z0))
    if denom <= 0 or len(z0) < 10:
        raise ValueError("degenerate pair set")
    phi = float(np.dot(z0, z1) / denom)
    phi = min(max(phi, 1e-6), 1.0 - 1e-6)
    innov = z1 - phi * z0
    q = float(np.var(innov, ddof=1))
    kappa = -np.log(phi) / dt
    sigma2 = q * 2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * dt))
    return float(kappa), float(np.sqrt(max(sigma2, 0.0))), innov


def fit_pair_from_pairs(y0, y1, p0, p1, dt: float) -> Dict[str, float]:
    """Correlated OU calibration from within-day consecutive pairs."""
    ky, sy, iy = fit_ar1_pairs(y0, y1, dt)
    kp, sp, ip = fit_ar1_pairs(p0, p1, dt)
    sy_i = np.std(iy, ddof=1)
    sp_i = np.std(ip, ddof=1)
    rho = (float(np.corrcoef(iy / sy_i, ip / sp_i)[0, 1])
           if sy_i > 0 and sp_i > 0 else 0.0)
    return {"kappa_y": ky, "sigma_y": sy, "kappa_p": kp, "sigma_p": sp,
            "rho": float(np.clip(rho, -0.95, 0.95))}


def fit_pair(y: np.ndarray, pz: np.ndarray, dt: float) -> Dict[str, float]:
    """Calibrate correlated OU pair; series must be aligned, same length."""
    ky, sy, iy = fit_ar1(y, dt)
    kp, sp, ip = fit_ar1(pz, dt)
    sy_i = np.std(iy, ddof=1)
    sp_i = np.std(ip, ddof=1)
    if sy_i > 0 and sp_i > 0:
        rho = float(np.corrcoef(iy / sy_i, ip / sp_i)[0, 1])
    else:
        rho = 0.0
    rho = float(np.clip(rho, -0.95, 0.95))
    return {"kappa_y": ky, "sigma_y": sy, "kappa_p": kp, "sigma_p": sp,
            "rho": rho}


def domain_bounds(train_series: np.ndarray, margin: float = 0.20) -> Tuple[float, float]:
    """Computational bounds: training range widened by `margin` of the range."""
    z = np.asarray(train_series, dtype=np.float64)
    lo, hi = float(np.min(z)), float(np.max(z))
    width = hi - lo
    if width <= 0:
        width = max(abs(hi), 1.0)
    return lo - margin * width, hi + margin * width
