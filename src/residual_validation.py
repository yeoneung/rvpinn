"""Sobol designs and independent residual validation (main.tex §6.4).

Three disjoint roles with separate frozen seeds:
  - training design: fresh scrambled Sobol every epoch (seed training_design)
  - adaptive candidate pool (seed adaptive_candidate)
  - tuning-validation design (seed tuning_validation): early stopping and
    checkpoint selection
  - final certification design (seed final_certification): NEVER used for
    optimization/selection; evaluated only after checkpoint freeze.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .config import ModelParams

DTYPE = torch.float64


def sobol_points(n: int, p: ModelParams, seed: int, use_p: bool = True,
                 device: str = "cpu") -> dict:
    """Scrambled Sobol points over [0,T]x[s]x[y](x[p])."""
    d = 4 if use_p else 3
    eng = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    u = eng.draw(n).to(dtype=DTYPE, device=device)
    out = {
        "t": u[:, 0] * p.T,
        "s": p.s_min + u[:, 1] * (p.s_max - p.s_min),
        "y": p.y_min + u[:, 2] * (p.y_max - p.y_min),
    }
    out["pz"] = (p.p_min + u[:, 3] * (p.p_max - p.p_min)) if use_p \
        else torch.zeros(n, dtype=DTYPE, device=device)
    return out


def boundary_points(n: int, p: ModelParams, seed: int, use_p: bool = True,
                    device: str = "cpu") -> dict:
    """Points on the reflecting y (and p) boundaries."""
    eng = torch.quasirandom.SobolEngine(4 if use_p else 3, scramble=True,
                                        seed=seed)
    u = eng.draw(n).to(dtype=DTYPE, device=device)
    t = u[:, 0] * p.T
    s = p.s_min + u[:, 1] * (p.s_max - p.s_min)
    n_y = n // 2 if use_p else n
    y = p.y_min + u[:, 2] * (p.y_max - p.y_min)
    pz = (p.p_min + u[:, 3] * (p.p_max - p.p_min)) if use_p \
        else torch.zeros(n, dtype=DTYPE, device=device)
    # first half: y at its bounds; second half: p at its bounds
    y = y.clone()
    pz = pz.clone()
    half = torch.arange(n, device=device) < n_y
    y_edge = torch.where(u[:, 2] < 0.5, torch.full_like(y, p.y_min),
                         torch.full_like(y, p.y_max))
    y = torch.where(half, y_edge, y)
    if use_p:
        p_edge = torch.where(u[:, 3] < 0.5, torch.full_like(pz, p.p_min),
                             torch.full_like(pz, p.p_max))
        pz = torch.where(~half, p_edge, pz)
    return {"t": t, "s": s, "y": y, "pz": pz, "is_y_edge": half}


def residual_metrics(r: torch.Tensor) -> dict:
    ra = r.detach().abs()
    return {"rms": float((ra ** 2).mean().sqrt()),
            "max": float(ra.max()),
            "p99": float(torch.quantile(ra, 0.99)) if ra.numel() < 2 ** 24
            else float(torch.quantile(ra[:: max(1, ra.numel() // 2**20)], 0.99)),
            "mean": float(ra.mean())}


def fill_distance_estimate(n: int, dim: int) -> float:
    """Approximate fill distance of an n-point Sobol design in [0,1]^dim
    (order-of-magnitude; labeled a sensitivity estimate, not a certificate)."""
    return math.sqrt(dim) / (2.0 * n ** (1.0 / dim))


def lipschitz_sensitivity_estimate(model, points: dict, prof, n_probe: int = 4096,
                                   seed: int = 0) -> float:
    """Empirical gradient-norm estimate of the HJB-residual Lipschitz
    constant via sampled finite differences on the *residual field*.

    This is an EMPIRICAL sensitivity estimate (main.tex Remark 5.9): it is
    reported as 'Lipschitz-corrected sensitivity estimate', never as a
    certified global bound.
    """
    from .pinn_value import hjb_residual
    p = model.p
    n = min(n_probe, points["t"].shape[0])
    idx = torch.arange(n)
    base = {k: points[k][idx].clone() for k in ("t", "s", "y", "pz")}
    r0, _, _ = hjb_residual(model, base["t"], base["s"], base["y"],
                            base["pz"], prof)
    scales = {"t": p.T, "s": p.s_max - p.s_min, "y": p.y_max - p.y_min,
              "pz": p.p_max - p.p_min}
    h = 1e-3
    grad_sq = torch.zeros(n, dtype=DTYPE, device=r0.device)
    for k in ("t", "s", "y", "pz"):
        if k == "pz" and not model.use_p:
            continue
        pert = {kk: vv.clone() for kk, vv in base.items()}
        step = h * scales[k]
        pert[k] = pert[k] + step
        r1, _, _ = hjb_residual(model, pert["t"], pert["s"], pert["y"],
                                pert["pz"], prof)
        # derivative w.r.t. normalized coordinate (unit cube)
        grad_sq += ((r1 - r0) / h) ** 2
    return float(grad_sq.sqrt().max())
