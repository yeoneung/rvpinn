"""Safe action sets (main.tex eqs. 9-10, 35-37).

Every controller — rules, MPC, DRL, direct HJB-PINN, RV-PINN-PI — must pass
its raw action through project_action_to_safe_set. Generic numpy/torch.
"""
from __future__ import annotations

from .config import ModelParams
from .dynamics import gmin, gmax


def q_c(s, p: ModelParams):
    return gmin((p.s_max - s) / p.delta_s, 1.0 * (s * 0 + 1.0))


def q_d(s, p: ModelParams):
    return gmin((s - p.s_min) / p.delta_s, 1.0 * (s * 0 + 1.0))


def cont_bounds(s, p: ModelParams):
    """Continuous admissible interval A_cont(s) = [-a_c*q_c, a_d*q_d].
    Values are clamped at 0 so that s marginally outside [s_min, s_max]
    still yields a valid (inward-pointing) interval."""
    lo = -p.a_c * gmax(q_c(s, p), 0.0 * s)
    hi = p.a_d * gmax(q_d(s, p), 0.0 * s)
    return lo, hi


def delta_bounds(s, p: ModelParams, dt: float):
    """One-step exact safe interval A_Delta(s) for update step dt (hours)."""
    lo = -gmin(p.a_c + 0.0 * s, p.E_max * (p.s_max - s) / (p.eta_c * dt))
    hi = gmin(p.a_d + 0.0 * s, p.eta_d * p.E_max * (s - p.s_min) / dt)
    lo = gmin(lo, 0.0 * s)   # guarantee 0 in the set even at boundary
    hi = gmax(hi, 0.0 * s)
    return lo, hi


def deploy_bounds(s, p: ModelParams, dt: float):
    """A_deploy(s) = A_cont(s) ∩ A_Delta(s); always contains 0."""
    lo1, hi1 = cont_bounds(s, p)
    lo2, hi2 = delta_bounds(s, p, dt)
    return gmax(lo1, lo2), gmin(hi1, hi2)


def project_action_to_safe_set(a, s, p: ModelParams, dt: float):
    """Clip into A_deploy without changing an already feasible float.

    Algebraic min/max identities can move an interior action by one ulp.
    That is economically material at a discontinuous tariff crossing.
    Keep the differentiable training-side bounds, but use native selectors
    for the final held-action projection (NumPy or PyTorch).
    """
    lo, hi = deploy_bounds(s, p, dt)
    if type(a).__module__.startswith("torch"):
        import torch
        return torch.minimum(torch.maximum(a, lo), hi)
    import numpy as np
    return np.clip(a, lo, hi)
