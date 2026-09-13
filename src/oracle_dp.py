"""Perfect-foresight deterministic DP oracle for the capacity-band tariff.

Given one realized day (hourly-held net load N_h and price C_h), computes
the exact optimal open-loop schedule of the full (nonconvex) running cost
by backward dynamic programming on a state-of-charge grid, using the same
deploy-time action bounds and battery update as every other controller.
Because the day is deterministic and the SoC transition s' = s + delta(a)
is affine in s, a fine grid with linear value interpolation is essentially
exact. This is an information-advantaged reference (like perfect-forecast
MPC) that, unlike the convex MPC surrogate, prices the capacity-band
surcharge exactly.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import ModelParams
from .costs import RegimeProfile, psi_eps, step_fee_smooth
from .dynamics import soc_step
from .safety import deploy_bounds


def _stage_cost(G, C, a, s, p: ModelParams):
    """Per-hour objective rate, identical to evaluation.run_day accounting
    (smoothed positive parts; smoothed capacity fee)."""
    imp = psi_eps(G, p.eps_g)
    exp_ = psi_eps(-G, p.eps_g)
    return (C * imp - p.alpha_s * C * exp_
            + p.lam_pk * imp * imp
            + p.lam1 * (np.sqrt(a * a + p.eps_a ** 2) - p.eps_a)
            + p.lam2 * a * a
            + p.lam_s * (s - p.s_ref) ** 2
            + step_fee_smooth(G, p))


def solve_day(day: Dict, p: ModelParams, n_s: int = 361,
              n_a: int = 161) -> np.ndarray:
    """Return the optimal action schedule a_k, k = 0..n_ctrl-1."""
    dt_c = p.dt_ctrl
    n_ctrl = int(round(p.T / dt_c))
    N_h, C_h = np.asarray(day["N"]), np.asarray(day["C"])

    S = np.linspace(p.s_min, p.s_max, n_s)
    lo, hi = deploy_bounds(S, p, dt_c)                       # (n_s,)
    frac = np.linspace(0.0, 1.0, n_a)
    A = lo[:, None] + (hi - lo)[:, None] * frac[None, :]     # (n_s, n_a)
    # always include the do-nothing action where feasible
    a0 = np.clip(0.0, lo, hi)
    A = np.concatenate([A, a0[:, None]], axis=1)             # (n_s, n_a+1)
    S_next = soc_step(np.repeat(S[:, None], A.shape[1], 1), A, dt_c, p)
    S_next = np.clip(S_next, p.s_min, p.s_max)

    V = p.lam_T * (S - p.s_tar) ** 2                         # terminal
    policy = np.zeros((n_ctrl, n_s), dtype=np.int64)
    for k in range(n_ctrl - 1, -1, -1):
        hh = int(k * dt_c) % 24
        G = float(N_h[hh]) - A                               # (n_s, n_a+1)
        cost = dt_c * _stage_cost(G, float(C_h[hh]), A, S[:, None], p)
        Vn = np.interp(S_next, S, V)
        Q = cost + Vn
        policy[k] = np.argmin(Q, axis=1)
        V = Q[np.arange(n_s), policy[k]]

    # forward pass from s0: re-parameterize the winning candidate index on
    # the current (off-grid) state's own feasible interval
    plan = np.zeros(n_ctrl)
    s = float(p.s0)
    for k in range(n_ctrl):
        j = int(policy[k][int(np.argmin(np.abs(S - s)))])
        la, ha = deploy_bounds(np.array([s]), p, dt_c)
        la0, ha0 = float(la[0]), float(ha[0])
        cand = np.concatenate([la0 + (ha0 - la0) * frac,
                               [np.clip(0.0, la0, ha0)]])
        a = float(cand[j])
        plan[k] = a
        s = float(np.clip(soc_step(s, a, dt_c, p), p.s_min, p.s_max))
    return plan


class OracleDPController:
    """Replays the per-day DP schedule; solved lazily on first call using
    the realized day passed through ctx (information-advantaged)."""

    name = "oracle_dp"

    def __init__(self, p: ModelParams, prof: RegimeProfile):
        self.p = p
        self.prof = prof
        self._plan = None

    def reset(self):
        self._plan = None

    def __call__(self, t, s, y, pz, ctx):
        p = self.p
        if self._plan is None:
            steps = np.arange(int(round(p.T / p.dt_ctrl))) * p.dt_ctrl
            day = {"N": np.asarray(ctx["N_of_t"](np.arange(24) + 0.0)),
                   "C": np.asarray(ctx["C_of_t"](np.arange(24) + 0.0))}
            self._plan = solve_day(day, p)
            self._steps = steps
        k = int(round(t / p.dt_ctrl))
        k = min(max(k, 0), len(self._plan) - 1)
        return float(self._plan[k])
