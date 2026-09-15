"""Deterministic and perfect-forecast MPC (baselines B4-B5) via CVXPY.

Convex epigraph formulation: grid exchange split into import/export parts
g+ >= 0, g- >= 0 with balance g+ - g- = N - (d - c). The battery action is
split into charge c >= 0 and discharge d >= 0; |a| = c + d and a^2 is
upper-bounded by c^2 + d^2 (exact under the complementarity that holds at
optimality). The taper is the linear constraint c <= a_c (s_max - s)/delta_s,
d <= a_d (s - s_min)/delta_s.

Negative forecast prices would make the exact import/export term concave;
in the (rare) affected hours the MPC surrogate uses export remuneration
alpha_s * max(C, 0) to preserve convexity. Deployment cost accounting is
unchanged. This deviation is recorded in the experiment report.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from .config import ModelParams


class MPCController:
    name = "mpc_deterministic"

    def __init__(self, p: ModelParams, horizon_hours: float = 24.0,
                 reopt_every_hours: float = 1.0, oracle: bool = False,
                 step_slope: float = 0.0):
        # step_slope > 0 adds the convex surrogate
        #   dt * step_slope * sum (gp - g_thr)^+
        # of the nonconvex capacity-band surcharge (Experiment E):
        #   envelope slope c_step/(G_ub - g_thr) = tightest convex
        #   minorant on [0, G_ub]; a large slope = soft-cap variant.
        self.step_slope = float(step_slope)
        self.p = p
        self.H = int(round(horizon_hours / p.dt_ctrl))
        self.reopt = reopt_every_hours
        self.oracle = oracle
        if oracle:
            self.name = "mpc_perfect_forecast"
        self._build()
        self._plan = None
        self._plan_t0 = None

    def _build(self):
        p, H = self.p, self.H
        dt = p.dt_ctrl
        self.c = cp.Variable(H, nonneg=True)
        self.d = cp.Variable(H, nonneg=True)
        self.gp = cp.Variable(H, nonneg=True)
        self.gm = cp.Variable(H, nonneg=True)
        self.s = cp.Variable(H + 1)
        self.N_par = cp.Parameter(H)
        self.Cb_par = cp.Parameter(H)      # buy price (can be negative)
        self.Cs_par = cp.Parameter(H)      # sell price, alpha_s*max(C,0)
        self.s0_par = cp.Parameter()
        cons = [self.s[0] == self.s0_par,
                self.s[1:] == self.s[:-1]
                + dt * (p.eta_c * self.c - self.d / p.eta_d) / p.E_max,
                self.s >= p.s_min, self.s <= p.s_max,
                self.c <= p.a_c, self.d <= p.a_d,
                self.c <= p.a_c * (p.s_max - self.s[:-1]) / p.delta_s,
                self.d <= p.a_d * (self.s[:-1] - p.s_min) / p.delta_s,
                self.gp - self.gm == self.N_par - (self.d - self.c)]
        cost = cp.sum(
            dt * (cp.multiply(self.Cb_par, self.gp)
                  - cp.multiply(self.Cs_par, self.gm)
                  + p.lam_pk * cp.square(self.gp)
                  + p.lam1 * (self.c + self.d)
                  + p.lam2 * (cp.square(self.c) + cp.square(self.d))
                  + self.step_slope * cp.pos(self.gp - p.g_thr))
        ) + p.lam_T * cp.square(self.s[H] - p.s_tar)
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def _solve_plan(self, t: float, s_now: float, ctx) -> np.ndarray:
        p, H = self.p, self.H
        dt = p.dt_ctrl
        steps_t = t + dt * np.arange(H)
        if self.oracle:
            N = ctx["N_of_t"](steps_t)      # realized future (oracle)
            C = ctx["C_of_t"](steps_t)
        else:
            N = ctx["N_forecast"](steps_t)  # regime forecast profile only
            C = ctx["C_forecast"](steps_t)
        self.N_par.value = np.asarray(N, dtype=np.float64)
        self.Cb_par.value = np.asarray(C, dtype=np.float64)
        self.Cs_par.value = p.alpha_s * np.maximum(np.asarray(C), 0.0)
        self.s0_par.value = float(s_now)
        try:
            self.prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6,
                            max_iter=40000, warm_start=True)
            ok = self.prob.status in ("optimal", "optimal_inaccurate")
        except (cp.SolverError, Exception):
            ok = False
        if not ok:
            try:
                self.prob.solve(solver=cp.CLARABEL)
                ok = self.prob.status in ("optimal", "optimal_inaccurate")
            except Exception:
                ok = False
        if not ok:
            return np.zeros(H)
        return np.asarray(self.d.value - self.c.value, dtype=np.float64)

    def __call__(self, t, s, y, pz, ctx):
        dt = self.p.dt_ctrl
        k_in_plan = None
        if self._plan is not None:
            k = int(round((t - self._plan_t0) / dt))
            if 0 <= k < len(self._plan) and (t - self._plan_t0) < self.reopt - 1e-9:
                k_in_plan = k
        if k_in_plan is None:
            self._plan = self._solve_plan(t, float(s), ctx)
            self._plan_t0 = t
            k_in_plan = 0
        return float(self._plan[k_in_plan])

    def reset(self):
        self._plan = None
        self._plan_t0 = None
