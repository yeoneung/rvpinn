"""Residual identity / policy-evaluation PDE vs Monte Carlo (2-state).

For two fixed feasible policies, the FD solution of the linear policy PDE at
(t=0, s=0.5, y=0) must agree with a common-random-number Monte Carlo estimate
of J^pi within discretization + sampling tolerance. This validates the
residual identity (Lemma 5.5): the exact solution has zero residual, so
V_FD ~ J^pi.
"""
import numpy as np
import pytest

from src.costs import psi_eps
from src.dynamics import ou_step_exact, reflect, soc_step
from src.finite_difference import FDGrid, interp_V, solve_policy_eval
from src.safety import cont_bounds


def _mc_cost(policy, params, profile, n_paths=4000, seed=123):
    """Monte Carlo J^pi(0, s0, 0) using the same model conventions as the
    FD solver (interpolated profile, reflected exact OU, pz = 0)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 12.0
    n_steps = int(round(params.T / dt))
    s = np.full(n_paths, 0.5)
    y = np.zeros(n_paths)
    total = np.zeros(n_paths)
    for k in range(n_steps):
        t = k * dt
        a = policy(t, s, y)
        # running cost (pz = 0, price deterministic)
        tt = np.full(n_paths, t)
        G = profile.n_bar(tt) + y - a
        C = profile.c_bar(tt)
        imp = psi_eps(G, params.eps_g)
        exp_ = psi_eps(-G, params.eps_g)
        ell = (C * imp - params.alpha_s * C * exp_
               + params.lam_pk * imp * imp
               + params.lam1 * (np.sqrt(a * a + params.eps_a ** 2)
                                - params.eps_a)
               + params.lam2 * a * a
               + params.lam_s * (s - params.s_ref) ** 2)
        total += ell * dt
        s = soc_step(s, a, dt, params)
        eps = rng.standard_normal(n_paths)
        y = reflect(ou_step_exact(y, params.kappa_y, params.sigma_y, dt, eps),
                    params.y_min, params.y_max)
    total += params.lam_T * (s - params.s_tar) ** 2
    return total.mean(), total.std() / np.sqrt(n_paths)


@pytest.mark.parametrize("which", ["zero", "taper_discharge"])
def test_pde_matches_monte_carlo(params, profile, which):
    if which == "zero":
        def policy(t, s, y):
            return np.zeros_like(np.asarray(s, dtype=float))
    else:
        def policy(t, s, y):
            s = np.asarray(s, dtype=float)
            lo, hi = cont_bounds(s, params)
            return 0.3 * hi
    grid = FDGrid(params, 81, 81, 193)
    V = solve_policy_eval(grid, params, profile,
                          lambda t, s, y: policy(t, s, y))
    v0 = float(np.asarray(interp_V(V, grid, 0.0, 0.5, 0.0)).item())
    mc, se = _mc_cost(policy, params, profile)
    # tolerance: 4 MC standard errors + 2% discretization allowance
    tol = 4 * se + 0.02 * max(abs(mc), 1.0)
    assert abs(v0 - mc) < tol, f"PDE {v0:.3f} vs MC {mc:.3f} (se {se:.3f})"
