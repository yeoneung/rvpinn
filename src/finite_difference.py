"""Monotone backward finite-difference solver for the 2-state (S, Y) problem.

Scheme (main.tex Experiment A): backward Euler in time, upwind first
derivatives selected by drift sign (monotone), centered diffusion with
reflected Neumann boundary in Y, state-dependent feasible action set at every
S node, Howard policy iteration at every time step. Price is deterministic
(P residual removed, pz = 0). All float64.

The same sweep with a frozen action field solves the linear policy-evaluation
PDE (used to compute reference costs of neural policies and for the
manufactured-solution test via an additive source).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import ModelParams
from .costs import RegimeProfile, psi_eps
from .hamiltonian import minimize_hamiltonian
from .safety import cont_bounds


class FDGrid:
    def __init__(self, p: ModelParams, Ns: int, Ny: int, Nt: int):
        self.s = np.linspace(p.s_min, p.s_max, Ns)
        self.y = np.linspace(p.y_min, p.y_max, Ny)
        self.t = np.linspace(0.0, p.T, Nt)
        self.Ns, self.Ny, self.Nt = Ns, Ny, Nt
        self.hs = self.s[1] - self.s[0]
        self.hy = self.y[1] - self.y[0]
        self.dt = self.t[1] - self.t[0]
        # flattened meshes, ordering index = i_s * Ny + i_y
        SS, YY = np.meshgrid(self.s, self.y, indexing="ij")
        self.S_flat = SS.ravel()
        self.Y_flat = YY.ravel()


def _upwind_derivatives(V: np.ndarray, grid: FDGrid) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided S-derivatives D+ and D- of V (Ns, Ny), one-sided at edges."""
    Dp = np.empty_like(V)
    Dm = np.empty_like(V)
    Dp[:-1, :] = (V[1:, :] - V[:-1, :]) / grid.hs
    Dp[-1, :] = (V[-1, :] - V[-2, :]) / grid.hs
    Dm[1:, :] = (V[1:, :] - V[:-1, :]) / grid.hs
    Dm[0, :] = (V[1, :] - V[0, :]) / grid.hs
    return Dp, Dm


def _greedy_policy(V: np.ndarray, t: float, grid: FDGrid, p: ModelParams,
                   prof: RegimeProfile) -> np.ndarray:
    """Monotone-consistent greedy action at every node.

    Discharge piece (a >= 0, b_S <= 0) is minimized against D-;
    charge piece (a <= 0, b_S >= 0) against D+.
    """
    Dp, Dm = _upwind_derivatives(V, grid)
    s = grid.S_flat
    y = grid.Y_flat
    tt = np.full_like(s, t)
    pz = np.zeros_like(s)
    lo, hi = cont_bounds(s, p)
    zero = np.zeros_like(s)
    a_pos, h_pos = minimize_hamiltonian(tt, s, y, pz, Dm.ravel(), p, prof,
                                        zero, hi)
    a_neg, h_neg = minimize_hamiltonian(tt, s, y, pz, Dp.ravel(), p, prof,
                                        lo, zero)
    a = np.where(h_pos <= h_neg, a_pos, a_neg)
    return a.reshape(grid.Ns, grid.Ny)


def _operator_and_cost(a_field: np.ndarray, t: float, grid: FDGrid,
                       p: ModelParams, prof: RegimeProfile,
                       source: Optional[Callable] = None
                       ) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Sparse upwind generator L (rows: L V) and running cost vector."""
    Ns, Ny = grid.Ns, grid.Ny
    n = Ns * Ny
    a = a_field.ravel()
    s = grid.S_flat
    y = grid.Y_flat

    bs = p.eta_c * np.maximum(-a, 0.0) / p.E_max \
        - np.maximum(a, 0.0) / (p.eta_d * p.E_max)
    by = -p.kappa_y * y
    dyy = 0.5 * p.sigma_y ** 2

    idx = np.arange(n)
    i_s = idx // Ny
    i_y = idx % Ny

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # --- S drift, upwind (one-sided fallback at edges keeps monotonicity
    #     because the tapered action set makes the drift inward there) ---
    bsp = np.maximum(bs, 0.0)
    bsm = np.minimum(bs, 0.0)
    up_ok = i_s < Ns - 1
    dn_ok = i_s > 0
    m = up_ok & (bsp > 0)
    add(idx[m], idx[m] + Ny, bsp[m] / grid.hs)
    add(idx[m], idx[m], -bsp[m] / grid.hs)
    m = dn_ok & (bsm < 0)
    add(idx[m], idx[m] - Ny, -bsm[m] / grid.hs)
    add(idx[m], idx[m], bsm[m] / grid.hs)

    # --- Y drift, upwind with reflecting Neumann (ghost = mirror) ---
    byp = np.maximum(by, 0.0)
    bym = np.minimum(by, 0.0)
    m = (i_y < Ny - 1) & (byp > 0)
    add(idx[m], idx[m] + 1, byp[m] / grid.hy)
    add(idx[m], idx[m], -byp[m] / grid.hy)
    m = (i_y > 0) & (bym < 0)
    add(idx[m], idx[m] - 1, -bym[m] / grid.hy)
    add(idx[m], idx[m], bym[m] / grid.hy)

    # --- Y diffusion, centered; Neumann mirror at edges (V_{-1} = V_1) ---
    coef = dyy / grid.hy ** 2
    m = (i_y > 0) & (i_y < Ny - 1)
    add(idx[m], idx[m] - 1, coef * np.ones(m.sum()))
    add(idx[m], idx[m] + 1, coef * np.ones(m.sum()))
    add(idx[m], idx[m], -2.0 * coef * np.ones(m.sum()))
    m = i_y == 0
    add(idx[m], idx[m] + 1, 2.0 * coef * np.ones(m.sum()))
    add(idx[m], idx[m], -2.0 * coef * np.ones(m.sum()))
    m = i_y == Ny - 1
    add(idx[m], idx[m] - 1, 2.0 * coef * np.ones(m.sum()))
    add(idx[m], idx[m], -2.0 * coef * np.ones(m.sum()))

    L = sp.csr_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    nbar = prof.n_bar(np.full(1, t))[0]
    cbar = prof.c_bar(np.full(1, t))[0]
    G = nbar + y - a
    C = cbar
    imp = psi_eps(G, p.eps_g)
    exp_ = psi_eps(-G, p.eps_g)
    ell = (C * imp - p.alpha_s * C * exp_
           + p.lam_pk * imp * imp
           + p.lam1 * (np.sqrt(a * a + p.eps_a ** 2) - p.eps_a)
           + p.lam2 * a * a
           + p.lam_s * (s - p.s_ref) ** 2)
    if source is not None:
        ell = ell + source(t, s, y)
    return L, ell


def solve_policy_eval(grid: FDGrid, p: ModelParams, prof: RegimeProfile,
                      policy: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                      terminal: Optional[Callable] = None,
                      source: Optional[Callable] = None) -> np.ndarray:
    """Backward Euler sweep for a fixed policy. Returns V (Nt, Ns, Ny)."""
    n = grid.Ns * grid.Ny
    V = np.empty((grid.Nt, grid.Ns, grid.Ny))
    if terminal is None:
        V[-1] = (p.lam_T * (grid.S_flat - p.s_tar) ** 2).reshape(grid.Ns, grid.Ny)
    else:
        V[-1] = terminal(grid.S_flat, grid.Y_flat).reshape(grid.Ns, grid.Ny)
    I = sp.identity(n, format="csr")
    for k in range(grid.Nt - 2, -1, -1):
        t = grid.t[k]
        a_field = policy(t, grid.S_flat, grid.Y_flat).reshape(grid.Ns, grid.Ny)
        L, ell = _operator_and_cost(a_field, t, grid, p, prof, source)
        rhs = V[k + 1].ravel() + grid.dt * ell
        V[k] = spla.spsolve((I - grid.dt * L).tocsc(), rhs).reshape(
            grid.Ns, grid.Ny)
    return V


def solve_hjb(grid: FDGrid, p: ModelParams, prof: RegimeProfile,
              howard_max: int = 60, howard_tol: float = 1e-7,
              value_tol: float = 1e-10, verbose: bool = False):
    """Howard policy iteration at every backward time step.

    Returns (V, A, diagnostics) with V (Nt, Ns, Ny), A (Nt-1, Ns, Ny).
    """
    n = grid.Ns * grid.Ny
    V = np.empty((grid.Nt, grid.Ns, grid.Ny))
    A = np.empty((grid.Nt - 1, grid.Ns, grid.Ny))
    V[-1] = (p.lam_T * (grid.S_flat - p.s_tar) ** 2).reshape(grid.Ns, grid.Ny)
    I = sp.identity(n, format="csr")
    howard_iters = []
    for k in range(grid.Nt - 2, -1, -1):
        t = grid.t[k]
        Vk = V[k + 1].copy()          # warm start for the greedy policy
        a_field = _greedy_policy(Vk, t, grid, p, prof)
        V_prev = None
        for it in range(howard_max):
            L, ell = _operator_and_cost(a_field, t, grid, p, prof)
            rhs = V[k + 1].ravel() + grid.dt * ell
            Vk = spla.spsolve((I - grid.dt * L).tocsc(), rhs).reshape(
                grid.Ns, grid.Ny)
            a_new = _greedy_policy(Vk, t, grid, p, prof)
            change = float(np.max(np.abs(a_new - a_field))) / max(p.a_max, 1.0)
            v_change = (float(np.max(np.abs(Vk - V_prev)))
                        / max(1.0, float(np.max(np.abs(Vk))))
                        if V_prev is not None else np.inf)
            a_field = a_new
            if change <= howard_tol or v_change <= value_tol:
                break
            V_prev = Vk
        howard_iters.append(it + 1)
        V[k] = Vk
        A[k] = a_field
        if verbose and k % max(1, grid.Nt // 10) == 0:
            print(f"  t={t:6.2f}h howard_iters={it+1} "
                  f"V0mid={Vk[grid.Ns//2, grid.Ny//2]:.4f}")
    return V, A, {"howard_iters": howard_iters}


def interp_V(V: np.ndarray, grid: FDGrid, t, s, y) -> np.ndarray:
    """Trilinear interpolation of the FD solution at query points."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((grid.t, grid.s, grid.y), V,
                                bounds_error=False, fill_value=None)
    pts = np.stack([np.asarray(t), np.asarray(s), np.asarray(y)], axis=-1)
    return f(pts)
