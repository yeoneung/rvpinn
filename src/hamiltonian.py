"""Approximate scalar Hamiltonian search for smooth policy-iteration training.

The map a -> b_S(a) has a kink at a = 0, so the feasible interval is split
into the charging piece [lo, min(hi,0)] and the discharging piece
[max(lo,0), hi]. On each piece a coarse vectorized grid brackets the best
candidate, which is refined by golden-section search; endpoints and the kink
are always compared explicitly. Pure float64 numpy, fully vectorized over
states. Coarse-grid bracketing with local refinement is not a certificate of
global minimization for a general nonconvex Hamiltonian. The discrete hard-cost
deployment search is implemented separately in deployed_policy.py.
"""
from __future__ import annotations

import numpy as np

from .config import ModelParams
from .costs import (RegimeProfile, common_running_cost_exact, psi_eps,
                    step_fee_smooth)

_INVPHI = (np.sqrt(5.0) - 1.0) / 2.0


def hamiltonian_terms(a, s, y, nbar, cbar, pz, v_s, p: ModelParams):
    """h(a) = b_S(a) v_s + ell(t,x,a) with profile values precomputed."""
    a = np.asarray(a, dtype=np.float64)
    a_pos = np.maximum(a, 0.0)
    a_neg = np.maximum(-a, 0.0)
    bs = p.eta_c * a_neg / p.E_max - a_pos / (p.eta_d * p.E_max)
    G = nbar + y - a
    C = cbar + pz
    imp = psi_eps(G, p.eps_g)
    exp_ = psi_eps(-G, p.eps_g)
    ell = (C * imp - p.alpha_s * C * exp_
           + p.lam_pk * imp * imp
           + p.lam1 * (np.sqrt(a * a + p.eps_a * p.eps_a) - p.eps_a)
           + p.lam2 * a * a
           + p.lam_s * (s - p.s_ref) ** 2
           + step_fee_smooth(G, p))
    return bs * v_s + ell


def _golden(fun, lo, hi, n_iter: int):
    lo = lo.copy()
    hi = hi.copy()
    for _ in range(n_iter):
        span = hi - lo
        c = hi - _INVPHI * span
        d = lo + _INVPHI * span
        move = fun(c) < fun(d)
        hi = np.where(move, d, hi)
        lo = np.where(move, lo, c)
    return 0.5 * (lo + hi)


def _piece_min(fun, lo, hi, n_grid: int, n_refine: int):
    """Approximate minimum per row: coarse grid plus local golden refinement."""
    frac = np.linspace(0.0, 1.0, n_grid)
    A = lo[:, None] + (hi - lo)[:, None] * frac[None, :]     # (N, n_grid)
    H = fun(A)
    j = np.argmin(H, axis=1)
    rows = np.arange(lo.shape[0])
    j_lo = np.clip(j - 1, 0, n_grid - 1)
    j_hi = np.clip(j + 1, 0, n_grid - 1)
    b_lo = A[rows, j_lo]
    b_hi = A[rows, j_hi]
    a_ref = _golden(fun, b_lo, b_hi, n_refine)
    h_ref = fun(a_ref)
    a_grid = A[rows, j]
    h_grid = H[rows, j]
    better = h_ref < h_grid
    return np.where(better, a_ref, a_grid), np.where(better, h_ref, h_grid)


def minimize_hamiltonian(t, s, y, pz, v_s, p: ModelParams,
                         prof: RegimeProfile, lo, hi,
                         n_grid: int = 129, n_refine: int = 48):
    """Vectorized argmin_a of h(a; z) over [lo, hi] per state.

    Returns (a_star, h_star). Feasible bounds must satisfy lo <= 0 <= hi
    is NOT required; general lo <= hi supported.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    pz = np.atleast_1d(np.asarray(pz, dtype=np.float64))
    v_s = np.atleast_1d(np.asarray(v_s, dtype=np.float64))
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), s.shape).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), s.shape).copy()
    nbar = np.asarray(prof.n_bar(t), dtype=np.float64)
    cbar = np.asarray(prof.c_bar(t), dtype=np.float64)

    def fun(a):
        if a.ndim == 2:
            return hamiltonian_terms(
                a, s[:, None], y[:, None], nbar[:, None], cbar[:, None],
                pz[:, None], v_s[:, None], p)
        return hamiltonian_terms(a, s, y, nbar, cbar, pz, v_s, p)

    # charging piece [lo, min(hi, 0)] and discharging piece [max(lo,0), hi];
    # empty pieces collapse to a point and are handled uniformly.
    neg_lo, neg_hi = lo, np.minimum(hi, 0.0)
    neg_lo = np.minimum(neg_lo, neg_hi)
    pos_lo, pos_hi = np.maximum(lo, 0.0), hi
    pos_lo = np.minimum(pos_lo, pos_hi)

    a1, h1 = _piece_min(fun, neg_lo, neg_hi, n_grid, n_refine)
    a2, h2 = _piece_min(fun, pos_lo, pos_hi, n_grid, n_refine)

    # explicit candidates: endpoints and the kink (where feasible)
    cands_a = [a1, a2, lo, hi]
    zero = np.zeros_like(lo)
    feas0 = (lo <= 0.0) & (hi >= 0.0)
    cands_a.append(np.where(feas0, zero, lo))
    A = np.stack(cands_a, axis=1)
    H = fun(A)
    j = np.argmin(H, axis=1)
    rows = np.arange(lo.shape[0])
    return A[rows, j], H[rows, j]


def brute_force_min(t, s, y, pz, v_s, p: ModelParams, prof: RegimeProfile,
                    lo, hi, n_grid: int = 4097):
    """Dense-grid reference used only in verification tests."""
    return minimize_hamiltonian(t, s, y, pz, v_s, p, prof, lo, hi,
                                n_grid=n_grid, n_refine=0)


def hard_common_hamiltonian_terms(a, s, y, nbar, cbar, pz, v_s,
                                  p: ModelParams):
    """Deployment Hamiltonian under the frozen hard economic objective."""
    a = np.asarray(a, dtype=np.float64)
    a_pos = np.maximum(a, 0.0)
    a_neg = np.maximum(-a, 0.0)
    bs = p.eta_c * a_neg / p.E_max - a_pos / (p.eta_d * p.E_max)
    G = nbar + y - a
    C = cbar + pz
    return bs * v_s + common_running_cost_exact(C, G, a, p)


def minimize_hard_common_hamiltonian(t, s, y, pz, v_s, p: ModelParams,
                                     prof: RegimeProfile, lo, hi,
                                     n_grid: int = 4097):
    """Auditable scalar search for the hard deployment Hamiltonian.

    A dense deterministic grid is augmented by every structural point: both
    feasible endpoints, zero, the exact tariff crossing, and the fee-active
    side immediately adjacent to that crossing.  This avoids silently jumping
    across the tariff discontinuity during a generic local optimization.  The
    remaining discretization regret is checked against a denser grid in the
    version-3 decision-gate tests.
    """
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    s = np.atleast_1d(np.asarray(s, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    pz = np.atleast_1d(np.asarray(pz, dtype=np.float64))
    v_s = np.atleast_1d(np.asarray(v_s, dtype=np.float64))
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), s.shape).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), s.shape).copy()
    nbar = np.asarray(prof.n_bar(t), dtype=np.float64)
    cbar = np.asarray(prof.c_bar(t), dtype=np.float64)

    frac = np.linspace(0.0, 1.0, int(n_grid), dtype=np.float64)
    grid = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
    zero = np.clip(np.zeros_like(lo), lo, hi)
    crossing = np.clip(nbar + y - p.g_thr, lo, hi)
    fee_side = np.clip(np.nextafter(crossing, -np.inf), lo, hi)
    candidates = np.concatenate(
        [grid, lo[:, None], hi[:, None], zero[:, None],
         crossing[:, None], fee_side[:, None]], axis=1)
    values = hard_common_hamiltonian_terms(
        candidates, s[:, None], y[:, None], nbar[:, None], cbar[:, None],
        pz[:, None], v_s[:, None], p)
    j = np.argmin(values, axis=1)
    rows = np.arange(s.shape[0])
    return candidates[rows, j], values[rows, j]


# ---------------------------------------------------------------------------
# Torch (GPU) implementation of the same piecewise approximate search.
# Identical algorithm: per-piece coarse grid + golden refinement + explicit
# endpoint/kink candidates. float64 throughout. Verified against the numpy
# solver in tests/test_hamiltonian_torch.py.
# ---------------------------------------------------------------------------

def _torch_h(a, s, y, nbar, cbar, pz, v_s, p: ModelParams):
    import torch
    a_pos = torch.clamp(a, min=0.0)
    a_neg = torch.clamp(-a, min=0.0)
    bs = p.eta_c * a_neg / p.E_max - a_pos / (p.eta_d * p.E_max)
    G = nbar + y - a
    C = cbar + pz
    imp = 0.5 * (G + torch.sqrt(G * G + p.eps_g ** 2))
    exp_ = 0.5 * (-G + torch.sqrt(G * G + p.eps_g ** 2))
    ell = (C * imp - p.alpha_s * C * exp_
           + p.lam_pk * imp * imp
           + p.lam1 * (torch.sqrt(a * a + p.eps_a ** 2) - p.eps_a)
           + p.lam2 * a * a
           + p.lam_s * (s - p.s_ref) ** 2
           + step_fee_smooth(G, p))
    return bs * v_s + ell


def _torch_piece_min(fun, lo, hi, n_grid: int, n_refine: int):
    import torch
    frac = torch.linspace(0.0, 1.0, n_grid, dtype=lo.dtype,
                          device=lo.device)
    A = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
    H = fun(A)
    j = torch.argmin(H, dim=1)
    rows = torch.arange(lo.shape[0], device=lo.device)
    j_lo = torch.clamp(j - 1, 0, n_grid - 1)
    j_hi = torch.clamp(j + 1, 0, n_grid - 1)
    b_lo = A[rows, j_lo]
    b_hi = A[rows, j_hi]
    for _ in range(n_refine):
        span = b_hi - b_lo
        c = b_hi - _INVPHI * span
        d = b_lo + _INVPHI * span
        move = fun(c[:, None]).squeeze(1) < fun(d[:, None]).squeeze(1)
        b_hi = torch.where(move, d, b_hi)
        b_lo = torch.where(move, b_lo, c)
    a_ref = 0.5 * (b_lo + b_hi)
    h_ref = fun(a_ref[:, None]).squeeze(1)
    a_grid = A[rows, j]
    h_grid = H[rows, j]
    better = h_ref < h_grid
    return torch.where(better, a_ref, a_grid), \
        torch.where(better, h_ref, h_grid)


def minimize_hamiltonian_torch(t, s, y, pz, v_s, p: ModelParams,
                               prof: RegimeProfile, lo, hi,
                               n_grid: int = 129, n_refine: int = 48):
    """GPU/torch variant of minimize_hamiltonian (same semantics)."""
    import torch
    nbar = prof.n_bar(t)
    cbar = prof.c_bar(t)
    zero = torch.zeros_like(lo)

    def fun(A):
        if A.dim() == 2:
            return _torch_h(A, s[:, None], y[:, None], nbar[:, None],
                            cbar[:, None], pz[:, None], v_s[:, None], p)
        return _torch_h(A, s, y, nbar, cbar, pz, v_s, p)

    neg_hi = torch.minimum(hi, zero)
    neg_lo = torch.minimum(lo, neg_hi)
    pos_lo = torch.maximum(lo, zero)
    pos_lo = torch.minimum(pos_lo, hi)

    a1, h1 = _torch_piece_min(fun, neg_lo, neg_hi, n_grid, n_refine)
    a2, h2 = _torch_piece_min(fun, pos_lo, hi, n_grid, n_refine)
    feas0 = (lo <= 0.0) & (hi >= 0.0)
    a0 = torch.where(feas0, zero, lo)
    A = torch.stack([a1, a2, lo, hi, a0], dim=1)
    H = fun(A)
    j = torch.argmin(H, dim=1)
    rows = torch.arange(lo.shape[0], device=lo.device)
    return A[rows, j], H[rows, j]
