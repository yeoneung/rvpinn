"""Hamiltonian minimizer vs dense brute force at 10,000 random states."""
import numpy as np

from src.hamiltonian import hamiltonian_terms, minimize_hamiltonian
from src.safety import cont_bounds


def _dense_reference(t, s, y, pz, v_s, params, profile, lo, hi, n=8193):
    """Independent dense-grid reference (chunked)."""
    nbar = np.asarray(profile.n_bar(t))
    cbar = np.asarray(profile.c_bar(t))
    best_h = np.full(s.shape, np.inf)
    best_a = np.zeros_like(s)
    frac = np.linspace(0.0, 1.0, n)
    chunk = 500
    for i0 in range(0, len(s), chunk):
        sl = slice(i0, i0 + chunk)
        A = lo[sl, None] + (hi - lo)[sl, None] * frac[None, :]
        H = hamiltonian_terms(A, s[sl, None], y[sl, None], nbar[sl, None],
                              cbar[sl, None], pz[sl, None], v_s[sl, None],
                              params)
        j = np.argmin(H, axis=1)
        r = np.arange(H.shape[0])
        best_h[sl] = H[r, j]
        best_a[sl] = A[r, j]
    return best_a, best_h


def test_solver_matches_brute_force(params, profile):
    rng = np.random.default_rng(42)
    n = 10000
    t = rng.uniform(0, params.T, n)
    s = rng.uniform(params.s_min, params.s_max, n)
    y = rng.uniform(params.y_min, params.y_max, n)
    pz = rng.uniform(params.p_min, params.p_max, n)
    # v_s scale: value has units EUR; dv/ds can be large
    v_s = rng.uniform(-2000.0, 2000.0, n)
    lo, hi = cont_bounds(s, params)
    a_sol, h_sol = minimize_hamiltonian(t, s, y, pz, v_s, params, profile,
                                        lo, hi)
    a_ref, h_ref = _dense_reference(t, s, y, pz, v_s, params, profile, lo, hi)
    # solver must never be worse than the dense grid beyond tolerance
    grid_step = (hi - lo) / 8192.0
    tol = 1e-6 + 1e-8 * np.abs(h_ref)
    assert np.all(h_sol <= h_ref + tol), \
        f"worst excess {np.max(h_sol - h_ref):.3e}"
    # actions within feasible interval
    assert np.all(a_sol >= lo - 1e-9) and np.all(a_sol <= hi + 1e-9)


def test_kink_handled(params, profile):
    # With G = -a and a large throughput cost lam1 exceeding both the export
    # remuneration (discharge side) and any import saving (charge side),
    # the kink a = 0 is the unique global minimizer.
    params.lam1 = 0.5          # > alpha_s * C and > C for all test prices
    n = 100
    t = np.full(n, 12.0)
    s = np.full(n, 0.5)
    y = np.full(n, -profile.n_bar(np.full(1, 12.0))[0])  # G = -a
    pz = np.zeros(n)
    v_s = np.zeros(n)
    lo, hi = cont_bounds(s, params)
    a, h = minimize_hamiltonian(t, s, y, pz, v_s, params, profile, lo, hi)
    assert np.all(np.abs(a) < 1.0)


def test_export_arbitrage_analytic(params, profile):
    # Flat interior case with G = -a: for a > 0 the smooth optimum solves
    # alpha_s*C = lam1 + 2*lam2*a; when that root exceeds the cap the
    # solver must return the cap a_d*q_d(s).
    n = 10
    t = np.full(n, 12.0)
    s = np.full(n, 0.5)
    y = np.full(n, -profile.n_bar(np.full(1, 12.0))[0])
    pz = np.zeros(n)
    v_s = np.zeros(n)
    lo, hi = cont_bounds(s, params)
    C = profile.c_bar(np.full(1, 12.0))[0]
    root = (params.alpha_s * C - params.lam1) / (2 * params.lam2)
    assert root > hi[0]        # premise of this scenario
    a, _ = minimize_hamiltonian(t, s, y, pz, v_s, params, profile, lo, hi)
    assert np.allclose(a, hi, atol=1e-6)
