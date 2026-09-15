"""FD policy improvement decreases cost (Prop. 5.3 up to discretization)."""
import numpy as np

from src.finite_difference import FDGrid, _greedy_policy, solve_policy_eval


def test_greedy_improves_on_zero_policy(params, profile):
    grid = FDGrid(params, 41, 41, 49)

    def pi0(t, s, y):
        return np.zeros_like(np.asarray(s, dtype=float))

    V0 = solve_policy_eval(grid, params, profile, pi0)

    # improved policy field: greedy w.r.t. V0 at each time level
    A1 = np.empty((grid.Nt - 1, grid.Ns, grid.Ny))
    for k in range(grid.Nt - 1):
        A1[k] = _greedy_policy(V0[k + 1], grid.t[k], grid, params, profile)

    def pi1(t, s, y):
        k = min(int(np.floor(t / grid.dt + 1e-12)), grid.Nt - 2)
        return A1[k].ravel()

    V1 = solve_policy_eval(grid, params, profile, pi1)
    # improvement everywhere at t=0 up to a small discretization allowance
    scale = np.max(np.abs(V0[0]))
    assert np.max(V1[0] - V0[0]) <= 0.02 * scale, \
        f"improvement violated by {np.max(V1[0]-V0[0]):.4f} (scale {scale:.1f})"
    # and strictly better on average (the zero policy is not optimal here)
    assert V1[0].mean() < V0[0].mean()
