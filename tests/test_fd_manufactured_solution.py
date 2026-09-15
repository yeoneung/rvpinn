"""FD policy-evaluation solver recovers a manufactured smooth solution
with the expected first-order refinement trend."""
import numpy as np

from src.finite_difference import FDGrid, solve_policy_eval
from src.safety import cont_bounds


def _manufactured(params):
    """V_m = Phi(s) + (T-t) * g(s) * cos(pi*(y-ymin)/W); Neumann-compatible."""
    W = params.y_max - params.y_min
    om = np.pi / W

    def V(t, s, y):
        g = 50.0 * (0.5 + (s - params.s_min) / (params.s_max - params.s_min))
        return params.lam_T * (s - params.s_tar) ** 2 \
            + (params.T - t) * g * np.cos(om * (y - params.y_min))

    def Vt(t, s, y):
        g = 50.0 * (0.5 + (s - params.s_min) / (params.s_max - params.s_min))
        return -g * np.cos(om * (y - params.y_min))

    def Vs(t, s, y):
        gp = 50.0 / (params.s_max - params.s_min)
        return 2 * params.lam_T * (s - params.s_tar) \
            + (params.T - t) * gp * np.cos(om * (y - params.y_min))

    def Vy(t, s, y):
        g = 50.0 * (0.5 + (s - params.s_min) / (params.s_max - params.s_min))
        return -(params.T - t) * g * om * np.sin(om * (y - params.y_min))

    def Vyy(t, s, y):
        g = 50.0 * (0.5 + (s - params.s_min) / (params.s_max - params.s_min))
        return -(params.T - t) * g * om ** 2 * np.cos(om * (y - params.y_min))

    return V, Vt, Vs, Vy, Vyy


def _solve_error(params, profile, Ns, Ny, Nt):
    V, Vt, Vs, Vy, Vyy = _manufactured(params)

    def policy(t, s, y):
        lo, hi = cont_bounds(np.asarray(s, dtype=float), params)
        return 0.25 * hi          # feasible discharging policy

    def bS(a):
        return -np.maximum(a, 0.0) / (params.eta_d * params.E_max) \
            + params.eta_c * np.maximum(-a, 0.0) / params.E_max

    def source(t, s, y):
        a = policy(t, s, y)
        lhs = (Vt(t, s, y) + bS(a) * Vs(t, s, y)
               - params.kappa_y * y * Vy(t, s, y)
               + 0.5 * params.sigma_y ** 2 * Vyy(t, s, y))
        # FD solves V_t + L V + (ell + f) = 0; choose f so V_m is exact:
        # f = -(lhs) - ell  =>  pass source = -lhs - ell; the solver adds
        # ell itself, so supply f_total = -lhs - ell via source callback.
        from src.costs import psi_eps
        tt = np.broadcast_to(np.asarray(t, dtype=float), np.shape(s))
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
        return -lhs - ell

    grid = FDGrid(params, Ns, Ny, Nt)
    Vnum = solve_policy_eval(grid, params, profile,
                             lambda t, s, y: policy(t, s, y),
                             source=source)
    TT, SS, YY = np.meshgrid(grid.t, grid.s, grid.y, indexing="ij")
    Vex = V(TT, SS, YY)
    return float(np.max(np.abs(Vnum - Vex)))


def test_refinement_trend(params, profile):
    e_coarse = _solve_error(params, profile, 21, 21, 25)
    e_fine = _solve_error(params, profile, 41, 41, 49)
    e_finest = _solve_error(params, profile, 81, 81, 97)
    # first-order scheme: roughly halving error per refinement; require
    # a robust monotone decrease with factor >= 1.5 per level
    assert e_fine < e_coarse / 1.5, (e_coarse, e_fine)
    assert e_finest < e_fine / 1.5, (e_fine, e_finest)
