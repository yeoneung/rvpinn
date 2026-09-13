"""Smooth training costs, hard deployment costs, and regime profiles.

Regime forecast profiles are hourly medians. Inside the PDE/Hamiltonian they
are evaluated by periodic linear interpolation between hour centers (Lipschitz
in t); realized data in simulation is held piecewise constant within the hour.
Generic numpy/torch.
"""
from __future__ import annotations

import numpy as np

from .config import ModelParams


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def psi_eps(z, eps: float):
    """Smooth positive part: 0.5*(z + sqrt(z^2 + eps^2))."""
    return 0.5 * (z + (z * z + eps * eps) ** 0.5)


def dpsi_eps(z, eps: float):
    return 0.5 * (1.0 + z / (z * z + eps * eps) ** 0.5)


def sigmoid(z):
    """Generic logistic function (numpy / torch / python float)."""
    if _is_torch(z):
        import torch
        return torch.sigmoid(z)
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))


def step_fee_smooth(G, p: ModelParams):
    """Smoothed capacity-band surcharge c_step * sigma((G - g_thr)/w_step).

    A flat fee rate (currency/h) charged while the grid import exceeds the
    contracted band g_thr; the logistic smoothing (width w_step) is used on
    the training/PDE side. Nonconvex in the action. c_step = 0 disables.
    """
    if p.c_step == 0.0:
        return 0.0 * G
    return p.c_step * sigmoid((G - p.g_thr) / p.w_step)


def step_fee_exact(G, p: ModelParams):
    """Exact surcharge indicator used in deployment cost accounting."""
    if p.c_step == 0.0:
        return 0.0 * np.asarray(G, dtype=np.float64)
    return p.c_step * (np.asarray(G, dtype=np.float64) > p.g_thr)


class RegimeProfile:
    """Hourly forecast profile with periodic linear interpolation.

    values[h] is the profile at hour-center t = h + 0.5.
    """

    def __init__(self, n_bar_hourly, c_bar_hourly, T: float = 24.0):
        self.n_hour = np.asarray(n_bar_hourly, dtype=np.float64)
        self.c_hour = np.asarray(c_bar_hourly, dtype=np.float64)
        self.T = float(T)
        assert len(self.n_hour) == 24 and len(self.c_hour) == 24

    def _interp(self, t, vals):
        if _is_torch(t):
            import torch
            v = torch.as_tensor(vals, dtype=t.dtype, device=t.device)
            x = torch.remainder(t - 0.5, self.T)
            i0 = torch.floor(x).long() % 24
            i1 = (i0 + 1) % 24
            w = x - torch.floor(x)
            return (1.0 - w) * v[i0] + w * v[i1]
        x = np.mod(np.asarray(t, dtype=np.float64) - 0.5, self.T)
        i0 = np.floor(x).astype(int) % 24
        i1 = (i0 + 1) % 24
        w = x - np.floor(x)
        return (1.0 - w) * vals[i0] + w * vals[i1]

    def n_bar(self, t):
        return self._interp(t, self.n_hour)

    def c_bar(self, t):
        return self._interp(t, self.c_hour)

    def hold(self, t, which: str = "n"):
        """Piecewise-constant hourly hold (realized-data convention)."""
        vals = self.n_hour if which == "n" else self.c_hour
        idx = np.floor(np.mod(np.asarray(t, dtype=np.float64), self.T)).astype(int) % 24
        return vals[idx]


def grid_exchange(t, y, a, prof: RegimeProfile):
    """G = n_bar(t) + y - a; positive = import."""
    return prof.n_bar(t) + y - a


def running_cost(t, s, y, pz, a, p: ModelParams, prof: RegimeProfile):
    """Smooth training cost ell(t,s,y,p,a), in EUR/h.

    The common-objective configuration sets lam_pk, lam2, and lam_s to zero.
    These optional regularizers are not part of hard deployment accounting.
    """
    G = grid_exchange(t, y, a, prof)
    C = prof.c_bar(t) + pz
    imp = psi_eps(G, p.eps_g)
    exp_ = psi_eps(-G, p.eps_g)
    return (C * imp
            - p.alpha_s * C * exp_
            + p.lam_pk * imp * imp
            + p.lam1 * ((a * a + p.eps_a * p.eps_a) ** 0.5 - p.eps_a)
            + p.lam2 * a * a
            + p.lam_s * (s - p.s_ref) ** 2
            + step_fee_smooth(G, p))


def terminal_cost(s, p: ModelParams):
    return p.lam_T * (s - p.s_tar) ** 2


def common_running_cost_exact(C, G, a, p: ModelParams):
    """Common hard economic running cost (currency/hour).

    This is the cost used to compare *all* controllers.  It intentionally
    includes the energy bill, linear throughput degradation, and band fee.
    Quadratic peak, quadratic action, and SoC-centering penalties are excluded.
    The tariff indicator and absolute throughput are not smoothed.
    """
    G_arr = np.asarray(G, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    C_arr = np.asarray(C, dtype=np.float64)
    energy = C_arr * np.maximum(G_arr, 0.0) \
        - p.alpha_s * C_arr * np.maximum(-G_arr, 0.0)
    degradation = p.lam1 * np.abs(a_arr)
    band = step_fee_exact(G_arr, p)
    return energy + degradation + band


def common_terminal_cost(s, p: ModelParams):
    """Common soft terminal charge used by every compared method."""
    return p.lam_T * (np.asarray(s, dtype=np.float64) - p.s_tar) ** 2


def bill_increment(C, G, dt: float, p: ModelParams):
    """Transaction bill: dt * (C*G^+ - alpha_s*C*G^-).
    Uses exact positive parts (no smoothing) for reporting."""
    Gp = np.maximum(G, 0.0)
    Gm = np.maximum(-G, 0.0)
    return dt * (C * Gp - p.alpha_s * C * Gm)
