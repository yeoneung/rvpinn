"""Tariff-aware grid-target rules with validation-tuned reserve and recharge."""
from __future__ import annotations

import numpy as np

from src.safety import deploy_bounds
from src.deployment_numerics import threshold_action_candidates
from experiments.audit_terminal_optimizer import analytic_terminal_minimum


class TariffPeakShaving:
    name = "validation_tuned_tariff_peak_shaving"

    def __init__(self, p, reserve_soc, charge_target_soc, charge_price,
                 charge_when_already_above_band, allow_partial_shaving):
        self.p = p
        self.reserve = float(reserve_soc)
        self.target = float(charge_target_soc)
        self.charge_price = float(charge_price)
        self.high_charge = bool(charge_when_already_above_band)
        self.partial = bool(allow_partial_shaving)

    def __call__(self, t, s, y, pz, ctx):
        p = self.p
        net, price = float(ctx["N_now"]), float(ctx["C_now"])
        lo, hi = (float(x[0]) for x in deploy_bounds(np.array([s]), p, p.dt_ctrl))
        if t + p.dt_ctrl >= p.T - 1e-10:
            return float(np.clip(analytic_terminal_minimum(p, s, net, price)[1], lo, hi))
        remaining = max(0.0, p.T - t - p.dt_ctrl)
        recovery_floor = max(p.s_min, p.s_tar - p.eta_c * p.a_c * remaining / p.E_max)
        if s < recovery_floor:
            return float(np.clip(-(recovery_floor - s) * p.E_max / (p.eta_c * p.dt_ctrl), lo, 0.0))
        floor = max(self.reserve, recovery_floor)
        shave_hi = max(0.0, min(hi, (s - floor) * p.eta_d * p.E_max / p.dt_ctrl))
        crossing = threshold_action_candidates(net, p.g_thr, 0.0, shave_hi)
        inactive = crossing[(net - crossing <= p.g_thr) & (crossing >= 0)]
        can_clear = net > p.g_thr and inactive.size > 0
        if net > p.g_thr and (can_clear or self.partial):
            action = float(inactive.min()) if can_clear else shave_hi
            def stage(a):
                g = net - a
                return price * (max(g, 0.0) - p.alpha_s * max(-g, 0.0)) + p.lam1 * abs(a) + p.c_step * (g > p.g_thr)
            if action > 0 and stage(action) < stage(0.0):
                return action
        if price <= self.charge_price and s < self.target:
            action = max(lo, -(self.target - s) * p.E_max / (p.eta_c * p.dt_ctrl))
            if net <= p.g_thr:
                candidates = threshold_action_candidates(net, p.g_thr, action, 0.0)
                inactive = candidates[net - candidates <= p.g_thr]
                action = max(action, float(inactive.min())) if inactive.size else 0.0
            elif not self.high_charge:
                action = 0.0
            return float(np.clip(action, lo, hi))
        return 0.0
