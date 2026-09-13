"""Chronological real-data evaluation engine (Experiment B).

Every method runs through run_day(): identical realized trajectory, identical
observation, identical safe projection, float64 accounting. Perturbation
hooks implement the robustness scenarios (Experiment C) without retuning.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ModelParams, ROOT
from .costs import (RegimeProfile, common_running_cost_exact,
                    common_terminal_cost, psi_eps, step_fee_smooth)
from .data_preprocess import assign_regime
from .dynamics import reflect, soc_step
from .regime_profiles import load_calibration
from .safety import project_action_to_safe_set
from .deployment_numerics import DEPLOYMENT_VERSION

PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "data", "splits")


def load_region_days(region: str = "primary",
                     part: str = "test",
                     cal: Optional[Dict] = None,
                     gamma_R_override: Optional[float] = None
                     ) -> List[Dict]:
    """Complete days of a split part with scaled net load and price."""
    cal = cal or load_calibration()
    rc = cal[region]
    zone, tz, split = rc["zone"], rc["tz"], rc["split"]
    df = pd.read_parquet(os.path.join(PROC, f"{region}_{zone}.parquet"))
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz)
    ts = df.index.tz_localize(None)
    lo, hi = split[part]
    df = df[(ts >= lo) & (ts < hi)]

    sc = rc["scaling"]
    gamma = gamma_R_override if gamma_R_override is not None else sc["gamma_R"]
    L = sc["P_base"] * df["load"] / sc["load_train_median"]
    R = gamma * sc["P_base"] * df["renew"] / sc["renew_train_q95"]
    N = (L - R).astype(np.float64)
    C = (df["price"] / 1000.0).astype(np.float64)
    Lr = L.astype(np.float64)
    Rr = R.astype(np.float64)

    try:
        import holidays as hol
        country = zone.split("_")[0]
        hdates = set(hol.country_holidays(
            country, years=sorted(set(df.index.year))).keys())
    except Exception:
        hdates = set()
    regs = assign_regime(df.index, hdates)

    days = []
    for day_ts, idxs in pd.Series(range(len(df)),
                                  index=df.index).groupby(df.index.normalize()):
        if len(idxs) != 24:
            continue
        i = idxs.values
        days.append({
            "date": str(day_ts.date()),
            "regime": regs.iloc[i[0]],
            "N": N.values[i], "C": C.values[i],
            "L": Lr.values[i], "R": Rr.values[i],
        })
    return days


def run_day(controller, day: Dict, p: ModelParams, prof: RegimeProfile,
            p_true: Optional[ModelParams] = None,
            collect_traj: bool = False) -> Dict:
    """Execute one controller on one realized day.

    p       : nominal parameters (safe set computed from these)
    p_true  : true dynamics parameters (defaults to p); the 'nominal safety
              parameters' robustness scenario passes p_true != p.
    """
    p_true = p_true or p
    dt_c, dt_s = p.dt_ctrl, p.dt_sim
    sub = int(round(dt_c / dt_s))
    n_ctrl = int(round(p.T / dt_c))
    N_h, C_h = day["N"], day["C"]

    def N_of_t(t):
        idx = np.clip(np.floor(np.asarray(t) % 24.0).astype(int), 0, 23)
        return N_h[idx]

    def C_of_t(t):
        idx = np.clip(np.floor(np.asarray(t) % 24.0).astype(int), 0, 23)
        return C_h[idx]

    ctx = {
        "date": day["date"],
        "regime": day["regime"],
        "N_of_t": N_of_t, "C_of_t": C_of_t,
        "N_forecast": lambda t: prof.n_bar(np.asarray(t, dtype=float)),
        "C_forecast": lambda t: prof.c_bar(np.asarray(t, dtype=float)),
    }
    if hasattr(controller, "reset"):
        controller.reset()

    s = float(p.s0)
    bill = 0.0
    objective = 0.0
    capacity_fee = 0.0
    degradation_cost = 0.0
    exceed_hours = 0.0
    exact_threshold_hours = 0.0
    near_threshold_hours = {5.0: 0.0, 10.0: 0.0, 20.0: 0.0}
    peak_import = 0.0
    export_energy = 0.0
    throughput = 0.0
    violations = 0
    max_violation = 0.0
    projections = 0
    exact_action_changes = 0
    tiny_positive_threshold_hours = 0.0
    reflections = 0
    latencies = []
    traj = {"t": [], "s": [], "a": [], "G": [], "N": [], "C": [], "y": [],
            "pz": []} if collect_traj else None

    for k in range(n_ctrl):
        t = k * dt_c
        hour = int(t) % 24
        N_now = float(N_h[hour])
        C_now = float(C_h[hour])
        y_raw = N_now - float(prof.n_bar(np.array([t]))[0])
        p_raw = C_now - float(prof.c_bar(np.array([t]))[0])
        y_obs = float(reflect(np.array([y_raw]), p.y_min, p.y_max)[0])
        p_obs = float(reflect(np.array([p_raw]), p.p_min, p.p_max)[0])
        # count only genuine domain exits (mirror mapping is identity inside
        # up to float rounding, which must not be counted)
        if abs(y_obs - y_raw) > 1e-9 * (p.y_max - p.y_min) \
                or abs(p_obs - p_raw) > 1e-9 * (p.p_max - p.p_min):
            reflections += 1
        ctx["N_now"], ctx["C_now"] = N_now, C_now

        t0 = time.perf_counter()
        a_raw = float(controller(t, s, y_obs, p_obs, ctx))
        latencies.append(time.perf_counter() - t0)
        # single shared safety gate, nominal parameters
        a = float(project_action_to_safe_set(
            np.array([a_raw]), np.array([s]), p, dt_c)[0])
        if abs(a - a_raw) > 1e-9 * max(1.0, abs(a_raw)):
            projections += 1
        exact_action_changes += int(a != a_raw)

        for j in range(sub):
            ts_ = t + j * dt_s
            hh = int(ts_) % 24
            G = float(N_h[hh]) - a
            C = float(C_h[hh])
            Gp, Gm = max(G, 0.0), max(-G, 0.0)
            bill += dt_s * (C * Gp - p.alpha_s * C * Gm)
            degradation_cost += dt_s * p.lam1 * abs(a)
            imp_s = psi_eps(G, p.eps_g)
            exp_s = psi_eps(-G, p.eps_g)
            objective += dt_s * (
                C * imp_s - p.alpha_s * C * exp_s
                + p.lam_pk * imp_s ** 2
                + p.lam1 * (np.sqrt(a * a + p.eps_a ** 2) - p.eps_a)
                + p.lam2 * a * a + p.lam_s * (s - p.s_ref) ** 2
                + float(step_fee_smooth(G, p)))
            if p.c_step > 0.0 and G > p.g_thr:
                capacity_fee += dt_s * p.c_step
                exceed_hours += dt_s
            if 0.0 < G - p.g_thr <= 1e-10:
                tiny_positive_threshold_hours += dt_s
            if abs(G - p.g_thr) <= 1e-6:
                exact_threshold_hours += dt_s
            for radius in near_threshold_hours:
                if abs(G - p.g_thr) <= radius:
                    near_threshold_hours[radius] += dt_s
            peak_import = max(peak_import, Gp)
            export_energy += dt_s * Gm
            throughput += dt_s * abs(a)
            # TRUE dynamics update (may differ from nominal)
            s = float(soc_step(s, a, dt_s, p_true))
            if s < p_true.s_min - 1e-8 or s > p_true.s_max + 1e-8:
                violations += 1
                max_violation = max(max_violation,
                                    p_true.s_min - s, s - p_true.s_max)
            if collect_traj and j == 0:
                traj["t"].append(t)
                traj["s"].append(s)
                traj["a"].append(a)
                traj["G"].append(G)
                traj["N"].append(float(N_h[hh]))
                traj["C"].append(C)
                traj["y"].append(y_obs)
                traj["pz"].append(p_obs)

    terminal_penalty = float(common_terminal_cost(s, p))
    objective += terminal_penalty
    common_cost = bill + capacity_fee + degradation_cost + terminal_penalty
    lat = np.array(latencies)
    out = {
        "deployment_version": DEPLOYMENT_VERSION,
        "action_search_version": getattr(controller, "action_search_version",
                                         "not_applicable"),
        "date": day["date"], "regime": day["regime"],
        "bill": bill, "objective": objective,
        "capacity_fee": capacity_fee,
        "degradation_cost": degradation_cost,
        "terminal_penalty": terminal_penalty,
        "exceed_hours": exceed_hours,
        "threshold_exact_hours": exact_threshold_hours,
        "threshold_near_5kw_hours": near_threshold_hours[5.0],
        "threshold_near_10kw_hours": near_threshold_hours[10.0],
        "threshold_near_20kw_hours": near_threshold_hours[20.0],
        "common_cost": common_cost,
        # Backward-compatible name.  In version 3 total_cost always means the
        # frozen common economic objective, not bill plus tariff alone.
        "total_cost": common_cost,
        "peak_import_kw": peak_import,
        "export_kwh": export_energy,
        "curtailment_kwh": 0.0,          # no inverter export cap in primary
        "throughput_kwh": throughput,
        "efc": throughput / (2.0 * p.E_max),
        "terminal_soc": s,
        "terminal_soc_dev": abs(s - p.s_tar),
        "soc_violations": violations,
        "max_violation": max_violation,
        "projection_events": projections,
        "exact_action_change_events": exact_action_changes,
        "tiny_positive_threshold_hours": tiny_positive_threshold_hours,
        "reflection_events": reflections,
        "latency_mean_ms": float(lat.mean() * 1e3),
        "latency_median_ms": float(np.median(lat) * 1e3),
        "latency_p95_ms": float(np.quantile(lat, 0.95) * 1e3),
    }
    if collect_traj:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


def evaluate_method(controller_factory: Callable[[str], object],
                    days: List[Dict],
                    params_by_regime: Dict[str, ModelParams],
                    profiles_by_regime: Dict[str, RegimeProfile],
                    p_true_by_regime: Optional[Dict[str, ModelParams]] = None,
                    collect_traj_dates: Optional[set] = None) -> List[Dict]:
    """Evaluate one method over a day list; controller built per regime."""
    results = []
    controllers = {}
    for day in days:
        reg = day["regime"]
        if reg not in params_by_regime:
            continue
        if reg not in controllers:
            controllers[reg] = controller_factory(reg)
        p = params_by_regime[reg]
        prof = profiles_by_regime[reg]
        p_true = (p_true_by_regime or {}).get(reg)
        ct = collect_traj_dates and day["date"] in collect_traj_dates
        results.append(run_day(controllers[reg], day, p, prof,
                               p_true=p_true, collect_traj=bool(ct)))
    return results


class NeuralController:
    """One-step hard-objective controller from a trained value network."""

    def __init__(self, model, p: ModelParams, prof: RegimeProfile):
        import torch
        from .deployed_policy import ACTION_SEARCH_VERSION
        self.action_search_version = ACTION_SEARCH_VERSION
        self.torch = torch
        self.model = model
        self.p = p
        self.prof = prof
        self.device = next(model.parameters()).device

    def __call__(self, t, s, y, pz, ctx):
        from .deployed_policy import deployed_one_step_action
        a, _ = deployed_one_step_action(
            self.model, float(t), float(s), float(y), float(pz), self.prof,
            dt=self.p.dt_ctrl,
            current_net=ctx.get("N_now"), current_price=ctx.get("C_now"))
        return a


class DRLController:
    """SB3 policy wrapper using the same observation encoding as the env."""

    def __init__(self, sb3_model, p: ModelParams, regimes: List[str],
                 regime: str):
        self.model = sb3_model
        self.p = p
        self.regimes = regimes
        self.regime = regime

    def __call__(self, t, s, y, pz, ctx):
        p = self.p
        onehot = np.zeros(len(self.regimes))
        onehot[self.regimes.index(self.regime)] = 1.0
        obs = np.concatenate([
            [2 * (s - p.s_min) / (p.s_max - p.s_min) - 1,
             2 * (y - p.y_min) / (p.y_max - p.y_min) - 1,
             2 * (pz - p.p_min) / (p.p_max - p.p_min) - 1,
             np.sin(2 * np.pi * t / p.T), np.cos(2 * np.pi * t / p.T)],
            onehot]).astype(np.float32)
        import torch
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)   # SB3 nets are float32
        try:
            u, _ = self.model.predict(obs, deterministic=True)
        finally:
            torch.set_default_dtype(prev_dtype)
        u = float(np.clip(np.asarray(u).ravel()[0], -1, 1))
        from .safety import deploy_bounds
        lo, hi = deploy_bounds(np.array([s]), p, p.dt_ctrl)
        return float(lo[0] + 0.5 * (u + 1.0) * (hi[0] - lo[0]))


def tune_threshold_rule(days_val: List[Dict],
                        params_by_regime, profiles_by_regime,
                        train_prices: np.ndarray,
                        lo_grid, hi_grid) -> Tuple[float, float, Dict]:
    """Grid-search price quantiles on the VALIDATION period only."""
    from .rules import NoStorage, PriceThreshold
    # Include the no-action limit so the tuned rule cannot be worse than the
    # no-storage reference on its own selection period.
    span = max(float(np.ptp(train_prices)), 1e-6)
    inactive_lo = float(np.min(train_prices) - span)
    inactive_hi = float(np.max(train_prices) + span)
    inactive = evaluate_method(
        lambda reg: NoStorage(), days_val,
        params_by_regime, profiles_by_regime)
    inactive_cost = float(np.mean([r["common_cost"] for r in inactive]))
    best = (0.0, 1.0, inactive_cost, inactive_lo, inactive_hi,
            "zero_action")
    for ql in lo_grid:
        for qh in hi_grid:
            c_lo = float(np.quantile(train_prices, ql))
            c_hi = float(np.quantile(train_prices, qh))
            if c_lo >= c_hi:
                continue
            res = evaluate_method(
                lambda reg: PriceThreshold(params_by_regime[reg], c_lo, c_hi),
                days_val, params_by_regime, profiles_by_regime)
            mean_cost = float(np.mean([r["common_cost"] for r in res]))
            if mean_cost < best[2]:
                best = (ql, qh, mean_cost, c_lo, c_hi,
                        "price_threshold")
    return best[3], best[4], {"q_lo": best[0], "q_hi": best[1],
                              "val_mean_common_cost": best[2],
                              "selected_kind": best[5]}
