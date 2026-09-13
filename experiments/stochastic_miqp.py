"""Two-stage nonanticipative exact-band MIQP comparator.

At each 15-minute update, the current charge/discharge decision is shared by
all scenarios.  Later decisions are scenario-contingent recourse.  Since that
recourse reveals more than a true multistage policy, the comparator is named
two-stage throughout; it is never presented as multistage stochastic MPC.
"""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Dict, Tuple

import numpy as np
from pyscipopt import Model, quicksum

from exact_miqp import ExactMIPController, SolveRecord
from src.deployment_numerics import (BAND_SEPARATION_KW,
                                    SOLVER_FEASIBILITY_TOL,
                                    recover_solver_action)
from src.safety import deploy_bounds


def _ou_sd(kappa: float, sigma: float, dt: float) -> float:
    if kappa <= 1e-12:
        return float(sigma * np.sqrt(dt))
    return float(sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt))
                                 / (2.0 * kappa)))


def _reflect_scalar(x: float, lo: float, hi: float) -> float:
    width = hi - lo
    if width <= 0:
        raise ValueError("invalid reflection interval")
    if lo <= x <= hi:
        return float(x)
    u = (x - lo) % (2.0 * width)
    return float(lo + (u if u <= width else 2.0 * width - u))


class ScenarioExactMIPController(ExactMIPController):
    """Rolling two-stage stochastic MIQP with a shared first action."""

    name = "stochastic_two_stage_exact_band_miqp"

    def __init__(self, p, n_scenarios: int = 8,
                 scenario_pool_size: int = None, **kwargs):
        kwargs["formulation"] = "miqp"
        kwargs["oracle"] = False
        super().__init__(p, **kwargs)
        if n_scenarios < 2:
            raise ValueError("n_scenarios must be at least two")
        self.n_scenarios = int(n_scenarios)
        if scenario_pool_size is not None \
                and int(scenario_pool_size) < self.n_scenarios:
            raise ValueError("scenario_pool_size must cover n_scenarios")
        self.scenario_pool_size = (None if scenario_pool_size is None
                                   else int(scenario_pool_size))
        self.name = "stochastic_two_stage_exact_band_miqp"

    def _scenario_seed(self, t: float, ctx) -> int:
        date_digits = int("".join(c for c in str(ctx.get("date", "0"))
                                  if c.isdigit()) or "0")
        step = int(round(t / self.p.dt_ctrl))
        if self.scenario_pool_size is None:
            # Preserve the version-3 stream exactly.
            entropy = [self.random_seed, date_digits, step,
                       self.n_scenarios]
        else:
            # A fixed pool size makes the first M paths nested across M.
            entropy = [self.random_seed, date_digits, step,
                       self.scenario_pool_size, 4]
        seq = np.random.SeedSequence(entropy)
        return int(seq.generate_state(1, dtype=np.uint32)[0])

    def _forecast_scenarios(self, t: float, horizon_steps: int,
                            y_now: float, p_now: float,
                            ctx) -> Tuple[np.ndarray, np.ndarray]:
        """Reflected exact-step OU scenarios conditional on the observation."""
        p, dt = self.p, self.p.dt_ctrl
        n_s = self.n_scenarios
        n_pool = (n_s if self.scenario_pool_size is None
                  else self.scenario_pool_size)
        y = np.full(n_pool, float(y_now), dtype=np.float64)
        price_error = np.full(n_pool, float(p_now), dtype=np.float64)
        net = np.empty((n_pool, horizon_steps), dtype=np.float64)
        price = np.empty_like(net)
        rng = np.random.default_rng(self._scenario_seed(t, ctx))

        phi_y = np.exp(-p.kappa_y * dt)
        phi_p = np.exp(-p.kappa_p * dt)
        sd_y = _ou_sd(p.kappa_y, p.sigma_y, dt)
        sd_p = _ou_sd(p.kappa_p, p.sigma_p, dt)
        den = p.kappa_y + p.kappa_p
        cov = (p.rho * p.sigma_y * p.sigma_p * dt if den <= 1e-12
               else p.rho * p.sigma_y * p.sigma_p
               * (1.0 - np.exp(-den * dt)) / den)
        corr = float(np.clip(cov / max(sd_y * sd_p, 1e-15), -1.0, 1.0))
        corr_scale = np.sqrt(max(0.0, 1.0 - corr * corr))

        nested_innovations = (
            rng.standard_normal((max(0, horizon_steps - 1), n_pool, 2))
            if self.scenario_pool_size is not None else None)
        for h in range(horizon_steps):
            clock = t + h * dt
            net[:, h] = np.asarray(ctx["N_forecast"](clock)) + y
            price[:, h] = np.asarray(ctx["C_forecast"](clock)) + price_error
            if h + 1 < horizon_steps:
                if nested_innovations is None:
                    z1 = rng.standard_normal(n_pool)
                    z2_independent = rng.standard_normal(n_pool)
                else:
                    z1 = nested_innovations[h, :, 0]
                    z2_independent = nested_innovations[h, :, 1]
                z2 = corr * z1 + corr_scale * z2_independent
                y = phi_y * y + sd_y * z1
                price_error = phi_p * price_error + sd_p * z2
                y = np.array([_reflect_scalar(v, p.y_min, p.y_max)
                              for v in y])
                price_error = np.array([
                    _reflect_scalar(v, p.p_min, p.p_max)
                    for v in price_error])
        if "N_now" in ctx:
            net[:, 0] = float(ctx["N_now"])
        if "C_now" in ctx:
            price[:, 0] = float(ctx["C_now"])
        return net[:n_s], price[:n_s]

    def _solve_plan(self, t: float, s_now: float, ctx,
                    y_now: float = 0.0, p_now: float = 0.0) -> np.ndarray:
        p, dt = self.p, self.p.dt_ctrl
        H = (max(1, int(round((p.T - t) / dt)))
             if self.shrinking_horizon else self.H)
        net, price = self._forecast_scenarios(
            t, H, y_now, p_now, ctx)
        Q = self.n_scenarios
        model = Model(f"stochastic_exact_band_miqp_{t:.2f}")
        if not self.verbose:
            model.hideOutput(True)
        model.setRealParam("limits/time", self.time_limit_s)
        model.setRealParam("limits/gap", self.mip_gap)
        model.setRealParam("numerics/feastol", SOLVER_FEASIBILITY_TOL)
        model.setIntParam("parallel/maxnthreads", 1)
        model.setIntParam("randomization/randomseedshift", self.random_seed)

        s, charge, discharge, gp, gm, z_band = [], [], [], [], [], []
        u_charge, u_import = [], []
        for q in range(Q):
            s.append([model.addVar(lb=p.s_min, ub=p.s_max,
                                   name=f"s_{q}_{k}")
                      for k in range(H + 1)])
            charge.append([model.addVar(lb=0.0, ub=p.a_c,
                                        name=f"c_{q}_{k}")
                           for k in range(H)])
            discharge.append([model.addVar(lb=0.0, ub=p.a_d,
                                           name=f"d_{q}_{k}")
                              for k in range(H)])
            grid_ub = np.maximum(net[q] + p.a_c, 0.0)
            export_ub = np.maximum(p.a_d - net[q], 0.0)
            gp.append([model.addVar(lb=0.0, ub=float(grid_ub[k]),
                                    name=f"gp_{q}_{k}")
                       for k in range(H)])
            gm.append([model.addVar(lb=0.0, ub=float(export_ub[k]),
                                    name=f"gm_{q}_{k}")
                       for k in range(H)])
            z_band.append([model.addVar(vtype="B", name=f"z_{q}_{k}")
                           for k in range(H)])
            u_charge.append([
                model.addVar(vtype="B", name=f"u_charge_{q}_{k}")
                for k in range(H)])
            u_import.append([
                model.addVar(vtype="B", name=f"u_import_{q}_{k}")
                for k in range(H)])
            model.addCons(s[q][0] == float(s_now), name=f"initial_{q}")

            for k in range(H):
                model.addCons(
                    s[q][k + 1] == s[q][k] + dt * (
                        p.eta_c * charge[q][k]
                        - discharge[q][k] / p.eta_d) / p.E_max,
                    name=f"soc_{q}_{k}")
                model.addCons(
                    charge[q][k]
                    <= p.a_c * (p.s_max - s[q][k]) / p.delta_s,
                    name=f"charge_taper_{q}_{k}")
                model.addCons(
                    discharge[q][k]
                    <= p.a_d * (s[q][k] - p.s_min) / p.delta_s,
                    name=f"discharge_taper_{q}_{k}")
                model.addCons(
                    dt * p.eta_c * charge[q][k]
                    <= p.E_max * (p.s_max - s[q][k]),
                    name=f"charge_safe_{q}_{k}")
                model.addCons(
                    dt * discharge[q][k] / p.eta_d
                    <= p.E_max * (s[q][k] - p.s_min),
                    name=f"discharge_safe_{q}_{k}")
                model.addCons(
                    gp[q][k] - gm[q][k] == float(net[q, k])
                    - discharge[q][k] + charge[q][k],
                    name=f"grid_{q}_{k}")
                if u_charge[q][k] is not None:
                    model.addCons(
                        charge[q][k] <= p.a_c * u_charge[q][k],
                        name=f"charge_mode_{q}_{k}")
                    model.addCons(
                        discharge[q][k] <= p.a_d * (1 - u_charge[q][k]),
                        name=f"discharge_mode_{q}_{k}")
                if u_import[q][k] is not None:
                    model.addCons(
                        gp[q][k] <= float(grid_ub[k]) * u_import[q][k],
                        name=f"import_mode_{q}_{k}")
                    model.addCons(
                        gm[q][k] <= float(export_ub[k])
                        * (1 - u_import[q][k]),
                        name=f"export_mode_{q}_{k}")

                g_lo = float(net[q, k] - p.a_d)
                g_hi = float(net[q, k] + p.a_c)
                eps = BAND_SEPARATION_KW
                if g_hi <= p.g_thr:
                    model.addCons(z_band[q][k] == 0,
                                  name=f"band0_{q}_{k}")
                elif g_lo > p.g_thr:
                    model.addCons(z_band[q][k] == 1,
                                  name=f"band1_{q}_{k}")
                else:
                    grid = gp[q][k] - gm[q][k]
                    model.addCons(
                        grid <= p.g_thr + (g_hi - p.g_thr) * z_band[q][k],
                        name=f"band_upper_{q}_{k}")
                    model.addCons(
                        grid >= p.g_thr + eps
                        - (p.g_thr + eps - g_lo) * (1 - z_band[q][k]),
                        name=f"band_lower_{q}_{k}")

        # The current physical action is the only nonanticipative decision in
        # the two-stage approximation.  At h=0 all scenarios also share the
        # observed net load and price by construction.
        for q in range(1, Q):
            model.addCons(charge[q][0] == charge[0][0],
                          name=f"nonant_charge_{q}")
            model.addCons(discharge[q][0] == discharge[0][0],
                          name=f"nonant_discharge_{q}")

        scenario_totals = []
        for q in range(Q):
            running = quicksum(
                dt * (float(price[q, k]) * gp[q][k]
                      - p.alpha_s * float(price[q, k]) * gm[q][k]
                      + p.lam1 * (charge[q][k] + discharge[q][k])
                      + p.c_step * z_band[q][k])
                for k in range(H))
            scenario_totals.append(
                running + p.lam_T * (s[q][H] - p.s_tar)
                * (s[q][H] - p.s_tar))
        total = quicksum(scenario_totals) / float(Q)
        epigraph = model.addVar(lb=-model.infinity(), name="objective")
        model.addCons(epigraph >= total, name="objective_definition")
        model.setObjective(epigraph, "minimize")

        # Provide the feasible no-storage policy as a primal start.  Tight
        # online budgets must still return an incumbent rather than silently
        # falling back because presolve consumed the entire time allowance.
        start = model.createSol()
        start_total = 0.0
        for q in range(Q):
            model.setSolVal(start, s[q][0], float(s_now))
            scenario_value = 0.0
            for k in range(H):
                model.setSolVal(start, s[q][k + 1], float(s_now))
                model.setSolVal(start, charge[q][k], 0.0)
                model.setSolVal(start, discharge[q][k], 0.0)
                gp0 = max(float(net[q, k]), 0.0)
                gm0 = max(-float(net[q, k]), 0.0)
                z0 = 1.0 if float(net[q, k]) > p.g_thr else 0.0
                model.setSolVal(start, gp[q][k], gp0)
                model.setSolVal(start, gm[q][k], gm0)
                model.setSolVal(start, z_band[q][k], z0)
                if u_charge[q][k] is not None:
                    model.setSolVal(start, u_charge[q][k], 0.0)
                if u_import[q][k] is not None:
                    model.setSolVal(start, u_import[q][k],
                                    1.0 if gp0 > 0.0 else 0.0)
                scenario_value += dt * (
                    float(price[q, k]) * gp0
                    - p.alpha_s * float(price[q, k]) * gm0
                    + p.c_step * z0)
            scenario_value += p.lam_T * (float(s_now) - p.s_tar) ** 2
            start_total += scenario_value / float(Q)
        model.setSolVal(start, epigraph, float(start_total))
        model.addSol(start)

        started = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - started
        status = str(model.getStatus())
        sol = model.getBestSol()
        has_solution = sol is not None
        gap = float(model.getGap()) if has_solution else float("inf")
        self.solve_records.append(SolveRecord(
            time=float(t), status=status, solve_s=float(elapsed), gap=gap,
            nodes=int(model.getNNodes()),
            primal_bound=self._finite_bound(model, model.getPrimalbound()),
            dual_bound=self._finite_bound(model, model.getDualbound()),
            has_solution=has_solution))
        if not has_solution:
            return np.zeros(H, dtype=np.float64)
        first = (model.getSolVal(sol, discharge[0][0])
                 - model.getSolVal(sol, charge[0][0]))
        record = self.solve_records[-1]
        record.first_charge_kw = float(model.getSolVal(sol, charge[0][0]))
        record.first_discharge_kw = float(model.getSolVal(sol, discharge[0][0]))
        record.first_action_raw = float(first)
        record.first_band_active = bool(model.getSolVal(sol, z_band[0][0]) > 0.5)
        lo, hi = deploy_bounds(np.array([s_now]), p, dt)
        first = recover_solver_action(
            first, float(net[0, 0]), p.g_thr, float(lo[0]), float(hi[0]),
            record.first_band_active)
        record.first_action_held = float(first)
        record.first_action_recovery_kw = float(first - record.first_action_raw)
        # Reoptimization is prespecified at every 15-minute step, so only this
        # shared first action can be deployed.  Padding preserves the parent
        # controller interface without pretending recourse actions are shared.
        out = np.zeros(H, dtype=np.float64)
        out[0] = float(first)
        return out

    @property
    def solver_summary(self) -> Dict[str, Any]:
        out = super().solver_summary
        out["n_scenarios"] = self.n_scenarios
        out["scenario_pool_size"] = self.scenario_pool_size
        out["scenario_stream_seed"] = self.random_seed
        out["information_structure"] = "two-stage, shared first action"
        return out
