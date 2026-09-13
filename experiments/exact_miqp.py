"""Common-objective exact-band rolling-horizon controllers for battery dispatch.

The primary formulation is an MIQP for the exact transaction bill, exact band
indicator, exact linear throughput charge, and the same soft quadratic
terminal charge used by every controller.  ``exact-band`` refers only to the
tariff representation.  Solver gaps and time limits remain explicit outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any, Dict, List, Optional

import numpy as np
from pyscipopt import Model, quicksum
from src.deployment_numerics import (DEPLOYMENT_VERSION, BAND_SEPARATION_KW,
                                    SOLVER_FEASIBILITY_TOL,
                                    recover_solver_action)
from src.safety import deploy_bounds


@dataclass
class SolveRecord:
    time: float
    status: str
    solve_s: float
    gap: float
    nodes: int
    primal_bound: float
    dual_bound: float
    has_solution: bool
    deployment_version: str = DEPLOYMENT_VERSION
    first_action_raw: float = float("nan")
    first_action_held: float = float("nan")
    first_action_recovery_kw: float = 0.0
    first_charge_kw: float = float("nan")
    first_discharge_kw: float = float("nan")
    first_band_active: Optional[bool] = None


class ExactMIPController:
    """Exact deterministic or perfect-forecast capacity-band MIP.

    The controller reoptimizes every ``reopt_every_hours`` and replays the
    intervening 15-minute actions.  A zero-action fallback is returned only if
    SCIP has no incumbent; every solve record is retained for reporting.
    """

    name = "mpc_exact_miqp"

    def __init__(self, p, horizon_hours: float = 24.0,
                 reopt_every_hours: float = 1.0, oracle: bool = False,
                 time_limit_s: float = 60.0, mip_gap: float = 1e-4,
                 random_seed: int = 0, verbose: bool = False,
                 formulation: str = "miqp", shrinking_horizon: bool = True,
                 band_mode: str = "exact", step_slope: float = 0.0):
        self.p = p
        self.H = int(round(horizon_hours / p.dt_ctrl))
        self.reopt = float(reopt_every_hours)
        self.oracle = bool(oracle)
        self.time_limit_s = float(time_limit_s)
        self.mip_gap = float(mip_gap)
        self.random_seed = int(random_seed)
        self.verbose = bool(verbose)
        if formulation not in {"milp", "miqp"}:
            raise ValueError("formulation must be 'milp' or 'miqp'")
        self.formulation = formulation
        self.shrinking_horizon = bool(shrinking_horizon)
        if band_mode not in {"exact", "envelope", "softcap"}:
            raise ValueError("unknown band_mode")
        self.band_mode = band_mode
        self.step_slope = float(step_slope)
        self.solve_records: List[SolveRecord] = []
        self._plan: Optional[np.ndarray] = None
        self._plan_t0: Optional[float] = None
        self.name = (f"mpc_exact_{formulation}" if band_mode == "exact"
                     else f"mpc_{band_mode}_{formulation}")
        if oracle:
            self.name += "_perfect_forecast"

    def reset(self):
        self._plan = None
        self._plan_t0 = None

    @property
    def solver_summary(self) -> Dict[str, Any]:
        records = self.solve_records
        if not records:
            return {"n_solves": 0, "n_with_solution": 0}
        gaps = np.asarray([r.gap for r in records if np.isfinite(r.gap)])
        times = np.asarray([r.solve_s for r in records])
        proven = sum(r.status == "optimal" for r in records)
        return {
            "n_solves": len(records),
            "n_with_solution": sum(r.has_solution for r in records),
            "n_optimal": proven,
            "proven_optimal_fraction": proven / len(records),
            "mean_solve_s": float(times.mean()),
            "median_solve_s": float(np.median(times)),
            "p95_solve_s": float(np.quantile(times, 0.95)),
            "max_solve_s": float(times.max()),
            "mean_gap": float(gaps.mean()) if gaps.size else float("nan"),
            "max_gap": float(gaps.max()) if gaps.size else float("nan"),
            "n_first_action_recoveries": sum(
                r.first_action_recovery_kw != 0.0 for r in records),
            "status_counts": {
                s: sum(r.status == s for r in records)
                for s in sorted(set(r.status for r in records))
            },
            "records": [asdict(r) for r in records],
        }

    def _forecast(self, t: float, horizon_steps: int, y_now: float,
                  p_now: float, ctx) -> tuple[np.ndarray, np.ndarray]:
        dt = self.p.dt_ctrl
        steps = t + dt * np.arange(horizon_steps)
        if self.oracle:
            n = ctx["N_of_t"](steps)
            price = ctx["C_of_t"](steps)
        else:
            # Conditional-mean forecast under the calibrated OU model.  This
            # uses exactly the current information available to the neural
            # feedback policy and avoids an artificially weak profile-only MPC.
            tau = steps - float(t)
            n = (ctx["N_forecast"](steps)
                 + float(y_now) * np.exp(-self.p.kappa_y * tau))
            price = (ctx["C_forecast"](steps)
                     + float(p_now) * np.exp(-self.p.kappa_p * tau))
        # Current physical observations are not reconstructed from a folded
        # residual. Future forecasts still use the common calibrated model.
        if "N_now" in ctx:
            n[0] = float(ctx["N_now"])
        if "C_now" in ctx:
            price[0] = float(ctx["C_now"])
        return (np.asarray(n, dtype=np.float64),
                np.asarray(price, dtype=np.float64))

    @staticmethod
    def _finite_bound(model: Model, value: float) -> float:
        return float(value) if np.isfinite(value) else float("nan")

    def _solve_plan(self, t: float, s_now: float, ctx,
                    y_now: float = 0.0, p_now: float = 0.0) -> np.ndarray:
        p, dt = self.p, self.p.dt_ctrl
        H = (max(1, int(round((p.T - t) / dt)))
             if self.shrinking_horizon else self.H)
        net, price = self._forecast(t, H, y_now, p_now, ctx)
        model = Model(f"battery_band_miqp_{t:.2f}")
        if not self.verbose:
            model.hideOutput(True)
        model.setRealParam("limits/time", self.time_limit_s)
        model.setRealParam("limits/gap", self.mip_gap)
        model.setRealParam("numerics/feastol", SOLVER_FEASIBILITY_TOL)
        model.setIntParam("parallel/maxnthreads", 1)
        model.setIntParam("randomization/randomseedshift", self.random_seed)

        s = [model.addVar(lb=p.s_min, ub=p.s_max, name=f"s_{k}")
             for k in range(H + 1)]
        charge = [model.addVar(lb=0.0, ub=p.a_c, name=f"c_{k}")
                  for k in range(H)]
        discharge = [model.addVar(lb=0.0, ub=p.a_d, name=f"d_{k}")
                     for k in range(H)]
        grid_ub = np.maximum(net + p.a_c, 0.0)
        export_ub = np.maximum(p.a_d - net, 0.0)
        gp = [model.addVar(lb=0.0, ub=float(grid_ub[k]), name=f"gp_{k}")
              for k in range(H)]
        gm = [model.addVar(lb=0.0, ub=float(export_ub[k]), name=f"gm_{k}")
              for k in range(H)]
        # Enforce physical direction for every incumbent, not just an exact
        # optimum: dominance is insufficient under time/gap-limited solves.
        u_charge = [
            model.addVar(vtype="B", name=f"u_charge_{k}")
            for k in range(H)
        ]
        u_import = [
            model.addVar(vtype="B", name=f"u_import_{k}")
            for k in range(H)
        ]
        z_band = ([model.addVar(vtype="B", name=f"z_band_{k}")
                   for k in range(H)] if self.band_mode == "exact" else None)
        excess = ([model.addVar(lb=0.0, name=f"band_excess_{k}")
                   for k in range(H)] if self.band_mode != "exact" else None)

        model.addCons(s[0] == float(s_now), name="initial_soc")
        threshold_eps = BAND_SEPARATION_KW
        for k in range(H):
            model.addCons(
                s[k + 1] == s[k] + dt * (
                    p.eta_c * charge[k] - discharge[k] / p.eta_d
                ) / p.E_max,
                name=f"soc_{k}")
            if u_charge[k] is not None:
                model.addCons(charge[k] <= p.a_c * u_charge[k],
                              name=f"charge_mode_{k}")
                model.addCons(discharge[k] <= p.a_d * (1 - u_charge[k]),
                              name=f"discharge_mode_{k}")
            model.addCons(
                charge[k] <= p.a_c * (p.s_max - s[k]) / p.delta_s,
                name=f"charge_taper_{k}")
            model.addCons(
                discharge[k] <= p.a_d * (s[k] - p.s_min) / p.delta_s,
                name=f"discharge_taper_{k}")
            model.addCons(
                dt * p.eta_c * charge[k]
                <= p.E_max * (p.s_max - s[k]),
                name=f"charge_discrete_safe_{k}")
            model.addCons(
                dt * discharge[k] / p.eta_d
                <= p.E_max * (s[k] - p.s_min),
                name=f"discharge_discrete_safe_{k}")
            model.addCons(
                gp[k] - gm[k] == float(net[k]) - discharge[k] + charge[k],
                name=f"grid_balance_{k}")
            if u_import[k] is not None:
                model.addCons(gp[k] <= float(grid_ub[k]) * u_import[k],
                              name=f"import_mode_{k}")
                model.addCons(gm[k] <= float(export_ub[k]) * (1 - u_import[k]),
                              name=f"export_mode_{k}")

            g_lo = float(net[k] - p.a_d)
            g_hi = float(net[k] + p.a_c)
            if self.band_mode != "exact":
                model.addCons(excess[k] >= gp[k] - gm[k] - p.g_thr,
                              name=f"band_excess_def_{k}")
            elif g_hi <= p.g_thr:
                model.addCons(z_band[k] == 0, name=f"band_fixed0_{k}")
            elif g_lo > p.g_thr:
                model.addCons(z_band[k] == 1, name=f"band_fixed1_{k}")
            else:
                model.addCons(
                    gp[k] - gm[k]
                    <= p.g_thr + (g_hi - p.g_thr) * z_band[k],
                    name=f"band_upper_{k}")
                model.addCons(
                    gp[k] - gm[k]
                    >= p.g_thr + threshold_eps
                    - (p.g_thr + threshold_eps - g_lo) * (1 - z_band[k]),
                    name=f"band_lower_{k}")

        band_cost = (quicksum(dt * p.c_step * z_band[k] for k in range(H))
                     if self.band_mode == "exact" else
                     quicksum(dt * self.step_slope * excess[k]
                              for k in range(H)))
        linear_running = quicksum(
            dt * (
                float(price[k]) * gp[k]
                - p.alpha_s * float(price[k]) * gm[k]
                + p.lam1 * (charge[k] + discharge[k])
            ) for k in range(H)
        ) + band_cost
        if self.formulation == "milp":
            # Legacy hard-terminal ablation.  It is never a main version-3
            # comparator because its terminal treatment differs from the
            # learned controller.
            model.addCons(s[H] == p.s_tar, name="terminal_soc")
            model.setObjective(linear_running, "minimize")
        else:
            # Frozen common objective: the only quadratic term is the common
            # soft terminal charge.  No hidden peak/action regularizer enters
            # this comparator.
            total = (linear_running
                     + p.lam_T * (s[H] - p.s_tar) * (s[H] - p.s_tar))
            obj = model.addVar(lb=-model.infinity(),
                               name="objective_epigraph")
            model.addCons(obj >= total, name="objective_definition")
            model.setObjective(obj, "minimize")

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
            has_solution=has_solution,
        ))
        if not has_solution:
            return np.zeros(H, dtype=np.float64)
        plan = np.asarray([
            model.getSolVal(sol, discharge[k]) - model.getSolVal(sol, charge[k])
            for k in range(H)
        ], dtype=np.float64)
        record = self.solve_records[-1]
        record.first_charge_kw = float(model.getSolVal(sol, charge[0]))
        record.first_discharge_kw = float(model.getSolVal(sol, discharge[0]))
        record.first_action_raw = float(plan[0])
        lo, hi = deploy_bounds(np.array([s_now]), p, dt)
        active = (bool(model.getSolVal(sol, z_band[0]) > 0.5)
                  if z_band is not None else True)
        record.first_band_active = active if z_band is not None else None
        plan[0] = recover_solver_action(
            plan[0], float(net[0]), p.g_thr, float(lo[0]), float(hi[0]), active)
        record.first_action_held = float(plan[0])
        record.first_action_recovery_kw = float(plan[0] - record.first_action_raw)
        return plan

    def __call__(self, t, s, y, pz, ctx):
        dt = self.p.dt_ctrl
        k_in_plan = None
        if self._plan is not None and self._plan_t0 is not None:
            k = int(round((t - self._plan_t0) / dt))
            if (0 <= k < len(self._plan)
                    and (t - self._plan_t0) < self.reopt - 1e-9):
                k_in_plan = k
        if k_in_plan is None:
            self._plan = self._solve_plan(float(t), float(s), ctx,
                                          float(y), float(pz))
            self._plan_t0 = float(t)
            k_in_plan = 0
        return float(self._plan[k_in_plan])


# Backward-compatible name retained for the initial smoke script.
ExactMIQPController = ExactMIPController
