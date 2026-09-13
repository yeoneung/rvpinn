"""Run version-3 comparators under one objective and 15-minute update rule."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exact_miqp import ExactMIPController  # noqa: E402
from stochastic_miqp import ScenarioExactMIPController  # noqa: E402
from src.evaluation import load_region_days, run_day  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.deployment_numerics import DEPLOYMENT_VERSION  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


class ZeroController:
    def __call__(self, t, s, y, pz, ctx):
        return 0.0


def _params(region: str, regime: str, threshold: float,
            fee: float, width: float):
    params, profiles, _ = regime_bundle(region, [regime])
    p = copy.deepcopy(params[regime])
    p.g_thr, p.c_step, p.w_step = threshold, fee, width
    return p, profiles[regime]


def _task(payload):
    (method, region, date, threshold, fee, width, time_limit,
     mip_gap, n_scenarios, scenario_pool_size, seed) = payload
    day = next(d for d in load_region_days(region, "test")
               if str(d["date"]) == date)
    p, prof = _params(region, day["regime"], threshold, fee, width)
    surrogate_slope = 0.0
    if method == "no_storage":
        controller = ZeroController()
    elif method == "convex_envelope_mpc":
        # Largest hinge slope that remains below the flat band fee on the
        # calibrated grid-exchange interval. The hard fee is still used in
        # deployment evaluation; only the online optimization representation
        # changes relative to conditional-mean exact-band MIQP.
        clock = np.arange(0.0, p.T, p.dt_ctrl)
        g_upper = (float(np.max(prof.n_bar(clock))) + p.y_max + p.a_c)
        surrogate_slope = p.c_step / max(g_upper - p.g_thr, 1e-9)
        controller = ExactMIPController(
            p, reopt_every_hours=p.dt_ctrl, oracle=False,
            time_limit_s=time_limit, mip_gap=mip_gap,
            random_seed=seed, formulation="miqp",
            shrinking_horizon=True, band_mode="envelope",
            step_slope=surrogate_slope)
    elif method == "deterministic_exact_band_miqp":
        controller = ExactMIPController(
            p, reopt_every_hours=p.dt_ctrl, oracle=False,
            time_limit_s=time_limit, mip_gap=mip_gap,
            random_seed=seed, formulation="miqp",
            shrinking_horizon=True)
    elif method == "stochastic_two_stage_exact_band_miqp":
        controller = ScenarioExactMIPController(
            p, n_scenarios=n_scenarios,
            scenario_pool_size=scenario_pool_size,
            reopt_every_hours=p.dt_ctrl,
            time_limit_s=time_limit, mip_gap=mip_gap,
            random_seed=seed, shrinking_horizon=True)
    else:
        raise ValueError(method)
    result = run_day(controller, day, p, prof)
    daily = {"method": method, "region": region,
             "threshold_kw": threshold, "fee_per_hour": fee,
             "smoothing_width_kw": width,
             "control_dt_min": 60.0 * p.dt_ctrl,
             "solver_time_limit_s": (time_limit if method != "no_storage"
                                      else 0.0),
             "n_scenarios": (n_scenarios if method.startswith("stochastic")
                              else 0), **result}
    daily["scenario_pool_size"] = (
        scenario_pool_size if method.startswith("stochastic") else 0)
    daily["scenario_stream_seed"] = (
        seed if method.startswith("stochastic") else 0)
    daily["surrogate_slope_eur_per_kwh"] = surrogate_slope
    solves = []
    if hasattr(controller, "solver_summary"):
        solves = [{"method": method, "region": region, "date": date,
                   "regime": day["regime"], "threshold_kw": threshold,
                   "fee_per_hour": fee, "n_scenarios": daily["n_scenarios"],
                   **row}
                  for row in controller.solver_summary.get("records", [])]
    return daily, solves


def _atomic_parquet(rows, path: Path):
    frame = pd.DataFrame(rows)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def select_evaluation_days(days, regimes, max_days):
    """Apply the evaluation budget within each requested seasonal regime."""
    if max_days is not None and max_days <= 0:
        raise ValueError('max-days must be positive')
    selected = []
    for regime in sorted(regimes):
        candidates = sorted((day for day in days if day['regime'] == regime),
                            key=lambda day: str(day['date']))
        selected.extend(candidates if max_days is None else candidates[:max_days])
    return sorted(selected, key=lambda day: str(day['date']))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="confirmatory")
    parser.add_argument("--regimes", default="winter_weekday")
    parser.add_argument("--methods", default=(
        "no_storage,deterministic_exact_band_miqp,"
        "stochastic_two_stage_exact_band_miqp"))
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--fee", type=float, default=40.0)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--max-days", type=int, default=30,
                        help="maximum held-out days per requested regime")
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--scenario-pool-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    regimes = {x.strip() for x in args.regimes.split(",") if x.strip()}
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    valid = {"no_storage", "convex_envelope_mpc",
             "deterministic_exact_band_miqp",
             "stochastic_two_stage_exact_band_miqp"}
    if set(methods) - valid:
        raise ValueError(f"unknown methods: {sorted(set(methods) - valid)}")
    days = select_evaluation_days(load_region_days(args.region, "test"),
                                  regimes, args.max_days)
    if not days:
        raise RuntimeError("no eligible held-out days")

    tag = (f"{args.region}_thr{args.threshold:g}_fee{args.fee:g}"
           f"_w{args.width:g}_n{args.scenarios}")
    if args.output_suffix:
        if not args.output_suffix.replace("-", "").replace("_", "").isalnum():
            raise ValueError("output suffix must be alphanumeric, '-' or '_'")
        tag += f"_{args.output_suffix}"
    daily_path = RESULTS / f"common_daily_{tag}.parquet"
    solve_path = RESULTS / f"common_solves_{tag}.parquet"
    daily_rows = (pd.read_parquet(daily_path).to_dict("records")
                  if daily_path.exists() else [])
    solve_rows = (pd.read_parquet(solve_path).to_dict("records")
                  if solve_path.exists() else [])
    if any(row.get("deployment_version") != DEPLOYMENT_VERSION
           for row in daily_rows + solve_rows):
        raise RuntimeError("legacy outputs cannot be resumed with corrected deployment; "
                           "use the isolated alignment workspace or a new output suffix")
    complete = {(str(row["date"]), str(row["method"]))
                for row in daily_rows}
    pending = [
        (method, args.region, str(day["date"]), args.threshold, args.fee,
         args.width, args.time_limit, args.mip_gap, args.scenarios,
         args.scenario_pool_size, args.seed)
        for method in methods for day in days
        if (str(day["date"]), method) not in complete
    ]
    workers = max(1, min(args.workers, len(pending))) if pending else 1
    print(f"{len(days)} days; {len(pending)} pending method-days; "
          f"{workers} workers", flush=True)
    if workers == 1:
        outputs = map(_task, pending)
        for daily, solves in outputs:
            daily_rows.append(daily)
            solve_rows.extend(solves)
            _atomic_parquet(daily_rows, daily_path)
            if solve_rows:
                _atomic_parquet(solve_rows, solve_path)
            print(daily["method"], daily["date"],
                  f"cost={daily['common_cost']:.3f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_task, item) for item in pending]
            for future in as_completed(futures):
                daily, solves = future.result()
                daily_rows.append(daily)
                solve_rows.extend(solves)
                _atomic_parquet(daily_rows, daily_path)
                if solve_rows:
                    _atomic_parquet(solve_rows, solve_path)
                print(daily["method"], daily["date"],
                      f"cost={daily['common_cost']:.3f}", flush=True)

    print(daily_path)
    if solve_rows:
        print(solve_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
