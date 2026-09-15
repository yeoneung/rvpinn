"""Run common-objective exact-band MIQP on held-out historical days."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VER3 = HERE.parent
sys.path.insert(0, str(VER3))
sys.path.insert(0, str(HERE))

from exact_miqp import ExactMIPController              # noqa: E402
from src.evaluation import load_region_days, run_day   # noqa: E402
from src.exp_common import load_stack, regime_bundle   # noqa: E402
from src.regime_profiles import load_calibration       # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
WEEKDAY_REGIMES = [
    "winter_weekday", "spring_weekday",
    "summer_weekday", "autumn_weekday",
]


def with_tariff(p, threshold: float, fee: float, width: float):
    out = copy.deepcopy(p)
    out.g_thr = float(threshold)
    out.c_step = float(fee)
    out.w_step = float(width)
    return out


def select_days(scope: str, max_days: Optional[int]):
    cal = load_calibration()
    days = load_region_days("primary", "test", cal)
    if scope == "main":
        days = [d for d in days if d["regime"] == "winter_weekday"][:30]
    elif scope == "seasonal":
        days = [d for d in days if d["regime"] in WEEKDAY_REGIMES]
    elif scope != "all":
        raise ValueError(scope)
    if max_days is not None:
        days = days[:max_days]
    return days


def solve_day_task(task):
    """Solve one independent day in a worker process.

    The worker reloads the frozen inputs instead of receiving controller
    objects or interpolation functions, which are not reliably picklable on
    Windows. Only the parent process writes the shared parquet files.
    """
    (mode, scope, target_date, day_index, n_days, threshold, fee, width,
     time_limit, mip_gap, det_reopt, pf_reopt) = task
    cfg = load_stack(str(VER3 / "configs" / "paper_full.yaml"))
    regimes = (["winter_weekday"] if scope == "main"
               else WEEKDAY_REGIMES if scope == "seasonal"
               else None)
    params, profiles, _ = regime_bundle("primary", regimes, cfg)
    params = {r: with_tariff(p, threshold, fee, width)
              for r, p in params.items()}
    day = next(d for d in select_days(scope, None)
               if str(d["date"]) == target_date)
    reg = day["regime"]
    p, prof = params[reg], profiles[reg]
    oracle = mode == "perfect_forecast"
    method = "exact_band_miqp_pf" if oracle else "exact_band_miqp"
    controller = ExactMIPController(
        p, oracle=oracle,
        reopt_every_hours=(pf_reopt if oracle else det_reopt),
        time_limit_s=time_limit, mip_gap=mip_gap, random_seed=0,
        formulation="miqp")
    result = run_day(controller, day, p, prof)
    daily_row = {"method": method, "scope": scope,
                 "threshold": threshold, "fee": fee, "width": width,
                 **result}
    solve_rows = [
        {"method": method, "date": day["date"], "regime": reg, **rec}
        for rec in controller.solver_summary["records"]
    ]
    message = (f"[{mode}] {day_index}/{n_days} {day['date']} {reg}: "
               f"total={result['total_cost']:.2f}, "
               f"latency={result['latency_mean_ms']:.1f} ms")
    return daily_row, solve_rows, message


def atomic_parquet(frame: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("main", "seasonal", "all"),
                    default="main")
    ap.add_argument("--modes", default="deterministic,perfect_forecast")
    ap.add_argument("--threshold", type=float, default=300.0)
    ap.add_argument("--fee", type=float, default=40.0)
    ap.add_argument("--width", type=float, default=10.0)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--mip-gap", type=float, default=0.01)
    ap.add_argument("--det-reopt", type=float, default=1.0,
                    help="hours between deterministic reoptimizations")
    ap.add_argument("--pf-reopt", type=float, default=24.0,
                    help="hours between perfect-forecast reoptimizations")
    ap.add_argument("--max-days", type=int)
    ap.add_argument("--rerun-gap-over", type=float,
                    help="rerun completed day/method pairs whose recorded "
                         "maximum relative gap exceeds this value")
    ap.add_argument("--workers", type=int, default=3,
                    help="independent day-level worker processes")
    args = ap.parse_args()

    cfg = load_stack(str(VER3 / "configs" / "paper_full.yaml"))
    regimes = (["winter_weekday"] if args.scope == "main"
               else WEEKDAY_REGIMES if args.scope == "seasonal"
               else None)
    params, profiles, _ = regime_bundle("primary", regimes, cfg)
    params = {r: with_tariff(p, args.threshold, args.fee, args.width)
              for r, p in params.items()}
    days = select_days(args.scope, args.max_days)

    tag = (f"{args.scope}_thr{args.threshold:g}_fee{args.fee:g}"
           f"_w{args.width:g}_forecastou_logicv2_shrink_pfreopt{args.pf_reopt:g}")
    if args.det_reopt != 1.0:
        tag += f"_detreopt{args.det_reopt:g}"
    daily_path = RESULTS / f"exact_milp_daily_{tag}.parquet"
    solve_path = RESULTS / f"exact_milp_solves_{tag}.parquet"
    daily_rows = (pd.read_parquet(daily_path).to_dict("records")
                  if daily_path.exists() else [])
    solve_rows = (pd.read_parquet(solve_path).to_dict("records")
                  if solve_path.exists() else [])
    if args.rerun_gap_over is not None and solve_rows:
        solve_frame = pd.DataFrame(solve_rows)
        bad = set()
        requested_methods = {
            "exact_band_miqp_pf" if mode.strip() == "perfect_forecast"
            else "exact_band_miqp"
            for mode in args.modes.split(",") if mode.strip()
        }
        for keys, sub in solve_frame.groupby(["date", "method"]):
            if str(keys[1]) not in requested_methods:
                continue
            finite = sub.loc[np.isfinite(sub["gap"]), "gap"]
            if (not len(finite)
                    or float(finite.max()) > args.rerun_gap_over + 1e-9
                    or not bool(sub["has_solution"].all())):
                bad.add((str(keys[0]), str(keys[1])))
        if bad:
            daily_rows = [
                row for row in daily_rows
                if (str(row["date"]), str(row["method"])) not in bad
            ]
            solve_rows = [
                row for row in solve_rows
                if (str(row["date"]), str(row["method"])) not in bad
            ]
            atomic_parquet(pd.DataFrame(daily_rows), daily_path)
            atomic_parquet(pd.DataFrame(solve_rows), solve_path)
            print(f"scheduled {len(bad)} completed day/method pairs for "
                  f"gap refinement", flush=True)
    completed = {(str(r["date"]), r["method"]) for r in daily_rows}

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    pending = []
    for mode in modes:
        if mode not in {"deterministic", "perfect_forecast"}:
            raise ValueError(f"unknown mode: {mode}")
        oracle = mode == "perfect_forecast"
        method = "exact_band_miqp_pf" if oracle else "exact_band_miqp"
        for i, day in enumerate(days, 1):
            key = (str(day["date"]), method)
            if key in completed:
                continue
            pending.append((mode, args.scope, str(day["date"]), i,
                            len(days), args.threshold, args.fee, args.width,
                            args.time_limit, args.mip_gap, args.det_reopt,
                            args.pf_reopt))

    workers = max(1, min(int(args.workers), len(pending))) if pending else 1
    if workers == 1:
        solved = map(solve_day_task, pending)
        for daily_row, new_solves, message in solved:
            daily_rows.append(daily_row)
            solve_rows.extend(new_solves)
            completed.add((str(daily_row["date"]), daily_row["method"]))
            atomic_parquet(pd.DataFrame(daily_rows), daily_path)
            atomic_parquet(pd.DataFrame(solve_rows), solve_path)
            print(message, flush=True)
    elif pending:
        print(f"solving {len(pending)} day/method pairs with "
              f"{workers} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(solve_day_task, task) for task in pending]
            for future in as_completed(futures):
                daily_row, new_solves, message = future.result()
                daily_rows.append(daily_row)
                solve_rows.extend(new_solves)
                completed.add((str(daily_row["date"]), daily_row["method"]))
                atomic_parquet(pd.DataFrame(daily_rows), daily_path)
                atomic_parquet(pd.DataFrame(solve_rows), solve_path)
                print(message, flush=True)

    daily = pd.DataFrame(daily_rows)
    solves = pd.DataFrame(solve_rows)
    summary = {
        "scope": args.scope,
        "n_days": len(days),
        "tariff": {"threshold": args.threshold, "fee": args.fee,
                   "training_width": args.width},
        "perfect_forecast_reoptimization_hours": args.pf_reopt,
        "deterministic_reoptimization_hours": args.det_reopt,
        "deterministic_forecast": "OU conditional mean from current error",
        "horizon": "shrinks to the common end of each evaluation day",
        "solver_mip_gap_target": args.mip_gap,
        "solver_time_limit_s": args.time_limit,
        "methods": {},
    }
    metrics = ["total_cost", "common_cost", "bill", "capacity_fee",
               "degradation_cost", "terminal_penalty", "exceed_hours",
               "objective", "peak_import_kw", "efc", "throughput_kwh",
               "terminal_soc_dev", "soc_violations", "latency_mean_ms"]
    for method, sub in daily.groupby("method"):
        ss = solves[solves["method"] == method]
        finite_gap = ss.loc[np.isfinite(ss["gap"]), "gap"]
        summary["methods"][method] = {
            "n_days": int(len(sub)),
            **{m: float(sub[m].mean()) for m in metrics},
            "n_solves": int(len(ss)),
            "proven_optimal_fraction": float((ss["status"] == "optimal").mean()),
            "mean_solve_s": float(ss["solve_s"].mean()),
            "p95_solve_s": float(ss["solve_s"].quantile(0.95)),
            "max_gap": float(finite_gap.max()) if len(finite_gap) else None,
            "status_counts": {str(k): int(v) for k, v in
                              ss["status"].value_counts().items()},
        }
    summary_path = RESULTS / f"exact_milp_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
