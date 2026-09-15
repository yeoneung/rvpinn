"""One-solve smoke test for the exact MIQP controller."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VER3 = HERE.parent
sys.path.insert(0, str(VER3))
sys.path.insert(0, str(HERE))

from src.evaluation import load_region_days  # noqa: E402
from src.exp_common import load_stack, regime_bundle  # noqa: E402
from exact_miqp import ExactMIQPController       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-hours", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--formulation", choices=("milp", "miqp"),
                    default="miqp")
    args = ap.parse_args()
    cfg = load_stack(str(VER3 / "configs" / "paper_smoke.yaml"))
    params, profiles, _ = regime_bundle(
        "primary", ["winter_weekday"], cfg)
    p = params["winter_weekday"]
    prof = profiles["winter_weekday"]
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    days = [d for d in load_region_days("primary", "test")
            if d["regime"] == "winter_weekday"][:1]
    day = days[0]
    n_hour = np.asarray(day["N"], dtype=float)
    c_hour = np.asarray(day["C"], dtype=float)

    def hourly(values, t):
        idx = np.clip(np.floor(np.asarray(t) % 24).astype(int), 0, 23)
        return values[idx]

    ctx = {
        "N_of_t": lambda t: hourly(n_hour, t),
        "C_of_t": lambda t: hourly(c_hour, t),
        "N_forecast": lambda t: prof.n_bar(np.asarray(t, dtype=float)),
        "C_forecast": lambda t: prof.c_bar(np.asarray(t, dtype=float)),
    }
    controller = ExactMIQPController(
        p, time_limit_s=args.time_limit, mip_gap=1e-4, random_seed=0,
        formulation=args.formulation)
    hours = range(24) if args.all_hours else (0,)
    action = 0.0
    for hour in hours:
        plan = controller._solve_plan(float(hour), p.s0, ctx)
        action = float(plan[0])
        rec = controller.solve_records[-1]
        print("hour", hour, "action_kw", action, "status", rec.status,
              "gap", rec.gap, "solve_s", rec.solve_s, flush=True)
    summary = controller.solver_summary
    print("status", summary["status_counts"])
    print("gap", summary["max_gap"])
    print("solve_s", summary["mean_solve_s"])
    if summary["n_with_solution"] != len(tuple(hours)):
        return 2
    if not np.isfinite(action) or action < -p.a_c or action > p.a_d:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
