"""Short-horizon solve check for the two-stage stochastic comparator."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.exp_common import regime_bundle  # noqa: E402
from stochastic_miqp import ScenarioExactMIPController  # noqa: E402


def main() -> int:
    params, profiles, _ = regime_bundle(
        "primary", ["winter_weekday"])
    p, profile = params["winter_weekday"], profiles["winter_weekday"]
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    ctx = {
        "date": "2019-01-03",
        "N_forecast": lambda t: profile.n_bar(np.asarray(t)),
        "C_forecast": lambda t: profile.c_bar(np.asarray(t)),
    }
    controller = ScenarioExactMIPController(
        p, n_scenarios=4, reopt_every_hours=p.dt_ctrl,
        time_limit_s=10.0, mip_gap=0.01, random_seed=4101)
    plan = controller._solve_plan(23.0, p.s0, ctx, 0.0, 0.0)
    record = controller.solve_records[-1]
    print({"action_kw": float(plan[0]), "status": record.status,
           "gap": record.gap, "solve_s": record.solve_s})
    return 0 if record.has_solution and np.isfinite(plan[0]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
