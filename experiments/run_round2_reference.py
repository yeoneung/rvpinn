"""Evaluate the frozen no-learning reference-value controller."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import load_region_days, run_day  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.reference_policy import ReferenceValueController  # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "round2_reference_daily.parquet"


def main() -> int:
    regime = "winter_weekday"
    params, profiles, _ = regime_bundle("confirmatory", [regime])
    prof = profiles[regime]
    days = [day for day in load_region_days("confirmatory", "test")
            if day["regime"] == regime][:30]
    if len(days) != 30:
        raise RuntimeError(f"expected 30 test days, found {len(days)}")

    rows = []
    for fee in (20.0, 40.0, 80.0):
        p = params[regime]
        p.g_thr, p.c_step, p.w_step = 300.0, fee, 10.0
        controller = ReferenceValueController(p, prof, n_grid=1025)
        fee_rows = []
        for day in days:
            result = run_day(controller, day, p, prof)
            row = {
                "method": controller.name,
                "region": "confirmatory",
                "partition": "test",
                "threshold_kw": p.g_thr,
                "fee_per_hour": fee,
                "smoothing_width_kw": p.w_step,
                "n_grid": controller.n_grid,
                **result,
            }
            rows.append(row)
            fee_rows.append(row)
        pd.DataFrame(rows).to_parquet(OUT, index=False)
        mean_cost = pd.DataFrame(fee_rows)["common_cost"].mean()
        print(f"fee={fee:g} mean_reference_cost={mean_cost:.6f}",
              flush=True)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
