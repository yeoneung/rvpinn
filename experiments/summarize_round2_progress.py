"""Print completion counts for restartable round-2 scenario runs."""
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "results"


def main() -> int:
    pattern = "common_daily_*_r2_stream*_pool16_tl*.parquet"
    for path in sorted(RESULTS.glob(pattern)):
        frame = pd.read_parquet(path)
        solve_path = path.with_name(path.name.replace("common_daily_",
                                                      "common_solves_"))
        diagnostics = ""
        if solve_path.exists():
            solves = pd.read_parquet(solve_path)
            finite = solves[
                np.isfinite(solves["gap"])
                & np.isfinite(solves["primal_bound"])
                & np.isfinite(solves["dual_bound"])
                & (solves["gap"] < 1e19)
            ]
            mean_gap = finite["gap"].mean() if len(finite) else np.nan
            diagnostics = (
                f", TL={(solves['status'] == 'timelimit').mean():.1%}, "
                f"finite_gap={len(finite) / len(solves):.1%}, "
                f"mean_gap={mean_gap:.3f}"
            )
        print(
            f"{path.name}: days={frame['date'].nunique():2d}/30, "
            f"mean_cost={frame['common_cost'].mean():.1f}{diagnostics}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
