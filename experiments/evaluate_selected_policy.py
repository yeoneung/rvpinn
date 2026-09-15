"""Evaluate one validation-selected restart on a held-out regime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (NeuralController, evaluate_method,
                            load_region_days)  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402
from src.deployment_numerics import DEPLOYMENT_VERSION  # noqa: E402
from src.deployed_policy import ACTION_SEARCH_VERSION  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--region", default="confirmatory")
    parser.add_argument("--regime", default="winter_weekday")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--fee", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = (ROOT / "checkpoints" / args.tag / args.region
               / args.regime / f"seed{args.seed}")
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("selection_deployment_version") != DEPLOYMENT_VERSION:
        raise RuntimeError(f"rerun historical validation before test evaluation: {run_dir}")
    if result.get("selection_action_search_version") != ACTION_SEARCH_VERSION:
        raise RuntimeError(f"rerun validation with the closed candidate grid: {run_dir}")
    params, profiles, _ = regime_bundle(args.region, [args.regime])
    p, prof = params[args.regime], profiles[args.regime]
    p.g_thr, p.c_step, p.w_step = args.threshold, args.fee, args.width
    days = [d for d in load_region_days(args.region, "test")
            if d["regime"] == args.regime][:args.max_days]
    model = load_checkpoint(result["selected_checkpoint"], p, args.device)
    rows = evaluate_method(
        lambda _: NeuralController(model, p, prof), days,
        {args.regime: p}, {args.regime: prof})
    frame = pd.DataFrame([{
        "method": "offline_value_policy", "tag": args.tag,
        "region": args.region, "regime_model": args.regime,
        "seed": args.seed, "threshold_kw": args.threshold,
        "fee_per_hour": args.fee, "smoothing_width_kw": args.width,
        **row,
    } for row in rows])
    out = HERE / "results" / f"selected_daily_{args.tag}.parquet"
    frame.to_parquet(out, index=False)
    print(f"{out}: mean common cost {frame['common_cost'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
