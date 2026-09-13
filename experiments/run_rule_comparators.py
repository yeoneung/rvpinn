"""Tune and evaluate a simple price-threshold rule under the common objective."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import (evaluate_method, load_region_days,  # noqa: E402
                            tune_threshold_rule)
from src.exp_common import regime_bundle  # noqa: E402
from src.rules import NoStorage, PriceThreshold  # noqa: E402

RESULTS = HERE / "results"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="confirmatory")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--fee", type=float, required=True)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--max-days", type=int, default=30)
    args = parser.parse_args()

    params, profiles, _ = regime_bundle(args.region, [args.regime])
    p, prof = params[args.regime], profiles[args.regime]
    p.g_thr, p.c_step, p.w_step = args.threshold, args.fee, args.width
    train_days = load_region_days(args.region, "train")
    train_prices = np.concatenate([day["C"] for day in train_days])
    val_days = [day for day in load_region_days(args.region, "val")
                if day["regime"] == args.regime][:args.max_days]
    test_days = [day for day in load_region_days(args.region, "test")
                 if day["regime"] == args.regime][:args.max_days]
    c_lo, c_hi, tuning = tune_threshold_rule(
        val_days, params, profiles, train_prices,
        (0.1, 0.2, 0.3), (0.7, 0.8, 0.9))
    controller_factory = (
        (lambda _: NoStorage()) if tuning["selected_kind"] == "zero_action"
        else (lambda _: PriceThreshold(p, c_lo, c_hi)))
    evaluated = evaluate_method(
        controller_factory, test_days, params, profiles)

    tag = (f"{args.region}_thr{args.threshold:g}_fee{args.fee:g}"
           f"_w{args.width:g}_n2")
    path = RESULTS / f"common_daily_{tag}.parquet"
    existing = pd.read_parquet(path)
    existing = existing[
        ~((existing["method"] == "validation_tuned_price_rule")
          & (existing["regime"] == args.regime))]
    rows = pd.DataFrame([{
        "method": "validation_tuned_price_rule",
        "region": args.region,
        "threshold_kw": args.threshold,
        "fee_per_hour": args.fee,
        "smoothing_width_kw": args.width,
        "control_dt_min": 60.0 * p.dt_ctrl,
        "solver_time_limit_s": 0.0,
        "n_scenarios": 0,
        "surrogate_slope_eur_per_kwh": 0.0,
        **row,
    } for row in evaluated])
    atomic_parquet(pd.concat([existing, rows], ignore_index=True), path)
    tuning.update({
        "region": args.region, "regime": args.regime,
        "partition": "validation", "n_validation_days": len(val_days),
        "threshold_kw": args.threshold, "fee_per_hour": args.fee,
        "smoothing_width_kw": args.width, "c_lo": c_lo, "c_hi": c_hi,
        "test_mean_common_cost": float(rows["common_cost"].mean()),
    })
    tune_path = RESULTS / (
        f"rule_tuning_{args.regime}_thr{args.threshold:g}"
        f"_fee{args.fee:g}_w{args.width:g}.json")
    tune_path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    print(f"{path}: rule mean {rows['common_cost'].mean():.3f}")
    print(tune_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
