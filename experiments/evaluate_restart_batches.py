"""Evaluate complete five-restart learned-policy batches on held-out days."""
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


class ZeroController:
    def __call__(self, t, s, y, pz, ctx):
        return 0.0


def select_batch_against_fallback(candidates, fallback_cost, margin):
    """Sequential validation selector with zero action as incumbent."""
    incumbent = None
    incumbent_cost = float(fallback_cost)
    decisions = []
    for candidate_index, candidate in enumerate(candidates):
        improvement = (incumbent_cost
                       - candidate["validation_mean_common_cost"])
        accept = improvement >= margin
        decisions.append({
            "candidate_seed": candidate["seed"],
            "incumbent_before": ("zero_action" if incumbent is None
                                  else candidates[incumbent]["seed"]),
            "improvement_eur_per_day": improvement,
            "decision": "accept" if accept else "retain_incumbent",
        })
        if accept:
            incumbent = candidate_index
            incumbent_cost = candidate["validation_mean_common_cost"]
    chosen = candidates[incumbent] if incumbent is not None else None
    return chosen, incumbent_cost, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--region", default="confirmatory")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--fee", type=float, required=True)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    params, profiles, _ = regime_bundle(args.region, [args.regime])
    p, prof = params[args.regime], profiles[args.regime]
    p.g_thr, p.c_step, p.w_step = args.threshold, args.fee, args.width
    days = [d for d in load_region_days(args.region, "test")
            if d["regime"] == args.regime][:args.max_days]
    validation_days = [d for d in load_region_days(args.region, "val")
                       if d["regime"] == args.regime][:args.max_days]
    if not days:
        raise RuntimeError("no held-out days")
    if not validation_days:
        raise RuntimeError("no validation days")
    fallback_validation = evaluate_method(
        lambda _: ZeroController(), validation_days,
        {args.regime: p}, {args.regime: prof})
    fallback_validation_mean = float(sum(
        row["common_cost"] for row in fallback_validation)
        / len(fallback_validation))

    restarts = []
    for seed in seeds:
        run_dir = (ROOT / "checkpoints" / args.tag / args.region
                   / args.regime / f"seed{seed}")
        result = json.loads((run_dir / "result.json").read_text(
            encoding="utf-8"))
        selection = json.loads((run_dir / "validation_selection.json").read_text(
            encoding="utf-8"))
        if selection.get("deployment_version") != DEPLOYMENT_VERSION:
            raise RuntimeError(f"rerun historical validation before test evaluation: {run_dir}")
        if selection.get("action_search_version") != ACTION_SEARCH_VERSION:
            raise RuntimeError(f"rerun validation with the closed candidate grid: {run_dir}")
        mean_val = next(
            x["mean_common_cost"] for x in selection["summaries"]
            if int(x["iteration"]) == int(selection["selected_iteration"]))
        restarts.append({
            "seed": seed,
            "batch": seed // 5,
            "checkpoint": result["selected_checkpoint"],
            "selected_iteration": result["selected_iteration"],
            "validation_mean_common_cost": mean_val,
        })

    batch_selections = []
    for batch in sorted({x["batch"] for x in restarts}):
        candidates = [x for x in restarts if x["batch"] == batch]
        if len(candidates) != 5:
            raise RuntimeError(f"batch {batch} has {len(candidates)} restarts")
        chosen, incumbent_cost, decisions = select_batch_against_fallback(
            candidates, fallback_validation_mean, args.margin)
        batch_selections.append({
            "batch": batch,
            "selected_kind": ("trained_policy" if chosen is not None
                               else "zero_action_fallback"),
            "selected_seed": (chosen["seed"] if chosen is not None else None),
            "selected_validation_mean_common_cost": incumbent_cost,
            "decisions": decisions,
        })

    selected_by_batch = {x["batch"]: x["selected_seed"]
                         for x in batch_selections}
    rows = []
    for restart in restarts:
        model = load_checkpoint(restart["checkpoint"], p, args.device)
        evaluated = evaluate_method(
            lambda _: NeuralController(model, p, prof), days,
            {args.regime: p}, {args.regime: prof})
        for row in evaluated:
            rows.append({
                "method": "offline_value_policy",
                "tag": args.tag, "region": args.region,
                "regime_model": args.regime,
                "seed": restart["seed"], "batch": restart["batch"],
                "is_fallback": False,
                "selected_by_batch": (
                    restart["seed"] == selected_by_batch[restart["batch"]]),
                "validation_mean_common_cost":
                    restart["validation_mean_common_cost"],
                "threshold_kw": args.threshold,
                "fee_per_hour": args.fee,
                "smoothing_width_kw": args.width,
                **row,
            })
        print(f"seed {restart['seed']} test mean "
              f"{sum(x['common_cost'] for x in evaluated)/len(evaluated):.3f}",
              flush=True)

    fallback_batches = [x["batch"] for x in batch_selections
                        if x["selected_kind"] == "zero_action_fallback"]
    if fallback_batches:
        fallback_test = evaluate_method(
            lambda _: ZeroController(), days,
            {args.regime: p}, {args.regime: prof})
        for batch in fallback_batches:
            for row in fallback_test:
                rows.append({
                    "method": "offline_value_policy",
                    "tag": args.tag, "region": args.region,
                    "regime_model": args.regime,
                    "seed": -(batch + 1), "batch": batch,
                    "is_fallback": True, "selected_by_batch": True,
                    "validation_mean_common_cost": fallback_validation_mean,
                    "threshold_kw": args.threshold,
                    "fee_per_hour": args.fee,
                    "smoothing_width_kw": args.width,
                    **row,
                })
            print(f"batch {batch} selected zero-action fallback; test mean "
                  f"{sum(x['common_cost'] for x in fallback_test)/len(fallback_test):.3f}",
                  flush=True)

    out = HERE / "results" / f"learned_daily_{args.tag}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    audit = {
        "deployment_version": DEPLOYMENT_VERSION,
        "action_search_version": ACTION_SEARCH_VERSION,
        "tag": args.tag, "region": args.region, "regime": args.regime,
        "partition": "test", "n_days": len(days),
        "restart_margin_eur_per_day": args.margin,
        "fallback": "zero_action",
        "fallback_validation_mean_common_cost": fallback_validation_mean,
        "restarts": restarts, "batch_selections": batch_selections,
        "daily_file": str(out),
    }
    audit_path = HERE / "results" / f"restart_batches_{args.tag}.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(out)
    print(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
