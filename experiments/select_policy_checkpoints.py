"""Freeze a training run by paired hard-cost historical validation."""
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
    parser.add_argument("run_dir")
    parser.add_argument("--region", default="confirmatory")
    parser.add_argument("--regime", default="winter_weekday")
    parser.add_argument("--threshold", type=float, default=300.0)
    parser.add_argument("--fee", type=float, default=40.0)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    params, profiles, _ = regime_bundle(args.region, [args.regime])
    p, prof = params[args.regime], profiles[args.regime]
    p.g_thr, p.c_step, p.w_step = args.threshold, args.fee, args.width
    days = [d for d in load_region_days(args.region, "val")
            if d["regime"] == args.regime][:args.max_days]
    if not days:
        raise RuntimeError("no validation days for requested regime")

    rows = []
    summaries = []
    for record in result["iterations"]:
        iteration = int(record["iteration"])
        checkpoint = str(record["checkpoint"])
        model = load_checkpoint(checkpoint, p, args.device)
        daily = evaluate_method(
            lambda _: NeuralController(model, p, prof), days,
            {args.regime: p}, {args.regime: prof})
        for row in daily:
            rows.append({"iteration": iteration, "checkpoint": checkpoint,
                         **row})
        summaries.append({
            "iteration": iteration,
            "checkpoint": checkpoint,
            "n_days": len(daily),
            "mean_common_cost": sum(x["common_cost"] for x in daily)
                                / len(daily),
            "mean_terminal_penalty": sum(x["terminal_penalty"] for x in daily)
                                     / len(daily),
        })
        print(summaries[-1], flush=True)

    incumbent = 0
    decisions = [{"candidate_iteration": summaries[0]["iteration"],
                  "decision": "initialize_incumbent"}]
    for candidate in range(1, len(summaries)):
        improvement = (summaries[incumbent]["mean_common_cost"]
                       - summaries[candidate]["mean_common_cost"])
        accept = improvement >= args.margin
        decisions.append({
            "candidate_iteration": summaries[candidate]["iteration"],
            "incumbent_before": summaries[incumbent]["iteration"],
            "paired_mean_improvement": improvement,
            "margin": args.margin,
            "decision": "accept" if accept else "retain_incumbent",
        })
        if accept:
            incumbent = candidate

    selected = summaries[incumbent]
    pd.DataFrame(rows).to_parquet(run_dir / "validation_daily.parquet",
                                  index=False)
    audit = {
        "deployment_version": DEPLOYMENT_VERSION,
        "action_search_version": ACTION_SEARCH_VERSION,
        "criterion": "paired historical validation mean of hard common cost",
        "partition": "validation",
        "region": args.region, "regime": args.regime,
        "n_days": len(days), "margin_eur_per_day": args.margin,
        "summaries": summaries, "decisions": decisions,
        "selected_iteration": selected["iteration"],
        "selected_checkpoint": selected["checkpoint"],
    }
    (run_dir / "validation_selection.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    result.setdefault("synthetic_crn_selected_iteration", result["selected_iteration"])
    result.setdefault("synthetic_crn_selected_checkpoint", result["selected_checkpoint"])
    result["selected_iteration"] = selected["iteration"]
    result["selected_checkpoint"] = selected["checkpoint"]
    result["selection_criterion"] = audit["criterion"]
    result["selection_partition"] = "validation"
    result["selection_n_days"] = len(days)
    result["selection_min_improvement_eur_per_day"] = args.margin
    result["selection_deployment_version"] = DEPLOYMENT_VERSION
    result["selection_action_search_version"] = ACTION_SEARCH_VERSION
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"selected iteration {selected['iteration']}: "
          f"{selected['mean_common_cost']:.3f} EUR/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
