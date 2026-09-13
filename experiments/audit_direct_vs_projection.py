"""Post-selection ablation of direct deployed-set search versus projection."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
sys.path.insert(0, str(ROOT))

from src.deployed_policy import projected_unrestricted_one_step_action  # noqa: E402
from src.evaluation import load_region_days, run_day  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402


class ProjectAfterOptimizationController:
    def __init__(self, model, profile):
        self.model = model
        self.profile = profile
        self.reset()

    def reset(self):
        self.n_calls = 0
        self.n_projected = 0
        self.defects = []

    def __call__(self, t, s, y, pz, ctx):
        action, raw_action, defect = projected_unrestricted_one_step_action(
            self.model, t, s, y, pz, self.profile,
            current_net=ctx.get("N_now"), current_price=ctx.get("C_now"))
        self.n_calls += 1
        self.n_projected += int(abs(action - raw_action) > 1e-9)
        self.defects.append(float(defect))
        return action


def main() -> int:
    tag = "v3_winter_fee40"
    audit = json.loads((RESULTS / f"restart_batches_{tag}.json")
                       .read_text(encoding="utf-8"))
    selected = [entry for entry in audit["batch_selections"]
                if entry.get("selected_seed") is not None]
    if not selected:
        raise RuntimeError("no trained central-fee policy selected")
    restart_by_seed = {int(row["seed"]): row for row in audit["restarts"]}
    params, profiles, _ = regime_bundle("confirmatory", ["winter_weekday"])
    p, profile = params["winter_weekday"], profiles["winter_weekday"]
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    days = [day for day in load_region_days("confirmatory", "test")
            if day["regime"] == "winter_weekday"][:30]

    rows = []
    for entry in selected:
        batch, seed = int(entry["batch"]), int(entry["selected_seed"])
        checkpoint = restart_by_seed[seed]["checkpoint"]
        model = load_checkpoint(checkpoint, p, "cuda")
        controller = ProjectAfterOptimizationController(model, profile)
        for day in days:
            result = run_day(controller, day, p, profile)
            rows.append({
                "batch": batch, "selected_seed": seed,
                "projection_fraction": controller.n_projected
                                       / max(controller.n_calls, 1),
                "mean_one_step_greedy_defect": float(
                    np.mean(controller.defects)),
                "max_one_step_greedy_defect": float(
                    np.max(controller.defects)),
                **result,
            })
    projected = pd.DataFrame(rows)
    projected.to_parquet(
        RESULTS / "direct_vs_projection_daily.parquet", index=False)

    direct_all = pd.read_parquet(
        RESULTS / f"learned_daily_{tag}.parquet")
    direct = direct_all[direct_all["selected_by_batch"]][
        ["batch", "date", "common_cost"]]
    paired = projected.merge(
        direct, on=["batch", "date"], suffixes=("_projected", "_direct"),
        validate="one_to_one")
    paired["projected_minus_direct"] = (
        paired["common_cost_projected"] - paired["common_cost_direct"])
    summary = {
        "label": "post-selection deployment-repair ablation",
        "n_batches": int(paired["batch"].nunique()),
        "n_days_per_batch": int(paired.groupby("batch").size().min()),
        "mean_direct_common_cost": float(
            paired["common_cost_direct"].mean()),
        "mean_projected_common_cost": float(
            paired["common_cost_projected"].mean()),
        "mean_projected_minus_direct": float(
            paired["projected_minus_direct"].mean()),
        "projection_fraction": float(projected["projection_fraction"].mean()),
        "mean_one_step_greedy_defect": float(
            projected["mean_one_step_greedy_defect"].mean()),
        "max_one_step_greedy_defect": float(
            projected["max_one_step_greedy_defect"].max()),
    }
    (RESULTS / "direct_vs_projection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    GENERATED.mkdir(parents=True, exist_ok=True)
    tex = (
        "In a post-selection ablation on the five central-fee batch policies, "
        f"optimizing first on the larger continuous-time action set and then "
        f"projecting changed {100.0 * summary['projection_fraction']:.1f}\\% "
        f"of held actions. Its mean common cost was "
        f"{summary['mean_projected_minus_direct']:+.1f} EUR/day relative to "
        f"direct deployed-set minimization, and its mean one-step greedy "
        f"defect was {summary['mean_one_step_greedy_defect']:.3f} EUR. "
        "This diagnostic was not used for policy selection."
    )
    (GENERATED / "deployment_repair_audit.tex").write_text(
        tex, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
