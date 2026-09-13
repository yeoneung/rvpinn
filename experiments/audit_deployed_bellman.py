"""Finite deployment-aligned Bellman audit for selected central-fee policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
sys.path.insert(0, str(ROOT))

from src.deployed_policy import (deployed_one_step_action,
                                 deployed_one_step_q)  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402


def scalar_value(model, t, s, y, pz) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        value = model(
            torch.tensor([t], dtype=torch.float64, device=device),
            torch.tensor([s], dtype=torch.float64, device=device),
            torch.tensor([y], dtype=torch.float64, device=device),
            torch.tensor([pz], dtype=torch.float64, device=device),
        )
    return float(value.detach().cpu().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v3_winter_fee40")
    parser.add_argument("--states-per-time", type=int, default=4)
    parser.add_argument("--deployed-grid", type=int, default=1025)
    parser.add_argument("--reference-grid", type=int, default=4097)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_path = RESULTS / f"restart_batches_{args.tag}.json"
    batches = json.loads(batch_path.read_text(encoding="utf-8"))
    selected = [(int(x["batch"]), int(x["selected_seed"]))
                for x in batches["batch_selections"]
                if x.get("selected_seed") is not None]
    skipped_fallback_batches = [int(x["batch"])
                                for x in batches["batch_selections"]
                                if x.get("selected_seed") is None]
    if not selected:
        raise RuntimeError("no trained central-fee policy selected")
    params, profiles, _ = regime_bundle(
        "confirmatory", ["winter_weekday"])
    p, prof = params["winter_weekday"], profiles["winter_weekday"]
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    rng = np.random.default_rng(args.seed)
    rows = []
    summaries = []

    for batch, selected_seed in selected:
        run_dir = (ROOT / "checkpoints" / args.tag / "confirmatory"
                   / "winter_weekday" / f"seed{selected_seed}")
        result = json.loads((run_dir / "result.json").read_text(
            encoding="utf-8"))
        model = load_checkpoint(result["selected_checkpoint"], p, args.device)
        for k in range(int(round(p.T / p.dt_ctrl))):
            t = k * p.dt_ctrl
            residuals, defects = [], []
            grid_defects, grid_action_diffs = [], []
            reference_action_diffs, quadrature_diffs = [], []
            for _ in range(args.states_per_time):
                s = float(rng.uniform(p.s_min, p.s_max))
                y = float(rng.uniform(p.y_min, p.y_max))
                pz = float(rng.uniform(p.p_min, p.p_max))
                a_dep, q_dep = deployed_one_step_action(
                    model, t, s, y, pz, prof, n_grid=args.deployed_grid)
                a_dense_same_q, q_dense_same_q = deployed_one_step_action(
                    model, t, s, y, pz, prof, n_grid=args.reference_grid,
                    quadrature_order=3)
                a_ref, q_ref = deployed_one_step_action(
                    model, t, s, y, pz, prof, n_grid=args.reference_grid,
                    quadrature_order=7)
                q_dep_reference = float(deployed_one_step_q(
                    model, t, s, y, pz, prof, [a_dep],
                    quadrature_order=7)[0])
                residuals.append(q_ref - scalar_value(model, t, s, y, pz))
                defects.append(max(0.0, q_dep_reference - q_ref))
                grid_defects.append(max(0.0, q_dep - q_dense_same_q))
                grid_action_diffs.append(abs(a_dep - a_dense_same_q))
                reference_action_diffs.append(abs(a_dep - a_ref))
                quadrature_diffs.append(abs(q_dep - q_dep_reference))
            rows.append({
                "batch": batch, "selected_seed": selected_seed,
                "time_index": k, "time_h": t,
                "n_states": args.states_per_time,
                "residual_min": float(np.min(residuals)),
                "residual_max": float(np.max(residuals)),
                "residual_oscillation": float(np.ptp(residuals)),
                "residual_abs_mean": float(np.mean(np.abs(residuals))),
                "deployed_action_defect_max": float(np.max(defects)),
                "deployed_action_defect_mean": float(np.mean(defects)),
                "same_quadrature_grid_defect_max": float(
                    np.max(grid_defects)),
                "same_quadrature_action_difference_max_kw": float(
                    np.max(grid_action_diffs)),
                "reference_action_difference_max_kw": float(
                    np.max(reference_action_diffs)),
                "deployed_action_quadrature_difference_max": float(
                    np.max(quadrature_diffs)),
            })

        terminal_errors = []
        for _ in range(128):
            s = float(rng.uniform(p.s_min, p.s_max))
            y = float(rng.uniform(p.y_min, p.y_max))
            pz = float(rng.uniform(p.p_min, p.p_max))
            terminal_errors.append(abs(
                scalar_value(model, p.T, s, y, pz)
                - p.lam_T * (s - p.s_tar) ** 2))
        batch_rows = [row for row in rows if row["batch"] == batch]
        terminal_error = float(np.max(terminal_errors))
        residual_sum = float(sum(row["residual_oscillation"]
                                 for row in batch_rows))
        defect_sum = float(sum(row["deployed_action_defect_max"]
                               for row in batch_rows))
        summaries.append({
            "batch": batch, "selected_seed": selected_seed,
            "n_time_indices": len(batch_rows),
            "states_per_time": args.states_per_time,
            "deployed_grid": args.deployed_grid,
            "reference_grid": args.reference_grid,
            "terminal_error_max": terminal_error,
            "sum_sampled_residual_oscillation": residual_sum,
            "sum_sampled_action_defect": defect_sum,
            "sum_same_quadrature_grid_defect": float(sum(
                row["same_quadrature_grid_defect_max"]
                for row in batch_rows)),
            "sampled_bound_rhs": 2.0 * terminal_error
                                 + residual_sum + defect_sum,
            "max_same_quadrature_action_difference_kw": max(
                row["same_quadrature_action_difference_max_kw"]
                for row in batch_rows),
            "max_reference_action_difference_kw": max(
                row["reference_action_difference_max_kw"]
                for row in batch_rows),
            "max_sampled_quadrature_difference": max(
                row["deployed_action_quadrature_difference_max"]
                for row in batch_rows),
        })

    pd.DataFrame(rows).to_parquet(
        RESULTS / "deployed_bellman_audit_by_time.parquet", index=False)
    summary_path = RESULTS / "deployed_bellman_audit.json"
    summary_path.write_text(json.dumps({
        "label": "finite uniform-state audit, not a continuous-state bound",
        "seed": args.seed,
        "skipped_zero_action_fallback_batches": skipped_fallback_batches,
        "batches": summaries,
    }, indent=2), encoding="utf-8")

    mean_bound = float(np.mean([x["sampled_bound_rhs"] for x in summaries]))
    max_defect = float(max(x["sum_sampled_action_defect"]
                           for x in summaries))
    max_grid_defect = float(max(x["sum_same_quadrature_grid_defect"]
                                for x in summaries))
    max_grid_action = float(max(
        x["max_same_quadrature_action_difference_kw"] for x in summaries))
    max_reference_action = float(max(
        x["max_reference_action_difference_kw"] for x in summaries))
    max_quadrature = float(max(x["max_sampled_quadrature_difference"]
                               for x in summaries))
    tex = (
        "Using seven nodes per disturbance dimension as a quadrature reference, "
        "the finite uniform-state audit produced a mean plug-in sum of "
        f"{mean_bound:.1f} EUR for the terms in "
        f"\\eqref{{eq:deployed-bound}}. "
        f"The largest accumulated deployed-action defect was {max_defect:.3f} "
        f"EUR against the combined dense-grid, seven-node reference. With "
        f"the deployed 3-by-3 quadrature held fixed, the largest accumulated "
        f"grid-only defect was {max_grid_defect:.3f} EUR and the largest "
        f"action difference between the 1,025- and 4,097-point searches was "
        f"{max_grid_action:.2f} kW. After also changing the quadrature order, "
        f"the largest reference-action difference was "
        f"{max_reference_action:.2f} kW. The largest sampled "
        f"change in the deployed action's one-step value between 3-by-3 and "
        f"7-by-7 quadrature was {max_quadrature:.3f} EUR. These sampled "
        "quantities diagnose the implementation; they neither bound the "
        "quadrature error uniformly nor replace continuous-state suprema."
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "bellman_audit.tex").write_text(tex, encoding="utf-8")
    print(summary_path)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
