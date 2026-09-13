"""Revision analysis for Reviewer 1, Comment 1.5 (and Reviewer 2.1/2.2).

Compares the submitted checkpoint-selection criterion (minimum
tuning-validation RMS residual) with the reviewer-proposed alternative
(acceptable residual threshold + lowest cost on independent validation
trajectories), using ONLY stored artifacts -- no retraining.

Part A (Experiment A, 2-state, FD ground truth):
  results/raw/expA/expA_metrics.json holds FD metrics (value_rmse,
  policy_rmse, cost_gap_eJ) for EVERY policy iteration of every seed of
  rvpinnpi and rvpinnpi_noadapt. We re-select per (method, seed) under
  the alternative criterion and compare realized FD cost gaps.

Part B (operational 3-state models, 8 regimes x 5 seeds):
  result.json per (regime, seed) stores per-iteration tuning RMS and the
  CRN validation rollout cost. Where the alternative criterion selects a
  different iteration, both checkpoints are evaluated on that regime's
  held-out test days and the daily-bill/objective difference is reported.

Output: results/rev/selection_expA.json, selection_operational.json,
        selection_operational_daily.parquet
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import (ROOT, OUT, remap,      # noqa: E402
                                    save_json, selection_thresholded)

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import torch                                                # noqa: E402

from src.evaluation import (NeuralController,               # noqa: E402
                            evaluate_method, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,           # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle,
                            seeds_of)
from src.policy_iteration import load_checkpoint            # noqa: E402
from src.regime_profiles import load_calibration            # noqa: E402

torch.set_default_dtype(torch.float64)
THETA = 1.25
THETA_GRID = [1.1, 1.25, 1.5, 2.0]


def alt_from_frame(g: pd.DataFrame, theta: float) -> int:
    rms = g["train_rms_residual"].values.astype(float)
    cost = g["rollout_cost"].values.astype(float)
    finite = np.isfinite(rms) & np.isfinite(cost)
    ok = finite & (rms <= theta * np.nanmin(rms[finite]))
    cand = np.where(ok)[0]
    return int(g["iteration"].values[cand[np.argmin(cost[cand])]])


def part_a() -> dict:
    m = pd.DataFrame(json.load(open(os.path.join(
        ROOT, "results", "raw", "expA", "expA_metrics.json"))))
    m = m[m.method.isin(["rvpinnpi", "rvpinnpi_noadapt"])]
    # drop the duplicate summary row (iteration = -1) that repeats the
    # selected checkpoint's metrics
    m = m[m.iteration >= 0]
    out = {"theta_primary": THETA, "runs": [], "theta_sensitivity": {}}
    for theta in THETA_GRID:
        rows = []
        for (meth, seed), g in m.groupby(["method", "seed"]):
            g = g.sort_values("iteration").reset_index(drop=True)
            # current criterion: argmin tuning-validation RMS residual
            gg = g[np.isfinite(g.train_rms_residual.astype(float))]
            cur = int(gg.iteration.values[
                np.argmin(gg.train_rms_residual.values)])
            alt = alt_from_frame(g, theta)
            gi = g.set_index("iteration")
            rows.append({
                "method": meth, "seed": int(seed),
                "iter_current": cur, "iter_alt": alt,
                "changed": alt != cur,
                "eJ_current": float(gi.loc[cur, "cost_gap_eJ"]),
                "eJ_alt": float(gi.loc[alt, "cost_gap_eJ"]),
                "value_rmse_current": float(gi.loc[cur, "value_rmse"]),
                "value_rmse_alt": float(gi.loc[alt, "value_rmse"]),
                "policy_rmse_current": float(gi.loc[cur, "policy_rmse"]),
                "policy_rmse_alt": float(gi.loc[alt, "policy_rmse"]),
                "rms_current": float(gi.loc[cur, "train_rms_residual"]),
                "rms_alt": float(gi.loc[alt, "train_rms_residual"]),
                "rollout_current": float(gi.loc[cur, "rollout_cost"]),
                "rollout_alt": float(gi.loc[alt, "rollout_cost"]),
            })
        df = pd.DataFrame(rows)
        agg = {
            "n_changed": int(df.changed.sum()),
            "n_runs": len(df),
            "mean_eJ_current": float(df.eJ_current.mean()),
            "mean_eJ_alt": float(df.eJ_alt.mean()),
            "median_eJ_current": float(df.eJ_current.median()),
            "median_eJ_alt": float(df.eJ_alt.median()),
        }
        # oracle: best possible e_J per run, for context
        if theta == THETA:
            oracle = m.groupby(["method", "seed"])["cost_gap_eJ"].min()
            agg["mean_eJ_oracle"] = float(oracle.mean())
            out["runs"] = rows
        out["theta_sensitivity"][str(theta)] = agg
    # per-method summary at the primary theta
    df = pd.DataFrame(out["runs"])
    out["by_method"] = {
        meth: {
            "mean_eJ_current": float(g.eJ_current.mean()),
            "mean_eJ_alt": float(g.eJ_alt.mean()),
            "mean_value_rmse_current": float(g.value_rmse_current.mean()),
            "mean_value_rmse_alt": float(g.value_rmse_alt.mean()),
            "mean_policy_rmse_current": float(g.policy_rmse_current.mean()),
            "mean_policy_rmse_alt": float(g.policy_rmse_alt.mean()),
        } for meth, g in df.groupby("method")}
    return out


def part_b() -> dict:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds_m = pick_seeds(cfg)
    regimes = pick_regimes(cfg, "primary")
    params, profs, _ = regime_bundle("primary", regimes, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal = load_calibration()
    all_days = load_region_days("primary", "test", cal)
    days_by_regime = {r: [d for d in all_days if d["regime"] == r]
                      for r in regimes}

    combos, daily_rows = [], []
    for reg in regimes:
        for ms in seeds_m:
            res_path = os.path.join(
                ckpt_dir("pinn_pi", "primary", reg, f"seed{ms}"),
                "result.json")
            if not os.path.exists(res_path):
                continue
            res = json.load(open(res_path))
            iters = res["iterations"]
            cur = int(res["selected_iteration"])
            alt = selection_thresholded(iters, THETA)
            rec = {"regime": reg, "seed": ms, "iter_current": cur,
                   "iter_alt": alt, "changed": alt != cur,
                   "rms_current":
                       iters[cur]["tuneval_policy_residual"]["rms"],
                   "rms_alt": iters[alt]["tuneval_policy_residual"]["rms"],
                   "rollout_current": iters[cur]["rollout_cost_crn"],
                   "rollout_alt": iters[alt]["rollout_cost_crn"],
                   "n_days": len(days_by_regime[reg])}
            if alt != cur:
                p, prof = params[reg], profs[reg]
                days = days_by_regime[reg]
                res_pair = {}
                for label, it in (("current", cur), ("alt", alt)):
                    model = load_checkpoint(
                        remap(iters[it]["checkpoint"]), p, device)
                    r = evaluate_method(
                        lambda _reg: NeuralController(model, p, prof),
                        days, {reg: p}, {reg: prof})
                    res_pair[label] = r
                    for d in r:
                        daily_rows.append({
                            "regime": reg, "seed": ms, "selection": label,
                            "iteration": it, "date": d["date"],
                            "bill": d["bill"], "objective": d["objective"],
                            "throughput_kwh": d["throughput_kwh"],
                            "soc_violations": d["soc_violations"]})
                rec["bill_current"] = float(np.mean(
                    [d["bill"] for d in res_pair["current"]]))
                rec["bill_alt"] = float(np.mean(
                    [d["bill"] for d in res_pair["alt"]]))
                rec["objective_current"] = float(np.mean(
                    [d["objective"] for d in res_pair["current"]]))
                rec["objective_alt"] = float(np.mean(
                    [d["objective"] for d in res_pair["alt"]]))
                print(f"{reg} seed{ms}: iter {cur}->{alt}, "
                      f"bill {rec['bill_current']:.2f}->{rec['bill_alt']:.2f}, "
                      f"obj {rec['objective_current']:.2f}->"
                      f"{rec['objective_alt']:.2f}", flush=True)
            combos.append(rec)

    if daily_rows:
        pd.DataFrame(daily_rows).to_parquet(
            os.path.join(OUT, "selection_operational_daily.parquet"))
    df = pd.DataFrame(combos)
    ch = df[df.changed]
    # overall test-set effect: day-weighted mean change over all combos
    # (unchanged combos contribute zero)
    tot_days = float((df.n_days * len(pick_seeds(cfg))).sum()) / len(seeds_m)
    if len(ch):
        w_bill = float((
            (ch.bill_alt - ch.bill_current) * ch.n_days).sum()
            / (df.n_days.sum()))
        w_obj = float((
            (ch.objective_alt - ch.objective_current) * ch.n_days).sum()
            / (df.n_days.sum()))
    else:
        w_bill = w_obj = 0.0
    return {"theta": THETA, "n_combos": len(df),
            "n_changed": int(df.changed.sum()),
            "weighted_mean_bill_change": w_bill,
            "weighted_mean_objective_change": w_obj,
            "total_regime_days": tot_days,
            "combos": combos}


def main() -> int:
    a = part_a()
    save_json("selection_expA.json", a)
    if os.environ.get("REV15_PART", "").lower() != "a":
        b = part_b()
        save_json("selection_operational.json", b)
    print("selection analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
