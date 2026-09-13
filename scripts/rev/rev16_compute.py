"""Revision analysis for Reviewer 1, Comment 1.6: total computational cost.

Aggregates, per method:
  - total training wall-clock time (stored per-iteration wall_s for the
    PINN methods; reconstructed from checkpoint file timestamps for the
    SB3 DRL runs, which did not store wall time);
  - approximate number of model evaluations during training;
  - average online decision time (latency, from the immutable results);
  - hardware (recorded in the run manifest).

Output: results/rev/compute_accounting.json and an extended LaTeX table
        results/rev/table06_compute_costs_revised.tex
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT, OUT, save_json     # noqa: E402

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

CK = os.path.join(ROOT, "checkpoints")

# frozen budgets (configs/paper_full.yaml)
ADAM_STEPS = 6000
BATCH = 4096
PI_ITERS = 8
N_REGIMES = 8
N_SEEDS = 5
DRL_TRANSITIONS_MAX = 1_000_000
MPC_RESOLVES_PER_DAY = 24
N_TEST_DAYS = 360


def pinn_pi_training_seconds(tag: str) -> dict:
    """Sum train_info.wall_s over iterations for every (regime|seed) run."""
    tot, runs = 0.0, 0
    per_run = []
    for res_path in glob.glob(os.path.join(CK, tag, "primary", "*",
                                           "seed*", "result.json")):
        res = json.load(open(res_path))
        if "iterations" in res:
            w = sum(it["train_info"]["wall_s"] for it in res["iterations"])
        else:
            w = res.get("wall_s", 0.0)
        tot += w
        runs += 1
        per_run.append(w)
    return {"total_s": tot, "n_runs": runs,
            "mean_s_per_run": float(np.mean(per_run)) if per_run else None}


def drl_training_seconds(algo: str) -> dict:
    """Reconstruct wall time from checkpoint mtimes (50k-transition
    spacing); labeled as reconstructed in the manuscript."""
    tot, runs, trans = 0.0, 0, 0
    for seed_dir in glob.glob(os.path.join(CK, "drl", "primary", algo,
                                           "seed*")):
        cks = sorted(glob.glob(os.path.join(seed_dir, "ck_*.zip")),
                     key=lambda q: int(os.path.basename(q)[3:-4]))
        if len(cks) < 2:
            continue
        mt = [os.path.getmtime(c) for c in cks]
        gaps = np.diff(sorted(mt))
        interval = float(np.median(gaps))
        dur = (max(mt) - min(mt)) + interval   # add the first segment
        tot += dur
        runs += 1
        trans += int(os.path.basename(cks[-1])[3:-4])
    return {"total_s": tot, "n_runs": runs, "total_transitions": trans}


def main() -> int:
    acc = {"hardware": "NVIDIA GeForce RTX 4080 SUPER, AMD64 16-core CPU, "
                       "46.9 GB RAM, Windows 10; torch 2.4.1 + CUDA 12.4, "
                       "float64 PDE math"}

    # ---- training time ----
    rv = pinn_pi_training_seconds("pinn_pi")
    dh = pinn_pi_training_seconds("direct_hjb")
    sac = drl_training_seconds("sac")
    td3 = drl_training_seconds("td3")
    acc["training"] = {
        "rvpinnpi": rv, "direct_hjb": dh, "sac": sac, "td3": td3,
        "mpc": {"total_s": 0.0,
                "note": "no offline training; forecast profiles and "
                        "calibration shared by all methods"},
        "rules": {"total_s": 0.0,
                  "note": "threshold rule: validation grid search, "
                          "shared tuning cost"}}

    # ---- approximate model evaluations during training ----
    # One Adam step evaluates the residual on BATCH collocation points;
    # each residual needs one forward + one backward pass (autograd
    # derivatives). Reported as residual-point evaluations.
    pinn_evals_per_run = ADAM_STEPS * BATCH * PI_ITERS
    acc["model_evaluations"] = {
        "rvpinnpi": {"residual_point_evals_per_regime_seed":
                     pinn_evals_per_run,
                     "total": pinn_evals_per_run * rv["n_runs"]},
        "direct_hjb": {"residual_point_evals_per_regime_seed":
                       pinn_evals_per_run,
                       "total": pinn_evals_per_run * dh["n_runs"]},
        "sac": {"env_transitions_total": sac["total_transitions"],
                "gradient_updates": "~1 per transition (SB3 default)"},
        "td3": {"env_transitions_total": td3["total_transitions"],
                "gradient_updates": "~1 per transition (SB3 default)"},
        "mpc_deterministic": {"online_solves":
                              MPC_RESOLVES_PER_DAY * N_TEST_DAYS},
    }

    # ---- online decision time from the immutable results ----
    master = os.path.join(ROOT, "results", "immutable",
                          "results_master.parquet")
    df = pd.read_parquet(master)
    b = df[(df.experiment == "expB_operation")
           & (df.metric == "latency_mean_ms") & (df.is_primary)]
    lat = b.groupby("method")["value"].mean().to_dict()
    acc["latency_mean_ms"] = {k: float(v) for k, v in lat.items()}

    save_json("compute_accounting.json", acc)

    # ---- extended LaTeX table ----
    def hours(x):
        return f"{x/3600.0:.1f}"

    lat_f = {k: (f"{v:.2f}" if v >= 0.01 else f"{v:.1e}")
             for k, v in lat.items()}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Total computational cost of every method. Training is "
        r"the summed wall-clock time over all regime models and seeds on "
        r"a single NVIDIA RTX 4080 SUPER workstation (torch 2.4.1, CUDA "
        r"12.4, float64 PDE arithmetic). Model evaluations count residual-"
        r"point evaluations (PINN methods; one forward--backward pass per "
        r"point) or simulator transitions (DRL); they are not directly "
        r"equivalent notions of computational effort (Section~\ref{"
        r"sec:results}). DRL wall time is reconstructed from checkpoint "
        r"timestamps. Latency is the mean online decision time per "
        r"15-minute control step on the held-out test days.}",
        r"\label{tab:compute}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Train (h, total) & Runs & Model evals & "
        r"Latency (ms) \\",
        r"\midrule",
        f"RV--PINN--PI & {hours(rv['total_s'])} & {rv['n_runs']} & "
        f"$\\approx 2.0\\times10^{{8}}$/run & {lat_f.get('rvpinnpi','--')} \\\\",
        f"Direct HJB--PINN & {hours(dh['total_s'])} & {dh['n_runs']} & "
        f"$\\approx 2.0\\times10^{{8}}$/run & {lat_f.get('direct_hjb','--')} \\\\",
        f"SAC & {hours(sac['total_s'])} & {sac['n_runs']} & "
        f"{sac['total_transitions']/1e6:.1f}M transitions & "
        f"{lat_f.get('sac','--')} \\\\",
        f"TD3 & {hours(td3['total_s'])} & {td3['n_runs']} & "
        f"{td3['total_transitions']/1e6:.1f}M transitions & "
        f"{lat_f.get('td3','--')} \\\\",
        f"MPC (det.) & 0 & -- & {MPC_RESOLVES_PER_DAY*N_TEST_DAYS} online "
        f"solves & {lat_f.get('mpc_deterministic','--')} \\\\",
        f"MPC (perfect) & 0 & -- & {MPC_RESOLVES_PER_DAY*N_TEST_DAYS} "
        f"online solves & {lat_f.get('mpc_perfect_forecast','--')} \\\\",
        f"Threshold rule & 0 & -- & validation grid search & "
        f"{lat_f.get('threshold_rule','--')} \\\\",
        f"Self-consumption & 0 & -- & -- & "
        f"{lat_f.get('self_consumption','--')} \\\\",
        f"No storage & 0 & -- & -- & {lat_f.get('no_storage','--')} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex = os.path.join(OUT, "table06_compute_costs_revised.tex")
    with open(tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
