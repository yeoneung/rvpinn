"""Experiment B: chronological held-out real-data operation.

Evaluates every baseline and learned method on the SAME test days with the
same realized trajectories, observations, and safe projection. Saves raw
daily paired outcomes (long format) and representative-day trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                             # noqa: E402

from src.evaluation import (DRLController,               # noqa: E402
                            NeuralController, evaluate_method,
                            load_region_days, tune_threshold_rule)
from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle,
                            seeds_of)
from src.mpc import MPCController                        # noqa: E402
from src.policy_iteration import load_checkpoint         # noqa: E402
from src.regime_profiles import load_calibration         # noqa: E402
from src.reproducibility import (write_run_manifest,     # noqa: E402
                                 finalize_run_manifest)
from src.rules import NoStorage, PriceThreshold, SelfConsumption  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(ROOT, "results", "raw", "expB")


def neural_factory(tag, region, regime_params, profs, seed, device):
    def factory(reg):
        res_path = os.path.join(ckpt_dir(tag, region, reg, f"seed{seed}"),
                                "result.json")
        with open(res_path) as f:
            res = json.load(f)
        ck = res.get("selected_checkpoint") or res.get("checkpoint")
        model = load_checkpoint(ck, regime_params[reg], device)
        return NeuralController(model, regime_params[reg], profs[reg])
    return factory


def drl_factory(algo, region, regime_params, regimes, seed):
    from stable_baselines3 import SAC, TD3
    summary = os.path.join(ckpt_dir("drl", region, algo, f"seed{seed}"),
                           "training_summary.json")
    with open(summary) as f:
        sm = json.load(f)
    Algo = {"sac": SAC, "td3": TD3}[algo]
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)   # SB3 checkpoints are float32
    try:
        model = Algo.load(sm["selected"]["checkpoint"], device="cpu")
    finally:
        torch.set_default_dtype(prev_dtype)

    def factory(reg):
        return DRLController(model, regime_params[reg], regimes, reg)
    return factory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--region", default="primary")
    ap.add_argument("--part", default="test")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()

    cfg = load_stack(args.config)
    seeds = seeds_of(cfg)
    seeds_m = pick_seeds(cfg)
    regimes = pick_regimes(cfg, args.region)
    params, profs, B0 = regime_bundle(args.region, regimes, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)

    cal = load_calibration()
    days = [d for d in load_region_days(args.region, args.part, cal)
            if d["regime"] in params]
    if args.max_days:
        days = days[: args.max_days]
    n_smoke = cfg.get("n_test_days_subset")
    if cfg.get("smoke") and n_smoke:
        days = days[: int(n_smoke)]
    print(f"{len(days)} evaluation days ({args.part}, {args.region})")

    manifest = os.path.join(OUT, f"run_manifest{args.tag_suffix}.json")
    write_run_manifest(manifest, cfg, seeds,
                       extra={"n_days": len(days), "part": args.part})

    # threshold-rule tuning: training prices + validation days ONLY
    val_days = [d for d in load_region_days(args.region, "val", cal)
                if d["regime"] in params]
    if cfg.get("smoke"):
        val_days = val_days[:4]
    train_days = load_region_days(args.region, "train", cal)
    train_prices = np.concatenate([d["C"] for d in train_days])
    bl = cfg["threshold_rule"]
    c_lo, c_hi, tinfo = tune_threshold_rule(
        val_days, params, profs, train_prices,
        bl["lower_quantile_grid"], bl["upper_quantile_grid"])
    print(f"threshold rule tuned on validation: c_lo={c_lo:.4f} "
          f"c_hi={c_hi:.4f} ({tinfo})")

    method_list = args.methods.split(",") if args.methods != "all" else [
        "no_storage", "self_consumption", "threshold_rule",
        "mpc_deterministic", "mpc_perfect_forecast",
        "sac", "td3", "direct_hjb", "rvpinnpi"]

    rows = []

    def record(res_list, method, seed):
        for r in res_list:
            r2 = {k: v for k, v in r.items() if k != "traj"}
            r2.update({"method": method, "seed": seed,
                       "region": args.region, "part": args.part})
            rows.append(r2)

    for method in method_list:
        t0 = time.time()
        if method == "no_storage":
            record(evaluate_method(lambda reg: NoStorage(), days, params,
                                   profs), method, -1)
        elif method == "self_consumption":
            record(evaluate_method(lambda reg: SelfConsumption(), days,
                                   params, profs), method, -1)
        elif method == "threshold_rule":
            record(evaluate_method(
                lambda reg: PriceThreshold(params[reg], c_lo, c_hi),
                days, params, profs), method, -1)
        elif method == "mpc_deterministic":
            record(evaluate_method(
                lambda reg: MPCController(params[reg]), days, params, profs),
                method, -1)
        elif method == "mpc_perfect_forecast":
            record(evaluate_method(
                lambda reg: MPCController(params[reg], oracle=True),
                days, params, profs), method, -1)
        elif method in ("sac", "td3"):
            for ms in seeds_m:
                try:
                    fac = drl_factory(method, args.region, params, regimes, ms)
                except FileNotFoundError:
                    print(f"  missing {method} seed{ms}; skipped")
                    continue
                record(evaluate_method(fac, days, params, profs), method, ms)
        elif method in ("rvpinnpi", "direct_hjb"):
            tag = "pinn_pi" if method == "rvpinnpi" else "direct_hjb"
            for ms in seeds_m:
                try:
                    fac = neural_factory(tag, args.region, params, profs,
                                         ms, device)
                    record(evaluate_method(fac, days, params, profs),
                           method, ms)
                except FileNotFoundError:
                    print(f"  missing {tag} seed{ms}; skipped")
        print(f"  {method}: {time.time()-t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(
            OUT, f"daily_{args.region}_{args.part}{args.tag_suffix}.parquet"))

    with open(os.path.join(OUT, f"threshold_tuning{args.tag_suffix}.json"),
              "w") as f:
        json.dump({"c_lo": c_lo, "c_hi": c_hi, **tinfo}, f, indent=2)

    # ---- representative days for Fig. 5 (deterministic rule) ----
    df = pd.DataFrame(rows)
    rv = df[(df["method"] == "rvpinnpi") & (df["seed"] == seeds_m[0])]
    if len(rv) == 0:
        rv = df[df["method"] == df["method"].iloc[0]]
    traj_days = {}
    for season in ("winter", "summer"):
        sub = rv[rv["regime"].str.startswith(season)]
        if len(sub) == 0:
            continue
        med = sub["bill"].median()
        traj_days[season] = sub.iloc[(sub["bill"] - med).abs().argsort()
                                     ].iloc[0]["date"]
    if traj_days:
        sel_dates = set(traj_days.values())
        day_objs = [d for d in days if d["date"] in sel_dates]
        traj_out = {}
        traj_methods = {
            "rvpinnpi": lambda: neural_factory("pinn_pi", args.region,
                                               params, profs, seeds_m[0],
                                               device),
            "mpc_deterministic": lambda: (lambda reg:
                                          MPCController(params[reg])),
            "self_consumption": lambda: (lambda reg: SelfConsumption()),
            "sac": lambda: drl_factory("sac", args.region, params, regimes,
                                       seeds_m[0]),
        }
        for mname, fac_fn in traj_methods.items():
            try:
                fac = fac_fn()
                res = evaluate_method(fac, day_objs, params, profs,
                                      collect_traj_dates=sel_dates)
            except FileNotFoundError:
                print(f"  fig5: missing checkpoints for {mname}; skipped")
                continue
            for r in res:
                if "traj" in r:
                    for k, v in r["traj"].items():
                        traj_out[f"{mname}__{r['date']}__{k}"] = v
        np.savez_compressed(
            os.path.join(OUT, f"fig5_trajectories{args.tag_suffix}.npz"),
            **traj_out,
            **{f"meta__{s}": np.array([d]) for s, d in traj_days.items()})
    finalize_run_manifest(manifest)
    print("Experiment B evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
