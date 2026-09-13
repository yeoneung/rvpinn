"""Build the immutable long-format results master from all raw outputs.

results/immutable/results_master.parquet + run_manifest.json (with SHA-256).
Figure/table scripts refuse to run when the hash mismatches unless
--allow-new-results is passed to them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import config_hash, load_config          # noqa: E402
from src.reproducibility import environment_info, file_sha256  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
IMM = os.path.join(ROOT, "results", "immutable")

ID_COLS = ["run_id", "experiment", "scenario", "region", "split", "date",
           "regime", "method", "method_seed", "training_seed", "checkpoint",
           "policy_iteration", "architecture", "metric", "value", "unit",
           "is_primary", "is_test", "source_file", "config_hash",
           "data_hash", "timestamp"]

UNITS = {
    "bill": "currency", "objective": "currency", "peak_import_kw": "kW",
    "export_kwh": "kWh", "curtailment_kwh": "kWh", "throughput_kwh": "kWh",
    "efc": "cycles", "terminal_soc": "-", "terminal_soc_dev": "-",
    "soc_violations": "count", "max_violation": "-",
    "projection_events": "count", "reflection_events": "count",
    "latency_mean_ms": "ms", "latency_median_ms": "ms",
    "latency_p95_ms": "ms",
    "value_linf": "currency", "value_rmse": "currency",
    "policy_rmse": "kW", "cost_gap_eJ": "currency",
    "train_rms_residual": "currency/h", "val_max_residual": "currency/h",
    "hjb_max_residual": "currency/h", "greedy_error": "currency/h",
    "policy_change": "-", "rollout_cost": "currency", "wall_s": "s",
    "peak_mem_mb": "MB", "cert_policy_max": "currency/h",
    "cert_hjb_max": "currency/h", "cert_lip_estimate": "currency/h",
    "e_T": "currency", "delta_greedy_mean": "currency/h",
    "delta_greedy_max": "currency/h", "action_rmse": "kW",
}

PRIMARY_METRICS = {"bill", "peak_import_kw", "efc", "terminal_soc_dev",
                   "soc_violations", "latency_mean_ms", "value_linf",
                   "value_rmse", "policy_rmse", "cost_gap_eJ"}


def melt_rows(df: pd.DataFrame, experiment: str, base: dict) -> list:
    """Wide rows -> long metric rows."""
    out = []
    skip = {"method", "seed", "date", "regime", "region", "part", "scenario",
            "scenario_value", "variant", "ablation", "iteration",
            "is_selected", "is_final", "selected_iteration", "spike_level"}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for _, r in df.iterrows():
        for col, val in r.items():
            if col in skip or val is None:
                continue
            if isinstance(val, str):
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            scen = r.get("scenario", experiment)
            if "variant" in r and pd.notna(r.get("variant")):
                scen = f"{r.get('ablation', '')}:{r['variant']}"
            out.append({
                **base,
                "experiment": experiment,
                "scenario": str(scen),
                "date": str(r.get("date", "")),
                "regime": str(r.get("regime", "")),
                "method": str(r.get("method", r.get("variant", ""))),
                "method_seed": int(r.get("seed", -1))
                if pd.notna(r.get("seed", -1)) else -1,
                "policy_iteration": int(r.get("iteration", -1))
                if pd.notna(r.get("iteration", -1)) else -1,
                "metric": col,
                "value": v,
                "unit": UNITS.get(col, "-"),
                "is_primary": col in PRIMARY_METRICS,
                "scenario_value": float(r.get("scenario_value", np.nan))
                if pd.notna(r.get("scenario_value", np.nan)) else np.nan,
                "timestamp": now,
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(*([args.config] if args.config else []))
    chash = config_hash(cfg)
    os.makedirs(IMM, exist_ok=True)

    dm_path = os.path.join(ROOT, "data", "data_manifest.json")
    data_hash = ""
    if os.path.exists(dm_path):
        with open(dm_path) as f:
            data_hash = json.load(f).get("sha256", "")[:16]

    base = {"run_id": f"consolidated_{int(time.time())}",
            "region": "primary", "split": "test", "is_test": True,
            "training_seed": -1, "checkpoint": "", "architecture": "w128d6",
            "config_hash": chash, "data_hash": data_hash,
            "source_file": ""}

    rows = []

    # Experiment A
    pa = os.path.join(RAW, "expA", "expA_metrics.json")
    if os.path.exists(pa):
        dfa = pd.DataFrame(json.load(open(pa)))
        b = dict(base)
        b.update({"split": "verification", "is_test": False,
                  "source_file": os.path.relpath(pa, ROOT)})
        rows += melt_rows(dfa, "expA_fd_verification", b)

    # Experiment B (all parquet daily files, primary & external, all parts)
    for pth in glob.glob(os.path.join(RAW, "expB", "daily_*.parquet")):
        if "minitest" in pth:
            continue
        dfb = pd.read_parquet(pth)
        b = dict(base)
        parts = os.path.basename(pth).replace(".parquet", "").split("_")
        b.update({"region": parts[1], "split": parts[2],
                  "is_test": parts[2] == "test",
                  "source_file": os.path.relpath(pth, ROOT)})
        rows += melt_rows(dfb, "expB_operation", b)

    # Experiment C
    pc = os.path.join(RAW, "expC", "robustness.parquet")
    if os.path.exists(pc):
        dfc = pd.read_parquet(pc)
        b = dict(base)
        b.update({"source_file": os.path.relpath(pc, ROOT)})
        rows += melt_rows(dfc, "expC_robustness", b)

    # Experiment D
    pd_ = os.path.join(RAW, "expD", "ablations.parquet")
    if os.path.exists(pd_):
        dfd = pd.read_parquet(pd_)
        b = dict(base)
        b.update({"split": "verification", "is_test": False,
                  "source_file": os.path.relpath(pd_, ROOT)})
        rows += melt_rows(dfd, "expD_ablations", b)

    master = pd.DataFrame(rows)
    for c in ID_COLS:
        if c not in master.columns:
            master[c] = "" if c not in ("value",) else np.nan
    out_path = os.path.join(IMM, "results_master.parquet")
    master.to_parquet(out_path, index=False)
    sha = file_sha256(out_path)

    # config snapshot
    snap = os.path.join(IMM, "config_snapshot")
    os.makedirs(snap, exist_ok=True)
    for f in glob.glob(os.path.join(ROOT, "configs", "*.yaml")):
        shutil.copy(f, snap)
    cal = os.path.join(ROOT, "data", "calibrated_parameters.json")
    if os.path.exists(cal):
        shutil.copy(cal, snap)

    manifest = {
        "results_master_sha256": sha,
        "n_rows": int(len(master)),
        "config_hash": chash,
        "data_hash": data_hash,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment_info(),
        "experiments": sorted(master["experiment"].unique().tolist())
        if len(master) else [],
    }
    with open(os.path.join(IMM, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"results_master.parquet: {len(master)} rows, sha256={sha[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
