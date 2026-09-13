"""Revision analyses for Reviewer 1, Comments 1.3 and 1.4.

1.3  Degradation-coefficient sensitivity: retrain RV-PINN-PI at
     lambda_1 factors {0.5, 2.0} (factors 0.0 and 1.0 already exist as
     ablation A7 and the primary run), 3 seeds, winter_weekday regime,
     and evaluate on the first 30 winter_weekday test days (same
     protocol as A7).

1.4  Peak-penalty sensitivity: retrain at lambda_pk factors
     {0.1, 10, 100} (factor 1.0 = primary), same protocol.

Both sweeps mirror scripts/run_ablations.py::A7 exactly: use_p=True
(3-state), pilot-frozen budgets from configs/paper_full.yaml, seeds from
configs/experiments.yaml. Existing rows for factors 0.0/1.0 are pulled
from the stored A7 evaluation and the primary Experiment-B checkpoints,
so nothing already trained is retrained.

Output: results/rev/sensitivity_daily.parquet (one row per day/seed/
variant) and results/rev/sensitivity_summary.json.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT, OUT, remap, save_json  # noqa: E402

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
import torch                                             # noqa: E402

from scripts.run_ablations import train_variant          # noqa: E402
from src.evaluation import (NeuralController,            # noqa: E402
                            evaluate_method, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_seeds, regime_bundle, seeds_of)
from src.policy_iteration import load_checkpoint         # noqa: E402
from src.regime_profiles import load_calibration         # noqa: E402

torch.set_default_dtype(torch.float64)
REGIME = "winter_weekday"
N_DAYS = 30
N_SEEDS = 3

LAM1_FACTORS = [0.5, 2.0]          # 0.0 (A7) and 1.0 (primary) exist
LAMPK_FACTORS = [0.1, 10.0, 100.0]  # 1.0 (primary) exists


def eval_days(model, p_eval, prof, days):
    """Evaluate a controller; cost accounting uses p_eval."""
    return evaluate_method(lambda reg: NeuralController(model, p_eval, prof),
                           days, {REGIME: p_eval}, {REGIME: prof})


def day_rows(res, sweep, factor, seed, lam_value):
    keep = ("date", "bill", "objective", "throughput_kwh", "efc",
            "peak_import_kw", "soc_violations", "terminal_soc_dev")
    return [{"sweep": sweep, "factor": factor, "seed": seed,
             "lam_value": lam_value, **{k: r[k] for k in keep}}
            for r in res]


def main() -> int:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds = seeds_of(cfg)
    seeds_m = pick_seeds(cfg)[:N_SEEDS]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
    p, prof = params[REGIME], profs[REGIME]
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] == REGIME][:N_DAYS]
    print(f"{len(days)} evaluation days, device={device}", flush=True)

    rows = []
    daily_path = os.path.join(OUT, "sensitivity_daily.parquet")

    def save():
        pd.DataFrame(rows).to_parquet(daily_path)

    # ---- existing comparators (no retraining) ----
    # lambda_1 factor 1.0 and lambda_pk factor 1.0: primary Experiment-B
    # checkpoints, evaluated here on the same 30-day subset for exact
    # comparability of the subsets.
    for ms in seeds_m:
        res = json.load(open(os.path.join(
            ckpt_dir("pinn_pi", "primary", REGIME, f"seed{ms}"),
            "result.json")))
        model = load_checkpoint(remap(res["selected_checkpoint"]), p, device)
        r = eval_days(model, p, prof, days)
        rows += day_rows(r, "lam1", 1.0, ms, p.lam1)
        rows += day_rows(r, "lampk", 1.0, ms, p.lam_pk)
    save()
    print("primary comparators evaluated", flush=True)

    # lambda_1 factor 0.0: stored A7 variant checkpoints
    for ms in seeds_m:
        p0 = copy.deepcopy(p)
        p0.lam1 = 0.0
        res_path = os.path.join(ckpt_dir("expD", "a7_lam1_0", f"seed{ms}"),
                                "result.json")
        if os.path.exists(res_path):
            res = json.load(open(res_path))
            model = load_checkpoint(remap(res["selected_checkpoint"]), p0,
                                    device)
            rows += day_rows(eval_days(model, p0, prof, days),
                             "lam1", 0.0, ms, 0.0)
    save()
    print("A7 (factor 0) evaluated", flush=True)

    # ---- new lambda_1 variants ----
    for fac in LAM1_FACTORS:
        for ms in seeds_m:
            pv = copy.deepcopy(p)
            pv.lam1 = p.lam1 * fac
            t0 = time.time()
            res, out = train_variant(f"rev13_lam1_{fac}", pv, prof, cfg,
                                     seeds, B0, ms, device, use_p=True)
            print(f"lam1 x{fac} seed{ms}: trained/loaded in "
                  f"{time.time()-t0:.0f}s", flush=True)
            model = load_checkpoint(remap(res["selected_checkpoint"]), pv,
                                    device)
            rows += day_rows(eval_days(model, pv, prof, days),
                             "lam1", fac, ms, pv.lam1)
            save()

    # ---- new lambda_pk variants ----
    for fac in LAMPK_FACTORS:
        for ms in seeds_m:
            pv = copy.deepcopy(p)
            pv.lam_pk = p.lam_pk * fac
            t0 = time.time()
            res, out = train_variant(f"rev14_lampk_{fac}", pv, prof, cfg,
                                     seeds, B0, ms, device, use_p=True)
            print(f"lampk x{fac} seed{ms}: trained/loaded in "
                  f"{time.time()-t0:.0f}s", flush=True)
            model = load_checkpoint(remap(res["selected_checkpoint"]), pv,
                                    device)
            rows += day_rows(eval_days(model, pv, prof, days),
                             "lampk", fac, ms, pv.lam_pk)
            save()

    # ---- summary ----
    df = pd.DataFrame(rows)
    summary = {}
    for sweep in ("lam1", "lampk"):
        sub = df[df.sweep == sweep]
        agg = sub.groupby("factor").agg(
            bill=("bill", "mean"),
            objective=("objective", "mean"),
            throughput_kwh=("throughput_kwh", "mean"),
            efc=("efc", "mean"),
            peak_import_kw=("peak_import_kw", "mean"),
            soc_violations=("soc_violations", "sum"),
            n=("bill", "size")).reset_index()
        summary[sweep] = agg.to_dict(orient="records")
    save_json("sensitivity_summary.json", summary)
    print("sensitivity sweeps complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
