"""Evaluation-only aggregation of the lambda_1 / lambda_pk sensitivity
sweeps. Trains NOTHING: it only loads variants whose result.json already
exists, evaluates any missing (sweep, factor, seed) combinations on the
30 winter-weekday test days, merges with rows already recorded by
rev13_14_sensitivity.py, and writes the final summary JSON plus the two
LaTeX table fragments for the manuscript."""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT, OUT, remap, save_json  # noqa: E402

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import torch                                              # noqa: E402

from src.evaluation import (NeuralController,             # noqa: E402
                            evaluate_method, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,         # noqa: E402
                            pick_seeds, regime_bundle, seeds_of)
from src.policy_iteration import load_checkpoint          # noqa: E402
from src.regime_profiles import load_calibration          # noqa: E402

torch.set_default_dtype(torch.float64)
REGIME = "winter_weekday"
N_DAYS = 30

# (sweep, factor) -> checkpoint directory parts (relative to checkpoints/)
VARIANTS = {
    ("lam1", 0.0):   ("expD", "a7_lam1_0"),
    ("lam1", 0.5):   ("expD", "rev13_lam1_0.5"),
    ("lam1", 1.0):   ("pinn_pi", "primary", REGIME),
    ("lam1", 2.0):   ("expD", "rev13_lam1_2.0"),
    ("lampk", 1.0):  ("pinn_pi", "primary", REGIME),
    ("lampk", 10.0): ("expD", "rev14_lampk_10.0"),
    ("lampk", 100.0): ("expD", "rev14_lampk_100.0"),
}


def main() -> int:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds_m = pick_seeds(cfg)[:3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params, profs, _ = regime_bundle("primary", [REGIME], cfg)
    p, prof = params[REGIME], profs[REGIME]
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] == REGIME][:N_DAYS]

    daily_path = os.path.join(OUT, "sensitivity_daily.parquet")
    df = (pd.read_parquet(daily_path) if os.path.exists(daily_path)
          else pd.DataFrame(columns=["sweep", "factor", "seed"]))
    rows = df.to_dict(orient="records")
    have = set(zip(df.get("sweep", []), df.get("factor", []),
                   df.get("seed", [])))

    keep = ("date", "bill", "objective", "throughput_kwh", "efc",
            "peak_import_kw", "soc_violations", "terminal_soc_dev")
    for (sweep, fac), parts in VARIANTS.items():
        pv = copy.deepcopy(p)
        if sweep == "lam1":
            pv.lam1 = p.lam1 * fac
        else:
            pv.lam_pk = p.lam_pk * fac
        for ms in seeds_m:
            if (sweep, fac, ms) in have:
                continue
            res_path = os.path.join(ROOT, "checkpoints", *parts,
                                    f"seed{ms}", "result.json")
            if not os.path.exists(res_path):
                continue
            res = json.load(open(res_path))
            ck = res.get("selected_checkpoint") or res.get("checkpoint")
            model = load_checkpoint(remap(ck), pv, device)
            r = evaluate_method(
                lambda reg: NeuralController(model, pv, prof),
                days, {REGIME: pv}, {REGIME: prof})
            lam_value = pv.lam1 if sweep == "lam1" else pv.lam_pk
            for d in r:
                rows.append({"sweep": sweep, "factor": fac, "seed": ms,
                             "lam_value": lam_value,
                             **{k: d[k] for k in keep}})
            print(f"evaluated {sweep} x{fac} seed{ms}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(daily_path)

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
            n_seeds=("seed", "nunique"),
            n=("bill", "size")).reset_index()
        summary[sweep] = agg.to_dict(orient="records")
    save_json("sensitivity_summary.json", summary)

    # ---- LaTeX fragment ----
    # Second revision (Reviewer 1, Comment 2): the table now carries the
    # percentage change of every indicator relative to the 1x reference of
    # each sweep. The writer lives in rev15_table08_pct.py so that this
    # script and the stand-alone rebuild emit the identical fragment.
    print(json.dumps(summary, indent=1))
    from scripts.rev.rev15_table08_pct import write_table08
    write_table08(summary, os.path.join(OUT, "table08_sensitivity.tex"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
