"""Revision analysis for Reviewer 1, Comment 1.8 (and Reviewer 2.3).

Quantifies, per method, how often and by how much the shared discrete
safety projection modifies the raw action on the held-out test days:
  - fraction of control steps with a modified action,
  - |a_raw - a_applied| statistics (mean over all steps, mean over
    modified steps, p95, max), in kW and relative to a_max = 500 kW.

Implementation: src.evaluation.project_action_to_safe_set is wrapped by
a recorder (the same monkeypatch mechanism as ablation A5/A6) and every
method is re-run through the identical evaluation engine on the
identical test days. MPC runs on a stratified subset (6 days/regime).

Output: results/rev/projection_daily.parquet,
        results/rev/projection_summary.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT, OUT, remap, save_json  # noqa: E402

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import torch                                              # noqa: E402

import src.evaluation as ev                               # noqa: E402
from src.evaluation import (DRLController,                # noqa: E402
                            NeuralController, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,         # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle)
from src.mpc import MPCController                         # noqa: E402
from src.policy_iteration import load_checkpoint          # noqa: E402
from src.regime_profiles import load_calibration          # noqa: E402
from src.rules import NoStorage, PriceThreshold, SelfConsumption  # noqa: E402

torch.set_default_dtype(torch.float64)
MPC_DAYS_PER_REGIME = 6
TOL = 1e-9


class Recorder:
    def __init__(self):
        self.diffs = []

    def wrap(self, orig):
        def patched(a, s, p, dt):
            out = orig(a, s, p, dt)
            self.diffs.append(float(np.abs(out - a)[0]))
            return out
        return patched

    def flush(self):
        d = np.array(self.diffs)
        self.diffs = []
        mod = d > TOL * 500.0
        return {"n_steps": int(d.size), "n_modified": int(mod.sum()),
                "mean_abs_all": float(d.mean()) if d.size else 0.0,
                "mean_abs_mod": float(d[mod].mean()) if mod.any() else 0.0,
                "p95_abs": float(np.quantile(d, 0.95)) if d.size else 0.0,
                "max_abs": float(d.max()) if d.size else 0.0}


def run_method(name, seed, factory, days, params, profs, rec, rows):
    controllers = {}
    for day in days:
        reg = day["regime"]
        if reg not in controllers:
            controllers[reg] = factory(reg)
        r = ev.run_day(controllers[reg], day, params[reg], profs[reg])
        stats = rec.flush()
        rows.append({"method": name, "seed": seed, "date": day["date"],
                     "regime": reg, "bill": r["bill"],
                     "projection_events_engine": r["projection_events"],
                     **stats})
    print(f"  {name} seed{seed}: {len(days)} days", flush=True)


def main() -> int:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds_m = pick_seeds(cfg)
    regimes = pick_regimes(cfg, "primary")
    params, profs, _ = regime_bundle("primary", regimes, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] in params]
    sub = []
    for reg in regimes:
        sub += [d for d in days if d["regime"] == reg][:MPC_DAYS_PER_REGIME]
    tuning = json.load(open(os.path.join(ROOT, "results", "raw", "expB",
                                         "threshold_tuning.json")))

    rec = Recorder()
    orig_proj = ev.project_action_to_safe_set
    ev.project_action_to_safe_set = rec.wrap(orig_proj)
    rows = []
    try:
        run_method("no_storage", -1, lambda reg: NoStorage(), days,
                   params, profs, rec, rows)
        run_method("self_consumption", -1, lambda reg: SelfConsumption(),
                   days, params, profs, rec, rows)
        run_method("threshold_rule", -1,
                   lambda reg: PriceThreshold(params[reg], tuning["c_lo"],
                                              tuning["c_hi"]),
                   days, params, profs, rec, rows)

        for algo in ("sac", "td3"):
            from stable_baselines3 import SAC, TD3
            Algo = {"sac": SAC, "td3": TD3}[algo]
            for ms in seeds_m:
                sm = json.load(open(os.path.join(
                    ckpt_dir("drl", "primary", algo, f"seed{ms}"),
                    "training_summary.json")))
                prev = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    model = Algo.load(remap(sm["selected"]["checkpoint"]),
                                      device="cpu")
                finally:
                    torch.set_default_dtype(prev)
                run_method(algo, ms,
                           lambda reg, _m=model: DRLController(
                               _m, params[reg], regimes, reg),
                           days, params, profs, rec, rows)

        for tag, name in (("pinn_pi", "rvpinnpi"),
                          ("direct_hjb", "direct_hjb")):
            for ms in seeds_m:
                models = {}
                for reg in regimes:
                    res = json.load(open(os.path.join(
                        ckpt_dir(tag, "primary", reg, f"seed{ms}"),
                        "result.json")))
                    ck = res.get("selected_checkpoint") or res["checkpoint"]
                    models[reg] = load_checkpoint(remap(ck), params[reg],
                                                  device)
                run_method(name, ms,
                           lambda reg, _mm=models: NeuralController(
                               _mm[reg], params[reg], profs[reg]),
                           days, params, profs, rec, rows)
                pd.DataFrame(rows).to_parquet(
                    os.path.join(OUT, "projection_daily.parquet"))

        run_method("mpc_deterministic", -1,
                   lambda reg: MPCController(params[reg]), sub,
                   params, profs, rec, rows)
        run_method("mpc_perfect_forecast", -1,
                   lambda reg: MPCController(params[reg], oracle=True), sub,
                   params, profs, rec, rows)
    finally:
        ev.project_action_to_safe_set = orig_proj

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "projection_daily.parquet"))

    summary = {}
    for meth, g in df.groupby("method"):
        n_steps = g.n_steps.sum()
        n_mod = g.n_modified.sum()
        w = g.n_steps / n_steps
        summary[meth] = {
            "n_days": int(g.date.nunique()),
            "fraction_steps_modified": float(n_mod / n_steps),
            "mean_abs_all_kw": float((g.mean_abs_all * w).sum()),
            "mean_abs_mod_kw": float(
                (g.mean_abs_mod * g.n_modified).sum() / max(n_mod, 1)),
            "p95_abs_kw": float(g.p95_abs.quantile(0.95)),
            "max_abs_kw": float(g.max_abs.max()),
            "mean_abs_all_rel_amax": float((g.mean_abs_all * w).sum() / 500.0),
        }
    save_json("projection_summary.json", summary)
    print("projection analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
