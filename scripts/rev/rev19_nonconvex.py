"""Nonconvex capacity-band tariff experiment.

A flat surcharge c_step (currency/h) applies while grid import exceeds
g_thr (kW). The running cost becomes nonconvex in the action, which the
scalar Hamiltonian minimization of RV-PINN-PI prices exactly, while
convex (rolling-horizon) MPC must use a convex surrogate. Protocol
mirrors the accepted representative-day scope of the lambda sweeps
(winter_weekday regime, 3 method seeds, 30 held-out test days).

Methods
  rvpinnpi_step   retrained RV-PINN-PI on the surcharge objective (GPU)
  rvpinnpi_primary primary-value checkpoints, fee-aware greedy deployment
  mpc_env         deterministic MPC + tightest convex minorant (envelope)
  mpc_cap         deterministic MPC + soft-cap linear over-penalty
  mpc_pf_env      perfect-forecast MPC + envelope (isolates the
                  convexification gap from the forecast gap)
  oracle_dp       perfect-foresight exact DP (nonconvex optimum reference)
  sac_step/td3_step  DRL retrained on the surcharge reward (CPU)
  self_consumption, no_storage  rules (evaluation only)

Usage (run stages separately; DRL is CPU-bound and can run in parallel
with the GPU PINN training):
  python scripts/rev/rev19_nonconvex.py --parts train        # GPU
  python scripts/rev/rev19_nonconvex.py --parts drl          # CPU
  python scripts/rev/rev19_nonconvex.py --parts eval,table
  python scripts/rev/rev19_nonconvex.py --smoke --parts train,drl,eval,table
"""
from __future__ import annotations

import argparse
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

REGIME = "winter_weekday"
N_DAYS = 30
N_SEEDS = 3

# ---- Experiment-E tariff -------------------------------------------------
C_STEP = 40.0        # currency/h flat surcharge while G > g_thr
G_THR = 300.0        # kW contracted import band (evening-peak level:
                     # exceeded a few hours per day, avoidable by shaving)
W_STEP = 10.0        # kW smoothing width (training side)
W_CAP = 25.0         # kW width of the soft-cap over-penalty surrogate

TAG = "e19_step"     # checkpoint tag under checkpoints/expD/


def make_pv(p):
    pv = copy.deepcopy(p)
    pv.c_step, pv.g_thr, pv.w_step = C_STEP, G_THR, W_STEP
    return pv


def envelope_slope(pv, prof):
    """Tightest convex minorant slope c_step/(G_ub - g_thr) with
    G_ub = max forecast net load + max forecast error + max charge rate."""
    g_ub = float(np.max(prof.n_hour)) + pv.y_max + pv.a_c
    return pv.c_step / max(g_ub - pv.g_thr, 1.0)


def bundle(smoke):
    from src.evaluation import load_region_days
    from src.exp_common import load_stack, pick_seeds, regime_bundle, seeds_of
    from src.regime_profiles import load_calibration
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
    p, prof = params[REGIME], profs[REGIME]
    pv = make_pv(p)
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] == REGIME][: (5 if smoke else N_DAYS)]
    seeds_m = pick_seeds(cfg)[: (1 if smoke else N_SEEDS)]
    return cfg, p, pv, prof, B0, days, seeds_m, seeds_of(cfg)


# ---------------------------------------------------------------- train --
def part_train(smoke):
    import torch
    torch.set_default_dtype(torch.float64)
    from scripts.run_ablations import train_variant
    cfg, p, pv, prof, B0, days, seeds_m, seeds = bundle(smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sf = 0.05 if smoke else 1.0
    for ms in seeds_m:
        t0 = time.time()
        res, out = train_variant(TAG, pv, prof, cfg, seeds, B0, ms, device,
                                 use_p=True, steps_factor=sf)
        print(f"trained {TAG} seed{ms} in {time.time()-t0:.0f}s -> {out}",
              flush=True)
    return 0


# ------------------------------------------------------------------ drl --
def part_drl(smoke, algos="sac,td3"):
    import torch
    torch.set_default_dtype(torch.float32)
    from stable_baselines3 import SAC, TD3
    from src.drl_env import MicrogridEnv
    from scripts.train_drl import validation_objective
    from src.exp_common import ckpt_dir
    from src.reproducibility import seed_everything
    cfg, p, pv, prof, B0, days, seeds_m, seeds = bundle(smoke)
    drl_cfg = cfg["drl"]
    total = 30_000 if smoke else int(drl_cfg["total_transitions"])
    eval_every = 10_000 if smoke else int(drl_cfg["eval_every_transitions"])
    n_eval = 4 if smoke else int(drl_cfg["n_eval_rollouts"])
    patience = int(drl_cfg.get("early_stop_patience_evals", 5))
    net_arch = list(drl_cfg.get("net_arch", [256, 256]))
    params_e = {REGIME: pv}
    profs_e = {REGIME: prof}
    wanted = set(algos.split(","))
    for algo_name, Algo in (("sac", SAC), ("td3", TD3)):
        if algo_name not in wanted:
            continue
        for ms in seeds_m:
            out = ckpt_dir("expE", f"{algo_name}_step", f"seed{ms}")
            best_path = os.path.join(out, "best_model.zip")
            summ_path = os.path.join(out, "training_summary.json")
            if os.path.exists(summ_path):
                print(f"skip {algo_name} seed{ms} (done)", flush=True)
                continue
            os.makedirs(out, exist_ok=True)
            seed_everything(61_000 + ms)
            env = MicrogridEnv(params_e, profs_e, seed=61_000 + ms,
                               fixed_regime=REGIME)
            model = Algo("MlpPolicy", env, seed=61_000 + ms, device="cpu",
                         policy_kwargs=dict(net_arch=net_arch), verbose=0)
            best, bad, done_tr, hist = np.inf, 0, 0, []
            t0 = time.time()
            while done_tr < total:
                model.learn(total_timesteps=eval_every,
                            reset_num_timesteps=False, progress_bar=False)
                done_tr += eval_every
                val = validation_objective(model, params_e, profs_e,
                                           n_eval, 91_000 + ms)
                hist.append({"transitions": done_tr, "val": val})
                print(f"{algo_name} seed{ms} {done_tr} val={val:.2f}",
                      flush=True)
                if val < best - 1e-9:
                    best, bad = val, 0
                    model.save(best_path)
                else:
                    bad += 1
                    if bad >= patience:
                        break
            with open(summ_path, "w") as f:
                json.dump({"algo": algo_name, "seed": ms, "best_val": best,
                           "transitions": done_tr,
                           "wall_s": time.time() - t0, "history": hist,
                           "tariff": {"c_step": C_STEP, "g_thr": G_THR,
                                      "w_step": W_STEP}}, f, indent=2)
            print(f"{algo_name} seed{ms} done best={best:.2f}", flush=True)
    return 0


# ----------------------------------------------------------------- eval --
def part_eval(smoke):
    import torch
    torch.set_default_dtype(torch.float64)
    from src.evaluation import (DRLController, NeuralController,
                                evaluate_method)
    from src.exp_common import ckpt_dir
    from src.mpc import MPCController
    from src.oracle_dp import OracleDPController
    from src.policy_iteration import load_checkpoint
    from src.rules import NoStorage, SelfConsumption
    cfg, p, pv, prof, B0, days, seeds_m, seeds = bundle(smoke)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params_e = {REGIME: pv}
    profs_e = {REGIME: prof}
    rows = []
    keep = ("date", "bill", "capacity_fee", "total_cost", "objective",
            "exceed_hours", "peak_import_kw", "efc", "throughput_kwh",
            "soc_violations", "terminal_soc_dev")

    def record(name, seed, res):
        for r in res:
            rows.append({"method": name, "seed": seed,
                         **{k: r[k] for k in keep}})
        print(f"evaluated {name} seed{seed}: mean total "
              f"{np.mean([r['total_cost'] for r in res]):.1f}", flush=True)

    def ev(factory):
        return evaluate_method(factory, days, params_e, profs_e)

    # retrained RV-PINN-PI
    for ms in seeds_m:
        res_path = os.path.join(ckpt_dir("expD", TAG, f"seed{ms}"),
                                "result.json")
        res = json.load(open(res_path))
        ck = res.get("selected_checkpoint") or res.get("checkpoint")
        model = load_checkpoint(remap(ck), pv, device)
        record("rvpinnpi_step", ms,
               ev(lambda reg: NeuralController(model, pv, prof)))

    # primary value function, fee-aware greedy deployment
    for ms in seeds_m:
        res = json.load(open(os.path.join(
            ckpt_dir("pinn_pi", "primary", REGIME, f"seed{ms}"),
            "result.json")))
        model = load_checkpoint(remap(res["selected_checkpoint"]), pv,
                                device)
        record("rvpinnpi_primary", ms,
               ev(lambda reg: NeuralController(model, pv, prof)))

    # MPC surrogates
    sl_env = envelope_slope(pv, prof)
    sl_cap = pv.c_step / W_CAP
    record("mpc_env", 0, ev(lambda reg: MPCController(pv,
                                                      step_slope=sl_env)))
    record("mpc_cap", 0, ev(lambda reg: MPCController(pv,
                                                      step_slope=sl_cap)))
    record("mpc_pf_env", 0, ev(lambda reg: MPCController(
        pv, oracle=True, step_slope=sl_env)))

    # exact nonconvex perfect-foresight DP
    record("oracle_dp", 0, ev(lambda reg: OracleDPController(pv, prof)))

    # rules
    record("self_consumption", 0, ev(lambda reg: SelfConsumption()))
    record("no_storage", 0, ev(lambda reg: NoStorage()))

    # DRL (float32 policy inference is handled inside DRLController)
    from stable_baselines3 import SAC, TD3
    for algo_name, Algo in (("sac", SAC), ("td3", TD3)):
        for ms in seeds_m:
            best_path = os.path.join(
                ckpt_dir("expE", f"{algo_name}_step", f"seed{ms}"),
                "best_model.zip")
            if not os.path.exists(best_path):
                print(f"missing {best_path}; skip", flush=True)
                continue
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)   # SB3 nets are float32
            try:
                model = Algo.load(best_path, device="cpu")
                model.policy.to(torch.float32)
            finally:
                torch.set_default_dtype(prev)
            record(f"{algo_name}_step", ms,
                   ev(lambda reg: DRLController(model, pv, [REGIME],
                                                REGIME)))

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "rev19_daily.parquet"))
    summary = {}
    for m, sub in df.groupby("method"):
        summary[m] = {
            "n_seeds": int(sub["seed"].nunique()), "n": int(len(sub)),
            **{k: float(sub[k].mean()) for k in keep if k != "date"}}
    save_json("rev19_summary.json",
              {"tariff": {"c_step": C_STEP, "g_thr": G_THR,
                          "w_step": W_STEP, "w_cap": W_CAP,
                          "envelope_slope": sl_env},
               "methods": summary})
    return 0


# ---------------------------------------------------------------- table --
LABELS = {
    "oracle_dp": "Perfect-foresight exact DP (oracle)",
    "mpc_pf_env": "Perfect-forecast MPC (convex envelope)",
    "mpc_env": "Deterministic MPC (convex envelope)",
    "mpc_cap": "Deterministic MPC (soft cap)",
    "rvpinnpi_step": "RV--PINN--PI (retrained)",
    "rvpinnpi_primary": "RV--PINN--PI (primary value)",
    "sac_step": "SAC (retrained)",
    "td3_step": "TD3 (retrained)",
    "self_consumption": "Self-consumption",
    "no_storage": "No storage",
}
ORDER = ["no_storage", "self_consumption", "mpc_env", "mpc_cap",
         "sac_step", "td3_step", "rvpinnpi_primary", "rvpinnpi_step",
         "mpc_pf_env", "oracle_dp"]


def part_table(smoke):
    with open(os.path.join(OUT, "rev19_summary.json")) as f:
        summ = json.load(f)["methods"]
    lines = []
    for m in ORDER:
        if m not in summ:
            continue
        r = summ[m]
        lines.append(
            f"{LABELS[m]} & {r['total_cost']:.1f} & {r['bill']:.1f} & "
            f"{r['capacity_fee']:.1f} & {r['exceed_hours']:.2f} & "
            f"{r['objective']:.1f} & {r['peak_import_kw']:.0f} & "
            f"{r['efc']:.2f} \\\\")
    body = "\n".join(lines)
    cs, gt = f"{C_STEP:.0f}", f"{G_THR:.0f}"
    tex = (
        "% Nonconvex capacity-band tariff\n"
        f"% (c_step = {cs} GBP/h above g_thr = {gt} kW),\n"
        "% winter-weekday regime, 30 held-out test days. Generated by\n"
        "% scripts/rev/rev19_nonconvex.py.\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Nonconvex capacity-band tariff (Experiment E): a flat\n"
        f"surcharge of {cs}\\,GBP/h applies while the grid import exceeds\n"
        f"{gt}\\,kW. Winter-weekday regime, 30 held-out test days; mean per\n"
        "day. Bill and fee in GBP/day; exceedance in h/day. The oracle is\n"
        "an information-advantaged perfect-foresight exact reference; the\n"
        "MPC rows use the tightest convex minorant (envelope) or a linear\n"
        "soft cap of the surcharge.}\n"
        "\\label{tab:nonconvex}\n\\small\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "Method & Total & Bill & Fee & Exceed. & Objective & Peak (kW) & "
        "EFC \\\\\n\\midrule\n" + body + "\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    out_tex = os.path.join(OUT, "tableE_nonconvex.tex")
    with open(out_tex, "w") as f:
        f.write(tex)
    print("saved", out_tex)
    print(tex)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="eval,table")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--algos", default="sac,td3",
                    help="drl part only: which algorithms to train")
    args = ap.parse_args()
    for part in args.parts.split(","):
        if part == "drl":
            rc = part_drl(args.smoke, args.algos)
        else:
            rc = {"train": part_train, "eval": part_eval,
                  "table": part_table}[part](args.smoke)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
