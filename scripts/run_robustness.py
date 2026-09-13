"""Experiment C: robustness and distribution shift, applied WITHOUT
retuning any controller (main.tex §7.8).

Scenario families:
  msigma    : OU innovation multiplier (simulator rollouts, CRN)
  spike     : synthetic price spikes on realized test days
  eff/cap   : efficiency / capacity misspecification on realized test days,
              run BOTH with known and nominal safety parameters
  rho       : correlation shift (simulator rollouts)
  gammaR    : renewable penetration on realized test days
  external  : external-region evaluation without refitting
  loso      : leave-one-season-out (train-regime models applied to the
              held-out season's days)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                             # noqa: E402

from src.evaluation import (NeuralController,            # noqa: E402
                            evaluate_method, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle,
                            seeds_of)
from src.mpc import MPCController                        # noqa: E402
from src.policy_iteration import load_checkpoint         # noqa: E402
from src.regime_profiles import load_calibration         # noqa: E402
from src.rules import SelfConsumption                    # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(ROOT, "results", "raw", "expC")


def perturbed_params(p, d_eff=0.0, cap_factor=1.0):
    q = copy.deepcopy(p)
    q.eta_c = min(0.999, max(0.5, p.eta_c + d_eff))
    q.eta_d = min(0.999, max(0.5, p.eta_d + d_eff))
    q.E_max = p.E_max * cap_factor
    return q


def method_factories(cfg, region, params, profs, regimes, seed0, device):
    """Robustness method set: rvpinnpi, direct_hjb, mpc, self-consumption,
    sac (first seed for learned methods; seed sweep in Exp B)."""
    def neural(tag):
        def factory(reg):
            res_path = os.path.join(
                ckpt_dir(tag, region, reg, f"seed{seed0}"), "result.json")
            with open(res_path) as f:
                res = json.load(f)
            ck = res.get("selected_checkpoint") or res.get("checkpoint")
            return NeuralController(load_checkpoint(ck, params[reg], device),
                                    params[reg], profs[reg])
        return factory

    out = {
        "rvpinnpi": neural("pinn_pi"),
        "direct_hjb": neural("direct_hjb"),
        "mpc_deterministic": lambda reg: MPCController(params[reg]),
        "self_consumption": lambda reg: SelfConsumption(),
    }
    try:
        from scripts.run_real_data_evaluation import drl_factory
        out["sac"] = drl_factory("sac", region, params, regimes, seed0)
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--families", default="spike,eff,cap,gammaR,external,loso,msigma,rho")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    cfg = load_stack(args.config)
    rb = cfg["robustness"]
    seeds_m = pick_seeds(cfg)
    regimes = pick_regimes(cfg, "primary")
    params, profs, B0 = regime_bundle("primary", regimes, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] in params]
    if args.max_days:
        days = days[: args.max_days]

    facs = method_factories(cfg, "primary", params, profs, regimes,
                            seeds_m[0], device)
    rows = []

    def record(res, method, scenario, value, extra=None):
        for r in res:
            r2 = dict(r)
            r2.pop("traj", None)
            r2.update({"method": method, "scenario": scenario,
                       "scenario_value": value, "seed": seeds_m[0]})
            if extra:
                r2.update(extra)
            rows.append(r2)
        pd.DataFrame(rows).to_parquet(os.path.join(OUT, "robustness.parquet"))

    fams = set(args.families.split(","))

    # ---- price spikes on realized days ----
    if "spike" in fams:
        train_days = load_region_days("primary", "train", cal)
        spike_level = float(np.percentile(
            np.concatenate([d["C"] for d in train_days]),
            float(rb["price_spike_percentile"])))
        for dur in rb["price_spike_durations_h"]:
            days_mod = []
            for d in days:
                d2 = dict(d)
                C = d2["C"].copy()
                h0 = 18 - dur // 2          # evening spike, deterministic
                C[h0:h0 + dur] = spike_level
                d2["C"] = C
                days_mod.append(d2)
            for m, fac in facs.items():
                try:
                    record(evaluate_method(fac, days_mod, params, profs),
                           m, "price_spike_h", dur,
                           {"spike_level": spike_level})
                except FileNotFoundError:
                    pass
            print(f"spike dur={dur} done", flush=True)

    # ---- efficiency / capacity errors: known vs nominal safety ----
    scenarios = []
    if "eff" in fams:
        scenarios += [("efficiency_pp", e / 100.0, 1.0)
                      for e in rb["efficiency_error_pp"]]
    if "cap" in fams:
        scenarios += [("capacity_pct", 0.0, 1.0 + c / 100.0)
                      for c in rb["capacity_error_pct"]]
    for (name, d_eff, capf) in scenarios:
        p_true = {r: perturbed_params(params[r], d_eff, capf)
                  for r in params}
        for safety_mode in ("known", "nominal"):
            p_nom = p_true if safety_mode == "known" else params
            for m, fac in facs.items():
                try:
                    res = evaluate_method(fac, days, p_nom, profs,
                                          p_true_by_regime=p_true)
                    val = d_eff * 100 if name == "efficiency_pp" \
                        else (capf - 1) * 100
                    record(res, m, f"{name}_{safety_mode}", val)
                except FileNotFoundError:
                    pass
        print(f"{name} d_eff={d_eff} capf={capf} done", flush=True)

    # ---- renewable penetration ----
    if "gammaR" in fams:
        for g in rb["gamma_R"]:
            days_g = [d for d in load_region_days("primary", "test", cal,
                                                  gamma_R_override=g)
                      if d["regime"] in params]
            if args.max_days:
                days_g = days_g[: args.max_days]
            for m, fac in facs.items():
                try:
                    record(evaluate_method(fac, days_g, params, profs),
                           m, "gamma_R", g)
                except FileNotFoundError:
                    pass
            print(f"gamma_R={g} done", flush=True)

    # ---- external region without refitting (affine renormalization) ----
    if "external" in fams and "external" in cal:
        ext_params, ext_profs, _ = regime_bundle("external", None, cfg)
        ext_days = [d for d in load_region_days("external", "test", cal)
                    if d["regime"] in ext_params]
        if args.max_days:
            ext_days = ext_days[: args.max_days]
        # controllers keep PRIMARY-trained networks; state normalization uses
        # the external region's affine profile/domain (documented transfer)
        def ext_neural(tag):
            def factory(reg):
                res_path = os.path.join(
                    ckpt_dir(tag, "primary", reg, f"seed{seeds_m[0]}"),
                    "result.json")
                with open(res_path) as f:
                    res = json.load(f)
                ck = res.get("selected_checkpoint") or res.get("checkpoint")
                model = load_checkpoint(ck, params[reg], device)
                return NeuralController(model, ext_params[reg],
                                        ext_profs[reg])
            return factory
        ext_facs = {"rvpinnpi": ext_neural("pinn_pi"),
                    "mpc_deterministic":
                        lambda reg: MPCController(ext_params[reg]),
                    "self_consumption": lambda reg: SelfConsumption()}
        for m, fac in ext_facs.items():
            try:
                record(evaluate_method(fac, ext_days, ext_params, ext_profs),
                       m, "external_region", 1.0)
            except FileNotFoundError:
                pass
        print("external region done", flush=True)

    # ---- leave-one-season-out: apply a *different* season's model ----
    if "loso" in fams:
        seasons = ["winter", "spring", "summer", "autumn"]
        fallback = {"winter": "autumn", "spring": "summer",
                    "summer": "spring", "autumn": "winter"}
        def loso_factory(reg):
            season, dtype = reg.split("_", 1)
            donor = f"{fallback[season]}_{dtype}"
            if donor not in params:
                donor = sorted(params)[0]
            res_path = os.path.join(
                ckpt_dir("pinn_pi", "primary", donor, f"seed{seeds_m[0]}"),
                "result.json")
            with open(res_path) as f:
                res = json.load(f)
            ck = res.get("selected_checkpoint") or res.get("checkpoint")
            model = load_checkpoint(ck, params[donor], device)
            return NeuralController(model, params[reg], profs[reg])
        try:
            record(evaluate_method(loso_factory, days, params, profs),
                   "rvpinnpi_loso", "loso", 1.0)
            print("loso done", flush=True)
        except FileNotFoundError:
            print("loso skipped (missing checkpoints)")

    # ---- simulator-based m_sigma and rho shift (CRN rollouts) ----
    if fams & {"msigma", "rho"}:
        from src.exp_common import trainer_cfg
        from src.policy_iteration import PINNPITrainer, PolicyFn
        reg0 = regimes[0]
        p0, prof0 = params[reg0], profs[reg0]
        res_path = os.path.join(
            ckpt_dir("pinn_pi", "primary", reg0, f"seed{seeds_m[0]}"),
            "result.json")
        model0 = None
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)
            model0 = load_checkpoint(res["selected_checkpoint"], p0, device)
        if model0 is not None:
            tr = PINNPITrainer(p0, prof0, trainer_cfg(cfg), seeds_of(cfg),
                               out_scale=B0 / p0.T, use_p=True,
                               method_seed=seeds_m[0])
            if "msigma" in fams:
                for m_sig in rb["m_sigma"]:
                    p_sig = copy.deepcopy(p0)
                    p_sig.sigma_y *= m_sig
                    p_sig.sigma_p *= m_sig
                    tr.p = p_sig
                    cost = tr.rollout_cost(model0, n_paths=256, seed=5101)
                    rows.append({"method": "rvpinnpi",
                                 "scenario": "m_sigma",
                                 "scenario_value": m_sig,
                                 "objective": cost, "seed": seeds_m[0],
                                 "regime": reg0, "date": "simulator"})
                    cost0 = tr.rollout_cost(None, n_paths=256, seed=5101)
                    rows.append({"method": "zero_action",
                                 "scenario": "m_sigma",
                                 "scenario_value": m_sig,
                                 "objective": cost0, "seed": -1,
                                 "regime": reg0, "date": "simulator"})
            if "rho" in fams:
                for dr in rb["rho_shift"]:
                    p_r = copy.deepcopy(p0)
                    p_r.rho = float(np.clip(p0.rho + dr, -0.95, 0.95))
                    tr.p = p_r
                    cost = tr.rollout_cost(model0, n_paths=256, seed=5101)
                    rows.append({"method": "rvpinnpi", "scenario": "rho_shift",
                                 "scenario_value": dr, "objective": cost,
                                 "seed": seeds_m[0], "regime": reg0,
                                 "date": "simulator"})
            tr.p = p0
            pd.DataFrame(rows).to_parquet(
                os.path.join(OUT, "robustness.parquet"))
            print("simulator scenarios done", flush=True)

    print(f"Experiment C complete: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
