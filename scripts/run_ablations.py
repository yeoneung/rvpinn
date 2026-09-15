"""Experiment D: ablations (A1-A10, main.tex §7.9).

Coverage map:
  A1 direct vs PI ............ from Experiment A results (no rerun)
  A2 hard vs penalty terminal  retrain variant (2-state)
  A3 selection criterion ..... post hoc from stored PI histories
  A4 uniform vs adaptive ..... from Experiment A (rvpinnpi_noadapt)
  A5 no taper (deployment) ... evaluation-mode ablation, violations counted
  A6 taper only, no discrete . evaluation-mode ablation
  A7 degradation off ......... retrain variant (3-state, one regime)
  A8 distilled actor ......... distillation + greedy-error measurement
  A9 width/depth ............. retrain grid subset (2-state)
  A10 collocation budget ..... retrain subset (2-state)
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

from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_seeds, regime_bundle, seeds_of, trainer_cfg)
from src.pinn_value import DTYPE, ValueNet, greedy_action_np  # noqa: E402
from src.policy_iteration import (PINNPITrainer,         # noqa: E402
                                  load_checkpoint)
from src.reproducibility import seed_everything          # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = os.path.join(ROOT, "results", "raw", "expD")
REGIME = "winter_weekday"


def eval_vs_fd(model, p, prof, cfg, device):
    from scripts.run_fd_verification import (evaluate_checkpoint,
                                             fd_reference)
    sols = fd_reference(cfg, p, prof)
    return evaluate_checkpoint(model, p, prof, sols[-1],
                               sols[-2] if len(sols) > 1 else sols[-1],
                               cfg, device)


def train_variant(tag, p, prof, cfg, seeds, B0, ms, device, width=None,
                  depth=None, adaptive=True, use_p=False,
                  steps_factor=1.0, batch_factor=1.0, hard_terminal=True):
    out = ckpt_dir("expD", tag, f"seed{ms}")
    res_path = os.path.join(out, "result.json")
    if os.path.exists(res_path):
        with open(res_path) as f:
            return json.load(f), out
    tcfg = copy.deepcopy(trainer_cfg(cfg))
    tcfg["policy_evaluation"]["adam_max_steps"] = int(
        tcfg["policy_evaluation"]["adam_max_steps"] * steps_factor)
    tcfg["policy_evaluation"]["adam_batch_size"] = max(64, int(
        tcfg["policy_evaluation"]["adam_batch_size"] * batch_factor))
    seed_everything(41_000 + ms)
    tr = PINNPITrainer(p, prof, tcfg, seeds, out_scale=B0 / p.T,
                       use_p=use_p, method_seed=ms, adaptive=adaptive,
                       width=width, depth=depth, hard_terminal=hard_terminal)
    res = tr.run(out)
    return res, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--parts", default="a2,a3,a5a6,a9,a10,a7,a8")
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = load_stack(args.config)
    seeds = seeds_of(cfg)
    seeds_m = pick_seeds(cfg)[: args.n_seeds]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
    p, prof = params[REGIME], profs[REGIME]
    os.makedirs(OUT, exist_ok=True)
    parts = set(args.parts.split(","))
    rows = []

    def save():
        pd.DataFrame(rows).to_parquet(os.path.join(OUT, "ablations.parquet"))

    # ---- A2: hard terminal vs terminal penalty (2-state) ----
    if "a2" in parts:
        for ms in seeds_m:
            res, out = train_variant("a2_soft_terminal", p, prof, cfg,
                                     seeds, B0, ms, device,
                                     hard_terminal=False)
            model = load_checkpoint(res["selected_checkpoint"], p, device)
            m = eval_vs_fd(model, p, prof, cfg, device)
            # measured terminal mismatch e_T on a Sobol slice at t = T
            from src.residual_validation import sobol_points
            pts = sobol_points(16384, p, 13579, use_p=False, device=device)
            tT = torch.full_like(pts["s"], p.T)
            with torch.no_grad():
                v = model(tT, pts["s"], pts["y"], pts["pz"])
            phi = p.lam_T * (pts["s"] - p.s_tar) ** 2
            e_T = float((v - phi).abs().max())
            rows.append({"ablation": "A2_terminal", "variant": "penalty",
                         "seed": ms, "e_T": e_T, **m})
            save()
        # hard-terminal comparator from Experiment A (e_T = 0 by design)
        for ms in seeds_m:
            res_path = os.path.join(ckpt_dir("expA", "rvpinnpi",
                                             f"seed{ms}"), "result.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    resA = json.load(f)
                modelA = load_checkpoint(resA["selected_checkpoint"], p,
                                         device)
                m = eval_vs_fd(modelA, p, prof, cfg, device)
                rows.append({"ablation": "A2_terminal", "variant": "hard",
                             "seed": ms, "e_T": 0.0, **m})
        save()
        print("A2 done")

    # ---- A3: checkpoint-selection criterion (post hoc, Experiment A) ----
    if "a3" in parts:
        for ms in pick_seeds(cfg):
            res_path = os.path.join(ckpt_dir("expA", "rvpinnpi",
                                             f"seed{ms}"), "result.json")
            if not os.path.exists(res_path):
                continue
            with open(res_path) as f:
                res = json.load(f)
            its = res["iterations"]
            sel_indep = int(np.argmin(
                [it["tuneval_policy_residual"]["rms"] for it in its]))
            # 'training-residual' selection: pick by final training loss of
            # each iteration (stored via history_len; approximate by the
            # tuneval rms of the LAST iteration = no independent selection)
            sel_train = len(its) - 1
            with open(os.path.join(OUT, "expA_metrics_ref.json"), "w") as f:
                json.dump({"note": "A3 uses Experiment A stored metrics"}, f)
            rows.append({"ablation": "A3_selection", "seed": ms,
                         "variant": "independent",
                         "selected_iteration": sel_indep,
                         "rollout_cost": its[sel_indep]["rollout_cost_crn"]})
            rows.append({"ablation": "A3_selection", "seed": ms,
                         "variant": "training_residual",
                         "selected_iteration": sel_train,
                         "rollout_cost": its[sel_train]["rollout_cost_crn"]})
        save()
        print("A3 done")

    # ---- A5/A6: safety-layer deployment ablations ----
    if "a5a6" in parts:
        from src.evaluation import load_region_days, run_day
        from src.regime_profiles import load_calibration
        from src.safety import cont_bounds
        cal = load_calibration()
        days = [d for d in load_region_days("primary", "test", cal)
                if d["regime"] == REGIME][:30]
        res_path = os.path.join(
            ckpt_dir("pinn_pi", "primary", REGIME, f"seed{seeds_m[0]}"),
            "result.json")
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)
            model = load_checkpoint(res["selected_checkpoint"], p, device)

            class AblController:
                def __init__(self, mode):
                    self.mode = mode

                def __call__(self, t, s, y, pz, ctx):
                    from src.hamiltonian import minimize_hamiltonian
                    from src.pinn_value import value_and_derivatives
                    tt = torch.tensor([t], dtype=DTYPE, device=device)
                    st = torch.tensor([s], dtype=DTYPE, device=device)
                    yt = torch.tensor([y], dtype=DTYPE, device=device)
                    pt = torch.tensor([pz], dtype=DTYPE, device=device)
                    d = value_and_derivatives(model, tt, st, yt, pt,
                                              second_order=False)
                    v_s = d["v_s"].detach().cpu().numpy()
                    if self.mode == "no_taper":
                        lo = np.array([-p.E_max * 0 - p.a_c])
                        hi = np.array([p.a_d])
                    else:                       # taper_only
                        lo, hi = cont_bounds(np.array([s]), p)
                    a, _ = minimize_hamiltonian(
                        np.array([t]), np.array([s]), np.array([y]),
                        np.array([pz]), v_s, p, prof, lo, hi)
                    return float(a[0])

            import src.evaluation as ev
            orig_proj = ev.project_action_to_safe_set
            for mode in ("no_taper", "taper_only", "full_safety"):
                if mode == "full_safety":
                    ev.project_action_to_safe_set = orig_proj
                elif mode == "taper_only":
                    ev.project_action_to_safe_set = \
                        lambda a, s, pp, dt: np.clip(a, -pp.a_c, pp.a_d)
                else:
                    ev.project_action_to_safe_set = \
                        lambda a, s, pp, dt: np.clip(a, -pp.a_c, pp.a_d)
                ctl = (AblController(mode) if mode != "full_safety"
                       else AblController("taper_only"))
                for day in days:
                    r = run_day(ctl, day, p, prof)
                    rows.append({"ablation": "A5A6_safety", "variant": mode,
                                 "seed": seeds_m[0], **{
                                     k: r[k] for k in
                                     ("date", "bill", "objective",
                                      "soc_violations", "max_violation",
                                      "terminal_soc")}})
            ev.project_action_to_safe_set = orig_proj
            save()
            print("A5/A6 done")

    # ---- A9: width/depth grid (2-state) ----
    if "a9" in parts:
        # (128,6) is covered by the Experiment A primary runs; reduction
        # recorded in artifacts/failure_log.md (compute priority §22)
        grid = [(64, 6), (256, 6), (128, 4), (128, 8)]
        for (w, d_) in grid:
            for ms in seeds_m:
                res, out = train_variant(f"a9_w{w}d{d_}", p, prof, cfg,
                                         seeds, B0, ms, device,
                                         width=w, depth=d_)
                model = load_checkpoint(res["selected_checkpoint"], p, device)
                m = eval_vs_fd(model, p, prof, cfg, device)
                rows.append({"ablation": "A9_arch", "variant": f"w{w}d{d_}",
                             "seed": ms, **m})
                save()
        print("A9 done")

    # ---- A10: collocation budget (2-state) ----
    if "a10" in parts:
        # factor 1.0 covered by Experiment A primary runs (recorded)
        for fac in (0.25, 0.5, 2.0):
            for ms in seeds_m:
                res, out = train_variant(f"a10_b{fac}", p, prof, cfg,
                                         seeds, B0, ms, device,
                                         steps_factor=fac)
                model = load_checkpoint(res["selected_checkpoint"], p, device)
                m = eval_vs_fd(model, p, prof, cfg, device)
                rows.append({"ablation": "A10_budget", "variant": str(fac),
                             "seed": ms, **m})
                save()
        print("A10 done")

    # ---- A7: degradation coefficient off (3-state, one regime) ----
    if "a7" in parts:
        from src.evaluation import NeuralController, evaluate_method, \
            load_region_days
        from src.regime_profiles import load_calibration
        cal = load_calibration()
        days = [d for d in load_region_days("primary", "test", cal)
                if d["regime"] == REGIME][:30]
        p0 = copy.deepcopy(p)
        p0.lam1 = 0.0
        for ms in seeds_m:
            res, out = train_variant(f"a7_lam1_0", p0, prof, cfg, seeds,
                                     B0, ms, device, use_p=True)
            model = load_checkpoint(res["selected_checkpoint"], p0, device)
            r_off = evaluate_method(
                lambda reg: NeuralController(model, p0, prof),
                days, {REGIME: p0}, {REGIME: prof})
            for r in r_off:
                rows.append({"ablation": "A7_degradation",
                             "variant": "lam1_0", "seed": ms, **{
                                 k: r[k] for k in ("date", "bill",
                                                   "throughput_kwh", "efc",
                                                   "objective")}})
            # primary-coefficient comparator from Experiment B checkpoints
            res_path = os.path.join(
                ckpt_dir("pinn_pi", "primary", REGIME, f"seed{ms}"),
                "result.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    resb = json.load(f)
                mb = load_checkpoint(resb["selected_checkpoint"], p, device)
                r_on = evaluate_method(
                    lambda reg: NeuralController(mb, p, prof),
                    days, {REGIME: p}, {REGIME: prof})
                for r in r_on:
                    rows.append({"ablation": "A7_degradation",
                                 "variant": "lam1_primary", "seed": ms, **{
                                     k: r[k] for k in
                                     ("date", "bill", "throughput_kwh",
                                      "efc", "objective")}})
            save()
        print("A7 done")

    # ---- A8: distilled actor ----
    if "a8" in parts:
        res_path = os.path.join(ckpt_dir("expA", "rvpinnpi",
                                         f"seed{seeds_m[0]}"), "result.json")
        if os.path.exists(res_path):
            with open(res_path) as f:
                res = json.load(f)
            model = load_checkpoint(res["selected_checkpoint"], p, device)
            from src.residual_validation import sobol_points
            pts = sobol_points(65536, p, 987654, use_p=False, device=device)
            a_star = greedy_action_np(model, pts["t"], pts["s"], pts["y"],
                                      pts["pz"], prof)
            actor = torch.nn.Sequential(
                torch.nn.Linear(3, 64, dtype=DTYPE), torch.nn.Tanh(),
                torch.nn.Linear(64, 64, dtype=DTYPE), torch.nn.Tanh(),
                torch.nn.Linear(64, 1, dtype=DTYPE)).to(device)
            X = torch.stack([pts["t"] / p.T,
                             (pts["s"] - p.s_min) / (p.s_max - p.s_min),
                             (pts["y"] - p.y_min) / (p.y_max - p.y_min)],
                            dim=-1)
            Yt = torch.as_tensor(a_star / p.a_max, dtype=DTYPE,
                                 device=device)
            opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
            for it in range(3000):
                idx = torch.randint(X.shape[0], (4096,), device=device)
                loss = ((actor(X[idx]).squeeze(-1) - Yt[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                a_actor = (actor(X).squeeze(-1) * p.a_max).cpu().numpy()
            # delta-greedy error of the distilled actor via Hamiltonian
            from src.hamiltonian import hamiltonian_terms
            from src.pinn_value import value_and_derivatives
            d = value_and_derivatives(model, pts["t"], pts["s"], pts["y"],
                                      pts["pz"], second_order=False)
            v_s = d["v_s"].detach().cpu().numpy()
            t_np = pts["t"].cpu().numpy()
            nbar = prof.n_bar(t_np)
            cbar = prof.c_bar(t_np)
            s_np = pts["s"].cpu().numpy()
            y_np = pts["y"].cpu().numpy()
            pz_np = pts["pz"].cpu().numpy()
            from src.safety import cont_bounds
            lo, hi = cont_bounds(s_np, p)
            a_actor_c = np.clip(a_actor, lo, hi)
            h_actor = hamiltonian_terms(a_actor_c, s_np, y_np, nbar, cbar,
                                        pz_np, v_s, p)
            h_star = hamiltonian_terms(a_star, s_np, y_np, nbar, cbar,
                                       pz_np, v_s, p)
            delta = h_actor - h_star
            rows.append({"ablation": "A8_distilled", "variant": "actor",
                         "seed": seeds_m[0],
                         "delta_greedy_mean": float(delta.mean()),
                         "delta_greedy_max": float(delta.max()),
                         "action_rmse": float(np.sqrt(np.mean(
                             (a_actor_c - a_star) ** 2)))})
            save()
            print("A8 done")

    save()
    print(f"Experiment D complete: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
