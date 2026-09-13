"""Experiment A: finite-difference verification of the 2-state (S, Y)
problem (price deterministic). Produces FD reference solutions, trains
matched neural methods, and computes all metrics for Figs. 3/4/8 + Table 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                             # noqa: E402

from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_seeds, regime_bundle, seeds_of, trainer_cfg)
from src.finite_difference import (FDGrid, interp_V,     # noqa: E402
                                   solve_hjb, solve_policy_eval)
from src.pinn_value import DTYPE, ValueNet, greedy_action_np  # noqa: E402
from src.policy_iteration import (DirectHJBTrainer,      # noqa: E402
                                  PINNPITrainer, PolicyFn, load_checkpoint)
from src.reproducibility import (seed_everything,        # noqa: E402
                                 write_run_manifest, finalize_run_manifest)

torch.set_default_dtype(torch.float64)
OUT = os.path.join(ROOT, "results", "raw", "expA")
REGIME = "winter_weekday"


def fd_reference(cfg, p, prof, force=False):
    os.makedirs(OUT, exist_ok=True)
    grids = cfg["experiment_a"]["grids"]
    solutions = []
    for (Ns, Ny, Nt) in grids:
        tag = f"fd_{Ns}x{Ny}x{Nt}"
        path = os.path.join(OUT, tag + ".npz")
        if os.path.exists(path) and not force:
            print(f"FD cached: {tag}")
            data = np.load(path)
            solutions.append((Ns, Ny, Nt, data["V"], data["A"]))
            continue
        print(f"FD solve {tag} ...", flush=True)
        t0 = time.time()
        grid = FDGrid(p, Ns, Ny, Nt)
        V, A, diag = solve_hjb(grid, p, prof, verbose=True)
        np.savez_compressed(path, V=V, A=A,
                            howard_iters=np.array(diag["howard_iters"]))
        print(f"  done in {time.time()-t0:.0f}s "
              f"(mean howard iters {np.mean(diag['howard_iters']):.1f})")
        solutions.append((Ns, Ny, Nt, V, A))
    # grid-convergence diagnostics on common sample points
    conv = []
    p_ref = solutions[-1]
    grid_ref = FDGrid(p, p_ref[0], p_ref[1], p_ref[2])
    rng = np.random.default_rng(999)
    tq = rng.uniform(0, p.T, 20000)
    sq = rng.uniform(p.s_min, p.s_max, 20000)
    yq = rng.uniform(p.y_min, p.y_max, 20000)
    v_ref = interp_V(p_ref[3], grid_ref, tq, sq, yq)
    for (Ns, Ny, Nt, V, A) in solutions[:-1]:
        g = FDGrid(p, Ns, Ny, Nt)
        v = interp_V(V, g, tq, sq, yq)
        conv.append({"grid": f"{Ns}x{Ny}x{Nt}",
                     "linf_vs_finest": float(np.max(np.abs(v - v_ref))),
                     "rmse_vs_finest": float(np.sqrt(np.mean((v - v_ref)**2))),
                     "value_range": float(v_ref.max() - v_ref.min())})
    with open(os.path.join(OUT, "fd_convergence.json"), "w") as f:
        json.dump(conv, f, indent=2)
    print("FD convergence:", conv)
    return solutions


def neural_policy_on_grid(model, grid: FDGrid, prof, device):
    """Greedy action field of a neural value on the FD grid (per time)."""
    from src.pinn_value import greedy_action_torch
    A = np.empty((grid.Nt - 1, grid.Ns, grid.Ny))
    s = torch.as_tensor(grid.S_flat, dtype=DTYPE, device=device)
    y = torch.as_tensor(grid.Y_flat, dtype=DTYPE, device=device)
    pz = torch.zeros_like(s)
    for k in range(grid.Nt - 1):
        t = torch.full_like(s, grid.t[k])
        a = greedy_action_torch(model, t, s, y, pz, prof)
        A[k] = a.detach().cpu().numpy().reshape(grid.Ns, grid.Ny)
    return A


def policy_cost_fd(A_field, grid: FDGrid, p, prof):
    """Reference cost of a frozen policy field via linear FD solve."""
    def policy(t, s_flat, y_flat):
        k = min(int(round(t / grid.dt)), grid.Nt - 2)
        return A_field[k].ravel()
    V = solve_policy_eval(grid, p, prof, policy)
    return float(interp_V(V, grid, np.array([0.0]), np.array([0.5]),
                          np.array([0.0]))[0]), V


def evaluate_checkpoint(model, p, prof, fd_fine, fd_mid, cfg, device):
    """All Experiment-A metrics for one neural value checkpoint."""
    (Ns, Ny, Nt, V_fd, A_fd) = fd_fine
    grid_f = FDGrid(p, Ns, Ny, Nt)
    n_eval = int(cfg["experiment_a"]["eval_points"])
    eng = torch.quasirandom.SobolEngine(3, scramble=True, seed=424242)
    u = eng.draw(n_eval).numpy().astype(np.float64)
    tq = u[:, 0] * p.T
    sq = p.s_min + u[:, 1] * (p.s_max - p.s_min)
    yq = p.y_min + u[:, 2] * (p.y_max - p.y_min)
    v_fd = interp_V(V_fd, grid_f, tq, sq, yq)
    with torch.no_grad():
        v_nn = model(torch.as_tensor(tq, device=device),
                     torch.as_tensor(sq, device=device),
                     torch.as_tensor(yq, device=device),
                     torch.zeros(n_eval, dtype=DTYPE, device=device))
    v_nn = v_nn.cpu().numpy()
    # FD policy interp (piecewise in t on A time levels)
    from scipy.interpolate import RegularGridInterpolator
    fA = RegularGridInterpolator((grid_f.t[:-1], grid_f.s, grid_f.y), A_fd,
                                 bounds_error=False, fill_value=None)
    a_fd = fA(np.stack([np.minimum(tq, grid_f.t[-2]), sq, yq], axis=-1))
    from src.pinn_value import greedy_action_torch
    a_nn = greedy_action_torch(
        model, torch.as_tensor(tq, device=device),
        torch.as_tensor(sq, device=device),
        torch.as_tensor(yq, device=device),
        torch.zeros(n_eval, dtype=DTYPE, device=device),
        prof).detach().cpu().numpy()
    # Decision-focused diagnostics. A finite-difference derivative of the
    # reference value gives the local Hamiltonian regret of the deployed
    # neural action; this is more interpretable than value RMSE alone near
    # bang-bang switching surfaces.
    hs = 0.25 * grid_f.hs
    s_lo = np.maximum(sq - hs, p.s_min)
    s_hi = np.minimum(sq + hs, p.s_max)
    v_lo = interp_V(V_fd, grid_f, tq, s_lo, yq)
    v_hi = interp_V(V_fd, grid_f, tq, s_hi, yq)
    v_s_fd = (v_hi - v_lo) / np.maximum(s_hi - s_lo, 1e-12)
    from src.hamiltonian import hamiltonian_terms
    nbar = np.asarray(prof.n_bar(tq))
    cbar = np.asarray(prof.c_bar(tq))
    h_nn = hamiltonian_terms(a_nn, sq, yq, nbar, cbar,
                             np.zeros_like(tq), v_s_fd, p)
    h_fd = hamiltonian_terms(a_fd, sq, yq, nbar, cbar,
                             np.zeros_like(tq), v_s_fd, p)
    h_regret = np.maximum(h_nn - h_fd, 0.0)
    action_abs = np.abs(a_nn - a_fd)
    sign_disagree = ((np.sign(a_nn) != np.sign(a_fd))
                     & (np.abs(a_nn) > 0.02 * p.a_max)
                     & (np.abs(a_fd) > 0.02 * p.a_max))
    # cost gap on the mid grid (runtime) using the FD linear solve
    (Nsm, Nym, Ntm, V_mid, _amid) = fd_mid
    grid_m = FDGrid(p, Nsm, Nym, Ntm)
    A_nn_field = neural_policy_on_grid(model, grid_m, prof, device)
    j_pi, _ = policy_cost_fd(A_nn_field, grid_m, p, prof)
    v_star_mid = float(interp_V(V_mid, grid_m, np.array([0.0]),
                                np.array([0.5]), np.array([0.0]))[0])
    return {
        "value_linf": float(np.max(np.abs(v_nn - v_fd))),
        "value_rmse": float(np.sqrt(np.mean((v_nn - v_fd) ** 2))),
        "policy_rmse": float(np.sqrt(np.mean((a_nn - a_fd) ** 2))),
        "policy_mae": float(np.mean(action_abs)),
        "policy_large_disagreement_fraction": float(
            np.mean(action_abs > 0.10 * p.a_max)),
        "policy_sign_disagreement_fraction": float(np.mean(sign_disagree)),
        "hamiltonian_regret_mean": float(np.mean(h_regret)),
        "hamiltonian_regret_p95": float(np.quantile(h_regret, 0.95)),
        "hamiltonian_regret_max": float(np.max(h_regret)),
        "cost_gap_eJ": j_pi - v_star_mid,
        "J_pi_fd": j_pi, "V_star_fd_mid": v_star_mid,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--methods", default="rvpinnpi,rvpinnpi_noadapt,direct_hjb",
                    help="comma-separated subset")
    args = ap.parse_args()
    cfg = load_stack(args.config)
    seeds = seeds_of(cfg)
    seeds_m = ([int(s) for s in args.seeds.split(",")] if args.seeds
               else pick_seeds(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
    p = params[REGIME]
    prof = profs[REGIME]

    os.makedirs(OUT, exist_ok=True)
    manifest = os.path.join(OUT, "run_manifest.json")
    write_run_manifest(manifest, cfg, seeds)

    solutions = fd_reference(cfg, p, prof)
    fd_fine = solutions[-1]
    fd_mid = solutions[-2] if len(solutions) >= 2 else solutions[-1]

    methods = {
        "rvpinnpi": dict(cls="pi", adaptive=True),
        "rvpinnpi_noadapt": dict(cls="pi", adaptive=False),
        "direct_hjb": dict(cls="direct"),
    }
    requested = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = set(requested) - set(methods)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    methods = {name: methods[name] for name in requested}
    rows = []
    for mname, mcfg in methods.items():
        for ms in seeds_m:
            out = ckpt_dir("expA", mname, f"seed{ms}")
            res_path = os.path.join(out, "result.json")
            if not os.path.exists(res_path) and not args.skip_train:
                print(f"=== expA {mname} seed{ms} ===", flush=True)
                seed_everything(31_000 + ms)
                if mcfg["cls"] == "pi":
                    tr = PINNPITrainer(p, prof, trainer_cfg(cfg), seeds,
                                       out_scale=B0 / p.T, use_p=False,
                                       method_seed=ms,
                                       adaptive=mcfg["adaptive"])
                    tr.run(out)
                else:
                    total = int(cfg["policy_evaluation"]["adam_max_steps"]) \
                        * int(cfg["policy_iteration"]["max_iterations"])
                    tr = DirectHJBTrainer(p, prof, trainer_cfg(cfg), seeds,
                                          out_scale=B0 / p.T, use_p=False,
                                          method_seed=ms)
                    tr.run(out, total_updates=total)
            with open(res_path) as f:
                res = json.load(f)
            # ---- final checkpoint metrics ----
            ck_path = res.get("selected_checkpoint") or res.get("checkpoint")
            model = load_checkpoint(ck_path, p, device)
            m = evaluate_checkpoint(model, p, prof, fd_fine, fd_mid, cfg,
                                    device)
            if "iterations" in res:      # PI variants: per-iteration metrics
                for it in res["iterations"]:
                    mod_i = load_checkpoint(it["checkpoint"], p, device)
                    mi = evaluate_checkpoint(mod_i, p, prof, fd_fine, fd_mid,
                                             cfg, device)
                    rows.append({
                        "method": mname, "seed": ms,
                        "iteration": it["iteration"],
                        "is_selected": it["iteration"] == res["selected_iteration"],
                        "train_rms_residual":
                            it["tuneval_policy_residual"]["rms"],
                        "val_max_residual":
                            it["tuneval_policy_residual"]["max"],
                        "hjb_max_residual": it["hjb_residual_max_tuneval"],
                        "greedy_error": it["greedy_error_tuneval"],
                        "policy_change": it["policy_change_norm"],
                        "rollout_cost": it["rollout_cost_crn"],
                        "wall_s": it["train_info"]["wall_s"],
                        "peak_mem_mb": it["train_info"].get("peak_mem_mb"),
                        **mi})
                cert = res.get("certification", {})
                rows.append({"method": mname, "seed": ms, "iteration": -1,
                             "is_selected": True, "is_final": True,
                             "cert_policy_max":
                                 cert.get("policy_residual", {}).get("max"),
                             "cert_hjb_max":
                                 cert.get("hjb_residual", {}).get("max"),
                             "cert_lip_estimate":
                                 cert.get("lipschitz_corrected_sensitivity_estimate"),
                             **m})
            else:
                rows.append({"method": mname, "seed": ms, "iteration": -1,
                             "is_selected": True, "is_final": True,
                             "train_rms_residual":
                                 res["tuneval_hjb_residual"]["rms"],
                             "val_max_residual":
                                 res["tuneval_hjb_residual"]["max"],
                             "hjb_max_residual":
                                 res["tuneval_hjb_residual"]["max"],
                             "wall_s": res["wall_s"], **m})
            with open(os.path.join(OUT, "expA_metrics.json"), "w") as f:
                json.dump(rows, f, indent=2, default=str)
            print(f"  {mname} seed{ms}: vL2={m['value_rmse']:.4g} "
                  f"piL2={m['policy_rmse']:.4g} eJ={m['cost_gap_eJ']:.4g}",
                  flush=True)

    # ---- surfaces for Fig. 3 at t = 12 h (fine grid + first requested seed) ----
    (Ns, Ny, Nt, V_fd, A_fd) = fd_fine
    grid_f = FDGrid(p, Ns, Ny, Nt)
    k12 = int(round(12.0 / grid_f.dt))
    surf = {"s": grid_f.s, "y": grid_f.y,
            "V_fd": V_fd[k12], "A_fd": A_fd[min(k12, Nt - 2)]}
    for mname in methods:
        surface_seed = seeds_m[0]
        res_path = os.path.join(ckpt_dir("expA", mname,
                                        f"seed{surface_seed}"),
                                "result.json")
        with open(res_path) as f:
            res = json.load(f)
        ck = res.get("selected_checkpoint") or res.get("checkpoint")
        model = load_checkpoint(ck, p, device)
        SS, YY = np.meshgrid(grid_f.s, grid_f.y, indexing="ij")
        tt = torch.full((SS.size,), 12.0, dtype=DTYPE, device=device)
        ss = torch.as_tensor(SS.ravel(), device=device)
        yy = torch.as_tensor(YY.ravel(), device=device)
        pz = torch.zeros_like(ss)
        with torch.no_grad():
            v = model(tt, ss, yy, pz).cpu().numpy().reshape(Ns, Ny)
        a = greedy_action_np(model, tt, ss, yy, pz, prof).reshape(Ns, Ny)
        surf[f"V_{mname}"] = v
        surf[f"A_{mname}"] = a
    np.savez_compressed(os.path.join(OUT, "fig3_surfaces.npz"), **surf)
    finalize_run_manifest(manifest)
    print("Experiment A complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
