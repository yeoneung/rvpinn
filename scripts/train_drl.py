"""Train SAC and TD3 (unified regime-conditioned) in the calibrated
simulator. Checkpoints every eval interval; selection by mean validation
objective across seeds happens in the evaluation script."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.drl_env import MicrogridEnv                     # noqa: E402
from src.exp_common import (ckpt_dir, load_stack,        # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle,
                            seeds_of)
from src.reproducibility import (seed_everything,        # noqa: E402
                                 write_run_manifest, finalize_run_manifest)


def validation_objective(model, params, profs, n_rollouts: int,
                         seed: int) -> float:
    """Mean episode cost on a FIXED validation rollout set (CRN)."""
    env = MicrogridEnv(params, profs, seed=seed)
    costs = []
    for i in range(n_rollouts):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total_r = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(act)
            total_r += r
        costs.append(-total_r * env.reward_scale)
    return float(np.mean(costs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--region", default="primary")
    ap.add_argument("--algos", default="sac,td3")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu is typically faster for these small MLPs and "
                         "leaves the GPU to the float64 PINN jobs")
    args = ap.parse_args()

    cfg = load_stack(args.config)
    regimes = pick_regimes(cfg, args.region)
    seeds_m = ([int(s) for s in args.seeds.split(",")] if args.seeds
               else pick_seeds(cfg))
    params, profs, B0 = regime_bundle(args.region, regimes, cfg)
    seeds = seeds_of(cfg)
    drl_cfg = cfg["drl"]
    total = int(drl_cfg["total_transitions"])
    eval_every = int(drl_cfg["eval_every_transitions"])
    n_eval = int(drl_cfg["n_eval_rollouts"])
    net_arch = list(drl_cfg.get("net_arch", [256, 256]))

    from stable_baselines3 import SAC, TD3

    manifest = os.path.join(ROOT, "results", "logs",
                            f"train_drl_{int(time.time())}.json")
    write_run_manifest(manifest, cfg, seeds)

    for algo_name in args.algos.split(","):
        Algo = {"sac": SAC, "td3": TD3}[algo_name]
        for ms in seeds_m:
            out = ckpt_dir("drl", args.region, algo_name, f"seed{ms}")
            summary_path = os.path.join(out, "training_summary.json")
            if os.path.exists(summary_path):
                print(f"skip {algo_name} seed{ms}")
                continue
            print(f"=== {algo_name} seed={ms} transitions={total} ===",
                  flush=True)
            seed_everything(10_000 + ms)
            env = MicrogridEnv(params, profs, seed=1000 + ms,
                               reward_scale=B0)
            model = Algo("MlpPolicy", env, seed=ms, verbose=0,
                         device=args.device,
                         policy_kwargs={"net_arch": net_arch},
                         learning_starts=min(1000, total // 4))
            history = []
            n_chunks = max(1, total // eval_every)
            best_val = float("inf")
            bad = 0
            patience = int(drl_cfg.get("early_stop_patience_evals", 5))
            for chunk in range(n_chunks):
                model.learn(total_timesteps=eval_every,
                            reset_num_timesteps=False, progress_bar=False)
                val = validation_objective(
                    model, params, profs, n_eval,
                    seed=int(seeds["tuning_validation"]))
                ck = os.path.join(out, f"ck_{(chunk+1)*eval_every}.zip")
                model.save(ck)
                history.append({"transitions": (chunk + 1) * eval_every,
                                "val_objective": val, "checkpoint": ck})
                print(f"  {algo_name} seed{ms} "
                      f"{(chunk+1)*eval_every} -> val {val:.2f}", flush=True)
                if val < best_val - 1e-9:
                    best_val = val
                    bad = 0
                else:
                    bad += 1
                if bad >= patience:
                    print(f"  early stop at "
                          f"{(chunk+1)*eval_every} transitions", flush=True)
                    break
            best = min(history, key=lambda h: h["val_objective"])
            with open(summary_path, "w") as f:
                json.dump({"history": history, "selected": best,
                           "regimes": regimes}, f, indent=2)
    finalize_run_manifest(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
