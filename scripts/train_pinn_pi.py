"""Train RV-PINN-PI models per regime and seed."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                             # noqa: E402

from src.exp_common import (ckpt_dir, load_stack, pick_regimes,  # noqa: E402
                            pick_seeds, regime_bundle, seeds_of,
                            trainer_cfg)
from src.policy_iteration import PINNPITrainer           # noqa: E402
from src.reproducibility import (seed_everything,        # noqa: E402
                                 write_run_manifest, finalize_run_manifest)

torch.set_default_dtype(torch.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--region", default="primary")
    ap.add_argument("--regimes", default=None,
                    help="comma-separated; default from config")
    ap.add_argument("--seeds", default=None, help="comma-separated ints")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--tag", default="pinn_pi")
    ap.add_argument("--threshold", type=float, default=300.0)
    ap.add_argument("--fee", type=float, default=40.0)
    ap.add_argument("--width", type=float, default=10.0)
    args = ap.parse_args()

    cfg = load_stack(args.config)
    regimes = (args.regimes.split(",") if args.regimes
               else pick_regimes(cfg, args.region))
    seeds_m = ([int(s) for s in args.seeds.split(",")] if args.seeds
               else pick_seeds(cfg))
    params, profs, B0 = regime_bundle(args.region, regimes, cfg)
    params = {reg: copy.deepcopy(p) for reg, p in params.items()}
    for p in params.values():
        p.g_thr = float(args.threshold)
        p.c_step = float(args.fee)
        p.w_step = float(args.width)
    seeds = seeds_of(cfg)

    manifest = os.path.join(ROOT, "results", "logs",
                            f"train_{args.tag}_{int(time.time())}.json")
    write_run_manifest(manifest, cfg, seeds)

    for reg in regimes:
        p = params[reg]
        prof = profs[reg]
        for ms in seeds_m:
            out = ckpt_dir(args.tag, args.region, reg, f"seed{ms}")
            done = os.path.join(out, "result.json")
            if os.path.exists(done):
                print(f"skip {reg} seed{ms} (already trained)")
                continue
            print(f"=== {args.tag} region={args.region} regime={reg} "
                  f"seed={ms} adaptive={not args.no_adaptive} ===", flush=True)
            seed_everything(int(seeds["training_design"]) + ms)
            tr = PINNPITrainer(p, prof, trainer_cfg(cfg), seeds,
                               out_scale=B0 / p.T, use_p=True,
                               method_seed=ms,
                               adaptive=not args.no_adaptive)
            result = tr.run(out)
            result["tariff"] = {
                "threshold_kw": args.threshold,
                "fee_per_hour": args.fee,
                "smoothing_width_kw": args.width,
            }
            result["region"] = args.region
            result["regime"] = reg
            result["method_seed"] = ms
            with open(done, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, default=str)
    finalize_run_manifest(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
