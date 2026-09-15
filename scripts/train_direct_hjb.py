"""Train direct HJB-PINN baselines with matched architecture and budget."""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch                                             # noqa: E402

from src.exp_common import (ckpt_dir, load_stack, pick_regimes,  # noqa: E402
                            pick_seeds, regime_bundle, seeds_of,
                            trainer_cfg)
from src.policy_iteration import DirectHJBTrainer        # noqa: E402
from src.reproducibility import (seed_everything,        # noqa: E402
                                 write_run_manifest, finalize_run_manifest)

torch.set_default_dtype(torch.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--region", default="primary")
    ap.add_argument("--regimes", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--updates", type=int, default=None,
                    help="total gradient updates (matched to PI budget)")
    args = ap.parse_args()

    cfg = load_stack(args.config)
    regimes = (args.regimes.split(",") if args.regimes
               else pick_regimes(cfg, args.region))
    seeds_m = ([int(s) for s in args.seeds.split(",")] if args.seeds
               else pick_seeds(cfg))
    params, profs, B0 = regime_bundle(args.region, regimes, cfg)
    seeds = seeds_of(cfg)
    # match total gradient updates of the PI run: adam_steps * max_iterations
    if args.updates is None:
        total = int(cfg["policy_evaluation"]["adam_max_steps"]) \
            * int(cfg["policy_iteration"]["max_iterations"])
    else:
        total = args.updates

    manifest = os.path.join(ROOT, "results", "logs",
                            f"train_direct_hjb_{int(time.time())}.json")
    write_run_manifest(manifest, cfg, seeds, extra={"total_updates": total})

    for reg in regimes:
        for ms in seeds_m:
            out = ckpt_dir("direct_hjb", args.region, reg, f"seed{ms}")
            if os.path.exists(os.path.join(out, "result.json")):
                print(f"skip {reg} seed{ms}")
                continue
            print(f"=== direct_hjb regime={reg} seed={ms} "
                  f"updates={total} ===", flush=True)
            seed_everything(int(seeds["training_design"]) + 77 + ms)
            tr = DirectHJBTrainer(params[reg], profs[reg], trainer_cfg(cfg),
                                  seeds, out_scale=B0 / params[reg].T,
                                  use_p=True, method_seed=ms)
            tr.run(out, total_updates=total)
    finalize_run_manifest(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
