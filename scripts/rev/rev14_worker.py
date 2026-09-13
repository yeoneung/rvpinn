"""Parallel worker: trains the lambda_pk x100 and x10 variants (seed 0)
while the main sensitivity sweep works through the queue from the front.
Uses the same train_variant cache directories, disjoint from the main
process's current position, so there is no write collision."""
from __future__ import annotations

import copy
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT                    # noqa: E402

import torch                                               # noqa: E402

from scripts.run_ablations import train_variant            # noqa: E402
from src.exp_common import load_stack, pick_seeds, regime_bundle, \
    seeds_of                                               # noqa: E402

torch.set_default_dtype(torch.float64)
REGIME = "winter_weekday"


def main() -> int:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds = seeds_of(cfg)
    ms = pick_seeds(cfg)[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params, profs, B0 = regime_bundle("primary", [REGIME], cfg)
    p, prof = params[REGIME], profs[REGIME]
    for fac in (100.0, 10.0):
        pv = copy.deepcopy(p)
        pv.lam_pk = p.lam_pk * fac
        t0 = time.time()
        train_variant(f"rev14_lampk_{fac}", pv, prof, cfg, seeds, B0, ms,
                      device, use_p=True)
        print(f"worker: lampk x{fac} seed{ms} done in "
              f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
