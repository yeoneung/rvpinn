"""Shared experiment plumbing: config stacks, regime artifact bundles."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .config import ROOT, load_config
from .regime_profiles import (all_regimes, b0_of, load_calibration,
                              regime_artifacts)


def load_stack(override: Optional[str] = None) -> Dict:
    return load_config(override) if override else load_config()


def trainer_cfg(cfg: Dict) -> Dict:
    keys = ("network", "policy_evaluation", "adaptive_refinement",
            "policy_iteration", "hamiltonian", "policy_selection")
    return {k: cfg[k] for k in keys if k in cfg}


def seeds_of(cfg: Dict) -> Dict:
    return cfg["seeds"]


def regime_bundle(region: str = "primary", regimes: Optional[List[str]] = None,
                  cfg: Optional[Dict] = None):
    """(params_by_regime, profiles_by_regime, B0)."""
    cfg = cfg or load_config()
    cal = load_calibration()
    regs = regimes or all_regimes(region, cal)
    params, profs = {}, {}
    for r in regs:
        p, prof = regime_artifacts(region, r, cfg, cal)
        params[r] = p
        profs[r] = prof
    return params, profs, b0_of(region, cal)


def pick_regimes(cfg: Dict, region: str = "primary") -> List[str]:
    subset = cfg.get("regimes_subset")
    return list(subset) if subset else all_regimes(region)


def pick_seeds(cfg: Dict) -> List[int]:
    subset = cfg.get("method_seeds_subset")
    return list(subset) if subset else list(cfg["seeds"]["method_seeds"])


def ckpt_dir(*parts: str) -> str:
    d = os.path.join(ROOT, "checkpoints", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def results_raw_dir() -> str:
    d = os.path.join(ROOT, "results", "raw")
    os.makedirs(d, exist_ok=True)
    return d
