"""Load calibrated artifacts into (ModelParams, RegimeProfile) per regime."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from .config import ModelParams, ROOT, load_config
from .costs import RegimeProfile

CAL_PATH = os.path.join(ROOT, "data", "calibrated_parameters.json")


def load_calibration(path: Optional[str] = None) -> Dict:
    with open(path or CAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def regime_artifacts(region: str = "primary", regime: str = "winter_weekday",
                     cfg: Optional[Dict] = None,
                     cal: Optional[Dict] = None,
                     include_legacy_regularizers: bool = False
                     ) -> Tuple[ModelParams, RegimeProfile]:
    cfg = cfg or load_config()
    cal = cal or load_calibration()
    rc = cal[region]
    coeff = rc["coefficients"]
    reg = rc["regimes"][regime]
    p = ModelParams.from_config(cfg)
    # Learning, online optimization, selection, and reporting share the common
    # economic objective. Quadratic peak and action terms in the calibration
    # artifact are enabled only for explicitly requested regularizer ablations.
    p.lam_pk = (float(coeff["lam_pk"])
                if include_legacy_regularizers else 0.0)
    p.lam1 = float(coeff["lam1"])
    p.lam2 = (float(coeff["lam2"])
              if include_legacy_regularizers else 0.0)
    p.lam_T = float(coeff["lam_T"])
    p.lam_s = float(coeff.get("lam_s", 0.0))
    p.eps_g = float(coeff["eps_g"])
    p.eps_a = float(coeff["eps_a"])
    p.alpha_s = float(coeff["alpha_s"])
    for k in ("kappa_y", "sigma_y", "kappa_p", "sigma_p", "rho",
              "y_min", "y_max", "p_min", "p_max"):
        setattr(p, k, float(reg[k]))
    prof = RegimeProfile(reg["n_bar"], reg["c_bar"], T=p.T)
    return p, prof


def all_regimes(region: str = "primary", cal: Optional[Dict] = None):
    cal = cal or load_calibration()
    return sorted(cal[region]["regimes"].keys())


def b0_of(region: str = "primary", cal: Optional[Dict] = None) -> float:
    cal = cal or load_calibration()
    return float(cal[region]["coefficients"]["B0"])
