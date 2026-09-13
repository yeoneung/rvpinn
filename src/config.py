"""Configuration loading and the physical parameter container.

All numerical experiment code receives a single ModelParams instance built
from configs/model.yaml plus regime-calibrated quantities. Every field is a
plain float so the object can be consumed by numpy, torch, and cvxpy code
without conversion surprises. Units: power kW, energy kWh, price EUR/kWh,
time hours.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(*paths: str) -> Dict[str, Any]:
    """Merge the standard config stack, then any extra override files."""
    cfg: Dict[str, Any] = {}
    standard = ["model.yaml", "data.yaml", "pinn_pi.yaml", "baselines.yaml",
                "experiments.yaml"]
    for name in standard:
        p = os.path.join(ROOT, "configs", name)
        if os.path.exists(p):
            cfg = deep_merge(cfg, load_yaml(p))
    for p in paths:
        cfg = deep_merge(cfg, load_yaml(p))
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@dataclasses.dataclass
class ModelParams:
    # battery
    E_max: float = 1000.0
    a_c: float = 500.0
    a_d: float = 500.0
    s_min: float = 0.10
    s_max: float = 0.90
    s0: float = 0.50
    s_tar: float = 0.50
    s_ref: float = 0.50
    eta_c: float = 0.95
    eta_d: float = 0.95
    delta_s: float = 0.05
    # horizon (hours)
    T: float = 24.0
    dt_ctrl: float = 0.25
    dt_sim: float = 5.0 / 60.0
    # cost
    alpha_s: float = 0.50
    lam_pk: float = 0.0
    lam1: float = 0.0375        # 300/(2*4000) EUR/kWh throughput proxy
    lam2: float = 0.0
    lam_s: float = 0.0
    lam_T: float = 0.0
    eps_g: float = 0.5          # 1e-3 * P_base
    eps_a: float = 0.5          # 1e-3 * max(a_c, a_d)
    # Nonconvex capacity-band tariff;
    # c_step = 0 disables the term everywhere
    c_step: float = 0.0         # currency/h surcharge while G > g_thr
    g_thr: float = 250.0        # kW import threshold
    w_step: float = 10.0        # kW smoothing width (training side)
    # OU (per regime)
    kappa_y: float = 1.0
    sigma_y: float = 50.0
    kappa_p: float = 1.0
    sigma_p: float = 0.02
    rho: float = 0.3
    y_min: float = -250.0
    y_max: float = 250.0
    p_min: float = -0.05
    p_max: float = 0.05

    @property
    def a_max(self) -> float:
        return max(self.a_c, self.a_d)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any],
                    calibrated: Optional[Dict[str, Any]] = None,
                    regime: Optional[str] = None) -> "ModelParams":
        b = cfg.get("battery", {})
        h = cfg.get("horizon", {})
        c = cfg.get("cost", {})
        p = cls(
            E_max=float(b.get("E_max_kwh", 1000.0)),
            a_c=float(b.get("a_c_max_kw", 500.0)),
            a_d=float(b.get("a_d_max_kw", 500.0)),
            s_min=float(b.get("s_min", 0.10)),
            s_max=float(b.get("s_max", 0.90)),
            s0=float(b.get("s0", 0.50)),
            s_tar=float(b.get("s_target", 0.50)),
            s_ref=float(b.get("s_ref", 0.50)),
            eta_c=float(b.get("eta_c", 0.95)),
            eta_d=float(b.get("eta_d", 0.95)),
            delta_s=float(b.get("delta_s", 0.05)),
            T=float(h.get("T_hours", 24.0)),
            dt_ctrl=float(h.get("control_dt_min", 15.0)) / 60.0,
            dt_sim=float(h.get("sim_dt_min", 5.0)) / 60.0,
            alpha_s=float(c.get("alpha_s", 0.50)),
            lam_s=float(c.get("lambda_s", 0.0)),
        )
        rc = float(c.get("replacement_cost_per_kwh", 300.0))
        cl = float(c.get("cycle_life_efc", 4000.0))
        p.lam1 = rc / (2.0 * cl)
        P_base = float(cfg.get("microgrid", {}).get("P_base_kw", 500.0))
        p.eps_g = float(c.get("eps_g_fraction", 1e-3)) * P_base
        p.eps_a = float(c.get("eps_a_fraction", 1e-3)) * p.a_max
        if calibrated:
            for k in ("lam_pk", "lam1", "lam2", "lam_T"):
                if k in calibrated:
                    setattr(p, k, float(calibrated[k]))
            reg = calibrated.get("regimes", {}).get(regime) if regime else None
            if reg:
                for k in ("kappa_y", "sigma_y", "kappa_p", "sigma_p", "rho",
                          "y_min", "y_max", "p_min", "p_max"):
                    if k in reg:
                        setattr(p, k, float(reg[k]))
        return p
