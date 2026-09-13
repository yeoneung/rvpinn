"""Rule-based baselines B1-B3. Raw actions are always passed through the
shared safe projection by the evaluation engine."""
from __future__ import annotations

import numpy as np

from .config import ModelParams


class NoStorage:
    name = "no_storage"

    def __call__(self, t, s, y, pz, ctx):
        return 0.0


class SelfConsumption:
    """Charge renewable surplus (N < 0), discharge against imports (N > 0)."""
    name = "self_consumption"

    def __call__(self, t, s, y, pz, ctx):
        N = ctx["N_now"]
        return float(N)          # a = N offsets the grid exchange exactly


class PriceThreshold:
    """Charge below the lower training-price quantile, discharge above the
    upper quantile. Quantiles tuned on the validation period only."""
    name = "threshold_rule"

    def __init__(self, p: ModelParams, c_lo: float, c_hi: float):
        self.p = p
        self.c_lo = float(c_lo)
        self.c_hi = float(c_hi)

    def __call__(self, t, s, y, pz, ctx):
        C = ctx["C_now"]
        if C <= self.c_lo:
            return -self.p.a_c
        if C >= self.c_hi:
            return self.p.a_d
        return 0.0
