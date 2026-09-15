"""Floating-point implementation of the unchanged strict hard-band tariff.

No tolerance is applied to billing: G > threshold remains fee-active.
Candidate construction and solver action recovery instead preserve the
intended side of a crossing before the action is held.
"""
from __future__ import annotations

import numpy as np

DEPLOYMENT_VERSION = "hard-band-aligned-v1"
SOLVER_FEASIBILITY_TOL = 1e-9
BAND_SEPARATION_KW = 1e-5


def threshold_action_candidates(n_now, threshold, lo, hi):
    """Crossing and feasible candidates verified on each side of G=N-a.

    nextafter(action) alone need not change N-action (notably when action is
    zero). In that case, construct the candidate from the adjacent *grid*
    value. Side membership is checked after subtraction, not assumed.
    """
    n_now, threshold, lo, hi = map(float, (n_now, threshold, lo, hi))
    crossing = n_now - threshold
    candidates = [float(np.clip(crossing, lo, hi))]
    if lo <= crossing <= hi:
        inactive = crossing
        if n_now - inactive > threshold:
            inactive = n_now - np.nextafter(threshold, -np.inf)
        for _ in range(4):
            if n_now - inactive <= threshold:
                break
            inactive = np.nextafter(inactive, np.inf)
        if lo <= inactive <= hi and n_now - inactive <= threshold:
            candidates.append(float(inactive))

        active = np.nextafter(crossing, -np.inf)
        if n_now - active <= threshold:
            active = n_now - np.nextafter(threshold, np.inf)
        for _ in range(4):
            if n_now - active > threshold:
                break
            active = np.nextafter(active, -np.inf)
        if lo <= active <= hi and n_now - active > threshold:
            candidates.append(float(active))
    return np.unique(np.asarray(candidates, dtype=np.float64))


def recover_solver_action(action, n_now, threshold, lo, hi,
                          intended_band_active,
                          recovery_limit=BAND_SEPARATION_KW):
    """Restore a solver's inactive-band first action within numerical limits.

    First apply the common feasible interval. Only an intended inactive
    binary and an overshoot no greater than the documented binary separation
    permit recovery to a verified feasible crossing. Real exceedances and
    active-band solutions are not reclassified; costs use the returned action.
    """
    held = float(np.clip(action, lo, hi))
    overshoot = float(n_now) - held - float(threshold)
    if not intended_band_active and 0.0 < overshoot <= recovery_limit:
        candidates = threshold_action_candidates(n_now, threshold, lo, hi)
        inactive = candidates[float(n_now) - candidates <= float(threshold)]
        if inactive.size:
            repaired = float(inactive[np.argmin(np.abs(inactive - held))])
            if abs(repaired - held) <= recovery_limit:
                held = repaired
    return held
