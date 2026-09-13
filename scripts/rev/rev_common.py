"""Shared helpers for the first-round revision analyses.

All revision scripts live in scripts/rev/ and write to results/rev/.
Checkpoint paths stored inside result.json files are absolute paths from
the original run location (C:\\Users\\lab\\Downloads\\energy\\...); remap()
rebases them onto the current project root.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "results", "rev")
os.makedirs(OUT, exist_ok=True)


def remap(path: str) -> str:
    """Rebase a stored absolute checkpoint path onto this ROOT."""
    norm = str(path).replace("/", "\\")
    m = re.search(r"\\checkpoints\\(.*)$", norm)
    if m is None:
        return path
    return os.path.join(ROOT, "checkpoints", m.group(1))


def load_result(*parts: str) -> dict:
    with open(os.path.join(ROOT, "checkpoints", *parts, "result.json")) as f:
        return json.load(f)


def selection_by_rms(iters) -> int:
    import numpy as np
    return int(np.argmin([it["tuneval_policy_residual"]["rms"]
                          for it in iters]))


def selection_thresholded(iters, theta: float = 1.25) -> int:
    """Reviewer-1.5 alternative: among iterations whose tuning-validation
    RMS residual is within theta x (minimum RMS), pick the lowest
    common-random-number validation rollout cost."""
    import numpy as np
    rms = np.array([it["tuneval_policy_residual"]["rms"] for it in iters])
    cost = np.array([it["rollout_cost_crn"] for it in iters])
    ok = rms <= theta * rms.min()
    cand = np.where(ok)[0]
    return int(cand[np.argmin(cost[cand])])


def save_json(name: str, obj) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"saved {path}")
    return path
