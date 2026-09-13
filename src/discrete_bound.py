"""Deployment-aligned finite-horizon Bellman audit.

The routines operate on finite state/action arrays and mirror the proposition
used in the version-3 paper.  They are deliberately elementary: the result is
a telescoping a-posteriori bound, not a new convergence theorem or a
continuous-state certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class BellmanAudit:
    residual_oscillation: np.ndarray
    deployed_action_defect: np.ndarray
    terminal_error: float
    bound_from_time: np.ndarray


def _validate(costs: Sequence[np.ndarray], transitions: Sequence[np.ndarray],
              values: Sequence[np.ndarray], policy: Sequence[np.ndarray],
              terminal_cost: np.ndarray) -> None:
    K = len(costs)
    if len(transitions) != K or len(policy) != K or len(values) != K + 1:
        raise ValueError("inconsistent horizon lengths")
    n_state = np.asarray(terminal_cost).shape[0]
    for k, (cost, trans, act) in enumerate(zip(costs, transitions, policy)):
        if cost.ndim != 2 or trans.ndim != 3:
            raise ValueError(f"stage {k}: expected cost[S,A], P[S,A,S]")
        if cost.shape != trans.shape[:2] or trans.shape[2] != n_state:
            raise ValueError(f"stage {k}: incompatible cost/transition shape")
        if np.asarray(values[k]).shape != (n_state,):
            raise ValueError(f"stage {k}: incompatible value shape")
        if np.asarray(act).shape != (n_state,):
            raise ValueError(f"stage {k}: incompatible policy shape")
        if np.any(np.asarray(act) < 0) or np.any(np.asarray(act) >= cost.shape[1]):
            raise ValueError(f"stage {k}: policy action outside action set")
        if not np.allclose(trans.sum(axis=2), 1.0, atol=1e-12):
            raise ValueError(f"stage {k}: transition rows do not sum to one")


def audit_bound(costs: Sequence[np.ndarray],
                transitions: Sequence[np.ndarray],
                values: Sequence[np.ndarray],
                policy: Sequence[np.ndarray],
                terminal_cost: np.ndarray) -> BellmanAudit:
    """Compute residual oscillations, held-action defects, and loss bound.

    `policy[k][x]` is the action actually held after all deployment/safety
    logic.  Therefore its defect captures action discretization, safety
    modification, tariff-switch errors, and any other implementation mismatch.
    """
    costs = [np.asarray(x, dtype=np.float64) for x in costs]
    transitions = [np.asarray(x, dtype=np.float64) for x in transitions]
    values = [np.asarray(x, dtype=np.float64) for x in values]
    policy = [np.asarray(x, dtype=np.int64) for x in policy]
    terminal_cost = np.asarray(terminal_cost, dtype=np.float64)
    _validate(costs, transitions, values, policy, terminal_cost)

    K = len(costs)
    oscillation = np.empty(K, dtype=np.float64)
    defect = np.empty(K, dtype=np.float64)
    for k in range(K):
        q = costs[k] + np.einsum("xay,y->xa", transitions[k], values[k + 1])
        tv = q.min(axis=1)
        residual = tv - values[k]
        held_q = q[np.arange(q.shape[0]), policy[k]]
        held_defect = held_q - tv
        oscillation[k] = residual.max() - residual.min()
        defect[k] = max(0.0, float(held_defect.max()))

    terminal_error = float(np.max(np.abs(terminal_cost - values[-1])))
    bound = np.empty(K + 1, dtype=np.float64)
    bound[K] = 2.0 * terminal_error
    for k in range(K - 1, -1, -1):
        bound[k] = bound[k + 1] + oscillation[k] + defect[k]
    return BellmanAudit(oscillation, defect, terminal_error, bound)


def solve_finite_mdp(costs: Sequence[np.ndarray],
                     transitions: Sequence[np.ndarray],
                     terminal_cost: np.ndarray,
                     policy: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
    """Exact backward recursion for testing/auditing a finite MDP."""
    terminal_cost = np.asarray(terminal_cost, dtype=np.float64)
    K = len(costs)
    out = np.empty((K + 1, terminal_cost.size), dtype=np.float64)
    out[K] = terminal_cost
    for k in range(K - 1, -1, -1):
        q = (np.asarray(costs[k], dtype=np.float64)
             + np.einsum("xay,y->xa", transitions[k], out[k + 1]))
        if policy is None:
            out[k] = q.min(axis=1)
        else:
            act = np.asarray(policy[k], dtype=np.int64)
            out[k] = q[np.arange(q.shape[0]), act]
    return out
