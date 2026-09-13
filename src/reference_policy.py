"""No-learning reference-value controller for the round-2 attribution audit."""
from __future__ import annotations

import numpy as np

from .costs import common_running_cost_exact
from .deployed_policy import ACTION_SEARCH_VERSION, deployed_action_candidates
from .dynamics import soc_step


def reference_one_step_q(p, prof, t: float, s: float, y: float, pz: float,
                         actions, dt: float = None,
                         current_net=None, current_price=None):
    """Hard stage cost plus the action-dependent reference continuation.

    With the version-3 common objective, the conditional-mean no-storage
    operating-cost term is independent of the battery action. It therefore
    cancels from the minimization, leaving only the terminal part of the
    reference value after the held-action state transition.
    """
    dt = float(p.dt_ctrl if dt is None else dt)
    actions = np.atleast_1d(np.asarray(actions, dtype=np.float64))
    n_now = float(prof.n_bar(np.array([t]))[0]) + float(y)
    c_now = float(prof.c_bar(np.array([t]))[0]) + float(pz)
    if current_net is not None:
        n_now = float(current_net)
    if current_price is not None:
        c_now = float(current_price)
    stage = dt * common_running_cost_exact(
        c_now, n_now - actions, actions, p)
    s_next = soc_step(np.full_like(actions, s), actions, dt, p)
    continuation = p.lam_T * (s_next - p.s_tar) ** 2
    return np.asarray(stage + continuation, dtype=np.float64)


def reference_one_step_action(p, prof, t: float, s: float, y: float,
                              pz: float, dt: float = None,
                              n_grid: int = 1025,
                              current_net=None, current_price=None):
    """Minimize the reference-value objective on the common deployed set."""
    dt = float(p.dt_ctrl if dt is None else dt)
    n_now = float(prof.n_bar(np.array([t]))[0]) + float(y)
    if current_net is not None:
        n_now = float(current_net)
    candidates = deployed_action_candidates(p, s, n_now, dt, n_grid)
    values = reference_one_step_q(
        p, prof, t, s, y, pz, candidates, dt=dt,
        current_net=current_net, current_price=current_price)
    index = int(np.argmin(values))
    return float(candidates[index]), float(values[index])


class ReferenceValueController:
    """Hard one-step deployment using the reference value without learning."""

    name = "reference_value_one_step"
    action_search_version = ACTION_SEARCH_VERSION

    def __init__(self, p, prof, n_grid: int = 1025):
        self.p = p
        self.prof = prof
        self.n_grid = int(n_grid)

    def __call__(self, t, s, y, pz, ctx):
        action, _ = reference_one_step_action(
            self.p, self.prof, float(t), float(s), float(y), float(pz),
            dt=self.p.dt_ctrl, n_grid=self.n_grid,
            current_net=ctx.get("N_now"), current_price=ctx.get("C_now"))
        return action
