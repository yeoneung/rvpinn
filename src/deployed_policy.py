"""Deployment-consistent one-step improvement for a learned value function."""
from __future__ import annotations

import numpy as np
import torch

from .costs import common_running_cost_exact
from .dynamics import reflect, soc_step
from .deployment_numerics import threshold_action_candidates
from .safety import cont_bounds, deploy_bounds, project_action_to_safe_set


ACTION_SEARCH_VERSION = "closed-feasible-grid-v1"


def deployed_action_candidates(p, s: float, n_now: float, dt: float,
                               n_grid: int = 1025):
    """Return the common threshold-aware candidate set on A_deploy(s)."""
    lo_arr, hi_arr = deploy_bounds(np.array([s]), p, dt)
    lo, hi = float(lo_arr[0]), float(hi_arr[0])
    frac = np.linspace(0.0, 1.0, int(n_grid))
    # The affine endpoint calculation can round above hi. Confine candidates
    # before objective evaluation, so the selected action is already feasible.
    grid = np.clip(lo + (hi - lo) * frac, lo, hi)
    candidates = np.unique(np.concatenate([
        grid, np.array([lo, hi, np.clip(0.0, lo, hi)]),
        threshold_action_candidates(n_now, p.g_thr, lo, hi)
    ]))
    return candidates


def _ou_joint_nodes(p, dt: float, order: int):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z = np.sqrt(2.0) * nodes
    w = weights / np.sqrt(np.pi)
    z1, z_aux = np.meshgrid(z, z, indexing="ij")
    weights2 = np.outer(w, w).ravel()
    ky, kp = p.kappa_y, p.kappa_p
    sd_y = (p.sigma_y * np.sqrt(dt) if ky <= 1e-12 else
            p.sigma_y * np.sqrt((1.0 - np.exp(-2.0 * ky * dt))
                                / (2.0 * ky)))
    sd_p = (p.sigma_p * np.sqrt(dt) if kp <= 1e-12 else
            p.sigma_p * np.sqrt((1.0 - np.exp(-2.0 * kp * dt))
                                / (2.0 * kp)))
    den = ky + kp
    cov = (p.rho * p.sigma_y * p.sigma_p * dt if den <= 1e-12 else
           p.rho * p.sigma_y * p.sigma_p
           * (1.0 - np.exp(-den * dt)) / den)
    corr = np.clip(cov / max(sd_y * sd_p, 1e-15), -1.0, 1.0)
    z2 = corr * z1 + np.sqrt(max(0.0, 1.0 - corr * corr)) * z_aux
    return (z1.ravel(), z2.ravel(), weights2, sd_y, sd_p)


def deployed_one_step_q(model, t: float, s: float, y: float, pz: float,
                        prof, actions, dt: float = None,
                        quadrature_order: int = 3,
                        current_net=None, current_price=None):
    """Evaluate the hard one-step objective at prescribed feasible actions."""
    p = model.p
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

    t_next = min(float(p.T), float(t) + dt)
    if t_next >= p.T - 1e-12:
        future = p.lam_T * (s_next - p.s_tar) ** 2
    else:
        z1, z2, weights, sd_y, sd_p = _ou_joint_nodes(
            p, dt, quadrature_order)
        y_next = np.exp(-p.kappa_y * dt) * float(y) + sd_y * z1
        p_next = np.exp(-p.kappa_p * dt) * float(pz) + sd_p * z2
        y_next = reflect(y_next, p.y_min, p.y_max)
        p_next = reflect(p_next, p.p_min, p.p_max)
        n_a, n_q = actions.size, weights.size
        device = next(model.parameters()).device
        with torch.no_grad():
            value = model(
                torch.full((n_a * n_q,), t_next,
                           dtype=torch.float64, device=device),
                torch.as_tensor(np.repeat(s_next, n_q),
                                dtype=torch.float64, device=device),
                torch.as_tensor(np.tile(y_next, n_a),
                                dtype=torch.float64, device=device),
                torch.as_tensor(np.tile(p_next, n_a),
                                dtype=torch.float64, device=device),
            ).detach().cpu().numpy().reshape(n_a, n_q)
        future = value @ weights
    return np.asarray(stage + future, dtype=np.float64)


def deployed_one_step_action(model, t: float, s: float, y: float, pz: float,
                             prof, dt: float = None, n_grid: int = 1025,
                             quadrature_order: int = 3,
                             current_net=None, current_price=None):
    """Minimize hard stage cost plus expected learned next value.

    The search is directly over ``A_deploy(s)`` and explicitly contains the
    hard-tariff crossing and its fee-active side.  OU innovations use a fixed
    tensor-product Gauss--Hermite rule and the same reflection map as
    simulation.  The function is scalar by design because it is the online
    held-action routine.
    """
    p = model.p
    dt = float(p.dt_ctrl if dt is None else dt)
    n_now = float(prof.n_bar(np.array([t]))[0]) + float(y)
    if current_net is not None:
        n_now = float(current_net)
    candidates = deployed_action_candidates(p, s, n_now, dt, n_grid)
    q_value = deployed_one_step_q(
        model, t, s, y, pz, prof, candidates, dt=dt,
        quadrature_order=quadrature_order,
        current_net=current_net, current_price=current_price)
    j = int(np.argmin(q_value))
    return float(candidates[j]), float(q_value[j])


def projected_unrestricted_one_step_action(
        model, t: float, s: float, y: float, pz: float, prof,
        dt: float = None, n_grid: int = 1025, quadrature_order: int = 3,
        current_net=None, current_price=None):
    """Legacy ablation: optimize on A_cont and then project to A_deploy."""
    p = model.p
    dt = float(p.dt_ctrl if dt is None else dt)
    lo_arr, hi_arr = cont_bounds(np.array([s]), p)
    lo, hi = float(lo_arr[0]), float(hi_arr[0])
    grid = np.linspace(lo, hi, int(n_grid))
    n_now = float(prof.n_bar(np.array([t]))[0]) + float(y)
    if current_net is not None:
        n_now = float(current_net)
    candidates = np.unique(np.concatenate([
        grid, np.array([lo, hi, np.clip(0.0, lo, hi)]),
        threshold_action_candidates(n_now, p.g_thr, lo, hi)
    ]))
    q_unrestricted = deployed_one_step_q(
        model, t, s, y, pz, prof, candidates, dt=dt,
        quadrature_order=quadrature_order,
        current_net=current_net, current_price=current_price)
    raw_action = float(candidates[int(np.argmin(q_unrestricted))])
    projected = float(project_action_to_safe_set(
        np.array([raw_action]), np.array([s]), p, dt)[0])
    q_projected = float(deployed_one_step_q(
        model, t, s, y, pz, prof, [projected], dt=dt,
        quadrature_order=quadrature_order,
        current_net=current_net, current_price=current_price)[0])
    _, q_direct = deployed_one_step_action(
        model, t, s, y, pz, prof, dt=dt, n_grid=n_grid,
        quadrature_order=quadrature_order,
        current_net=current_net, current_price=current_price)
    return projected, raw_action, max(0.0, q_projected - q_direct)
