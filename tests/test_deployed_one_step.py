import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.deployed_policy import (deployed_one_step_action,
                                 deployed_action_candidates,
                                 deployed_one_step_q,
                                 projected_unrestricted_one_step_action)
from src.dynamics import soc_step
from src.pinn_value import ValueNet
from src.safety import deploy_bounds, project_action_to_safe_set


@pytest.mark.parametrize('state', [
    0.10604776311973389, 0.13752854290594674, 0.138373438604247,
])
def test_candidate_mesh_stays_inside_float_endpoints(params, state):
    # Regression states from the endpoint-rounding audit. In each case,
    # lo + (hi - lo) is one or more ulps above hi.
    lo, hi = deploy_bounds(np.array([state]), params, params.dt_ctrl)
    candidates = deployed_action_candidates(params, state, 400., params.dt_ctrl)
    assert np.all(np.isfinite(candidates))
    assert candidates[0] == lo[0]
    assert candidates[-1] == hi[0]
    assert np.all((lo[0] <= candidates) & (candidates <= hi[0]))
    held = project_action_to_safe_set(
        candidates, np.full_like(candidates, state), params, params.dt_ctrl)
    assert np.array_equal(held, candidates)


def test_endpoint_optimum_needs_no_post_search_correction(params, profile,
                                                         monkeypatch):
    monkeypatch.setattr('src.deployed_policy.deployed_one_step_q',
                        lambda model, t, s, y, pz, prof, actions, **kw:
                        -np.asarray(actions))
    state = 0.10604776311973389
    action, _ = deployed_one_step_action(
        SimpleNamespace(p=params), 0., state, 0., 0., profile)
    held = project_action_to_safe_set(
        np.array([action]), np.array([state]), params, params.dt_ctrl)[0]
    assert action == held


def test_candidate_mesh_is_closed_across_soc_and_tariff_locations(params):
    rng = np.random.default_rng(20260909)
    states = np.r_[params.s_min, params.s_max,
                   np.nextafter(params.s_min, params.s_max),
                   np.nextafter(params.s_max, params.s_min),
                   rng.uniform(params.s_min, params.s_max, 1000)]
    for state in states:
        lo, hi = deploy_bounds(np.array([state]), params, params.dt_ctrl)
        net = float(params.g_thr + rng.uniform(lo[0]-1., hi[0]+1.))
        candidates = deployed_action_candidates(params, state, net, params.dt_ctrl)
        assert candidates[0] == lo[0]
        assert candidates[-1] == hi[0]
        assert 0. in candidates
        assert np.all((lo[0] <= candidates) & (candidates <= hi[0]))


def test_one_step_search_is_feasible_and_recovers_terminal_target(
        params, profile):
    params.c_step = 0.0
    params.lam1 = 0.0
    params.lam_T = 10000.0
    model = ValueNet(params, width=8, depth=1, use_p=True,
                     reference_baseline="terminal_only", seed=4)
    # Remove the random correction so the next value at T is exactly terminal.
    for layer in model.layers:
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    s = 0.55
    action, objective = deployed_one_step_action(
        model, params.T - params.dt_ctrl, s, 0.0, 0.0, profile,
        n_grid=4097)
    reevaluated = deployed_one_step_q(
        model, params.T - params.dt_ctrl, s, 0.0, 0.0, profile, [action])
    assert np.isclose(objective, reevaluated[0])
    lo, hi = deploy_bounds(np.array([s]), params, params.dt_ctrl)
    assert lo[0] <= action <= hi[0]
    next_soc = float(soc_step(s, action, params.dt_ctrl, params))
    # Energy cost perturbs the exact target slightly, but the high common
    # terminal charge must prevent the depletion seen with a myopic action.
    assert abs(next_soc - params.s_tar) < 0.005

    projected, _, defect = projected_unrestricted_one_step_action(
        model, params.T - params.dt_ctrl, s, 0.0, 0.0, profile,
        n_grid=1025)
    assert lo[0] <= projected <= hi[0]
    assert defect >= 0.0
