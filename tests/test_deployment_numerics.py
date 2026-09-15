import numpy as np
import pytest
import torch

from src.costs import RegimeProfile, step_fee_exact
from src.deployment_numerics import (threshold_action_candidates,
                                    recover_solver_action)
from src.dynamics import reflect
from src.evaluation import run_day
from src.reference_policy import ReferenceValueController, reference_one_step_q
from src.safety import deploy_bounds, project_action_to_safe_set


def test_safe_projection_preserves_every_feasible_float(params):
    rng = np.random.default_rng(20260909)
    s = rng.uniform(params.s_min, params.s_max, 10000)
    lo, hi = deploy_bounds(s, params, params.dt_ctrl)
    a = lo + (hi - lo) * rng.random(len(s))
    assert np.array_equal(project_action_to_safe_set(a, s, params,
                                                    params.dt_ctrl), a)
    # The original observed failure: a harmless clip changed a tariff side.
    n = 345.81784079454
    a = np.array([n - 300.0])
    held = project_action_to_safe_set(a, np.array([0.5]), params,
                                      params.dt_ctrl)
    assert held[0] == a[0]
    assert n - held[0] == 300.0


def test_torch_safe_projection_preserves_feasible_float_and_gradient(params):
    a = torch.tensor([45.81784079454001], dtype=torch.float64,
                     requires_grad=True)
    s = torch.tensor([0.5], dtype=torch.float64)
    held = project_action_to_safe_set(a, s, params, params.dt_ctrl)
    assert torch.equal(held, a)
    held.sum().backward()
    assert a.grad.item() == 1.0


def test_reflection_is_bitwise_identity_inside_domain():
    values = np.array([0.0, 1e-12, -40.123456789, 45.81784079454001])
    assert np.array_equal(reflect(values, -536.6, 533.9), values)


@pytest.mark.parametrize('net', [300.0, 345.81784079454, 120.0, 731.346777])
def test_crossing_candidates_really_cover_both_fee_sides(net):
    candidates = threshold_action_candidates(net, 300.0, -500.0, 500.0)
    grid = net - candidates
    assert np.any(grid <= 300.0)
    assert np.any(grid > 300.0)
    assert np.all(candidates >= -500.0)
    assert np.all(candidates <= 500.0)


@pytest.mark.parametrize('net', [-500.0, -200.0, 300.0, 800.0, 1200.0])
def test_crossing_candidates_at_or_outside_endpoints_are_feasible(net):
    candidates = threshold_action_candidates(net, 300.0, -500.0, 500.0)
    assert np.all(np.isfinite(candidates))
    assert np.all((-500.0 <= candidates) & (candidates <= 500.0))


def test_billing_keeps_the_strict_tariff_definition(params):
    params.g_thr, params.c_step = 300.0, 40.0
    grid = np.array([np.nextafter(300.0, -np.inf), 300.0,
                     np.nextafter(300.0, np.inf), 300.01])
    assert np.array_equal(step_fee_exact(grid, params), [0., 0., 40., 40.])


def test_solver_recovery_changes_action_not_cost_classification():
    net = 345.81784079454
    raw = net - 300.0 - 8e-7
    held = recover_solver_action(raw, net, 300.0, -500., 500., False)
    assert net - held <= 300.0
    assert 0.0 < held - raw <= 1e-5
    assert recover_solver_action(raw, net, 300., -500., 500., True) == raw
    # A real violation, or an unavailable feasible crossing, is not forgiven.
    far = net - 300.01
    assert recover_solver_action(far, net, 300., -500., 500., False) == far
    assert recover_solver_action(500., 801., 300., -500., 500., False) == 500.


def test_reference_bills_actual_observation_not_reconstructed_error(params):
    params.g_thr, params.c_step = 300.0, 40.0
    prof = RegimeProfile(np.full(24, 300.), np.full(24, .001))
    actual_net = 345.81784079454
    ctrl = ReferenceValueController(params, prof)
    action = ctrl(0., .5, 0., 0., {'N_now': actual_net, 'C_now': .001})
    assert actual_net - action <= 300.0
    assert abs(action - (actual_net - 300.0)) < 1e-10
    q = reference_one_step_q(params, prof, 0., .5, 0., 0., [action],
                             current_net=actual_net, current_price=.001)
    assert np.isfinite(q).all()


def test_realized_day_no_longer_charges_projection_roundoff(params):
    params.g_thr, params.c_step = 300., 40.
    net = 345.81784079454
    prof = RegimeProfile(np.full(24, net), np.full(24, .001))
    day = {'date': '2019-01-07', 'regime': 'winter_weekday',
           'N': np.full(24, net), 'C': np.full(24, .001)}
    out = run_day(ReferenceValueController(params, prof), day, params, prof,
                  collect_traj=True)
    assert out['traj']['G'][0] <= 300.0
    assert out['exact_action_change_events'] == 0
    assert out['tiny_positive_threshold_hours'] == 0.0
