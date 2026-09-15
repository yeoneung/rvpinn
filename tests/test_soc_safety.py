"""Exact one-step safety of A_Delta and the deployment projection."""
import numpy as np

from src.dynamics import soc_step
from src.safety import cont_bounds, delta_bounds, deploy_bounds, \
    project_action_to_safe_set


def test_endpoint_exactness(params):
    dt = params.dt_ctrl
    # at s_max, charging bound must be exactly 0; at s_min discharging 0
    lo, hi = delta_bounds(np.array([params.s_max]), params, dt)
    assert lo[0] == 0.0
    lo, hi = delta_bounds(np.array([params.s_min]), params, dt)
    assert hi[0] == 0.0
    # extreme feasible actions land exactly on the boundary
    s = np.array([0.88])
    lo, hi = delta_bounds(s, params, dt)
    s_next = soc_step(s, lo, dt, params)   # max charge
    assert np.isclose(s_next[0], params.s_max, atol=1e-12)
    s = np.array([0.12])
    lo, hi = delta_bounds(s, params, dt)
    s_next = soc_step(s, hi, dt, params)   # max discharge
    assert np.isclose(s_next[0], params.s_min, atol=1e-12)


def test_million_pair_stress(params):
    rng = np.random.default_rng(20260731)
    n = 1_000_000
    s = rng.uniform(params.s_min, params.s_max, n)
    a_raw = rng.uniform(-3 * params.a_c, 3 * params.a_d, n)
    dt = params.dt_ctrl
    a = project_action_to_safe_set(a_raw, s, params, dt)
    s_next = soc_step(s, a, dt, params)
    assert np.all(s_next >= params.s_min - 1e-12)
    assert np.all(s_next <= params.s_max + 1e-12)
    # projected actions respect power limits
    assert np.all(a >= -params.a_c - 1e-12)
    assert np.all(a <= params.a_d + 1e-12)


def test_deploy_contains_zero(params):
    rng = np.random.default_rng(7)
    s = rng.uniform(params.s_min, params.s_max, 10000)
    lo, hi = deploy_bounds(s, params, params.dt_ctrl)
    assert np.all(lo <= 1e-15)
    assert np.all(hi >= -1e-15)
    assert np.all(lo <= hi)


def test_taper_vanishes_at_bounds(params):
    lo, hi = cont_bounds(np.array([params.s_min]), params)
    assert hi[0] == 0.0          # no discharge at empty
    lo, hi = cont_bounds(np.array([params.s_max]), params)
    assert lo[0] == 0.0          # no charge at full
