"""Positive action (discharge) must decrease SoC and grid import."""
import numpy as np

from src.costs import grid_exchange
from src.dynamics import b_S, soc_step


def test_positive_action_decreases_soc(params):
    s0 = 0.5
    s1 = soc_step(s0, 100.0, 0.25, params)
    assert s1 < s0
    # exact value: 0.5 - 0.25*100/(0.95*1000)
    assert np.isclose(s1, 0.5 - 0.25 * 100.0 / (0.95 * 1000.0))


def test_negative_action_increases_soc(params):
    s0 = 0.5
    s1 = soc_step(s0, -100.0, 0.25, params)
    assert s1 > s0
    assert np.isclose(s1, 0.5 + 0.25 * 0.95 * 100.0 / 1000.0)


def test_positive_action_decreases_grid_import(params, profile):
    t, y = 12.0, 50.0
    g0 = grid_exchange(t, y, 0.0, profile)
    g1 = grid_exchange(t, y, 100.0, profile)
    assert g1 == g0 - 100.0


def test_drift_signs(params):
    assert b_S(200.0, params) < 0       # discharge drains
    assert b_S(-200.0, params) > 0      # charge fills
    assert b_S(0.0, params) == 0.0


def test_efficiency_asymmetry(params):
    # discharging 100 kW drains SoC faster than charging 100 kW fills it
    assert abs(b_S(100.0, params)) > abs(b_S(-100.0, params))
