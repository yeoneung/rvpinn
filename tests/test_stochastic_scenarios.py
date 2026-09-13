import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("pyscipopt")

HERE = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(HERE))
from stochastic_miqp import ScenarioExactMIPController  # noqa: E402


def test_scenarios_share_observed_first_stage_and_are_reproducible(
        params, profile):
    controller = ScenarioExactMIPController(
        params, n_scenarios=6, reopt_every_hours=params.dt_ctrl,
        time_limit_s=1.0)
    ctx = {
        "date": "2019-02-01",
        "N_forecast": lambda t: profile.n_bar(np.asarray(t)),
        "C_forecast": lambda t: profile.c_bar(np.asarray(t)),
    }
    n1, c1 = controller._forecast_scenarios(
        3.0, 12, 25.0, -0.01, ctx)
    n2, c2 = controller._forecast_scenarios(
        3.0, 12, 25.0, -0.01, ctx)
    assert n1.shape == (6, 12) and c1.shape == (6, 12)
    assert np.allclose(n1, n2) and np.allclose(c1, c2)
    assert np.max(np.abs(n1[:, 0] - n1[0, 0])) < 1e-12
    assert np.max(np.abs(c1[:, 0] - c1[0, 0])) < 1e-12
    y = n1 - profile.n_bar(3.0 + params.dt_ctrl * np.arange(12))[None, :]
    pe = c1 - profile.c_bar(3.0 + params.dt_ctrl * np.arange(12))[None, :]
    assert np.all(y >= params.y_min) and np.all(y <= params.y_max)
    assert np.all(pe >= params.p_min) and np.all(pe <= params.p_max)


def test_round2_scenario_sets_are_nested_across_counts(params, profile):
    ctx = {
        "date": "2019-02-01",
        "N_forecast": lambda t: profile.n_bar(np.asarray(t)),
        "C_forecast": lambda t: profile.c_bar(np.asarray(t)),
    }
    forecasts = []
    for count in (2, 8, 16):
        controller = ScenarioExactMIPController(
            params, n_scenarios=count, scenario_pool_size=16,
            random_seed=7301, reopt_every_hours=params.dt_ctrl,
            time_limit_s=1.0)
        forecasts.append(controller._forecast_scenarios(
            3.0, 12, 25.0, -0.01, ctx))
    for smaller, larger in zip(forecasts, forecasts[1:]):
        n_small, c_small = smaller
        n_large, c_large = larger
        assert np.allclose(n_small, n_large[:len(n_small)])
        assert np.allclose(c_small, c_large[:len(c_small)])
