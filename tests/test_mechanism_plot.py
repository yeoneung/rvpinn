import numpy as np
import pytest

from src.config import ModelParams
from src.dynamics import b_S
import pandas as pd
from experiments.make_round2_mechanism_figure import (
    RecordedActionController, _check_reported_economics, _soc_plot_coordinates)
from experiments.verify_round2_results import check_mechanism_layout


def mechanism_layout():
    keys = ['learned', 'reference', 'm2', 'm16']
    metadata = {'fee_per_hour': 40., 'threshold_kw': 300., 'scenario_count': 16,
                'scenario_stream': 7301, 'scenario_pool_size': 16, 'time_limit_s': 5.,
                'primary_scenario_count': 2, 'primary_scenario_stream': 4101,
                'learned_batch': 0}
    return (metadata, pd.DataFrame({'controller': keys}),
            pd.DataFrame({'controller': np.repeat(keys, 96)}))


def test_mechanism_requires_exactly_the_four_reported_curves():
    metadata, summary, trajectory = mechanism_layout()
    check_mechanism_layout(metadata, summary, trajectory)
    with pytest.raises(RuntimeError, match='incomplete'):
        check_mechanism_layout(metadata, summary,
            pd.concat([trajectory, pd.DataFrame({'controller': ['extra'] * 96})]))
    with pytest.raises(RuntimeError, match='incomplete'):
        check_mechanism_layout(metadata, summary, trajectory.iloc[:-1])


@pytest.mark.parametrize('key,wrong', [('primary_scenario_count', 8),
    ('primary_scenario_stream', 7301), ('learned_batch', 1), ('time_limit_s', 15.)])
def test_mechanism_cannot_change_primary_stream_batch_or_budget(key, wrong):
    metadata, summary, trajectory = mechanism_layout()
    with pytest.raises(RuntimeError, match='setting differs'):
        check_mechanism_layout(dict(metadata, **{key: wrong}), summary, trajectory)


def test_recorded_miqp_actions_and_missing_incumbent_are_replayed_exactly():
    records = pd.DataFrame({'time': np.arange(96) / 4.,
                            'has_solution': [False] + [True] * 95,
                            'first_action_held': [np.nan] + list(np.arange(95) / 3.)})
    controller = RecordedActionController(records.iloc[::-1])
    assert controller(0., None, None, None, None) == 0.
    assert controller(.25, None, None, None, None) == 0.
    assert controller(23.75, None, None, None, None) == 94. / 3.
    with pytest.raises(RuntimeError, match='complete decision clock'):
        RecordedActionController(records.iloc[:-1])
    with pytest.raises(RuntimeError, match='outside'):
        controller(24., None, None, None, None)


def test_mechanism_cannot_replace_reported_cost_with_a_fresh_solver_outcome():
    result = dict(bill=10., capacity_fee=20., degradation_cost=3.,
                  terminal_penalty=1., common_cost=34., exceed_hours=.5,
                  throughput_kwh=80., terminal_soc=.5, exact_action_change_events=0)
    _check_reported_economics(result, result)
    with pytest.raises(RuntimeError, match='reported common_cost'):
        _check_reported_economics(result, dict(result, common_cost=33.))


def test_soc_plot_uses_control_boundaries_not_first_substep_times():
    p = ModelParams()
    times = np.arange(0., p.T, p.dt_ctrl)
    actions = 5.0 * np.sin(times)
    drift = b_S(actions, p)
    boundaries = np.r_[p.s0, p.s0 + np.cumsum(p.dt_ctrl * drift)]
    raw = {'t': times, 'a': actions,
           's': boundaries[:-1] + p.dt_sim * drift}
    plot_t, plot_s = _soc_plot_coordinates(raw, boundaries[-1], p)
    assert np.allclose(plot_t, np.r_[times, p.T])
    assert np.allclose(plot_s, boundaries, atol=1e-12, rtol=0)
    assert plot_s[0] == p.s0 and plot_s[-1] == boundaries[-1]
    with pytest.raises(RuntimeError, match='held-action dynamics'):
        _soc_plot_coordinates(raw, boundaries[-1] + 0.01, p)
