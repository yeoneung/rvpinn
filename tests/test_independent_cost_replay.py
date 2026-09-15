"""Independently replay the solver action log, including no-incumbent fallbacks."""
import numpy as np
import pandas as pd
import pytest

from experiments.audit_release_inputs import replay_solver_costs
from src.evaluation import run_day


def recorded_example(params, profile, fallback=False):
    params.c_step = 40.
    params.g_thr = 300.
    day = {'date': '2019-01-02', 'regime': 'winter_weekday',
           'N': np.linspace(180., 410., 24), 'C': np.linspace(-0.02, 0.12, 24)}
    actions = np.zeros(96) if fallback else np.tile([-10., -10., 8., 8.], 24)

    class ReplayController:
        def __call__(self, t, s, y, pz, ctx):
            return actions[int(round(t / params.dt_ctrl))]

    daily = pd.DataFrame([{'method': 'test_miqp', 'threshold_kw': 300.,
                           'fee_per_hour': 40., **run_day(ReplayController(), day, params, profile)}])
    solves = pd.DataFrame({'method': 'test_miqp', 'date': day['date'],
                           'time': np.arange(96) / 4., 'has_solution': not fallback,
                           'first_action_held': np.full(96, np.nan) if fallback else actions})
    return solves, daily, {day['date']: day}, {'winter_weekday': params}


@pytest.mark.parametrize('fallback', [False, True])
def test_logged_actions_reproduce_all_cost_components(params, profile, fallback):
    assert replay_solver_costs(*recorded_example(params, profile, fallback)) == 1


@pytest.mark.parametrize('field', ['bill', 'capacity_fee', 'terminal_soc', 'common_cost'])
def test_logged_action_replay_rejects_incorrect_economic_summary(params, profile, field):
    solves, daily, days, p = recorded_example(params, profile)
    daily.loc[0, field] += 0.001
    with pytest.raises(RuntimeError, match=f'mismatch in {field}'):
        replay_solver_costs(solves, daily, days, p)


def test_logged_action_replay_does_not_assume_unrecorded_safety_changes(params, profile):
    solves, daily, days, p = recorded_example(params, profile)
    daily.loc[0, 'exact_action_change_events'] = 1
    with pytest.raises(RuntimeError, match='post-call safety change'):
        replay_solver_costs(solves, daily, days, p)


def test_logged_action_replay_checks_within_day_feasibility(params, profile):
    solves, daily, days, p = recorded_example(params, profile)
    solves.loc[:, 'first_action_held'] = params.a_d
    with pytest.raises(RuntimeError, match='outside replayed safe set'):
        replay_solver_costs(solves, daily, days, p)
