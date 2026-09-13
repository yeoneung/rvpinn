import json

import numpy as np
import pandas as pd
import pytest

from experiments import audit_release_inputs as audit
from src.deployed_policy import ACTION_SEARCH_VERSION
from src.deployment_numerics import DEPLOYMENT_VERSION


@pytest.mark.parametrize('tag,expected', [
    ('v3_winter_fee80', (300, 80, 10)),
    ('v3_summer_fee40', (300, 40, 10)),
    ('v3_sens_t250_f20_w5', (250, 20, 5)),
])
def test_validation_accounting_uses_the_frozen_run_tariff(tag, expected):
    assert audit.selection_tariff(tag) == expected


def test_unknown_validation_tag_cannot_silently_use_the_central_fee():
    with pytest.raises(RuntimeError, match='unrecognized validation selection tag'):
        audit.selection_tariff('development_unspecified')


def test_tariff_and_solver_budget_metadata_are_not_inferred_from_filename():
    frame = pd.DataFrame([{
        'method': audit.ONLINE[2], 'region': 'confirmatory',
        'threshold_kw': 300., 'fee_per_hour': 40., 'smoothing_width_kw': 10.,
        'control_dt_min': 15., 'solver_time_limit_s': 5.,
        'n_scenarios': 8, 'scenario_pool_size': 16, 'scenario_stream_seed': 7301,
    }])
    name = 'common_daily_confirmatory_thr300_fee40_w10_n8_r2_stream7301_pool16_tl5.parquet'
    audit.check_online_parameters(frame, name)
    for column, wrong in (('fee_per_hour', 80.), ('threshold_kw', 250.),
                          ('solver_time_limit_s', 15.), ('n_scenarios', 16),
                          ('scenario_stream_seed', 7302), ('scenario_pool_size', 8)):
        damaged = frame.copy()
        damaged[column] = wrong
        with pytest.raises(RuntimeError):
            audit.check_online_parameters(damaged, name)


def selection_example(tmp_path, monkeypatch, params, fallback=False):
    tag, regime = 'v3_winter_fee40', 'winter_weekday'
    costs = [11., 12., 13., 14., 15.] if fallback else [9., 9.05, 8.95, 8.5, 8.45]
    winner = None if fallback else 3  # seed 4 improves by less than the fixed margin.
    params.s0 = params.s_tar
    day = {'N': np.full(24, 10.), 'C': np.full(24, 1. / 24.)}
    restarts = [{'seed': seed, 'batch': 0, 'selected_iteration': 2,
                 'validation_mean_common_cost': value} for seed, value in enumerate(costs)]
    record = {
        'action_search_version': ACTION_SEARCH_VERSION,
        'deployment_version': DEPLOYMENT_VERSION, 'tag': tag, 'regime': regime,
        'restart_margin_eur_per_day': 0.1, 'fallback_validation_mean_common_cost': 10.,
        'restarts': restarts,
        'batch_selections': [{'batch': 0, 'selected_seed': winner,
                              'selected_kind': 'zero_action_fallback' if fallback else 'trained_policy',
                              'selected_validation_mean_common_cost': 10. if fallback else costs[winner]}],
    }
    file = tmp_path / 'experiments' / 'results' / f'restart_batches_{tag}.json'
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(record), encoding='utf-8')
    for seed in range(5):
        selection = tmp_path / 'checkpoints' / tag / 'confirmatory' / regime / f'seed{seed}' / 'validation_selection.json'
        selection.parent.mkdir(parents=True)
        selection.write_text(json.dumps({'selected_iteration': 2}), encoding='utf-8')

    def validation(path):
        seed = int(path.parent.name[4:])
        return pd.DataFrame({'iteration': [0, 1, 2, 3], 'common_cost': [costs[seed]] * 4})

    monkeypatch.setattr(audit, 'read_aligned_parquet', validation)
    rows = [{'seed': seed, 'batch': 0, 'is_fallback': False,
             'selected_by_batch': seed == winner, 'validation_mean_common_cost': value}
            for seed, value in enumerate(costs)]
    if fallback:
        rows.append({'seed': -1, 'batch': 0, 'is_fallback': True,
                     'selected_by_batch': True, 'validation_mean_common_cost': 10.})
    return (tmp_path, pd.DataFrame(rows), tag, regime, 40, 5, [day] * 30, params), file, record


@pytest.mark.parametrize('fallback', [False, True])
def test_complete_batch_selection_reproduces_incumbent_and_margin(
        tmp_path, monkeypatch, params, fallback):
    arguments, _, _ = selection_example(tmp_path, monkeypatch, params, fallback)
    assert audit.check_batch_selection(*arguments) == 1


@pytest.mark.parametrize('damage', ['margin', 'fallback_cost', 'winner', 'test_flags'])
def test_complete_batch_audit_rejects_changed_selection(
        tmp_path, monkeypatch, params, damage):
    arguments, file, record = selection_example(tmp_path, monkeypatch, params)
    if damage == 'margin':
        record['restart_margin_eur_per_day'] = 0.01
    elif damage == 'fallback_cost':
        record['fallback_validation_mean_common_cost'] = 9.9
    elif damage == 'winner':
        record['batch_selections'][0]['selected_seed'] = 4
    else:
        arguments[1]['selected_by_batch'] = arguments[1]['seed'].eq(4)
    file.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(RuntimeError):
        audit.check_batch_selection(*arguments)
