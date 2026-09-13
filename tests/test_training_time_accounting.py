"""Offline compute includes recorded full training runs, not just optimizer time."""
import hashlib
import json

import pytest

from experiments.analyze_confirmatory import (cost_premium_compute_threshold,
                                               recorded_training_time)


def write_manifest(tmp_path, suffix='1', seed='3', **updates):
    record = {
        'command': ('C:\\old workspace\\scripts\\train_pinn_pi.py '
                    '--tag v3_winter_fee40 --region confirmatory '
                    f'--regimes winter_weekday --seeds {seed} --no-adaptive'),
        'start_time': '2026-09-06T23:59:00',
        'end_time': '2026-09-07T00:03:00',
        'exit_status': 'completed',
    }
    record.update(updates)
    path = tmp_path / f'train_v3_winter_fee40_{suffix}.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


def test_training_run_time_includes_postfit_work_and_provenance(tmp_path):
    path = write_manifest(tmp_path)
    result = recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 225.0)
    assert result['training_wall_s'] == 240.0
    assert result['value_fit_wall_s'] == 225.0
    assert result['training_manifest'] == f'results/logs/{path.name}'
    assert result['training_manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('seed', ['2', '3,4'])
def test_training_run_requires_exact_single_seed(tmp_path, seed):
    write_manifest(tmp_path, seed=seed)
    with pytest.raises(ValueError, match='found 0'):
        recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 225.0)


def test_training_run_rejects_ambiguous_manifests(tmp_path):
    write_manifest(tmp_path)
    write_manifest(tmp_path, suffix='2')
    with pytest.raises(ValueError, match='found 2'):
        recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 225.0)


def test_training_run_does_not_use_failed_or_skipped_execution(tmp_path):
    write_manifest(tmp_path, exit_status='failed')
    with pytest.raises(ValueError, match='found 0'):
        recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 225.0)
    write_manifest(tmp_path, end_time='2026-09-06T23:59:02')
    with pytest.raises(ValueError, match='cannot cover'):
        recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 225.0)


def test_training_run_allows_only_timestamp_quantization_slack(tmp_path):
    write_manifest(tmp_path)
    assert recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 240.5)['training_wall_s'] == 240.0
    with pytest.raises(ValueError, match='cannot cover'):
        recorded_training_time(tmp_path, 'v3_winter_fee40', 3, 241.1)


def test_compute_only_amortization_does_not_hide_operating_premium():
    import math
    assert cost_premium_compute_threshold(10., 100.) == 360.
    assert cost_premium_compute_threshold(-10., 100.) == 0.
    assert math.isnan(cost_premium_compute_threshold(10., 0.))
    assert math.isnan(cost_premium_compute_threshold(10., -100.))
    with pytest.raises(ValueError, match='finite'):
        cost_premium_compute_threshold(float('nan'), 100.)
