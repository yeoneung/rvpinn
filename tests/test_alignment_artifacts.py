import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

EXP = Path(__file__).resolve().parents[1] / 'experiments'
sys.path.insert(0, str(EXP))
from result_validation import (finite_solver_gaps, read_aligned_parquet,
                               require_paired_dates)
from src.deployment_numerics import DEPLOYMENT_VERSION
from src.deployed_policy import ACTION_SEARCH_VERSION


def test_gap_summary_excludes_sentinels_and_unavailable_bounds():
    frame = pd.DataFrame({
        'gap': [0.01, 1e20, 0.1, 0.2, 0.3, float('nan')],
        'primal_bound': [100., 100., 1e20, 100., 100., 100.],
        'dual_bound': [99., 1., 90., -float('inf'), 90., 90.],
        'has_solution': [True, True, True, True, False, True],
    })
    assert finite_solver_gaps(frame).tolist() == [0.01]
    assert len(frame) == 6  # Diagnostic filtering must not drop recorded solves.


def test_day_pairing_rejects_equal_size_mismatches_and_duplicates():
    days = pd.DataFrame({'date': ['2019-01-02', '2019-01-03']})
    require_paired_dates(days, days.iloc[::-1], expected=2)
    for other in (pd.DataFrame({'date': ['2019-01-02', '2019-01-04']}),
                  pd.DataFrame({'date': ['2019-01-02', '2019-01-02']})):
        with pytest.raises(RuntimeError, match='identical unique'):
            require_paired_dates(days, other, expected=2)


def test_aggregation_requires_corrected_search_for_affected_policies(tmp_path):
    path = tmp_path / 'learned_daily_example.parquet'
    base = {'deployment_version': [DEPLOYMENT_VERSION], 'is_fallback': [False]}
    pd.DataFrame(base).to_parquet(path)
    with pytest.raises(RuntimeError, match='missing closed-grid'):
        read_aligned_parquet(path)
    base['action_search_version'] = ['old-grid']
    pd.DataFrame(base).to_parquet(path)
    with pytest.raises(RuntimeError, match='mixed or legacy action-search'):
        read_aligned_parquet(path)
    base['action_search_version'] = [ACTION_SEARCH_VERSION]
    pd.DataFrame(base).to_parquet(path)
    assert len(read_aligned_parquet(path)) == 1
    base['is_fallback'] = [True]
    base['action_search_version'] = ['not_applicable']
    pd.DataFrame(base).to_parquet(path)
    assert len(read_aligned_parquet(path)) == 1


def test_aggregation_rejects_legacy_and_mixed_records(tmp_path):
    path = tmp_path / 'daily.parquet'
    pd.DataFrame({'common_cost': [1.]}).to_parquet(path)
    with pytest.raises(RuntimeError, match='not a corrected'):
        read_aligned_parquet(path)
    pd.DataFrame({'deployment_version': [DEPLOYMENT_VERSION, 'legacy']}).to_parquet(path)
    with pytest.raises(RuntimeError, match='mixed or legacy'):
        read_aligned_parquet(path)
    pd.DataFrame({'deployment_version': [DEPLOYMENT_VERSION]}).to_parquet(path)
    assert len(read_aligned_parquet(path)) == 1


def test_parameter_scale_is_available_without_checkpoint_files():
    spec = importlib.util.spec_from_file_location('parameter_export', EXP / 'make_round2_parameter_tables.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scale = module._selected_scale('v3_winter_fee40', 'winter_weekday', 0)
    assert 300. < scale < 303.
