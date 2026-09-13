"""Factorial reporting preserves the complete design and observed direction."""
import numpy as np
import pandas as pd
import pytest

from experiments import analyze_seasonal_sensitivity as report


def cells():
    rows = []
    for threshold in (250, 300, 350):
        for fee in (20, 40, 80):
            for width in (5, 10, 20):
                anchor = threshold == 300 and width == 10
                rows.append({
                    'threshold_kw': threshold, 'fee_per_hour': fee,
                    'smoothing_width_kw': width,
                    'n_training_restarts': (25 if fee == 40 else 5) if anchor else 1,
                    'n_selected_batches': (5 if fee == 40 else 1) if anchor else 1,
                    'n_days': 30, 'mean_learned_cost': 100. + width,
                    'mean_deterministic_cost': 120., 'mean_no_storage_cost': 125.,
                    'learned_minus_deterministic': width - 20.,
                    'learned_minus_no_storage': width - 25.,
                    'mean_terminal_penalty': .1, 'mean_smooth_minus_hard_cost': -2.,
                    'mean_exact_threshold_hours': .5,
                    'mean_near_5kw_hours': .7, 'mean_near_10kw_hours': 1.,
                    'mean_near_20kw_hours': 2.,
                })
    return pd.DataFrame(rows)


@pytest.mark.parametrize('reverse', [False, True])
def test_factorial_text_and_table_follow_all_cells(tmp_path, monkeypatch, reverse):
    frame = cells()
    if reverse:
        frame['learned_minus_deterministic'] *= -1
        frame['learned_minus_no_storage'] *= -1
    monkeypatch.setattr(report, 'RESULTS', tmp_path)
    monkeypatch.setattr(report, 'GENERATED', tmp_path)
    summary = report.write_factorial(frame)
    assert summary['factorial_cells'] == 27
    assert summary['offline_better_than_deterministic_cells'] == (0 if reverse else 18)
    assert summary['offline_better_than_no_storage_cells'] == (0 if reverse else 27)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / 'sensitivity_cells.csv'), frame)
    prose = (tmp_path / 'factorial_results.tex').read_text(encoding='utf-8')
    detail = (tmp_path / 'factorial_supplement.tex').read_text(encoding='utf-8')
    assert f"MIQP in {0 if reverse else 18} cells" in prose
    assert '24 cells use one restart' in prose
    assert 'unequal selection effort' in prose
    assert 'best width' not in prose and 'significant' not in prose
    assert len([line for line in detail.splitlines() if line.startswith(('250 &', '300 &', '350 &'))]) == 27


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'wrong_level', 'nonfinite'])
def test_factorial_report_rejects_incomplete_or_corrupt_design(tmp_path, monkeypatch, corruption):
    frame = cells()
    if corruption == 'missing':
        frame = frame.iloc[:-1]
    elif corruption == 'duplicate':
        frame.iloc[0] = frame.iloc[1]
    elif corruption == 'wrong_level':
        frame.loc[0, 'threshold_kw'] = 200
    else:
        frame.loc[0, 'mean_learned_cost'] = np.nan
    monkeypatch.setattr(report, 'RESULTS', tmp_path)
    monkeypatch.setattr(report, 'GENERATED', tmp_path)
    with pytest.raises(RuntimeError, match='27 finite, unique cells'):
        report.write_factorial(frame)
    assert not list(tmp_path.iterdir())
