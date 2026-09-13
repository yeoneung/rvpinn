from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from analyze_round2_scenarios import (  # noqa: E402
    resample_learned_minus_scenario,
)
import analyze_round2_scenarios as report


def test_three_way_bootstrap_keeps_paired_dates_and_randomization_levels():
    dates = [f"2019-01-{day:02d}" for day in range(1, 11)]
    learned = pd.DataFrame([
        {"date": date, "batch": batch, "common_cost": 5.0}
        for date in dates for batch in (0, 1)
    ])
    comparator = pd.DataFrame([
        {"date": date, "stream": stream, "common_cost": 7.0}
        for date in dates for stream in (7301, 7302, 7303)
    ])
    out = resample_learned_minus_scenario(
        learned, comparator, seed=3, n_boot=200, block=3)
    assert out["n_days"] == 10
    assert out["n_training_batches"] == 2
    assert out["n_scenario_streams"] == 3
    assert np.isclose(out["mean_diff"], -2.0)
    assert np.isclose(out["one_sided_95_upper"], -2.0)
    assert out["superiority"]


def test_scenario_reporting_cannot_omit_pending_high_budget_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'RESULTS', tmp_path)
    for fee, count, budget in report.SETTINGS:
        for stream in report.STREAMS:
            if (fee, count, budget, stream) == (80, 16, 15, 7303):
                continue
            for path in report.paths(fee, count, stream, budget):
                path.touch()
    monkeypatch.setattr(report, 'load_setting',
                        lambda *args: pytest.fail('must check coverage before analysis'))
    with pytest.raises(FileNotFoundError, match='incomplete fixed scenario design'):
        report.main()
