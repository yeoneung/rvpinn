"""Read-only checks of the saved-iteration diagnostic and its manuscript table."""
import importlib.util
from pathlib import Path

import pytest
from experiments.manuscript_checks import enabled

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pi_progression", ROOT / "experiments/analyze_policy_iteration_progression.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("costs, expected", [
    ([100.0, 99.95, 101.0, 100.0], 0),
    ([100.0, 90.0, 80.0, 85.0], 2),
    ([100.0, 90.0, 80.0, 79.5], 3),
])
def test_sequential_margin(costs, expected):
    assert MODULE.selected_iteration(costs) == expected


def test_saved_records_and_rendered_table():
    frame, report = MODULE.collect()
    assert len(frame) == 100
    assert report["selected_count_by_round"] == {"1": 0, "2": 0, "3": 23, "4": 2}
    assert report["round3_better_than_round1_count"] == 25
    assert report["round4_worse_than_round3_count"] == 23
    assert report["round3_vs_round1_reduction_percent"] == pytest.approx(12.3578102351)
    assert report["max_daily_vs_saved_mean_error_eur"] < 1e-8
    expected = MODULE.render(report)
    if enabled():
        assert expected == (ROOT / "manuscript/generated/policy_iteration_progression.tex").read_text(
            encoding="utf-8")
    assert "winter validation across 25 training seeds" in expected
    assert "combined effects of policy updates and the training" in expected
