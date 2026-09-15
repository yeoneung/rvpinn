import numpy as np
import pandas as pd
import pytest

from experiments.analyze_confirmatory import (hierarchical_paired_stats,
                                               paired_stats,
                                               relative_cost_description)
from experiments.evaluate_restart_batches import select_batch_against_fallback


def test_cost_prose_handles_wins_losses_and_equal_costs():
    assert relative_cost_description(90., 100.) == r'10.0\% lower'
    assert relative_cost_description(110., 100.) == r'10.0\% higher'
    assert relative_cost_description(100., 100.) == 'unchanged'
    with pytest.raises(ValueError):
        relative_cost_description(1., 0.)


def test_paired_blocks_use_chronological_dates_and_reject_missing_pairs():
    dates = pd.date_range('2019-01-01', periods=12).astype(str)
    comparator = pd.DataFrame({'date': dates, 'common_cost': 100.})
    learned = pd.DataFrame({'date': dates, 'common_cost': np.arange(12.) + 94.})
    original = paired_stats(learned, comparator, seed=35)
    shuffled = paired_stats(learned.sample(frac=1., random_state=13),
                            comparator.sample(frac=1., random_state=12), seed=35)
    assert original == shuffled
    with pytest.raises(ValueError, match='identical evaluation dates'):
        paired_stats(learned, comparator.iloc[:-1], seed=35)
    batches = pd.concat([learned.assign(batch=b) for b in range(2)])
    with pytest.raises(ValueError, match='identical evaluation dates'):
        hierarchical_paired_stats(batches, comparator.iloc[:-1], seed=35, n_boot=10)


def test_hierarchical_bootstrap_resamples_complete_batches():
    dates = pd.date_range("2019-01-01", periods=30).astype(str)
    comparator = pd.DataFrame({"date": dates, "common_cost": 100.0})
    learned = pd.concat([
        pd.DataFrame({"batch": batch, "date": dates,
                      "common_cost": 88.0 + batch})
        for batch in range(5)
    ], ignore_index=True)
    result = hierarchical_paired_stats(
        learned, comparator, seed=6101, n_boot=500)
    assert result["n_days"] == 30
    assert result["n_training_batches"] == 5
    assert np.isclose(result["mean_diff"], -10.0)
    assert result["one_sided_95_upper"] < 0.0


def test_batch_selector_can_retain_or_replace_zero_action():
    candidates = [
        {"seed": 0, "validation_mean_common_cost": 101.0},
        {"seed": 1, "validation_mean_common_cost": 95.0},
        {"seed": 2, "validation_mean_common_cost": 95.05},
    ]
    chosen, cost, decisions = select_batch_against_fallback(
        candidates, fallback_cost=100.0, margin=0.10)
    assert chosen["seed"] == 1
    assert np.isclose(cost, 95.0)
    assert decisions[0]["incumbent_before"] == "zero_action"

    chosen, cost, _ = select_batch_against_fallback(
        candidates[:1], fallback_cost=100.0, margin=0.10)
    assert chosen is None
    assert np.isclose(cost, 100.0)
