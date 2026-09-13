"""Seasonal coverage repair preserves rather than replaces original outcomes."""
import pandas as pd
import pytest

from experiments.merge_corrected_inputs import merge_disjoint_frames


def frame(date, cost=10.):
    return pd.DataFrame({'method': ['m'], 'date': [date], 'common_cost': [cost]})


def test_disjoint_merge_preserves_values_and_orders_keys():
    left, right = frame('2019-04-01', 123.456789012345), frame('2019-01-02', -5.25)
    merged = merge_disjoint_frames(left, right, ['method', 'date'], 2)
    assert merged['date'].tolist() == ['2019-01-02', '2019-04-01']
    assert merged['common_cost'].tolist() == [-5.25, 123.456789012345]
    assert left['date'].tolist() == ['2019-04-01']


def test_merge_rejects_replacement_even_when_values_agree():
    with pytest.raises(RuntimeError, match='overwrite'):
        merge_disjoint_frames(frame('a'), frame('a'), ['method', 'date'], 2)


def test_merge_rejects_duplicate_source_keys():
    duplicate = pd.concat([frame('a'), frame('a')], ignore_index=True)
    with pytest.raises(RuntimeError, match='duplicate'):
        merge_disjoint_frames(duplicate, frame('b'), ['method', 'date'], 3)


def test_merge_requires_complete_count_and_matching_schema():
    with pytest.raises(RuntimeError, match='expected 3'):
        merge_disjoint_frames(frame('a'), frame('b'), ['method', 'date'], 3)
    with pytest.raises(RuntimeError, match='schemas differ'):
        merge_disjoint_frames(frame('a'), frame('b').assign(extra=1), ['method', 'date'], 2)


def test_merge_rejects_missing_endpoint_key():
    with pytest.raises(RuntimeError, match='missing'):
        merge_disjoint_frames(frame(None), frame('b'), ['method', 'date'], 2)
