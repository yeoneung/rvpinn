"""Chronological split integrity and train-only statistics."""
import numpy as np
import pandas as pd

from src.data_preprocess import (assert_no_overlap, assign_regime,
                                 chronological_split, regime_hourly_medians,
                                 scale_microgrid, split_slice)


def _index(years=6):
    return pd.date_range("2015-01-01", periods=years * 365 * 24, freq="h")


def test_split_no_overlap_and_order():
    idx = _index()
    split = chronological_split(idx)
    assert_no_overlap(split)
    assert split["train"][0] < split["train"][1] <= split["val"][0]
    assert split["val"][1] <= split["test"][0] < split["test"][1]


def test_split_complete_years():
    idx = _index(5)
    split = chronological_split(idx)
    # 5 complete years: 3 train, 1 val, 1 test
    assert split["train"] == ("2015-01-01", "2018-01-01")
    assert split["val"] == ("2018-01-01", "2019-01-01")
    assert split["test"] == ("2019-01-01", "2020-01-01")


def test_train_stats_ignore_future():
    idx = _index()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"load": rng.uniform(100, 500, len(idx))}, index=idx)
    split = chronological_split(idx)
    train = split_slice(df, split, "train")
    regs = assign_regime(train.index)
    prof_a = regime_hourly_medians(train, "load", regs)
    # poison validation/test values; training stats must be identical
    df2 = df.copy()
    df2.loc[df2.index >= split["val"][0], "load"] = 1e9
    train2 = split_slice(df2, split, "train")
    prof_b = regime_hourly_medians(train2, "load", assign_regime(train2.index))
    for k in prof_a:
        assert np.array_equal(prof_a[k], prof_b[k])


def test_scaling_stats_train_only():
    idx = _index()
    rng = np.random.default_rng(1)
    load = pd.Series(rng.uniform(100, 500, len(idx)), index=idx)
    renew = pd.Series(rng.uniform(0, 300, len(idx)), index=idx)
    split = chronological_split(idx)
    mask = (idx >= split["train"][0]) & (idx < split["train"][1])
    L1, R1, st1 = scale_microgrid(load, renew, mask.values if hasattr(mask, 'values') else mask, 500.0, 1.2)
    load2 = load.copy()
    load2[~mask] = 1e9
    renew2 = renew.copy()
    renew2[~mask] = 1e9
    L2, R2, st2 = scale_microgrid(load2, renew2, mask.values if hasattr(mask, 'values') else mask, 500.0, 1.2)
    assert st1 == st2


def test_regime_labels():
    idx = pd.date_range("2020-01-06", periods=24, freq="h")  # a Monday
    regs = assign_regime(idx)
    assert set(regs) == {"winter_weekday"}
    idx = pd.date_range("2020-07-04", periods=24, freq="h")  # a Saturday
    regs = assign_regime(idx)
    assert set(regs) == {"summer_weekend_holiday"}
