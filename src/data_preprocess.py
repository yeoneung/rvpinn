"""Preprocessing: chronological split, regimes, profiles, scaling.

Every statistic (medians, quantiles, OU parameters, domain bounds) is
computed strictly on the training slice and then frozen.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEASON_OF_MONTH = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}


def chronological_split(index: pd.DatetimeIndex,
                        train_frac: float = 0.60,
                        val_frac: float = 0.20) -> Dict[str, Tuple[str, str]]:
    """Split a datetime index chronologically by complete calendar years
    where possible; fall back to fractional day counts otherwise. Returns
    {'train': (start, end), 'val': ..., 'test': ...} as ISO date strings,
    end exclusive."""
    days = pd.DatetimeIndex(sorted(set(index.normalize())))
    years = sorted(set(days.year))
    complete_years = [y for y in years
                      if (days.year == y).sum() >= 360]
    if len(complete_years) >= 5:
        n = len(complete_years)
        n_train = max(1, int(round(train_frac * n)))
        n_val = max(1, int(round(val_frac * n)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        y_train = complete_years[:n_train]
        y_val = complete_years[n_train:n_train + n_val]
        y_test = complete_years[n_train + n_val:]
        def rng(ys):
            return (f"{ys[0]}-01-01", f"{ys[-1] + 1}-01-01")
        return {"train": rng(y_train), "val": rng(y_val), "test": rng(y_test)}
    # fractional fallback (>= 24 / 6 / 6 months enforced by caller's data)
    n = len(days)
    i1 = int(round(train_frac * n))
    i2 = int(round((train_frac + val_frac) * n))
    return {
        "train": (str(days[0].date()), str(days[i1].date())),
        "val": (str(days[i1].date()), str(days[i2].date())),
        "test": (str(days[i2].date()),
                 str((days[-1] + pd.Timedelta(days=1)).date())),
    }


def split_slice(df: pd.DataFrame, split: Dict[str, Tuple[str, str]],
                part: str) -> pd.DataFrame:
    lo, hi = split[part]
    return df.loc[(df.index >= lo) & (df.index < hi)]


def assert_no_overlap(split: Dict[str, Tuple[str, str]]) -> None:
    tr, va, te = split["train"], split["val"], split["test"]
    assert tr[1] <= va[0] and va[1] <= te[0], f"overlapping split {split}"


def assign_regime(index: pd.DatetimeIndex,
                  holiday_dates: Optional[set] = None) -> pd.Series:
    """Regime label per timestamp: season x weekday/weekend_holiday,
    using LOCAL calendar time (index must already be tz-converted)."""
    holiday_dates = holiday_dates or set()
    season = index.month.map(SEASON_OF_MONTH)
    is_we = index.dayofweek >= 5
    is_hol = pd.Index([d in holiday_dates for d in index.date])
    day_type = np.where(is_we | is_hol, "weekend_holiday", "weekday")
    return pd.Series([f"{s}_{d}" for s, d in zip(season, day_type)],
                     index=index)


def regime_hourly_medians(df_train: pd.DataFrame, col: str,
                          regimes: pd.Series) -> Dict[str, np.ndarray]:
    """Robust hourly median profile per regime, from TRAINING data only."""
    out = {}
    hours = df_train.index.hour
    for reg in sorted(regimes.unique()):
        m = (regimes == reg).values
        prof = np.empty(24)
        for h in range(24):
            sel = m & (hours == h)
            vals = df_train.loc[sel, col].dropna()
            prof[h] = float(np.median(vals)) if len(vals) else np.nan
        out[reg] = prof
    return out


def scale_microgrid(load_raw: pd.Series, renew_raw: pd.Series,
                    train_mask: np.ndarray, P_base: float,
                    gamma_R: float) -> Tuple[pd.Series, pd.Series, Dict]:
    """Shape-preserving scaling with TRAIN-only statistics (eqs. 25-27)."""
    med = float(np.median(load_raw.values[train_mask]))
    q95 = float(np.quantile(renew_raw.values[train_mask], 0.95))
    stats = {"load_train_median": med, "renew_train_q95": q95,
             "P_base": P_base, "gamma_R": gamma_R}
    L = P_base * load_raw / med
    R = gamma_R * P_base * renew_raw / q95
    return L, R, stats


def interpolate_short_gaps(s: pd.Series, max_gap: int = 2) -> pd.Series:
    """Time-interpolate gaps of at most `max_gap` consecutive hours."""
    filled = s.interpolate(method="time", limit=max_gap, limit_area="inside")
    return filled


def drop_incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    """Remove days that still contain NaNs after short-gap interpolation."""
    day = df.index.normalize()
    bad_days = df.isna().any(axis=1).groupby(day).any()
    keep = ~day.map(bad_days).values
    return df.loc[keep]
