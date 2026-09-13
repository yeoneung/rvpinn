"""Reject legacy or mixed numerical implementations during aggregation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.deployment_numerics import DEPLOYMENT_VERSION
from src.deployed_policy import ACTION_SEARCH_VERSION


def finite_solver_gaps(frame):
    """Ordinary reported gaps with an incumbent and usable primal/dual bounds.

    SCIP's finite infinity sentinel (typically 1e20) is not an economic bound.
    This filters diagnostic summaries only; no solve or daily cost is dropped.
    """
    fields = ['gap', 'primal_bound', 'dual_bound']
    mask = (np.isfinite(frame[fields]).all(axis=1)
            & frame[fields].abs().lt(1e19).all(axis=1)
            & frame['gap'].ge(0.0) & frame['has_solution'].eq(True))
    return frame.loc[mask, 'gap']


def format_gap_percent(value):
    """Show relative-gap ratios as percentages, retaining unavailable values."""
    if not np.isfinite(value) or abs(value) >= 1e19:
        return '--'
    if value < 0.:
        raise ValueError('a relative solver gap cannot be negative')
    return f'{100. * value:.2f}\\%'


def require_paired_dates(first, second, expected=30):
    """Reject equal-size but mismatched or duplicated daily endpoints."""
    if (len(first) != expected or len(second) != expected
            or first['date'].duplicated().any()
            or second['date'].duplicated().any()
            or set(first['date']) != set(second['date'])):
        raise RuntimeError(f'expected {expected} identical unique evaluation dates')


def read_aligned_parquet(path):
    frame = pd.read_parquet(path)
    if frame.empty or 'deployment_version' not in frame:
        raise RuntimeError(f'not a corrected deployment artifact: {path}')
    if not frame['deployment_version'].eq(DEPLOYMENT_VERSION).all():
        raise RuntimeError(f'mixed or legacy deployment versions: {path}')
    candidate_artifact = Path(path).name.startswith((
        'learned_daily_', 'selected_daily_', 'round2_reference_daily',
        'validation_daily'))
    if candidate_artifact:
        if 'action_search_version' not in frame:
            raise RuntimeError(f'missing closed-grid action-search version: {path}')
        valid = frame['action_search_version'].eq(ACTION_SEARCH_VERSION)
        if 'is_fallback' in frame:
            valid |= (frame['is_fallback'].eq(True)
                      & frame['action_search_version'].eq('not_applicable'))
        if not valid.all():
            raise RuntimeError(f'mixed or legacy action-search versions: {path}')
    return frame
