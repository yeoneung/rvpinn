"""Narrowing the logistic width does not change its midpoint tariff."""
import numpy as np
import pytest

from src.costs import step_fee_exact, step_fee_smooth


@pytest.mark.parametrize('fee', [20., 40., 80.])
def test_exact_threshold_keeps_half_fee_for_every_positive_width(params, fee):
    params.g_thr, params.c_step = 300., fee
    for width in (20., 10., 5., 1., 1e-9):
        params.w_step = width
        assert step_fee_exact(np.array([300.]), params)[0] == 0.
        assert step_fee_smooth(np.array([300.]), params)[0] == fee / 2.


def test_fixed_trace_limit_retains_exact_threshold_occupation(params):
    params.g_thr, params.c_step, params.w_step = 300., 40., 1e-6
    exchange = np.array([299., 300., 300., 301.])
    dt = .25
    tariff_gap = dt * np.sum(step_fee_smooth(exchange, params)
                             - step_fee_exact(exchange, params))
    exact_threshold_hours = dt * np.count_nonzero(exchange == params.g_thr)
    assert tariff_gap == params.c_step * exact_threshold_hours / 2.
