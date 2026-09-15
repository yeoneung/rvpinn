"""Check the independent piecewise-quadratic oracle without running SCIP."""
import numpy as np
import pytest

from experiments.audit_terminal_optimizer import (
    analytic_terminal_minimum, hard_terminal_objective, physical_bounds)
from src.config import ModelParams


@pytest.mark.parametrize('price', [-.2, 0., .2])
@pytest.mark.parametrize('soc', [.1, .3, .5, .7, .9])
def test_terminal_analytic_minimum_dominates_dense_grid(price, soc):
    p = ModelParams(g_thr=300., c_step=40., lam_T=10819.79)
    for net in [-50., 0., 300., 345.81784079454, 850.]:
        lower, upper = physical_bounds(p, soc)
        minimum, action = analytic_terminal_minimum(p, soc, net, price)
        assert lower <= action <= upper
        assert hard_terminal_objective(p, soc, net, price, action) == minimum
        grid = np.linspace(lower, upper, 2001)
        values = [hard_terminal_objective(p, soc, net, price, a) for a in grid]
        assert minimum <= min(values) + 1e-10
        assert min(values) - minimum <= .4


def test_terminal_analytic_minimum_has_known_stationary_discharge():
    p = ModelParams(g_thr=300., c_step=0., lam_T=10819.79)
    price = .2
    slope = p.dt_ctrl / (p.E_max * p.eta_d)
    expected = p.dt_ctrl * (price - p.lam1) / (2. * p.lam_T * slope ** 2)
    _, action = analytic_terminal_minimum(p, .5, 850., price)
    assert np.isclose(action, expected, rtol=0., atol=1e-8)
