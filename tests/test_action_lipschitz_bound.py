"""Physical-coordinate Lipschitz constant used in the scalar-search result."""
import numpy as np
import pytest

from src.config import ModelParams


@pytest.mark.parametrize('price', [-.2, 0., .2])
@pytest.mark.parametrize('alpha', [.5, 1.5])
def test_jump_removed_q_obeys_the_physical_slope_bound(price, alpha):
    p = ModelParams(alpha_s=alpha)
    rng = np.random.default_rng(70910)
    soc, net, value_slope = .5, 345.81784079454, 700.
    actions = np.sort(np.r_[rng.uniform(-500., 500., 1000), 0., net - 300., net])
    following = soc + p.dt_ctrl / p.E_max * (
        p.eta_c * np.maximum(-actions, 0.) - np.maximum(actions, 0.) / p.eta_d)
    grid = net - actions
    # A continuation with a known global SoC Lipschitz constant, averaged
    # over action-independent disturbances. Its level/sign are unrestricted.
    phase = np.array([-.4, 0., .8])
    values = value_slope * np.sin(following[:, None] + phase)
    continuation = values @ np.array([.2, .5, .3])
    without_jump = p.dt_ctrl * (
        price * np.maximum(grid, 0.) - alpha * price * np.maximum(-grid, 0.)
        + p.lam1 * np.abs(actions)) + continuation
    bound = p.dt_ctrl * (
        max(1., alpha) * abs(price) + p.lam1
        + max(p.eta_c, 1. / p.eta_d) / p.E_max * value_slope)
    assert np.all(np.abs(np.diff(without_jump)) <= bound * np.diff(actions) + 1e-10)
