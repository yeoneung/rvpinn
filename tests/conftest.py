import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ModelParams          # noqa: E402
from src.costs import RegimeProfile         # noqa: E402


@pytest.fixture
def params():
    p = ModelParams()
    # small but nontrivial calibrated-style coefficients for testing
    p.lam_pk = 2e-4
    p.lam2 = 1e-5
    p.lam_T = 500.0
    p.kappa_y = 1.2
    p.sigma_y = 60.0
    p.kappa_p = 2.0
    p.sigma_p = 0.03
    p.rho = 0.35
    p.y_min, p.y_max = -300.0, 300.0
    p.p_min, p.p_max = -0.08, 0.08
    return p


@pytest.fixture
def profile():
    hours = np.arange(24)
    n_bar = 250.0 + 150.0 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    c_bar = 0.10 + 0.05 * np.sin(2 * np.pi * (hours - 8) / 24.0)
    return RegimeProfile(n_bar, c_bar)
