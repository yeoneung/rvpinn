"""Identical seeds must reproduce identical simulations and networks."""
import numpy as np
import torch

from src.dynamics import simulate_exogenous
from src.pinn_value import ValueNet
from src.reproducibility import seed_everything


def test_simulation_reproducible(params):
    rng1 = np.random.default_rng(5101)
    rng2 = np.random.default_rng(5101)
    Y1, P1 = simulate_exogenous(params, 50, 100, 0.25, rng1)
    Y2, P2 = simulate_exogenous(params, 50, 100, 0.25, rng2)
    assert np.array_equal(Y1, Y2) and np.array_equal(P1, P2)


def test_network_init_reproducible(params):
    m1 = ValueNet(params, width=16, depth=2, seed=7)
    m2 = ValueNet(params, width=16, depth=2, seed=7)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    m3 = ValueNet(params, width=16, depth=2, seed=8)
    diff = any(not torch.equal(p1, p3)
               for p1, p3 in zip(m1.parameters(), m3.parameters()))
    assert diff


def test_seed_everything_isolates(params):
    seed_everything(1101)
    a = np.random.rand(5)
    seed_everything(1101)
    b = np.random.rand(5)
    assert np.array_equal(a, b)
