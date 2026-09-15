"""Mirror reflection correctness for arbitrarily large overshoots."""
import numpy as np

from src.dynamics import reflect, simulate_exogenous


def test_reflection_in_bounds():
    rng = np.random.default_rng(1)
    z = rng.uniform(-1e6, 1e6, 100000)
    r = reflect(z, -2.0, 3.0)
    assert np.all(r >= -2.0) and np.all(r <= 3.0)


def test_reflection_identity_inside():
    z = np.linspace(-1.99, 2.99, 1000)
    assert np.allclose(reflect(z, -2.0, 3.0), z)


def test_single_mirror():
    # one reflection at the top: hi + d -> hi - d
    assert np.isclose(reflect(np.array([3.5]), -2.0, 3.0)[0], 2.5)
    # one reflection at the bottom: lo - d -> lo + d
    assert np.isclose(reflect(np.array([-2.7]), -2.0, 3.0)[0], -1.3)


def test_double_mirror():
    # width 5, overshoot beyond one full fold: -2 - 5 - 1 = -8 maps like
    # reflect twice: -8 -> fold at -2 gives 4 above? verify via explicit fold
    z = -8.0
    lo, hi = -2.0, 3.0
    # manual repeated reflection
    x = z
    for _ in range(100):
        if x < lo:
            x = 2 * lo - x
        elif x > hi:
            x = 2 * hi - x
        else:
            break
    assert np.isclose(reflect(np.array([z]), lo, hi)[0], x)


def test_paths_stay_in_bounds(params):
    rng = np.random.default_rng(5)
    Y, P = simulate_exogenous(params, 200, 288, 5.0 / 60.0, rng)
    assert np.all(Y >= params.y_min) and np.all(Y <= params.y_max)
    assert np.all(P >= params.p_min) and np.all(P <= params.p_max)
