"""GPU/torch Hamiltonian minimizer agrees with the numpy solver."""
import numpy as np
import torch

from src.hamiltonian import minimize_hamiltonian, minimize_hamiltonian_torch
from src.safety import cont_bounds

torch.set_default_dtype(torch.float64)


def test_torch_matches_numpy(params, profile):
    rng = np.random.default_rng(77)
    n = 4000
    t = rng.uniform(0, params.T, n)
    s = rng.uniform(params.s_min, params.s_max, n)
    y = rng.uniform(params.y_min, params.y_max, n)
    pz = rng.uniform(params.p_min, params.p_max, n)
    v_s = rng.uniform(-4000.0, 4000.0, n)
    lo, hi = cont_bounds(s, params)
    a_np, h_np = minimize_hamiltonian(t, s, y, pz, v_s, params, profile,
                                      lo, hi)
    tt = torch.tensor(t)
    a_t, h_t = minimize_hamiltonian_torch(
        tt, torch.tensor(s), torch.tensor(y), torch.tensor(pz),
        torch.tensor(v_s), params, profile,
        torch.tensor(np.asarray(lo)), torch.tensor(np.asarray(hi)))
    h_t = h_t.numpy()
    # both must find (essentially) the same minimum value
    tol = 1e-6 + 1e-8 * np.abs(h_np)
    assert np.all(h_t <= h_np + tol) and np.all(h_np <= h_t + tol), \
        f"max discrepancy {np.max(np.abs(h_t - h_np)):.3e}"
