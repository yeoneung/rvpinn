import numpy as np
import torch

from src.deployed_policy import (deployed_action_candidates,
                                 deployed_one_step_q)
from src.pinn_value import ValueNet
from src.reference_policy import (reference_one_step_action,
                                  reference_one_step_q)


def test_reference_reduction_matches_zero_correction_action_differences(
        params, profile):
    params.lam_pk = 0.0
    params.lam2 = 0.0
    params.lam_s = 0.0
    params.c_step = 40.0
    params.g_thr = 300.0
    params.lam_T = 8000.0
    model = ValueNet(
        params, width=8, depth=1, use_p=True,
        reference_baseline="zero_action_conditional_mean", seed=19)
    model.attach_profile(profile)
    for layer in model.layers:
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    states = [
        (0.0, 0.50, 20.0, 0.01),
        (8.25, 0.22, -35.0, -0.02),
        (18.75, 0.81, 55.0, 0.03),
        (23.75, 0.58, 5.0, 0.00),
    ]
    for t, s, y, pz in states:
        n_now = float(profile.n_bar(np.array([t]))[0]) + y
        candidates = deployed_action_candidates(
            params, s, n_now, params.dt_ctrl, n_grid=257)
        full = deployed_one_step_q(
            model, t, s, y, pz, profile, candidates,
            quadrature_order=3)
        reduced = reference_one_step_q(
            params, profile, t, s, y, pz, candidates)
        # The omitted no-storage operating term is constant in action.
        assert np.ptp((full - reduced)) < 1e-8
        a_ref, _ = reference_one_step_action(
            params, profile, t, s, y, pz, n_grid=257)
        full_min = float(np.min(full))
        tied = candidates[np.abs(full - full_min) <= 1e-8]
        assert float(np.min(np.abs(tied - a_ref))) <= 1e-10
