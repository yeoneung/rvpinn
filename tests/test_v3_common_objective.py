"""Gates introduced by the version-3 common-comparison protocol."""
import numpy as np

from src.costs import common_running_cost_exact, common_terminal_cost
from src.hamiltonian import (hard_common_hamiltonian_terms,
                             minimize_hard_common_hamiltonian)
from src.evaluation import run_day
from src.safety import deploy_bounds


def test_common_cost_has_only_frozen_components(params):
    params.c_step = 40.0
    params.g_thr = 300.0
    params.lam1 = 0.0375
    params.lam_pk = 999.0
    params.lam2 = 999.0
    params.lam_s = 999.0
    C = np.array([0.2, 0.2])
    G = np.array([301.0, -100.0])
    a = np.array([20.0, -10.0])
    got = common_running_cost_exact(C, G, a, params)
    expected = np.array([
        0.2 * 301.0 + 0.0375 * 20.0 + 40.0,
        -params.alpha_s * 0.2 * 100.0 + 0.0375 * 10.0,
    ])
    assert np.allclose(got, expected)
    assert common_terminal_cost(params.s_tar, params) == 0.0


def test_hard_search_matches_dense_reference(params, profile):
    rng = np.random.default_rng(20260906)
    n = 160
    params.c_step = 80.0
    params.g_thr = 300.0
    params.lam1 = 0.0375
    t = rng.uniform(0.0, params.T, n)
    s = rng.uniform(params.s_min, params.s_max, n)
    y = rng.uniform(params.y_min, params.y_max, n)
    pz = rng.uniform(params.p_min, params.p_max, n)
    v_s = rng.uniform(-3000.0, 3000.0, n)
    lo, hi = deploy_bounds(s, params, params.dt_ctrl)
    a, h = minimize_hard_common_hamiltonian(
        t, s, y, pz, v_s, params, profile, lo, hi, n_grid=4097)

    # Independent denser reference, augmented by the discontinuity itself and
    # its fee-active side.  The coarse search is allowed only the Lipschitz
    # discretization error between adjacent action nodes.
    frac = np.linspace(0.0, 1.0, 32769)
    nbar = np.asarray(profile.n_bar(t))
    cbar = np.asarray(profile.c_bar(t))
    best = np.full(n, np.inf)
    for i0 in range(0, n, 20):
        sl = slice(i0, i0 + 20)
        grid = lo[sl, None] + (hi - lo)[sl, None] * frac[None, :]
        crossing = np.clip(nbar[sl] + y[sl] - params.g_thr,
                           lo[sl], hi[sl])
        grid = np.concatenate([
            grid,
            crossing[:, None],
            np.nextafter(crossing, -np.inf)[:, None],
        ], axis=1)
        vals = hard_common_hamiltonian_terms(
            grid, s[sl, None], y[sl, None], nbar[sl, None], cbar[sl, None],
            pz[sl, None], v_s[sl, None], params)
        best[sl] = vals.min(axis=1)
    max_slope = (np.abs(v_s) / (params.eta_d * params.E_max)
                 + np.maximum(np.abs(cbar + pz),
                              params.alpha_s * np.abs(cbar + pz))
                 + params.lam1)
    grid_error = max_slope * (hi - lo) / 4096.0
    assert np.all(h <= best + grid_error + 1e-9)
    assert np.all(a >= lo - 1e-12) and np.all(a <= hi + 1e-12)


def test_tariff_crossing_is_an_explicit_candidate(params, profile):
    params.c_step = 80.0
    params.g_thr = 300.0
    params.lam1 = 0.0
    t = np.array([12.0])
    s = np.array([0.5])
    y = np.array([350.0 - profile.n_bar(t)[0]])
    pz = np.array([0.0])
    C = float(profile.c_bar(t)[0])
    # Make additional discharge mildly unattractive within either tariff
    # region, while the 80-EUR discontinuous saving still justifies reaching
    # the threshold.
    v_s = np.array([-1.10 * C * params.eta_d * params.E_max])
    lo, hi = deploy_bounds(s, params, params.dt_ctrl)
    a, _ = minimize_hard_common_hamiltonian(
        t, s, y, pz, v_s, params, profile, lo, hi, n_grid=65)
    # With a large step fee and positive price, the optimum shaves import
    # exactly to the 300-kW threshold: a = 350 - 300 = 50 kW.
    assert np.isclose(a[0], 50.0, atol=1e-12)


def test_exact_threshold_occupation_is_recorded(params, profile):
    params.c_step = 40.0
    params.g_thr = 300.0
    day = {
        "date": "2019-01-02", "regime": "winter_weekday",
        "N": np.full(24, 310.0), "C": np.full(24, 0.10),
    }

    class ThresholdController:
        def __call__(self, t, s, y, pz, ctx):
            return ctx["N_now"] - params.g_thr

    result = run_day(ThresholdController(), day, params, profile)
    assert np.isclose(result["threshold_exact_hours"], 24.0)
    assert np.isclose(result["capacity_fee"], 0.0)
