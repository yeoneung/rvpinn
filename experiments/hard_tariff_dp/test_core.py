from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.hard_tariff_dp.core import GridDP, reflected_transition, terminal_action, stage, following_soc
from experiments.audit_terminal_optimizer import analytic_terminal_minimum
from experiments.controller_comparisons.heuristic import TariffPeakShaving
from src.config import ModelParams
from src.dynamics import reflect


class Profile:
    def n_bar(self, t): return np.asarray(t)*0+350.
    def c_bar(self, t): return np.asarray(t)*0+.1


def test_transition_matches_continuous_reflection_monte_carlo():
    y = np.linspace(-2., 3., 61)
    P = reflected_transition(y, .3, 1.2, .25)
    assert np.all(P >= 0) and np.max(abs(P.sum(1)-1)) < 1e-12
    rng = np.random.default_rng(123)
    z = rng.normal(size=200000)
    sd = 1.2*np.sqrt(-np.expm1(-2*.3*.25)/(2*.3))
    for i in (0, 20, 60):
        samples = reflect(np.exp(-.3*.25)*y[i]+sd*z, y[0], y[-1])
        assert abs(P[i]@y-samples.mean()) < .008


def test_terminal_and_vector_rule_match_independent_scalar_algorithms():
    p = ModelParams(T=1., c_step=40., g_thr=300., lam_T=10819.79, y_min=-200., y_max=200.)
    grid = GridDP(p, Profile(), 11, 9, 33)
    setting = dict(reserve_soc=.3, charge_target_soc=.9, charge_price=.2,
                   charge_when_already_above_band=True, allow_partial_shaving=False)
    vector = grid.rule_actions(setting)
    scalar = TariffPeakShaving(p, **setting)
    for k, t in enumerate(grid.times):
        for i, s in enumerate(grid.s):
            for j, y in enumerate(grid.y):
                net = 350.+y
                a = scalar(t, s, y, 0., dict(N_now=net, C_now=.1))
                assert abs(a-vector[k, i, j]) < 1e-9
                target = analytic_terminal_minimum(p, s, net, .1)[0]
                action = terminal_action(p, np.array(s), np.array(net), .1)
                q = stage(p, net, .1, action)+p.lam_T*(following_soc(p, s, action)-p.s_tar)**2
                assert abs(q-target) < 1e-7


def test_dp_policy_evaluation_and_regret_identity():
    p = ModelParams(T=1., c_step=40., g_thr=300., lam_T=10819.79, y_min=-200., y_max=200.)
    grid = GridDP(p, Profile(), 21, 25, 65)
    V, A = grid.solve()
    np.testing.assert_allclose(grid.evaluate(A), grid.initial_costs(V[0]), atol=1e-10)
    rule = grid.reference_actions()
    assert np.min(grid.evaluate(rule)-grid.initial_costs(V[0])) > -1e-8
    assert grid.regret(V, A, rule)['minimum_regret'] > -1e-7
    assert grid.regret(V, A, A)['boundary_max'] < 1e-8


def test_action_refinement_stable_with_interpolation_knots():
    p = ModelParams(T=1., c_step=40., g_thr=300., lam_T=10819.79, y_min=-200., y_max=200.)
    small, large = GridDP(p, Profile(), 21, 25, 33), GridDP(p, Profile(), 21, 25, 129)
    v1, _ = small.solve()
    v2, _ = large.solve()
    np.testing.assert_allclose(v1, v2, atol=1e-8)
