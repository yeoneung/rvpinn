from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import ModelParams
from src.safety import deploy_bounds
from heuristic import TariffPeakShaving
from experiments.audit_terminal_optimizer import hard_terminal_objective


def rule(p, **changes):
    args = dict(reserve_soc=0.1, charge_target_soc=0.9, charge_price=0.1,
                charge_when_already_above_band=True, allow_partial_shaving=False)
    args.update(changes)
    return TariffPeakShaving(p, **args)


def test_threshold_and_no_oracle_information():
    p = ModelParams(g_thr=300., c_step=40.)
    def forbidden(*args):
        raise AssertionError("Future observations must not be accessed")
    c = dict(N_now=345.81784079454, C_now=.05, N_of_t=forbidden, C_of_t=forbidden)
    action = rule(p)(8., .5, 0., 0., c)
    assert c["N_now"] - action <= p.g_thr
    assert abs(action - (c["N_now"] - p.g_thr)) < 1e-10


def test_recharge_does_not_create_new_fee():
    p = ModelParams(g_thr=300., c_step=40.)
    for net in (250., 299.9, np.nextafter(300., -np.inf), 300.):
        a = rule(p)(0., .5, 0., 0., dict(N_now=net, C_now=.02))
        assert a <= 0 and net - a <= p.g_thr


def test_fee_changes_action_when_energy_price_is_negative():
    p = ModelParams(g_thr=300., c_step=0.)
    ctx = dict(N_now=340., C_now=-.1)
    assert rule(p, charge_price=-1.)(8., .5, 0., 0., ctx) == 0
    p.c_step = 80.
    assert rule(p, charge_price=-1.)(8., .5, 0., 0., ctx) > 0


def test_actions_are_feasible():
    p = ModelParams(g_thr=300., c_step=80.)
    rng = np.random.default_rng(19271)
    for _ in range(250):
        t, s = rng.integers(0, 96) / 4., rng.uniform(p.s_min, p.s_max)
        a = rule(p)(t, s, 0., 0., dict(N_now=rng.uniform(-200, 1000), C_now=rng.uniform(-.2, .3)))
        lo, hi = (float(x[0]) for x in deploy_bounds(np.array([s]), p, p.dt_ctrl))
        assert lo <= a <= hi


def test_terminal_action_beats_dense_grid():
    p = ModelParams(g_thr=300., c_step=40., lam_T=10819.79)
    for s in (.1, .3, .5, .7, .9):
        n, price = 345.81784079454, .07
        a = rule(p)(23.75, s, 0., 0., dict(N_now=n, C_now=price))
        lo, hi = (float(x[0]) for x in deploy_bounds(np.array([s]), p, p.dt_ctrl))
        baseline = min(hard_terminal_objective(p, s, n, price, x) for x in np.linspace(lo, hi, 4097))
        assert hard_terminal_objective(p, s, n, price, a) <= baseline + 1e-8
