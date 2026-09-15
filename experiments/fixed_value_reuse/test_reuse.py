import subprocess
import sys
import numpy as np
import pytest
import torch

from common import HERE, PUBLIC, bundle, protocol, make_cases, update_state
from batched_policy import batched_actions
from src.policy_iteration import load_checkpoint
from src.deployed_policy import deployed_one_step_action
from src.safety import deploy_bounds


@pytest.fixture(scope='module')
def model():
    torch.set_num_threads(1)
    p, _ = bundle()
    return load_checkpoint(str(PUBLIC / protocol()['checkpoint']), p, 'cpu').eval()


@pytest.mark.parametrize('t', [0., 12., 23.75])
def test_batch_matches_scalar(model, t):
    p, prof = bundle()
    s = np.array([.1, .1000001, .2, .5, .8, .8999999, .9])
    y = np.linspace(p.y_min, p.y_max, len(s))
    pe = np.linspace(p.p_min, p.p_max, len(s))
    net = prof.n_bar(np.full(len(s), t)) + y
    price = prof.c_bar(np.full(len(s), t)) + pe
    a, q = batched_actions(model, t, s, y, pe, net, price, max_rows=15000)
    reference = np.array([deployed_one_step_action(model, t, ss, yy, pp, prof,
                                                  current_net=nn, current_price=cc)
                          for ss, yy, pp, nn, cc in zip(s, y, pe, net, price)])
    np.testing.assert_allclose(q, reference[:, 1], rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(a, reference[:, 0], rtol=0, atol=1e-5)


def test_paths_are_shared_across_soc_without_parameter_changes():
    cases = make_cases()
    assert len(cases) == 12
    for i in range(4):
        for j in (i+4, i+8):
            np.testing.assert_array_equal(cases[i]['net'], cases[j]['net'])
            np.testing.assert_array_equal(cases[i]['price'], cases[j]['price'])
    assert len({tuple(c['net']) for c in cases}) == 4


def test_hard_cost_and_safe_dynamics():
    p, _ = bundle()
    s = np.array([.2, .5, .8])
    a = np.array([-100., 0., 100.])
    net = p.g_thr + a
    sn, held, costs, violations = update_state(s, a, net, np.full(3, .1), p)
    assert np.all(costs[:, 1] == 0)
    assert np.all(violations == 0)
    assert np.all((sn >= p.s_min) & (sn <= p.s_max))
    np.testing.assert_array_equal(a, held)
    _, _, active, _ = update_state(s, a, net + 1e-6, np.full(3, .1), p)
    np.testing.assert_allclose(active[:, 1], p.dt_ctrl*p.c_step)


def test_scip_runs_in_a_torch_free_process():
    code = "from simulation import warm_scip_worker; import sys; assert 'torch' not in sys.modules; assert warm_scip_worker()"
    subprocess.run([sys.executable, '-c', code], cwd=HERE, check=True)
