from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

pytest.importorskip('pyscipopt')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from exact_miqp import ExactMIPController
from stochastic_miqp import ScenarioExactMIPController
from src.costs import common_running_cost_exact, common_terminal_cost
from src.dynamics import soc_step
from src.safety import deploy_bounds, project_action_to_safe_set


def _check_terminal_miqp(params, price, scenario):
    params.g_thr, params.c_step, params.lam1 = 300., 40., .0375
    net = 345.81784079454
    kwargs = dict(reopt_every_hours=params.dt_ctrl, time_limit_s=2., mip_gap=1e-6)
    ctrl = (ScenarioExactMIPController(params, n_scenarios=2, **kwargs)
            if scenario else ExactMIPController(params, **kwargs))
    ctx = {'N_now': net, 'C_now': price, 'date': '2019-01-07',
           'N_forecast': lambda t: np.asarray(t) * 0. + 300.,
           'C_forecast': lambda t: np.asarray(t) * 0. + .1}
    # Deliberately inconsistent residuals: the current observation is authoritative.
    action = ctrl(params.T - params.dt_ctrl, .5, -100., .02, ctx)
    record = ctrl.solve_records[-1]
    assert record.has_solution
    assert min(record.first_charge_kw, record.first_discharge_kw) <= 1e-8
    assert action == project_action_to_safe_set(np.array([action]), np.array([.5]),
                                               params, params.dt_ctrl)[0]
    if record.first_band_active is False:
        assert net - action <= params.g_thr
    lo, hi = deploy_bounds(np.array([.5]), params, params.dt_ctrl)
    assert lo[0] <= action <= hi[0]
    realized = params.dt_ctrl * common_running_cost_exact(price, net-action, action, params)
    realized += common_terminal_cost(soc_step(.5, action, params.dt_ctrl, params), params)
    assert abs(float(realized) - record.primal_bound) < 1e-4


@pytest.mark.parametrize('scenario', [False, True])
@pytest.mark.parametrize('price', [.001, -.01])
def test_terminal_miqp_incumbent_matches_physical_hard_objective(price, scenario):
    # On Windows, SCIP and Torch can load different Intel OpenMP DLL copies.
    # Match the experimental process isolation; never allow duplicate runtimes.
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(price), str(int(scenario))],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == '__main__':
    from src.config import ModelParams
    params = ModelParams()
    params.lam_T = 500.0
    _check_terminal_miqp(params, float(sys.argv[1]), bool(int(sys.argv[2])))
