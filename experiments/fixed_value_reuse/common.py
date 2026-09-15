"""Pure NumPy public-study access; keep Torch out of SCIP worker processes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PUBLIC = Path(os.environ.get('RVPINN_REUSE_SOURCE', str(HERE.parents[1]))).resolve()
RESULTS = Path(os.environ.get('RVPINN_REUSE_RESULTS', str(HERE / 'results'))).resolve()
sys.path.insert(0, str(PUBLIC))
sys.path.insert(0, str(PUBLIC / 'experiments'))
from src.exp_common import regime_bundle
from src.dynamics import soc_step
from src.safety import deploy_bounds, project_action_to_safe_set


def protocol():
    return json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle():
    cfg = protocol()
    params, profiles, _ = regime_bundle(cfg['region'], [cfg['regime']])
    p, prof = params[cfg['regime']], profiles[cfg['regime']]
    p.c_step, p.g_thr, p.w_step = cfg['fee'], cfg['threshold'], cfg['width']
    return p, prof


def gpu_status():
    try:
        line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free',
                                        '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()[0]
        util, mem = map(float, line.split(','))
        return dict(utilization_percent=util, free_mib=mem)
    except (OSError, ValueError, subprocess.SubprocessError):
        return dict(utilization_percent=None, free_mib=None)


def context(path_id, net, price, prof):
    return {'date': f'reuse_path_{path_id}', 'N_now': float(net), 'C_now': float(price),
            'N_forecast': lambda t: prof.n_bar(np.asarray(t, dtype=float)),
            'C_forecast': lambda t: prof.c_bar(np.asarray(t, dtype=float))}


def make_cases():
    from stochastic_miqp import ScenarioExactMIPController
    cfg = protocol()
    p, prof = bundle()
    generator = ScenarioExactMIPController(p, n_scenarios=cfg['n_exogenous_paths'],
                                         scenario_pool_size=cfg['n_exogenous_paths'],
                                         random_seed=cfg['outer_path_seed'])
    n0, c0 = prof.n_bar(np.array([0.]))[0], prof.c_bar(np.array([0.]))[0]
    net, price = generator._forecast_scenarios(0., 96, 0., 0., context(0, n0, c0, prof))
    times = np.arange(96) * p.dt_ctrl
    cases = []
    for s0 in cfg['initial_soc']:
        for path_id in range(cfg['n_exogenous_paths']):
            cases.append(dict(id=len(cases), path_id=path_id, initial_soc=s0,
                              net=net[path_id].tolist(), price=price[path_id].tolist(),
                              y=(net[path_id] - prof.n_bar(times)).tolist(),
                              pz=(price[path_id] - prof.c_bar(times)).tolist()))
    return cases


def update_state(s, a, net, price, p):
    """Exactly billed cost; three five-minute state substeps per held action."""
    s, a = np.asarray(s, dtype=float), np.asarray(a, dtype=float)
    lo, hi = deploy_bounds(s, p, p.dt_ctrl)
    violation = np.maximum(lo - a, 0.) + np.maximum(a - hi, 0.)
    a = project_action_to_safe_set(a, s, p, p.dt_ctrl)
    grid = np.asarray(net) - a
    bill = p.dt_ctrl * np.asarray(price) * (np.maximum(grid, 0.) - p.alpha_s * np.maximum(-grid, 0.))
    fee = p.dt_ctrl * p.c_step * (grid > p.g_thr)
    degradation = p.dt_ctrl * p.lam1 * np.abs(a)
    for _ in range(int(round(p.dt_ctrl / p.dt_sim))):
        s = soc_step(s, a, p.dt_sim, p)
    return s, a, np.stack([bill, fee, degradation], axis=-1), violation


def finish_rows(cases, states, components, actions, violations, p):
    rows = []
    for i, case in enumerate(cases):
        terminal = p.lam_T * (states[i] - p.s_tar) ** 2
        rows.append(dict(case_id=case['id'], path_id=case['path_id'], initial_soc=case['initial_soc'],
                         bill=float(components[i, 0]), band_fee=float(components[i, 1]),
                         degradation=float(components[i, 2]), terminal=float(terminal),
                         cost=float(components[i].sum() + terminal),
                         exceed_hours=float(components[i, 1] / p.c_step),
                         terminal_soc=float(states[i]), max_action_violation_kw=float(violations[i]),
                         actions=actions[i].tolist()))
    return rows
