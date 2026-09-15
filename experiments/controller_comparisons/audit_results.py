"""Independent hard-cost/action audit for completed controller_comparisons records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.exp_common import regime_bundle
from src.evaluation import load_region_days, run_day
from experiments.controller_comparisons.heuristic import TariffPeakShaving


def check_path(p, actions, nets, prices, initial_soc, terminal_soc, expected_cost, first_substep_states=None):
    s, parts, violation, state_error = initial_soc, np.zeros(3), 0., 0.
    assert len(actions) == len(nets) == len(prices) == 96
    for k, a in enumerate(actions):
        lo = -min(p.a_c*min((p.s_max-s)/p.delta_s, 1.),
                  p.E_max*(p.s_max-s)/(p.eta_c*p.dt_ctrl))
        hi = min(p.a_d*min((s-p.s_min)/p.delta_s, 1.),
                 p.eta_d*p.E_max*(s-p.s_min)/p.dt_ctrl)
        violation = max(violation, lo-a, a-hi)
        g = nets[k]-a
        parts += p.dt_ctrl*np.array([prices[k]*(max(g, 0.)-p.alpha_s*max(-g, 0.)),
                                     p.c_step*float(g > p.g_thr), p.lam1*abs(a)])
        for substep in range(int(round(p.dt_ctrl/p.dt_sim))):
            s += p.dt_sim*(p.eta_c*max(-a, 0.)-max(a, 0.)/p.eta_d)/p.E_max
            # run_day records s AFTER the first five-minute state update.
            if substep == 0 and first_substep_states is not None:
                state_error = max(state_error, abs(first_substep_states[k]-s))
        violation = max(violation, p.s_min-s, s-p.s_max)
    cost = float(parts.sum()+p.lam_T*(s-p.s_tar)**2)
    error = dict(cost_error_eur=abs(cost-expected_cost),
                 soc_error=max(state_error, abs(s-terminal_soc)), bound_violation=max(0., violation))
    assert error['cost_error_eur'] < 1e-7 and error['soc_error'] < 1e-10 and error['bound_violation'] < 1e-7, error
    return error


def check_daily(p, daily):
    traj = daily['traj']
    assert np.max(np.abs(np.array(traj['G'])-(np.array(traj['N'])-np.array(traj['a'])))) < 1e-10
    return check_path(p, traj['a'], traj['N'], traj['C'], p.s0, daily['terminal_soc'],
                      daily['common_cost'], traj['s'])


def check_solver_record(record, net, held, threshold):
    """Check observed exact-band records, without treating dual bounds as truth."""
    assert held == record['first_action_held'], 'Held action differs from solver record'
    assert held-record['first_action_raw'] == record['first_action_recovery_kw']
    if record['first_band_active'] is not None:
        assert bool(net-held > threshold) == record['first_band_active'], 'Recorded and billed fee sides differ'


def audit(replay_heuristic=False):
    params, profiles, _ = regime_bundle('confirmatory')
    errors, sources = [], []
    budget_action_records = 0
    for family, pattern in [('surrogate','*.json'), ('solver_budget','closed_*.json')]:
        for path in sorted((HERE/'results'/family).glob(pattern)):
            r = json.loads(path.read_text())
            if 'daily' not in r:
                continue
            p = deepcopy(params['winter_weekday'])
            p.c_step, p.g_thr, p.w_step = r['job']['fee'], 300., 10.
            errors.append(check_daily(p, r['daily']))
            assert len(r['solves']) == 96
            for a, record in zip(r['daily']['traj']['a'], r['solves']):
                assert abs(a-record.get('final_action', record['first_action_held'])) < 1e-7
            if family == 'solver_budget':
                for net, a, record in zip(r['daily']['traj']['N'], r['daily']['traj']['a'], r['solves']):
                    check_solver_record(record, net, a, p.g_thr)
                    budget_action_records += 1
            sources.append(dict(path=str(path.relative_to(HERE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in sorted((HERE/'results/solver_budget').glob('fixed_fee*.json')):
        r = json.loads(path.read_text())
        assert len(r['solves']) == 1
        check_solver_record(r['solves'][0], r['current_net'], r['action'], 300.)
        budget_action_records += 1
        sources.append(dict(path=str(path.relative_to(HERE)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    reuse = ROOT/'experiments/fixed_value_reuse/results'
    cases = json.loads((reuse/'manifest.json').read_text())['cases']
    p = deepcopy(params['winter_weekday'])
    p.c_step, p.g_thr, p.w_step = 40., 300., 10.
    for path in sorted((HERE/'results/heuristic_reuse').glob('n*_r*.json')):
        for row in json.loads(path.read_text())['rows']:
            c = cases[row['case_id']]
            errors.append(check_path(p, row['actions'], c['net'], c['price'], c['initial_soc'],
                                     row['terminal_soc'], row['cost']))
    if replay_heuristic:
        days = {d['date']:d for d in load_region_days('confirmatory','test')}
        for selection in json.loads((HERE/'results/heuristic/selection.json').read_text()):
            regime, fee = selection['regime'], selection['fee']
            p = deepcopy(params[regime])
            p.c_step, p.g_thr, p.w_step = fee, 300., 10.
            controller = TariffPeakShaving(p, **selection['setting'])
            saved = json.loads((HERE/f'results/heuristic/test_{regime}_fee{fee}.json').read_text())
            for row in saved['days']:
                replay = run_day(controller, days[row['date']], p, profiles[regime], collect_traj=True)
                assert abs(replay['common_cost']-row['common_cost']) < 1e-8
                errors.append(check_daily(p,replay))
    report = dict(audited_trajectories=len(errors), budget_action_records=budget_action_records,
                  heuristic_replayed=replay_heuristic,
                  maximum_errors={k:max(e[k] for e in errors) for k in errors[0]}, inputs=sources)
    out = HERE/'results'/('accounting_audit_with_heuristic.json' if replay_heuristic else 'accounting_audit.json')
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'inputs'},indent=2))


if __name__ == '__main__':
    audit('--replay-heuristic' in sys.argv)
