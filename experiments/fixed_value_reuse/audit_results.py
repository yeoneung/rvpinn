"""Independent reconstruction of every trajectory's hard cost and safety."""
import json
import numpy as np
from common import HERE, RESULTS, bundle


def audit():
    p, _ = bundle()
    manifest = json.loads((RESULTS/'manifest.json').read_text())
    cases = {c['id']:c for c in manifest['cases']}
    audited, max_cost_error, max_soc_error, worst_violation = 0, 0., 0., 0.
    for path in sorted(RESULTS.glob('*.json')):
        if not (path.name.startswith('neural_n') or (path.name.startswith('miqp_n') and 'partial' not in path.name)):
            continue
        group = json.loads(path.read_text())
        for entry in group['results']:
            if 'error' in entry:
                raise RuntimeError(entry['error'])
            rows = entry['rows'] if 'rows' in entry else [entry['row']]
            for row in rows:
                case = cases[row['case_id']]
                s, costs = case['initial_soc'], np.zeros(3)
                for k, a in enumerate(row['actions']):
                    # Independently spell out both continuous taper and held-action bounds.
                    low = -min(p.a_c*min((p.s_max-s)/p.delta_s,1.),
                               p.E_max*(p.s_max-s)/(p.eta_c*p.dt_ctrl))
                    high = min(p.a_d*min((s-p.s_min)/p.delta_s,1.),
                               p.eta_d*p.E_max*(s-p.s_min)/p.dt_ctrl)
                    worst_violation = max(worst_violation, low-a, a-high, 0.)
                    g, price = case['net'][k]-a, case['price'][k]
                    costs += p.dt_ctrl*np.array([price*(max(g,0.)-p.alpha_s*max(-g,0.)),
                                                 p.c_step*float(g>p.g_thr), p.lam1*abs(a)])
                    for _ in range(3):
                        s += p.dt_sim*(p.eta_c*max(-a,0.)-max(a,0.)/p.eta_d)/p.E_max
                    worst_violation = max(worst_violation, p.s_min-s, s-p.s_max, 0.)
                total = float(costs.sum()+p.lam_T*(s-p.s_tar)**2)
                max_cost_error=max(max_cost_error,abs(total-row['cost']))
                max_soc_error=max(max_soc_error,abs(s-row['terminal_soc']))
                audited += 1
    result=dict(audited_trajectories=audited, max_cost_error_eur=max_cost_error,
                max_terminal_soc_error=max_soc_error, max_bound_violation=worst_violation)
    (RESULTS/'accounting_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert max_cost_error < 1e-7 and max_soc_error < 1e-10 and worst_violation < 1e-7, result
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    audit()
