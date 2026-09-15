"""Compare the benchmark's interpolated continuation with native neural search.

States are selected deterministically without using either policy's cost.
This finite diagnostic is not a uniform interpolation/expectation certificate.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.hard_tariff_dp.train import bundle, HERE
from experiments.hard_tariff_dp.core import GridDP
from src.policy_iteration import load_checkpoint
from src.deployed_policy import deployed_action_candidates, deployed_one_step_q


def main():
    torch.set_num_threads(1)
    protocol, _, p, prof, _ = bundle()
    out = HERE/'results'
    selection = json.loads((out/'learned_selection.json').read_text())['selected']
    model = load_checkpoint(str(out/f"seed{selection['seed']}/iter{selection['iteration']:02d}.pt"), p,
                            'cuda' if torch.cuda.is_available() else 'cpu')
    spec = protocol['grids'][-1]
    grid = GridDP(p, prof, *spec)
    tag = 'x'.join(map(str, spec))
    with np.load(out/f'policies_{tag}.npz') as data:
        actions = data['learned']
    records, states = [], []
    # Four dispatch times, three initial SoCs, two load regimes: 24 states.
    for k in (16, 40, 64, 88):
        for s0 in (.2, .5, .8):
            i = int(np.argmin(abs(grid.s-s0)))
            for net_target in (p.g_thr, p.g_thr+100.):
                y0 = np.clip(net_target-float(prof.n_bar(np.array([grid.times[k]]))[0]),p.y_min,p.y_max)
                j = int(np.argmin(abs(grid.y-y0)))
                states.append((k,i,j,'boundary_design'))
    rng = np.random.default_rng(9287)
    states += [(int(rng.integers(grid.K)), int(rng.integers(grid.ns)),
                int(rng.integers(grid.ny)), 'uniform_grid') for _ in range(64)]
    for k,i,j,design in states:
        s, y, a = float(grid.s[i]), float(grid.y[j]), float(actions[k,i,j])
        net = float(prof.n_bar(np.array([grid.times[k]]))[0])+y
        candidates = np.r_[deployed_action_candidates(p,s,net,p.dt_ctrl,1025),a]
        q3 = deployed_one_step_q(model,grid.times[k],s,y,0.,prof,candidates,quadrature_order=3)
        q7 = deployed_one_step_q(model,grid.times[k],s,y,0.,prof,candidates,quadrature_order=7)
        records.append(dict(k=k,s=s,y=y,design=design,grid_action=a,
            native_action=float(candidates[np.argmin(q3[:-1])]),
            high_order_action=float(candidates[np.argmin(q7[:-1])]),
            grid_regret_q3=float(max(0.,q3[-1]-q3[:-1].min())),
            grid_regret_q7=float(max(0.,q7[-1]-q7[:-1].min())),
            max_q3_q7_difference=float(np.max(abs(q3-q7)))))
    result = dict(states=records,
        max_grid_action_difference_q3=max(abs(r['grid_action']-r['native_action']) for r in records),
        max_grid_regret_q3=max(r['grid_regret_q3'] for r in records),
        max_grid_regret_q7=max(r['grid_regret_q7'] for r in records),
        mean_grid_regret_q7=float(np.mean([r['grid_regret_q7'] for r in records])))
    (out/'deployment_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='states'},indent=2))


if __name__ == '__main__': main()
