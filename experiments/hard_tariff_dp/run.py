"""Solve the frozen reduced problem; no primary experiment is rerun."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.hard_tariff_dp.train import bundle, HERE
from experiments.hard_tariff_dp.core import GridDP

OUT = HERE/'results'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def solve_grid(spec):
    protocol, _, p, prof, _ = bundle()
    grid = GridDP(p, prof, *spec)
    tag = 'x'.join(map(str, spec))
    path = OUT/f'dp_{tag}.npz'
    if path.exists():
        with np.load(path) as archive:
            V, A = archive['V'], archive['A']
        return grid, V, A
    started = time.perf_counter()
    print('Solving DP', tag, flush=True)
    V, A = grid.solve()
    np.savez_compressed(path, V=V, A=A, s=grid.s, y=grid.y, transition=grid.P)
    diag = dict(grid=spec, actual_y_points=grid.ny, wall_s=time.perf_counter()-started,
                initial_costs=grid.initial_costs(V[0]).tolist(),
                self_evaluation_error=float(np.max(abs(grid.evaluate(A)-grid.initial_costs(V[0])))))
    dump(OUT/f'dp_{tag}.json', diag)
    print(diag, flush=True)
    return grid, V, A


def select_rule(grid):
    path = OUT/'rule_selection.json'
    if path.exists(): return json.loads(path.read_text())
    manifest = json.loads((ROOT/'experiments/controller_comparisons/results/heuristic/manifest.json').read_text())
    rows, incumbent, best = [], 0, np.inf
    for i, setting in enumerate(manifest['candidates']):
        costs = grid.evaluate(grid.rule_actions(setting))
        rows.append(dict(index=i, setting=setting, initial_costs=costs.tolist()))
        if i == 0 or best-costs[1] >= .10:
            incumbent, best = i, float(costs[1])
        if i % 16 == 0:
            print('Rule settings', i, 'selected cost', best, flush=True)
    result = dict(selected=rows[incumbent], candidates=rows)
    dump(path, result)
    return result


def select_learned(grid):
    import torch
    from src.policy_iteration import load_checkpoint
    torch.set_num_threads(1)
    protocol, _, p, _, _ = bundle()
    manifest = json.loads((OUT/'training_manifest.json').read_text())
    assert len(manifest['seeds']) == len(protocol['training_seeds']), 'Training is not complete'
    path = OUT/'learned_selection.json'
    if path.exists(): return json.loads(path.read_text())
    rows, winners = [], []
    for seed in protocol['training_seeds']:
        candidate_path = OUT/f'seed{seed}_selection.json'
        if candidate_path.exists():
            report = json.loads(candidate_path.read_text())
            rows += report['rounds']; winners.append(report['winner'])
            continue
        best, winner, seed_rows = np.inf, None, []
        for iteration in range(protocol['training_rounds']):
            checkpoint = OUT/f'seed{seed}/iter{iteration:02d}.pt'
            model = load_checkpoint(str(checkpoint), p, 'cuda' if torch.cuda.is_available() else 'cpu')
            started = time.perf_counter()
            actions = grid.learned_actions(model)
            costs = grid.evaluate(actions)
            row = dict(seed=seed, iteration=iteration, initial_costs=costs.tolist(),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       evaluation_wall_s=time.perf_counter()-started)
            rows.append(row); seed_rows.append(row)
            if winner is None or best-costs[1] >= .10:
                best, winner = float(costs[1]), row
            print('Neural selection', row, flush=True)
        winners.append(winner)
        dump(candidate_path, dict(rounds=seed_rows, winner=winner))
    best = float(grid.evaluate(grid.rule_actions(None))[1])
    selected = None
    for row in winners:
        if best-row['initial_costs'][1] >= .10:
            best, selected = row['initial_costs'][1], row
    result = dict(selected=selected, winners=winners, rounds=rows, selected_cost=best)
    dump(path, result)
    return result


def evaluate_grid(spec, rule, learned):
    import torch
    from src.policy_iteration import load_checkpoint
    torch.set_num_threads(1)
    path = OUT/('evaluation_'+'x'.join(map(str, spec))+'.json')
    if path.exists(): return json.loads(path.read_text())
    grid, V, A = solve_grid(spec)
    policies = dict(reference=grid.reference_actions(), rule=grid.rule_actions(rule['selected']['setting']))
    selected = learned['selected']
    if selected is None:
        policies['learned'] = grid.rule_actions(None)
    else:
        ckpt = OUT/f"seed{selected['seed']}/iter{selected['iteration']:02d}.pt"
        model = load_checkpoint(str(ckpt), grid.p, 'cuda' if torch.cuda.is_available() else 'cpu')
        policies['learned'] = grid.learned_actions(model)
    optimum = grid.initial_costs(V[0])
    result = dict(grid=spec, actual_y_points=grid.ny, optimal_initial_costs=optimum.tolist(), policies={})
    for name, policy in policies.items():
        costs = grid.evaluate(policy)
        result['policies'][name] = dict(initial_costs=costs.tolist(), gaps=(costs-optimum).tolist(),
                                       regret=grid.regret(V, A, policy))
        assert result['policies'][name]['regret']['minimum_regret'] >= -1e-6
    np.savez_compressed(OUT/('policies_'+'x'.join(map(str, spec))+'.npz'), **policies)
    dump(path, result)
    print('Grid evaluation', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['reference', 'selection', 'evaluation'], required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol, _, _, _, _ = bundle()
    grid, _, _ = solve_grid(protocol['selection_grid'])
    rule = select_rule(grid)
    if args.stage == 'reference':
        for spec in protocol['grids'][1:]: solve_grid(spec)
        return
    learned = select_learned(grid)
    if args.stage == 'evaluation':
        results = [evaluate_grid(spec, rule, learned) for spec in protocol['grids']]
        dump(OUT/'summary.json', dict(protocol=protocol, selected_rule=rule['selected'],
                                     selected_learned=learned['selected'], grids=results))


if __name__ == '__main__':
    main()
