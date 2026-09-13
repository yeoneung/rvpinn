"""Recompute the retained development gate without changing its stored evidence."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('use a new empty audit output directory')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    script = root / 'scripts' / 'run_fd_verification.py'
    spec = importlib.util.spec_from_file_location('gate_reproduction_source', script)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.OUT = str(output)
    cfg = gate.load_stack(str(root / 'configs' / 'gate_pilot.yaml'))
    params, profiles, _ = gate.regime_bundle('primary', ['winter_weekday'], cfg)
    p, profile = params['winter_weekday'], profiles['winter_weekday']
    solutions = gate.fd_reference(cfg, p, profile, force=True)
    old_path = root / 'results' / 'raw' / 'expA' / 'expA_metrics.json'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    checkpoint_dir = root / 'checkpoints' / 'expA' / 'rvpinnpi_noadapt' / 'seed2'
    rows, errors, hashes = [], [], {}
    for path in (script, old_path, root / 'configs' / 'gate_pilot.yaml'):
        hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for iteration in range(4):
        checkpoint = checkpoint_dir / f'iter{iteration:02d}.pt'
        hashes[checkpoint.relative_to(root).as_posix()] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model = gate.load_checkpoint(str(checkpoint), p, 'cuda')
        actual = gate.evaluate_checkpoint(model, p, profile, solutions[-1],
                                         solutions[-2], cfg, 'cuda')
        matches = [row for row in old if row['method'] == 'rvpinnpi_noadapt'
                   and row['seed'] == 2 and row['iteration'] == iteration]
        if len(matches) != 1:
            raise RuntimeError('development gate iteration provenance is ambiguous')
        expected = matches[0]
        differences = {key: float(value - expected[key]) for key, value in actual.items()}
        failed = [key for key, value in actual.items()
                  if not np.isclose(value, expected[key], atol=1e-6, rtol=1e-8)]
        errors.extend(f'iteration {iteration}: {key}' for key in failed)
        rows.append({'iteration': iteration, 'retained_as_selected': expected['is_selected'],
                     'recomputed': actual, 'difference_from_archived': differences})
        print('iteration', iteration, 'max metric difference', max(map(abs, differences.values())), flush=True)
    report = {'status': 'failed' if errors else 'development_gate_reproduced',
              'scope': 'stored weights; freshly solved two FD grids; original 5000-point design',
              'workspace': str(root), 'source_sha256': hashes, 'iterations': rows,
              'errors': errors}
    (output / 'gate_reproduction_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['status'])
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
