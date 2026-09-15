"""Check the stored hard-tariff comparison and tables without retraining."""
import copy
import json
import hashlib
from pathlib import Path

import numpy as np

from experiments.hard_tariff_dp.report import render
from experiments.manuscript_checks import enabled
from experiments.hard_tariff_dp.train import portable_checkpoint_fields

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def records():
    return tuple(json.loads((HERE / 'results' / name).read_text()) for name in
                 ('summary.json', 'learned_selection.json', 'deployment_audit.json'))


def test_rendered_tables_match_stored_results():
    main, supplement = render(*records())
    assert 'Boundary P95' in main
    assert r'\widehat{\mathcal P}_{h,k}^{a}V_{k+1}^{\mathrm{DP}}' in main
    assert r'Q_k^{\mathrm{DP}}(x,\pi_k(x))-V_k^{\mathrm{DP}}(x)' in main
    assert 'greedy-search defect relative to the learned value' in main
    assert 'DP-optimal continuation for every policy' in supplement
    assert 'Model-based checkpoint selection' in supplement
    if enabled():
        assert main == (ROOT / 'manuscript/hard_tariff_dp_results.tex').read_text()
        assert supplement == (ROOT / 'manuscript/hard_tariff_dp_details.tex').read_text()


def test_gaps_are_computed_from_common_initial_costs():
    summary, selection, audit = records()
    assert len(summary['grids']) == 3
    assert len(selection['rounds']) == 20
    assert {r['seed'] for r in selection['rounds']} == set(range(5))
    for grid in summary['grids']:
        optimum = np.asarray(grid['optimal_initial_costs'])
        for policy in grid['policies'].values():
            np.testing.assert_allclose(np.asarray(policy['initial_costs']) - optimum,
                                       policy['gaps'], atol=1e-8)
            assert min(policy['gaps']) >= -1e-8
            assert policy['regret']['minimum_regret'] >= -1e-7
    assert len(audit['states']) == 88
    assert sum(r['design'] == 'boundary_design' for r in audit['states']) == 24
    for order in (3, 7):
        assert audit[f'max_grid_regret_q{order}'] == max(
            r[f'grid_regret_q{order}'] for r in audit['states'])


def test_cost_prose_uses_input_results():
    summary, selection, audit = records()
    altered = copy.deepcopy(summary)
    row = altered['grids'][-1]['policies']['learned']
    row['initial_costs'][1] += 100
    row['gaps'][1] += 100
    main, _ = render(altered, selection, audit)
    assert f"Learned & {row['initial_costs'][1]:.2f}" in main
    percentage = 100 * row['gaps'][1] / altered['grids'][-1]['optimal_initial_costs'][1]
    assert f'{percentage:.2f}\\% above DP' in main


def test_checkpoint_record_paths_are_portable_and_weights_match_manifest():
    manifest = json.loads((HERE / 'results/training_manifest.json').read_text())
    assert hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
    for seed in manifest['seeds']:
        directory = HERE / 'results' / f"seed{seed['seed']}"
        for name, digest in seed['checkpoints'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
        for name in ('iterations.json', 'result.json'):
            value = json.loads((directory / name).read_text())
            assert portable_checkpoint_fields(value) == value
            rows = value if isinstance(value, list) else value['iterations']
            for row in rows:
                assert not Path(row['checkpoint']).is_absolute()
                assert '\\' not in row['checkpoint']
                assert (ROOT / row['checkpoint']).is_file()


def test_path_normalization_preserves_training_metrics():
    path = HERE / 'results/seed0/iter00.pt'
    original = {'checkpoint': str(path), 'metrics': [1.25, {'cost': -0.5}],
                'selected_checkpoint': str(path)}
    expected = path.relative_to(ROOT).as_posix()
    converted = portable_checkpoint_fields(original)
    assert converted == {'checkpoint': expected, 'metrics': original['metrics'],
                         'selected_checkpoint': expected}
    assert original['checkpoint'] == str(path)
