"""Build a provenance-tracked primary-only review without touching live outputs.

This is not the complete release analysis: seasons, factorial comparators and
larger scenario settings still require their separate coverage gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import analyze_confirmatory as reporting
import make_v3_figures as plotting
import analyze_round2_reference as reference_reporting
import audit_release_inputs as input_audit
import result_validation
from result_validation import read_aligned_parquet


def audit_primary_snapshot(output, learned):
    """Independent physical replay and validation-selection checks for the core."""
    inputs = output / 'inputs'
    dependency_hashes = {}
    for folder in ('configs', 'data'):
        for source in (learned / folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                relative = source.relative_to(learned)
                canonical = input_audit.ROOT / relative
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if not canonical.exists() or hashlib.sha256(canonical.read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f'primary-audit model input differs: {relative}')
                dependency_hashes[relative.as_posix()] = digest
    params, _, _ = input_audit.regime_bundle('confirmatory', ['winter_weekday'])
    tests = input_audit.load_region_days('confirmatory', 'test')
    validation = [d for d in input_audit.load_region_days('confirmatory', 'val')
                  if d['regime'] == 'winter_weekday'][:30]
    expected = {str(d['date']) for d in tests if d['regime'] == 'winter_weekday'}
    expected = set(sorted(expected)[:30])
    test_days = {str(day['date']): day for day in tests}
    replayed, batches = 0, 0
    for fee in (20, 40, 80):
        name = f'common_daily_confirmatory_thr300_fee{fee}_w10_n2.parquet'
        daily = read_aligned_parquet(inputs / name)
        daily = daily[daily['regime'] == 'winter_weekday']
        if set(daily['method']) != set(input_audit.ONLINE):
            raise RuntimeError('primary comparator set is incomplete')
        for _, group in daily.groupby('method'):
            if len(group) != 30 or set(group['date']) != expected or group['date'].duplicated().any():
                raise RuntimeError('primary date matrix is incomplete')
        input_audit.check_online_parameters(daily, name)
        input_audit.check_daily_accounting(daily, params)
        solves = read_aligned_parquet(inputs / name.replace('common_daily_', 'common_solves_'))
        solves = solves[solves['regime'] == 'winter_weekday']
        if len(solves) != 3 * 30 * 96:
            raise RuntimeError('primary solve matrix is incomplete')
        input_audit.check_solve_records(solves, test_days)
        replayed += input_audit.replay_solver_costs(solves, daily, test_days, params)
        tag = f'v3_winter_fee{fee}'
        policies = read_aligned_parquet(inputs / f'learned_daily_{tag}.parquet')
        input_audit.check_daily_accounting(policies, params)
        if not policies['exact_action_change_events'].eq(0).all():
            raise RuntimeError('learned primary actions changed at the common safety gate')
        batches += input_audit.check_batch_selection(
            output, policies, tag, 'winter_weekday', fee, 25 if fee == 40 else 5,
            validation, params['winter_weekday'])
    reference = read_aligned_parquet(inputs / 'round2_reference_daily.parquet')
    input_audit.check_daily_accounting(reference, params)
    if not reference['exact_action_change_events'].eq(0).all():
        raise RuntimeError('reference primary actions changed at the safety gate')
    if replayed != 270 or batches != 7:
        raise RuntimeError('primary audit cardinalities do not reconcile')
    return {'status': 'primary_input_checks_passed_not_release',
            'independently_replayed_solver_days': replayed,
            'independently_reproduced_batch_selections': batches,
            'source_model_data_sha256': dependency_hashes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    online, learned, output = [path.resolve() for path in
                               (args.online, args.learned, args.output)]
    if any(output == root or root in output.parents for root in (online, learned)):
        raise RuntimeError('the review snapshot must be outside both frozen input workspaces')
    source_state = json.loads((online / 'alignment_progress.json').read_text(encoding='utf-8'))
    for fee in (20, 40, 80):
        for job in (f'primary_fee{fee}', f'rule_fee{fee}_winter_weekday'):
            if source_state['jobs'].get(job, {}).get('status') != 'complete':
                raise RuntimeError(f'primary review waits for complete source job: {job}')
    guard_state = json.loads((learned / 'alignment_progress.json').read_text(encoding='utf-8'))
    if any(value['status'] == 'running' and name.startswith(('select_', 'evaluate_'))
           for name, value in guard_state['jobs'].items()):
        raise RuntimeError('learned-policy inputs are still changing')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('use a new empty primary review output directory')
    inputs, generated, figures = [output / name for name in ('inputs', 'generated', 'figures')]
    for path in (inputs, generated, figures):
        path.mkdir(parents=True, exist_ok=True)
    sources = []
    for fee in (20, 40, 80):
        tag = f'v3_winter_fee{fee}'
        for kind in ('daily', 'solves'):
            sources.append(online / 'experiments' / 'results' /
                           f'common_{kind}_confirmatory_thr300_fee{fee}_w10_n2.parquet')
        sources.append(online / 'experiments' / 'results' /
                       f'rule_tuning_winter_weekday_thr300_fee{fee}_w10.json')
        sources.extend(learned / 'experiments' / 'results' / name for name in
                       (f'learned_daily_{tag}.parquet', f'restart_batches_{tag}.json'))
    sources.append(learned / 'experiments' / 'results' / 'confirmatory_model_loading.csv')
    sources.append(learned / 'experiments' / 'results' / 'round2_reference_daily.parquet')
    provenance = []

    def copy_input(source, target):
        if source.suffix == '.parquet':
            read_aligned_parquet(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'input changed during copy: {source}')
        provenance.append({'source': str(source), 'snapshot': target.relative_to(output).as_posix(),
                           'sha256': digest})

    for source in sources:
        copy_input(source, inputs / source.name)
    # Include every additional input read by the statistics generator.  This
    # makes the review independent of subsequent derived files in the live run.
    for fee in (20, 40, 80):
        tag = f'v3_winter_fee{fee}'
        name = f'restart_batches_{tag}.json'
        copy_input(learned / 'experiments' / 'results' / name,
                   output / 'experiments' / 'results' / name)
        for seed in range(25 if fee == 40 else 5):
            relative = Path('checkpoints') / tag / 'confirmatory' / 'winter_weekday' / f'seed{seed}'
            for name in ('result.json', 'validation_daily.parquet', 'validation_selection.json'):
                copy_input(learned / relative / name, output / relative / name)
        for source in sorted((learned / 'results' / 'logs').glob(f'train_{tag}_*.json')):
            copy_input(source, output / 'results' / 'logs' / source.name)
    audit = audit_primary_snapshot(output, learned)
    (output / 'PRIMARY_INPUT_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    reporting.ROOT, reporting.RESULTS, reporting.GENERATED = output, inputs, generated
    reporting.main()
    reference_reporting.ROOT, reference_reporting.RESULTS, reference_reporting.GENERATED = output, inputs, generated
    reference_reporting.main()
    plotting.RESULTS, plotting.FIGURES = inputs, figures
    plotting.make_cost_latency()
    scripts = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (reporting, reference_reporting, plotting, result_validation, input_audit)}
    scripts[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'reporting_code').mkdir()
    for name in scripts:
        shutil.copy2(Path(__file__).parent / name, output / 'reporting_code' / name)
    manifest = {'status': 'primary_review_only_not_release', 'submission_ready': False,
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'online_workspace': str(online), 'learned_workspace': str(learned),
                'raw_input_sha256': provenance, 'reporting_script_sha256': scripts,
                'primary_input_audit': 'PRIMARY_INPUT_AUDIT.json'}
    (output / 'PRIMARY_REVIEW_PROVENANCE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(manifest['status'])


if __name__ == '__main__':
    main()
