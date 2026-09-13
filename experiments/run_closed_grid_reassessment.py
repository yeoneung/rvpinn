"""Reassess the corrected candidate grid without changing an active old run.

All validation selection and all learned/reference evaluations are repeated.
Unchanged online comparators are copied byte-for-byte only after dependency
checks and their complete original runs. No old learned records are imported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployed_policy import ACTION_SEARCH_VERSION
from src.deployment_numerics import DEPLOYMENT_VERSION
from result_validation import read_aligned_parquet
from run_alignment_reassessment import implementation_digest, main_cells, now


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_online_dependencies(source):
    """Check the entire shared source tree and the online entry points.

    The only run_day difference is an extra output metadata field. Changes to
    NeuralController/reference/grid code do not enter any online comparator.
    All physical, tariff, optimization, observation and data inputs must match.
    """
    compared = {}
    excluded = {'deployed_policy.py', 'reference_policy.py', 'evaluation.py'}
    for old in (source / 'src').glob('*.py'):
        if old.name in excluded:
            continue
        new = ROOT / 'src' / old.name
        if old.read_text(encoding='utf-8') != new.read_text(encoding='utf-8'):
            raise RuntimeError(f'online dependency changed: {old.name}')
        compared[f'src/{old.name}'] = {'source_sha256': sha(old), 'target_sha256': sha(new)}
    old = source / 'src' / 'evaluation.py'
    new = ROOT / 'src' / 'evaluation.py'
    normalized = new.read_text(encoding='utf-8').replace(
        '        "action_search_version": getattr(controller, "action_search_version",\n'
        '                                         "not_applicable"),\n', '').replace(
        '        from .deployed_policy import ACTION_SEARCH_VERSION\n'
        '        self.action_search_version = ACTION_SEARCH_VERSION\n', '')
    if normalized != old.read_text(encoding='utf-8'):
        raise RuntimeError('evaluation changed beyond online-neutral metadata')
    compared['src/evaluation.py'] = {
        'source_sha256': sha(old), 'target_sha256': sha(new),
        'verified_difference': 'output metadata and unused NeuralController metadata only'}
    for name in ('exact_miqp.py', 'stochastic_miqp.py', 'run_common_comparators.py',
                 'run_rule_comparators.py'):
        old, new = source / 'experiments' / name, ROOT / 'experiments' / name
        if old.read_text(encoding='utf-8') != new.read_text(encoding='utf-8'):
            raise RuntimeError(f'online entry point changed: {name}')
        compared[f'experiments/{name}'] = {'source_sha256': sha(old), 'target_sha256': sha(new)}
    for folder in ('configs', 'data'):
        old_files = {p.relative_to(source): p for p in (source / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts}
        new_files = {p.relative_to(ROOT): p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts}
        if old_files.keys() != new_files.keys():
            raise RuntimeError(f'input file set changed: {folder}')
        for relative, old in old_files.items():
            digest = sha(old)
            if digest != sha(new_files[relative]):
                raise RuntimeError(f'online input changed: {relative}')
            compared[relative.as_posix()] = {'sha256': digest}
    return compared


def online_jobs():
    names = []
    for fee in (20, 40, 80):
        names.append(f'primary_fee{fee}')
        seasons = ('winter', 'spring', 'summer', 'autumn') if fee == 40 else ('winter',)
        names.extend(f'rule_fee{fee}_{s}_weekday' for s in seasons)
    names.extend(f'factorial_t{t}_f{f}' for t in (250, 350) for f in (20, 40, 80))
    names.append('primary_fee80_tl15')
    for stream in (7301, 7302, 7303):
        names.extend(f'scenario_s{stream}_f{fee}_m{n}_tl5'
                     for fee, n in ((40, 2), (40, 8), (40, 16), (80, 16)))
        names.append(f'scenario_s{stream}_f80_m16_tl15')
    return names


def source_state(source):
    return json.loads((source / 'alignment_progress.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--check-reuse', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    dependencies = verify_online_dependencies(source)
    if args.check_reuse:
        print(json.dumps({'checked_dependencies': len(dependencies),
                          'online_reuse_eligible': True}))
        return
    marker = json.loads((ROOT / 'ALIGNMENT_WORKSPACE.json').read_text(encoding='utf-8'))
    if marker.get('action_search_version') != ACTION_SEARCH_VERSION:
        raise RuntimeError('prepare a new closed-grid workspace first')
    if source == ROOT:
        raise RuntimeError('source and corrected workspace must be distinct')
    state_path = ROOT / 'alignment_progress.json'
    digest = implementation_digest()
    state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
             else {'created_utc': now(), 'jobs': {}})
    if state.get('implementation_sha256', digest) != digest:
        raise RuntimeError('frozen corrected implementation changed')
    if state.get('status') == 'running' and state.get('pid'):
        import psutil
        if psutil.pid_exists(state['pid']):
            raise RuntimeError(f'corrected runner already active: {state["pid"]}')
    state.update(status='running', pid=os.getpid(), deployment_version=DEPLOYMENT_VERSION,
                 action_search_version=ACTION_SEARCH_VERSION,
                 implementation_sha256=digest, source_workspace=str(source),
                 release_status='not_submission_ready')

    def save():
        state['updated_utc'] = now()
        temp = state_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(state_path)

    save()
    logdir = ROOT / 'experiments' / 'logs' / 'closed_grid'
    logdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1')

    def job(name, script, *arguments):
        if state['jobs'].get(name, {}).get('status') == 'complete':
            return
        command = [sys.executable, str(ROOT / 'experiments' / script),
                   *map(str, arguments)]
        state['jobs'][name] = {'status': 'running', 'started_utc': now(),
                              'command': command, 'log': str(logdir / f'{name}.log')}
        save()
        print(f'START {name}', flush=True)
        with (logdir / f'{name}.log').open('a', encoding='utf-8') as log:
            log.write(f'\nSTART {now()}\n')
            log.flush()
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        state['jobs'][name].update(status='complete' if result.returncode == 0 else 'failed',
                                   exit_code=result.returncode, finished_utc=now())
        save()
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect its retained log')
        print(f'DONE {name}', flush=True)

    try:
        state['phase'] = 'waiting_for_original_gpu_lane'
        save()
        last_old_gpu_job = 'evaluate_v3_sens_t350_f80_w20'
        while True:
            original = source_state(source)
            if original['jobs'].get(last_old_gpu_job, {}).get('status') == 'complete':
                break
            if original.get('status') != 'running':
                raise RuntimeError('original run stopped before its GPU lane completed')
            time.sleep(30)
        state['phase'] = 'repeat_all_affected_validation_and_evaluation'
        save()
        job('reference', 'run_round2_reference.py')
        for tag, regime, fee, seeds in main_cells():
            for seed in seeds:
                run_dir = ROOT / 'checkpoints' / tag / 'confirmatory' / regime / f'seed{seed}'
                job(f'select_{tag}_seed{seed}', 'select_policy_checkpoints.py', run_dir,
                    '--regime', regime, '--fee', fee, '--device', 'cuda')
            job(f'evaluate_{tag}', 'evaluate_restart_batches.py', '--tag', tag,
                '--regime', regime, '--fee', fee, '--seeds', ','.join(map(str, seeds)),
                '--device', 'cuda')
        for threshold in (250, 300, 350):
            for fee in (20, 40, 80):
                for width in (5, 10, 20):
                    if threshold == 300 and width == 10:
                        continue
                    tag = f'v3_sens_t{threshold}_f{fee}_w{width}'
                    run_dir = ROOT / 'checkpoints' / tag / 'confirmatory' / 'winter_weekday' / 'seed0'
                    settings = ('--threshold', threshold, '--fee', fee, '--width', width,
                                '--device', 'cuda')
                    job(f'select_{tag}', 'select_policy_checkpoints.py', run_dir, *settings)
                    job(f'evaluate_{tag}', 'evaluate_selected_policy.py', '--tag', tag, *settings)
        state['phase'] = 'waiting_for_unchanged_online_comparators'
        save()
        while True:
            original = source_state(source)
            statuses = {name: original['jobs'].get(name, {}).get('status')
                        for name in online_jobs()}
            failed = [name for name, status in statuses.items() if status == 'failed']
            if failed:
                raise RuntimeError(f'original online jobs failed: {failed}')
            complete = all(status == 'complete' for status in statuses.values())
            # The old audit failure is expected for its uncorrected mesh.
            # Wait for the old runner to finish its GPU audits before ours.
            if complete and original.get('status') != 'running':
                break
            if not complete and original.get('status') != 'running':
                raise RuntimeError('original runner stopped with online jobs incomplete')
            state['source_online_jobs_complete'] = sum(s == 'complete' for s in statuses.values())
            state['source_online_jobs_expected'] = len(statuses)
            save()
            time.sleep(30)
        dependencies = verify_online_dependencies(source)
        source_results = source / 'experiments' / 'results'
        target_results = ROOT / 'experiments' / 'results'
        imported = []
        for pattern in ('common_daily_*.parquet', 'common_solves_*.parquet', 'rule_tuning_*.json'):
            for old in sorted(source_results.glob(pattern)):
                if old.suffix == '.parquet':
                    frame = read_aligned_parquet(old)
                    allowed = {'no_storage', 'deterministic_exact_band_miqp',
                               'stochastic_two_stage_exact_band_miqp',
                               'convex_envelope_mpc', 'validation_tuned_price_rule'}
                    if not set(frame['method']).issubset(allowed):
                        raise RuntimeError(f'affected policy cannot be reused: {old}')
                new = target_results / old.name
                digest = sha(old)
                if new.exists() and sha(new) != digest:
                    raise RuntimeError(f'conflicting online artifact: {new}')
                if not new.exists():
                    shutil.copy2(old, new)
                if sha(new) != digest:
                    raise RuntimeError(f'copy integrity failure: {new}')
                imported.append({'file': old.name, 'sha256': digest})
        provenance = {'source_workspace': str(source), 'imported_utc': now(),
                      'source_implementation_sha256': original['implementation_sha256'],
                      'target_implementation_sha256': state['implementation_sha256'],
                      'policy': 'byte-identical reuse of unaffected online procedures only',
                      'dependencies': dependencies, 'artifacts': imported}
        (ROOT / 'ONLINE_REUSE_PROVENANCE.json').write_text(
            json.dumps(provenance, indent=2), encoding='utf-8')
        state['phase'] = 'audits_analysis_and_draft_compilation'
        save()
        for script in ('audit_fixed_policy_gate.py', 'audit_selected_derivatives.py',
                       'audit_direct_vs_projection.py', 'audit_deployed_bellman.py',
                       'audit_model_loading.py', 'audit_reflected_forecast.py',
                       'analyze_confirmatory.py', 'analyze_seasonal_sensitivity.py',
                       'analyze_round2_reference.py', 'analyze_round2_scenarios.py',
                       'make_v3_figures.py', 'make_round2_parameter_tables.py'):
            job(Path(script).stem, script)
        for controller in ('learned', 'reference', 'm2', 'm16'):
            job(f'mechanism_{controller}', 'make_round2_mechanism_figure.py', '--controller', controller)
        job('mechanism_assemble', 'make_round2_mechanism_figure.py', '--assemble')
        job('verify_round2', 'verify_round2_results.py')
        job('verify_alignment', 'verify_alignment_results.py')
        job('compile_drafts', 'compile_manuscripts.py')
        state['status'] = 'results_ready_pending_manuscript_reconciliation'
        state['phase'] = 'manual_numerical_prose_and_submission_review_required'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
