"""Complete omitted seasonal comparator days in a separate immutable snapshot.

The original runner passed several regimes to a globally truncated CLI. This
follow-up uses one regime per invocation of that same frozen online code. It
does not alter any completed winter run, checkpoint, solver budget, or seed.
All source online jobs must finish before its six single-thread workers start.
Final aggregation must merge these disjoint seasonal rows with the corrected
main artifacts and rerun the complete release checks.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_online_jobs():
    names = [f'primary_fee{fee}' for fee in (20, 40, 80)]
    for fee in (20, 40, 80):
        seasons = ('winter', 'spring', 'summer', 'autumn') if fee == 40 else ('winter',)
        names.extend(f'rule_fee{fee}_{season}_weekday' for season in seasons)
    names.extend(f'factorial_t{threshold}_f{fee}'
                 for threshold in (250, 350) for fee in (20, 40, 80))
    names.append('primary_fee80_tl15')
    for stream in (7301, 7302, 7303):
        names.extend(f'scenario_s{stream}_f{fee}_m{count}_tl5'
                     for fee, count in ((40, 2), (40, 8), (40, 16), (80, 16)))
        names.append(f'scenario_s{stream}_f80_m16_tl15')
    return names


def prepare(source, target):
    if target.exists():
        raise RuntimeError(f'new supplemental workspace required: {target}')
    target.mkdir()
    for name in ('src', 'configs', 'data'):
        shutil.copytree(source / name, target / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    (target / 'experiments').mkdir()
    for path in (source / 'experiments').glob('*.py'):
        shutil.copy2(path, target / 'experiments' / path.name)
    (target / 'experiments' / 'results').mkdir()
    for name in ('requirements.txt',):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    files = {path.relative_to(target).as_posix(): sha(path)
             for folder in ('src', 'configs', 'data', 'experiments')
             for path in (target / folder).rglob('*') if path.is_file()}
    manifest = {
        'created_utc': now(), 'source_workspace': str(source),
        'purpose': 'complete first 30 test weekdays separately in each omitted season',
        'frozen_files': files, 'primary_fee': 40, 'seasons': ['spring', 'summer', 'autumn'],
        'seeds_budgets_models_changed': False,
        'release_status': 'not_submission_ready',
    }
    (target / 'SEASONAL_COMPLETION_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if source.parent != ROOT or target.parent != ROOT or source == target:
        raise RuntimeError('source and target must be distinct direct children of the study workspace')
    if not (source / 'ALIGNMENT_WORKSPACE.json').exists():
        raise RuntimeError('source must be the frozen alignment workspace')
    if not target.exists():
        prepare(source, target)
    manifest_path = target / 'SEASONAL_COMPLETION_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['source_workspace'] != str(source):
        raise RuntimeError('supplemental source does not match')
    for relative, digest in manifest['frozen_files'].items():
        if sha(target / relative) != digest:
            raise RuntimeError(f'frozen supplemental input changed: {relative}')
    if args.prepare_only:
        print(f'prepared {target}', flush=True)
        return

    state_path = target / 'seasonal_completion_progress.json'
    state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
             else {'created_utc': now(), 'jobs': {}})
    if state.get('status') == 'running' and psutil.pid_exists(state.get('pid', -1)):
        raise RuntimeError('supplemental runner already active')
    state.update(status='running', pid=os.getpid(),
                 phase='waiting_for_original_online_workers',
                 runner_sha256=sha(Path(__file__)), release_status='not_submission_ready')

    def save():
        state['updated_utc'] = now()
        temp = state_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(state_path)

    save()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1')
    logdir = target / 'experiments' / 'logs' / 'seasonal_completion'
    logdir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            original = json.loads((source / 'alignment_progress.json').read_text(encoding='utf-8'))
            complete = sum(original['jobs'].get(name, {}).get('status') == 'complete'
                           for name in expected_online_jobs())
            state['source_online_complete'] = complete
            state['source_online_expected'] = len(expected_online_jobs())
            save()
            if original['status'] != 'running':
                if complete != len(expected_online_jobs()):
                    raise RuntimeError('original workflow stopped with incomplete online jobs')
                break
            if not psutil.pid_exists(original['pid']):
                raise RuntimeError('original runner disappeared while marked running')
            time.sleep(30)
        state['phase'] = 'complete_disjoint_seasonal_evaluations'
        save()
        for season in ('spring', 'summer', 'autumn'):
            if state['jobs'].get(season, {}).get('status') == 'complete':
                continue
            command = [sys.executable, str(target / 'experiments' / 'run_common_comparators.py'),
                       '--fee', '40', '--regimes', f'{season}_weekday', '--max-days', '30',
                       '--workers', '6', '--methods',
                       'no_storage,deterministic_exact_band_miqp,stochastic_two_stage_exact_band_miqp,convex_envelope_mpc']
            state['jobs'][season] = {'status': 'running', 'started_utc': now(),
                                      'command': command, 'workers': 6}
            save()
            print(f'START {season}', flush=True)
            with (logdir / f'{season}.log').open('a', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=target, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            state['jobs'][season].update(
                status='complete' if result.returncode == 0 else 'failed',
                exit_code=result.returncode, finished_utc=now())
            save()
            if result.returncode:
                raise RuntimeError(f'{season} completion failed; inspect retained log')
            print(f'DONE {season}', flush=True)

        import pandas as pd
        daily = pd.read_parquet(target / 'experiments/results/common_daily_confirmatory_thr300_fee40_w10_n2.parquet')
        solves = pd.read_parquet(target / 'experiments/results/common_solves_confirmatory_thr300_fee40_w10_n2.parquet')
        if (len(daily) != 3 * 4 * 30
                or not daily.groupby(['regime', 'method']).size().eq(30).all()
                or len(solves) != 3 * 3 * 30 * 96
                or not solves.groupby(['regime', 'method', 'date']).size().eq(96).all()):
            raise RuntimeError('incomplete supplemental seasonal cardinalities')
        state['artifact_sha256'] = {
            path.name: sha(path) for path in (target / 'experiments/results').glob('*.parquet')}
        state['status'] = 'complete_pending_disjoint_merge_and_release_audit'
        state['phase'] = 'manual_reconciliation_required'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
