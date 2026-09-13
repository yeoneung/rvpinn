"""Restartable correction workflow, isolated from the archived original study.

One GPU lane and at most three two-worker MIQP lanes run concurrently.
All commands and completion states are retained. Tables are regenerated only
after every evaluation lane succeeds; manuscript release remains a separate
review because numerical prose must be reconciled with the corrected results.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployment_numerics import DEPLOYMENT_VERSION

STATE = ROOT / 'alignment_progress.json'
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def implementation_digest():
    digest = hashlib.sha256()
    for folder, pattern in [('src', '*.py'), ('experiments', '*.py'),
                             ('configs', '*.yaml')]:
        for path in sorted((ROOT / folder).rglob(pattern)):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main_cells():
    return [(f'v3_{season}_fee{fee}', f'{season}_weekday', fee, seeds)
            for season, fee, seeds in [
                ('winter', 20, list(range(5))), ('winter', 40, list(range(25))),
                ('winter', 80, list(range(5))), ('spring', 40, list(range(5))),
                ('summer', 40, list(range(5))), ('autumn', 40, list(range(5)))]]


def main():
    if not (ROOT / 'ALIGNMENT_WORKSPACE.json').exists():
        raise RuntimeError('run prepare_alignment_workspace.py, then execute its copied runner')
    digest = implementation_digest()
    state = (json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists()
             else {'created_utc': now(), 'jobs': {}})
    if state.get('implementation_sha256', digest) != digest:
        raise RuntimeError('implementation changed; do not mix outputs across code revisions')
    previous_pid = state.get('pid')
    if state.get('status') == 'running' and previous_pid and os.name == 'nt':
        listing = subprocess.run(['tasklist', '/FI', f'PID eq {previous_pid}',
                                  '/FO', 'CSV', '/NH'], capture_output=True, text=True)
        if f'"{previous_pid}"' in listing.stdout:
            raise RuntimeError(f'runner already active: PID {previous_pid}')
    state.update(status='running', pid=os.getpid(), updated_utc=now(),
                 deployment_version=DEPLOYMENT_VERSION, implementation_sha256=digest,
                 max_concurrent_single_thread_miqp_solves=6,
                 release_status='not_submission_ready')

    def save():
        state['updated_utc'] = now()
        temp = STATE.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(STATE)

    save()
    logdir = ROOT / 'experiments' / 'logs' / 'alignment'
    logdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', PYTHONUTF8='1')

    def job(name, script, *args):
        command = [sys.executable, str(ROOT / 'experiments' / script), *map(str, args)]
        with LOCK:
            if state['jobs'].get(name, {}).get('status') == 'complete':
                return
            state['jobs'][name] = {'status': 'running', 'started_utc': now(),
                                   'command': command, 'log': str(logdir / f'{name}.log')}
            save()
        print(f'START {name}', flush=True)
        with (logdir / f'{name}.log').open('a', encoding='utf-8') as log:
            log.write(f'\nSTART {now()}\n')
            log.flush()
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        with LOCK:
            state['jobs'][name].update(
                status='complete' if completed.returncode == 0 else 'failed',
                finished_utc=now(), exit_code=completed.returncode)
            save()
        if completed.returncode:
            raise RuntimeError(f'{name} failed; inspect {logdir / (name + ".log")}')
        print(f'DONE {name}', flush=True)

    def learner_lane():
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
                    args = ('--threshold', threshold, '--fee', fee, '--width', width,
                            '--device', 'cuda')
                    job(f'select_{tag}', 'select_policy_checkpoints.py', run_dir, *args)
                    job(f'evaluate_{tag}', 'evaluate_selected_policy.py', '--tag', tag, *args)

    def primary_lane():
        for fee in (20, 40, 80):
            regimes = (['winter_weekday', 'spring_weekday', 'summer_weekday', 'autumn_weekday']
                       if fee == 40 else ['winter_weekday'])
            job(f'primary_fee{fee}', 'run_common_comparators.py', '--fee', fee,
                '--regimes', ','.join(regimes), '--workers', 2,
                '--methods', 'no_storage,deterministic_exact_band_miqp,stochastic_two_stage_exact_band_miqp,convex_envelope_mpc')
            for regime in regimes:
                job(f'rule_fee{fee}_{regime}', 'run_rule_comparators.py',
                    '--regime', regime, '--fee', fee)
        for threshold in (250, 350):
            for fee in (20, 40, 80):
                job(f'factorial_t{threshold}_f{fee}', 'run_common_comparators.py',
                    '--threshold', threshold, '--fee', fee, '--workers', 2,
                    '--methods', 'no_storage,deterministic_exact_band_miqp')
        job('primary_fee80_tl15', 'run_common_comparators.py', '--fee', 80,
            '--time-limit', 15, '--output-suffix', 'tl15', '--workers', 2,
            '--methods', 'deterministic_exact_band_miqp,stochastic_two_stage_exact_band_miqp')

    def scenario_lane(stream):
        for fee, count in [(40, 2), (40, 8), (40, 16), (80, 16)]:
            job(f'scenario_s{stream}_f{fee}_m{count}_tl5', 'run_common_comparators.py',
                '--fee', fee, '--scenarios', count, '--scenario-pool-size', 16,
                '--seed', stream, '--workers', 2,
                '--output-suffix', f'r2_stream{stream}_pool16_tl5',
                '--methods', 'stochastic_two_stage_exact_band_miqp')
        # Repeat the previously executed 15-second robustness setting for
        # every stream, regardless of the corrected performance ranking.
        job(f'scenario_s{stream}_f80_m16_tl15', 'run_common_comparators.py',
            '--fee', 80, '--scenarios', 16, '--scenario-pool-size', 16,
            '--seed', stream, '--workers', 2, '--time-limit', 15,
            '--output-suffix', f'r2_stream{stream}_pool16_tl15',
            '--methods', 'stochastic_two_stage_exact_band_miqp')

    failures = []
    try:
        with ThreadPoolExecutor(max_workers=1) as gpu, ThreadPoolExecutor(max_workers=3) as online:
            futures = [gpu.submit(learner_lane), online.submit(primary_lane)]
            futures.extend(online.submit(scenario_lane, stream) for stream in (7301, 7302, 7303))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures.append(str(error))
                    with LOCK:
                        state['lane_failures'] = failures.copy()
                        save()
        if failures:
            raise RuntimeError('; '.join(failures))
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
        state['next_action'] = ('Reconcile numerical prose, uncertainty statements and claims; '
                                'review generated tables, then rebuild canonical submission files.')
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        with LOCK:
            save()


if __name__ == '__main__':
    main()
