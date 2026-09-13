"""Finish local corrected reports after every frozen run has completed.

Default: read-only readiness report. --run --wait starts the local continuation
queue. It preserves previous raw artifacts through the checked merge, audits
all inputs, regenerates reports and compiles visibly pending drafts. It never
removes submission warnings, edits handwritten conclusions, packages or pushes
a release. A failed stage stops the queue with its log and backup intact.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from merge_corrected_inputs import ready_inputs

STATE_NAME = 'CORRECTED_DRAFT_FINALIZATION.json'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_state_atomic(state_path, state, *, attempts=40, retry_delay=0.25,
                       replace=None, sleep=None):
    """Retain valid JSON while tolerating brief Windows sharing conflicts.

    A persistent permission error still fails and retains the temporary file;
    unrelated I/O errors are not retried or hidden.
    """
    if attempts < 1 or retry_delay < 0:
        raise ValueError('state-write attempts must be positive and delay nonnegative')
    temporary = state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    replace = os.replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    for attempt in range(attempts):
        try:
            replace(temporary, state_path)
            return attempt
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            sleep(retry_delay)


def frozen_dependencies(root):
    """Generated results/PDFs may change; code and handwritten sources may not."""
    paths = set()
    for folder in ('src', 'scripts', 'experiments', 'tests'):
        paths.update(path for path in (root / folder).rglob('*.py')
                     if '__pycache__' not in path.parts)
    for folder in ('configs', 'data'):
        paths.update(path for path in (root / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    paths.update((root / 'results' / 'logs').glob('train_*.json'))
    paths.update((root / 'checkpoints').rglob('*.pt'))
    for folder, names in (
            ('manuscript', ('main.tex', 'supplement.tex', 'references.bib',
                            'alignment_pending.tex')),
            ('', ('cover_letter.tex', 'highlights.txt', 'highlights.docx'))):
        paths.update(root / folder / name for name in names)
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(paths)}


def check_dependencies(root, expected):
    current = frozen_dependencies(root)
    changed = sorted(name for name in set(current) | set(expected)
                     if current.get(name) != expected.get(name))
    if changed:
        raise RuntimeError(f'protected inputs changed after queue start: {changed}')


def blocking_failures(states, reasons):
    """Expected old analysis failures do not excuse incomplete raw/audit jobs."""
    failures = []
    for name, state in states.items():
        if state.get('status') in ('failed', 'interrupted', 'terminated'):
            pending = [reason for reason in reasons if reason.startswith(name + ':')]
            if pending:
                failures.append(f'{name}: stopped with outstanding prerequisites: {pending}')
    return failures


def missing_running_processes(states, pid_exists=psutil.pid_exists):
    return [name for name, state in states.items()
            if state.get('status') == 'running'
            and (not isinstance(state.get('pid'), int) or state['pid'] <= 0
                 or not pid_exists(state['pid']))]


def stage_commands():
    """The complete local pipeline; no publication or warning removal."""
    scripts = [
        ('tests', ['-m', 'pytest', 'tests', '-q']),
        ('merge_raw_inputs', ['experiments/merge_corrected_inputs.py', '--apply']),
        ('input_accounting', ['experiments/audit_release_inputs.py', '--workspace', '.',
                              '--report', 'experiments/results/corrected_input_audit.json']),
    ]
    for name in ('audit_fixed_policy_gate', 'audit_selected_derivatives',
                 'audit_direct_vs_projection', 'audit_deployed_bellman',
                 'audit_model_loading', 'audit_reflected_forecast',
                 'analyze_confirmatory', 'analyze_seasonal_sensitivity',
                 'analyze_round2_reference', 'analyze_round2_scenarios',
                 'make_round2_parameter_tables', 'make_v3_figures'):
        scripts.append((name, [f'experiments/{name}.py']))
    for controller in ('learned', 'reference', 'm2', 'm16'):
        scripts.append((f'mechanism_{controller}',
                        ['experiments/make_round2_mechanism_figure.py',
                         '--controller', controller]))
    scripts.extend([
        ('mechanism_assemble', ['experiments/make_round2_mechanism_figure.py', '--assemble']),
        ('verify_round2', ['experiments/verify_round2_results.py']),
        ('verify_alignment', ['experiments/verify_alignment_results.py']),
        ('input_accounting_final', ['experiments/audit_release_inputs.py', '--workspace', '.',
                                    '--report', 'experiments/results/corrected_input_audit.json']),
        ('assemble_complete_draft_sections', ['experiments/assemble_corrected_sections.py']),
        ('compile_pending_drafts', ['experiments/compile_manuscripts.py']),
        ('format_pending_drafts', ['experiments/audit_manuscript_format.py',
                                   '--report', 'experiments/results/corrected_draft_format.json']),
    ])
    return scripts


def output_hashes(root):
    paths = []
    for folder in ('experiments/results', 'manuscript/generated', 'manuscript/figures'):
        paths.extend(path for path in (root / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts)
    paths += [root / 'manuscript' / f'{name}.pdf' for name in ('main', 'supplement')]
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='execute the local continuation pipeline')
    parser.add_argument('--wait', action='store_true', help='wait for every frozen-run prerequisite')
    args = parser.parse_args()
    if args.wait and not args.run:
        parser.error('--wait requires --run; the default is read-only')
    sources = [ROOT / name for name in
               ('alignment_run', 'alignment_guard_run', 'alignment_seasonal_run')]
    reasons, states = ready_inputs(*sources)
    if not args.run:
        print(json.dumps({'ready_to_start': not reasons, 'reasons': reasons,
                          'stopped_prerequisites': blocking_failures(states, reasons),
                          'submission_ready': False}, indent=2))
        return
    if reasons and not args.wait:
        raise RuntimeError('prerequisites are incomplete; use --run --wait to queue local continuation')
    if (ROOT / 'CORRECTED_INPUT_MERGE.json').exists():
        raise RuntimeError('a canonical merge record already exists; inspect it before rerunning')
    dependencies = frozen_dependencies(ROOT)
    state_path = ROOT / STATE_NAME
    state = {'status': 'running', 'phase': 'waiting_for_frozen_inputs',
             'pid': os.getpid(), 'created_utc': now(), 'submission_ready': False,
             'protected_sha256': dependencies, 'jobs': {},
             'scope': 'checked local reports and visibly pending PDFs only'}
    # Exclusive creation protects an already running or previously failed queue.
    with state_path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)
    logdir = ROOT / 'experiments' / 'logs' / 'corrected_finalization'
    logdir.mkdir(parents=True, exist_ok=True)

    def save():
        state['updated_utc'] = now()
        retries = write_state_atomic(state_path, state)
        if retries:
            print(f'Recovered state-file replacement after {retries} retries.', flush=True)

    try:
        while True:
            check_dependencies(ROOT, dependencies)
            reasons, states = ready_inputs(*sources)
            failures = blocking_failures(states, reasons)
            if failures:
                raise RuntimeError('; '.join(failures))
            missing = missing_running_processes(states)
            if missing:
                raise RuntimeError(f'prerequisite runners disappeared while marked active: {missing}')
            state['pending_prerequisites'] = reasons
            save()
            if not reasons:
                break
            time.sleep(30)
        state['phase'] = 'checked_merge_audits_reports_and_draft_compilation'
        save()
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        for name, arguments in stage_commands():
            check_dependencies(ROOT, dependencies)
            if not (ROOT / 'manuscript' / 'alignment_pending.tex').exists():
                raise RuntimeError('draft warning was removed during numerical reconciliation')
            command = [sys.executable, *arguments]
            log_path = logdir / f'{name}.log'
            state['jobs'][name] = {'status': 'running', 'started_utc': now(),
                                   'command': command, 'log': str(log_path)}
            save()
            print(f'START {name}', flush=True)
            with log_path.open('x', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            state['jobs'][name].update(
                status='complete' if result.returncode == 0 else 'failed',
                exit_code=result.returncode, finished_utc=now())
            save()
            if result.returncode:
                raise RuntimeError(f'{name} failed; inspect its retained log')
            print(f'DONE {name}', flush=True)
        check_dependencies(ROOT, dependencies)
        state.update(status='reports_ready_pending_manual_reconciliation',
                     phase='review_complete_tables_and_numerical_message',
                     output_sha256=output_hashes(ROOT),
                     remaining=['manually reconcile every full-result statement with the checked tables',
                                'visually inspect the compiled complete <=30-page main paper',
                                'check submission materials and the separately requested public release'])
    except BaseException as error:
        state.update(status='failed', error=str(error))
        raise
    finally:
        save()
    print(state['status'], flush=True)


if __name__ == '__main__':
    main()
