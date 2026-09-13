import json
import os

import pytest

from experiments.finalize_corrected_draft import (
    blocking_failures, check_dependencies, frozen_dependencies,
    missing_running_processes, stage_commands, write_state_atomic)
from experiments.assemble_corrected_sections import (
    assembly_audit_commands, checked_include_text, section_sources)


def test_expected_old_analysis_failure_does_not_waive_missing_raw_outputs():
    states = {'original': {'status': 'failed'}, 'corrected': {'status': 'running'}}
    reasons = ['corrected: runner is still active']
    assert blocking_failures(states, reasons) == []
    reasons.append('original: incomplete primary_fee80')
    assert len(blocking_failures(states, reasons)) == 1


def test_missing_corrected_audit_or_failed_seasonal_job_stops_continuation():
    states = {'corrected': {'status': 'failed'}, 'seasonal': {'status': 'failed'}}
    reasons = ['corrected: incomplete audit_deployed_bellman',
               'seasonal: completion and cardinality check pending']
    assert len(blocking_failures(states, reasons)) == 2


def test_stale_running_flag_does_not_wait_forever_for_a_dead_process():
    states = {'alive': {'status': 'running', 'pid': 10},
              'dead': {'status': 'running', 'pid': 11},
              'invalid': {'status': 'running', 'pid': -1},
              'complete': {'status': 'complete', 'pid': 12}}
    assert missing_running_processes(states, pid_exists=lambda pid: pid == 10) == ['dead', 'invalid']


def test_pipeline_has_complete_figure_and_audit_order_but_no_release():
    stages = stage_commands()
    names = [name for name, _ in stages]
    assert len(names) == len(set(names))
    assert names.index('merge_raw_inputs') < names.index('input_accounting')
    assert names.index('input_accounting') < names.index('analyze_confirmatory')
    assert names.index('mechanism_m2') < names.index('mechanism_assemble')
    assert names.index('mechanism_assemble') < names.index('verify_round2')
    assert names.index('verify_alignment') < names.index('compile_pending_drafts')
    assert names.index('input_accounting_final') < names.index('assemble_complete_draft_sections')
    assert names.index('assemble_complete_draft_sections') < names.index('compile_pending_drafts')
    assert not any('git' in argument or '--release' == argument
                   for _, arguments in stages for argument in arguments)
    assert not any('submission' in name for name in names)


def test_completed_section_assembly_is_limited_to_generated_includes():
    sections = section_sources()
    assert set(sections) == {'additional_results.tex', 'additional_supplement.tex'}
    assert {name for names in sections.values() for name in names} == {
        'mechanism_results', 'round2_scenarios', 'supplement_results', 'round2_scenario_supplement'}
    text = checked_include_text(sections['additional_results.tex'])
    assert text.count(r'\input{generated/') == 2
    assert 'Document formatting is checked separately.' in text
    assert 'alignment_pending' not in text and 'documentclass' not in text


def test_section_assembly_passes_required_workspace_to_input_audit(tmp_path):
    commands = assembly_audit_commands(tmp_path)
    assert len(commands) == 3
    assert commands[0][1:] == [str(tmp_path / 'experiments' / 'verify_round2_results.py')]
    assert commands[1][1:] == [str(tmp_path / 'experiments' / 'verify_alignment_results.py')]
    assert commands[2][1:] == [str(tmp_path / 'experiments' / 'audit_release_inputs.py'),
                               '--workspace', str(tmp_path)]
    assert not any('--partial' in command for command in commands)


def make_minimal_dependencies(root):
    for relative in ('src/model.py', 'experiments/report.py', 'data/frozen.json',
                     'manuscript/main.tex', 'manuscript/supplement.tex',
                     'manuscript/references.bib', 'manuscript/alignment_pending.tex',
                     'cover_letter.tex', 'highlights.txt', 'highlights.docx'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')


def test_queue_protects_code_and_handwritten_sources_but_allows_generated_outputs(tmp_path):
    make_minimal_dependencies(tmp_path)
    expected = frozen_dependencies(tmp_path)
    generated = tmp_path / 'manuscript/generated/table.tex'
    generated.parent.mkdir()
    generated.write_text('updated generated result')
    check_dependencies(tmp_path, expected)
    (tmp_path / 'src/model.py').write_text('changed')
    with pytest.raises(RuntimeError, match='protected inputs changed'):
        check_dependencies(tmp_path, expected)


def test_queue_detects_new_or_removed_protected_inputs(tmp_path):
    make_minimal_dependencies(tmp_path)
    expected = frozen_dependencies(tmp_path)
    (tmp_path / 'experiments/new_report.py').write_text('new')
    with pytest.raises(RuntimeError, match='new_report.py'):
        check_dependencies(tmp_path, expected)


def test_state_write_retries_transient_windows_sharing_conflicts(tmp_path):
    path = tmp_path / 'progress.json'
    old, new = {'status': 'old'}, {'status': 'new'}
    path.write_text(json.dumps(old), encoding='utf-8')
    calls, delays = [], []

    def busy_then_replace(source, target):
        assert json.loads(target.read_text(encoding='utf-8')) == old
        calls.append((source, target))
        if len(calls) < 3:
            raise PermissionError('simulated Windows reader-sharing conflict')
        os.replace(source, target)

    retries = write_state_atomic(path, new, replace=busy_then_replace, sleep=delays.append)
    assert retries == 2 and delays == [0.25, 0.25]
    assert json.loads(path.read_text(encoding='utf-8')) == new
    assert not path.with_suffix('.json.tmp').exists()


def test_persistent_state_write_error_is_bounded_and_retains_both_records(tmp_path):
    path = tmp_path / 'progress.json'
    old, new = {'status': 'old'}, {'status': 'failed'}
    path.write_text(json.dumps(old), encoding='utf-8')
    calls, delays = [], []

    def always_busy(source, target):
        calls.append((source, target))
        raise PermissionError('persistent denial')

    with pytest.raises(PermissionError, match='persistent denial'):
        write_state_atomic(path, new, attempts=3, replace=always_busy, sleep=delays.append)
    assert len(calls) == 3 and delays == [0.25, 0.25]
    assert json.loads(path.read_text(encoding='utf-8')) == old
    assert json.loads(path.with_suffix('.json.tmp').read_text(encoding='utf-8')) == new


def test_unrelated_state_write_errors_are_not_retried(tmp_path):
    path = tmp_path / 'progress.json'
    delays = []

    def unrelated_error(source, target):
        raise OSError('unrelated filesystem failure')

    with pytest.raises(OSError, match='unrelated filesystem failure'):
        write_state_atomic(path, {'status': 'new'}, replace=unrelated_error,
                           sleep=delays.append)
    assert delays == []


@pytest.mark.parametrize('attempts,delay', [(0, 0.25), (1, -0.1)])
def test_invalid_state_write_retry_configuration_does_not_touch_files(tmp_path, attempts, delay):
    path = tmp_path / 'progress.json'
    with pytest.raises(ValueError, match='state-write attempts'):
        write_state_atomic(path, {}, attempts=attempts, retry_delay=delay)
    assert not list(tmp_path.iterdir())
