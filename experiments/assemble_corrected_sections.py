"""Assemble verified numerical sections in a guarded analysis workspace.

Only two generated include files are replaced. Handwritten interpretation,
the workspace guard, submission materials and public repositories are untouched.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def section_sources():
    return {
        'additional_results.tex': ('mechanism_results', 'round2_scenarios'),
        'additional_supplement.tex': ('supplement_results', 'round2_scenario_supplement'),
    }


def checked_include_text(names):
    return ('% Numerical inputs passed the local artifact checks.\n'
            '% Document formatting is checked separately.\n'
            + ''.join('\\input{generated/' + name + '}\n' for name in names))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assembly_audit_commands(root):
    commands = [[sys.executable, str(root / 'experiments' / script)]
                for script in ('verify_round2_results.py', 'verify_alignment_results.py')]
    commands.append([sys.executable, str(root / 'experiments' / 'audit_release_inputs.py'),
                     '--workspace', str(root)])
    return commands


def main():
    manuscript = ROOT / 'manuscript'
    generated = manuscript / 'generated'
    record_path = ROOT / 'CORRECTED_SECTION_ASSEMBLY.json'
    if record_path.exists():
        raise RuntimeError('a section-assembly record already exists; inspect it before repeating')
    if not (manuscript / 'alignment_pending.tex').exists():
        raise RuntimeError('section assembly must retain the visible working-draft banner')
    merge_path = ROOT / 'CORRECTED_INPUT_MERGE.json'
    merge = json.loads(merge_path.read_text(encoding='utf-8'))
    if merge['status'] != 'raw_inputs_merged_audits_analysis_and_manuscript_pending':
        raise RuntimeError('checked corrected-input merge is not complete')
    for relative, digest in merge['output_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f'merged raw input changed before section assembly: {relative}')
    for command in assembly_audit_commands(ROOT):
        subprocess.run(command, cwd=ROOT, check=True)
    sources = section_sources()
    dependency_names = {name + '.tex' for names in sources.values() for name in names}
    dependency_names.update(('factorial_results.tex', 'factorial_supplement.tex'))
    newest_raw = max((ROOT / relative).stat().st_mtime for relative in merge['output_sha256'])
    for name in dependency_names:
        path = generated / name
        if not path.exists() or path.stat().st_mtime < newest_raw:
            raise RuntimeError(f'corrected result text is absent or stale: {name}')
    original = {name: (generated / name).read_text(encoding='utf-8') for name in sources}
    record = {
        'status': 'section_assembly_in_progress_not_release', 'submission_ready': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'merge_sha256': sha(merge_path),
        'previous_generated_includes': original,
        'checked_result_text_sha256': {name: sha(generated / name) for name in sorted(dependency_names)},
    }
    with record_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    try:
        for name, includes in sources.items():
            (generated / name).write_text(checked_include_text(includes), encoding='utf-8')
        record.update(status='full_result_sections_assembled_pending_manual_review',
                      generated_include_sha256={name: sha(generated / name) for name in sources})
    except BaseException as error:
        record.update(status='section_assembly_failed_not_release', error=str(error))
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(record['status'])


if __name__ == '__main__':
    main()
