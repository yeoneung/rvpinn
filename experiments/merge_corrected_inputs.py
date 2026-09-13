"""Merge finished correction artifacts without altering any frozen run.

The default is a read-only readiness check.  --apply preserves the previous
canonical artifacts before replacing raw inputs and selection metadata.  It
does not regenerate statistics, remove the draft warning, or publish anything.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from audit_release_inputs import expected_online_files
from result_validation import read_aligned_parquet


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_disjoint_frames(original, supplemental, keys, expected_rows):
    """Keep every original row, reject overlap, and merge only disjoint days."""
    if set(original.columns) != set(supplemental.columns):
        raise RuntimeError('seasonal input schemas differ')
    for frame in (original, supplemental):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise RuntimeError('missing or duplicate within-source keys')
    left = pd.MultiIndex.from_frame(original[keys])
    right = pd.MultiIndex.from_frame(supplemental[keys])
    if len(left.intersection(right)):
        raise RuntimeError('seasonal merge would overwrite an existing endpoint')
    combined = pd.concat([original, supplemental[original.columns]], ignore_index=True)
    if len(combined) != expected_rows:
        raise RuntimeError(f'expected {expected_rows} merged rows, found {len(combined)}')
    return combined.sort_values(keys).reset_index(drop=True)


def ready_inputs(original, corrected, seasonal):
    reasons = []
    states = {}
    for name, root, filename in (
            ('original', original, 'alignment_progress.json'),
            ('corrected', corrected, 'alignment_progress.json'),
            ('seasonal', seasonal, 'seasonal_completion_progress.json')):
        path = root / filename
        if not path.exists():
            reasons.append(f'{name}: no progress record')
            continue
        states[name] = json.loads(path.read_text(encoding='utf-8'))
        if states[name].get('status') == 'running':
            reasons.append(f'{name}: runner is still active')
    online_names = [f'primary_fee{fee}' for fee in (20, 40, 80)]
    for fee in (20, 40, 80):
        seasons = ('winter', 'spring', 'summer', 'autumn') if fee == 40 else ('winter',)
        online_names.extend(f'rule_fee{fee}_{season}_weekday' for season in seasons)
    online_names.extend(f'factorial_t{threshold}_f{fee}'
                        for threshold in (250, 350) for fee in (20, 40, 80))
    online_names.append('primary_fee80_tl15')
    for stream in (7301, 7302, 7303):
        online_names.extend(f'scenario_s{stream}_f{fee}_m{count}_tl{budget}'
                            for fee, count, budget in ((40, 2, 5), (40, 8, 5),
                                                       (40, 16, 5), (80, 16, 5),
                                                       (80, 16, 15)))
    for name in online_names:
        if states.get('original', {}).get('jobs', {}).get(name, {}).get('status') != 'complete':
            reasons.append(f'original: incomplete {name}')
    audit_names = ('audit_fixed_policy_gate', 'audit_selected_derivatives',
                   'audit_direct_vs_projection', 'audit_deployed_bellman',
                   'audit_model_loading', 'audit_reflected_forecast')
    for name in audit_names:
        if states.get('corrected', {}).get('jobs', {}).get(name, {}).get('status') != 'complete':
            reasons.append(f'corrected: incomplete {name}')
    if states.get('seasonal', {}).get('status') != 'complete_pending_disjoint_merge_and_release_audit':
        reasons.append('seasonal: completion and cardinality check pending')
    if not (corrected / 'ONLINE_REUSE_PROVENANCE.json').exists():
        reasons.append('corrected: checked online import pending')
    return reasons, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    original, corrected, seasonal = [ROOT / name for name in
                                     ('alignment_run', 'alignment_guard_run',
                                      'alignment_seasonal_run')]
    reasons, states = ready_inputs(original, corrected, seasonal)
    if reasons:
        print(json.dumps({'ready_for_merge': False, 'reasons': reasons}, indent=2))
        if args.apply:
            raise RuntimeError('do not merge while required outputs are incomplete or changing')
        return
    # Runtime model/data/weights must still be the same as the corrected run.
    # Canonical CLI/reporting changes are separate from these frozen inputs.
    for folder in ('src', 'configs', 'data'):
        for source in (corrected / folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                target = ROOT / source.relative_to(corrected)
                if not target.is_file() or sha(source) != sha(target):
                    raise RuntimeError(f'canonical model/data changed: {target}')
    for source in (corrected / 'checkpoints').rglob('*.pt'):
        target = ROOT / source.relative_to(corrected)
        if not target.exists() or sha(source) != sha(target):
            raise RuntimeError(f'checkpoint weights changed: {target}')
    manifest = json.loads((seasonal / 'SEASONAL_COMPLETION_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, digest in manifest['frozen_files'].items():
        if sha(seasonal / relative) != digest:
            raise RuntimeError(f'frozen seasonal dependency changed: {relative}')

    imported = json.loads((corrected / 'ONLINE_REUSE_PROVENANCE.json').read_text(encoding='utf-8'))
    imports = {record['file']: record['sha256'] for record in imported['artifacts']}
    results_relative = Path('experiments') / 'results'
    results = corrected / results_relative
    names = []
    for name in expected_online_files():
        names.extend((name, name.replace('common_daily_', 'common_solves_')))
    names.extend(path.name for pattern in ('learned_daily_*.parquet',
                  'selected_daily_*.parquet', 'round2_reference_daily.parquet',
                  'restart_batches_*.json', 'rule_tuning_*.json')
                 for path in sorted(results.glob(pattern)))
    if len(names) != 93:
        raise RuntimeError(f'expected 81 raw parquet and 12 selection/rule artifacts; got {len(names)}')
    copies = {}
    for name in names:
        source = results / name
        if name.startswith(('common_', 'rule_tuning_')):
            if sha(source) != imports.get(name) or sha(source) != sha(original / results_relative / name):
                raise RuntimeError(f'online reuse integrity failure: {name}')
        if source.suffix == '.parquet':
            read_aligned_parquet(source)
        copies[results_relative / name] = source
    selections = sorted((corrected / 'checkpoints').glob('v3_*/confirmatory/*/seed*/validation_selection.json'))
    if len(selections) != 74:
        raise RuntimeError('expected 74 complete checkpoint selections')
    for selection in selections:
        for name in ('validation_selection.json', 'validation_daily.parquet', 'result.json'):
            source = selection.parent / name
            copies[source.relative_to(corrected)] = source
    merged = {}
    merge_provenance = []
    for kind, keys, rows in (('daily', ['method', 'date'], 600),
                            ('solves', ['method', 'date', 'time'], 34560)):
        name = f'common_{kind}_confirmatory_thr300_fee40_w10_n2.parquet'
        source = results / name
        additional = seasonal / results_relative / name
        if sha(additional) != states['seasonal']['artifact_sha256'].get(name):
            raise RuntimeError(f'seasonal output changed after completion: {name}')
        merged[results_relative / name] = merge_disjoint_frames(
            read_aligned_parquet(source), read_aligned_parquet(additional), keys, rows)
        merge_provenance.append({'target': (results_relative / name).as_posix(),
                                 'base_sha256': sha(source), 'supplemental_sha256': sha(additional),
                                 'supplemental_source': str(additional), 'keys': keys, 'rows': rows})
    plan = {'status': 'ready_for_raw_input_merge_not_release', 'submission_ready': False,
            'sources': [{'source': str(source), 'target': relative.as_posix(), 'sha256': sha(source)}
                        for relative, source in sorted(copies.items())],
            'disjoint_seasonal_merges': merge_provenance}
    if not args.apply:
        print(json.dumps({'ready_for_merge': True, 'replacement_files': len(copies),
                          'disjoint_merges': merge_provenance}, indent=2))
        return
    if not (ROOT / 'manuscript' / 'alignment_pending.tex').exists():
        raise RuntimeError('the manuscript must remain visibly marked as pending during reconciliation')
    record = ROOT / 'CORRECTED_INPUT_MERGE.json'
    if record.exists():
        raise RuntimeError('a merge record already exists; inspect it before any repeated replacement')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = ROOT / 'alignment_archive' / f'pre_corrected_merge_{stamp}'
    archive.mkdir(parents=True, exist_ok=False)
    # Preserve all existing result tables/figural diagnostics, including outputs
    # not overwritten here. Subsequent analysis must regenerate those explicitly.
    shutil.copytree(ROOT / results_relative, archive / results_relative)
    for relative in copies:
        if relative.parts[0] == 'checkpoints' and (ROOT / relative).exists():
            saved = archive / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, saved)
    plan['backup_sha256'] = {}
    for backup in archive.rglob('*'):
        if backup.is_file():
            relative = backup.relative_to(archive)
            digest = sha(backup)
            if digest != sha(ROOT / relative):
                raise RuntimeError(f'backup integrity failure before any replacement: {relative}')
            plan['backup_sha256'][relative.as_posix()] = digest
    plan['backup_directory'] = str(archive)
    plan['status'] = 'raw_input_merge_in_progress_not_release'
    record.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    try:
        for relative, source in copies.items():
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in merged:
                merged[relative].to_parquet(target, index=False)
            else:
                shutil.copy2(source, target)
                if sha(source) != sha(target):
                    raise RuntimeError(f'copy integrity failure: {relative}')
        plan['output_sha256'] = {relative.as_posix(): sha(ROOT / relative) for relative in copies}
        plan['status'] = 'raw_inputs_merged_audits_analysis_and_manuscript_pending'
    except BaseException as error:
        plan.update(status='merge_failed_restore_or_resume_from_retained_backup', error=str(error))
        raise
    finally:
        record.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(plan['status'])


if __name__ == '__main__':
    main()
