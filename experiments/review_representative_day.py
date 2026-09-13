"""Source-checked representative-day replay without mutating live runs."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
import make_round2_mechanism_figure as mechanism
from verify_round2_results import verify_mechanism


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    online, learned, output = [path.resolve() for path in (args.online, args.learned, args.output)]
    if output.exists():
        raise FileExistsError('representative-day review requires a new output directory')
    if any(output == root or root in output.parents for root in (online, learned)):
        raise RuntimeError('review output must not be within a frozen run')
    state = json.loads((online / 'alignment_progress.json').read_text(encoding='utf-8'))
    for name in ('primary_fee40', 'scenario_s7301_f40_m16_tl5'):
        if state['jobs'].get(name, {}).get('status') != 'complete':
            raise RuntimeError(f'review waits for completed source: {name}')
    guard_state = json.loads((learned / 'alignment_progress.json').read_text(encoding='utf-8'))
    for name in ('reference', 'evaluate_v3_winter_fee40'):
        if guard_state['jobs'].get(name, {}).get('status') != 'complete':
            raise RuntimeError(f'review waits for completed corrected policy: {name}')
    for folder in ('src', 'configs', 'data'):
        for source in (ROOT / folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                other = learned / source.relative_to(ROOT)
                if not other.exists() or sha(source) != sha(other):
                    raise RuntimeError('canonical replay model differs from the corrected run')
    results, figures, generated = [output / name for name in ('results', 'figures', 'generated')]
    for path in (results, figures, generated):
        path.mkdir(parents=True)
    sources = [online / 'experiments' / 'results' / name for name in (
        'common_daily_confirmatory_thr300_fee40_w10_n2.parquet',
        'common_solves_confirmatory_thr300_fee40_w10_n2.parquet',
        'common_daily_confirmatory_thr300_fee40_w10_n16_r2_stream7301_pool16_tl5.parquet',
        'common_solves_confirmatory_thr300_fee40_w10_n16_r2_stream7301_pool16_tl5.parquet')]
    sources += [learned / 'experiments' / 'results' / name for name in (
        'learned_daily_v3_winter_fee40.parquet', 'restart_batches_v3_winter_fee40.json',
        'round2_reference_daily.parquet')]
    inputs = []
    for source in sources:
        digest = sha(source)
        target = results / source.name
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise RuntimeError('representative-day input changed during snapshot')
        inputs.append({'source': str(source), 'snapshot': target.relative_to(output).as_posix(),
                       'sha256': digest})
    mechanism.ROOT, mechanism.RESULTS = learned, results
    mechanism.FIGURE = figures / 'fig_round2_mechanism.pdf'
    mechanism.TEX = generated / 'round2_mechanism.tex'
    for key in mechanism.CONTROLLERS:
        mechanism._run_controller(key)
    mechanism._assemble()
    verify_mechanism(results)
    scripts = [Path(__file__), Path(mechanism.__file__), ROOT / 'experiments' / 'verify_round2_results.py']
    (output / 'reporting_code').mkdir()
    for source in scripts:
        shutil.copy2(source, output / 'reporting_code' / source.name)
    manifest = {
        'status': 'representative_day_review_passed_not_release',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_provenance': inputs,
        'script_sha256': {path.name: sha(path) for path in scripts},
        'policy_scope': 'first selected winter batch, unchanged reference, recorded primary M2 and M16 stream 7301 actions',
        'submission_ready': False}
    (output / 'REPRESENTATIVE_DAY_REVIEW.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(manifest['status'])


if __name__ == '__main__':
    main()
