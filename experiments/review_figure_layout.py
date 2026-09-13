"""Re-render checked numerical summaries without repeating or changing analysis."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import make_v3_figures as plotting


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--factorial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary, factorial, output = [path.resolve() for path in
                                 (args.primary, args.factorial, args.output)]
    if output.exists():
        raise FileExistsError('figure-layout review requires a new directory')
    if any(output == root or root in output.parents for root in (primary, factorial)):
        raise RuntimeError('preserve the checked input snapshots')
    manifests = [primary / 'PRIMARY_REVIEW_PROVENANCE.json',
                 factorial / 'FACTORIAL_REVIEW_PROVENANCE.json']
    manifest_hashes = {str(path): sha(path) for path in manifests}
    for path in manifests:
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') not in ('primary_review_only_not_release', 'factorial_review_passed_not_release'):
            raise RuntimeError('figure input is not a completed review snapshot')
    output.mkdir(parents=True)
    source_paths = (primary / 'inputs' / 'confirmatory_method_summary.csv',
                    factorial / 'experiments/results/sensitivity_cells.csv')
    source_hashes = {str(path): sha(path) for path in source_paths}
    for path in source_paths:
        shutil.copy2(path, output / path.name)
        if sha(output / path.name) != source_hashes[str(path)]:
            raise RuntimeError('numerical figure input changed during copy')
    plotting.RESULTS = plotting.FIGURES = output
    plotting.make_cost_latency()
    plotting.make_factorial()
    scripts = [Path(__file__), Path(plotting.__file__)]
    for path in scripts:
        shutil.copy2(path, output / path.name)
    record = {
        'status': 'checked_numeric_inputs_rerendered_not_release', 'submission_ready': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_review_manifest_sha256': manifest_hashes,
        'input_csv_sha256': source_hashes,
        'plotting_code_sha256': {path.name: sha(path) for path in scripts},
        'figure_sha256': {path.name: sha(path) for path in output.iterdir()
                          if path.suffix in ('.pdf', '.png')},
        'changes': ['larger text at manuscript scale', 'unclipped concise colorbar label',
                    'actual positive measured latency without a plotting floor',
                    'common logarithmic latency axis across fee panels'],
    }
    (output / 'FIGURE_LAYOUT_REVIEW.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(record['status'])


if __name__ == '__main__':
    main()
