"""Check completed factorial inputs without importing unfinished experiments."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
import analyze_seasonal_sensitivity as reporting
import audit_release_inputs as audit
import make_v3_figures as plotting
import result_validation
from result_validation import read_aligned_parquet
from src.deployed_policy import ACTION_SEARCH_VERSION
from src.deployment_numerics import DEPLOYMENT_VERSION


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_factorial_statistics(cells, results):
    """Recompute descriptive means directly from balanced raw records."""
    for row in cells.itertuples():
        threshold, fee, width = map(int, (row.threshold_kw, row.fee_per_hour,
                                         row.smoothing_width_kw))
        comp = pd.read_parquet(results / f'common_daily_confirmatory_thr{threshold}_fee{fee}_w10_n2.parquet')
        comp = comp[comp['regime'] == 'winter_weekday']
        if threshold == 300 and width == 10:
            observed = pd.read_parquet(results / f'learned_daily_v3_winter_fee{fee}.parquet')
            observed = observed[observed['selected_by_batch']]
        else:
            observed = pd.read_parquet(results / f'selected_daily_v3_sens_t{threshold}_f{fee}_w{width}.parquet')
        mean = float(observed['common_cost'].mean())
        det = float(comp.loc[comp['method'] == 'deterministic_exact_band_miqp', 'common_cost'].mean())
        zero = float(comp.loc[comp['method'] == 'no_storage', 'common_cost'].mean())
        expected = {'mean_learned_cost': mean, 'mean_deterministic_cost': det,
                    'mean_no_storage_cost': zero, 'learned_minus_deterministic': mean - det,
                    'learned_minus_no_storage': mean - zero,
                    'mean_terminal_penalty': float(observed['terminal_penalty'].mean()),
                    'mean_smooth_minus_hard_cost': float(observed['objective'].mean() - mean),
                    'mean_exact_threshold_hours': float(observed['threshold_exact_hours'].mean())}
        expected.update({f'mean_near_{radius}kw_hours': float(observed[f'threshold_near_{radius}kw_hours'].mean())
                         for radius in (5, 10, 20)})
        if any(not np.isclose(getattr(row, name), value, atol=1e-9, rtol=0)
               for name, value in expected.items()):
            raise RuntimeError(f'factorial descriptive statistics do not reconcile: {threshold}/{fee}/{width}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--online', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    online, learned, output = [path.resolve() for path in
                               (args.online, args.learned, args.output)]
    if output.exists():
        raise FileExistsError('factorial review requires a new output directory')
    if any(output == root or root in output.parents for root in (online, learned)):
        raise RuntimeError('review must not write into a frozen run')
    state = json.loads((online / 'alignment_progress.json').read_text(encoding='utf-8'))
    guard = json.loads((learned / 'alignment_progress.json').read_text(encoding='utf-8'))
    for threshold in (250, 300, 350):
        for fee in (20, 40, 80):
            name = f'primary_fee{fee}' if threshold == 300 else f'factorial_t{threshold}_f{fee}'
            if state['jobs'].get(name, {}).get('status') != 'complete':
                raise RuntimeError(f'factorial review waits for complete source job: {name}')
    for fee in (20, 40, 80):
        seasons = ('winter', 'spring', 'summer', 'autumn') if fee == 40 else ('winter',)
        for season in seasons:
            if state['jobs'].get(f'rule_fee{fee}_{season}_weekday', {}).get('status') != 'complete':
                raise RuntimeError('primary comparator file is still changing')
    if any(entry.get('status') == 'running' and name.startswith(('select_', 'evaluate_'))
           for name, entry in guard['jobs'].items()):
        raise RuntimeError('corrected learned selections or evaluations are changing')
    dependencies = {}
    for folder in ('data', 'configs'):
        for source in (ROOT / folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                relative = source.relative_to(ROOT)
                digest = sha(source)
                if sha(learned / relative) != digest or sha(online / relative) != digest:
                    raise RuntimeError(f'factorial model/data mismatch: {relative}')
                dependencies[relative.as_posix()] = digest
    results, generated, figures = [output / name for name in
                                  ('experiments/results', 'generated', 'figures')]
    for path in (results, generated, figures):
        path.mkdir(parents=True)
    provenance = []

    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise RuntimeError('factorial input changed during snapshot')
        provenance.append({'source': str(source), 'snapshot': relative.as_posix(), 'sha256': digest})

    relative_results = Path('experiments/results')
    params, _, _ = audit.regime_bundle('confirmatory', ['winter_weekday'])
    test_list = audit.load_region_days('confirmatory', 'test')
    validation_list = [day for day in audit.load_region_days('confirmatory', 'val')
                       if day['regime'] == 'winter_weekday'][:30]
    expected_dates = {str(day['date']) for day in
                      [d for d in test_list if d['regime'] == 'winter_weekday'][:30]}
    validation_dates = {str(day['date']) for day in validation_list}
    test_days = {str(day['date']): day for day in test_list}
    replayed, selections, batches, weight_hashes = 0, 0, 0, {}
    for threshold in (250, 300, 350):
        for fee in (20, 40, 80):
            name = f'common_daily_confirmatory_thr{threshold}_fee{fee}_w10_n2.parquet'
            for filename in (name, name.replace('common_daily_', 'common_solves_')):
                copy(online / relative_results / filename, relative_results / filename)
            daily = read_aligned_parquet(results / name)
            daily = daily[daily['regime'] == 'winter_weekday']
            audit.check_online_parameters(daily, name)
            audit.check_daily_accounting(daily, params)
            methods = set(audit.ONLINE) if threshold == 300 else set(audit.ONLINE[:2])
            if set(daily['method']) != methods:
                raise RuntimeError('factorial comparator family mismatch')
            for _, group in daily.groupby('method'):
                if len(group) != 30 or set(group['date']) != expected_dates:
                    raise RuntimeError('factorial comparator dates differ from the protocol')
            p = params['winter_weekday']
            for row in daily[daily['method'] == 'no_storage'].itertuples():
                day = test_days[str(row.date)]
                net, price = np.asarray(day['N']), np.asarray(day['C'])
                bill = float(np.sum(price * np.maximum(net, 0.)
                                    - p.alpha_s * price * np.maximum(-net, 0.)))
                cost = bill + fee * np.count_nonzero(net > threshold) + p.lam_T * (p.s0 - p.s_tar) ** 2
                if not np.isclose(row.common_cost, cost, atol=1e-7, rtol=0):
                    raise RuntimeError('factorial zero-action cost does not reproduce from observations')
            solves = read_aligned_parquet(results / name.replace('common_daily_', 'common_solves_'))
            solves = solves[solves['regime'] == 'winter_weekday']
            expected_rows = 30 * 96 * (3 if threshold == 300 else 1)
            if len(solves) != expected_rows:
                raise RuntimeError('incomplete factorial solver matrix')
            audit.check_solve_records(solves, test_days)
            replayed += audit.replay_solver_costs(solves, daily, test_days, params)
            for width in (5, 10, 20):
                anchor = threshold == 300 and width == 10
                tag = f'v3_winter_fee{fee}' if anchor else f'v3_sens_t{threshold}_f{fee}_w{width}'
                n_seeds = (25 if fee == 40 else 5) if anchor else 1
                if guard['jobs'].get(f'evaluate_{tag}', {}).get('status') != 'complete':
                    raise RuntimeError('factorial policy evaluation is not complete')
                filename = f'{"learned" if anchor else "selected"}_daily_{tag}.parquet'
                copy(learned / relative_results / filename, relative_results / filename)
                frame = read_aligned_parquet(results / filename)
                audit.check_tariff_metadata(frame, threshold, fee, width)
                audit.check_daily_accounting(frame, params)
                if not frame['exact_action_change_events'].eq(0).all():
                    raise RuntimeError('factorial held actions changed at the safety gate')
                trained = frame[~frame['is_fallback']] if anchor else frame
                if set(trained['seed']) != set(range(n_seeds)):
                    raise RuntimeError('factorial training seed set changed')
                for _, group in trained.groupby('seed'):
                    if len(group) != 30 or set(group['date']) != expected_dates:
                        raise RuntimeError('incomplete factorial seed/date matrix')
                for seed in range(n_seeds):
                    relative = Path('checkpoints') / tag / 'confirmatory' / 'winter_weekday' / f'seed{seed}'
                    for filename in ('validation_selection.json', 'validation_daily.parquet', 'result.json'):
                        copy(learned / relative / filename, relative / filename)
                    chosen = json.loads((output / relative / 'validation_selection.json').read_text(encoding='utf-8'))
                    if (chosen['deployment_version'] != DEPLOYMENT_VERSION
                            or chosen['action_search_version'] != ACTION_SEARCH_VERSION
                            or chosen['partition'] != 'validation'
                            or chosen['region'] != 'confirmatory'
                            or chosen['regime'] != 'winter_weekday'
                            or chosen['n_days'] != 30
                            or chosen['margin_eur_per_day'] != .10):
                        raise RuntimeError('factorial checkpoint selection protocol changed')
                    val = read_aligned_parquet(output / relative / 'validation_daily.parquet')
                    audit.check_daily_accounting(val.assign(fee_per_hour=fee), params)
                    if (not val['regime'].eq('winter_weekday').all()
                            or not val['exact_action_change_events'].eq(0).all()):
                        raise RuntimeError('factorial validation regime or held action changed')
                    if set(val['iteration']) != {0, 1, 2, 3}:
                        raise RuntimeError('factorial checkpoint set is incomplete')
                    incumbent, incumbent_cost = 0, None
                    for iteration, group in val.groupby('iteration', sort=True):
                        if len(group) != 30 or set(group['date']) != validation_dates:
                            raise RuntimeError('factorial validation dates changed')
                        mean = float(group['common_cost'].mean())
                        if incumbent_cost is None or incumbent_cost - mean >= .10:
                            incumbent, incumbent_cost = int(iteration), mean
                        weight = relative / f'iter{int(iteration):02d}.pt'
                        if sha(learned / weight) != sha(ROOT / weight):
                            raise RuntimeError('factorial checkpoint weights changed')
                        weight_hashes[weight.as_posix()] = sha(learned / weight)
                    recorded = json.loads((output / relative / 'result.json').read_text(encoding='utf-8'))
                    if incumbent != chosen['selected_iteration'] or incumbent != recorded['selected_iteration']:
                        raise RuntimeError('factorial validation winner does not reproduce')
                    selections += 1
                if anchor:
                    filename = f'restart_batches_{tag}.json'
                    copy(learned / relative_results / filename, relative_results / filename)
                    batches += audit.check_batch_selection(
                        output, frame, tag, 'winter_weekday', fee, n_seeds,
                        validation_list, params['winter_weekday'])
    if (replayed, selections, batches, len(weight_hashes)) != (450, 59, 7, 236):
        raise RuntimeError('factorial review cardinalities do not reconcile')
    reporting.RESULTS, reporting.GENERATED = results, generated
    cells = reporting.factorial_cells()
    check_factorial_statistics(cells, results)
    summary = reporting.write_factorial(cells)
    plotting.RESULTS, plotting.FIGURES = results, figures
    plotting.make_factorial()
    scripts = [Path(__file__), Path(reporting.__file__), Path(audit.__file__),
               Path(plotting.__file__), Path(result_validation.__file__)]
    (output / 'reporting_code').mkdir()
    for path in scripts:
        shutil.copy2(path, output / 'reporting_code' / path.name)
    manifest = {
        'status': 'factorial_review_passed_not_release', 'submission_ready': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'raw_input_sha256': provenance, 'data_config_sha256': dependencies,
        'checkpoint_sha256': weight_hashes,
        'reporting_script_sha256': {path.name: sha(path) for path in scripts},
        'checks': {'independently_replayed_solver_days': replayed,
                   'independently_checked_zero_action_days': 270,
                   'independently_recomputed_factorial_cells': len(cells),
                   'validation_selections': selections, 'batch_selections': batches},
        'summary': summary}
    (output / 'FACTORIAL_REVIEW_PROVENANCE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'checks': manifest['checks'], **summary}, indent=2))


if __name__ == '__main__':
    main()
