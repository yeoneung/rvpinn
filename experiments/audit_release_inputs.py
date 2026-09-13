"""Independent, read-only coverage and accounting audit of corrected artifacts.

Partial mode checks available records but never reports release readiness.
This supplements, rather than replaces, the numerical/theory-specific audits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from src.evaluation import load_region_days
from src.exp_common import regime_bundle
from src.deployment_numerics import DEPLOYMENT_VERSION
from src.deployed_policy import ACTION_SEARCH_VERSION
from result_validation import read_aligned_parquet

SEASONS = ('winter', 'spring', 'summer', 'autumn')
ONLINE = ('no_storage', 'deterministic_exact_band_miqp',
          'stochastic_two_stage_exact_band_miqp', 'convex_envelope_mpc',
          'validation_tuned_price_rule')


def selection_tariff(tag):
    central = re.fullmatch(r'v3_(?:winter|spring|summer|autumn)_fee(\d+)', tag)
    factorial = re.fullmatch(r'v3_sens_t(\d+)_f(\d+)_w(\d+)', tag)
    if central:
        return 300, int(central.group(1)), 10
    if factorial:
        return tuple(map(int, factorial.groups()))
    raise RuntimeError(f'unrecognized validation selection tag: {tag}')


def expected_online_files():
    specs = {}
    for fee in (20, 40, 80):
        regs = SEASONS if fee == 40 else ('winter',)
        specs[f'common_daily_confirmatory_thr300_fee{fee}_w10_n2.parquet'] = (regs, ONLINE)
    for threshold in (250, 350):
        for fee in (20, 40, 80):
            specs[f'common_daily_confirmatory_thr{threshold}_fee{fee}_w10_n2.parquet'] = (
                ('winter',), ONLINE[:2])
    specs['common_daily_confirmatory_thr300_fee80_w10_n2_tl15.parquet'] = (
        ('winter',), ONLINE[1:3])
    for stream in (7301, 7302, 7303):
        for fee, count, budget in ((40, 2, 5), (40, 8, 5), (40, 16, 5), (80, 16, 5), (80, 16, 15)):
            name = f'common_daily_confirmatory_thr300_fee{fee}_w10_n{count}_r2_stream{stream}_pool16_tl{budget}.parquet'
            specs[name] = (('winter',), (ONLINE[2],))
    return specs


def check_tariff_metadata(frame, threshold, fee, width):
    for column, value in (('threshold_kw', threshold), ('fee_per_hour', fee),
                          ('smoothing_width_kw', width)):
        if column not in frame or not frame[column].eq(float(value)).all():
            raise RuntimeError(f'wrong experiment metadata: {column}')
    if 'region' in frame and not frame['region'].eq('confirmatory').all():
        raise RuntimeError('wrong experiment region')


def check_online_parameters(frame, filename):
    match = re.fullmatch(r'common_daily_confirmatory_thr(\d+)_fee(\d+)_w(\d+)_n(\d+)(.*?)\.parquet', filename)
    if not match:
        raise RuntimeError('unrecognized expected online filename')
    threshold, fee, width, count = map(int, match.groups()[:4])
    suffix = match.group(5)
    check_tariff_metadata(frame, threshold, fee, width)
    if not frame['control_dt_min'].eq(15.).all():
        raise RuntimeError('wrong online update interval')
    budget = 15. if suffix.endswith('tl15') else 5.
    controlled = frame[frame['method'].isin(ONLINE[1:4])]
    if not controlled['solver_time_limit_s'].eq(budget).all():
        raise RuntimeError('wrong online solver time budget')
    stochastic = frame[frame['method'] == ONLINE[2]]
    if not stochastic['n_scenarios'].eq(count).all():
        raise RuntimeError('wrong scenario count')
    if 'r2_stream' in suffix:
        stream = int(re.search(r'r2_stream(\d+)_pool16', suffix).group(1))
        if not stochastic['scenario_pool_size'].eq(16).all():
            raise RuntimeError('wrong nested scenario pool')
    else:
        stream = 4101
        if not stochastic['scenario_pool_size'].isna().all():
            raise RuntimeError('primary stochastic comparator must retain its original stream convention')
    if not stochastic['scenario_stream_seed'].eq(stream).all():
        raise RuntimeError('wrong scenario stream seed')


def check_batch_selection(workspace, frame, tag, regime, fee, n_seeds, val_days, p):
    """Reconstruct the zero-action incumbent and every five-restart decision."""
    path = workspace / 'experiments' / 'results' / f'restart_batches_{tag}.json'
    audit = json.loads(path.read_text(encoding='utf-8'))
    if (audit.get('action_search_version') != ACTION_SEARCH_VERSION
            or audit.get('deployment_version') != DEPLOYMENT_VERSION
            or audit['restart_margin_eur_per_day'] != 0.10
            or audit['tag'] != tag or audit['regime'] != regime):
        raise RuntimeError('batch selection protocol mismatch')
    values = []
    for day in val_days:
        net, price = np.asarray(day['N']), np.asarray(day['C'])
        values.append(float(np.sum(price * np.maximum(net, 0.)
                                  - p.alpha_s * price * np.maximum(-net, 0.))
                            + fee * np.count_nonzero(net > 300.)
                            + p.lam_T * (p.s0 - p.s_tar) ** 2))
    fallback = float(np.mean(values))
    if not np.isclose(fallback, audit['fallback_validation_mean_common_cost'], atol=1e-7, rtol=0):
        raise RuntimeError('zero-action validation incumbent does not reproduce')
    restarts = audit['restarts']
    if [row['seed'] for row in restarts] != list(range(n_seeds)):
        raise RuntimeError('restart selection order/set changed')
    by_seed = {}
    for row in restarts:
        seed = row['seed']
        run_dir = workspace / 'checkpoints' / tag / 'confirmatory' / regime / f'seed{seed}'
        selected = json.loads((run_dir / 'validation_selection.json').read_text(encoding='utf-8'))
        if row['selected_iteration'] != selected['selected_iteration'] or row['batch'] != seed // 5:
            raise RuntimeError('restart/checkpoint selection mismatch')
        daily = read_aligned_parquet(run_dir / 'validation_daily.parquet')
        mean = float(daily.loc[daily['iteration'] == selected['selected_iteration'], 'common_cost'].mean())
        if not np.isclose(mean, row['validation_mean_common_cost'], atol=1e-7, rtol=0):
            raise RuntimeError('restart validation cost does not reproduce')
        observed = frame[~frame['is_fallback'] & frame['seed'].eq(seed)]
        if not np.allclose(observed['validation_mean_common_cost'], mean, atol=1e-7, rtol=0):
            raise RuntimeError('test records do not identify the selected validation value')
        by_seed[seed] = float(row['validation_mean_common_cost'])
    selections = audit['batch_selections']
    if [row['batch'] for row in selections] != list(range(n_seeds // 5)):
        raise RuntimeError('batch selection set/order changed')
    for entry in selections:
        batch = entry['batch']
        winner, incumbent = None, fallback
        for seed in range(5 * batch, 5 * batch + 5):
            if incumbent - by_seed[seed] >= 0.10:
                winner, incumbent = seed, by_seed[seed]
        kind = 'zero_action_fallback' if winner is None else 'trained_policy'
        if entry['selected_seed'] != winner or entry['selected_kind'] != kind:
            raise RuntimeError('batch winner does not reproduce from validation')
        if not np.isclose(entry['selected_validation_mean_common_cost'], incumbent, atol=1e-7, rtol=0):
            raise RuntimeError('batch incumbent cost mismatch')
        selected = frame[frame['selected_by_batch'] & frame['batch'].eq(batch)]
        expected_seed = -(batch + 1) if winner is None else winner
        if not selected['seed'].eq(expected_seed).all() or not selected['is_fallback'].eq(winner is None).all():
            raise RuntimeError('test rows do not represent the validation-selected batch policy')
    return len(selections)


def check_daily_accounting(frame, params):
    fields = ['common_cost', 'total_cost', 'bill', 'capacity_fee', 'degradation_cost',
              'terminal_penalty', 'throughput_kwh', 'exceed_hours', 'terminal_soc',
              'terminal_soc_dev', 'efc', 'latency_mean_ms', 'latency_p95_ms']
    if not np.isfinite(frame[fields]).all().all():
        raise RuntimeError('nonfinite economic or physical record')

    def close(actual, expected, name):
        if not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise RuntimeError(f'inconsistent {name}')

    close(frame['common_cost'], frame[['bill', 'capacity_fee', 'degradation_cost', 'terminal_penalty']].sum(axis=1), 'component sum')
    close(frame['total_cost'], frame['common_cost'], 'total-cost alias')
    close(frame['capacity_fee'], frame['fee_per_hour'] * frame['exceed_hours'], 'threshold-fee units')
    close(frame['exceed_hours'] * 4., np.round(frame['exceed_hours'] * 4.), 'held-action exceedance duration')
    if not frame['soc_violations'].eq(0).all():
        raise RuntimeError('SoC violations')
    for regime, sub in frame.groupby('regime'):
        p = params[regime]
        close(sub['degradation_cost'], p.lam1 * sub['throughput_kwh'], 'degradation units')
        close(sub['efc'], sub['throughput_kwh'] / (2. * p.E_max), 'cycle units')
        close(sub['terminal_soc_dev'], (sub['terminal_soc'] - p.s_tar).abs(), 'terminal deviation')
        close(sub['terminal_penalty'], p.lam_T * (sub['terminal_soc'] - p.s_tar) ** 2, 'terminal cost')
        if not sub['terminal_soc'].between(p.s_min - 1e-10, p.s_max + 1e-10).all():
            raise RuntimeError('infeasible terminal SoC')
    for field in ('exceed_hours', 'threshold_exact_hours', 'threshold_near_5kw_hours',
                  'threshold_near_10kw_hours', 'threshold_near_20kw_hours'):
        if not frame[field].between(-1e-9, 24. + 1e-9).all():
            raise RuntimeError(f'invalid duration: {field}')
    if not ((frame['threshold_near_5kw_hours'] <= frame['threshold_near_10kw_hours'] + 1e-9)
            & (frame['threshold_near_10kw_hours'] <= frame['threshold_near_20kw_hours'] + 1e-9)).all():
        raise RuntimeError('nonnested threshold neighborhoods')


def check_solve_records(frame, test_days):
    if frame.duplicated(['method', 'date', 'time']).any():
        raise RuntimeError('duplicate solver call')
    for _, group in frame.groupby(['method', 'date']):
        if len(group) != 96 or not np.allclose(sorted(group['time']), np.arange(96) / 4., atol=1e-12, rtol=0):
            raise RuntimeError('incomplete within-day solver clock')
    if not frame['has_solution'].isin([True, False]).all() or not frame['solve_s'].ge(0.).all():
        raise RuntimeError('invalid solver status/time record')
    valid = frame[frame['has_solution']].copy()
    first_fields = ['first_charge_kw', 'first_discharge_kw', 'first_action_raw',
                    'first_action_held', 'first_action_recovery_kw']
    if not np.isfinite(valid[first_fields]).all().all():
        raise RuntimeError('nonfinite first-stage incumbent record')
    if not np.allclose(valid['first_discharge_kw'] - valid['first_charge_kw'],
                       valid['first_action_raw'], atol=1e-9, rtol=0):
        raise RuntimeError('first-stage flow/action mismatch')
    if (np.minimum(valid['first_charge_kw'], valid['first_discharge_kw']) > 1e-6).any():
        raise RuntimeError('simultaneous first-stage battery flows')
    if valid['first_action_recovery_kw'].abs().gt(1e-5 + 1e-8).any():
        raise RuntimeError('first-action recovery exceeds declared bound')
    if not np.allclose(valid['first_action_held'] - valid['first_action_raw'],
                       valid['first_action_recovery_kw'], atol=1e-10, rtol=0):
        raise RuntimeError('first-action recovery metadata disagree')
    valid = valid[valid['method'].isin(ONLINE[1:3])]
    for row in valid.itertuples():
        net = float(test_days[str(row.date)]['N'][int(row.time)])
        active = net - row.first_action_held > row.threshold_kw
        if active != bool(row.first_band_active):
            raise RuntimeError(f'first-stage billed band mismatch: {row.method}/{row.date}/{row.time}')


def replay_solver_costs(solves, daily, test_days, params):
    """Independently reconstruct economic bills from 96 logged held actions.

    This does not call the evaluator's cost, safety, or dynamics functions.
    Zero action is used for a documented no-incumbent fallback. A nonzero
    post-call safety change is rejected: the log would then be insufficient
    to reconstruct the action actually billed without an additional trace.
    """
    replayed = 0
    for (method, date), records in solves.groupby(['method', 'date']):
        selected = daily[(daily['method'] == method) & (daily['date'] == date)]
        if len(selected) != 1:
            raise RuntimeError(f'replay needs one daily record: {method}/{date}')
        row = selected.iloc[0]
        if row['exact_action_change_events'] != 0:
            raise RuntimeError('post-call safety change prevents logged-action replay')
        records = records.sort_values('time')
        p, day = params[row['regime']], test_days[str(date)]
        actions = np.where(records['has_solution'], records['first_action_held'], 0.)
        if len(actions) != 96 or not np.isfinite(actions).all():
            raise RuntimeError('replay requires 96 finite deployed actions')
        times = records['time'].to_numpy()
        net = np.asarray(day['N'], dtype=float)[times.astype(int)]
        price = np.asarray(day['C'], dtype=float)[times.astype(int)]
        grid = net - actions
        dt = float(p.dt_ctrl)
        bill = float(dt * np.sum(price * np.maximum(grid, 0.)
                                  - p.alpha_s * price * np.maximum(-grid, 0.)))
        exceed = float(dt * np.count_nonzero(grid > row['threshold_kw']))
        band = float(row['fee_per_hour'] * exceed)
        throughput = float(dt * np.sum(np.abs(actions)))
        degradation = float(p.lam1 * throughput)
        soc = float(p.s0)
        for action in actions:
            lo = -min(p.a_c * max(min((p.s_max - soc) / p.delta_s, 1.), 0.),
                      max(p.E_max * (p.s_max - soc) / (p.eta_c * dt), 0.))
            hi = min(p.a_d * max(min((soc - p.s_min) / p.delta_s, 1.), 0.),
                     max(p.eta_d * p.E_max * (soc - p.s_min) / dt, 0.))
            if not lo - 1e-8 <= action <= hi + 1e-8:
                raise RuntimeError(f'logged action outside replayed safe set: {method}/{date}')
            soc += dt * (p.eta_c * max(-action, 0.) / p.E_max
                         - max(action, 0.) / (p.eta_d * p.E_max))
        terminal = float(p.lam_T * (soc - p.s_tar) ** 2)
        expected = {'bill': bill, 'capacity_fee': band,
                    'degradation_cost': degradation, 'terminal_penalty': terminal,
                    'throughput_kwh': throughput, 'exceed_hours': exceed,
                    'terminal_soc': soc, 'common_cost': bill + band + degradation + terminal}
        for field, value in expected.items():
            if not np.isclose(row[field], value, atol=1e-7, rtol=0.):
                raise RuntimeError(f'logged-action replay mismatch in {field}: {method}/{date}')
        replayed += 1
    return replayed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    results = workspace / 'experiments' / 'results'
    errors, missing = [], []
    counts = {'daily_files': 0, 'daily_rows': 0, 'solve_files': 0, 'solve_rows': 0,
              'validation_selections': 0, 'validation_rows': 0, 'checkpoint_files': 0,
              'independently_replayed_solver_days': 0, 'reproduced_batch_selections': 0}
    hashes = {}
    # Parameter-based checks are valid only against the same frozen inputs.
    for folder in ('data', 'configs'):
        for original in (ROOT / folder).rglob('*'):
            if original.is_file() and '__pycache__' not in original.parts:
                other = workspace / original.relative_to(ROOT)
                if not other.exists() or original.read_bytes() != other.read_bytes():
                    raise RuntimeError(f'audit parameter input mismatch: {original.relative_to(ROOT)}')
    params, _, _ = regime_bundle('confirmatory', [f'{s}_weekday' for s in SEASONS])
    day_lists = {part: load_region_days('confirmatory', part) for part in ('test', 'val')}
    expected_dates = {(part, f'{season}_weekday'): [day['date'] for day in day_lists[part]
                       if day['regime'] == f'{season}_weekday'][:30]
                      for part in day_lists for season in SEASONS}
    test_days = {str(day['date']): day for day in day_lists['test']}

    def dates_ok(frame, regime, part='test'):
        return len(frame) == 30 and set(frame['date']) == set(expected_dates[part, regime])

    def read(path):
        frame = read_aligned_parquet(path)
        hashes[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return frame

    expected_daily = set(expected_online_files())
    for name, (seasons, methods) in expected_online_files().items():
        path = results / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            frame = read(path)
            check_online_parameters(frame, name)
            expected_groups = {(f'{s}_weekday', m) for s in seasons for m in methods}
            groups = set(frame.groupby(['regime', 'method']).groups)
            if groups - expected_groups:
                raise RuntimeError('unexpected regime/method groups')
            for regime, method in expected_groups:
                sub = frame[(frame['regime'] == regime) & (frame['method'] == method)]
                if not dates_ok(sub, regime):
                    missing.append(f'{name}:{regime}/{method}:{len(sub)}/30')
        except Exception as error:
            errors.append(f'{name}: {error}')

    main_cells = [(f'v3_{season}_fee{fee}', f'{season}_weekday', fee, seeds)
                  for season, fee, seeds in [('winter', 20, 5), ('winter', 40, 25),
                      ('winter', 80, 5), ('spring', 40, 5), ('summer', 40, 5), ('autumn', 40, 5)]]
    for tag, regime, fee, n_seeds in main_cells:
        name = f'learned_daily_{tag}.parquet'
        expected_daily.add(name)
        path = results / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            frame = read(path)
            check_tariff_metadata(frame, 300, fee, 10)
            trained = frame[~frame['is_fallback']]
            if set(trained['seed']) != set(range(n_seeds)):
                raise RuntimeError('unexpected training seed set')
            for _, group in trained.groupby('seed'):
                if not dates_ok(group, regime):
                    raise RuntimeError('incomplete seed/date matrix')
            chosen = frame[frame['selected_by_batch']]
            if set(chosen['batch']) != set(range(n_seeds // 5)):
                raise RuntimeError('unexpected selected batch set')
            for _, group in chosen.groupby('batch'):
                if not dates_ok(group, regime):
                    raise RuntimeError('incomplete selected-batch/date matrix')
            counts['reproduced_batch_selections'] += check_batch_selection(
                workspace, frame, tag, regime, fee, n_seeds,
                [day for day in day_lists['val'] if day['regime'] == regime][:30], params[regime])
        except Exception as error:
            errors.append(f'{name}: {error}')

    for threshold in (250, 300, 350):
        for fee in (20, 40, 80):
            for width in (5, 10, 20):
                if threshold == 300 and width == 10:
                    continue
                name = f'selected_daily_v3_sens_t{threshold}_f{fee}_w{width}.parquet'
                expected_daily.add(name)
                path = results / name
                if not path.exists():
                    missing.append(name)
                else:
                    frame = read(path)
                    check_tariff_metadata(frame, threshold, fee, width)
                    if not dates_ok(frame, 'winter_weekday'):
                        errors.append(f'{name}: incomplete factorial dates')
    name = 'round2_reference_daily.parquet'
    expected_daily.add(name)
    if not (results / name).exists():
        missing.append(name)
    else:
        frame = read(results / name)
        if len(frame) != 90 or set(frame['fee_per_hour']) != {20, 40, 80}:
            errors.append('incomplete reference tariff matrix')
        for _, group in frame.groupby('fee_per_hour'):
            check_tariff_metadata(group, 300, group['fee_per_hour'].iloc[0], 10)
            if not dates_ok(group, 'winter_weekday'):
                errors.append('incomplete reference dates')

    for name in sorted(expected_daily):
        path = results / name
        if not path.exists():
            continue
        try:
            frame = read(path)
            check_daily_accounting(frame, params)
            if name.startswith(('learned_', 'selected_', 'round2_reference')):
                if not frame['exact_action_change_events'].eq(0).all():
                    raise RuntimeError('candidate policy changed by safety gate')
            counts['daily_files'] += 1
            counts['daily_rows'] += len(frame)
        except Exception as error:
            errors.append(f'{name}: {error}')
    for daily_name in expected_online_files():
        name = daily_name.replace('common_daily_', 'common_solves_')
        path = results / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            frame = read(path)
            check_solve_records(frame, test_days)
            daily = read(results / daily_name)
            expected = {(str(row.method), str(row.date))
                        for row in daily.itertuples()
                        if row.method in ONLINE[1:4]}
            actual = set(zip(frame['method'], frame['date']))
            if actual != expected:
                missing.append(f'{name}: daily/solve endpoint mismatch')
            counts['independently_replayed_solver_days'] += replay_solver_costs(
                frame, daily, test_days, params)
            counts['solve_files'] += 1
            counts['solve_rows'] += len(frame)
        except Exception as error:
            errors.append(f'{name}: {error}')

    for path in sorted((workspace / 'checkpoints').glob('v3_*/confirmatory/*/seed*/validation_selection.json')):
        audit = json.loads(path.read_text(encoding='utf-8'))
        if audit.get('action_search_version') != ACTION_SEARCH_VERSION:
            missing.append(f'{path.parent.relative_to(workspace)}: closed-grid selection pending')
            continue
        try:
            if audit['deployment_version'] != DEPLOYMENT_VERSION or audit['partition'] != 'validation':
                raise RuntimeError('invalid selection provenance')
            if audit['margin_eur_per_day'] != 0.10:
                raise RuntimeError('checkpoint selection margin changed')
            validation = read(path.parent / 'validation_daily.parquet')
            _, fee, _ = selection_tariff(path.parents[3].name)
            check_daily_accounting(validation.assign(fee_per_hour=fee), params)
            if audit['region'] != 'confirmatory' or audit['n_days'] != 30:
                raise RuntimeError('validation selection region or date count changed')
            if not validation['regime'].eq(audit['regime']).all():
                raise RuntimeError('validation regime metadata mismatch')
            if not validation['exact_action_change_events'].eq(0).all():
                raise RuntimeError('validation candidate changed by safety gate')
            if set(validation['iteration']) != {0, 1, 2, 3}:
                raise RuntimeError('incomplete checkpoint iteration set')
            incumbent, cost = 0, None
            for iteration, group in validation.groupby('iteration', sort=True):
                if not dates_ok(group, audit['regime'], 'val'):
                    raise RuntimeError('validation date matrix mismatch')
                mean = float(group['common_cost'].mean())
                if cost is None or cost - mean >= audit['margin_eur_per_day']:
                    incumbent, cost = int(iteration), mean
                checkpoint = path.parent / f'iter{int(iteration):02d}.pt'
                canonical = ROOT / checkpoint.relative_to(workspace)
                if not checkpoint.exists() or checkpoint.read_bytes() != canonical.read_bytes():
                    raise RuntimeError('stored checkpoint weights changed or missing')
                hashes[checkpoint.relative_to(workspace).as_posix()] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                counts['checkpoint_files'] += 1
            if incumbent != audit['selected_iteration']:
                raise RuntimeError('validation selection does not reproduce')
            recorded = json.loads((path.parent / 'result.json').read_text(encoding='utf-8'))
            if recorded['selected_iteration'] != incumbent:
                raise RuntimeError('result metadata does not identify the validation winner')
            counts['validation_selections'] += 1
            counts['validation_rows'] += len(validation)
        except Exception as error:
            errors.append(f'{path.relative_to(workspace)}: {error}')
    if counts['validation_selections'] != 74:
        missing.append(f"validation selections: {counts['validation_selections']}/74")
    report = {'workspace': str(workspace), 'partial_mode': args.partial,
              'status': ('failed' if errors else 'partial_not_release' if args.partial or missing
                         else 'input_accounting_and_coverage_passed'),
              'submission_ready': False, 'counts': counts,
              'missing': missing, 'errors': errors, 'artifact_sha256': hashes}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'artifact_sha256'}, indent=2))
    return int(bool(errors) or (bool(missing) and not args.partial))


if __name__ == '__main__':
    raise SystemExit(main())
