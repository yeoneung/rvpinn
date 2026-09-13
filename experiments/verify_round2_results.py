"""Fail fast on incomplete or internally inconsistent version-4 artifacts."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from analyze_round2_scenarios import COUNTS, STREAMS, paths
from result_validation import read_aligned_parquet


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def check_mechanism_layout(metadata, summary, trajectory):
    if (set(summary['controller']) != {'learned', 'reference', 'm2', 'm16'}
            or len(summary) != 4
            or set(trajectory['controller']) != {'learned', 'reference', 'm2', 'm16'}
            or len(trajectory) != 384
            or not trajectory.groupby('controller').size().eq(96).all()):
        raise RuntimeError('incomplete representative-day artifacts')
    expected = {'fee_per_hour': 40., 'threshold_kw': 300., 'scenario_count': 16,
                'scenario_stream': 7301, 'scenario_pool_size': 16, 'time_limit_s': 5.,
                'primary_scenario_count': 2, 'primary_scenario_stream': 4101,
                'learned_batch': 0}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError('representative-day setting differs from the protocol')


def verify_mechanism(results):
    metadata = json.loads((results / 'round2_mechanism_day_metadata.json').read_text(encoding='utf-8'))
    summary = pd.read_csv(results / 'round2_mechanism_day_summary.csv')
    trajectory = pd.read_parquet(results / 'round2_mechanism_day_trajectory.parquet')
    check_mechanism_layout(metadata, summary, trajectory)
    from make_round2_mechanism_figure import _check_reported_economics
    from src.deployment_numerics import DEPLOYMENT_VERSION
    for key in ('learned', 'reference', 'm2', 'm16'):
        record = json.loads((results / f'round2_mechanism_{key}_summary.json').read_text(encoding='utf-8'))
        if (record.get('deployment_version') != DEPLOYMENT_VERSION
                or record.get('reported_economics_verified') is not True
                or record.get('date') != metadata['date']):
            raise RuntimeError('representative day lacks a corrected source check')
        source_rows = []
        for name, digest in record['source_sha256'].items():
            if Path(name).name != name:
                raise RuntimeError('mechanism provenance must name a local result file')
            path = results / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError('representative-day input hash changed')
            if 'daily' in name:
                frame = read_aligned_parquet(path)
                frame = frame[frame['date'].astype(str) == metadata['date']]
                if key == 'learned':
                    frame = frame[frame['selected_by_batch'] & (frame['batch'] == 0)]
                elif key == 'reference':
                    frame = frame[frame['fee_per_hour'] == 40]
                else:
                    frame = frame[frame['method'] == 'stochastic_two_stage_exact_band_miqp']
                source_rows.extend(frame.to_dict('records'))
        if len(source_rows) != 1:
            raise RuntimeError('mechanism requires one matching reported daily row')
        _check_reported_economics(dict(record, exact_action_change_events=0), source_rows[0])
        displayed = summary[summary['controller'] == key].iloc[0]
        _check_reported_economics(dict(displayed, exact_action_change_events=0), record)
        path = results / f'round2_mechanism_{key}_trajectory.parquet'
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['trajectory_sha256']:
            raise RuntimeError('representative-day trajectory hash changed')
        individual = pd.read_parquet(path).reset_index(drop=True)
        combined = trajectory[trajectory['controller'] == key].drop(columns='controller').reset_index(drop=True)
        if not combined.equals(individual):
            raise RuntimeError('combined mechanism trajectory differs from its checked input')


def main() -> int:
    reference = read_aligned_parquet(RESULTS / "round2_reference_daily.parquet")
    assert len(reference) == 90
    assert set(reference["fee_per_hour"].astype(int)) == {20, 40, 80}
    assert reference.groupby("fee_per_hour")["date"].nunique().eq(30).all()
    assert np.isfinite(reference["common_cost"]).all()

    settings = [(40, count, 5) for count in COUNTS] + [(80, 16, 5), (80, 16, 15)]
    for fee, count, budget in settings:
        for stream in STREAMS:
            daily_path, solve_path = paths(fee, count, stream, budget)
            daily = read_aligned_parquet(daily_path)
            solves = read_aligned_parquet(solve_path)
            assert len(daily) == 30 and daily["date"].nunique() == 30
            assert len(solves) == 30 * 96
            assert set(daily["n_scenarios"].astype(int)) == {count}
            assert set(daily["scenario_pool_size"].astype(int)) == {16}
            assert set(daily["scenario_stream_seed"].astype(int)) == {stream}
            assert np.isfinite(daily["common_cost"]).all()
            assert solves["has_solution"].isin([True, False]).all()
            assert (solves["solve_s"] >= 0.0).all()

    verify_mechanism(RESULTS)

    print("round-2 artifact verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
