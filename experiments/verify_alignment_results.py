"""Release gate for versioned correction outputs, without changing artifacts."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.deployment_numerics import DEPLOYMENT_VERSION
from src.deployed_policy import ACTION_SEARCH_VERSION
from result_validation import read_aligned_parquet


def main():
    results = ROOT / 'experiments' / 'results'
    daily_paths = sorted(results.glob('*daily*.parquet'))
    if not daily_paths:
        raise RuntimeError('no daily artifacts')
    for path in daily_paths:
        frame = read_aligned_parquet(path)
        if 'common_cost' in frame:
            assert np.isfinite(frame['common_cost']).all(), path
            components = frame[['bill', 'capacity_fee', 'degradation_cost', 'terminal_penalty']].sum(axis=1)
            assert np.allclose(frame['common_cost'], components, atol=1e-7, rtol=0), path
            assert frame['soc_violations'].eq(0).all(), path
        if path.name.startswith(('learned_daily_', 'selected_daily_', 'round2_reference_daily')):
            assert frame['exact_action_change_events'].eq(0).all(), path
    for path in results.glob('common_solves_*.parquet'):
        frame = read_aligned_parquet(path)
        feasible = frame[frame['has_solution']]
        assert (np.minimum(feasible['first_charge_kw'], feasible['first_discharge_kw']) <= 1e-6).all(), path
        assert (feasible['first_action_recovery_kw'].abs() <= 1e-5 + 1e-8).all(), path
    selections = list((ROOT / 'checkpoints').glob('v3_*/confirmatory/*/seed*/validation_selection.json'))
    assert len(selections) == 74, len(selections)
    for path in selections:
        selected = json.loads(path.read_text(encoding='utf-8'))
        assert selected['deployment_version'] == DEPLOYMENT_VERSION, path
        assert selected['action_search_version'] == ACTION_SEARCH_VERSION, path
        assert selected['partition'] == 'validation' and selected['n_days'] == 30, path
    # Core cardinalities are fixed independently of which policy wins.
    for fee, seeds in [(20, 5), (40, 25), (80, 5)]:
        frame = read_aligned_parquet(results / f'learned_daily_v3_winter_fee{fee}.parquet')
        assert len(frame[~frame['is_fallback']]) == seeds * 30
        assert frame[frame['selected_by_batch']].groupby('batch').size().eq(30).all()
    print(f'alignment gate passed: {len(daily_paths)} daily files; 74 validation selections')


if __name__ == '__main__':
    main()
