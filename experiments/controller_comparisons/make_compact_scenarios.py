"""Keep scenario-resolution context beside budget results; preserve full tables."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['fee', 'n_scenarios', 'time_limit_s']
SETTINGS = {(40, 2, 5), (40, 8, 5), (40, 16, 5), (80, 16, 5), (80, 16, 15)}


def render(settings, solves):
    assert set(map(tuple, settings[KEYS].to_numpy())) == SETTINGS
    assert set(map(tuple, solves[KEYS].to_numpy())) == SETTINGS
    assert settings.n_streams.eq(3).all() and settings.n_days_per_stream.eq(30).all()
    assert solves.n_streams.eq(3).all() and solves.n_solves.eq(8640).all()
    assert len(settings) == len(solves) == 5
    data = settings.set_index(KEYS)
    diagnostics = solves.set_index(KEYS)
    costs = [data.loc[(40, m, 5), 'mean_common_cost'] for m in (2, 8, 16)]
    limits = [100*diagnostics.loc[(40, m, 5), 'time_limit_fraction'] for m in (2, 8, 16)]
    high = [data.loc[(80, 16, b), 'mean_common_cost'] for b in (5, 15)]
    high_limit = 100*diagnostics.loc[(80, 16, 15), 'time_limit_fraction']
    return (
        'On the 30 winter test days, the three-stream scenario-resolution study\n'
        'gave mean MIQP costs of '
        f'{costs[0]:.1f}, {costs[1]:.1f}, and {costs[2]:.1f} EUR/day\n'
        'for $M=2,8,16$ at 40 EUR/h under five-second solve limits.\n'
        f'Time-limit frequencies were {limits[0]:.1f}\\%, {limits[1]:.1f}\\%, and {limits[2]:.1f}\\%.\n'
        'At 80 EUR/h, increasing the $M=16$ limit from 5 to 15 seconds reduced\n'
        f'mean cost from {high[0]:.1f} to {high[1]:.1f} EUR/day, but {high_limit:.1f}\\% of calls\n'
        'still reached the time limit. The supplement reports\n'
        'stream-level results, paired intervals, and solver diagnostics.\n')


def main():
    results = ROOT/'experiments/results'
    text = render(pd.read_csv(results/'comparative_scenario_setting_summary.csv'),
                  pd.read_csv(results/'comparative_scenario_solver_summary.csv'))
    (ROOT/'manuscript/generated/compact_scenarios.tex').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
