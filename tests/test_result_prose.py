"""Reporting must follow the current estimates, including reversed outcomes."""
import pandas as pd
import pytest

from experiments import analyze_round2_reference as reference_report
from experiments.analyze_confirmatory import COMPARATORS, paired_comparison_table


def test_compact_primary_table_keeps_full_multiplicity_family_in_supplement():
    pairs = pd.DataFrame([
        {'fee': fee, 'comparator': method, 'mean_diff': 12.,
         'percent_diff': 1.2, 'one_sided_95_upper': 14., 'holm_p_all': 1.}
        for fee in (20, 40, 80) for method in COMPARATORS])
    primary = '\n'.join(paired_comparison_table(pairs, primary_only=True))
    full = '\n'.join(paired_comparison_table(pairs))
    assert primary.count(' & 12.0 & ') == 6
    assert full.count(' & 12.0 & ') == 15
    assert 'all 15 comparisons' in primary and 'all 15 comparisons' in full
    assert 'No storage &' not in primary and full.count('No storage &') == 3
    assert 'tab:paired}' in primary and 'tab:paired-all}' in full


@pytest.mark.parametrize('difference,superior', [(-12.0, True), (12.0, False)])
def test_reference_report_does_not_preserve_old_advantage(
        tmp_path, monkeypatch, difference, superior):
    rows, pairs = [], []
    for fee in (20, 40, 80):
        for method in ('reference_value_one_step', 'offline_value_policy'):
            delta = difference if method == 'offline_value_policy' else 0.0
            rows.append({
                'fee': fee, 'method': method,
                'mean_common_cost': 100.0 + delta,
                'mean_bill': 60.0 + delta / 2.0,
                'mean_capacity_fee': 20.0 + delta / 4.0,
                'mean_degradation_cost': 20.0 + delta / 4.0,
                'mean_terminal_penalty': 0.0,
                'mean_throughput_kwh': 200.0,
                'mean_exceed_hours': 20.0 / fee,
                'mean_latency_mean_ms': 5.0,
            })
        pairs.append({
            'fee': fee, 'mean_diff': difference,
            'percent_diff': difference,
            'one_sided_95_upper': difference + 1.0,
            'superiority': superior,
        })
    monkeypatch.setattr(reference_report, 'GENERATED', tmp_path)
    reference_report._write_latex(pd.DataFrame(rows), pd.DataFrame(pairs))
    prose = (tmp_path / 'round2_reference.tex').read_text(encoding='utf-8')
    detail = (tmp_path / 'round2_reference_supplement.tex').read_text(encoding='utf-8')
    outcome = 'established' if superior else 'not established'
    assert detail.count(f'superiority was {outcome}.') == 3
    assert detail.count(f'cost difference was {difference:.1f} EUR/day') == 3
    assert f'changing the band fee by {difference / 4.0:.1f} EUR/day' in prose
    assert '120.3' not in prose and '444.5' not in prose
    if not superior:
        assert 'superiority was established.' not in detail
        assert 'inconclusive' not in prose
        assert 'was not met at any' in prose
    else:
        assert 'all three anchors' in prose
