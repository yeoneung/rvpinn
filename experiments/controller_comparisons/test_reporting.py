"""Completed-comparator results, selection, and manuscript scope must agree."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from experiments.manuscript_checks import enabled

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def test_hinge_selection_reconstructs_all_validation_costs():
    manifest=json.loads((HERE/'results/surrogate/manifest.json').read_text())
    selections=json.loads((HERE/'results/surrogate/selection.json').read_text())
    assert len(manifest['jobs'])==810
    for fee,selection in selections.items():
        best=None
        for candidate in selection['candidates']:
            jobs=[j for j in manifest['jobs'] if j['fee']==int(fee) and j['width']==candidate['width']]
            assert len(jobs)==30 and len({j['date'] for j in jobs})==30
            costs=[json.loads((HERE/'results/surrogate'/(j['id']+'.json')).read_text())['daily']['common_cost'] for j in jobs]
            assert np.mean(costs)==pytest.approx(candidate['mean_cost'],abs=1e-9)
            if best is None or best['mean_cost']-candidate['mean_cost']>=.10:
                best=candidate
        assert best==selection['selected']


def test_selected_hinge_summary_and_complete_solve_counts():
    out=HERE/'results/surrogate'
    summary=pd.read_csv(out/'summary.csv')
    days=pd.read_parquet(out/'selected_test_daily.parquet')
    solves=pd.read_parquet(out/'selected_test_solves.parquet')
    assert len(days)==90 and len(solves)==8640
    for row in summary.itertuples():
        d=days[days.fee==row.fee]; s=solves[solves.fee==row.fee]
        assert len(d)==30 and d.date.nunique()==30 and len(s)==2880
        assert d.common_cost.mean()==pytest.approx(row.mean_cost)
        assert s.has_solution.all() and not (s.status=='timelimit').any()
        assert s.threshold_polish_kw.abs().max()<=1e-5


def test_cost_latency_figure_includes_competitive_rules():
    frame=pd.read_csv(HERE/'results/cost_latency_points.csv')
    rule=pd.read_csv(HERE/'results/heuristic/summary.csv')
    assert len(frame)==18 and frame.groupby('fee').method.nunique().eq(6).all()
    assert frame.latency_ms.gt(0).all()
    for r in rule[rule.regime=='winter_weekday'].itertuples():
        point=frame[(frame.fee==r.fee)&(frame.method=='tariff_rule')].iloc[0]
        assert point.cost==pytest.approx(r.rule_cost)
        assert point.latency_ms==pytest.approx(r.rule_latency_ms)
    if enabled():
        original=(ROOT/'manuscript/generated/confirmatory_results.tex').read_text()
        expanded=(ROOT/'manuscript/generated/controller_comparison_results.tex').read_text()
        import re
        pattern=r'\\begin\{figure\}\[t\].*?\\end\{figure\}'
        assert expanded.count('Tariff rule (expl.)') == 3
        assert 'tariff-aware rule is an exploratory comparison' in expanded
        for line in original.splitlines():
            if re.match(r'^(20|40|80| & )', line) and '&' in line:
                assert line in expanded


@pytest.mark.skipif(not enabled(), reason="Current manuscript is maintained separately")
def test_reference_table_is_preserved_outside_main_headlines():
    generated=ROOT/'manuscript/generated'
    table=(generated/'reference_attribution_table.tex').read_text().strip()
    assert table in (generated/'comparative_reference.tex').read_text()
    main=(ROOT/'manuscript/main.tex').read_text()
    assert r'\input{generated/compact_reference}' in main
    assert r'\input{generated/comparative_reference}' not in main
    assert '24.6' not in main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    assert '0.05--0.37' in (generated/'compact_reference.tex').read_text()


@pytest.mark.skipif(not enabled(), reason="Current manuscript is maintained separately")
def test_main_retains_one_policy_bound_and_no_dangling_grid_proposition():
    main=(ROOT/'manuscript/main.tex').read_text()
    supplement=(ROOT/'manuscript/supplement.tex').read_text()
    assert main.count(r'\begin{proposition}')==1
    assert r'\ref{prop:grid-defect}' not in main
    assert r'\input{deployment_analysis}' in supplement
    assert '1745.6' in main and '33.1 EUR/day' in main
    assert 'not a certified bound on the empirical' in main


def test_compact_scenario_summary_preserves_full_comparison():
    from make_compact_scenarios import render
    results=ROOT/'experiments/results'
    generated=ROOT/'manuscript/generated'
    settings=pd.read_csv(results/'comparative_scenario_setting_summary.csv')
    solves=pd.read_csv(results/'comparative_scenario_solver_summary.csv')
    compact=render(settings,solves)
    for value in settings.mean_common_cost:
        assert f'{value:.1f}' in compact
    if enabled():
        assert compact==(generated/'compact_scenarios.tex').read_text()
        assert r'\input{generated/compact_scenarios}' in (ROOT/'manuscript/main.tex').read_text()
        assert r'\input{generated/additional_results}' not in (ROOT/'manuscript/main.tex').read_text()
        assert r'\input{generated/comparative_scenarios}' in (ROOT/'manuscript/supplement.tex').read_text()
        assert r'\label{tab:comparative-scenarios}' in (generated/'comparative_scenarios.tex').read_text()


def test_complete_budget_outputs_match_all_frozen_jobs():
    from analyze_solver_budget import collect, fixed_summary, closed_summary, render
    fixed, _, missing_fixed = collect('fixed')
    days, solves, missing_closed = collect('closed')
    assert not missing_fixed and not missing_closed
    assert len(fixed) == 576 and len(days) == 36 and len(solves) == 3456
    assert days.groupby(['fee', 'scenarios', 'limit_s']).size().eq(3).all()
    f, _ = fixed_summary(fixed)
    c = closed_summary(days, solves)
    assert len(f) == 16 and len(c) == 12
    assert set(c.limit_s) == {5, 60, 300}
    expected_main, expected_supplement = render(f, c)
    generated = ROOT/'manuscript/generated'
    assert expected_main.count(r'\begin{table}') == 1
    assert expected_supplement.count(r'\begin{table}') == 2
    for r in c.itertuples():
        assert f'{r.mean_cost:.1f}' in expected_main
    if enabled():
        assert (generated/'extended_budget.tex').read_text() == expected_main
        assert (generated/'extended_budget_supplement.tex').read_text() == expected_supplement
        assert r'\input{generated/extended_budget}' in (ROOT/'manuscript/main.tex').read_text()
        assert r'\input{generated/extended_budget_supplement}' in (ROOT/'manuscript/supplement.tex').read_text()


def test_representative_day_uses_retained_models_and_same_date():
    out=HERE/'results/heuristic_mechanism'
    manifest=json.loads((out/'manifest.json').read_text())
    original=ROOT/'experiments/results'
    old=json.loads((original/'comparative_mechanism_day_metadata.json').read_text())
    assert manifest['date']==old['date']=='2019-01-07'
    for key,sha in manifest['retained_summary_sha256'].items():
        assert hashlib.sha256((original/f'comparative_mechanism_{key}_summary.json').read_bytes()).hexdigest()==sha
    assert len(pd.read_parquet(out/'rule_trajectory.parquet'))==96
