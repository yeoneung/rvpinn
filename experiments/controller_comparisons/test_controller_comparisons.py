"""Focused contract tests; do not run long simulations or neural training."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from src.config import ModelParams
from src.deployment_numerics import BAND_SEPARATION_KW
from run_surrogates import ThresholdPolishedSurrogate, ExactMIPController
from audit_results import check_path


def test_surrogate_polish_preserves_strict_fee(monkeypatch):
    p = ModelParams(g_thr=300.,c_step=40.)
    raw = 40.-0.5*BAND_SEPARATION_KW
    monkeypatch.setattr(ExactMIPController,'__call__',lambda *args:raw)
    controller = ThresholdPolishedSurrogate(p)
    a = controller(8.,.5,0.,0.,dict(N_now=340.,C_now=.05))
    assert 340.-a <= p.g_thr
    assert 0. < a-raw <= BAND_SEPARATION_KW
    raw = 40.-2*BAND_SEPARATION_KW
    assert controller(8.,.5,0.,0.,dict(N_now=340.,C_now=.05)) == raw
    assert 340.-raw > p.g_thr  # No fee tolerance is introduced.


def test_frozen_source_hashes():
    for name in ('fixed','closed'):
        path = HERE/f'results/solver_budget/{name}_manifest.json'
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        assert m['protocol_sha256'] == hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
        assert m['script_sha256'] == hashlib.sha256((HERE/'run_solver_budget.py').read_bytes()).hexdigest()
    m = json.loads((HERE/'results/surrogate/manifest.json').read_text())
    for name, sha in m['hashes'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == sha
    execution = json.loads((HERE/'results/solver_budget/adaptive_execution.json').read_text())
    for record in execution:
        assert record['runner_sha256'] == hashlib.sha256((HERE/'run_closed_adaptive.py').read_bytes()).hexdigest()


def test_fixed_scenarios_match_across_budgets():
    hashes = {}
    budgets = {}
    for path in (HERE/'results/solver_budget').glob('fixed_fee*.json'):
        r = json.loads(path.read_text())
        j = r['job']
        key = tuple(j[k] for k in ('fee','scenarios','stream','date','hour','soc'))
        hashes.setdefault(key, set()).add(r['scenario_sha256'])
        budgets.setdefault(key, set()).add(j['limit_s'])
    assert len(hashes) == 144 and all(len(v) == 1 for v in hashes.values())
    assert all(v == {5, 15, 60, 300} for v in budgets.values())


def test_independent_accounting_detects_one_wrong_fee():
    p = ModelParams(g_thr=300., c_step=40.)
    actions, net, prices = [0.]*96, [301.]*96, [.05]*96
    cost = p.T*(301*.05+40)+p.lam_T*(.5-p.s_tar)**2
    assert check_path(p,actions,net,prices,.5,.5,cost)['cost_error_eur'] < 1e-8
    with pytest.raises(AssertionError):
        check_path(p,actions,net,prices,.5,.5,cost-p.dt_ctrl*p.c_step)


def test_validation_selection_not_reuse_retuning():
    m = json.loads((HERE/'results/heuristic_reuse/manifest.json').read_text())
    selection = json.loads((HERE/'results/heuristic/selection.json').read_text())
    selected = next(s for s in selection if s['fee']==40 and s['regime']=='winter_weekday')
    assert m['setting'] == selected['setting']


def test_budget_summary_excludes_solver_infinity_without_dropping_actions():
    from analyze_solver_budget import fixed_summary
    base=dict(fee=40,scenarios=16,stream=7301,date='2018-01-02',hour=8,soc=.5,
              scenario_sha256='same',has_solution=True,status='timelimit',solve_s=5.)
    rows=[dict(**base,limit_s=5,action=0.,primal_bound=1500.,dual_bound=-1e20,gap=1e20),
          dict(**base,limit_s=60,action=100.,primal_bound=1200.,dual_bound=1000.,gap=.2)]
    summary, matched=fixed_summary(pd.DataFrame(rows))
    assert len(matched)==2 and len(summary)==2
    five=summary[summary.limit_s==5].iloc[0]
    sixty=summary[summary.limit_s==60].iloc[0]
    assert five.incumbent_fraction==1 and five.finite_gap_fraction==0
    assert np.isnan(five.median_finite_gap_percent)
    assert sixty.median_finite_gap_percent==pytest.approx(20.)
    assert sixty.p95_finite_gap_percent==pytest.approx(20.)
    assert np.isnan(five.p95_finite_gap_percent)
    assert sixty.mean_primal_improvement_eur==300. and sixty.mean_action_change_kw==100.
    assert np.isnan(sixty.mean_dual_improvement_eur)


def test_solver_record_check_detects_wrong_held_action_or_fee_side():
    from audit_results import check_solver_record
    record = dict(first_action_held=50., first_action_raw=50.,
                  first_action_recovery_kw=0., first_band_active=False)
    check_solver_record(record, 350., 50., 300.)
    with pytest.raises(AssertionError, match='Held action'):
        check_solver_record(record, 350., 50.+1e-12, 300.)
    with pytest.raises(AssertionError, match='fee sides'):
        check_solver_record(dict(record, first_band_active=True), 350., 50., 300.)
