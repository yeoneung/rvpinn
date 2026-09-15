"""Paired solver-budget reporting; no inference from conditional gap averages."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from result_validation import finite_solver_gaps
OUT = HERE / 'results/solver_budget'


def collect(kind, complete=True):
    manifest = json.loads((OUT/f'{kind}_manifest.json').read_text())
    rows, solves, missing = [], [], []
    for j in manifest['jobs']:
        path = OUT/(j['id']+'.json')
        if not path.exists():
            missing.append(j['id'])
            continue
        r = json.loads(path.read_text())
        assert r['job'] == j
        common = {k:v for k,v in j.items() if k not in ('id','kind')}
        for rec in r['solves']:
            solves.append(dict(**common, **rec))
        if kind == 'fixed':
            assert len(r['solves']) == 1
            rows.append(dict(**common, elapsed_s=r['elapsed_s'], action=r['action'],
                             scenario_sha256=r['scenario_sha256'], **r['solves'][0]))
        else:
            assert len(r['solves']) == 96
            rows.append(dict(**common, elapsed_s=r['elapsed_s'],
                             **{k:v for k,v in r['daily'].items() if k not in ('traj','date')}))
    if complete:
        assert not missing, f'{kind}: {len(missing)} unfinished jobs'
    return pd.DataFrame(rows), pd.DataFrame(solves), missing


def fixed_summary(df):
    keys = ['fee','scenarios','stream','date','hour','soc']
    assert (df.groupby(keys).scenario_sha256.nunique()==1).all()
    baseline = df[df.limit_s==5][keys+['action','primal_bound','dual_bound']].rename(
        columns={k:k+'_5s' for k in ('action','primal_bound','dual_bound')})
    joined = df.merge(baseline,on=keys,validate='many_to_one')
    for field in ('primal_bound','dual_bound','primal_bound_5s','dual_bound_5s'):
        joined.loc[~np.isfinite(joined[field]) | (joined[field].abs()>=1e19),field]=np.nan
    joined['action_change_kw'] = abs(joined.action-joined.action_5s)
    joined['primal_improvement_eur'] = joined.primal_bound_5s-joined.primal_bound
    joined['dual_improvement_eur'] = joined.dual_bound-joined.dual_bound_5s
    joined['absolute_gap_eur'] = joined.primal_bound-joined.dual_bound
    groups=[]
    for (fee,m,b), g in joined.groupby(['fee','scenarios','limit_s']):
        gaps = finite_solver_gaps(g)
        groups.append(dict(fee=fee,scenarios=m,limit_s=b,states=len(g),
                           incumbent_fraction=float(g.has_solution.mean()),
                           timelimit_fraction=float((g.status=='timelimit').mean()),
                           finite_gap_fraction=float(len(gaps)/len(g)),
                           median_finite_gap_percent=float(100*gaps.median()),
                           p95_finite_gap_percent=float(100*gaps.quantile(.95)),
                           mean_finite_gap_percent=float(100*gaps.mean()),
                           median_absolute_gap_eur=float(g.absolute_gap_eur.replace([np.inf,-np.inf],np.nan).median()),
                           mean_primal_improvement_eur=float(g.primal_improvement_eur.replace([np.inf,-np.inf],np.nan).mean()),
                           mean_dual_improvement_eur=float(g.dual_improvement_eur.replace([np.inf,-np.inf],np.nan).mean()),
                           mean_action_change_kw=float(g.action_change_kw.mean()),
                           changed_action_fraction=float((g.action_change_kw>1e-5).mean()),
                           mean_solve_s=float(g.solve_s.mean())))
    return pd.DataFrame(groups), joined


def closed_summary(days, solves):
    rows=[]
    for (fee,m,b), g in days.groupby(['fee','scenarios','limit_s']):
        s=solves[(solves.fee==fee)&(solves.scenarios==m)&(solves.limit_s==b)]
        dates=set(g.date)
        assert len(g)==3 and len(dates)==3 and len(s)==288
        learned=pd.read_parquet(ROOT/f'experiments/results/learned_daily_study_winter_fee{fee}.parquet')
        learned=learned[learned.selected_by_batch & learned.date.isin(dates)]
        expected_batches={40:5,80:1}[fee]
        assert learned.batch.nunique()==expected_batches and len(learned)==3*expected_batches
        assert learned.groupby('batch').date.nunique().eq(3).all()
        assert np.isfinite(learned.common_cost).all()
        lcost=float(learned.groupby('date').common_cost.mean().mean())
        primary=pd.read_parquet(ROOT/f'experiments/results/common_daily_confirmatory_thr300_fee{fee}_w10_n2.parquet')
        primary=primary[(primary.method=='stochastic_two_stage_exact_band_miqp') & primary.date.isin(dates)]
        assert len(primary)==3 and primary.date.nunique()==3
        heuristic=json.loads((HERE/f'results/heuristic/test_winter_weekday_fee{fee}.json').read_text())
        h=pd.DataFrame(heuristic['days']); h=h[h.date.isin(dates)]
        assert len(h)==3
        gaps=finite_solver_gaps(s)
        rows.append(dict(fee=fee,scenarios=m,limit_s=b,days=len(g),mean_cost=float(g.common_cost.mean()),
                         learned_cost=lcost,rule_cost=float(h.common_cost.mean()),
                         primary_miqp_cost=float(primary.common_cost.mean()),
                         cost_minus_learned=float(g.common_cost.mean()-lcost),
                         total_simulation_s=float(g.elapsed_s.sum()),mean_solve_s=float(s.solve_s.mean()),
                         p95_solve_s=float(s.solve_s.quantile(.95)),
                         timelimit_fraction=float((s.status=='timelimit').mean()),
                         incumbent_fraction=float(s.has_solution.mean()),
                         finite_gap_fraction=float(len(gaps)/len(s)),
                         median_finite_gap_percent=float(100*gaps.median()),
                         p95_finite_gap_percent=float(100*gaps.quantile(.95)),
                         mean_finite_gap_percent=float(100*gaps.mean())))
    return pd.DataFrame(rows)


def render(fixed, closed):
    lines=[r'\begin{table}[tbp]',r'\centering',
           r'\caption{Closed-loop budget sensitivity on three common winter test days. Costs are EUR/day; limit is seconds per solve. The $M=8,16$ runs use nested scenario stream 7301. Learned, rule, and primary $M=2$ costs use the same dates; $M=2$ uses the independent stream 4101. Each daily evaluation contains 96 solves.}',
           r'\label{tab:extended-budget}',r'\begin{tabular}{rrrrrrr}',r'\toprule',
           r'Fee & $M$ & Limit & MIQP cost & Learned & Rule & Primary $M=2$ \\',r'\midrule']
    for r in closed.itertuples():
        lines.append(f'{r.fee} & {r.scenarios} & {r.limit_s} & {r.mean_cost:.1f} & {r.learned_cost:.1f} & {r.rule_cost:.1f} & {r.primary_miqp_cost:.1f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
              'The budget comparison separates online optimization effort from scenario',
              'resolution. Closed-loop replay measures realized daily cost. Matched-state',
              'solves hold scenario arrays, horizon, and SoC fixed across budgets to',
              'measure changes in the primal objective and selected action.', '']
    supplement=[r'\begin{table}[htbp]',r'\centering',r'\scriptsize',
                r'\caption{Matched-state budget sweep: 36 state--stream combinations per fee, scenario count, and budget. TL denotes time-limit termination. Mean primal gain and absolute action change compare each run with its five-second counterpart. Gains refer to the remaining-horizon scenario objective, not realized daily costs. Finite reports gap availability; gap is the median among available finite solver gaps, not a measured optimality loss.}',
                r'\label{tab:fixed-budget}',r'\begin{tabular}{rrrrrrrr}',r'\toprule',
                r'Fee & $M$ & Limit & TL (\%) & Finite (\%) & Gap (\%) & Gain (EUR) & Action (kW) \\',r'\midrule']
    for r in fixed.itertuples():
        supplement.append(f'{r.fee} & {r.scenarios} & {r.limit_s} & {100*r.timelimit_fraction:.1f} & {100*r.finite_gap_fraction:.1f} & {r.median_finite_gap_percent:.1f} & {r.mean_primal_improvement_eur:.1f} & {r.mean_action_change_kw:.1f}'+r' \\')
    supplement += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
                   r'\begin{table}[htbp]',r'\centering',r'\scriptsize',
                   r'\setlength{\tabcolsep}{5pt}',
                   r'\caption{Closed-loop solver diagnostics. All 288 calls per row enter the time-limit (TL), incumbent (Inc.), and finite-gap availability fractions. Gap columns report the median and 95th percentile conditional on finite availability. Limit and mean are seconds; timing covers the solver, not the full controller call.}',
                   r'\label{tab:closed-budget-diagnostics}',r'\begin{tabular}{rrrrrrrrr}',r'\toprule',
                   r'Fee & $M$ & Limit & Mean & TL (\%) & Inc. (\%) & Finite (\%) & Gap med. (\%) & Gap P95 (\%) \\',r'\midrule']
    for r in closed.itertuples():
        supplement.append(f'{r.fee} & {r.scenarios} & {r.limit_s} & {r.mean_solve_s:.1f} & {100*r.timelimit_fraction:.1f} & {100*r.incumbent_fraction:.1f} & {100*r.finite_gap_fraction:.1f} & {r.median_finite_gap_percent:.1f} & {r.p95_finite_gap_percent:.1f}'+r' \\')
    supplement += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    return '\n'.join(lines), '\n'.join(supplement)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--partial',action='store_true'); args=parser.parse_args()
    fixed, _, missing_fixed=collect('fixed',not args.partial)
    summary, matched=fixed_summary(fixed)
    print('Fixed completed:',len(fixed),'remaining:',len(missing_fixed))
    print(summary.to_string(index=False))
    if args.partial:
        _,_,missing=collect('closed',False)
        print('Closed remaining:',len(missing))
        return
    days, solves,_=collect('closed')
    assert len(fixed)==576 and len(days)==36 and len(solves)==3456
    assert days.groupby(['fee','scenarios','limit_s']).size().eq(3).all()
    closed=closed_summary(days,solves)
    assert len(closed)==12
    summary.to_csv(OUT/'fixed_summary.csv',index=False)
    matched.to_parquet(OUT/'fixed_matched.parquet',index=False)
    days.to_parquet(OUT/'closed_daily.parquet',index=False)
    solves.to_parquet(OUT/'closed_solves.parquet',index=False)
    closed.to_csv(OUT/'closed_summary.csv',index=False)
    main_text,supp_text=render(summary,closed)
    (ROOT/'manuscript/generated/extended_budget.tex').write_text(main_text,encoding='utf-8')
    (ROOT/'manuscript/generated/extended_budget_supplement.tex').write_text(supp_text,encoding='utf-8')
    print(closed.to_string(index=False))


if __name__=='__main__':
    main()
