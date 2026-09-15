"""Report the entire validation slope sweep and the validation-selected tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results/surrogate'
sys.path.insert(0,str(ROOT/'experiments'))
from result_validation import finite_solver_gaps
from analyze_confirmatory import paired_stats, hierarchical_paired_stats


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text())
    val, daily, solves, pairs=[],[],[],[]
    for j in manifest['jobs']:
        r=json.loads((OUT/(j['id']+'.json')).read_text())
        assert r['job']==j
        val.append(dict(**j,slope=r['slope'],cost=r['daily']['common_cost']))
    validation=pd.DataFrame(val)
    sweep=validation.groupby(['fee','slope'],dropna=False).agg(
        width=('width','first'),mean_validation_cost=('cost','mean'),days=('date','nunique')).reset_index()
    assert len(sweep)==27 and sweep.days.eq(30).all()
    for fee,selection in selected.items():
        width=selection['selected']['width']
        name='envelope' if width is None else f'width{width}'
        files=sorted(OUT.glob(f'test_fee{fee}_{name}_*.json'))
        assert len(files)==30
        rows=[]
        for path in files:
            r=json.loads(path.read_text()); j=r['job']
            assert j['part']=='test' and j['width']==width and j['fee']==int(fee)
            rows.append(dict(fee=int(fee),width=width,slope=r['slope'],
                             **{k:v for k,v in r['daily'].items() if k!='traj'}))
            solves += [dict(fee=int(fee),date=j['date'],width=width,**s) for s in r['solves']]
        frame=pd.DataFrame(rows)
        assert frame.date.nunique()==30
        learned=pd.read_parquet(ROOT/f'experiments/results/learned_daily_study_winter_fee{fee}.parquet')
        learned=learned[learned.selected_by_batch]
        stat=(hierarchical_paired_stats(learned,frame,seed=39183+int(fee))
              if learned.batch.nunique()>1 else paired_stats(learned,frame,seed=39183+int(fee)))
        pairs.append(dict(fee=int(fee),**stat))
        daily.extend(rows)
    days=pd.DataFrame(daily); solve=pd.DataFrame(solves); pair=pd.DataFrame(pairs)
    primary=pd.read_csv(ROOT/'experiments/results/confirmatory_method_summary.csv')
    heuristic=pd.read_csv(HERE/'results/heuristic/summary.csv')
    summaries=[]
    for fee,g in days.groupby('fee'):
        s=solve[solve.fee==fee]; gaps=finite_solver_gaps(s)
        p=primary[primary.fee==fee].set_index('method')
        rule=heuristic[(heuristic.fee==fee)&(heuristic.regime=='winter_weekday')].iloc[0]
        summaries.append(dict(fee=int(fee),width=selected[str(fee)]['selected']['width'],slope=float(g.slope.iloc[0]),
                              mean_cost=float(g.common_cost.mean()),
                              learned_cost=float(p.loc['offline_value_policy','mean_common_cost']),
                              rule_cost=float(rule.rule_cost),
                              primary_miqp_cost=float(p.loc['stochastic_two_stage_exact_band_miqp','mean_common_cost']),
                              mean_latency_ms=float(g.latency_mean_ms.mean()),
                              timelimit_fraction=float((s.status=='timelimit').mean()),
                              incumbent_fraction=float(s.has_solution.mean()),
                              finite_gap_fraction=float(len(gaps)/len(s)),
                              median_finite_gap_percent=float(100*gaps.median()),
                              polish_calls=int(s.threshold_polish_kw.abs().gt(0).sum()),
                              maximum_polish_kw=float(s.threshold_polish_kw.abs().max())))
    summary=pd.DataFrame(summaries)
    validation.to_csv(OUT/'validation_daily.csv',index=False)
    sweep.to_csv(OUT/'validation_summary.csv',index=False)
    days.to_parquet(OUT/'selected_test_daily.parquet',index=False)
    solve.to_parquet(OUT/'selected_test_solves.parquet',index=False)
    summary.to_csv(OUT/'summary.csv',index=False)
    pair.to_csv(OUT/'paired_comparisons.csv',index=False)
    lines=[r'\begin{table}[tbp]',r'\centering',
           r'\caption{Validation-selected convex hinge surrogates on the 30 winter test days. Slope is EUR/(kW h), and costs are EUR/day. All controllers are evaluated under the hard discontinuous tariff.}',
           r'\label{tab:hinge-sweep}',r'\begin{tabular}{rrrrrr}',r'\toprule',
           r'Fee & Slope & Hinge & Learned & Rule & Primary MIQP \\',r'\midrule']
    for r in summary.itertuples():
        lines.append(f'{r.fee} & {r.slope:.4f} & {r.mean_cost:.1f} & {r.learned_cost:.1f} & {r.rule_cost:.1f} & {r.primary_miqp_cost:.1f}'+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
              'The sweep broadens the convex comparison beyond the greatest lower',
              'envelope. Validation hard cost selects the slope. The supplement',
              'reports every validation setting and the solver diagnostics.',
              'General hinge slopes preserve convexity but can exceed the hard fee.','']
    supp=[r'\begin{table}[htbp]',r'\centering',
          r'\caption{Complete convex-slope validation sweep: mean hard cost in EUR/day over 30 validation days. Width denotes $c_f/\gamma$ for hinge slope $\gamma$; the envelope uses the greatest-lower-envelope slope.}',
          r'\label{tab:hinge-validation}',r'\begin{tabular}{lrrr}',r'\toprule',
          r'Effective width (kW) & Fee 20 & Fee 40 & Fee 80 \\',r'\midrule']
    for width in [None,5,10,20,50,100,250,500,1000]:
        group=sweep[sweep.width.isna()] if width is None else sweep[sweep.width==width]
        assert len(group)==3
        costs=group.set_index('fee').mean_validation_cost
        supp.append(('Envelope' if width is None else str(width))+' & '+' & '.join(f'{costs[f]:.1f}' for f in (20,40,80))+r' \\')
    supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
             r'\begin{table}[htbp]',r'\centering',r'\small',
             r'\caption{Selected-hinge test diagnostics, 2,880 solves per fee. TL is the time-limit fraction. Cost intervals are paired two-sided 95\% intervals for learned minus hinge cost, in EUR/day. Central-fee intervals also resample learned-policy batches.}',
             r'\label{tab:hinge-diagnostics}',r'\begin{tabular}{rrrrrr}',r'\toprule',
             r'Fee & Latency (ms) & TL (\%) & Incumbent (\%) & Polish calls & Cost interval \\',r'\midrule']
    for r in summary.itertuples():
        stat=pair[pair.fee==r.fee].iloc[0]
        supp.append(f'{r.fee} & {r.mean_latency_ms:.1f} & {100*r.timelimit_fraction:.1f} & {100*r.incumbent_fraction:.1f} & {r.polish_calls} & [{stat.ci95_low:.1f}, {stat.ci95_high:.1f}]'+r' \\')
    supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (ROOT/'manuscript/generated/hinge_sweep.tex').write_text('\n'.join(lines),encoding='utf-8')
    (ROOT/'manuscript/generated/hinge_sweep_supplement.tex').write_text('\n'.join(supp),encoding='utf-8')
    print(summary.to_string(index=False)); print(sweep.to_string(index=False)); print(pair.to_string(index=False))


if __name__=='__main__':
    main()
