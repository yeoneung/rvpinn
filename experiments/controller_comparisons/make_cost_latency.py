"""Add the completed rule and selected hinge to the common-day cost/time figure."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    (ROOT/'manuscript/figures').mkdir(parents=True, exist_ok=True)
    (ROOT/'manuscript/generated').mkdir(parents=True, exist_ok=True)
    primary=pd.read_csv(ROOT/'experiments/results/confirmatory_method_summary.csv')
    rule=pd.read_csv(HERE/'results/heuristic/summary.csv')
    hinge=pd.read_csv(HERE/'results/surrogate/summary.csv')
    order=('convex_envelope_mpc','deterministic_exact_band_miqp',
           'stochastic_two_stage_exact_band_miqp','offline_value_policy','tariff_rule','selected_hinge')
    labels=('Envelope MPC','CM MIQP','Primary stochastic MIQP','Learned','Tariff-aware rule','Selected hinge')
    colors=('#999999','#CC79A7','#D55E00','#0072B2','#009E73','#E69F00')
    markers=('^','D','P','X','o','s')
    rows=[]
    for fee in (20,40,80):
        p=primary[primary.fee==fee].set_index('method')
        baseline=float(p.loc['no_storage','mean_common_cost'])
        for method in order[:4]:
            r=p.loc[method]
            rows.append(dict(fee=fee,method=method,cost=r.mean_common_cost,latency_ms=r.mean_latency_mean_ms,baseline=baseline))
        r=rule[(rule.regime=='winter_weekday')&(rule.fee==fee)].iloc[0]
        h=hinge[hinge.fee==fee].iloc[0]
        rows.extend([dict(fee=fee,method='tariff_rule',cost=r.rule_cost,latency_ms=r.rule_latency_ms,baseline=baseline),
                     dict(fee=fee,method='selected_hinge',cost=h.mean_cost,latency_ms=h.mean_latency_ms,baseline=baseline)])
    points=pd.DataFrame(rows)
    assert len(points)==18 and np.isfinite(points[['cost','latency_ms']]).all().all() and points.latency_ms.gt(0).all()
    points.to_csv(HERE/'results/cost_latency_points.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(8.7,3.5),sharex=True,sharey=True,layout='constrained')
    handles=[]
    for ax,fee in zip(axes,(20,40,80)):
        frame=points[points.fee==fee].set_index('method')
        for method,label,color,marker in zip(order,labels,colors,markers):
            r=frame.loc[method]
            point=ax.scatter(r.latency_ms,100*(r.cost/r.baseline-1),s=45,
                             marker=marker,facecolors='none' if method=='selected_hinge' else color,
                             edgecolors=color if method=='selected_hinge' else 'black',linewidths=.9)
            if fee==20:
                handles.append(point)
        ax.axhline(0,color='0.5',linestyle='--',linewidth=.7)
        ax.set_xscale('log'); ax.set_title(f'Fee {fee} EUR/h',fontsize=12)
        ax.set_xlabel('Mean call latency (ms)\n(log scale)',fontsize=11)
        ax.grid(alpha=.25,which='both',linewidth=.4)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel('Cost vs no storage (%)',fontsize=11)
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=10)
    fig.savefig(ROOT/'manuscript/figures/fig_cost_latency_rules.pdf')
    fig.savefig(HERE/'results/cost_latency.png',dpi=160)
    plt.close(fig)
    original=(ROOT/'manuscript/generated/confirmatory_results.tex').read_text(encoding='utf-8')
    pattern=r'\\begin\{figure\}\[t\].*?\\end\{figure\}'
    blocks=re.findall(pattern,original,flags=re.S)
    assert len(blocks)==1 and 'fig_cost_latency.pdf' in blocks[0]
    replacement=(r'\begin{figure}[t]'+'\n'+r'\centering'+'\n'
        r'\includegraphics[width=\linewidth]{figures/fig_cost_latency_rules.pdf}'+'\n'
        r'\caption{Operating cost and online computation on the same 30 winter days. The dashed line is no-storage cost. Primary controllers are shown alongside exploratory tariff-rule and validation-selected hinge comparisons. Learned evaluation uses the GPU; other procedures use one CPU thread per controller. Rule and hinge timings were measured separately on the shared workstation. Latency excludes offline selection and training.}'+'\n'
        r'\label{fig:cost-latency}'+'\n'+r'\end{figure}')
    updated=original.replace(blocks[0],replacement)
    assert updated.replace(replacement,'')==original.replace(blocks[0],'')
    note = (' The tariff-aware rule is an exploratory comparison selected on validation data; '
            'its timings were measured separately on the same workstation.')
    updated = updated.replace('The price rule is validation-tuned.}',
                              'The price rule is validation-tuned.'+note+'}')
    for fee in (20,40,80):
        row=rule[(rule.regime=='winter_weekday')&(rule.fee==fee)].iloc[0]
        daily=json.loads((HERE/f'results/heuristic/test_winter_weekday_fee{fee}.json').read_text())['days']
        p95=np.mean([d['latency_p95_ms'] for d in daily])
        baseline=float(primary[(primary.fee==fee)&(primary.method=='no_storage')].iloc[0].mean_common_cost)
        anchor=f' & Price rule & {baseline:.1f} & 0.00 & 0.0/0.0 '+r'\\'
        addition=(f' & Tariff rule (expl.) & {row.rule_cost:.1f} & '
                  f'{row.rule_terminal_penalty:.2f} & {row.rule_latency_ms:.3f}/{p95:.3f} '+r'\\')
        assert anchor in updated
        updated=updated.replace(anchor,anchor+'\n'+addition,1)
    (ROOT/'manuscript/generated/controller_comparison_results.tex').write_text(updated,encoding='utf-8')
    print(points.to_string(index=False))


if __name__=='__main__':
    main()
