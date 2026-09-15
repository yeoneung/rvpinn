"""Compare the tariff rule with retained learned/M=2 trajectories on the fixed day."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE))
from src.exp_common import regime_bundle
from experiments.make_comparative_mechanism_figure import _soc_plot_coordinates
from audit_results import check_daily


def main():
    (ROOT/'manuscript/figures').mkdir(parents=True, exist_ok=True)
    (ROOT/'manuscript/generated').mkdir(parents=True, exist_ok=True)
    original=ROOT/'experiments/results'
    meta=json.loads((original/'comparative_mechanism_day_metadata.json').read_text())
    date=meta['date']
    selection=json.loads((HERE/'results/heuristic/selection.json').read_text())
    selected=next(s for s in selection if s['fee']==40 and s['regime']=='winter_weekday')
    params,profiles,_=regime_bundle('confirmatory',['winter_weekday'])
    p,prof=params['winter_weekday'],profiles['winter_weekday']
    p.c_step,p.g_thr,p.w_step=40.,300.,10.
    saved=json.loads((HERE/'results/heuristic/test_winter_weekday_fee40.json').read_text())
    reported=next(d for d in saved['days'] if d['date']==date)
    result=dict(reported)
    result['traj']=pd.read_parquet(HERE/'results/heuristic_mechanism/rule_trajectory.parquet').to_dict('list')
    check_daily(p,result)
    trajectories={'rule':pd.DataFrame(result['traj'])}
    costs={'rule':reported}
    inputs={}
    for key in ('learned','m2'):
        trajectory_path=original/f'comparative_mechanism_{key}_trajectory.parquet'
        summary_path=original/f'comparative_mechanism_{key}_summary.json'
        trajectories[key]=pd.read_parquet(trajectory_path)
        costs[key]=json.loads(summary_path.read_text())
        assert costs[key]['date']==date
        assert costs[key]['trajectory_sha256']==hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
        inputs[key]=hashlib.sha256(summary_path.read_bytes()).hexdigest()
    out=HERE/'results/heuristic_mechanism'; out.mkdir(parents=True,exist_ok=True)
    trajectories['rule'].to_parquet(out/'rule_trajectory.parquet',index=False)
    pd.DataFrame([dict(controller=k,**{n:costs[k][n] for n in
                 ('bill','capacity_fee','degradation_cost','terminal_penalty','common_cost','exceed_hours')})
                 for k in ('learned','m2','rule')]).to_csv(out/'summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(date=date,
        date_selection=meta['selection_rule'],setting=selected['setting'],retained_summary_sha256=inputs),indent=2),encoding='utf-8')
    labels={'learned':'Learned','m2':'Primary MIQP (M=2)','rule':'Tariff-aware rule'}
    styles={'learned':('#0072B2','-'),'m2':('#D55E00','--'),'rule':('#009E73','-.')}
    fig,axes=plt.subplots(3,1,figsize=(7.2,6.5),sharex=True,layout='constrained')
    for key in ('learned','m2','rule'):
        frame=trajectories[key]; color,style=styles[key]
        clock=np.r_[frame.t.to_numpy(),p.T]
        axes[0].step(clock,np.r_[frame.G.to_numpy(),frame.G.iloc[-1]],where='post',
                     label=labels[key],color=color,linestyle=style,linewidth=1.2)
        st,sv=_soc_plot_coordinates(frame,costs[key]['terminal_soc'],p)
        axes[1].plot(st,sv,color=color,linestyle=style,linewidth=1.2)
        axes[2].step(clock,np.r_[frame.a.to_numpy(),frame.a.iloc[-1]],where='post',
                     color=color,linestyle=style,linewidth=1.2)
    axes[0].axhline(p.g_thr,color='0.35',linestyle=':',linewidth=1.)
    axes[0].annotate('Tariff threshold: 300 kW', xy=(23.8,p.g_thr),
                     xytext=(0,4), textcoords='offset points',
                     ha='right', va='bottom', fontsize=8, color='0.25',
                     bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    axes[0].set_ylabel('Grid import (kW)'); axes[1].set_ylabel('State of charge')
    axes[2].set_ylabel('Battery power (kW)'); axes[2].set_xlabel('Hour')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.25),ncol=3,frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2); ax.set_xlim(0,24)
    axes[2].set_xticks(np.arange(0,25,4))
    fig.savefig(ROOT/'manuscript/figures/fig_tariff_mechanism.pdf')
    fig.savefig(out/'figure.png',dpi=150)
    plt.close(fig)
    lines=[r'\begin{table}[tbp]',r'\centering',r'\small',
           r'\caption{Cost decomposition on the protocol-selected winter day (7 January 2019), in EUR. Exceedance duration is in hours.}',
           r'\label{tab:tariff-mechanism}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Controller & Bill & Fee & Degradation & Terminal & Total & Exceed h \\',r'\midrule']
    for key in ('learned','m2','rule'):
        r=costs[key]
        label={'learned':'Learned','m2':r'MIQP ($M=2$)','rule':'Tariff rule'}[key]
        lines.append(f"{label} & {r['bill']:.1f} & {r['capacity_fee']:.1f} & {r['degradation_cost']:.1f} & {r['terminal_penalty']:.2f} & {r['common_cost']:.1f} & {r['exceed_hours']:.2f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (ROOT/'manuscript/generated/heuristic_mechanism.tex').write_text('\n'.join(lines),encoding='utf-8')
    print(pd.read_csv(out/'summary.csv').to_string(index=False))


if __name__=='__main__':
    main()
