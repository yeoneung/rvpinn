"""Generate the reduced-model tables from stored results only."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def render(summary, selection, audit):
    fine, middle = summary['grids'][-1], summary['grids'][-2]
    opt = fine['optimal_initial_costs'][1]
    change = abs(opt-middle['optimal_initial_costs'][1])
    rows = [r'\begin{table}[tbp]', r'\centering', r'\small',
        r'\caption{Hard-tariff reduced-model comparison from $(S,Y)=(0.5,0)$. Costs and gaps are EUR/day. Boundary regret is in EUR per decision over grid states where the DP action leaves import within 10 kW of the threshold; it is not trajectory weighted.}',
        r'\label{tab:hard-dp}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Policy & Cost & Gap to DP & Boundary mean & Boundary P95 \\', r'\midrule',
        f'DP & {opt:.2f} & 0.00 & 0.000 & 0.000 '+r'\\']
    for key, label in [('learned','Learned'),('rule','Tariff rule'),('reference','Reference only')]:
        row = fine['policies'][key]; regret = row['regret']
        rows.append(f"{label} & {row['initial_costs'][1]:.2f} & {row['gaps'][1]:.2f} & {regret['boundary_mean']:.3f} & {regret['boundary_p95']:.3f} "+r'\\')
    rows += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    main = [
        'The hard-tariff two-state check uses deterministic price, winter profiles,',
        '15-minute held actions, and the common terminal cost. Coarse-grid expected',
        'cost selects checkpoints from five training seeds and the tariff-rule setting.',
        'All policies use the same refined numerical transition, with the learned',
        'continuation interpolated on that grid.',
        f'Refining from 81 to 161 SoC points and 162 to 322 load-error points',
        f'changes the initial DP cost by {change:.2f} EUR/day.',
        '',
        'Using the DP-optimal continuation, define one-step decision regret by',
        r'\begin{align}',
        r'Q_k^{\mathrm{DP}}(x,a)&=c_k(x,a)+(\widehat{\mathcal P}_{h,k}^{a}V_{k+1}^{\mathrm{DP}})(x), \label{eq:dp-regret-q}\\',
        r'\mathcal R_k^\pi(x)&=Q_k^{\mathrm{DP}}(x,\pi_k(x))-V_k^{\mathrm{DP}}(x). \label{eq:dp-regret}',
        r'\end{align}',
        r'Here $V_k^{\mathrm{DP}}(x)=\min_{a\in\cA_{\mathrm{dep}}(s)}Q_k^{\mathrm{DP}}(x,a)$,',
        r'$c_k$ is the hard stage cost, and $\widehat{\mathcal P}_{h,k}^{a}$ is the',
        'DP transition and interpolation operator; the final step evaluates',
        r'$V_K^{\mathrm{DP}}=\Phi$ exactly. Thus $\mathcal R_k^\pi$ measures the cost',
        'of the current action followed by DP-optimal continuation, not the',
        'greedy-search defect relative to the learned value.',
        *rows,
        f"The selected learned policy is {100*fine['policies']['learned']['gaps'][1]/opt:.2f}\\% above DP,",
        f"compared with {100*fine['policies']['rule']['gaps'][1]/opt:.2f}\\% for the tariff-aware rule.",
        'The supplement reports the selection results, grid refinement, and a',
        'direct-neural-search check of the interpolation used here.', '']
    supp = [r'\section{Hard-tariff reduced-model DP comparison}\label{sec:hard-dp-details}',
        'The central winter model fixes price error to zero and retains the hard',
        '40 EUR/h fee above 300 kW, the 24-hour horizon, and 15-minute held actions.',
        'No historical outcomes enter this comparison. The five training seeds',
        'are 0--4, each with four rounds of at most 2,000 Adam steps and 80 L-BFGS',
        'iterations, using the main architecture with the price coordinate removed.',
        '',
        'The DP integrates reflected Gaussian OU transitions against linear',
        'load-grid basis functions and interpolates continuation values in SoC.',
        'Zero load error is added to each uniform load grid. Candidate actions',
        'include feasible endpoints, zero, zero import, verified fee-crossing sides,',
        'uniform action grids of 129, 257, and 513 points, and actions reaching',
        'next-period SoC grid knots. These knots capture the remaining kinks of',
        'the interpolated objective. The final step minimizes the piecewise',
        'quadratic terminal-stage objective directly; the terminal penalty is not',
        'interpolated. Every policy uses this same numerical transition for cost',
        'and regret evaluation. Refinement checks numerical stability rather than',
        'providing a global certificate.',
        '',
        'Selection uses expected cost from $(0.5,0)$ on the coarsest grid.',
        'The sequential 0.10 EUR/day rule first selects a checkpoint per seed,',
        'then a seed against the zero-action fallback. The tariff rule selects',
        'among the same 96 settings and zero-action candidate used in the',
        'historical study, but using this model-based criterion.',
        f"Seed {selection['selected']['seed']}, round {selection['selected']['iteration']+1}, is selected.",
        'The reference-only policy minimizes hard current cost plus the',
        'one-step terminal term. Neural action selection uses the tabulated',
        'learned continuation and the common transition operator.',
        '',r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Grid refinement from $(S,Y)=(0.5,0)$. Entries are expected cost in EUR/day. Policy selection is held fixed after the first grid.}',
        r'\begin{tabular}{rrrrrr}',r'\toprule',r'SoC & Load error & DP & Learned & Rule & Reference \\',r'\midrule']
    for g in summary['grids']:
        costs = [g['optimal_initial_costs'][1]]+[g['policies'][k]['initial_costs'][1] for k in ('learned','rule','reference')]
        supp.append(f"{g['grid'][0]} & {g['actual_y_points']} & "+' & '.join(f'{v:.2f}' for v in costs)+r' \\')
    supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
        r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Model-based checkpoint selection: expected cost in EUR/day on the coarsest grid. Rounds are one-based.}',
        r'\begin{tabular}{rrrrrr}',r'\toprule',r'Seed & Round 1 & Round 2 & Round 3 & Round 4 & Selected \\',r'\midrule']
    for seed in range(5):
        costs = [r['initial_costs'][1] for r in selection['rounds'] if r['seed']==seed]
        winner = next(r for r in selection['winners'] if r['seed']==seed)
        supp.append(str(seed)+' & '+' & '.join(f'{v:.2f}' for v in costs)+f" & {winner['iteration']+1}"+r' \\')
    supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
        r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Initial-SoC sensitivity on the finest grid, with initial load error zero. Costs are EUR/day.}',
        r'\begin{tabular}{rrrrr}',r'\toprule',r'Initial SoC & DP & Learned & Rule & Reference \\',r'\midrule']
    for i,s in enumerate((.2,.5,.8)):
        costs = [fine['optimal_initial_costs'][i]]+[fine['policies'][k]['initial_costs'][i] for k in ('learned','rule','reference')]
        supp.append(f'{s:.1f} & '+' & '.join(f'{v:.2f}' for v in costs)+r' \\')
    supp += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
        'Boundary regret uses the DP-optimal continuation for every policy,',
        'rather than the learned continuation or the evaluated policy value.',
        'It is evaluated over the grid states where DP leaves import',
        f"within 10 kW of the threshold ({fine['policies']['learned']['regret']['boundary_states']:,} states on the finest grid).",
        'The averages weight those states equally, not by a policy occupancy measure.',
        'A separate check uses 24 boundary-focused states: hours 4, 10, 16, and 22; SoC',
        '0.2, 0.5, and 0.8; and load-grid points nearest imports of 300 and 400 kW.',
        'Another 64 grid states are sampled uniformly with seed 9287.',
        'Direct neural search uses 1,025 actions plus crossing candidates and',
        'three- or seven-node quadrature per disturbance dimension.',
        f"The largest one-step objective loss of the grid-selected action was {audit['max_grid_regret_q3']:.4f} EUR",
        f"against the production quadrature search and {audit['max_grid_regret_q7']:.4f} EUR against the higher-order search.",
        'This finite check measures the numerical change from tabulating the',
        'continuation value; it does not certify a uniform error bound.', '']
    return '\n'.join(main), '\n'.join(supp)


def main():
    out=HERE/'results'
    summary=json.loads((out/'summary.json').read_text())
    selection=json.loads((out/'learned_selection.json').read_text())
    audit=json.loads((out/'deployment_audit.json').read_text())
    main_tex,supp_tex=render(summary,selection,audit)
    (ROOT/'manuscript').mkdir(parents=True, exist_ok=True)
    (ROOT/'manuscript/hard_tariff_dp_results.tex').write_text(main_tex,encoding='utf-8')
    (ROOT/'manuscript/hard_tariff_dp_details.tex').write_text(supp_tex,encoding='utf-8')


if __name__=='__main__': main()
