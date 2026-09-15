"""Report completed validation-tuned tariff-rule comparisons without replay."""
from pathlib import Path
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_confirmatory import paired_stats, hierarchical_paired_stats

OUT = HERE / "results/heuristic"
GENERATED = ROOT / "manuscript/generated"


def main():
    selections = json.loads((OUT / "selection.json").read_text())
    primary = pd.read_csv(ROOT / "experiments/results/confirmatory_method_summary.csv")
    seasonal = pd.read_csv(ROOT / "experiments/results/seasonal_method_summary.csv")
    rows, pairs, days = [], [], []
    for selected in selections:
        regime, fee = selected["regime"], selected["fee"]
        result = json.loads((OUT / f"test_{regime}_fee{fee}.json").read_text())
        assert result["setting"] == selected["setting"] and result["candidate_id"] == selected["candidate_id"]
        daily = pd.DataFrame(result["days"])
        assert len(daily) == 30 and daily.date.nunique() == 30
        assert daily.soc_violations.sum() == 0
        daily["fee"], daily["method"] = fee, "tariff_peak_shaving"
        days.append(daily)
        if regime == "winter_weekday":
            comparison = primary[primary.fee == fee].set_index("method")
            learned = pd.read_parquet(ROOT / f"experiments/results/learned_daily_study_winter_fee{fee}.parquet")
            learned = learned[learned.selected_by_batch]
            if learned.batch.nunique() > 1:
                stat = hierarchical_paired_stats(learned, daily, seed=19271 + fee)
            else:
                stat = paired_stats(learned, daily, seed=19271 + fee)
            pairs.append(dict(fee=fee, **stat))
        else:
            comparison = seasonal[seasonal.regime == regime].set_index("method")
        learned_cost = float(comparison.loc["offline_value_policy", "mean_common_cost"])
        mip_cost = float(comparison.loc["stochastic_two_stage_exact_band_miqp", "mean_common_cost"])
        validation_seconds = sum(json.loads(path.read_text())["elapsed_s"] for path in
                                 OUT.glob(f"val_{regime}_fee{fee}_candidate*.json"))
        rows.append(dict(regime=regime, fee=fee, rule_cost=float(daily.common_cost.mean()),
                         learned_cost=learned_cost, miqp_cost=mip_cost,
                         learned_minus_rule=learned_cost-float(daily.common_cost.mean()),
                         rule_latency_ms=float(daily.latency_mean_ms.mean()),
                         summed_validation_simulation_s=validation_seconds,
                         rule_efc=float(daily.efc.mean()), rule_exceed_hours=float(daily.exceed_hours.mean()),
                         rule_terminal_penalty=float(daily.terminal_penalty.mean()), **selected["setting"]))
    summary = pd.DataFrame(rows)
    pair = pd.DataFrame(pairs)
    summary.to_csv(OUT / "summary.csv", index=False)
    pair.to_csv(OUT / "paired_comparisons.csv", index=False)
    pd.concat(days, ignore_index=True).to_parquet(OUT / "selected_test_daily.parquet", index=False)
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{Exploratory comparison with a validation-tuned tariff-aware peak-shaving rule on the same 30 winter test days. Costs and differences are in EUR/day. Differences are learned minus rule. Intervals are two-sided 95\% paired date-block intervals; the central fee also resamples the five learned-policy batches.}",
             r"\label{tab:tariff-rule}", r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Fee & Rule & Learned & Difference & Interval \\", r"\midrule"]
    for row in summary[summary.regime == "winter_weekday"].itertuples():
        stat = pair[pair.fee == row.fee].iloc[0]
        lines.append(f"{row.fee} & {row.rule_cost:.1f} & {row.learned_cost:.1f} & {row.learned_minus_rule:.1f} & [{stat.ci95_low:.1f}, {stat.ci95_high:.1f}]" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"The tariff-aware rule had lower mean cost than learning at 40 and 80 EUR/h; learning had lower mean cost at 20 EUR/h (Table~\ref{tab:tariff-rule}).",
              "At the central fee, the rule also had lower mean cost than learning in",
              "each of the other three seasons.", ""]
    (GENERATED / "heuristic_results.tex").write_text("\n".join(lines), encoding="utf-8")
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Tariff-aware rule: selected settings and descriptive test outcomes. Reserve SoC is 0.1 in all selected rules. All allow charging while the band fee is already active and disallow partial shaving that cannot eliminate the fee. Costs are in EUR/day; $C_{\rm ch}$ is in EUR/kWh and latency in ms. Seasonal settings are selected separately on validation data.}",
             r"\label{tab:tariff-rule-details}", r"\scriptsize",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Season & Fee & Target SoC & $C_{\rm ch}$ & Rule cost & Learned cost & EFC/day & Latency \\", r"\midrule"]
    for r in summary.itertuples():
        lines.append(f"{r.regime.split('_')[0].capitalize()} & {r.fee} & {r.charge_target_soc:.1f} & {r.charge_price:.4f} & {r.rule_cost:.1f} & {r.learned_cost:.1f} & {r.rule_efc:.2f} & {r.rule_latency_ms:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    central = summary[(summary.regime == "winter_weekday") & (summary.fee == 40)].iloc[0]
    lines += [f"The central-winter validation grid used {central.summed_validation_simulation_s:.1f} s",
              "summed over the 97 candidate simulations, each covering 30 validation days.",
              "These are sums of per-job wall times from four one-thread workers, not",
              "end-to-end elapsed time; they exclude input preparation and worker startup.",
              "", ""]
    (GENERATED / "heuristic_supplement.tex").write_text("\n".join(lines), encoding="utf-8")
    (GENERATED / "heuristic_abstract.tex").write_text(
        "A validation-tuned tariff-aware rule also achieved lower mean cost than learning at the two higher fee anchors and across the central-fee seasonal comparisons.\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(pair.to_string(index=False))


if __name__ == "__main__":
    main()
