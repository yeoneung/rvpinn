"""Analyze learned-minus-reference attribution comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
sys.path.insert(0, str(HERE))

from analyze_confirmatory import (hierarchical_paired_stats, holm_adjust,
                                  paired_stats)  # noqa: E402
from result_validation import read_aligned_parquet

LEARNED = {
    20: "learned_daily_v3_winter_fee20.parquet",
    40: "learned_daily_v3_winter_fee40.parquet",
    80: "learned_daily_v3_winter_fee80.parquet",
}
COMPONENTS = (
    "common_cost", "bill", "capacity_fee", "degradation_cost",
    "terminal_penalty", "throughput_kwh", "exceed_hours",
    "terminal_soc_dev", "latency_mean_ms", "latency_p95_ms",
)


def _write_latex(methods: pd.DataFrame, pairs: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[tbp]",
        r"\centering",
        r"\caption{Exploratory attribution to the learned continuation correction. Costs and bounds are in EUR/day and fees in EUR/h. The reference retains the same hard one-step search without the neural correction. Differences are learned minus reference; the one-sided 95\% upper bound resamples five batches at 40 EUR/h and is conditional on one batch at the outer fees.}",
        r"\label{tab:reference-attribution}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rrrrrl}",
        r"\toprule",
        r"Fee & Reference & Learned & Difference & Upper bound & Superiority \\",
        r"\midrule",
    ]
    for fee in (20, 40, 80):
        ref = methods[(methods["fee"] == fee)
                      & (methods["method"] == "reference_value_one_step")]
        learned = methods[(methods["fee"] == fee)
                          & (methods["method"] == "offline_value_policy")]
        pair = pairs[pairs["fee"] == fee].iloc[0]
        conclusion = "established" if bool(pair["superiority"]) else "not established"
        lines.append(
            f"{fee} & {ref.iloc[0]['mean_common_cost']:.1f} & "
            f"{learned.iloc[0]['mean_common_cost']:.1f} & "
            f"{pair['mean_diff']:.1f} ({pair['percent_diff']:.1f}\\%) & "
            f"{pair['one_sided_95_upper']:.1f} & {conclusion} \\\\")
    low = methods[methods["fee"] == 20].set_index("method")
    medium = methods[methods["fee"] == 40].set_index("method")
    high = methods[methods["fee"] == 80].set_index("method")

    def component_difference(frame, name):
        return (frame.loc["offline_value_policy", f"mean_{name}"]
                - frame.loc["reference_value_one_step", f"mean_{name}"])

    attribution = []
    for pair in pairs.sort_values('fee').itertuples():
        outcome = ('established' if pair.superiority else 'not established')
        attribution.append(
            f'At {int(pair.fee)} EUR/h, the learned-minus-reference cost '
            f'difference was {pair.mean_diff:.1f} EUR/day, with a one-sided '
            f'upper bound of {pair.one_sided_95_upper:.1f} EUR/day; '
            f'superiority was {outcome}.')
    supported = pairs.loc[pairs['superiority'], 'fee'].astype(int).tolist()
    if len(supported) == 3:
        overview = (
            r'The learned continuation reduced cost relative to the reference-only '
            r'controller at all three anchors, with one-sided upper bounds below zero '
            r'(Table~\ref{tab:reference-attribution}).')
    elif not supported:
        overview = (
            r'The learned-superiority rule was not met at any reference-comparison '
            r'anchor (Table~\ref{tab:reference-attribution}).')
    else:
        overview = (
            'The learned-superiority rule was met at fee rates of '
            + ', '.join(map(str, supported))
            + r' EUR/h in the reference comparison (Table~\ref{tab:reference-attribution}).')

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
        overview,
        (f"At 40 EUR/h, the learned correction changed the transaction bill "
         f"by {component_difference(medium, 'bill'):.1f} EUR/day and "
         f"degradation by {component_difference(medium, 'degradation_cost'):.1f} "
         f"EUR/day, while changing the band fee by "
         f"{component_difference(medium, 'capacity_fee'):.1f} EUR/day. "
         "The supplement gives all cost components. Mean controller-call latency across "
         f"fees was {methods[methods['method'] == 'reference_value_one_step']['mean_latency_mean_ms'].mean():.2f} "
         "ms for the reference controller and "
         f"{methods[methods['method'] == 'offline_value_policy']['mean_latency_mean_ms'].mean():.2f} "
         "ms for the learned controller."),
        "",
    ])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "round2_reference.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    supplement = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Learned-correction attribution by cost component. Fees are in EUR/h, cost components in EUR/day, throughput in kWh/day, and exceedance in hours/day. The central learned row averages five complete training batches.}",
        r"\label{tab:supp-reference-components}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rlrrrrrr}",
        r"\toprule",
        r"Fee & Controller & Bill & Band fee & Degradation & Terminal & Throughput & Exceed h \\",
        r"\midrule",
    ]
    labels = {"reference_value_one_step": "No-learning reference",
              "offline_value_policy": "Offline learned"}
    for fee in (20, 40, 80):
        for method in ("reference_value_one_step", "offline_value_policy"):
            row = methods[(methods["fee"] == fee)
                          & (methods["method"] == method)].iloc[0]
            supplement.append(
                f"{fee} & {labels[method]} & {row['mean_bill']:.1f} & "
                f"{row['mean_capacity_fee']:.1f} & "
                f"{row['mean_degradation_cost']:.1f} & "
                f"{row['mean_terminal_penalty']:.2f} & "
                f"{row['mean_throughput_kwh']:.0f} & "
                f"{row['mean_exceed_hours']:.2f} \\\\")
    supplement.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
        ' '.join(attribution),
        "",
    ])
    (GENERATED / "round2_reference_supplement.tex").write_text(
        "\n".join(supplement), encoding="utf-8")


def main() -> int:
    reference = read_aligned_parquet(RESULTS / "round2_reference_daily.parquet")
    if len(reference) != 90:
        raise RuntimeError(f"expected 90 reference rows, found {len(reference)}")
    method_rows, pair_rows, batch_rows = [], [], []
    for fee in (20, 40, 80):
        ref = reference[reference["fee_per_hour"] == fee].copy()
        learned_all = read_aligned_parquet(RESULTS / LEARNED[fee])
        chosen = learned_all[learned_all["selected_by_batch"]].copy()
        if len(ref) != 30 or len(chosen) != 30 * chosen["batch"].nunique():
            raise RuntimeError(f"incomplete fee {fee} attribution inputs")

        learned_daily = chosen.groupby("date", as_index=False)[
            list(COMPONENTS)].mean()
        for method, frame in (("reference_value_one_step", ref),
                              ("offline_value_policy", learned_daily)):
            method_rows.append({
                "fee": fee, "method": method, "n_days": 30,
                "n_training_batches": (chosen["batch"].nunique()
                                       if method == "offline_value_policy"
                                       else 0),
                **{f"mean_{name}": float(frame[name].mean())
                   for name in COMPONENTS},
            })
        for batch, sub in chosen.groupby("batch"):
            joined = sub[["date", "common_cost"]].merge(
                ref[["date", "common_cost"]], on="date",
                suffixes=("_learned", "_reference"), validate="one_to_one")
            batch_rows.append({
                "fee": fee, "batch": int(batch), "n_days": len(joined),
                "mean_learned_minus_reference": float(
                    (joined["common_cost_learned"]
                     - joined["common_cost_reference"]).mean()),
                "mean_learned_cost": float(
                    joined["common_cost_learned"].mean()),
                "mean_reference_cost": float(
                    joined["common_cost_reference"].mean()),
            })
        stats = (hierarchical_paired_stats(chosen, ref, 8400 + fee)
                 if chosen["batch"].nunique() > 1
                 else paired_stats(learned_daily, ref, 8400 + fee))
        pair_rows.append({"fee": fee, "comparator":
                          "reference_value_one_step", **stats})

    methods = pd.DataFrame(method_rows)
    pairs = pd.DataFrame(pair_rows)
    pairs["holm_p"] = holm_adjust(pairs["one_sided_p"].to_numpy())
    batches = pd.DataFrame(batch_rows)
    methods.to_csv(RESULTS / "round2_reference_method_summary.csv",
                   index=False)
    pairs.to_csv(RESULTS / "round2_reference_pairwise.csv", index=False)
    batches.to_csv(RESULTS / "round2_reference_batch_summary.csv",
                   index=False)
    _write_latex(methods, pairs)

    summary = {
        "label": "exploratory learned-continuation attribution analysis",
        "fees": [20, 40, 80],
        "n_days_per_fee": 30,
        "central_training_batches": 5,
        "learned_superior_fees": [
            int(row.fee) for row in pairs.itertuples() if row.superiority],
    }
    (RESULTS / "round2_reference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(pairs.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
