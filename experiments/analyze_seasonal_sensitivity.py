"""Generate seasonal and factorial mechanism summaries from frozen outputs.

The winter anchors are confirmatory. Seasons and the 3x3x3 grid are
descriptive mechanism analyses; the noncentral factorial cells use one
restart and therefore do not support cellwise significance claims.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from result_validation import (finite_solver_gaps, format_gap_percent, read_aligned_parquet,
                               require_paired_dates)
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)

SEASONS = {
    "winter_weekday": "Winter",
    "spring_weekday": "Spring",
    "summer_weekday": "Summer",
    "autumn_weekday": "Autumn",
}
SEASON_TAGS = {
    "winter_weekday": "v3_winter_fee40",
    "spring_weekday": "v3_spring_fee40",
    "summer_weekday": "v3_summer_fee40",
    "autumn_weekday": "v3_autumn_fee40",
}
METHODS = (
    "no_storage",
    "validation_tuned_price_rule",
    "convex_envelope_mpc",
    "deterministic_exact_band_miqp",
    "stochastic_two_stage_exact_band_miqp",
)
DISPLAY = {
    "no_storage": "No storage",
    "validation_tuned_price_rule": "Validation-tuned price rule",
    "convex_envelope_mpc": "Convex-envelope MPC",
    "deterministic_exact_band_miqp": "Conditional-mean MIQP",
    "stochastic_two_stage_exact_band_miqp": "Two-stage stochastic MIQP",
    "offline_value_policy": "Offline procedure",
}
COMPACT_SOLVER_DISPLAY = {
    "convex_envelope_mpc": "Envelope MPC",
    "deterministic_exact_band_miqp": "CM MIQP",
    "stochastic_two_stage_exact_band_miqp": "Stoch. MIQP",
}


def learned_selected(tag: str) -> pd.DataFrame:
    path = RESULTS / f"learned_daily_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = read_aligned_parquet(path)
    chosen = frame[frame["selected_by_batch"]].copy()
    for _, batch in chosen.groupby('batch'):
        require_paired_dates(batch, batch)
    # If there are independent batches, average them by date. This estimates
    # the mean result of the complete five-restart procedure without test-set
    # selection.
    numeric = [c for c in chosen.columns
               if c != "date" and pd.api.types.is_numeric_dtype(chosen[c])]
    return chosen.groupby("date", as_index=False)[numeric].mean()


def comparator_file(threshold: int, fee: int) -> Path:
    return RESULTS / (
        f"common_daily_confirmatory_thr{threshold}_fee{fee}_w10_n2.parquet")


def fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


def factorial_cells():
    """Load the complete 27-cell design independently of seasonal summaries."""
    cells = []
    for threshold in (250, 300, 350):
        for fee in (20, 40, 80):
            comp = read_aligned_parquet(comparator_file(threshold, fee))
            comp = comp[comp["regime"] == "winter_weekday"]
            det = comp[comp["method"] == "deterministic_exact_band_miqp"]
            zero = comp[comp["method"] == "no_storage"]
            if len(det) != 30 or len(zero) != 30:
                raise RuntimeError(f"incomplete sensitivity comparators: {threshold}/{fee}")
            comp_dates = det[["date", "common_cost"]].rename(
                columns={"common_cost": "deterministic_cost"})
            zero_dates = zero[["date", "common_cost"]].rename(
                columns={"common_cost": "no_storage_cost"})
            for width in (5, 10, 20):
                if threshold == 300 and width == 10:
                    learned = learned_selected(f"v3_winter_fee{fee}")
                    selected_batches = 5 if fee == 40 else 1
                    n_restarts = 25 if fee == 40 else 5
                else:
                    tag = f"v3_sens_t{threshold}_f{fee}_w{width}"
                    learned = read_aligned_parquet(RESULTS / f"selected_daily_{tag}.parquet")
                    selected_batches, n_restarts = 1, 1
                require_paired_dates(learned, det)
                require_paired_dates(learned, zero)
                paired = (learned[["date", "common_cost"]]
                          .merge(comp_dates, on="date", validate="one_to_one")
                          .merge(zero_dates, on="date", validate="one_to_one"))
                near = {}
                near["mean_exact_threshold_hours"] = (
                    float(learned["threshold_exact_hours"].mean())
                    if "threshold_exact_hours" in learned.columns else np.nan)
                for radius in (5, 10, 20):
                    name = f"threshold_near_{radius}kw_hours"
                    near[f"mean_near_{radius}kw_hours"] = (
                        float(learned[name].mean()) if name in learned.columns else np.nan)
                cells.append({
                    "threshold_kw": threshold, "fee_per_hour": fee,
                    "smoothing_width_kw": width, "n_training_restarts": n_restarts,
                    "n_selected_batches": selected_batches, "n_days": len(paired),
                    "mean_learned_cost": float(paired["common_cost"].mean()),
                    "mean_deterministic_cost": float(paired["deterministic_cost"].mean()),
                    "mean_no_storage_cost": float(paired["no_storage_cost"].mean()),
                    "learned_minus_deterministic": float(
                        (paired["common_cost"] - paired["deterministic_cost"]).mean()),
                    "learned_minus_no_storage": float(
                        (paired["common_cost"] - paired["no_storage_cost"]).mean()),
                    "mean_terminal_penalty": float(learned["terminal_penalty"].mean()),
                    "mean_smooth_minus_hard_cost": float(
                        (learned["objective"] - learned["common_cost"]).mean()),
                    **near,
                })
    return pd.DataFrame(cells)


def write_factorial(sensitivity):
    """Keep descriptive factorial results separate from seasonal/budget output."""
    keys = ["threshold_kw", "fee_per_hour", "smoothing_width_kw"]
    expected = {(threshold, fee, width) for threshold in (250, 300, 350)
                for fee in (20, 40, 80) for width in (5, 10, 20)}
    if (len(sensitivity) != 27 or sensitivity.duplicated(keys).any()
            or set(map(tuple, sensitivity[keys].to_numpy())) != expected
            or not np.isfinite(sensitivity.drop(columns=keys).to_numpy(dtype=float)).all()):
        raise RuntimeError("factorial reporting requires all 27 finite, unique cells")
    sensitivity.to_csv(RESULTS / "sensitivity_cells.csv", index=False)
    wins_det = int((sensitivity["learned_minus_deterministic"] < 0).sum())
    wins_zero = int((sensitivity["learned_minus_no_storage"] < 0).sum())
    grouped_width = sensitivity.groupby("smoothing_width_kw")["mean_learned_cost"].mean()
    near10 = sensitivity["mean_near_10kw_hours"]
    exact = sensitivity["mean_exact_threshold_hours"]
    smooth_hard = sensitivity["mean_smooth_minus_hard_cost"]
    text = (
        f"Across the 27 descriptive factorial cells, the offline policy had "
        f"lower mean cost than conditional-mean MIQP in {wins_det} cells and "
        f"lower mean cost than no storage in {wins_zero} cells. "
        f"The three anchors reuse complete five-restart procedures; the other "
        f"24 cells use one restart. Width contrasts therefore also reflect "
        f"unequal selection effort, and the cell counts are descriptive."
        f" Realized grid exchange lay within 10 kW of the threshold for "
        f"{near10.min():.2f}--{near10.max():.2f} hours/day across cells; "
        f"exact-threshold occupation within $10^{{-6}}$ kW ranged from "
        f"{exact.min():.2f} to {exact.max():.2f} hours/day. The "
        f"corresponding mean smooth-minus-hard realized-cost difference "
        f"ranged from {smooth_hard.min():.1f} to {smooth_hard.max():.1f} EUR/day.")
    figure = "\n".join([
        r"\begin{figure}[t]", r"\centering",
        r"\includegraphics[width=\linewidth]{figures/fig_factorial.pdf}",
        (r"\caption{Descriptive threshold--fee--smoothing results. Entries "
         r"are mean offline-minus-conditional-mean-MIQP common costs in "
         r"EUR/day; negative values favor the offline policy. The three "
         r"anchors reuse the primary procedures; other cells use one restart.}"),
        r"\label{fig:factorial}", r"\end{figure}"])
    (GENERATED / "factorial_results.tex").write_text(text + "\n\n" + figure, encoding="utf-8")
    lines = [r"\begin{table}[ht]", r"\centering", r"\scriptsize",
        (r"\caption{Threshold--fee--smoothing factorial. $\Delta$ is offline "
         r"minus conditional-mean MIQP cost. Threshold and width are in kW, "
         r"fee in EUR/h, and costs in EUR/day. Near 10 is the number of "
         r"hours/day within 10 kW of the threshold.}"),
        r"\label{tab:supp-factorial}", r"\begin{tabular}{rrrrrrrr}", r"\toprule",
        r"Threshold & Fee & Width & Offline cost & $\Delta$ & Smooth--hard & Near 10 & Restarts \\",
        r"\midrule"]
    for row in sensitivity.itertuples():
        lines.append(
            f"{int(row.threshold_kw)} & {int(row.fee_per_hour)} & "
            f"{int(row.smoothing_width_kw)} & {fmt(row.mean_learned_cost)} & "
            f"{fmt(row.learned_minus_deterministic)} & "
            f"{fmt(row.mean_smooth_minus_hard_cost)} & "
            f"{fmt(row.mean_near_10kw_hours, 2)} & {int(row.n_training_restarts)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (GENERATED / "factorial_supplement.tex").write_text("\n".join(lines), encoding="utf-8")
    return {
        "factorial_cells": len(sensitivity),
        "offline_better_than_deterministic_cells": wins_det,
        "offline_better_than_no_storage_cells": wins_zero,
        "best_mean_smoothing_width_kw": int(grouped_width.idxmin()),
        "width_mean_cost_spread_eur_per_day": float(grouped_width.max() - grouped_width.min()),
        "width_summary_scope": "descriptive pooled means; unequal restart effort, not an isolated width effect",
    }


def main() -> int:
    br = r"\\"
    seasonal_rows = []
    central = read_aligned_parquet(comparator_file(300, 40))
    for regime, tag in SEASON_TAGS.items():
        learned = learned_selected(tag)
        batch_audit = json.loads((
            RESULTS / f"restart_batches_{tag}.json"
        ).read_text(encoding="utf-8"))
        n_batches = len(batch_audit["batch_selections"])
        n_fallback = sum(
            entry.get("selected_kind") == "zero_action_fallback"
            for entry in batch_audit["batch_selections"])
        seasonal_rows.append({
            "regime": regime,
            "method": "offline_value_policy",
            "n_days": len(learned),
            "n_selected_batches": n_batches,
            "n_fallback_batches": n_fallback,
            "mean_common_cost": float(learned["common_cost"].mean()),
            "mean_terminal_penalty": float(
                learned["terminal_penalty"].mean()),
            "mean_latency_ms": float(learned["latency_mean_ms"].mean()),
        })
        for method in METHODS:
            sub = central[(central["regime"] == regime)
                          & (central["method"] == method)].copy()
            require_paired_dates(learned, sub)
            seasonal_rows.append({
                "regime": regime,
                "method": method,
                "n_days": len(sub),
                "n_selected_batches": np.nan,
                "n_fallback_batches": np.nan,
                "mean_common_cost": float(sub["common_cost"].mean()),
                "mean_terminal_penalty": float(
                    sub["terminal_penalty"].mean()),
                "mean_latency_ms": float(sub["latency_mean_ms"].mean()),
            })
    seasonal = pd.DataFrame(seasonal_rows)
    seasonal.to_csv(RESULTS / "seasonal_method_summary.csv", index=False)

    # Descriptive solver-budget sensitivity: the two high-fee exact-band
    # formulations receive 15 rather than 5 seconds at every reoptimization.
    # Separate files preserve both budgets in the reported comparison.
    base_budget_daily = read_aligned_parquet(comparator_file(300, 80))
    tight_daily_path = RESULTS / (
        "common_daily_confirmatory_thr300_fee80_w10_n2_tl15.parquet")
    base_budget_solves = read_aligned_parquet(
        RESULTS / "common_solves_confirmatory_thr300_fee80_w10_n2.parquet")
    tight_solves_path = RESULTS / (
        "common_solves_confirmatory_thr300_fee80_w10_n2_tl15.parquet")
    if not tight_daily_path.exists() or not tight_solves_path.exists():
        raise FileNotFoundError("15-second high-fee solver-budget check")
    tight_budget_daily = read_aligned_parquet(tight_daily_path)
    tight_budget_solves = read_aligned_parquet(tight_solves_path)
    budget_rows = []
    for method in ("deterministic_exact_band_miqp",
                   "stochastic_two_stage_exact_band_miqp"):
        for seconds, daily_frame, solve_frame in (
                (5, base_budget_daily, base_budget_solves),
                (15, tight_budget_daily, tight_budget_solves)):
            daily_sub = daily_frame[
                (daily_frame["regime"] == "winter_weekday")
                & (daily_frame["method"] == method)]
            solve_sub = solve_frame[
                (solve_frame["regime"] == "winter_weekday")
                & (solve_frame["method"] == method)]
            if len(daily_sub) != 30 or len(solve_sub) != 30 * 96:
                raise RuntimeError(
                    f"incomplete solver-budget check: {method}/{seconds}s")
            finite_gap = finite_solver_gaps(solve_sub)
            budget_rows.append({
                "method": method, "time_limit_s": seconds,
                "n_days": len(daily_sub),
                "mean_common_cost": float(daily_sub["common_cost"].mean()),
                "mean_latency_ms": float(daily_sub["latency_mean_ms"].mean()),
                "incumbent_fraction": float(solve_sub["has_solution"].mean()),
                "finite_gap_fraction": float(len(finite_gap) / len(solve_sub)),
                "mean_gap": float(finite_gap.mean()),
                "time_limit_fraction": float(
                    (solve_sub["status"] == "timelimit").mean()),
                "p95_solve_s": float(solve_sub["solve_s"].quantile(0.95)),
            })
    budget = pd.DataFrame(budget_rows)
    base_cost = budget[budget["time_limit_s"] == 5].set_index("method")[
        "mean_common_cost"]
    budget["cost_change_vs_5s"] = budget.apply(
        lambda row: row["mean_common_cost"] - base_cost.loc[row["method"]],
        axis=1)
    budget.to_csv(RESULTS / "high_fee_solver_budget_summary.csv", index=False)

    sensitivity = factorial_cells()
    summary = write_factorial(sensitivity)

    season_table = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Seasonal replication at 40 EUR/h. Costs are daily means "
         r"in EUR/day. Winter averages five validation-selected batches; "
         r"each other season uses one five-restart batch.}"),
        r"\label{tab:seasons}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        f"Season & No storage & Convex & CM MIQP & Stoch. MIQP & Offline {br}",
        r"\midrule",
    ]
    for regime, label in SEASONS.items():
        sub = seasonal[seasonal["regime"] == regime].set_index("method")
        season_table.append(
            f"{label} & {fmt(sub.loc['no_storage', 'mean_common_cost'])} & "
            f"{fmt(sub.loc['convex_envelope_mpc', 'mean_common_cost'])} & "
            f"{fmt(sub.loc['deterministic_exact_band_miqp', 'mean_common_cost'])} & "
            f"{fmt(sub.loc['stochastic_two_stage_exact_band_miqp', 'mean_common_cost'])} & "
            f"{fmt(sub.loc['offline_value_policy', 'mean_common_cost'])} {br}")
    season_table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    budget_15 = budget[budget["time_limit_s"] == 15].set_index("method")
    budget_sentence = (
        " At the 80 EUR/h anchor, increasing the per-solve limit from 5 to "
        "15 seconds changed mean daily cost by "
        f"{budget_15.loc['deterministic_exact_band_miqp', 'cost_change_vs_5s']:.1f} "
        "EUR for conditional-mean MIQP and "
        f"{budget_15.loc['stochastic_two_stage_exact_band_miqp', 'cost_change_vs_5s']:.1f} "
        "EUR for two-stage stochastic MIQP; this descriptive sensitivity "
        "analysis is separate from the primary comparison.")
    seasonal_offline = seasonal[
        seasonal["method"] == "offline_value_policy"]
    seasonal_fallback_sentence = (
        " Across the four seasonal regimes, zero action was retained in "
        f"{int(seasonal_offline['n_fallback_batches'].sum())} of "
        f"{int(seasonal_offline['n_selected_batches'].sum())} complete "
        "five-restart batches.")
    (GENERATED / "mechanism_results.tex").write_text(
        "\n".join(season_table + ["", budget_sentence + seasonal_fallback_sentence]),
        encoding="utf-8")

    restart = pd.read_csv(RESULTS / "confirmatory_restart_summary.csv")
    selection = pd.read_csv(RESULTS / "confirmatory_selection_summary.csv")
    solver = pd.read_csv(RESULTS / "confirmatory_solver_summary.csv")
    offline = pd.read_csv(RESULTS / "confirmatory_offline_compute.csv")
    amortization = pd.read_csv(RESULTS / "confirmatory_amortization.csv")
    components = pd.read_csv(RESULTS / "confirmatory_method_summary.csv")
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Winter common-objective decomposition. Cost components are in EUR/day and fee rates in EUR/h.}",
        r"\label{tab:supp-components}",
        r"\begin{tabularx}{\linewidth}{r>{\raggedright\arraybackslash}Xrrrrr}",
        r"\toprule",
        f"Fee & Method & Bill & Band fee & Degradation & Terminal & Total {br}",
        r"\midrule",
    ]
    for fee in (20, 40, 80):
        sub = components[components["fee"] == fee]
        for i, method in enumerate((*METHODS, "offline_value_policy")):
            row = sub[sub["method"] == method].iloc[0]
            lines.append(
                f"{fee if i == 0 else ''} & {DISPLAY[method]} & "
                f"{row['mean_bill']:.1f} & {row['mean_capacity_fee']:.1f} & "
                f"{row['mean_degradation_cost']:.1f} & "
                f"{row['mean_terminal_penalty']:.2f} & "
                f"{row['mean_common_cost']:.1f} {br}")
        if fee != 80:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (r"\caption{Descriptive solver-budget sensitivity at 80 EUR/h. Costs and "
         r"cost changes are in EUR/day; latency is mean controller-call time. "
         r"The primary deployment budget is 5 s. "
         r"Finite gap is the availability fraction; mean gaps are conditional on availability.}"),
        r"\label{tab:supp-budget}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        f"Method & Limit & Cost & Cost change & Latency (ms) & Incumbent & Finite gap & Mean gap & Time limit {br}",
        r"\midrule",
    ])
    for method in ("deterministic_exact_band_miqp",
                   "stochastic_two_stage_exact_band_miqp"):
        sub = budget[budget["method"] == method].sort_values("time_limit_s")
        for i, (_, row) in enumerate(sub.iterrows()):
            lines.append(
                f"{DISPLAY[method] if i == 0 else ''} & "
                f"{int(row['time_limit_s'])} s & "
                f"{row['mean_common_cost']:.1f} & "
                f"{row['cost_change_vs_5s']:.1f} & "
                f"{row['mean_latency_ms']:.0f} & "
                f"{100.0 * row['incumbent_fraction']:.1f}\\% & "
                f"{100.0 * row['finite_gap_fraction']:.1f}\\% & "
                f"{format_gap_percent(row['mean_gap'])} & "
                f"{100.0 * row['time_limit_fraction']:.1f}\\% {br}")
        if method == "deterministic_exact_band_miqp":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}",
                  r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Restart and measured offline-computation diagnostics. Fees are in EUR/h; cost ranges and standard deviations are in EUR/day. Offline time is in minutes per five-restart batch.}",
        r"\label{tab:supp-restarts}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        (f"Fee & Restarts & Batches & Fallbacks & Validation range & Test SD & "
         f"Selected test range & Min/batch {br}"),
        r"\midrule",
    ])
    for fee in (20, 40, 80):
        sub = restart[restart["fee"] == fee]
        off = offline[offline["fee"] == fee]
        selected = selection[selection["fee"] == fee]
        vrange = (f"{sub['validation_mean_common_cost'].min():.1f}--"
                  f"{sub['validation_mean_common_cost'].max():.1f}")
        trange = (f"{selected['selected_test_mean_common_cost'].min():.1f}--"
                  f"{selected['selected_test_mean_common_cost'].max():.1f}")
        n_fallback = int(
            (selected["selected_kind"] == "zero_action_fallback").sum())
        lines.append(
            f"{fee} & {len(sub)} & {len(off)} & {n_fallback} & {vrange} & "
            f"{sub['test_mean_common_cost'].std(ddof=1):.1f} & {trange} & "
            f"{off['total_measured_offline_s'].mean() / 60.0:.1f} {br}")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}",
                  r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        (r"\caption{Compute-equivalent amortization against the better of "
         r"the two online MIQPs. Cost $\Delta$ is offline minus comparator in "
         r"EUR/day; fees are in EUR/h. The last column ignores operating-cost differences "
         r"and values offline and online compute seconds equally.}"),
        r"\label{tab:supp-amortization}",
        r"\begin{tabularx}{\linewidth}{r>{\raggedright\arraybackslash}Xrrr}",
        r"\toprule",
        rf"Fee & Online comparator & Cost $\Delta$ & Saved s/day & Break-even days {br}",
        r"\midrule",
    ])
    for _, row in amortization.iterrows():
        lines.append(
            f"{int(row['fee'])} & {DISPLAY[row['online_comparator']]} & "
            f"{row['learned_minus_online_eur_per_day']:.1f} & "
            f"{row['online_compute_saved_s_per_day']:.1f} & "
            f"{row['compute_equivalent_break_even_days']:.1f} {br}")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Winter online-solver diagnostics. Fees are in EUR/h. Gaps are reported only "
        r"for solves returning finite solver bounds; finite gap is their availability fraction. "
        r"Solve time excludes model construction. CM and Stoch. denote conditional-mean "
        r"and two-stage stochastic MIQP; Envelope denotes the convex-tariff-surrogate MPC.}",
        r"\label{tab:supp-solvers}",
        r"\begin{tabularx}{\linewidth}{r>{\raggedright\arraybackslash}Xrrrrr}",
        r"\toprule",
        f"Fee & Method & Solves & Incumbent & Finite gap & Mean gap & P95 solve (s) {br}",
        r"\midrule",
    ])
    for fee in (20, 40, 80):
        sub = solver[solver["fee"] == fee]
        for i, method in enumerate(("convex_envelope_mpc",
                                    "deterministic_exact_band_miqp",
                                    "stochastic_two_stage_exact_band_miqp")):
            row = sub[sub["method"] == method].iloc[0]
            lines.append(
                f"{fee if i == 0 else ''} & {COMPACT_SOLVER_DISPLAY[method]} & "
                f"{int(row['n_solves'])} & "
                f"{100.0 * row['incumbent_fraction']:.1f}\\% & "
                f"{100.0 * row['finite_gap_fraction']:.1f}\\% & "
                f"{format_gap_percent(row['mean_gap'])} & {row['p95_solve_s']:.2f} {br}")
        if fee != 80:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Seasonal terminal and online-latency diagnostics. Terminal penalties are in EUR/day; latency is mean controller-call time.}",
        r"\label{tab:supp-season}",
        r"\begin{tabularx}{\linewidth}{l>{\raggedright\arraybackslash}Xrr}",
        r"\toprule",
        f"Season & Method & Terminal penalty & Mean latency (ms) {br}",
        r"\midrule",
    ])
    for regime, label in SEASONS.items():
        sub = seasonal[seasonal["regime"] == regime]
        for i, method in enumerate((*METHODS, "offline_value_policy")):
            row = sub[sub["method"] == method].iloc[0]
            lines.append(
                f"{label if i == 0 else ''} & {DISPLAY[method]} & "
                f"{fmt(row['mean_terminal_penalty'], 2)} & "
                f"{fmt(row['mean_latency_ms'], 1)} {br}")
        if regime != "autumn_weekday":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    (GENERATED / "supplement_results.tex").write_text(
        "\n".join(lines), encoding="utf-8")

    (RESULTS / "seasonal_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
