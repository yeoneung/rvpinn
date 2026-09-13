"""Analyze nested scenario-count sensitivity and solver diagnostics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from result_validation import finite_solver_gaps, format_gap_percent, read_aligned_parquet
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
COUNTS = (2, 8, 16)
STREAMS = (7301, 7302, 7303)
SETTINGS = ((40, 2, 5), (40, 8, 5), (40, 16, 5), (80, 16, 5), (80, 16, 15))
LEARNED = {
    40: "learned_daily_v3_winter_fee40.parquet",
    80: "learned_daily_v3_winter_fee80.parquet",
}
COMPONENTS = (
    "common_cost", "bill", "capacity_fee", "degradation_cost",
    "terminal_penalty", "throughput_kwh", "exceed_hours",
    "terminal_soc_dev", "latency_mean_ms", "latency_p95_ms",
)


def paths(fee: int, count: int, stream: int, budget: int = 5):
    suffix = f"r2_stream{stream}_pool16_tl{budget}"
    tag = f"confirmatory_thr300_fee{fee}_w10_n{count}_{suffix}"
    return (RESULTS / f"common_daily_{tag}.parquet",
            RESULTS / f"common_solves_{tag}.parquet")


def resample_learned_minus_scenario(learned, comparator, seed,
                                    n_boot=10000, block=7):
    """Resample paired dates, training batches, and scenario streams."""
    lmat = learned.pivot(index="date", columns="batch",
                         values="common_cost").sort_index()
    cmat = comparator.pivot(index="date", columns="stream",
                            values="common_cost").sort_index()
    if not lmat.index.equals(cmat.index):
        raise RuntimeError("learned and scenario dates do not match")
    if lmat.isna().any().any() or cmat.isna().any().any():
        raise RuntimeError("incomplete date-by-randomization matrix")
    lv, cv = lmat.to_numpy(), cmat.to_numpy()
    n_days, n_batches = lv.shape
    n_streams = cv.shape[1]
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    n_blocks = int(np.ceil(n_days / block))

    def sample(center=0.0):
        out = np.empty(n_boot)
        for b in range(n_boot):
            starts = rng.integers(0, n_days, size=n_blocks)
            dates = np.concatenate(
                [(start + offsets) % n_days for start in starts])[:n_days]
            batches = rng.integers(0, n_batches, size=n_batches)
            streams = rng.integers(0, n_streams, size=n_streams)
            difference = (lv[np.ix_(dates, batches)].mean()
                          - cv[np.ix_(dates, streams)].mean())
            out[b] = difference - center
        return out

    point = float(lv.mean() - cv.mean())
    boot = sample()
    null_boot = sample(center=point)
    return {
        "n_days": n_days,
        "n_training_batches": n_batches,
        "n_scenario_streams": n_streams,
        "mean_diff": point,
        "mean_learned": float(lv.mean()),
        "mean_comparator": float(cv.mean()),
        "percent_diff": float(100.0 * point / cv.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "one_sided_p": float((1 + np.sum(null_boot <= point))
                              / (1 + n_boot)),
        "superiority": bool(np.quantile(boot, 0.95) < 0.0),
    }


def load_setting(fee: int, count: int, budget: int = 5):
    daily_parts, solve_parts = [], []
    for stream in STREAMS:
        daily_path, solve_path = paths(fee, count, stream, budget)
        if not daily_path.exists() or not solve_path.exists():
            raise FileNotFoundError(
                f"missing fee={fee} M={count} stream={stream} budget={budget}")
        daily = read_aligned_parquet(daily_path)
        solves = read_aligned_parquet(solve_path)
        daily["stream"] = stream
        solves["stream"] = stream
        if len(daily) != 30 or daily["date"].nunique() != 30:
            raise RuntimeError(f"incomplete daily file: {daily_path}")
        if len(solves) != 30 * 96:
            raise RuntimeError(f"incomplete solve file: {solve_path}")
        daily_parts.append(daily)
        solve_parts.append(solves)
    return (pd.concat(daily_parts, ignore_index=True),
            pd.concat(solve_parts, ignore_index=True))


def _write_latex(settings, streams, solvers, pairs) -> None:
    merged = (settings.merge(solvers, on=["fee", "n_scenarios",
                                          "time_limit_s"],
                             suffixes=("", "_solver"))
              .merge(pairs, on=["fee", "n_scenarios", "time_limit_s"],
                     suffixes=("", "_pair")))
    main_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Exploratory nested scenario-count sensitivity over three fixed streams. Fees are in EUR/h; costs, differences, and bounds are in EUR/day. Costs show the mean [between-stream SD]. Difference is learned minus stochastic MIQP; the one-sided 95\% upper bound resamples paired dates, training batches, and scenario streams. Latency is controller-call mean/mean daily P95; TL is time-limit frequency.}",
        r"\label{tab:round2-scenarios}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Fee & $M$ & Limit & MIQP cost [SD] & Difference & Upper & Latency (ms) & TL \\",
        r"\midrule",
    ]
    for row in merged.sort_values(["fee", "time_limit_s", "n_scenarios"]).itertuples():
        main_lines.append(
            f"{int(row.fee)} & {int(row.n_scenarios)} & "
            f"{int(row.time_limit_s)} s & {row.mean_common_cost:.1f} "
            f"[{row.between_stream_cost_sd:.1f}] & "
            f"{row.mean_diff:.1f} & {row.one_sided_95_upper:.1f} & "
            f"{row.mean_latency_mean_ms:.0f}/{row.mean_latency_p95_ms:.0f} & "
            f"{100.0 * row.time_limit_fraction:.1f}\\% \\\\")
    main_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    central = merged[(merged["fee"] == 40)
                     & (merged["time_limit_s"] == 5)].sort_values(
                         "n_scenarios")
    first, last = central.iloc[0], central.iloc[-1]
    learned_direction = ("less" if last["mean_diff"] < 0 else "more")
    main_lines.append(
        f"At 40 EUR/h, increasing the scenario count from 2 to 16 under the "
        f"fixed five-second budget changed mean stochastic-MIQP cost from "
        f"{first['mean_common_cost']:.1f} to {last['mean_common_cost']:.1f} "
        f"EUR/day; the time-limit frequency changed from "
        f"{100.0 * first['time_limit_fraction']:.1f}\\% to "
        f"{100.0 * last['time_limit_fraction']:.1f}\\%. The learned policy was "
        f"{abs(last['mean_diff']):.1f} EUR/day {learned_direction} costly than the "
        f"16-scenario comparator; its one-sided upper bound was "
        f"{last['one_sided_95_upper']:.1f} EUR/day.")
    high = merged[(merged["fee"] == 80)
                  & (merged["n_scenarios"] == 16)].sort_values(
                      "time_limit_s")
    high5 = high[high["time_limit_s"] == 5].iloc[0]
    if len(high) > 1:
        high15 = high[high["time_limit_s"] == 15].iloc[0]
        main_lines.append(
            f" At 80 EUR/h, the 16-scenario cost changed from "
            f"{high5['mean_common_cost']:.1f} to "
            f"{high15['mean_common_cost']:.1f} EUR/day when the per-solve "
            f"limit increased from 5 to 15 seconds. Finite gaps were available "
            f"for {100.0 * high5['finite_gap_fraction']:.1f}\\% and "
            f"{100.0 * high15['finite_gap_fraction']:.1f}\\% of calls; their "
            f"conditional means were {format_gap_percent(high5['mean_gap'])} "
            f"and {format_gap_percent(high15['mean_gap'])}. These means do not "
            f"measure matched solver progress because availability and "
            f"closed-loop states differ. At 15 seconds, the time-limit frequency was "
            f"{100.0 * high15['time_limit_fraction']:.1f}\\%. The learned-minus-MIQP "
            f"upper bound at 15 seconds was "
            f"{high15['one_sided_95_upper']:.1f} EUR/day.")
    else:
        main_lines.append(
            f" At 80 EUR/h, the five-second 16-scenario run cost "
            f"{high5['mean_common_cost']:.1f} EUR/day and had a "
            f"{100.0 * high5['time_limit_fraction']:.1f}\\% time-limit "
            f"frequency.")
    main_lines.append("")

    learned_rows = []
    for fee in (40, 80):
        learned = read_aligned_parquet(RESULTS / LEARNED[fee])
        learned = learned[learned["selected_by_batch"]]
        learned_rows.append({
            "fee": fee, "n_scenarios": 0, "time_limit_s": 0,
            "method": "Offline learned",
            **{f"mean_{name}": float(learned[name].mean())
               for name in ("bill", "capacity_fee", "degradation_cost",
                            "terminal_penalty", "common_cost")},
        })
    comp = settings.copy()
    comp["method"] = "Stochastic MIQP"
    component_rows = pd.concat(
        [pd.DataFrame(learned_rows), comp], ignore_index=True,
        sort=False).sort_values(["fee", "time_limit_s", "n_scenarios"])

    supp_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Scenario-count cost decomposition, averaged over dates and available randomization units. Fees are in EUR/h and cost components in EUR/day.}",
        r"\label{tab:supp-round2-components}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rrrlrrrrr}",
        r"\toprule",
        r"Fee & $M$ & Limit & Method & Bill & Band fee & Degradation & Terminal & Total \\",
        r"\midrule",
    ]
    for row in component_rows.itertuples():
        m = "--" if int(row.n_scenarios) == 0 else str(int(row.n_scenarios))
        budget = "--" if int(row.time_limit_s) == 0 else f"{int(row.time_limit_s)} s"
        supp_lines.append(
            f"{int(row.fee)} & {m} & {budget} & {row.method} & "
            f"{row.mean_bill:.1f} & {row.mean_capacity_fee:.1f} & "
            f"{row.mean_degradation_cost:.1f} & "
            f"{row.mean_terminal_penalty:.2f} & "
            f"{row.mean_common_cost:.1f} \\\\")
    supp_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Scenario-stream and solver diagnostics. Fees are in EUR/h; stream columns are mean costs in EUR/day. Finite gap is the share of calls with finite primal and dual bounds and an ordinary reported gap; the mean gap is conditional on that event. TL is time-limit frequency.}",
        r"\label{tab:supp-round2-solvers}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{rrrrrrrrrrr}",
        r"\toprule",
        r"Fee & $M$ & Limit & Stream 1 & Stream 2 & Stream 3 & Incumbent & Finite gap & Mean gap & TL & P95 solve \\",
        r"\midrule",
    ])
    for row in merged.sort_values(["fee", "time_limit_s", "n_scenarios"]).itertuples():
        sub = streams[(streams["fee"] == row.fee)
                      & (streams["n_scenarios"] == row.n_scenarios)
                      & (streams["time_limit_s"] == row.time_limit_s)]
        costs = sub.set_index("stream")["mean_common_cost"]
        supp_lines.append(
            f"{int(row.fee)} & {int(row.n_scenarios)} & "
            f"{int(row.time_limit_s)} s & "
            f"{costs.loc[7301]:.1f} & {costs.loc[7302]:.1f} & "
            f"{costs.loc[7303]:.1f} & "
            f"{100.0 * row.incumbent_fraction:.1f}\\% & "
            f"{100.0 * row.finite_gap_fraction:.1f}\\% & "
            f"{format_gap_percent(row.mean_gap)} & "
            f"{100.0 * row.time_limit_fraction:.1f}\\% & "
            f"{row.p95_solve_s:.2f} s \\\\")
    supp_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])

    learned = read_aligned_parquet(RESULTS / LEARNED[40])
    learned = (learned[learned["selected_by_batch"]]
               .groupby("date", as_index=True)[
                   ["bill", "capacity_fee", "degradation_cost",
                    "terminal_penalty", "common_cost"]].mean())
    reference = read_aligned_parquet(RESULTS / "round2_reference_daily.parquet")
    reference = (reference[reference["fee_per_hour"] == 40]
                 .set_index("date")[["bill", "capacity_fee",
                                      "degradation_cost", "terminal_penalty",
                                      "common_cost"]])
    m16_parts = []
    for stream in STREAMS:
        daily_path, _ = paths(40, 16, stream, 5)
        daily = read_aligned_parquet(daily_path).copy()
        daily["stream"] = stream
        m16_parts.append(daily)
    m16 = (pd.concat(m16_parts, ignore_index=True)
           .groupby("date", as_index=True)[
               ["bill", "capacity_fee", "degradation_cost",
                "terminal_penalty", "common_cost"]].mean())
    if not learned.index.equals(reference.index) or not learned.index.equals(m16.index):
        raise RuntimeError("mechanism-distribution dates do not align")
    labels = {"bill": "Bill", "capacity_fee": "Band fee",
              "degradation_cost": "Degradation",
              "terminal_penalty": "Terminal", "common_cost": "Total"}
    supp_lines.extend([
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{Distribution of paired daily component differences at 40 EUR/h. Entries are learned minus comparator in EUR/day; learned batches and scenario streams are averaged within date.}",
        r"\label{tab:supp-round2-component-distribution}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparator & Component & Mean & P10 & Median & P90 \\",
        r"\midrule",
    ])
    for comparator_name, comparator in (
            ("No-learning reference", reference),
            ("Stochastic MIQP ($M=16$)", m16)):
        for component in labels:
            diff = learned[component] - comparator[component]
            supp_lines.append(
                f"{comparator_name} & {labels[component]} & {diff.mean():.1f} & "
                f"{diff.quantile(0.10):.1f} & {diff.median():.1f} & "
                f"{diff.quantile(0.90):.1f} \\\\")
        supp_lines.append(r"\addlinespace")
    supp_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "round2_scenarios.tex").write_text(
        "\n".join(main_lines), encoding="utf-8")
    (GENERATED / "round2_scenario_supplement.tex").write_text(
        "\n".join(supp_lines), encoding="utf-8")


def main() -> int:
    setting_rows, stream_rows, solver_rows, pair_rows = [], [], [], []
    # All five settings belong to the fixed sensitivity design. A missing
    # result must not silently narrow the reported comparison family.
    for fee, count, budget in SETTINGS:
        for stream in STREAMS:
            for path in paths(fee, count, stream, budget):
                if not path.exists():
                    raise FileNotFoundError(f'incomplete fixed scenario design: {path.name}')

    for fee, count, budget in SETTINGS:
        daily, solves = load_setting(fee, count, budget)
        for stream, sub in daily.groupby("stream"):
            stream_rows.append({
                "fee": fee, "n_scenarios": count,
                "time_limit_s": budget, "stream": int(stream),
                "n_days": len(sub),
                **{f"mean_{name}": float(sub[name].mean())
                   for name in COMPONENTS},
            })
        setting_rows.append({
            "fee": fee, "n_scenarios": count, "time_limit_s": budget,
            "n_streams": len(STREAMS), "n_days_per_stream": 30,
            "between_stream_cost_sd": float(
                daily.groupby("stream")["common_cost"].mean().std(ddof=1)),
            **{f"mean_{name}": float(daily[name].mean())
               for name in COMPONENTS},
        })
        # SCIP may encode an unavailable relative gap with a large finite
        # sentinel (typically 1e20). Treat a gap as available only when both
        # solver bounds are finite and the reported value is below that
        # sentinel range.
        finite_gap = finite_solver_gaps(solves)
        solver_rows.append({
            "fee": fee, "n_scenarios": count, "time_limit_s": budget,
            "n_streams": len(STREAMS), "n_solves": len(solves),
            "incumbent_fraction": float(solves["has_solution"].mean()),
            "finite_gap_fraction": float(len(finite_gap) / len(solves)),
            "mean_gap": float(finite_gap.mean()),
            "p95_gap": float(finite_gap.quantile(0.95)),
            "mean_solve_s": float(solves["solve_s"].mean()),
            "p95_solve_s": float(solves["solve_s"].quantile(0.95)),
            "time_limit_fraction": float(
                (solves["status"] == "timelimit").mean()),
        })
        learned = read_aligned_parquet(RESULTS / LEARNED[fee])
        learned = learned[learned["selected_by_batch"]].copy()
        stats = resample_learned_minus_scenario(
            learned, daily, seed=9200 + fee + 10 * count + budget)
        pair_rows.append({
            "fee": fee, "n_scenarios": count,
            "time_limit_s": budget, **stats})

    settings_frame = pd.DataFrame(setting_rows)
    streams_frame = pd.DataFrame(stream_rows)
    solvers_frame = pd.DataFrame(solver_rows)
    pairs_frame = pd.DataFrame(pair_rows)
    settings_frame.to_csv(RESULTS / "round2_scenario_setting_summary.csv",
                          index=False)
    streams_frame.to_csv(RESULTS / "round2_scenario_stream_summary.csv",
                         index=False)
    solvers_frame.to_csv(RESULTS / "round2_scenario_solver_summary.csv",
                         index=False)
    pairs_frame.to_csv(RESULTS / "round2_scenario_pairwise.csv", index=False)
    _write_latex(settings_frame, streams_frame, solvers_frame, pairs_frame)

    high = solvers_frame[(solvers_frame["fee"] == 80)
                         & (solvers_frame["n_scenarios"] == 16)
                         & (solvers_frame["time_limit_s"] == 5)].iloc[0]
    trigger = bool(
        high["incumbent_fraction"] < 1.0
        or high["mean_gap"] > 0.02
        or high["time_limit_fraction"] > 0.25)
    summary = {
        "label": "exploratory nested scenario-count sensitivity",
        "scenario_counts": list(COUNTS),
        "scenario_streams": list(STREAMS),
        "scenario_pool_size": 16,
        "high_fee_15s_triggered": trigger,
        "high_fee_15s_complete": bool(
            ((solvers_frame["fee"] == 80)
             & (solvers_frame["time_limit_s"] == 15)).any()),
    }
    (RESULTS / "round2_scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(settings_frame.to_string(index=False))
    print(solvers_frame.to_string(index=False))
    print(pairs_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
