"""Generate confirmatory statistics and LaTeX from frozen per-day outputs."""
from __future__ import annotations

import hashlib
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from result_validation import finite_solver_gaps, read_aligned_parquet
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)

FEES = (20, 40, 80)
LEARNED_TAG = {
    20: "v3_winter_fee20",
    40: "v3_winter_fee40",
    80: "v3_winter_fee80",
}
COMPARATORS = (
    "no_storage",
    "validation_tuned_price_rule",
    "convex_envelope_mpc",
    "deterministic_exact_band_miqp",
    "stochastic_two_stage_exact_band_miqp",
)
PRIMARY_ONLINE = (
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
COMPACT_DISPLAY = {
    "no_storage": "No storage",
    "validation_tuned_price_rule": "Price rule",
    "convex_envelope_mpc": "Envelope MPC",
    "deterministic_exact_band_miqp": "CM MIQP",
    "stochastic_two_stage_exact_band_miqp": "Stochastic MIQP",
    "offline_value_policy": "Offline procedure",
}


def relative_cost_description(learned_cost, comparator_cost):
    """Describe the observed direction without assuming a learned-policy win."""
    if not np.isfinite([learned_cost, comparator_cost]).all() or comparator_cost <= 0:
        raise ValueError('relative-cost prose requires finite costs and a positive denominator')
    difference = 100.0 * (learned_cost - comparator_cost) / comparator_cost
    if difference == 0.0:
        return 'unchanged'
    direction = 'lower' if difference < 0 else 'higher'
    return f'{abs(difference):.1f}\\% {direction}'


def recorded_training_time(log_dir, tag, seed, value_fit_s):
    """Recover one complete restart's elapsed time, without double counting fit.

    The outer manifest includes trainer construction, fitting, diagnostics,
    certification, and result writing. Its timestamps have second resolution.
    Imports and input preparation before the manifest are outside this timer.
    Multi-restart manifests cannot be allocated to individual restarts here.
    """
    matches = []
    for path in sorted(Path(log_dir).glob(f'train_{tag}_*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('exit_status') != 'completed':
            continue
        tokens = shlex.split(record.get('command', ''), posix=False)
        flags = {}
        for index, token in enumerate(tokens):
            if not token.startswith('--'):
                continue
            if '=' in token:
                key, value = token.split('=', 1)
            elif index + 1 < len(tokens) and not tokens[index + 1].startswith('--'):
                key, value = token, tokens[index + 1]
            else:
                continue
            if key in flags:
                raise ValueError(f'duplicate command flag in {path}: {key}')
            flags[key] = value.strip('\"\'')
        expected = {'--tag': tag, '--region': 'confirmatory',
                    '--regimes': 'winter_weekday', '--seeds': str(int(seed))}
        if all(flags.get(key) == value for key, value in expected.items()):
            matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f'{tag} seed {seed}: expected one completed single-restart '
                         f'training manifest, found {len(matches)}')
    path, record = matches[0]
    elapsed = (datetime.fromisoformat(record['end_time'])
               - datetime.fromisoformat(record['start_time'])).total_seconds()
    if (not np.isfinite(value_fit_s) or value_fit_s < 0.0 or elapsed <= 0.0
            or elapsed + 1.0 < value_fit_s):
        raise ValueError(f'training manifest cannot cover recorded value fitting: {path}')
    return {'training_wall_s': elapsed,
            'value_fit_wall_s': float(value_fit_s),
            'training_manifest': f'results/logs/{path.name}',
            'training_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def cost_premium_compute_threshold(cost_premium, saved_seconds):
    """EUR/compute-hour needed to offset a positive daily operating premium.

    This is the strict lower threshold before any upfront learning expense,
    not an assumed market price or a claim that latency monetizes itself.
    A nonpositive premium needs no such minimum; no time saving cannot offset
    a positive premium by this mechanism.
    """
    if not np.isfinite([cost_premium, saved_seconds]).all():
        raise ValueError('compute trade-off inputs must be finite')
    if cost_premium <= 0.:
        return 0.
    return 3600. * cost_premium / saved_seconds if saved_seconds > 0. else np.nan


def moving_block_means(values, n_boot=10000, block=7, seed=6101):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n < 2:
        raise ValueError("at least two paired days required")
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    offsets = np.arange(block)
    out = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        selected = []
        for start in rng.choice(starts, size=n_blocks, replace=True):
            selected.extend(((start + offsets) % n).tolist())
        out[b] = values[np.asarray(selected[:n])].mean()
    return out


def paired_stats(learned, comparator, seed):
    if set(learned['date']) != set(comparator['date']):
        raise ValueError('paired comparison requires identical evaluation dates')
    joined = learned[["date", "common_cost"]].merge(
        comparator[["date", "common_cost"]], on="date",
        suffixes=("_learned", "_comparator"), validate="one_to_one").sort_values('date')
    diff = (joined["common_cost_learned"]
            - joined["common_cost_comparator"]).to_numpy()
    boot = moving_block_means(diff, seed=seed)
    null_boot = moving_block_means(diff - diff.mean(), seed=seed + 10000)
    return {
        "n_days": len(diff),
        "n_training_batches": 1,
        "mean_diff": float(diff.mean()),
        "mean_learned": float(joined["common_cost_learned"].mean()),
        "mean_comparator": float(joined["common_cost_comparator"].mean()),
        "percent_diff": float(100.0 * diff.mean()
                              / joined["common_cost_comparator"].mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "one_sided_p": float((1 + np.sum(null_boot <= diff.mean()))
                             / (1 + len(null_boot))),
        "superiority": bool(np.quantile(boot, 0.95) < 0.0),
    }


def hierarchical_paired_stats(learned, comparator, seed, n_boot=10000,
                              block=7):
    """Resample complete training batches and paired circular date blocks."""
    if set(learned['date']) != set(comparator['date']):
        raise ValueError('paired comparison requires identical evaluation dates')
    joined = learned[["batch", "date", "common_cost"]].merge(
        comparator[["date", "common_cost"]], on="date",
        suffixes=("_learned", "_comparator"), validate="many_to_one")
    joined["difference"] = (joined["common_cost_learned"]
                            - joined["common_cost_comparator"])
    matrix = joined.pivot(index="date", columns="batch",
                          values="difference").sort_index().to_numpy()
    if np.isnan(matrix).any():
        raise RuntimeError("incomplete date-by-training-batch comparison")
    n_days, n_batches = matrix.shape
    if n_batches < 2:
        raise ValueError("hierarchical bootstrap requires repeated batches")
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    n_blocks = int(np.ceil(n_days / block))

    def resample_means(values):
        out = np.empty(n_boot)
        for replicate in range(n_boot):
            starts = rng.integers(0, n_days, size=n_blocks)
            date_index = np.concatenate(
                [(start + offsets) % n_days for start in starts])[:n_days]
            batch_index = rng.integers(0, n_batches, size=n_batches)
            out[replicate] = values[np.ix_(date_index, batch_index)].mean()
        return out

    point = float(matrix.mean())
    boot = resample_means(matrix)
    null_boot = resample_means(matrix - point)
    comparator_mean = float(joined.drop_duplicates("date")[
        "common_cost_comparator"].mean())
    return {
        "n_days": n_days,
        "n_training_batches": n_batches,
        "mean_diff": point,
        "mean_learned": float(joined["common_cost_learned"].mean()),
        "mean_comparator": comparator_mean,
        "percent_diff": float(100.0 * point / comparator_mean),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "one_sided_p": float((1 + np.sum(null_boot <= point))
                             / (1 + len(null_boot))),
        "superiority": bool(np.quantile(boot, 0.95) < 0.0),
    }


def fmt(value, digits=1):
    return f"{value:.{digits}f}"


def paired_comparison_table(pairs, primary_only=False):
    subset = pairs[pairs['comparator'].isin(PRIMARY_ONLINE)] if primary_only else pairs
    scope = 'Primary MIQP comparisons' if primary_only else 'All prespecified paired comparisons'
    label = 'tab:paired' if primary_only else 'tab:paired-all'
    lines = [r'\begin{table}[tbp]', r'\centering',
             '\\caption{' + scope + '. Differences and bounds are learned minus comparator '
             'in EUR/day; fee rates are in EUR/h. The one-sided 95\\% upper bound uses '
             'paired circular date blocks and, at the central fee, complete training '
             'batches. Superiority requires a negative upper bound. Holm $p$ adjusts '
             f'all {len(pairs)} comparisons.}}',
             f'\\label{{{label}}}', r'\small' if primary_only else r'\scriptsize',
             r'\begin{tabularx}{\linewidth}{r>{\raggedright\arraybackslash}Xrrrr}',
             r'\toprule',
             r'Fee & Comparator & Difference & Relative & Upper bound & Holm $p$ \\',
             r'\midrule']
    for row in subset.itertuples():
        lines.append(f'{int(row.fee)} & {COMPACT_DISPLAY[row.comparator]} & '
                     f'{row.mean_diff:.1f} & {row.percent_diff:.1f}\\% & '
                     f'{row.one_sided_95_upper:.1f} & {row.holm_p_all:.3f} \\\\')
    return lines + [r'\bottomrule', r'\end{tabularx}', r'\end{table}']


def holm_adjust(p_values):
    """Holm step-down adjustment, returned in original order."""
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(p)
    adjusted[order] = adjusted_sorted
    return adjusted


def main() -> int:
    pairwise = []
    method_rows = []
    restart_rows = []
    selection_rows = []
    offline_rows = []
    learned_by_fee = {}
    comparator_by_fee = {}

    for fee in FEES:
        comp_path = RESULTS / (
            f"common_daily_confirmatory_thr300_fee{fee}_w10_n2.parquet")
        learned_path = RESULTS / f"learned_daily_{LEARNED_TAG[fee]}.parquet"
        if not comp_path.exists() or not learned_path.exists():
            raise FileNotFoundError(
                f"incomplete fee {fee}: {comp_path.exists()=}, "
                f"{learned_path.exists()=}")
        comp = read_aligned_parquet(comp_path)
        comp = comp[comp["regime"] == "winter_weekday"].copy()
        learned_all = read_aligned_parquet(learned_path)
        if "is_fallback" not in learned_all.columns:
            learned_all["is_fallback"] = False
        trained_all = learned_all[~learned_all["is_fallback"]].copy()
        chosen = learned_all[learned_all["selected_by_batch"]].copy()
        selection_audit = json.loads((
            RESULTS / f"restart_batches_{LEARNED_TAG[fee]}.json"
        ).read_text(encoding="utf-8"))
        for selected in selection_audit["batch_selections"]:
            batch = int(selected["batch"])
            chosen_batch = chosen[chosen["batch"] == batch]
            if len(chosen_batch) != 30:
                raise RuntimeError(
                    f"fee {fee} batch {batch}: expected 30 selected days, "
                    f"found {len(chosen_batch)}")
            selection_rows.append({
                "fee": fee, "batch": batch,
                "selected_kind": selected.get(
                    "selected_kind", "trained_policy"),
                "selected_seed": selected.get("selected_seed"),
                "selected_validation_mean_common_cost": selected[
                    "selected_validation_mean_common_cost"],
                "selected_test_mean_common_cost": float(
                    chosen_batch["common_cost"].mean()),
            })
        learned_by_fee[fee] = chosen
        comparator_by_fee[fee] = comp

        for seed, sub in trained_all.groupby("seed"):
            run_dir = (ROOT / "checkpoints" / LEARNED_TAG[fee]
                       / "confirmatory" / "winter_weekday"
                       / f"seed{int(seed)}")
            run_result = json.loads((run_dir / "result.json").read_text(
                encoding="utf-8"))
            value_fit_s = float(sum(
                record["train_info"]["wall_s"]
                for record in run_result["iterations"]))
            training_record = recorded_training_time(
                ROOT / 'results' / 'logs', LEARNED_TAG[fee], seed, value_fit_s)
            validation_daily = read_aligned_parquet(
                run_dir / "validation_daily.parquet")
            validation_controller_s = float(
                (validation_daily["latency_mean_ms"] * 96.0 / 1000.0).sum())
            restart_rows.append({
                "fee": fee, "seed": int(seed),
                "batch": int(sub["batch"].iloc[0]),
                "selected_by_batch": bool(sub["selected_by_batch"].iloc[0]),
                "validation_mean_common_cost": float(
                    sub["validation_mean_common_cost"].iloc[0]),
                "test_mean_common_cost": float(sub["common_cost"].mean()),
                **training_record,
                "validation_controller_s": validation_controller_s,
            })

        # For the central cell, average the independently selected batch
        # policies within date before pairing. This makes the complete
        # randomized procedure, not the better batch, the reported estimator.
        learned_daily = chosen.groupby("date", as_index=False).agg({
            "common_cost": "mean", "bill": "mean", "capacity_fee": "mean",
            "degradation_cost": "mean", "terminal_penalty": "mean",
            "terminal_soc_dev": "mean", "latency_mean_ms": "mean",
            "latency_p95_ms": "mean", "soc_violations": "mean",
            "projection_events": "mean", "reflection_events": "mean",
        })
        restart_fee = [row for row in restart_rows if row["fee"] == fee]
        for batch in sorted({row["batch"] for row in restart_fee}):
            members = [row for row in restart_fee if row["batch"] == batch]
            training_s = sum(row["training_wall_s"] for row in members)
            validation_s = sum(row["validation_controller_s"]
                               for row in members)
            offline_rows.append({
                "fee": fee, "batch": batch,
                "n_restarts": len(members),
                "training_wall_s": training_s,
                "value_fit_wall_s": sum(row['value_fit_wall_s'] for row in members),
                "validation_controller_s": validation_s,
                "total_measured_offline_s": training_s + validation_s,
            })
        method_rows.append({
            "fee": fee, "method": "offline_value_policy",
            "n_days": len(learned_daily),
            **{f"mean_{m}": float(learned_daily[m].mean()) for m in (
                "common_cost", "bill", "capacity_fee", "degradation_cost",
                "terminal_penalty", "terminal_soc_dev", "latency_mean_ms",
                "latency_p95_ms", "soc_violations", "projection_events",
                "reflection_events")},
        })
        for method in COMPARATORS:
            sub = comp[comp["method"] == method].copy()
            if len(sub) != len(learned_daily):
                raise RuntimeError(
                    f"fee {fee} method {method}: expected "
                    f"{len(learned_daily)} days, found {len(sub)}")
            method_rows.append({
                "fee": fee, "method": method, "n_days": len(sub),
                **{f"mean_{m}": float(sub[m].mean()) for m in (
                    "common_cost", "bill", "capacity_fee",
                    "degradation_cost", "terminal_penalty",
                    "terminal_soc_dev", "latency_mean_ms", "latency_p95_ms",
                    "soc_violations", "projection_events",
                    "reflection_events")},
            })
            stats_seed = 6101 + fee + len(pairwise)
            stats = (hierarchical_paired_stats(chosen, sub, stats_seed)
                     if chosen["batch"].nunique() > 1
                     else paired_stats(learned_daily, sub, stats_seed))
            pairwise.append({"fee": fee, "comparator": method, **stats})

    methods = pd.DataFrame(method_rows)
    pairs = pd.DataFrame(pairwise)
    pairs["holm_p_all"] = holm_adjust(pairs["one_sided_p"].to_numpy())
    pairs["holm_p_primary"] = np.nan
    primary_mask = pairs["comparator"].isin(PRIMARY_ONLINE)
    pairs.loc[primary_mask, "holm_p_primary"] = holm_adjust(
        pairs.loc[primary_mask, "one_sided_p"].to_numpy())
    restarts = pd.DataFrame(restart_rows)
    selections = pd.DataFrame(selection_rows)
    offline = pd.DataFrame(offline_rows)
    loading = pd.read_csv(RESULTS / "confirmatory_model_loading.csv")
    if len(loading) != len(offline):
        raise RuntimeError("model-loading audit does not cover every batch")
    offline = offline.merge(
        loading[["fee", "batch", "model_loading_s"]],
        on=["fee", "batch"], how="left", validate="one_to_one")
    if offline["model_loading_s"].isna().any():
        raise RuntimeError("missing model-loading measurement")
    offline["total_measured_offline_s"] += offline["model_loading_s"]
    methods.to_csv(RESULTS / "confirmatory_method_summary.csv", index=False)
    pairs.to_csv(RESULTS / "confirmatory_pairwise.csv", index=False)
    restarts.to_csv(RESULTS / "confirmatory_restart_summary.csv", index=False)
    selections.to_csv(RESULTS / "confirmatory_selection_summary.csv", index=False)
    offline.to_csv(RESULTS / "confirmatory_offline_compute.csv", index=False)

    # Solver diagnostics are kept separate from economic outcomes.
    solver_rows = []
    for fee in FEES:
        path = RESULTS / (
            f"common_solves_confirmatory_thr300_fee{fee}_w10_n2.parquet")
        solves = read_aligned_parquet(path)
        # The central-fee file is later augmented by the seasonal runs.  Keep
        # the confirmatory solver table tied to the frozen winter endpoint.
        solves = solves[solves["regime"] == "winter_weekday"].copy()
        for method, sub in solves.groupby("method"):
            gap = finite_solver_gaps(sub)
            solver_rows.append({
                "fee": fee, "method": method, "n_solves": len(sub),
                "incumbent_fraction": float(sub["has_solution"].mean()),
                "finite_gap_fraction": float(len(gap) / len(sub)),
                "target_gap_fraction": float((gap <= 0.01 + 1e-12).mean()),
                "target_gap_fraction_scope": "conditional on an incumbent and finite primal/dual bounds",
                "target_gap_fraction_all_calls": float((gap <= 0.01 + 1e-12).sum() / len(sub)),
                "mean_gap": float(gap.mean()), "max_gap": float(gap.max()),
                "mean_solve_s": float(sub["solve_s"].mean()),
                "p95_solve_s": float(sub["solve_s"].quantile(0.95)),
                "time_limit_fraction": float((sub["status"] == "timelimit").mean()),
            })
    solver = pd.DataFrame(solver_rows)
    solver.to_csv(RESULTS / "confirmatory_solver_summary.csv", index=False)

    table = [
        "\\begin{table}[tbp]",
        "\\centering",
        "\\caption{Held-out winter performance under the common objective. "
        "Cost and terminal penalty are daily means in EUR/day; latency shows "
        "the controller-call mean and mean daily P95. "
        "The offline row averages independently selected five-restart "
        "batches when repeated batches are available. CM denotes conditional mean; "
        "stochastic MIQP uses two scenarios. The price rule is validation-tuned.}",
        "\\label{tab:confirmatory}",
        "\\small",
        "\\begin{tabularx}{\\linewidth}{r>{\\raggedright\\arraybackslash}Xrrr}",
        "\\toprule",
        "Fee & Method & Cost & Terminal & Latency (ms) \\\\",
        "(EUR/h) & & (EUR/day) & penalty & mean/P95 \\\\",
        "\\midrule",
    ]
    for fee in FEES:
        sub = methods[methods["fee"] == fee]
        order = ["no_storage", "validation_tuned_price_rule",
                 "convex_envelope_mpc",
                 "deterministic_exact_band_miqp",
                 "stochastic_two_stage_exact_band_miqp",
                 "offline_value_policy"]
        for idx, method in enumerate(order):
            row = sub[sub["method"] == method].iloc[0]
            table.append(
                f"{fee if idx == 0 else ''} & {COMPACT_DISPLAY[method]} & "
                f"{fmt(row['mean_common_cost'])} & "
                f"{fmt(row['mean_terminal_penalty'], 2)} & "
                f"{fmt(row['mean_latency_mean_ms'], 1)}/{fmt(row['mean_latency_p95_ms'], 1)} \\\\")
        if fee != FEES[-1]:
            table.append("\\addlinespace")
    table.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}"])

    narrative = paired_comparison_table(pairs, primary_only=True)

    best_online = []
    for fee in FEES:
        sub = methods[(methods["fee"] == fee)
                      & methods["method"].isin(PRIMARY_ONLINE)]
        best = sub.loc[sub["mean_common_cost"].idxmin()]
        learned_cost = float(methods[(methods["fee"] == fee)
                                     & (methods["method"] ==
                                        "offline_value_policy")]
                             ["mean_common_cost"].iloc[0])
        best_online.append((fee, learned_cost, best["method"],
                            float(best["mean_common_cost"])))
    amortization_rows = []
    for fee, learned_cost, method, online_cost in best_online:
        learned_lat = float(methods[(methods["fee"] == fee)
                                    & (methods["method"] ==
                                       "offline_value_policy")]
                            ["mean_latency_mean_ms"].iloc[0])
        online_lat = float(methods[(methods["fee"] == fee)
                                   & (methods["method"] == method)]
                           ["mean_latency_mean_ms"].iloc[0])
        offline_s = float(offline[offline["fee"] == fee]
                          ["total_measured_offline_s"].mean())
        saved_s_day = 96.0 * (online_lat - learned_lat) / 1000.0
        amortization_rows.append({
            "fee": fee, "online_comparator": method,
            "learned_minus_online_eur_per_day": learned_cost - online_cost,
            "mean_offline_s_per_complete_batch": offline_s,
            "online_compute_saved_s_per_day": saved_s_day,
            "premium_offset_threshold_eur_per_compute_hour": cost_premium_compute_threshold(
                learned_cost - online_cost, saved_s_day),
            "compute_equivalent_break_even_days": (
                offline_s / saved_s_day if saved_s_day > 0 else np.nan),
        })
    amortization = pd.DataFrame(amortization_rows)
    amortization.to_csv(RESULTS / "confirmatory_amortization.csv", index=False)
    # Valuing compute can change which online comparator is preferable.
    # Keep both comparisons, rather than presenting the lower operating-cost
    # (potentially slower) comparator as the best joint cost/time choice.
    tradeoff_rows = []
    for fee in FEES:
        sub = methods[methods['fee'] == fee].set_index('method')
        learned = sub.loc['offline_value_policy']
        for method in PRIMARY_ONLINE:
            comparator = sub.loc[method]
            premium = float(learned['mean_common_cost'] - comparator['mean_common_cost'])
            saved = float(96. * (comparator['mean_latency_mean_ms']
                                 - learned['mean_latency_mean_ms']) / 1000.)
            tradeoff_rows.append({
                'fee': fee, 'online_comparator': method,
                'learned_minus_online_eur_per_day': premium,
                'online_compute_saved_s_per_day': saved,
                'premium_offset_threshold_eur_per_compute_hour':
                    cost_premium_compute_threshold(premium, saved),
                'scope': 'pairwise necessary valuation before any upfront expense; not a compute market price',
            })
    pd.DataFrame(tradeoff_rows).to_csv(RESULTS / 'confirmatory_compute_tradeoffs.csv', index=False)
    sentences = []
    for fee, learned_cost, method, online_cost in best_online:
        pct = 100.0 * (learned_cost - online_cost) / online_cost
        direction = "lower" if pct < 0 else "higher"
        sentences.append(
            f"At {fee} EUR/h, the complete offline procedure averaged "
            f"{learned_cost:.1f} EUR/day, {abs(pct):.1f}\\% {direction} than "
            f"the better online MIQP ({DISPLAY[method]}, {online_cost:.1f} EUR/day).")
    if len({item[2] for item in best_online}) == 1:
        cost_values = [f'{item[1]:.1f}' for item in best_online]
        costs = ', '.join(cost_values[:-1]) + ', and ' + cost_values[-1]
        relative = [relative_cost_description(item[1], item[3]) for item in best_online]
        directions = [value.rsplit(' ', 1)[-1] for value in relative]
        if len(set(directions)) == 1 and directions[0] in ('higher', 'lower'):
            magnitudes = [value.rsplit(' ', 1)[0] for value in relative]
            comparisons = (', '.join(magnitudes[:-1]) + ', and ' + magnitudes[-1]
                           + ' ' + directions[0])
        else:
            comparisons = ', '.join(relative[:-1]) + ', and ' + relative[-1]
        label = DISPLAY[best_online[0][2]]
        method = label[0].lower() + label[1:]
        sentences = [
            f'At fee rates of 20, 40, and 80 EUR/h, the offline procedure averaged '
            f'{costs} EUR/day. These costs were {comparisons}, respectively, '
            f'than the primary {method} '
            '(Table~\\ref{tab:confirmatory}).']
    primary_pairs = pairs[pairs['comparator'].isin(PRIMARY_ONLINE)]
    sentences.append(
        f"The prespecified learned-superiority rule was met in "
        f"{int(primary_pairs['superiority'].sum())} of six MIQP comparisons "
        "(Table~\\ref{tab:paired}); the supplement reports all baseline comparisons.")
    restart_counts = []
    for fee, _, _, online_cost in best_online:
        sub = restarts[restarts["fee"] == fee]
        n_lower = int((sub["test_mean_common_cost"] < online_cost).sum())
        restart_counts.append(f"{n_lower}/{len(sub)} at {fee} EUR/h")
    restart_sentence = (
        "As a test-set stability diagnostic, individual restarts had lower "
        "mean cost than the better online MIQP in "
        + ", ".join(restart_counts)
        + "; this diagnostic did not alter validation-based selection.")
    low_pair = pairs[(pairs['fee'] == 20)
                     & (pairs['comparator'] == 'no_storage')].iloc[0]
    low_fee_scope_sentence = (
        f"At 20 EUR/h, the one-sided upper bound against no storage was "
        f"{low_pair['one_sided_95_upper']:.1f} EUR/day; superiority "
        + ("was established." if bool(low_pair['superiority'])
           else "was not established."))
    central_selected = selections[selections["fee"] == 40]
    batch_sentence = (
        f"The five validation-selected central-fee batch policies ranged from "
        f"{central_selected['selected_test_mean_common_cost'].min():.1f} to "
        f"{central_selected['selected_test_mean_common_cost'].max():.1f} EUR/day on "
        f"the common test days.")
    fallback_by_fee = selections.groupby("fee")["selected_kind"].apply(
        lambda values: int((values == "zero_action_fallback").sum()))
    fallback_sentence = (
        "The zero-action feasible incumbent was retained in "
        + ", ".join(
            f"{fallback_by_fee.get(fee, 0)}/{len(selections[selections['fee'] == fee])} "
            f"batches at {fee} EUR/h" for fee in FEES)
        + "; retained fallback outcomes are not attributed to neural control.")
    rule_choices = []
    for fee in FEES:
        rule_audit = json.loads((
            RESULTS / f"rule_tuning_winter_weekday_thr300_fee{fee}_w10.json"
        ).read_text(encoding="utf-8"))
        kind = rule_audit.get("selected_kind", "price_threshold")
        label = ("zero action" if kind == "zero_action"
                 else "price-threshold policy")
        rule_choices.append(f"{fee} EUR/h: {label}")
    rule_sentence = (
        "Validation selected the threshold-rule baseline as "
        + ", ".join(rule_choices)
        + "; a zero-action choice remains labeled as such in its tuning log.")
    mean_offline_s = float(offline["total_measured_offline_s"].mean())
    trained_loading = loading.loc[
        loading["selected_kind"] == "trained_policy", "model_loading_s"]
    mean_loading_s = (float(trained_loading.mean())
                      if len(trained_loading) else 0.0)
    avg_lat_l = float(methods[methods["method"] == "offline_value_policy"]
                      ["mean_latency_mean_ms"].mean())
    avg_lat_m = float(methods[methods["method"]
                              == "deterministic_exact_band_miqp"]
                      ["mean_latency_mean_ms"].mean())
    compute_sentence = (
        f"A complete five-restart batch used {mean_offline_s / 60.0:.1f} "
        f"minutes of recorded training-run time, validation-controller computation, "
        f"and selected-checkpoint loading. Among trained selections, mean "
        f"checkpoint loading was {mean_loading_s:.2f} s. "
        f"At 96 calls per day, the measured online controller time was "
        f"{96.0 * avg_lat_l / 1000.0:.2f} s/day for the offline policy and "
        f"{96.0 * avg_lat_m / 1000.0:.2f} s/day for conditional-mean MIQP.")
    finite_break_even = amortization["compute_equivalent_break_even_days"].dropna()
    amortization_sentence = (
        f"On compute time alone, assigning the same unit value to offline and "
        f"online seconds, the measured offline work equals the accumulated "
        f"online time saving after "
        f"{finite_break_even.min():.1f}--{finite_break_even.max():.1f} "
        f"deployment days relative to the better of the two online MIQPs. "
        f"Equation~\\eqref{{eq:amortization}} also includes the observed "
        f"operating-cost difference for any chosen monetary value of compute "
        f"time.")
    tradeoff_figure = [
        "\\begin{figure}[t]", "\\centering",
        "\\includegraphics[width=\\linewidth]{figures/fig_cost_latency.pdf}",
        "\\caption{Operating cost and online computation under the common "
        "benchmark. Cost is relative to no storage; latency is measured from "
        "controller call to returned held action. Learned evaluation uses the GPU; "
        "each MIQP solve uses one CPU thread.}",
        "\\label{fig:cost-latency}", "\\end{figure}",
    ]
    concise_compute = (
        f"A complete five-restart batch required {mean_offline_s / 60.:.1f} minutes "
        f"of recorded training, validation, and checkpoint loading. Mean online "
        f"call latency was {avg_lat_l:.1f} ms, versus {avg_lat_m:.1f} ms for "
        f"conditional-mean MIQP (Figure~\\ref{{fig:cost-latency}}).")
    choice_scope = (
        f"Zero action was retained in {int(fallback_by_fee.sum())} of "
        f"{len(selections)} learned-selection batches.")
    if all('zero action' in value for value in rule_choices):
        choice_scope += " The price-rule baseline selected zero action at all three fees."
    else:
        choice_scope += " " + rule_sentence
    body = "\n".join(table + [""] + tradeoff_figure
                    + ["", " ".join(sentences), batch_sentence, choice_scope, "",
                       concise_compute, ""] + narrative)
    (GENERATED / "confirmatory_results.tex").write_text(body,
                                                         encoding="utf-8")
    full_inference = ([r'\subsection{Primary inference and implementation accounting}', ""]
                      + paired_comparison_table(pairs)
                      + ["", restart_sentence, low_fee_scope_sentence, fallback_sentence,
                         rule_sentence, compute_sentence,
                         amortization_sentence.replace(
                             r'Equation~\eqref{eq:amortization}',
                             "The main paper's break-even condition"), ""])
    (GENERATED / 'confirmatory_inference_supplement.tex').write_text(
        '\n'.join(full_inference), encoding='utf-8')

    superior_online = sum(
        bool(pairs[(pairs["fee"] == fee)
                   & (pairs["comparator"].isin(PRIMARY_ONLINE))]
             ["superiority"].all()) for fee in FEES)
    fallback_total = int(sum(fallback_by_fee.get(fee, 0) for fee in FEES))
    comparisons_vs_best = [relative_cost_description(learned_cost, online_cost)
                           for _, learned_cost, _, online_cost in best_online]
    primary_pairs = pairs[pairs["comparator"].isin(PRIMARY_ONLINE)]
    if bool(primary_pairs["superiority"].all()):
        inference_sentence = (
            "All six prespecified one-sided 95\\% upper "
            "bounds against the two MIQPs were below zero. ")
    else:
        inference_sentence = (
            f"The prespecified learned-superiority rule was met in "
            f"{int(primary_pairs['superiority'].sum())} of six MIQP comparisons. ")
    abstract = (
        f"On 30 held-out 2019 winter weekdays, its mean cost was "
        f"{comparisons_vs_best[0]}, {comparisons_vs_best[1]}, "
        f"and {comparisons_vs_best[2]} relative to the lower-cost online MIQP "
        f"at fee rates of 20, 40, and 80 EUR/h, respectively. "
        + inference_sentence
        + "Central-fee inference resampled five independent five-restart "
        "batches; each outer-anchor result is conditional on one such batch. "
        f"Its mean GPU-assisted online latency was {avg_lat_l:.1f} ms versus "
        f"{avg_lat_m:.1f} ms for single-threaded conditional-mean MIQP."
    )
    (GENERATED / "abstract_result.tex").write_text(abstract,
                                                    encoding="utf-8")
    summary = {"best_online_by_fee": best_online,
               "n_fee_anchors_superior_to_both_online": superior_online,
               "fallback_batches_by_fee": {
                   int(fee): int(fallback_by_fee.get(fee, 0))
                   for fee in FEES},
               "offline_mean_latency_ms": avg_lat_l,
               "deterministic_miqp_mean_latency_ms": avg_lat_m,
               "mean_complete_batch_offline_s": float(
                   offline["total_measured_offline_s"].mean())}
    (RESULTS / "confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
