"""Pair stored rule and primary MIQP daily costs; never rerun controllers.

Run from any directory. Read stored outcomes without executing controllers.
Inference is conditional on the selected rule and recorded MIQP scenario stream.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT
N_BOOT, BLOCK = 10000, 7
BASE_SEED = 20261005


def moving_block_means(values, seed, n_boot=N_BOOT, block=BLOCK):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite paired differences")
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.choice(np.arange(n), size=(n_boot, int(np.ceil(n / block))))
    indices = (starts[:, :, None] + np.arange(block)) % n
    return values[indices.reshape(n_boot, -1)[:, :n]].mean(axis=1)


def pair_costs(rule, miqp):
    for frame in (rule, miqp):
        if frame.date.duplicated().any():
            raise ValueError("Duplicate daily outcomes")
    if set(rule.date) != set(miqp.date):
        raise ValueError("Unmatched evaluation dates")
    paired = rule.merge(miqp, on="date", validate="one_to_one",
                        suffixes=("_rule", "_miqp")).sort_values("date")
    if not np.isfinite(paired[["common_cost_rule", "common_cost_miqp"]]).all().all():
        raise ValueError("Nonfinite daily cost")
    return paired


def self_test():
    np.testing.assert_allclose(moving_block_means(np.ones(30) * 2.4, 17), 2.4)
    values = np.arange(30, dtype=float) - 9
    np.testing.assert_allclose(moving_block_means(-values, 17),
                               -moving_block_means(values, 17))
    # Match the existing analysis' replicate-by-replicate circular-block algorithm.
    rng = np.random.default_rng(17)
    expected = []
    for _ in range(20):
        selected = []
        for start in rng.choice(np.arange(30), size=5, replace=True):
            selected.extend(((start + np.arange(7)) % 30).tolist())
        expected.append(values[np.asarray(selected[:30])].mean())
    np.testing.assert_allclose(moving_block_means(values, 17, n_boot=20), expected)
    frame = pd.DataFrame({"date": ["a", "b"], "common_cost": [1., 2.]})
    assert list(pair_costs(frame, frame.iloc[::-1]).date) == ["a", "b"]
    for bad in [frame.iloc[[0, 0]], frame.iloc[:1]]:
        try:
            pair_costs(frame, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid pairing accepted")


def main():
    self_test()
    rows, provenance = [], []
    for fee in (20, 40, 80):
        rule_path = PUBLIC / f"experiments/controller_comparisons/results/heuristic/test_winter_weekday_fee{fee}.json"
        miqp_path = PUBLIC / f"experiments/results/common_daily_confirmatory_thr300_fee{fee}_w10_n2.parquet"
        rule = pd.DataFrame(json.loads(rule_path.read_text())["days"])[["date", "common_cost"]]
        raw = pd.read_parquet(miqp_path)
        miqp = raw.loc[(raw.method == "stochastic_two_stage_exact_band_miqp") &
                       (raw.regime == "winter_weekday")].copy()
        for column, expected in [("threshold_kw", 300), ("fee_per_hour", fee),
                                 ("n_scenarios", 2), ("solver_time_limit_s", 5),
                                 ("scenario_stream_seed", 4101)]:
            assert miqp[column].eq(expected).all(), (fee, column)
        miqp = miqp[["date", "common_cost"]]
        for frame in (rule, miqp):
            frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        paired = pair_costs(rule, miqp)
        assert len(paired) == 30
        differences = (paired.common_cost_rule - paired.common_cost_miqp).to_numpy()
        ci = np.percentile(moving_block_means(differences, BASE_SEED + fee), [2.5, 97.5])
        rows.append(dict(fee=fee, n_days=len(paired), seed=BASE_SEED + fee,
                         rule_mean=float(paired.common_cost_rule.mean()),
                         miqp_mean=float(paired.common_cost_miqp.mean()),
                         difference=float(differences.mean()), ci95=ci.tolist(),
                         daily_pairs=paired.to_dict(orient="records")))
        for path in (rule_path, miqp_path):
            provenance.append(dict(path=path.relative_to(PUBLIC).as_posix(),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(contrast="rule minus primary stochastic MIQP, EUR/day",
                  method="two-sided percentile circular moving-block bootstrap",
                  n_boot=N_BOOT, block_eligible_days=BLOCK,
                  scope="Exploratory; conditional on validation-selected rules and recorded MIQP stream 4101. No equivalence margin or multiplicity adjustment.",
                  self_tests_passed=True, inputs=provenance, results=rows)
    output = Path(__file__).resolve().parent / "results/heuristic"
    output.mkdir(exist_ok=True)
    (output / "rule_miqp_paired.json").write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    tex = [r"\begin{table}[tbp]", r"\centering", r"\caption{Tariff-aware rule versus primary two-scenario stochastic MIQP on 30 paired winter test days. Costs and differences are in EUR/day. Two-sided 95\% percentile intervals use 10,000 circular moving-block replicates with seven-eligible-day blocks, conditional on the selected rules and recorded MIQP scenario stream. Positive differences favor MIQP.}", r"\label{tab:rule-miqp-paired}", r"\begin{tabular}{rrrrr}", r"\toprule", r"Fee (EUR/h) & Rule & MIQP & Rule $-$ MIQP & 95\% interval \\", r"\midrule"]
    for row in rows:
        lo, hi = row["ci95"]
        tex.append(f'{row["fee"]} & {row["rule_mean"]:.1f} & {row["miqp_mean"]:.1f} & ${row["difference"]:+.1f}$ & $[{lo:.1f},\\,{hi:.1f}]$ ' + r"\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    generated = ROOT / "manuscript/generated"
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "rule_miqp_paired.tex").write_bytes("\n".join(tex).encode("utf-8"))
    print(json.dumps([{k: v for k, v in row.items() if k != "daily_pairs"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
