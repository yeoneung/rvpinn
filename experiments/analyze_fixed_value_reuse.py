"""Aggregate stored reuse trajectories and timings without running controllers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/fixed_value_reuse"
RESULTS = Path(os.environ.get("RVPINN_REUSE_RESULTS", str(HERE / "results")))
GENERATED = ROOT / "manuscript/generated"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def collect(results=RESULTS):
    cfg, manifest = read(HERE / "protocol.json"), read(results / "manifest.json")
    assert hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest() == manifest["protocol_sha256"]
    assert hashlib.sha256((ROOT / cfg["checkpoint"]).read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
    cases = manifest["cases"]
    assert len(cases) == 12 and len({c["path_id"] for c in cases}) == 4
    assert {c["initial_soc"] for c in cases} == {0.2, 0.5, 0.8}
    assert {c["id"] for c in cases} == set(range(12))
    for path_id in range(4):
        same_path = [c for c in cases if c["path_id"] == path_id]
        assert len(same_path) == 3
        for c in same_path[1:]:
            assert c["net"] == same_path[0]["net"] and c["price"] == same_path[0]["price"]
    timing_rows, quality_rows, input_paths = [], [], [HERE / "protocol.json", results / "manifest.json"]
    max_action_difference, max_cost_difference = 0.0, 0.0
    for n in (1, 4, 12):
        path = results / f"miqp_n{n}.json"
        input_paths.append(path)
        mip = read(path)
        assert len(mip["results"]) == n
        assert mip["workers"] == min(n, 4)
        assert {r["row"]["case_id"] for r in mip["results"]} == set(range(n))
        timing_rows.append(dict(n=n, method="miqp", repetition=0, device="cpu",
                                workers=mip["workers"], wall_s=mip["wall_s"], setup_s=mip["process_setup_s"]))
        for r in mip["results"]:
            assert "error" not in r and len(r["row"]["actions"]) == 96
            assert len(r["solves"]) == 96
            if n == 12:
                quality_rows.append(dict(method="miqp", **{k: v for k, v in r["row"].items() if k != "actions"}))
        for rep in range(3):
            path = results / f"neural_n{n}_r{rep}.json"
            input_paths.append(path)
            neural = read(path)
            entries = {r["method"]: r for r in neural["results"]}
            assert set(entries) == {"scalar", "batched", "reference"}
            for method, entry in entries.items():
                assert entry["device"] == "cpu", "Do not mix devices in the CPU manuscript benchmark"
                assert len(entry["rows"]) == n and entry["n"] == n and entry["repetition"] == rep
                assert {r["case_id"] for r in entry["rows"]} == set(range(n))
                timing_rows.append(dict(n=n, method=method, repetition=rep, device="cpu", workers=1,
                                        wall_s=entry["full_call_wall_s"], setup_s=0.0))
                for row in entry["rows"]:
                    assert len(row["actions"]) == 96
                    if n == 12 and rep == 0:
                        quality_rows.append(dict(method=method, **{k: v for k, v in row.items() if k != "actions"}))
            scalar = sorted(entries["scalar"]["rows"], key=lambda r: r["case_id"])
            batched = sorted(entries["batched"]["rows"], key=lambda r: r["case_id"])
            max_action_difference = max(max_action_difference, float(np.max(np.abs(
                np.array([r["actions"] for r in scalar]) - np.array([r["actions"] for r in batched])))))
            max_cost_difference = max(max_cost_difference, max(abs(a["cost"] - b["cost"]) for a, b in zip(scalar, batched)))
    assert max_action_difference < 1e-12 and max_cost_difference < 1e-8
    timing = pd.DataFrame(timing_rows)
    assert (timing["wall_s"] > 0).all()
    summaries = timing.groupby(["n", "method", "device", "workers"]).agg(
        repeats=("wall_s", "size"), median_wall_s=("wall_s", "median"),
        min_wall_s=("wall_s", "min"), max_wall_s=("wall_s", "max")).reset_index()
    quality = pd.DataFrame(quality_rows)
    assert len(quality) == 48 and np.isfinite(quality.cost).all()
    for row in quality.itertuples():
        case = cases[int(row.case_id)]
        assert row.path_id == case["path_id"] and row.initial_soc == case["initial_soc"]
    costs = quality.pivot(index="case_id", columns="method", values="cost")
    cost_means = costs.mean()
    times = summaries[summaries.n == 12].set_index("method").median_wall_s
    offline = pd.read_csv(ROOT / "experiments/results/confirmatory_offline_compute.csv")
    historical = float(offline[(offline.fee == 40) & (offline.batch == 0)].total_measured_offline_s.iloc[0])
    assert abs(historical - manifest["historical_training_and_selection_s"]) < 1e-8
    solves = [s for r in read(results / "miqp_n12.json")["results"] for s in r["solves"]]
    relative = 100 * (costs.scalar - costs.miqp) / costs.miqp
    metrics = dict(
        n_cases=12, independent_paths=4, scalar_wall_s=float(times.scalar),
        miqp_wall_s=float(times.miqp), batched_wall_s=float(times.batched),
        wall_ratio=float(times.miqp / times.scalar),
        cost_premium_percent=float(100 * (cost_means.scalar / cost_means.miqp - 1)),
        reference_reduction_percent=float(100 * (1 - cost_means.scalar / cost_means.reference)),
        min_case_premium_percent=float(relative.min()), max_case_premium_percent=float(relative.max()),
        historical_offline_s=historical, learned_total_with_offline_s=float(historical + times.scalar),
        projected_compute_only_crossover_cases=math.ceil(historical / ((times.miqp - times.scalar) / 12)),
        max_scalar_batch_action_difference_kw=max_action_difference,
        max_scalar_batch_cost_difference_eur=max_cost_difference,
        miqp_status_counts=pd.Series([s["status"] for s in solves]).value_counts().to_dict(),
        primary_miqp_calls=len(solves),
        model_loading_and_warmup_s=read(results / "neural_setup.json")["model_loading_and_warmup_s"]["cpu"],
        inputs=[dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in input_paths],
    )
    return timing, summaries, quality, metrics


def render(timing, quality, m):
    abstract = (
        f"A fixed-model CPU reuse benchmark took {m['scalar_wall_s']:.1f} s versus "
        f"{m['miqp_wall_s']:.1f} s for parallel MIQP over 12 path--initial-state combinations, "
        f"excluding setup and training, with {m['cost_premium_percent']:.1f}\\% higher mean operating cost.\n")
    main = (
        r"\paragraph{Reuse in repeated simulation}" + "\n"
        "The reuse benchmark applies one selected central-winter model to four\n"
        "model-generated disturbance paths and three initial SoCs, with all other\n"
        "settings fixed. For these 12 combinations, CPU simulation took\n"
        f"{m['scalar_wall_s']:.1f} s with the learned controller versus {m['miqp_wall_s']:.1f} s with up to four\n"
        f"single-threaded MIQP workers, a {m['wall_ratio']:.1f}-fold wall-time ratio. Mean operating\n"
        f"cost was {m['cost_premium_percent']:.1f}\\% higher than MIQP and {m['reference_reduction_percent']:.1f}\\% lower than the no-learning\n"
        "reference. Timings exclude setup and training. Adding the selected batch's\n"
        f"{m['historical_offline_s']:.1f} s of training and validation leaves offline computation unamortized\n"
        "at this workload. The comparison measures repeated evaluation within a\n"
        "fixed tariff and seasonal model. The supplement reports timing\n"
        "repetitions and cost variation.\n")
    lines = [
        r"Tables~\ref{tab:reuse-time} and~\ref{tab:reuse-cost} report computation and operating cost separately.", "",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Fixed-model reuse: CPU simulation wall time in seconds. Learned and reference entries are medians of three repetitions; MIQP has one timed repetition per workload. Each method excludes setup and training. The reference and learned policies use one process; MIQP uses one worker for $N=1$ and four for $N=4,12$, with one thread per worker.}",
        r"\label{tab:reuse-time}", r"\begin{tabular}{rrrrr}", r"\toprule",
        r"Cases $N$ & Reference & Learned & Batched learned & MIQP \\", r"\midrule",
    ]
    for n in (1, 4, 12):
        times = timing[timing.n == n].set_index("method").median_wall_s
        lines.append(f"{n} & {times.reference:.3f} & {times.scalar:.2f} & {times.batched:.2f} & {times.miqp:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"\begin{table}[htbp]", r"\centering",
              r"\caption{Mean daily operating cost over four disturbance paths, by initial SoC. Costs are in EUR/day. Premium is the percentage difference of mean learned and MIQP costs. Cost statistics use one $N=12$ run; timing repetitions use the same cases.}",
              r"\label{tab:reuse-cost}", r"\begin{tabular}{rrrrr}", r"\toprule",
              r"Initial SoC & Reference & Learned & MIQP & Premium \\", r"\midrule"]
    for s in (0.2, 0.5, 0.8):
        costs = quality[quality.initial_soc == s].groupby("method").cost.mean()
        premium = 100 * (costs.scalar / costs.miqp - 1)
        lines.append(f"{s:.1f} & {costs.reference:.1f} & {costs.scalar:.1f} & {costs.miqp:.1f} & {premium:.2f}" + r"\% \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              f"The overall mean cost premium was {m['cost_premium_percent']:.1f}\\%, while individual-case premiums",
              f"ranged from {m['min_case_premium_percent']:.2f}\\% to {m['max_case_premium_percent']:.2f}\\%.",
              f"At $N=12$, CPU batching took {m['batched_wall_s']:.1f} s and was slightly slower than scalar",
              "evaluation. The two implementations returned the same daily costs,",
              r"with a maximum action discrepancy below $10^{-12}$ kW across all repetitions.",
              "Both reuse implementations evaluate the same learned policy.", "",
              f"The selected batch's recorded training and validation time was {m['historical_offline_s']:.1f} s.",
              f"Including it gives {m['learned_total_with_offline_s']:.1f} s for learning at $N=12$, versus",
              f"{m['miqp_wall_s']:.1f} s for MIQP. Extrapolating the measured $N=12$ throughput at the same",
              f"resource allocation gives a compute-only crossover near {m['projected_compute_only_crossover_cases']} day--initial-state",
              "cases. This throughput-based projection concerns computation alone;",
              "economic break-even also depends on the operating-cost difference.", ""]
    return {"fixed_value_reuse_abstract.tex": abstract, "fixed_value_reuse_main.tex": main,
            "fixed_value_reuse_supplement.tex": "\n".join(lines)}


def main():
    raw_timing, timing, quality, metrics = collect()
    raw_timing.to_csv(RESULTS / "manuscript_timing_records.csv", index=False)
    timing.to_csv(RESULTS / "manuscript_timing_summary.csv", index=False)
    quality.to_csv(RESULTS / "manuscript_quality.csv", index=False)
    (RESULTS / "manuscript_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for name, content in render(timing, quality, metrics).items():
        (GENERATED / name).write_text(content, encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
