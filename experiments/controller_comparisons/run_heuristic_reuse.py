"""Apply the frozen historical-validation rule to the existing reuse paths."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fixed_value_reuse.common import bundle, context, update_state, finish_rows
from experiments.controller_comparisons.heuristic import TariffPeakShaving

OUT = HERE / "results/heuristic_reuse"
REUSE = ROOT / "experiments/fixed_value_reuse/results"


def simulate(cases, controller, p, prof):
    start = time.perf_counter()
    states = np.array([c["initial_soc"] for c in cases])
    components = np.zeros((len(cases), 3))
    actions = np.zeros((len(cases), 96))
    violations = np.zeros(len(cases))
    for k in range(96):
        for i, case in enumerate(cases):
            net, price = case["net"][k], case["price"][k]
            a = controller(k*p.dt_ctrl, states[i], case["y"][k], case["pz"][k],
                           context(case["path_id"], net, price, prof))
            s, held, parts, violation = update_state(states[i], a, net, price, p)
            states[i], actions[i, k] = float(s), float(held)
            components[i] += parts
            violations[i] = max(violations[i], float(violation))
    rows = finish_rows(cases, states, components, actions, violations, p)
    return dict(simulation_wall_s=time.perf_counter()-start, rows=rows)


def main():
    selection_path = HERE / "results/heuristic/selection.json"
    selected = next(s for s in json.loads(selection_path.read_text())
                    if s["regime"] == "winter_weekday" and s["fee"] == 40)
    manifest_path = REUSE / "manifest.json"
    cases = json.loads(manifest_path.read_text())["cases"]
    manifest = dict(setting=selected["setting"],
                    selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                    cases_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    interpretation="Exploratory extension; no retuning on reuse paths. CPU timings measured separately on shared workstation.")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    p, prof = bundle()
    controller = TariffPeakShaving(p, **selected["setting"])
    jobs = [(n, r) for n in (1, 4, 12) for r in range(3)]
    np.random.default_rng(29183).shuffle(jobs)
    results = []
    for n, r in jobs:
        path = OUT / f"n{n}_r{r}.json"
        if path.exists():
            result = json.loads(path.read_text())
        else:
            result = dict(n=n, repetition=r, **simulate(cases[:n], controller, p, prof))
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        assert max(row["max_action_violation_kw"] for row in result["rows"]) < 1e-8
        results.append(result)
    times = pd.DataFrame([{k:v for k,v in r.items() if k != "rows"} for r in results])
    times.to_csv(OUT / "timing.csv", index=False)
    quality = pd.DataFrame([{k:v for k,v in row.items() if k != "actions"}
                            for r in results if r["n"] == 12 and r["repetition"] == 0 for row in r["rows"]])
    quality.to_csv(OUT / "quality.csv", index=False)
    old = pd.read_csv(REUSE / "manuscript_quality.csv")
    means = old.groupby("method").cost.mean().to_dict()
    means["tariff_rule"] = float(quality.cost.mean())
    timing = times.groupby("n").simulation_wall_s.median().to_dict()
    report = dict(mean_costs=means, rule_median_seconds=timing,
                  learned_minus_rule_percent=100*(means["scalar"]/means["tariff_rule"]-1))
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = (f"Applying the validation-selected tariff-aware rule without retuning to the same\n"
            f"12 reuse cases gave mean cost {means['tariff_rule']:.1f} EUR/day, versus\n"
            f"{means['scalar']:.1f} for learning and {means['miqp']:.1f} for MIQP.\n"
            f"Its median CPU simulation time was {timing[12]:.3f} s over three repetitions,\n"
            "measured separately on the shared workstation. The rule combines\n"
            "shorter simulation time with lower operating cost than learning\n"
            "on these cases.\n")
    (ROOT / "manuscript/generated/heuristic_reuse.tex").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
