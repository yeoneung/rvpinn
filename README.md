# Simulation-based comparison of offline value learning and online mixed-integer control for battery dispatch

Code, processed data, stored model weights, and evaluation outputs for the
study by **Yeoneung Kim**, Department of Industrial Engineering,
Yonsei University.

The current branch contains the experimental implementation, settings, and
results, including tariff-aware control, convex-slope sensitivity, extended
solver budgets, policy-iteration progression, fixed-model reuse, and a
hard-tariff reduced-model dynamic-programming comparison.

- [Controller comparisons and sensitivity](experiments/controller_comparisons/)
- [Fixed-model reuse](experiments/fixed_value_reuse/)
- [Hard-tariff DP comparison and five-seed results](experiments/hard_tariff_dp/)
- [Reproducibility notes](REPRODUCIBILITY.md)

This repository distributes experimental code, data, weights, and results.
Submission documents are maintained separately. Stored results and selected
weights can be inspected without retraining.

The study compares offline stochastic value learning, tariff-aware rules, and
rolling-horizon exact-band MIQP controllers for battery dispatch under a
discontinuous import-band fee. All reported controllers share the hard
economic objective, current observations, terminal treatment, feasible action set, and
15-minute decision interval.

## Main findings

The two-state hard-tariff benchmark compares policies under a common refined
numerical transition, retaining the 15-minute held action and terminal cost.
Coarse-grid expected cost selects a checkpoint from five training seeds. From
(SoC, load error) = (0.5, 0), the finest-grid expected daily costs are
1348.12 EUR for DP, 1385.15 EUR for learning, 1372.88 EUR for the tariff-aware
rule, and 1627.70 EUR for the reference-only policy. The learned and rule
gaps to DP are 2.75% and 1.84%. The last DP grid refinement changes cost
by 0.21 EUR/day. These are fitted-model accuracy results, not historical
replay costs or a global optimality certificate. The experiment directory
contains all five seeds' checkpoints, selection records, grid-refinement
outputs, boundary regrets, and a direct-neural-search check.

Deployment rules matter independently of value learning. Narrowing the smooth
tariff width does not remove its discrepancy with the billed fee at the
threshold. In the central-fee deployment ablation, optimizing over a larger
action set and then projecting costs 17.1 EUR/day more than direct minimization
over feasible held actions. The matched no-learning controller then isolates
the contribution of the learned continuation correction.

The primary evaluation uses the first 30 eligible 2019 winter weekdays from
IT_NORD; seasonal replication uses 30 eligible weekdays per season. Central-fee
inference includes five independent five-restart batches; outer-fee inference
is conditional on one batch. Nested scenario comparisons retain all three
prespecified streams and both evaluated solver budgets.

The learned-policy operating cost is 1.18%, 2.27%, and 1.35% above
primary stochastic MIQP at fees 20, 40, and 80 EUR/h. Only the high-fee
conditional-mean MIQP comparison meets the prespecified learned-superiority
rule. The matched no-learning reference costs more than learning at all three
anchors. These are distinct results. Historical neural online timing uses the GPU;
each MIQP solve uses one CPU thread. Across fees, their mean online latencies
are 3.1 ms and 1955.6 ms for learning and primary stochastic MIQP, respectively.
Both are below the 15-minute decision interval; lower latency does not by
itself establish an economic or binding response-time advantage.

A validation-tuned tariff-aware rule costs 1271.5, 1458.4, and
1744.7 EUR/day at the three winter fee anchors. It therefore costs less than
learning at fees 40 and 80, while learning costs less at fee 20. The rule
also has lower mean cost in every central-fee seasonal comparison. Its mean
winter controller-call latency is about 0.05 ms on the CPU, measured separately
on the shared workstation. The reference-only ablation isolates the neural
correction; the tariff-aware rule provides a practical non-learning comparison.

A nine-setting convex-hinge validation sweep retains the envelope at fees
20/40 and selects an effective width of 1000 kW at fee 80. The selected
surrogate's test costs are 1315.2, 1750.6, and 2623.8 EUR/day: all higher than
both learning and the tariff-aware rule. Full validation settings, selected
test outcomes, and solver diagnostics are retained, including the absence
of time-limit terminations in the selected-hinge test runs.

Learning's operating cost exceeds primary stochastic MIQP in all four seasonal means: 2.27% in
winter, 1.76% in spring, 2.24% in summer, and 5.47% in autumn at 40 EUR/h.
These seasonal contrasts are descriptive, not additional confirmatory tests.

Across the 25 central-fee training seeds, mean validation cost fell by 12.4%
from the first trained round to the third, with improvement for all 25 seeds.
The fourth round increased cost for 23 seeds. The manuscript reports all four
rounds and their selection frequencies. This diagnostic uses saved validation
records; it is not a held-out or equal-training-budget policy-update ablation.

At 40 EUR/h and a five-second solve budget, nested-scenario MIQP mean costs
are 1456.1, 1680.3, and 1708.1 EUR/day for 2, 8, and 16 scenarios, versus
1489.1 EUR/day for learning. At 80 EUR/h, the 16-scenario cost falls from
2469.5 to 2342.1 EUR/day when the solve limit increases from 5 to 15 seconds,
but 96.4% of the latter calls still reach the time limit. Finite-gap
availability changes from 48.5% to 97.7%; the conditional mean gaps of 897.5%
and 1167.0% are not a matched comparison of solver progress. These results
concern budgeted implementations, not globally solved stochastic control.
The finding is an operating-cost/computation trade-off, not general learned
cost dominance.

The budget sensitivity analysis contains 576 matched-state solves and
36 closed-loop days. On three common test dates, the five-second ranking
against learning reverses for M=8 at 60 seconds and M=16 at 300 seconds at
both fees. This is descriptive evidence, not a new 30-day superiority test.

| Fee (EUR/h) | Scenarios | 5 s cost | 60 s cost | 300 s cost | Learned cost |
| --- | --- | --- | --- | --- | --- |
| 40 | 8 | 1257.5 | 956.4 | 953.9 | 1007.9 |
| 40 | 16 | 1299.9 | 1040.4 | 958.0 | 1007.9 |
| 80 | 8 | 1789.1 | 1104.8 | 1115.0 | 1138.8 |
| 80 | 16 | 1927.8 | 1293.1 | 1113.1 | 1138.8 |

Costs are EUR/day on these dates only. The rule costs 955.2 and 1108.5;
primary M=2 costs 962.5 and 1118.6, using stream 4101 independently of
the nested stream 7301. More solve time is not uniformly
beneficial: high-fee M=8 costs 10.2 EUR/day more at 300 than at 60 seconds,
despite a lower median finite solver gap. Computation is timed separately
and does not advance simulated time; actuation delays are not modeled.

An exploratory fixed-model reuse benchmark applies one selected central-winter
value function to four model-generated paths and three initial SoCs. Over the
12 combinations, CPU simulation takes 39.5 s for scalar learned control versus
620.3 s for MIQP using up to four one-thread workers. Mean operating cost is
3.8% higher than MIQP. Setup and
training are excluded from these timings. Including the recorded 1,403.2 s of
training and selection means offline work is not yet amortized at 12 cases.
This is reuse within a fixed model, not cross-tariff transfer or reproduction
of MIQP decisions at lower cost. Protocol, full trajectories, and timing
records are in [experiments/fixed_value_reuse](experiments/fixed_value_reuse/).
The same validation-selected tariff rule, without retuning on these paths,
costs 1406.0 EUR/day versus 1452.9 for learning and 1399.6 for MIQP. Its
median CPU simulation time is 0.142 seconds over three repetitions. Reuse
speed against MIQP alone consequently does not establish the value of learning.

The comparator sensitivity studies and their frozen designs are in
[experiments/controller_comparisons](experiments/controller_comparisons/). They are exploratory
analyses, separate from the primary confirmatory tests.

## Repository layout

~~~text
configs/              frozen experiment and model specifications
data/                 processed inputs, splits, calibration, and checksums
experiments/          comparator, audit, analysis, and figure scripts
experiments/results/  daily, solve-level, and summary evaluation outputs
checkpoints/          stored weights and validation-selection records
results/              gate and supporting numerical outputs
scripts/              preprocessing, training, and core evaluation pipeline
src/                  dynamics, costs, policies, safety, and utilities
tests/                numerical and reproducibility tests
~~~

The checkpoint tree is approximately 94.4 MiB and includes stored weights
and selection records, so replay does not depend on stochastic retraining.
The parameter-table exporter uses `data/checkpoint_metadata.json`. The
controller_comparisons and hard-tariff DP directories contain their own comparison
records. Run metadata use portable repository-relative paths;
experimental settings and numerical-version identifiers are retained for
reproducibility. Checksums identify the distributed files.

## Environment and checks

~~~bash
conda env create -f environment.yml -n rvpinn
conda activate rvpinn
python -m pytest tests experiments/controller_comparisons experiments/hard_tariff_dp -q
python experiments/verify_comparative_results.py
~~~

The implementation includes regression tests for bit-preserving
safety projection, current-observation accounting, strict threshold billing,
and physically feasible MIQP incumbents. Solver integration tests run in a
separate process to avoid loading conflicting Torch/SCIP OpenMP runtimes on
Windows. SCIP 10.0 with PySCIPOpt 6.2.1 is used for the mixed-integer models.

Result aggregators reject incompatible or mixed `deployment_version`
records. The additional release gate is:

~~~bash
python experiments/verify_evaluation_results.py
python experiments/audit_release_inputs.py --workspace .
~~~

The included input audit records checks of 4,080 daily rows and 118,080 solve
rows, independent replay of 1,230 solver-days, and reconstruction of 10 batch
selections. These numerical checks do not establish neural-training convergence
or global continuous-state error bounds. Stored outputs can be inspected
without rerunning the computationally substantial experiments.

Regenerate final statistics and tables with:

~~~bash
python experiments/analyze_confirmatory.py
python experiments/analyze_seasonal_sensitivity.py
python experiments/analyze_comparative_reference.py
python experiments/analyze_policy_iteration_progression.py
python experiments/analyze_fixed_value_reuse.py
python experiments/analyze_comparative_scenarios.py
python experiments/make_comparative_parameter_tables.py
python experiments/make_study_figures.py
python experiments/controller_comparisons/analyze_heuristic.py
python experiments/controller_comparisons/run_heuristic_reuse.py
python experiments/controller_comparisons/analyze_surrogates.py
python experiments/controller_comparisons/analyze_solver_budget.py
python experiments/controller_comparisons/make_compact_attribution.py
python experiments/controller_comparisons/make_compact_scenarios.py
python experiments/controller_comparisons/make_cost_latency.py
python experiments/controller_comparisons/make_heuristic_mechanism.py
python experiments/hard_tariff_dp/report.py
~~~

Current-paper source checks are optional because this repository excludes
the submission documents. With the matching current manuscript available,
run the full local consistency checks in PowerShell:

~~~powershell
$env:RVPINN_CHECK_MANUSCRIPT='1'
python -m pytest tests experiments/controller_comparisons experiments/hard_tariff_dp -q
~~~

The default tests still check numerical records, selection, and reporting
calculations. Only current-paper inclusion and source-equality checks are
optional. Regeneration commands create local tables under
`manuscript/generated/` and figures under `manuscript/figures/`; the DP exporter
writes its two result includes directly under `manuscript/`. These outputs
are not a standalone paper. Run the commands in the order shown so that
composite reports can read the preceding generated summaries.

PowerShell orchestration scripts use python by default. Set RVPINN_PYTHON to
an alternative interpreter path when required.

## Additional analyses

The reference-controller and scenario-resolution analyses are documented in
the supplement and reproducibility notes. To rerun them if needed:

~~~powershell
python experiments\run_comparative_reference.py
powershell -ExecutionPolicy Bypass -File experiments\run_comparative_scenario_sensitivity.ps1
powershell -ExecutionPolicy Bypass -File experiments\run_comparative_mechanism.ps1
python experiments\analyze_comparative_reference.py
python experiments\analyze_comparative_scenarios.py
~~~

The complete experiment is computationally substantial. All long-running
drivers are restartable from their saved per-day outputs.

## Data

The input is the Open Power System Data time-series package, release
2020-10-06. Processed inputs used in the paper are included. The original
source URL and SHA-256 checksum are retained with the data manifests.
Upstream data remain subject to their original licenses and attribution
requirements.

## License

The code is released under the [MIT License](LICENSE).
