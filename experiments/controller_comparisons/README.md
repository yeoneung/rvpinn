# Controller comparisons and sensitivity

These experiments compare tariff-aware rules, convex penalty slopes, and
online solve budgets under the common battery-dispatch objective. They use
the same learned weights and primary evaluation data. The sensitivity and
reuse analyses are exploratory, separate from the primary inferential tests.

## Frozen design

`protocol.json` fixes the settings and sample dates before the added runs.
Each runner saves a hash manifest and refuses a changed protocol on resume.
`surrogate_implementation.json` additionally specifies the hinge controllers'
threshold-aware floating-point treatment before their evaluations.

- **Tariff-aware rule:** 96 reserve/recharge/price/switch settings plus a
  zero-action incumbent, selected on 30 validation days. All six selections
  finish before any test evaluation. Tests cover three winter fees and three
  additional seasons at the central fee; seasonal rules are retuned separately.
- **Solver effort:** 576 fixed-state solves, using two validation dates, two
  dispatch times, three SoCs, three nested scenario streams, two fees, two
  scenario counts, and four budgets (5, 15, 60, 300 seconds). Scenario-array
  hashes must agree across matched budgets.
- **Closed-loop budget comparison:** 36 full-day evaluations, using the first
  three winter test dates, fees 40/80, M=8/16, stream 7301, and budgets
  5/60/300 seconds. This smaller sample diagnoses budget sensitivity rather
  than replacing the 30-day primary test.
- **Convex penalty:** the original greatest lower envelope and eight hinge
  slopes, with effective widths 5, 10, 20, 50, 100, 250, 500, 1000 kW.
  All 810 validation-day runs precede selection and the 90 selected test-day
  runs. Selection starts at the envelope, replacing it only for at least
  0.10 EUR/day lower validation hard cost. The code's `softcap` mode is a
  linear hinge, not a capped penalty. General slopes need not minorize the fee.
- **Reuse extension:** the fixed central-winter heuristic is applied without
  retuning to the existing four paths and three initial states. No learned or
  MIQP reuse results are recomputed. Timing repetitions are not new quality
  samples.

The common objective always bills the strict indicator `grid > threshold`.
For hinge controllers, a raw overshoot of at most 1e-5 kW can be moved to a
verified feasible fee-inactive crossing; the displacement and held action
are recorded. This is an action adjustment, not a billing tolerance.

## Running and reporting

From the repository root, in an environment containing the recorded study
dependencies and PySCIPOpt:

```sh
python experiments/controller_comparisons/run_heuristic.py
python experiments/controller_comparisons/analyze_heuristic.py
python experiments/controller_comparisons/run_heuristic_reuse.py
python experiments/controller_comparisons/run_solver_budget.py --kind fixed --workers 4
python experiments/controller_comparisons/run_solver_budget.py --kind closed --workers 4
python experiments/controller_comparisons/run_surrogates.py --workers 2
python experiments/controller_comparisons/analyze_solver_budget.py
python experiments/controller_comparisons/analyze_surrogates.py
python experiments/controller_comparisons/make_cost_latency.py
python experiments/controller_comparisons/make_heuristic_mechanism.py
python experiments/controller_comparisons/make_compact_attribution.py
python experiments/controller_comparisons/make_compact_scenarios.py
python experiments/controller_comparisons/audit_results.py --replay-heuristic
python -m pytest experiments/controller_comparisons -q
```

Do not run all CPU pools simultaneously without checking available resources.
Each solver worker is one-threaded and Torch-free. This avoids the Windows
OpenMP-runtime conflict; these MIQP solves do not use CUDA. The added rules
also need no neural evaluations. Completed per-job results are skipped on
resume. An interrupted full-day solver run restarts that entire day, not a
changed within-day continuation. Native solver logs are retained per job.
Do not edit a running experiment's source or frozen manifest.

`run_closed_adaptive.py` is an alternative executor for the same frozen
closed-loop manifest, not a different controller. Use it only after the
surrogate sweep is complete and the original closed-loop executor has
stopped. It allows four active jobs while the fixed-state pool is running,
then eight, and records allocation events. Never run both closed-loop
executors concurrently. In the recorded schedule all 12 five-second days
were preserved; two unfinished 60-second days restarted after their partial
logs were retained during the resource transition.

Analysis scripts read stored results without replaying MIQP. The budget and
surrogate manuscript exporters require complete planned results. For a
non-publishing progress report use `analyze_solver_budget.py --partial`.
The independent audit spells out the state update, hard cost, and feasible
bounds rather than calling the simulation's cost function. `run_day` records
trajectory SoC after the first five-minute substep; the audit respects this
recording convention and starts each historical day from configured `s0`.
For the stored fixed-state and closed-loop exact-band records, the audit also
checks exact equality of recorded and held actions, logged recovery amounts,
and agreement between the band binary and the realized strict fee side.
These are checks of observed records, not certificates of solver dual bounds.

## Budget findings and accounting checks

The budget sweep contains 576 fixed-state
solves and 36 closed-loop days (3456 calls), with no omitted jobs. M=8 reverses
the five-second cost ranking against learning at 60 seconds, and M=16 at
300 seconds, at both fees on the three common dates. The high-fee M=8 mean
nevertheless rises from 1104.8 to 1115.0 EUR/day between 60 and 300 seconds;
solver progress does not guarantee lower historical replay cost. Full fixed
and closed-loop summaries, including finite-gap availability and P95 gaps,
are in `results/solver_budget/`. The 36-day design remains exploratory.

The final independent accounting audit covers 1167 stored/replayed trajectories
and 4032 fixed/closed budget action records. Maximum discrepancies are below
1e-8 EUR in cost and 1e-8 in the corresponding state/action units. These are
implementation checks, not additional statistically independent test samples
or certificates of global optimality.

## Rule findings

On 30 winter test dates, the tariff-aware rule costs 1271.5, 1458.4, and
1744.7 EUR/day at fees 20, 40, and 80 EUR/h, respectively. Learned costs are
1258.2, 1489.1, and 1769.7. The rule also has lower mean cost in all three
additional central-fee seasonal comparisons. The reference-only reduction
therefore identifies an architectural correction, not general superiority
to practical non-learning control.

On the 12 stored reuse cases, rule, learned, and primary MIQP mean costs are
1406.0, 1452.9, and 1399.6 EUR/day. Rule CPU simulation takes a median
0.142 seconds over three repetitions, measured separately on the shared
workstation. The learned controller's faster repeated simulation than MIQP
is not a unique advantage over the rule.

Raw outputs, selections, timing repetitions, summaries, and audits are under
`results/`. This repository distributes experiments, not submission documents.
Default tests validate the experiments without requiring the manuscript;
`RVPINN_CHECK_MANUSCRIPT=1` enables optional local manuscript-consistency checks.

## Convex-surrogate findings

Validation selects the envelope at fees 20 and 40, and the hinge of width
1000 kW (slope 0.08 EUR/(kW h)) at fee 80. Their held-out hard costs are
1315.2, 1750.6, and 2623.8 EUR/day. All 8640 selected test solves returned
an incumbent, and none reached the five-second limit. Thus the selected
controllers' higher realized costs cannot be attributed to time-limit
termination in these test runs. This result concerns the tested, validation-
selected penalty family, not every possible convex controller.

The high-fee selection improves validation cost but not the historical test
mean relative to the envelope comparator; no test-based reselection is
made. All 810 validation-day outputs, 90 selected test-day outputs, solver
records, and native logs are retained. The sweep supports accounting for the
hard tariff, not a unique advantage of neural learning over the tariff-aware
rule.
