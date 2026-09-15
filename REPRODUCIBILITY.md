# Reproducibility notes

The repository contains code, processed inputs, stored weights, and outputs
for historical replay, comparator sensitivity, policy-iteration progression,
fixed-model reuse, and the hard-tariff reduced-model DP comparison.
Submission documents are maintained separately. Inspecting the reported
results does not require retraining or rerunning the online solvers.

## Evaluation design

The following historical design is separate from the model-based
[hard-tariff DP comparison](experiments/hard_tariff_dp/README.md).
That benchmark retains the central winter tariff with deterministic price,
five new two-state training seeds, and three state-grid resolutions. Its
checkpoint and rule selection uses coarse-grid expected cost, not historical
validation or test outcomes. The learned continuation is tabulated for a
common numerical transition; an 88-state audit compares the resulting search
with direct neural evaluation. The frozen protocol, weights, numerical
outputs, and scripts are in `experiments/hard_tariff_dp/`.

- Geographic holdout: IT_NORD.
- Training years: 2016--2017; validation year: 2018; test year: 2019.
- Primary test set: first 30 eligible winter weekdays.
- Decision interval: 15 minutes, with five-minute plant substeps.
- Primary threshold and smoothing width: 300 kW and 10 kW.
- Fee anchors: 20, 40, and 80 EUR/h.
- Central anchor: five independent batches of five training restarts.
- Outer anchors and noncentral seasonal cells: one five-restart batch.
- Seasonal evaluation: first 30 eligible weekdays within each season.

All controllers use the same hard economic objective, observed information,
terminal treatment, and feasible action set. Neural training uses a smooth
objective; deployment and reporting use the hard tariff. Validation selection
uses 2018 costs and includes a feasible zero-action incumbent.

The primary comparisons use paired daily outcomes. Central-fee uncertainty
resamples complete training batches and paired seven-day circular date blocks;
outer-fee uncertainty is conditional on the available training batch. An
interval containing zero is inconclusive, not evidence of equivalence.

Seasonal and factorial comparisons are descriptive sensitivity analyses.
The matched no-learning controller and expanded scenario study are additional
attribution and sensitivity analyses; they do not replace the primary tests.
The nested scenario study retains all three streams (7301, 7302, 7303), all
scenario counts (2, 8, 16), and both executed high-fee solver budgets (5 and
15 seconds). No scenario count is selected by test cost. The representative
day is selected by median no-storage threshold-exceedance duration, with ties
resolved by the earliest date, not by learned or MIQP performance.

## Comparator and numerical conventions

The added tariff-aware rule uses 96 settings plus zero action, with all
validation selections completed before test evaluation. Its six test cells
cover three winter fees and three additional central-fee seasons. Seasonal
rules are selected separately. A second extension compares the envelope with
eight hinge slopes, selecting on validation hard cost before 90 selected
test-day runs. In this slope family the incumbent is the envelope, not the
heuristic grid's zero-action incumbent. The `softcap` code mode denotes an
uncapped linear hinge. The frozen runner and
`experiments/controller_comparisons/surrogate_implementation.json` specify the deployed
threshold treatment; billing remains the strict hard indicator.

Longer-budget MIQP experiments use 576 matched-state solves and 36
three-date closed-loop day runs. Individual completions and unchanged scenario
hashes are retained. No neural model or original primary result is rerun.
The closed-loop executor can expand from four to eight active one-thread
jobs when the fixed-state pool finishes; scheduling changes do not change
the model, scenario stream, solve limit, or validation/test selection.
See `experiments/controller_comparisons/README.md` for scripts and result locations.

The exploratory fixed-model reuse design is separate from historical replay:
one validation-selected model, four independent model-generated full-day
disturbance paths, and three initial SoCs. Its 12 combinations are not 12
independent paths. No retraining or tariff/season transfer is involved. The
stored results preserve per-action decisions, realized costs, solve statuses,
and all timing repetitions. Regenerate the manuscript summary without online
solving with `python experiments/analyze_fixed_value_reuse.py`; see
[the reuse protocol notes](experiments/fixed_value_reuse/README.md) for reruns.
The compute-only crossover is an extrapolation of measured throughput, not a
measured crossover or financial break-even claim.

`Exact-band` describes the mixed-integer representation of the discontinuous
fee, not a guarantee that every time-limited solve is optimal. The stochastic
comparator is a two-stage approximation with a shared first action and
scenario-contingent future recourse. Solver status, feasible incumbent
availability, relative gap, and time-limit frequency are reported separately.

The daily replay outputs use `deployment_version = hard-band-aligned-v1`;
the learned one-step search uses `action_search_version = closed-feasible-grid-v1`.
This search identifier does not apply to the MIQP and tariff-rule controllers.
The numerical implementation
includes strict threshold billing, bit-preserving safety projection,
current-observation accounting, a closed feasible action grid, and physically
feasible first-action recovery from MIQP incumbents. Aggregators reject
incompatible or mixed numerical-version records.

## Environment and verification

Use `environment.yml` to create the environment. Observed direct-dependency
versions are in `requirements-reproduction.txt`; `requirements.txt` gives
lower bounds. The observed dependency record is not a full transitive lock or
a fresh-install test. Hardware, CUDA, Python, and timing details are in
[HARDWARE_SOFTWARE.md](HARDWARE_SOFTWARE.md).

~~~bash
python -m pytest tests experiments/controller_comparisons experiments/hard_tariff_dp -q
python experiments/verify_comparative_results.py
python experiments/verify_evaluation_results.py
python experiments/audit_release_inputs.py --workspace .
~~~

These commands check numerical regressions, the stored result coverage,
accounting identities, MIQP fee-side consistency, validation decisions, and
checkpoint weights. The full input audit is more substantial than the result
checks: it replays logged solver actions and reconstructs validation selection,
without retraining or reoptimizing MIQPs.

The included audit `experiments/results/input_audit.json` records
4,080 daily rows, 118,080 solve rows, 74 validation selections, and 8,880
validation rows. It independently replays 1,230 solver-days and reconstructs
10 batch selections. These are numerical consistency checks, not a proof of
neural-training convergence or global continuous-state error bounds.

Manuscript formatting is checked in the separately maintained submission
package, not by the numerical audits.

## Regenerate summaries and figures

~~~bash
python experiments/analyze_confirmatory.py
python experiments/analyze_seasonal_sensitivity.py
python experiments/analyze_comparative_reference.py
python experiments/analyze_comparative_scenarios.py
python experiments/make_comparative_parameter_tables.py
python experiments/make_study_figures.py
python experiments/analyze_policy_iteration_progression.py
python experiments/analyze_fixed_value_reuse.py
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

These commands require the complete planned outputs; budget and surrogate
exporters refuse an incomplete experiment. The heuristic reuse runner reads
its nine retained timing records when present, without repeating them.
The representative-day figure reads stored learned, primary-MIQP, and
tariff-rule trajectories and checks their accounting. It does not rerun
controllers. Its reported controller
latencies are original measurements, not figure-generation times.

The optional core exporter `experiments/make_comparative_mechanism_figure.py`
retains learned, reference, M=2, and M=16 trajectory checks. Its MIQP modes
replay logged actions; its learned/reference modes reevaluate those policies,
with the reported GPU setting for learning. These reevaluations are not
required to regenerate the current main figure from the stored trajectories.

The reference, scenario, and seasonal exporters reproduce the distributed
tables and their explanatory text. Regression tests check reporting against
numerical records, including the interpretation of conditional solver gaps.
Set `RVPINN_CHECK_MANUSCRIPT=1` to additionally check the current manuscript
sources when they are available locally. These paper checks are disabled by
default in the experiments-only release; numerical tests remain enabled.
Run these exporters in the order shown: composite reports read earlier
generated summaries. They create local LaTeX tables and figure PDFs under
`manuscript/`, which is an ignored output directory, not a distributed paper.

## Included artifacts

The repository includes processed data, per-day evaluations, per-solve MIQP
diagnostics, validation-selection records, audit outputs, and table/figure
exporters. The main checkpoint tree contains approximately 94.4 MiB of
weights and records; the five DP training seeds are stored in their experiment directory.
Data source URLs, input checksums, and upstream licensing information remain
with the data manifests.

Run metadata use repository-relative paths for portable verification and
reporting. Checksums identify the distributed input and checkpoint files.
The configurations and manifests specify experimental settings, validation
selection, seeds, and timing measurements.

Temporary scheduling workspaces and build caches are not distributed. Long-running experiment
drivers are for deliberate new evaluations, not for checking the stored
findings. PowerShell drivers use `python` unless `RVPINN_PYTHON` specifies
another interpreter.
