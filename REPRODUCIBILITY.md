# Reproducibility notes

Version `v1.0.0` contains the code, processed inputs, stored weights, and
evaluation outputs underlying the accompanying manuscript. Inspecting the
reported results does not require retraining or rerunning the online solvers.

## Evaluation design

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

`Exact-band` describes the mixed-integer representation of the discontinuous
fee, not a guarantee that every time-limited solve is optimal. The stochastic
comparator is a two-stage approximation with a shared first action and
scenario-contingent future recourse. Solver status, feasible incumbent
availability, relative gap, and time-limit frequency are reported separately.

The included outputs use `deployment_version = hard-band-aligned-v1` and
`action_search_version = closed-feasible-grid-v1`. The numerical implementation
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
python -m pytest tests -q
python experiments/verify_round2_results.py
python experiments/verify_alignment_results.py
python experiments/audit_release_inputs.py --workspace .
~~~

These commands check numerical regressions, the stored result coverage,
accounting identities, MIQP fee-side consistency, validation decisions, and
checkpoint weights. The full input audit is more substantial than the result
checks: it replays logged solver actions and reconstructs validation selection,
without retraining or reoptimizing MIQPs.

The included audit `experiments/results/corrected_input_audit.json` records
4,080 daily rows, 118,080 solve rows, 74 validation selections, and 8,880
validation rows. It independently replays 1,230 solver-days and reconstructs
10 batch selections. These are numerical consistency checks, not a proof of
neural-training convergence or global continuous-state error bounds.

`submission_ready` fields in numerical audit files do not assess manuscript
formatting or journal submission. Document checks are reported separately by
the manuscript format checker below.

## Regenerate summaries and figures

~~~bash
python experiments/analyze_confirmatory.py
python experiments/analyze_seasonal_sensitivity.py
python experiments/analyze_round2_reference.py
python experiments/analyze_round2_scenarios.py
python experiments/make_round2_parameter_tables.py
python experiments/make_v3_figures.py
python experiments/make_round2_mechanism_figure.py --controller learned
python experiments/make_round2_mechanism_figure.py --controller reference
python experiments/make_round2_mechanism_figure.py --controller m2
python experiments/make_round2_mechanism_figure.py --controller m16
python experiments/make_round2_mechanism_figure.py --assemble
~~~

The representative-day MIQP figure replays the 96 logged held actions; it does
not reoptimize the time-limited problems. Its call latencies are the original
measurements, not replay times. Learned/reference trajectory generation checks
economic outputs against the stored day and retains source and trajectory
hashes. Neural reevaluation uses the GPU setting of the reported run.

The reference, scenario, and seasonal exporters reproduce the distributed
tables and their explanatory text. Regression tests compare their output with
the manuscript, including the interpretation of conditional solver gaps.
Run numerical reporting in a working copy to retain the distributed PDFs
until the regenerated sources have been compiled and checked.

## Compile the manuscript

~~~bash
python experiments/compile_manuscripts.py
python experiments/audit_manuscript_format.py --release
~~~

The compiler uses `pdflatex`, `bibtex`, and two further LaTeX passes for each
PDF. The official `elsarticle` class and numeric bibliography style are
included. `pdfinfo` and `pdffonts` must also be available. The local format
check covers the 30-page main-paper cap, abstract, keywords, highlights,
references, layout, and source/PDF synchronization; it does not assess
editorial acceptance or complete author-side submission declarations.

## Included artifacts

The repository includes processed data, per-day evaluations, per-solve MIQP
diagnostics, validation-selection records, audit outputs, generated tables,
manuscript PDFs, and approximately 94.4 MiB of checkpoint weights and records.
Data source URLs, input checksums, and upstream licensing information remain
with the data manifests.

Run metadata use repository-relative paths for portable verification and
reporting. Checkpoint weights and numerical evaluation artifacts retain their
original bytes; checksums for text metadata identify the distributed files.
Experimental settings, selection decisions, seeds, and recorded timings are
unchanged. Internal script names and numerical-version identifiers are retained
so that stored configurations and outputs remain compatible.

Temporary scheduling workspaces, duplicate archives, private review documents,
and build caches are not part of this distribution. Long-running experiment
drivers are for deliberate new evaluations, not for checking the stored
findings. PowerShell drivers use `python` unless `RVPINN_PYTHON` specifies
another interpreter.
