# Simulation-based comparison of offline value learning and online mixed-integer control for battery dispatch

Code, processed data, stored model weights, evaluation outputs, and manuscript
sources for the study by **Yeoneung Kim**, Department of Industrial Engineering,
Yonsei University.

**Version 1.0.0 — 13 September 2026.** Research code and results accompanying
the manuscript. This repository does not imply journal acceptance or publication.

- [Main manuscript (PDF)](manuscript/main.pdf)
- [Supplementary material (PDF)](manuscript/supplement.pdf)
- [Fixed version used by the manuscript](https://github.com/yeoneung/rvpinn/tree/v1.0.0)
- [Reproducibility notes](REPRODUCIBILITY.md)
- [Version notes](RELEASE_NOTES.md)

The manuscript and supplement are 29 and 16 pages, respectively. Both compile
without unresolved references or overfull boxes. Stored results and selected
weights are included; retraining is not required to inspect these findings.

The study compares an offline stochastic value-learning procedure with
rolling-horizon exact-band MIQP controllers for battery dispatch under a
discontinuous import-band fee. All reported controllers share the hard
economic objective, information, terminal treatment, feasible action set, and
15-minute decision interval.

## Main findings

The primary evaluation uses the first 30 eligible 2019 winter weekdays from
IT_NORD; seasonal replication uses 30 eligible weekdays per season. Central-fee
inference includes five independent five-restart batches; outer-fee inference
is conditional on one batch. Nested scenario comparisons retain all three
prespecified streams and both evaluated solver budgets.

The learned-policy operating cost is 1.18%, 2.27%, and 1.35% above
primary stochastic MIQP at fees 20, 40, and 80 EUR/h. Only the high-fee
conditional-mean MIQP comparison meets the prespecified learned-superiority
rule. The matched no-learning reference costs more than learning at all three
anchors. These are distinct results. Neural online timing uses the GPU;
each MIQP solve uses one CPU thread.

The learned correction reduces cost relative to the matched no-learning
reference by 2.90%, 11.93%, and 24.57% at the three fee anchors. Its operating
cost exceeds primary stochastic MIQP in all four seasonal means: 2.27% in
winter, 1.76% in spring, 2.24% in summer, and 5.47% in autumn at 40 EUR/h.
These seasonal contrasts are descriptive, not additional confirmatory tests.

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

## Repository layout

~~~text
configs/              frozen experiment and model specifications
data/                 processed inputs, splits, calibration, and checksums
experiments/          comparator, audit, analysis, and figure scripts
experiments/results/  daily, solve-level, and summary evaluation outputs
checkpoints/          stored weights and validation-selection records
manuscript/           current manuscript sources, tables, figures, and PDFs
results/              gate and supporting numerical outputs
scripts/              preprocessing, training, and core evaluation pipeline
src/                  dynamics, costs, policies, safety, and utilities
tests/                numerical and reproducibility tests
~~~

The checkpoint tree is approximately 94.4 MiB and includes stored weights
and selection records, so replay does not depend on stochastic retraining.
The parameter-table exporter uses `data/checkpoint_metadata.json`. Use the
fixed version above to keep the paper, its implementation, and the numerical
sources together. Run metadata use portable repository-relative paths;
experimental settings and numerical-version identifiers are retained for
reproducibility. Checksums identify the distributed files.

## Environment and checks

~~~bash
conda env create -f environment.yml -n rvpinn
conda activate rvpinn
python -m pytest tests -q
python experiments/verify_round2_results.py
~~~

The implementation includes regression tests for bit-preserving
safety projection, current-observation accounting, strict threshold billing,
and physically feasible MIQP incumbents. Solver integration tests run in a
separate process to avoid loading conflicting Torch/SCIP OpenMP runtimes on
Windows. SCIP 10.0 with PySCIPOpt 6.2.1 is used for the mixed-integer models.

Result aggregators reject legacy or mixed `deployment_version`
records. The additional release gate is:

~~~bash
python experiments/verify_alignment_results.py
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

To compile the manuscript and supplement without running numerical experiments:

~~~bash
python experiments/compile_manuscripts.py
~~~

This uses `pdflatex` and `bibtex` with the included official `elsarticle` class
and numeric bibliography style. The format check validates document limits,
references, and layout; author declarations and journal submission are separate
steps. The reference, scenario, and seasonal table exporters reproduce the
distributed numerical content, explanatory captions, and interpretation.
Regression tests check these renderers against the manuscript and preserve
the distinction between conditional solver gaps and matched solver progress.

PowerShell orchestration scripts use python by default. Set RVPINN_PYTHON to
an alternative interpreter path when required.

## Additional analyses

The reference-controller and scenario-resolution analyses are documented in
the supplement and reproducibility notes. To rerun them if needed:

~~~powershell
python experiments\run_round2_reference.py
powershell -ExecutionPolicy Bypass -File experiments\run_round2_scenario_sensitivity.ps1
powershell -ExecutionPolicy Bypass -File experiments\run_round2_mechanism.ps1
python experiments\analyze_round2_reference.py
python experiments\analyze_round2_scenarios.py
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
