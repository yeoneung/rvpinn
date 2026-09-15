# Fixed-model value-function reuse

This exploratory benchmark reuses one selected central-winter checkpoint on
four model-generated disturbance paths and three initial states of charge.
The 12 combinations are not 12 independent disturbance paths. Tariff,
equipment, profiles, dynamics, and terminal cost remain fixed.

The stored `results/` contain complete action trajectories, solve records,
timings, input paths, and numerical diagnostics. Their provenance is recorded
in `protocol.json` and `results/manifest.json`. No retraining is needed to
inspect or regenerate the manuscript results. From the repository root:

```bash
python experiments/analyze_fixed_value_reuse.py
python experiments/fixed_value_reuse/audit_results.py
python -m pytest tests/test_fixed_value_reuse_reporting.py -q
```

The manuscript reports CPU measurements: one learned-policy process versus
up to four single-threaded MIQP workers, with setup and historical training
accounted for separately. The batching implementation produces the same
policy, but was slightly slower on CPU. Timing repetitions are not additional
independent quality samples. Operating-cost differences remain as reuse grows.

## Repeating the benchmark

Keep the distributed measurement records. Set `RVPINN_REUSE_RESULTS` to a new
explicit output directory for a new benchmark. `RVPINN_REUSE_SOURCE` can point
to a separate copy of this code. The default source is this repository.
The driver verifies the checkpoint checksum specified in the protocol.
The protocol and manifest record the settings, input paths, checkpoint,
and timing environment for the distributed measurements.

Run the two timed phases separately, with OMP/MKL/OPENBLAS thread counts set
to one, using the documented research environment:

```bash
python experiments/fixed_value_reuse/run_reuse.py neural
python experiments/fixed_value_reuse/run_reuse.py miqp
```

The driver can choose CUDA when its recorded utilization rule permits;
CPU and CUDA measurements must be reported separately. An interrupted MIQP
timing group preserves its partial output and refuses to silently resume an
uninterrupted wall-time claim. Torch and SCIP use separate processes on
Windows to avoid conflicting OpenMP runtimes.
