# Hardware and software record

The reported experiments used a Windows 10 workstation with:

- an AMD Ryzen 7 9700X with 8 cores and 16 logical processors;
- one NVIDIA GeForce RTX 4080 SUPER with 16,376 MiB of memory;
- NVIDIA driver 560.94;
- Python 3.9.19;
- PyTorch 2.4.1 with CUDA 12.4;
- NumPy 1.26.4, pandas 2.2.1, and SciPy 1.13.1; and
- PySCIPOpt 6.2.1 with SCIP 10.0.

Learned-value training, validation, and held-out evaluation used the GPU.
Neural call latency includes transferring one-step values back to the CPU;
it is not a CPU-only neural benchmark. Each MIQP solve and BLAS backend uses
one CPU thread. Device-specific compute prices can differ; equal-second
valuation is an illustrative assumption, not a measured computing bill.

Batch execution ranged from two day workers per scenario stream (at most six
concurrent solves) to a shared pool of at most eight day workers. Scenario
and seasonal jobs could overlap with GPU evaluation and diagnostics. Timings
are workstation controller-call measurements under these schedules, not
isolated kernels or measurements under identical background load. The
single-thread limit, per-solve budgets, solver tolerances, and random seeds
are separate from the batch scheduling configuration.

Controller-call latency includes forecasting, quadrature or solver work,
action search, and controller-specific feasibility logic. The common post-call
safety gate and simulation accounting occur outside that timer. Model loading
is measured separately.

Offline training time uses the stored start/end run manifests, including
trainer construction, fitting, post-fit diagnostics, certification, and result
writing. Timestamps have one-second resolution. Optimizer-only totals are a
diagnostic subset and are not added again. Each restart's manifest path and
SHA-256 accompany the compute table. Validation accounting sums controller-call
times for every candidate checkpoint; shared plant simulation and file I/O
are outside those timers. Imports and input preparation before the training
manifest are also outside this accounting.

The separate fixed-model reuse benchmark uses CPU-only end-to-end simulation
timings, not the historical GPU-assisted controller-call timings above. Learned
and reference modes use one process; MIQP uses one worker for one case and four
for four or twelve cases, each with one thread. Simulation timers include
controller calls, hard-cost accounting, and state updates. Model/library setup,
worker startup, path generation, and offline training are excluded and recorded
separately. Neural and reference modes have three repetitions; MIQP has one per
workload. CPU batching was slightly slower than scalar learned evaluation.
Details and raw timing records are in `experiments/fixed_value_reuse/`.

Controller comparisons use pure-NumPy tariff rules and Torch-free SCIP
worker processes. The heuristic grid uses four workers and the hinge sweep
two. Fixed-state MIQP runs use four workers. Closed-loop evaluations use
between two and eight concurrent workers, with one solver thread per worker.
The executor records worker allocation and completion status in
`experiments/controller_comparisons/results/solver_budget/`.

The central-winter rule's 97-candidate validation grid used 41.8 seconds
summed over individual candidate simulations on the 30 validation days.
This is not the four-worker elapsed wall time and excludes setup and input
preparation. The separate rule-reuse timing includes simulation and state/cost
accounting, as in the CPU reuse comparison. Timings were measured separately
on the shared workstation; background load was not controlled across runs.
These comparisons reuse stored neural outputs. CUDA is not an execution
backend for these SCIP models.

The extended-budget comparison contains 576 fixed-state solves and 36
closed-loop days. The three high-fee M=16 300-second days took 7.76, 7.56,
and 7.79 hours. Each solve uses one thread. Longer budgets
measure computation independently of simulated time, not an actuation-delay
model or an experimentally established real-time deadline.
