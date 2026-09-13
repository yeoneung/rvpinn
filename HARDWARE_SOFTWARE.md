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
