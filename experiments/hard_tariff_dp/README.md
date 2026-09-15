# Hard-tariff reduced-model decision accuracy

This separate benchmark retains the 40 EUR/h strict import-band fee, 300 kW
threshold, 24-hour horizon, 15-minute held actions, and quadratic terminal
cost. Price error is fixed to zero. Profiles and physical/economic parameters
come from the existing confirmatory winter calibration; no historical test
results are overwritten.

`protocol.json` fixes the design before training and DP comparisons. Five
new two-state PINN-PI models use the existing 5-by-96 architecture and four
rounds of up to 2,000 Adam steps plus 80 L-BFGS iterations. The smooth PDE
trainer is unchanged. Its internal rollout diagnostic is not the selection
criterion in this experiment. Checkpoints, seeds, and 96 tariff-rule settings
are selected by expected hard cost at (SoC=0.5, load error=0) on the coarsest
DP grid, using the existing sequential 0.10 EUR improvement margin.

## Common numerical model

`core.py` independently implements the discrete cost and held-action map.
The reflected Gaussian OU transition integrates piecewise-linear load-grid
basis functions analytically. Continuation values use linear interpolation
in SoC. The terminal stage evaluates the original quadratic directly.
Action candidates include uniform grids, feasible endpoints, zero import,
zero action, verified tariff-crossing sides, and actions reaching next-SoC
grid knots. These knots capture all additional kinks of the interpolated
continuation objective. Increasing only the uniform action grid is therefore
redundant once these breakpoints are present, as tested in `test_core.py`.

All policy costs and regrets use this same grid transition, not different
Monte Carlo paths or historical observations. The learned continuation is
tabulated on each grid and searched with the common operator. This is a
controlled discretized evaluation of the learned value, not an assertion
that interpolation and production Gauss-Hermite deployment are identical.
`audit_deployment.py` compares it with direct neural searches at 24
boundary-focused and 64 uniformly sampled grid states, using both production
and higher-order quadrature.

Grid refinement measures numerical stability, not a uniform error certificate.
With Q_DP,k(x,a) = hard_stage_cost_k(x,a) + (P_hat_h,k^a V_DP,k+1)(x),
one-step regret is Q_DP,k(x,pi_k(x)) - V_DP,k(x), where V_DP,k minimizes
Q_DP,k over feasible actions. P_hat denotes the common numerical transition
and interpolation; the final step evaluates the terminal quadratic directly.
Thus regret uses the DP-optimal continuation, not the learned-value search
objective. Boundary summaries use grid states where the DP action leaves
import within 10 kW of the threshold. Units are EUR per decision, and the
grid-state average is not trajectory weighted.

## Run

The finest-grid expected daily costs from (0.5, 0) are 1348.12 EUR for DP,
1385.15 EUR for the selected learned controller, 1372.88 EUR for the tariff rule,
and 1627.70 EUR for the reference-only controller. The learned and rule gaps
to DP are 2.75% and 1.84%, respectively. Seed 0, round 4, is selected by the
coarse-grid criterion. The final DP refinement changes cost by 0.21 EUR/day.
These are fitted-model results, not historical replay costs.

From the repository root, with the existing experiment dependencies:

```powershell
python -m pytest experiments/hard_tariff_dp/test_core.py -q
python experiments/hard_tariff_dp/train.py
python experiments/hard_tariff_dp/run.py --stage reference
python experiments/hard_tariff_dp/run.py --stage evaluation
python experiments/hard_tariff_dp/audit_deployment.py
python experiments/hard_tariff_dp/report.py
```

Training uses CUDA when available; DP uses NumPy/SciPy on the CPU. Completed
training seeds and numerical outputs are reused. Inspect partial outputs
before restarting; do not mix results from changed protocols or algorithms.
These scripts do not run or replace the historical comparison experiments.
