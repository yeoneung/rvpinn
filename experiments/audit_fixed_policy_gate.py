"""Audit the learned zero-action policy value before policy improvement."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.costs import running_cost, terminal_cost  # noqa: E402
from src.dynamics import ou_step_exact, reflect  # noqa: E402
from src.exp_common import load_stack, regime_bundle  # noqa: E402
from src.finite_difference import FDGrid, interp_V, solve_policy_eval  # noqa: E402
from src.pinn_value import value_and_derivatives  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402


def main() -> int:
    cfg = load_stack(str(ROOT / "configs" / "gate_pilot.yaml"))
    params, profiles, _ = regime_bundle(
        "primary", ["winter_weekday"], cfg)
    p, prof = params["winter_weekday"], profiles["winter_weekday"]
    grid = FDGrid(p, 31, 31, 37)

    def zero_policy(t, s, y):
        return np.zeros_like(s)

    value_fd = solve_policy_eval(grid, p, prof, zero_policy)
    checkpoint = (ROOT / "checkpoints" / "expA" / "rvpinnpi_noadapt"
                  / "seed2" / "iter00.pt")
    model = load_checkpoint(str(checkpoint), p, "cpu")

    n_audit = 5000
    design = torch.quasirandom.SobolEngine(3, scramble=True, seed=4199)
    u = design.draw(n_audit).double().numpy()
    t = u[:, 0] * p.T
    s = p.s_min + u[:, 1] * (p.s_max - p.s_min)
    y = p.y_min + u[:, 2] * (p.y_max - p.y_min)
    value_reference = interp_V(value_fd, grid, t, s, y)
    tensors = [torch.as_tensor(x, dtype=torch.float64) for x in (t, s, y)]
    zeros = torch.zeros(n_audit, dtype=torch.float64)
    with torch.no_grad():
        value_learned = model(tensors[0], tensors[1], tensors[2], zeros)
    value_learned = value_learned.numpy()

    hs = 0.25 * grid.hs
    s_low = np.maximum(s - hs, p.s_min)
    s_high = np.minimum(s + hs, p.s_max)
    grad_reference = (
        interp_V(value_fd, grid, t, s_high, y)
        - interp_V(value_fd, grid, t, s_low, y)
    ) / np.maximum(s_high - s_low, 1e-12)
    derivatives = value_and_derivatives(
        model, tensors[0], tensors[1], tensors[2], zeros,
        second_order=False)
    grad_learned = derivatives["v_s"].detach().numpy()

    initial_fd = float(interp_V(
        value_fd, grid, np.array([0.0]), np.array([p.s0]),
        np.array([0.0]))[0])
    with torch.no_grad():
        initial_learned = float(model(
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([p.s0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64)).item())

    # Independent zero-policy Monte Carlo under the same reflected diffusion.
    n_paths = 20000
    rng = np.random.default_rng(4201)
    state_y = np.zeros(n_paths, dtype=np.float64)
    state_s = np.full(n_paths, p.s0, dtype=np.float64)
    total = np.zeros(n_paths, dtype=np.float64)
    dt = p.dt_sim
    for k in range(int(round(p.T / dt))):
        time_h = k * dt
        total += dt * running_cost(
            time_h, state_s, state_y, np.zeros_like(state_y),
            np.zeros_like(state_y), p, prof)
        innovation = rng.standard_normal(n_paths)
        state_y = reflect(
            ou_step_exact(state_y, p.kappa_y, p.sigma_y, dt, innovation),
            p.y_min, p.y_max)
    total += terminal_cost(state_s, p)
    mc_mean = float(total.mean())
    mc_se = float(total.std(ddof=1) / np.sqrt(n_paths))

    value_error = value_learned - value_reference
    gradient_error = grad_learned - grad_reference
    output = {
        "label": "zero-action fixed-policy audit before policy improvement",
        "checkpoint": str(checkpoint),
        "finite_difference_grid": [grid.Ns, grid.Ny, grid.Nt],
        "n_sobol_audit_points": n_audit,
        "value_rmse": float(np.sqrt(np.mean(value_error ** 2))),
        "value_max_abs_error": float(np.max(np.abs(value_error))),
        "soc_gradient_rmse": float(np.sqrt(np.mean(gradient_error ** 2))),
        "soc_gradient_max_abs_error": float(np.max(np.abs(gradient_error))),
        "initial_value_learned": initial_learned,
        "initial_value_finite_difference": initial_fd,
        "initial_value_monte_carlo": mc_mean,
        "monte_carlo_standard_error": mc_se,
        "n_monte_carlo_paths": n_paths,
    }
    out = ROOT / "results" / "raw" / "expA" / "fixed_policy_audit.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    generated = ROOT / "manuscript" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    tex = (
        "Before policy improvement, the zero-action evaluation had value "
        f"RMSE {output['value_rmse']:.2f} EUR and SoC-gradient RMSE "
        f"{output['soc_gradient_rmse']:.2f} EUR per unit SoC against the "
        "independent finite-difference solution. At the initial state, the "
        f"learned, finite-difference, and {n_paths:,}-path Monte Carlo values "
        f"were {initial_learned:.2f}, {initial_fd:.2f}, and {mc_mean:.2f} "
        f"EUR (Monte Carlo standard error {mc_se:.2f}).")
    (generated / "fixed_policy_audit.tex").write_text(tex, encoding="utf-8")
    print(out)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
