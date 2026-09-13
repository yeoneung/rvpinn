"""Finite-difference audit of physical-coordinate derivatives after selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
GENERATED = ROOT / "manuscript" / "generated"
sys.path.insert(0, str(ROOT))

from src.exp_common import regime_bundle  # noqa: E402
from src.pinn_value import value_and_derivatives  # noqa: E402
from src.policy_iteration import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v3_winter_fee40")
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4117)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_path = RESULTS / f"restart_batches_{args.tag}.json"
    batches = json.loads(batch_path.read_text(encoding="utf-8"))
    selected = [(int(x["batch"]), int(x["selected_seed"]))
                for x in batches["batch_selections"]
                if x.get("selected_seed") is not None]
    skipped_fallback_batches = [int(x["batch"])
                                for x in batches["batch_selections"]
                                if x.get("selected_seed") is None]
    if not selected:
        raise RuntimeError("no trained central-fee policy selected")
    params, profiles, _ = regime_bundle(
        "confirmatory", ["winter_weekday"])
    p, prof = params["winter_weekday"], profiles["winter_weekday"]
    p.g_thr, p.c_step, p.w_step = 300.0, 40.0, 10.0
    rng = np.random.default_rng(args.seed)
    rows = []

    for batch, selected_seed in selected:
        run_dir = (ROOT / "checkpoints" / args.tag / "confirmatory"
                   / "winter_weekday" / f"seed{selected_seed}")
        result = json.loads((run_dir / "result.json").read_text(
            encoding="utf-8"))
        model = load_checkpoint(result["selected_checkpoint"], p, args.device)
        device = next(model.parameters()).device
        n = args.points
        t = torch.tensor(rng.uniform(0.2, p.T - 0.2, n),
                         dtype=torch.float64, device=device)
        s = torch.tensor(rng.uniform(p.s_min + 0.02, p.s_max - 0.02, n),
                         dtype=torch.float64, device=device)
        y = torch.tensor(rng.uniform(0.8 * p.y_min, 0.8 * p.y_max, n),
                         dtype=torch.float64, device=device)
        pz = torch.tensor(rng.uniform(0.8 * p.p_min, 0.8 * p.p_max, n),
                          dtype=torch.float64, device=device)
        ad = value_and_derivatives(model, t, s, y, pz, second_order=False)

        def value(tt, ss, yy, pp):
            with torch.no_grad():
                return model(tt, ss, yy, pp)

        steps = {"t": 1e-5 * p.T, "s": 2e-6,
                 "y": 1e-5 * (p.y_max - p.y_min),
                 "p": 1e-5 * (p.p_max - p.p_min)}
        fd = {
            "v_t": (value(t + steps["t"], s, y, pz)
                    - value(t - steps["t"], s, y, pz)) / (2 * steps["t"]),
            "v_s": (value(t, s + steps["s"], y, pz)
                    - value(t, s - steps["s"], y, pz)) / (2 * steps["s"]),
            "v_y": (value(t, s, y + steps["y"], pz)
                    - value(t, s, y - steps["y"], pz)) / (2 * steps["y"]),
            "v_p": (value(t, s, y, pz + steps["p"])
                    - value(t, s, y, pz - steps["p"])) / (2 * steps["p"]),
        }
        row = {"batch": batch, "selected_seed": selected_seed,
               "n_points": n}
        for name, finite_difference in fd.items():
            error = torch.max(torch.abs(
                ad[name].detach() - finite_difference)).item()
            scale = torch.max(torch.abs(finite_difference)).item() + 1e-8
            row[f"{name}_max_abs_error"] = float(error)
            row[f"{name}_relative_error"] = float(error / scale)
        with torch.no_grad():
            terminal = model(torch.full_like(t, p.T), s, y, pz)
        phi = p.lam_T * (s - p.s_tar) ** 2
        row["terminal_max_abs_error"] = float(
            torch.max(torch.abs(terminal - phi)).item())
        rows.append(row)

    output = {
        "label": "post-selection derivative implementation audit",
        "seed": args.seed,
        "points_per_batch": args.points,
        "skipped_zero_action_fallback_batches": skipped_fallback_batches,
        "batches": rows,
    }
    out_path = RESULTS / "selected_derivative_audit.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    derivative_max = max(
        row[f"{name}_relative_error"] for row in rows
        for name in ("v_t", "v_s", "v_y", "v_p"))
    terminal_max = max(row["terminal_max_abs_error"] for row in rows)
    def sci_tex(value: float) -> str:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"

    tex = (
        f"Across the selected trained central-fee batch policies, the largest "
        f"physical-coordinate first-derivative discrepancy relative to "
        f"central finite differences was {sci_tex(derivative_max)}; the "
        f"largest terminal-condition error was {sci_tex(terminal_max)} EUR."
    )
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / "derivative_audit.tex").write_text(tex, encoding="utf-8")
    print(out_path)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
