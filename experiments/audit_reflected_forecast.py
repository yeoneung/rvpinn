"""Measure the error of the unreflected OU conditional-mean forecast."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src.dynamics import reflect  # noqa: E402
from src.evaluation import load_region_days  # noqa: E402
from src.exp_common import regime_bundle  # noqa: E402

HORIZONS = (1.0, 4.0, 12.0, 24.0)


def ou_reflected_mean(x0, kappa, sigma, lo, hi, horizon, dt,
                      n_paths, seed):
    rng = np.random.default_rng(seed)
    x = np.full(n_paths, float(x0))
    phi = np.exp(-kappa * dt)
    sd = (sigma * np.sqrt(dt) if kappa <= 1e-12 else
          sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt))
                          / (2.0 * kappa)))
    for _ in range(int(round(horizon / dt))):
        x = reflect(phi * x + sd * rng.standard_normal(n_paths), lo, hi)
    return float(x.mean())


def main() -> int:
    region = "confirmatory"
    params, profiles, _ = regime_bundle(region)
    days = load_region_days(region, "val")
    rows = []
    for regime in sorted(params):
        p, prof = params[regime], profiles[regime]
        observations = []
        for day in days:
            if day["regime"] != regime:
                continue
            for hour in range(24):
                observations.append((
                    float(day["N"][hour] - prof.n_bar(hour)),
                    float(day["C"][hour] - prof.c_bar(hour))))
        if not observations:
            continue
        obs = np.asarray(observations)
        order = np.argsort(np.maximum(
            np.abs(obs[:, 0]) / max(abs(p.y_min), abs(p.y_max)),
            np.abs(obs[:, 1]) / max(abs(p.p_min), abs(p.p_max))))
        # Even coverage plus the most boundary-adjacent states.
        index = np.unique(np.concatenate([
            order[np.linspace(0, len(order) - 1, 21).astype(int)],
            order[-4:],
        ]))
        for h in HORIZONS:
            errors_y, errors_p = [], []
            for j, idx in enumerate(index):
                y0 = float(reflect(np.array([obs[idx, 0]]),
                                   p.y_min, p.y_max)[0])
                p0 = float(reflect(np.array([obs[idx, 1]]),
                                   p.p_min, p.p_max)[0])
                y_mc = ou_reflected_mean(
                    y0, p.kappa_y, p.sigma_y, p.y_min, p.y_max,
                    h, p.dt_ctrl, 2000, 71000 + j + int(10 * h))
                p_mc = ou_reflected_mean(
                    p0, p.kappa_p, p.sigma_p, p.p_min, p.p_max,
                    h, p.dt_ctrl, 2000, 81000 + j + int(10 * h))
                errors_y.append(abs(y_mc - y0 * np.exp(-p.kappa_y * h)))
                errors_p.append(abs(p_mc - p0 * np.exp(-p.kappa_p * h)))
            rows.append({
                "region": region, "regime": regime, "horizon_h": h,
                "n_initial_states": len(index), "n_paths": 2000,
                "load_error_mean_abs_kw": float(np.mean(errors_y)),
                "load_error_max_abs_kw": float(np.max(errors_y)),
                "price_error_mean_abs_eur_per_kwh": float(np.mean(errors_p)),
                "price_error_max_abs_eur_per_kwh": float(np.max(errors_p)),
                "load_error_mean_fraction_of_box": float(
                    np.mean(errors_y) / (p.y_max - p.y_min)),
                "price_error_mean_fraction_of_box": float(
                    np.mean(errors_p) / (p.p_max - p.p_min)),
            })
    path = HERE / "results" / "reflected_forecast_audit.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(path)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
