"""Independent one-interval optimization audit; not a new test-set endpoint.

Enumerates the analytic stationary point on each continuous quadratic piece
and fee-side endpoints. Does not call production cost, SoC, or safety helpers.
Run separately from torch-based tests on Windows to isolate OpenMP runtimes.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def physical_bounds(p, soc):
    lower = -min(p.a_c, p.a_c * (p.s_max - soc) / p.delta_s,
                 p.E_max * (p.s_max - soc) / (p.eta_c * p.dt_ctrl))
    upper = min(p.a_d, p.a_d * (soc - p.s_min) / p.delta_s,
                p.eta_d * p.E_max * (soc - p.s_min) / p.dt_ctrl)
    return lower, upper


def hard_terminal_objective(p, soc, net, price, action):
    grid = net - action
    following = soc + p.dt_ctrl / p.E_max * (
        p.eta_c * max(-action, 0.) - max(action, 0.) / p.eta_d)
    return (p.dt_ctrl * (price * max(grid, 0.)
                        - p.alpha_s * price * max(-grid, 0.)
                        + p.lam1 * abs(action) + p.c_step * (grid > p.g_thr))
            + p.lam_T * (following - p.s_tar) ** 2)


def analytic_terminal_minimum(p, soc, net, price):
    """Finite enumeration of all piecewise-quadratic minimum candidates."""
    lower, upper = physical_bounds(p, soc)
    cuts = sorted(set([lower, upper] + [
        value for value in (0., net, net - p.g_thr)
        if lower <= value <= upper]))
    candidates = set(cuts)
    for left, right in zip(cuts[:-1], cuts[1:]):
        if left == right:
            continue
        midpoint = left + (right - left) / 2.
        soc_slope = p.dt_ctrl / p.E_max * (
            1. / p.eta_d if midpoint >= 0. else p.eta_c)
        price_factor = price if net - midpoint >= 0. else p.alpha_s * price
        throughput_slope = p.lam1 if midpoint >= 0. else -p.lam1
        linear = p.dt_ctrl * (-price_factor + throughput_slope)
        quadratic = p.lam_T * soc_slope ** 2
        if quadratic > 0.:
            root = (2. * p.lam_T * soc_slope * (soc - p.s_tar)
                    - linear) / (2. * quadratic)
            candidates.add(float(np.clip(root, left, right)))
    # Include fee sides through both action and grid-exchange arithmetic.
    for grid_side in (-np.inf, np.inf):
        candidates.add(float(np.clip(net - np.nextafter(p.g_thr, grid_side),
                                     lower, upper)))
    for point in list(candidates):
        for direction in (-np.inf, np.inf):
            candidates.add(float(np.clip(np.nextafter(point, direction), lower, upper)))
    return min((hard_terminal_objective(p, soc, net, price, action), action)
               for action in candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mip-gap', type=float, default=1e-8)
    parser.add_argument('--time-limit', type=float, default=2.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('terminal audit output already exists')
    from exact_miqp import ExactMIPController
    from stochastic_miqp import ScenarioExactMIPController
    from src.config import ModelParams

    p = ModelParams(g_thr=300., c_step=40., lam_T=10819.79)
    scenarios = [False, True]
    states = [p.s_min, p.s_min + 1e-6, .3, .5, .7, p.s_max - 1e-6, p.s_max]
    nets = [-50., 0., np.nextafter(300., -np.inf), 300.,
            np.nextafter(300., np.inf), 345.81784079454, 850.]
    prices = [-.2, 0., .2]
    rows, errors = [], []
    for scenario, soc, net, price in product(scenarios, states, nets, prices):
        kwargs = dict(reopt_every_hours=p.dt_ctrl, time_limit_s=args.time_limit,
                      mip_gap=args.mip_gap, random_seed=0)
        controller = (ScenarioExactMIPController(p, n_scenarios=2, **kwargs)
                      if scenario else ExactMIPController(p, **kwargs))
        ctx = {'N_now': net, 'C_now': price, 'date': '2019-01-07',
               'N_forecast': lambda t: np.asarray(t) * 0. + 300.,
               'C_forecast': lambda t: np.asarray(t) * 0. + .1}
        # Inconsistent residuals check that the actual observation is authoritative.
        try:
            action = controller(p.T - p.dt_ctrl, soc, -100., .02, ctx)
        except Exception as exc:
            errors.append(f'case {len(rows)}: {type(exc).__name__}: {exc}')
            rows.append(dict(scenario=scenario, soc=soc, net=net, price=price,
                             exception=f'{type(exc).__name__}: {exc}'))
            continue
        record = controller.solve_records[-1]
        minimum, analytic_action = analytic_terminal_minimum(p, soc, net, price)
        realized = hard_terminal_objective(p, soc, net, price, action)
        lower, upper = physical_bounds(p, soc)
        row = dict(scenario=scenario, soc=soc, net=net, price=price,
                   action=float(action), analytic_action=analytic_action,
                   realized_cost=float(realized), analytic_cost=float(minimum),
                   excess_cost=float(realized - minimum), record=asdict(record))
        index = len(rows)
        if not record.has_solution:
            errors.append(f'case {index}: missing incumbent')
        if not lower <= action <= upper:
            errors.append(f'case {index}: infeasible held action')
        if abs(realized - record.primal_bound) > 1e-4:
            errors.append(f'case {index}: hard cost/primal-bound discrepancy')
        if realized < minimum - 1e-4:
            errors.append(f'case {index}: cost below the independent minimum')
        if record.dual_bound > minimum + 1e-4:
            errors.append(f'case {index}: lower bound above the independent minimum')
        if (record.status == 'optimal'
                and abs(realized - minimum) > 1e-4):
            errors.append(f'case {index}: claimed optimum disagrees with analytic minimum')
        if realized - minimum > max(0., record.primal_bound - record.dual_bound) + 2e-4:
            errors.append(f'case {index}: analytic suboptimality exceeds recorded absolute gap')
        rows.append(row)
    report = {'status': 'failed' if errors else 'terminal_optimizer_checks_passed',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'one-interval synthetic diagnostics, not full-horizon optimality',
              'solver_time_limit_s': args.time_limit, 'solver_gap_target': args.mip_gap,
              'cases': len(rows), 'parameters': asdict(p),
              'near_exact_cases_at_1e_minus_4_eur': sum(
                  abs(row['excess_cost']) <= 1e-4 for row in rows if 'excess_cost' in row),
              'interpretation': ('A gap-limited incumbent need not equal the analytic minimum. '
                                 'Checks cover feasibility, physical cost, lower bounds, and '
                                 'claimed optima; observed optimization defects are retained.'),
              'max_absolute_cost_error': max(
                  (abs(row['excess_cost']) for row in rows if 'excess_cost' in row),
                  default=None),
              'errors': errors, 'rows': rows,
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'parameters')}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
