"""Pure NumPy worker and shared accounting; Torch is imported only on its path."""
import time
import numpy as np
from common import bundle, protocol, context, update_state, finish_rows


def simulate_neural(cases, model, mode):
    from src.deployed_policy import deployed_one_step_action
    from batched_policy import batched_actions
    import torch
    p, prof = bundle()
    soc = np.array([c['initial_soc'] for c in cases], dtype=float)
    arrays = {k: np.array([c[k] for c in cases]) for k in ('y', 'pz', 'net', 'price')}
    comp, actions, violations = np.zeros((len(cases), 3)), np.zeros((len(cases), 96)), np.zeros(len(cases))
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    for k in range(96):
        t = k * p.dt_ctrl
        y, pz, net, price = [arrays[name][:, k] for name in ('y', 'pz', 'net', 'price')]
        if mode == 'scalar':
            a = np.array([deployed_one_step_action(model, t, s, yy, pp, prof,
                                                  current_net=n, current_price=c)[0]
                          for s, yy, pp, n, c in zip(soc, y, pz, net, price)])
        elif mode == 'reference':
            from src.reference_policy import reference_one_step_action
            a = np.array([reference_one_step_action(p, prof, t, s, yy, pp,
                                                   current_net=n, current_price=c)[0]
                          for s, yy, pp, n, c in zip(soc, y, pz, net, price)])
        else:
            a, _ = batched_actions(model, t, soc, y, pz, net, price)
        soc, a, costs, violation = update_state(soc, a, net, price, p)
        comp += costs
        actions[:, k] = a
        violations = np.maximum(violations, violation)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return dict(simulation_wall_s=elapsed, rows=finish_rows(cases, soc, comp, actions, violations, p))


def warm_scip_worker():
    from pyscipopt import Model
    model = Model(); model.hideOutput()
    x = model.addVar(); model.setObjective(x); model.optimize()
    return True


def simulate_miqp(case):
    from stochastic_miqp import ScenarioExactMIPController
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    import os
    cfg = protocol()
    p, prof = bundle()
    started = time.perf_counter()
    controller = ScenarioExactMIPController(p, n_scenarios=2, random_seed=cfg['miqp_seed'],
                                          time_limit_s=cfg['miqp_solver_limit_s'], mip_gap=cfg['miqp_relative_gap'],
                                          reopt_every_hours=p.dt_ctrl)
    soc, comp, actions, violations = np.array([case['initial_soc']]), np.zeros((1, 3)), np.zeros((1, 96)), np.zeros(1)
    call_times = []
    with threadpool_limits(limits=1):
        for k in range(96):
            t = k * p.dt_ctrl
            net, price = case['net'][k], case['price'][k]
            tick = time.perf_counter()
            a = controller(t, soc[0], case['y'][k], case['pz'][k], context(case['path_id'], net, price, prof))
            call_times.append(time.perf_counter() - tick)
            soc, held, costs, violation = update_state(soc, np.array([a]), net, price, p)
            actions[:, k] = held
            violations = np.maximum(violations, violation)
            comp += costs
    row = finish_rows([case], soc, comp, actions, violations, p)[0]
    return dict(simulation_wall_s=time.perf_counter() - started, row=row, worker_pid=os.getpid(),
                call_s=call_times, solves=[asdict(r) for r in controller.solve_records])
