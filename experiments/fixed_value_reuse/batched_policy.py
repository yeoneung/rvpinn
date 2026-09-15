"""Batch only neural evaluation; preserve the published candidate sets and GH rule."""
import numpy as np
import torch
from common import protocol
from src.deployed_policy import deployed_action_candidates, _ou_joint_nodes
from src.costs import common_running_cost_exact
from src.dynamics import reflect, soc_step


def batched_actions(model, t, soc, y, pz, net, price, max_rows=None):
    p = model.p
    soc, y, pz, net, price = [np.atleast_1d(np.asarray(x, dtype=np.float64))
                             for x in (soc, y, pz, net, price)]
    if len(soc) > protocol()['max_batch_states']:
        chunks = []
        for start in range(0, len(soc), protocol()['max_batch_states']):
            sl = slice(start, start + protocol()['max_batch_states'])
            chunks.append(batched_actions(model, t, soc[sl], y[sl], pz[sl], net[sl], price[sl], max_rows))
        return np.concatenate([r[0] for r in chunks]), np.concatenate([r[1] for r in chunks])
    candidates = [deployed_action_candidates(p, s, n, p.dt_ctrl, 1025) for s, n in zip(soc, net)]
    sizes = np.array([len(a) for a in candidates])
    starts = np.r_[0, sizes.cumsum()]
    all_a = np.concatenate(candidates)
    repeats = lambda x: np.repeat(x, sizes)
    sn = soc_step(repeats(soc), all_a, p.dt_ctrl, p)
    stage = p.dt_ctrl * common_running_cost_exact(repeats(price), repeats(net) - all_a, all_a, p)
    tn = min(p.T, t + p.dt_ctrl)
    if tn >= p.T - 1e-12:
        future = p.lam_T * (sn - p.s_tar) ** 2
    else:
        z1, z2, weights, sd_y, sd_p = _ou_joint_nodes(p, p.dt_ctrl, 3)
        yq = reflect(np.exp(-p.kappa_y * p.dt_ctrl) * y[:, None] + sd_y * z1, p.y_min, p.y_max)
        pq = reflect(np.exp(-p.kappa_p * p.dt_ctrl) * pz[:, None] + sd_p * z2, p.p_min, p.p_max)
        nq = len(weights)
        ys = np.repeat(yq, sizes, axis=0).ravel()
        ps = np.repeat(pq, sizes, axis=0).ravel()
        ss = np.repeat(sn, nq)
        device = next(model.parameters()).device
        max_rows = max_rows or protocol()['max_model_rows']
        values = []
        with torch.no_grad():
            for offset in range(0, len(ss), max_rows):
                sl = slice(offset, offset + max_rows)
                tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
                values.append(model(torch.full((len(ss[sl]),), tn, dtype=torch.float64, device=device),
                                    tensor(ss[sl]), tensor(ys[sl]), tensor(ps[sl])).detach().cpu().numpy())
        v = np.concatenate(values).reshape(-1, nq)
        # Match scalar matrix-vector reductions separately for every state.
        future = np.concatenate([v[starts[i]:starts[i+1]] @ weights for i in range(len(soc))])
    q = stage + future
    idx = np.array([starts[i] + np.argmin(q[starts[i]:starts[i+1]]) for i in range(len(soc))])
    return all_a[idx], q[idx]
