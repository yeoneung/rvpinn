import numpy as np

from src.discrete_bound import audit_bound, solve_finite_mdp


def test_deployment_aligned_bound_dominates_true_policy_loss():
    rng = np.random.default_rng(90210)
    K, S, A = 8, 7, 4
    costs = [rng.uniform(-0.2, 2.0, (S, A)) for _ in range(K)]
    transitions = []
    for _ in range(K):
        raw = rng.uniform(size=(S, A, S))
        transitions.append(raw / raw.sum(axis=2, keepdims=True))
    terminal = rng.uniform(0.0, 3.0, S)
    approx = [rng.normal(size=S) for _ in range(K + 1)]
    policy = [rng.integers(0, A, size=S) for _ in range(K)]

    optimal = solve_finite_mdp(costs, transitions, terminal)
    deployed = solve_finite_mdp(costs, transitions, terminal, policy)
    audit = audit_bound(costs, transitions, approx, policy, terminal)
    gap = deployed - optimal
    for k in range(K + 1):
        assert np.all(gap[k] >= -1e-12)
        assert np.all(gap[k] <= audit.bound_from_time[k] + 1e-12)


def test_exact_value_and_greedy_policy_give_zero_bound():
    rng = np.random.default_rng(11)
    K, S, A = 5, 4, 3
    costs = [rng.uniform(size=(S, A)) for _ in range(K)]
    transitions = []
    for _ in range(K):
        raw = rng.uniform(size=(S, A, S))
        transitions.append(raw / raw.sum(axis=2, keepdims=True))
    terminal = rng.uniform(size=S)
    value = solve_finite_mdp(costs, transitions, terminal)
    policy = []
    for k in range(K):
        q = costs[k] + np.einsum("xay,y->xa", transitions[k], value[k + 1])
        policy.append(q.argmin(axis=1))
    audit = audit_bound(costs, transitions, list(value), policy, terminal)
    assert np.max(np.abs(audit.bound_from_time)) < 1e-12


def test_time_only_value_offsets_do_not_create_policy_loss():
    # Residual oscillation should ignore state-independent temporal offsets.
    # This checks the manuscript's sharper span bound, not training convergence.
    rng = np.random.default_rng(709)
    K, S, A = 5, 4, 3
    costs = [rng.normal(size=(S, A)) for _ in range(K)]
    transitions = []
    for _ in range(K):
        raw = rng.uniform(size=(S, A, S))
        transitions.append(raw / raw.sum(axis=2, keepdims=True))
    terminal = rng.normal(size=S)
    value = solve_finite_mdp(costs, transitions, terminal)
    offsets = np.array([4., -3., 2., -5., 1., 0.])
    approximate = value + offsets[:, None]
    policy = [(costs[k] + np.einsum('xay,y->xa', transitions[k],
                                   approximate[k + 1])).argmin(axis=1)
              for k in range(K)]
    audit = audit_bound(costs, transitions, list(approximate), policy, terminal)
    deployed = solve_finite_mdp(costs, transitions, terminal, policy)
    assert np.allclose(deployed, value, atol=1e-12, rtol=0)
    assert np.max(np.abs(audit.bound_from_time)) < 1e-12
