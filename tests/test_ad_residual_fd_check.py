"""Autograd PDE derivatives agree with central finite differences on the
network output at random points (protocol Stage 0 check #7)."""
import numpy as np
import torch

from src.pinn_value import ValueNet, value_and_derivatives

torch.set_default_dtype(torch.float64)


def test_autograd_matches_finite_differences(params):
    model = ValueNet(params, width=16, depth=2, use_p=True, out_scale=10.0,
                     seed=3)
    rng = np.random.default_rng(8)
    n = 200
    t = torch.tensor(rng.uniform(0.1, params.T - 0.1, n))
    s = torch.tensor(rng.uniform(params.s_min + 0.01, params.s_max - 0.01, n))
    y = torch.tensor(rng.uniform(params.y_min * 0.9, params.y_max * 0.9, n))
    pz = torch.tensor(rng.uniform(params.p_min * 0.9, params.p_max * 0.9, n))

    d = value_and_derivatives(model, t, s, y, pz)

    def f(tt, ss, yy, pp):
        with torch.no_grad():
            return model(tt, ss, yy, pp)

    h_t, h_s = 1e-5 * params.T, 1e-6
    h_y = 1e-4 * (params.y_max - params.y_min)
    h_p = 1e-4 * (params.p_max - params.p_min)

    fd_t = (f(t + h_t, s, y, pz) - f(t - h_t, s, y, pz)) / (2 * h_t)
    fd_s = (f(t, s + h_s, y, pz) - f(t, s - h_s, y, pz)) / (2 * h_s)
    fd_y = (f(t, s, y + h_y, pz) - f(t, s, y - h_y, pz)) / (2 * h_y)
    fd_p = (f(t, s, y, pz + h_p) - f(t, s, y, pz - h_p)) / (2 * h_p)
    fd_yy = (f(t, s, y + h_y, pz) - 2 * f(t, s, y, pz)
             + f(t, s, y - h_y, pz)) / h_y ** 2
    fd_pp = (f(t, s, y, pz + h_p) - 2 * f(t, s, y, pz)
             + f(t, s, y, pz - h_p)) / h_p ** 2
    fd_yp = (f(t, s, y + h_y, pz + h_p) - f(t, s, y + h_y, pz - h_p)
             - f(t, s, y - h_y, pz + h_p) + f(t, s, y - h_y, pz - h_p)) \
        / (4 * h_y * h_p)

    def check(ad, fd, name, rtol=1e-4, atol=1e-6):
        ad = ad.detach()
        err = (ad - fd).abs().max().item()
        scale = fd.abs().max().item() + atol
        assert err / scale < rtol, f"{name}: rel err {err/scale:.2e}"

    check(d["v_t"], fd_t, "v_t")
    check(d["v_s"], fd_s, "v_s")
    check(d["v_y"], fd_y, "v_y")
    check(d["v_p"], fd_p, "v_p")
    check(d["v_yy"], fd_yy, "v_yy", rtol=1e-3)
    check(d["v_pp"], fd_pp, "v_pp", rtol=1e-3)
    check(d["v_yp"], fd_yp, "v_yp", rtol=1e-3)


def test_hard_terminal_exact(params):
    model = ValueNet(params, width=16, depth=2, use_p=True, out_scale=10.0,
                     seed=5)
    n = 500
    rng = np.random.default_rng(2)
    s = torch.tensor(rng.uniform(params.s_min, params.s_max, n))
    y = torch.tensor(rng.uniform(params.y_min, params.y_max, n))
    pz = torch.tensor(rng.uniform(params.p_min, params.p_max, n))
    tT = torch.full((n,), params.T, dtype=torch.float64)
    with torch.no_grad():
        v = model(tT, s, y, pz)
    phi = params.lam_T * (s - params.s_tar) ** 2
    assert (v - phi).abs().max().item() < 1e-10


def test_reference_policy_ansatz_chain_rule(params, profile):
    model = ValueNet(
        params, width=12, depth=2, use_p=True, out_scale=10.0, seed=17,
        reference_baseline="zero_action_conditional_mean",
        reference_quadrature_points=12).attach_profile(profile)
    rng = np.random.default_rng(17)
    n = 40
    t = torch.tensor(rng.uniform(0.2, params.T - 0.2, n))
    s = torch.tensor(rng.uniform(params.s_min + 0.02,
                                 params.s_max - 0.02, n))
    y = torch.tensor(rng.uniform(0.8 * params.y_min,
                                 0.8 * params.y_max, n))
    pz = torch.tensor(rng.uniform(0.8 * params.p_min,
                                  0.8 * params.p_max, n))
    d = value_and_derivatives(model, t, s, y, pz, second_order=False)

    def value(tt, ss, yy, pp):
        with torch.no_grad():
            return model(tt, ss, yy, pp)

    steps = {"t": 1e-5 * params.T, "s": 2e-6,
             "y": 1e-5 * (params.y_max - params.y_min),
             "p": 1e-5 * (params.p_max - params.p_min)}
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
    for name in fd:
        err = torch.max(torch.abs(d[name].detach() - fd[name])).item()
        scale = torch.max(torch.abs(fd[name])).item() + 1e-8
        assert err / scale < 3e-4, (name, err / scale)

    t_terminal = torch.full((n,), params.T)
    with torch.no_grad():
        terminal = model(t_terminal, s, y, pz)
    phi = params.lam_T * (s - params.s_tar) ** 2
    assert torch.max(torch.abs(terminal - phi)).item() < 1e-10
