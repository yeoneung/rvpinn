"""Hard-terminal value networks and smooth-training autograd residuals.

The default reference ansatz is
v_theta(t, x) = V_cm^0(t, x) + (T - t) * scale * N_theta(normalized inputs).
A terminal-only baseline Phi(s) is also supported for diagnostic comparisons.
The terminal condition holds exactly, so e_T = 0 up to float64 rounding.
All PDE work is float64. Supports the 3-state (S, Y, P) problem and the
2-state (S, Y) verification problem (use_p=False).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .config import ModelParams
from .costs import RegimeProfile, psi_eps, step_fee_smooth

DTYPE = torch.float64


def torch_running_cost(t, s, y, pz, a, p: ModelParams, prof: RegimeProfile):
    """ell in torch (a treated as a constant tensor)."""
    G = prof.n_bar(t) + y - a
    C = prof.c_bar(t) + pz
    imp = psi_eps(G, p.eps_g)
    exp_ = psi_eps(-G, p.eps_g)
    return (C * imp - p.alpha_s * C * exp_
            + p.lam_pk * imp * imp
            + p.lam1 * ((a * a + p.eps_a ** 2) ** 0.5 - p.eps_a)
            + p.lam2 * a * a
            + p.lam_s * (s - p.s_ref) ** 2
            + step_fee_smooth(G, p))


def torch_b_S(a, p: ModelParams):
    a_pos = torch.clamp(a, min=0.0)
    a_neg = torch.clamp(-a, min=0.0)
    return p.eta_c * a_neg / p.E_max - a_pos / (p.eta_d * p.E_max)


def value_rate_scale(p: ModelParams, prof: RegimeProfile) -> float:
    """Upper-bound-style estimate of max |L^a Phi + ell| (currency/hour).

    The hard-terminal ansatz v = Phi + (T-t) * scale * N requires N to reach
    (V - Phi)/((T-t) scale); near t = T this is (L^a Phi + ell)/scale, so
    `scale` must match that rate or the network cannot represent the value
    within a reasonable weight range (diagnosed in the pilot run).
    """
    max_bS = max(p.eta_c * p.a_c, p.a_d / p.eta_d) / p.E_max
    dphi = 2.0 * p.lam_T * max(p.s_max - p.s_tar, p.s_tar - p.s_min)
    c_max = float(np.max(np.abs(prof.c_hour))) + max(abs(p.p_min),
                                                     abs(p.p_max))
    g_max = float(np.max(np.abs(prof.n_hour))) \
        + max(abs(p.y_min), abs(p.y_max)) + p.a_max
    ell_max = (c_max * g_max + p.lam_pk * g_max ** 2
               + p.lam1 * p.a_max + p.lam2 * p.a_max ** 2
               + p.lam_s * (p.s_max - p.s_ref) ** 2
               + p.c_step)
    return float(max_bS * dphi + ell_max)


class ValueNet(torch.nn.Module):
    def __init__(self, p: ModelParams, width: int = 128, depth: int = 6,
                 fourier_k=(1, 2, 3), use_p: bool = True,
                 out_scale: float = 1.0, seed: int = 0,
                 hard_terminal: bool = True,
                 reference_baseline: str = "terminal_only",
                 reference_quadrature_points: int = 24):
        super().__init__()
        self.p = p
        self.use_p = use_p
        self.hard_terminal = hard_terminal
        self.reference_baseline = str(reference_baseline)
        self.reference_quadrature_points = int(reference_quadrature_points)
        self.fourier_k = tuple(fourier_k)
        self.out_scale = float(out_scale)     # ~ B0 / T, currency per hour
        in_dim = (4 if use_p else 3) + 2 * len(self.fourier_k)
        g = torch.Generator().manual_seed(seed)
        layers = []
        dims = [in_dim] + [width] * depth + [1]
        for i in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[i], dims[i + 1], dtype=DTYPE)
            # With a reference-policy ansatz, start from the reference itself
            # rather than adding an economically huge random correction.
            if i == len(dims) - 2 and reference_baseline != "terminal_only":
                torch.nn.init.zeros_(lin.weight)
            else:
                torch.nn.init.xavier_normal_(lin.weight, generator=g)
            torch.nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = torch.nn.ModuleList(layers)

    def _features(self, t, s, y, pz):
        p = self.p
        tn = 2.0 * t / p.T - 1.0
        sn = 2.0 * (s - p.s_min) / (p.s_max - p.s_min) - 1.0
        yn = 2.0 * (y - p.y_min) / (p.y_max - p.y_min) - 1.0
        feats = [tn, sn, yn]
        if self.use_p:
            pn = 2.0 * (pz - p.p_min) / (p.p_max - p.p_min) - 1.0
            feats.append(pn)
        for k in self.fourier_k:
            w = 2.0 * math.pi * k / p.T
            feats.append(torch.sin(w * t))
            feats.append(torch.cos(w * t))
        return torch.stack(feats, dim=-1)

    def net(self, t, s, y, pz):
        h = self._features(t, s, y, pz)
        for lin in self.layers[:-1]:
            h = torch.tanh(lin(h))
        return self.layers[-1](h).squeeze(-1)

    def forward(self, t, s, y, pz):
        p = self.p
        if not self.hard_terminal:      # ablation A2: terminal via penalty
            return self.out_scale * self.net(t, s, y, pz)
        reference = self._reference_value(t, s, y, pz)
        return (reference
                + (p.T - t) * self.out_scale * self.net(t, s, y, pz))

    def _reference_value(self, t, s, y, pz):
        """Differentiable no-storage conditional-mean cost-to-go baseline.

        This is a reference-policy control variate, not an assertion that a
        conditional-mean path equals the value under reflected stochastic
        dynamics.  The terminal value remains exact by construction.
        """
        p = self.p
        phi = p.lam_T * (s - p.s_tar) ** 2
        if self.reference_baseline == "terminal_only":
            return phi
        if self.reference_baseline != "zero_action_conditional_mean":
            raise ValueError(
                f"unknown reference baseline: {self.reference_baseline}")
        if not hasattr(self, "_profile"):
            raise RuntimeError("attach_profile must be called for this baseline")
        q = self.reference_quadrature_points
        if q < 1:
            raise ValueError("reference_quadrature_points must be positive")
        frac = ((torch.arange(q, dtype=t.dtype, device=t.device) + 0.5)
                / float(q))
        remaining = torch.clamp(p.T - t, min=0.0)
        tau = remaining[:, None] * frac[None, :]
        tq = t[:, None] + tau
        yq = y[:, None] * torch.exp(-p.kappa_y * tau)
        pq = pz[:, None] * torch.exp(-p.kappa_p * tau)
        sq = s[:, None].expand_as(tq)
        aq = torch.zeros_like(tq)
        ell = torch_running_cost(tq, sq, yq, pq, aq, p, self._profile)
        return phi + remaining * ell.mean(dim=1)

    def attach_profile(self, prof: RegimeProfile):
        """Attach the fixed regime profile used by the reference baseline."""
        self._profile = prof
        return self


def value_and_derivatives(model: ValueNet, t, s, y, pz,
                          second_order: bool = True):
    """v and all PDE derivatives via autograd. Inputs are 1-D tensors."""
    t = t.detach().requires_grad_(True)
    s = s.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    pz = pz.detach().requires_grad_(True)
    v = model(t, s, y, pz)
    ones = torch.ones_like(v)
    inputs = [t, s, y] + ([pz] if model.use_p else [])
    grads = torch.autograd.grad(v, inputs, grad_outputs=ones,
                                create_graph=True)
    v_t, v_s, v_y = grads[0], grads[1], grads[2]
    v_p = grads[3] if model.use_p else torch.zeros_like(v)
    out = {"v": v, "v_t": v_t, "v_s": v_s, "v_y": v_y, "v_p": v_p,
           "t": t, "s": s, "y": y, "pz": pz}
    if second_order:
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones,
                                   create_graph=True)[0]
        out["v_yy"] = v_yy
        if model.use_p:
            v_pp = torch.autograd.grad(v_p, pz, grad_outputs=ones,
                                       create_graph=True)[0]
            v_yp = torch.autograd.grad(v_y, pz, grad_outputs=ones,
                                       create_graph=True)[0]
            out["v_pp"] = v_pp
            out["v_yp"] = v_yp
        else:
            out["v_pp"] = torch.zeros_like(v)
            out["v_yp"] = torch.zeros_like(v)
    return out


def policy_residual(model: ValueNet, t, s, y, pz, a,
                    prof: RegimeProfile, d: Optional[dict] = None):
    """r_v^pi = v_t + L^a v + ell(t,x,a); `a` is a constant tensor."""
    p = model.p
    if d is None:
        d = value_and_derivatives(model, t, s, y, pz)
    gen = (torch_b_S(a, p) * d["v_s"]
           - p.kappa_y * d["y"] * d["v_y"]
           - (p.kappa_p * d["pz"] * d["v_p"] if model.use_p else 0.0)
           + 0.5 * p.sigma_y ** 2 * d["v_yy"]
           + (0.5 * p.sigma_p ** 2 * d["v_pp"] if model.use_p else 0.0)
           + (p.rho * p.sigma_y * p.sigma_p * d["v_yp"] if model.use_p else 0.0))
    ell = torch_running_cost(d["t"], d["s"], d["y"], d["pz"], a, p, prof)
    return d["v_t"] + gen + ell, d


def greedy_action_torch(model: ValueNet, t, s, y, pz, prof: RegimeProfile,
                        bounds="deploy", dt: Optional[float] = None):
    """Greedy action fully on the model's device (GPU-fast path).
    Returns a detached float64 torch tensor."""
    import torch
    from .hamiltonian import minimize_hamiltonian_torch
    from .safety import cont_bounds, deploy_bounds
    with torch.enable_grad():        # callers may be inside no_grad()
        d = value_and_derivatives(model, t, s, y, pz, second_order=False)
    v_s = d["v_s"].detach()
    s_d = s.detach()
    p = model.p
    if bounds == "deploy":
        lo, hi = deploy_bounds(s_d, p, dt if dt is not None else p.dt_ctrl)
    else:
        lo, hi = cont_bounds(s_d, p)
    with torch.no_grad():
        a, _ = minimize_hamiltonian_torch(
            t.detach(), s_d, y.detach(), pz.detach(), v_s, p, prof, lo, hi)
    return a


def greedy_action_np(model: ValueNet, t, s, y, pz, prof: RegimeProfile,
                     bounds="deploy", dt: Optional[float] = None,
                     objective: str = "smooth"):
    """Greedy (Hamiltonian-minimizing) action at points, via the numpy
    solver on detached v_s. Returns a float64 numpy array."""
    from .hamiltonian import (minimize_hamiltonian,
                              minimize_hard_common_hamiltonian)
    from .safety import cont_bounds, deploy_bounds
    d = value_and_derivatives(model, t, s, y, pz, second_order=False)
    v_s = d["v_s"].detach().cpu().numpy()
    s_np = s.detach().cpu().numpy()
    p = model.p
    if bounds == "deploy":
        lo, hi = deploy_bounds(s_np, p, dt if dt is not None else p.dt_ctrl)
    else:
        lo, hi = cont_bounds(s_np, p)
    solver = (minimize_hard_common_hamiltonian
              if objective == "hard_common" else minimize_hamiltonian)
    if objective not in {"smooth", "hard_common"}:
        raise ValueError(f"unknown greedy objective: {objective}")
    a, _ = solver(t.detach().cpu().numpy(), s_np,
                  y.detach().cpu().numpy(), pz.detach().cpu().numpy(),
                  v_s, p, prof, lo, hi)
    return a


def hjb_residual(model: ValueNet, t, s, y, pz, prof: RegimeProfile):
    """r_v^* via Danskin: minimizer computed on detached v_s (GPU torch
    solver), then substituted as a constant into the torch residual."""
    a_t = greedy_action_torch(model, t, s, y, pz, prof,
                              bounds="deploy", dt=model.p.dt_ctrl)
    r, d = policy_residual(model, t, s, y, pz, a_t, prof)
    return r, a_t, d


def neumann_residual(model: ValueNet, t, s, y, pz):
    """(v_y at y-boundaries, v_p at p-boundaries) for penalty and reporting."""
    d = value_and_derivatives(model, t, s, y, pz, second_order=False)
    return d["v_y"], d["v_p"]
