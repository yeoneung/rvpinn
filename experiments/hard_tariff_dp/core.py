"""Independent discrete hard-cost DP on an interpolated (SoC, load-error) grid.

Gaussian reflected OU transition probabilities integrate load-grid hat functions.
No smooth tariff, PDE residual, historical trajectory, or MIQP is used here.
The final step evaluates the quadratic terminal cost without interpolation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr


def reflected_transition(y, kappa, sigma, dt):
    lo, hi = float(y[0]), float(y[-1])
    length = hi - lo
    mu = np.exp(-kappa * dt) * y[:, None]
    sd = sigma * np.sqrt(dt if kappa == 0 else -np.expm1(-2*kappa*dt)/(2*kappa))
    if sd <= 0:
        raise ValueError('This benchmark requires positive load diffusion')
    left, right = y[:-1][None, :], y[1:][None, :]
    P = np.zeros((len(y), len(y)))
    # Enough mirror images that omitted Gaussian tails are negligible.
    images = int(np.ceil((abs(mu).max()+abs(lo)+10*sd)/length)) + 2
    for image in range(-images, images+1):
        for center in (mu-2*image*length, 2*lo+2*image*length-mu):
            a, b = (left-center)/sd, (right-center)/sd
            mass = ndtr(b)-ndtr(a)
            moment = center*mass + sd/np.sqrt(2*np.pi)*(np.exp(-a*a/2)-np.exp(-b*b/2))
            lower = (right*mass-moment)/(right-left)
            upper = (moment-left*mass)/(right-left)
            P[:, :-1] += lower
            P[:, 1:] += upper
    if np.min(P) < -1e-10 or np.max(np.abs(P.sum(axis=1)-1)) > 1e-9:
        raise ArithmeticError('Invalid reflected OU transition matrix')
    P = np.maximum(P, 0.)
    return P/P.sum(axis=1, keepdims=True)


def bounds(p, s):
    lo = -np.minimum.reduce([np.full_like(s, p.a_c), p.a_c*(p.s_max-s)/p.delta_s,
                            p.E_max*(p.s_max-s)/(p.eta_c*p.dt_ctrl)])
    hi = np.minimum.reduce([np.full_like(s, p.a_d), p.a_d*(s-p.s_min)/p.delta_s,
                           p.eta_d*p.E_max*(s-p.s_min)/p.dt_ctrl])
    return lo, hi


def following_soc(p, s, a):
    return s + p.dt_ctrl/p.E_max*(p.eta_c*np.maximum(-a, 0)-np.maximum(a, 0)/p.eta_d)


def stage(p, net, price, a):
    g = net-a
    return p.dt_ctrl*(price*(np.maximum(g, 0)-p.alpha_s*np.maximum(-g, 0))
                      +p.lam1*np.abs(a)+p.c_step*(g > p.g_thr))


def crossings(p, net):
    cross = net-p.g_thr
    inactive = np.where(net-cross <= p.g_thr, cross, net-np.nextafter(p.g_thr, -np.inf))
    active = np.nextafter(cross, -np.inf)
    active = np.where(net-active > p.g_thr, active, net-np.nextafter(p.g_thr, np.inf))
    for _ in range(4):
        inactive = np.where(net-inactive > p.g_thr, np.nextafter(inactive, np.inf), inactive)
        active = np.where(net-active <= p.g_thr, np.nextafter(active, -np.inf), active)
    return cross, inactive, active


def terminal_action(p, s, net, price):
    """Enumerate all piecewise quadratic stationary points and breakpoints."""
    s, net = np.broadcast_arrays(s, net)
    lo, hi = bounds(p, s)
    raw = [lo, hi, np.zeros_like(s), net, *crossings(p, net)]
    for sign in (-1., 1.):
        b = p.dt_ctrl/p.E_max*(p.eta_c if sign < 0 else 1/p.eta_d)
        for factor in (1., p.alpha_s):
            linear = p.dt_ctrl*(-price*factor+p.lam1*sign)
            root = (2*p.lam_T*b*(s-p.s_tar)-linear)/(2*p.lam_T*b*b)
            raw.append(root)
    a = np.clip(np.stack(raw, axis=-1), lo[..., None], hi[..., None])
    q = stage(p, net[..., None], price, a)+p.lam_T*(following_soc(p, s[..., None], a)-p.s_tar)**2
    ix = np.argmin(q, axis=-1)
    return np.take_along_axis(a, ix[..., None], axis=-1)[..., 0]


@dataclass
class GridDP:
    p: object
    profile: object
    ns: int
    ny: int
    na: int

    def __post_init__(self):
        p = self.p
        self.s = np.linspace(p.s_min, p.s_max, self.ns)
        self.y = np.unique(np.r_[np.linspace(p.y_min, p.y_max, self.ny), 0.])
        self.ny = len(self.y)
        self.S, self.Y = np.meshgrid(self.s, self.y, indexing='ij')
        self.K = int(round(p.T/p.dt_ctrl))
        self.times = np.arange(self.K)*p.dt_ctrl
        self.net = np.asarray(self.profile.n_bar(self.times))[:, None]+self.y[None, :]
        self.price = np.asarray(self.profile.c_bar(self.times))
        self.P = reflected_transition(self.y, p.kappa_y, p.sigma_y, p.dt_ctrl)

    def expected(self, value):
        return value @ self.P.T

    def interpolate_s(self, expected, next_s, y_index):
        z = np.clip((next_s-self.s[0])/(self.s[1]-self.s[0]), 0., self.ns-1)
        i = np.minimum(np.floor(z).astype(int), self.ns-2)
        w = z-i
        return (1-w)*expected[i, y_index]+w*expected[i+1, y_index]

    def q(self, k, actions, expected):
        now = stage(self.p, self.net[k][None, :], self.price[k], actions)
        sn = following_soc(self.p, self.S, actions)
        if k == self.K-1:
            return now+self.p.lam_T*(sn-self.p.s_tar)**2
        return now+self.interpolate_s(expected, sn, np.arange(self.ny)[None, :])

    def minimize(self, k, expected):
        p = self.p
        if k == self.K-1:
            a = terminal_action(p, self.S, self.net[k][None, :], self.price[k])
            return self.q(k, a, expected), a
        values, actions = np.empty_like(self.S), np.empty_like(self.S)
        for start in range(0, self.ns, 8):
            stop = min(start+8, self.ns)
            s = self.s[start:stop, None, None]
            net = self.net[k][None, :, None]
            lo, hi = bounds(p, s)
            uniform = lo+(hi-lo)*np.linspace(0, 1, self.na)[None, None, :]
            delta = self.s[None, None, :]-s
            knots = np.where(delta >= 0, -delta*p.E_max/(p.eta_c*p.dt_ctrl),
                             -delta*p.eta_d*p.E_max/p.dt_ctrl)
            pieces = [uniform, knots, np.zeros_like(net), net,
                      *[c for c in crossings(p, net)]]
            a = np.concatenate([np.broadcast_to(x, (stop-start, self.ny, x.shape[-1])) for x in pieces], axis=-1)
            a = np.clip(a, lo, hi)
            sn = following_soc(p, s, a)
            q = stage(p, net, self.price[k], a)+self.interpolate_s(expected, sn, np.arange(self.ny)[None, :, None])
            ix = np.argmin(q, axis=-1)
            actions[start:stop] = np.take_along_axis(a, ix[..., None], axis=-1)[..., 0]
            values[start:stop] = np.take_along_axis(q, ix[..., None], axis=-1)[..., 0]
        return values, actions

    def solve(self):
        V = np.empty((self.K+1, self.ns, self.ny))
        A = np.empty((self.K, self.ns, self.ny))
        V[-1] = self.p.lam_T*(self.S-self.p.s_tar)**2
        for k in range(self.K-1, -1, -1):
            V[k], A[k] = self.minimize(k, self.expected(V[k+1]))
        return V, A

    def evaluate(self, policy):
        V = self.p.lam_T*(self.S-self.p.s_tar)**2
        for k in range(self.K-1, -1, -1):
            V = self.q(k, policy[k], self.expected(V))
        return self.initial_costs(V)

    def initial_costs(self, value):
        j = int(np.argmin(np.abs(self.y)))
        return np.interp([.2, .5, .8], self.s, value[:, j])

    def rule_actions(self, setting):
        if setting is None:
            return np.zeros((self.K, self.ns, self.ny))
        p, s = self.p, self.S
        lo, hi = bounds(p, s)
        fields = []
        for k, t in enumerate(self.times):
            net, price = self.net[k][None, :], self.price[k]
            if k == self.K-1:
                fields.append(terminal_action(p, s, net, price))
                continue
            recovery = max(p.s_min, p.s_tar-p.eta_c*p.a_c*max(0, p.T-t-p.dt_ctrl)/p.E_max)
            floor = max(setting['reserve_soc'], recovery)
            shave_hi = np.maximum(0, np.minimum(hi, (s-floor)*p.eta_d*p.E_max/p.dt_ctrl))
            inactive = crossings(p, net)[1]
            clear = (net > p.g_thr) & (inactive >= 0) & (inactive <= shave_hi)
            shave = np.where(clear, inactive, shave_hi)
            use_shave = (net > p.g_thr) & (clear | setting['allow_partial_shaving']) & (shave > 0)
            use_shave &= stage(p, net, price, shave) < stage(p, net, price, np.zeros_like(s))
            charge = np.maximum(lo, -(setting['charge_target_soc']-s)*p.E_max/(p.eta_c*p.dt_ctrl))
            charge = np.where(net <= p.g_thr, np.maximum(charge, inactive), charge)
            if not setting['charge_when_already_above_band']:
                charge = np.where(net > p.g_thr, 0., charge)
            charge = np.clip(charge, lo, hi)
            a = np.where((price <= setting['charge_price']) & (s < setting['charge_target_soc']), charge, 0.)
            a = np.where(use_shave, shave, a)
            a = np.where(s < recovery, np.clip(-(recovery-s)*p.E_max/(p.eta_c*p.dt_ctrl), lo, 0.), a)
            fields.append(a)
        return np.asarray(fields)

    def reference_actions(self):
        return np.asarray([terminal_action(self.p, self.S, self.net[k][None, :], self.price[k])
                           for k in range(self.K)])

    def learned_actions(self, model):
        import torch
        fields = []
        device = next(model.parameters()).device
        s = torch.as_tensor(self.S.ravel(), device=device)
        y = torch.as_tensor(self.Y.ravel(), device=device)
        with torch.no_grad():
            for k in range(self.K):
                t = torch.full_like(s, (k+1)*self.p.dt_ctrl)
                v = model(t, s, y, torch.zeros_like(s)).cpu().numpy().reshape(self.S.shape)
                _, a = self.minimize(k, self.expected(v))
                fields.append(a)
        return np.asarray(fields)

    def regret(self, V, optimal, policy):
        all_regrets, boundary = [], []
        for k in range(self.K):
            regret = self.q(k, policy[k], self.expected(V[k+1]))-V[k]
            mask = np.abs(self.net[k][None, :]-optimal[k]-self.p.g_thr) <= 10.
            all_regrets.append(regret.ravel())
            boundary.append(regret[mask])
        overall, near = np.concatenate(all_regrets), np.concatenate(boundary)
        return dict(mean_regret=float(overall.mean()), minimum_regret=float(overall.min()),
                    boundary_states=int(len(near)), boundary_mean=float(near.mean()),
                    boundary_p95=float(np.quantile(near, .95)), boundary_max=float(near.max()))
