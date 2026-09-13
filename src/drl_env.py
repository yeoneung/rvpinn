"""Gymnasium environment for SAC/TD3 training in the calibrated simulator.

Unified regime-conditioned implementation: one model with a regime one-hot,
identical for SAC and TD3. Environment dynamics, cost accounting, and the
safe projection are float64; only the observation/action interface exposed
to SB3 is float32. Actions in [-1, 1] are mapped affinely onto the current
deployment-safe interval, so every executed action is feasible by
construction (identical safety gate as all other methods).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

from .config import ModelParams
from .costs import RegimeProfile, psi_eps, step_fee_smooth
from .dynamics import correlated_normals, ou_step_exact, reflect, soc_step
from .safety import deploy_bounds


class MicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params_by_regime: Dict[str, ModelParams],
                 profiles_by_regime: Dict[str, RegimeProfile],
                 seed: int = 0, reward_scale: Optional[float] = None,
                 fixed_regime: Optional[str] = None):
        super().__init__()
        self.regimes: List[str] = sorted(params_by_regime.keys())
        self.params = params_by_regime
        self.profiles = profiles_by_regime
        self.fixed_regime = fixed_regime
        self.rng = np.random.default_rng(seed)
        n_reg = len(self.regimes)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 + n_reg,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        p0 = self.params[self.regimes[0]]
        self.n_ctrl = int(round(p0.T / p0.dt_ctrl))
        self.sub = int(round(p0.dt_ctrl / p0.dt_sim))
        # one common reward scale for SAC and TD3
        self.reward_scale = reward_scale or 1.0

    def _obs(self) -> np.ndarray:
        p = self.p
        t = self.k * p.dt_ctrl
        onehot = np.zeros(len(self.regimes))
        onehot[self.regimes.index(self.regime)] = 1.0
        o = np.concatenate([
            [2 * (self.s - p.s_min) / (p.s_max - p.s_min) - 1,
             2 * (self.y - p.y_min) / (p.y_max - p.y_min) - 1,
             2 * (self.pz - p.p_min) / (p.p_max - p.p_min) - 1,
             np.sin(2 * np.pi * t / p.T), np.cos(2 * np.pi * t / p.T)],
            onehot])
        return o.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.regime = self.fixed_regime or \
            self.regimes[self.rng.integers(len(self.regimes))]
        self.p = self.params[self.regime]
        self.prof = self.profiles[self.regime]
        self.s = self.p.s0
        self.y = 0.0
        self.pz = 0.0
        self.k = 0
        return self._obs(), {}

    def step(self, action):
        p, prof = self.p, self.prof
        t = self.k * p.dt_ctrl
        u = float(np.clip(np.asarray(action, dtype=np.float64).ravel()[0],
                          -1.0, 1.0))
        lo, hi = deploy_bounds(np.array([self.s]), p, p.dt_ctrl)
        a = float(lo[0] + 0.5 * (u + 1.0) * (hi[0] - lo[0]))
        cost = 0.0
        for j in range(self.sub):
            ts = t + j * p.dt_sim
            nbar = float(prof.n_bar(np.array([ts]))[0])
            cbar = float(prof.c_bar(np.array([ts]))[0])
            G = nbar + self.y - a
            C = cbar + self.pz
            imp = psi_eps(G, p.eps_g)
            exp_ = psi_eps(-G, p.eps_g)
            ell = (C * imp - p.alpha_s * C * exp_
                   + p.lam_pk * imp * imp
                   + p.lam1 * (np.sqrt(a * a + p.eps_a ** 2) - p.eps_a)
                   + p.lam2 * a * a
                   + p.lam_s * (self.s - p.s_ref) ** 2
                   + step_fee_smooth(G, p))
            cost += ell * p.dt_sim
            self.s = float(soc_step(self.s, a, p.dt_sim, p))
            e1, e2 = correlated_normals(self.rng, p.rho, ())
            self.y = float(reflect(
                ou_step_exact(np.array([self.y]), p.kappa_y, p.sigma_y,
                              p.dt_sim, np.array([e1])),
                p.y_min, p.y_max)[0])
            self.pz = float(reflect(
                ou_step_exact(np.array([self.pz]), p.kappa_p, p.sigma_p,
                              p.dt_sim, np.array([e2])),
                p.p_min, p.p_max)[0])
        self.k += 1
        terminated = self.k >= self.n_ctrl
        if terminated:
            cost += p.lam_T * (self.s - p.s_tar) ** 2
        reward = -cost / self.reward_scale
        return self._obs(), float(reward), terminated, False, {}
