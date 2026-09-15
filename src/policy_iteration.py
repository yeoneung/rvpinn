"""RV-PINN-PI trainer and direct HJB-PINN trainer (main.tex §6, Alg. 1).

Point-set discipline:
  - training batches: fresh scrambled Sobol every step (seed stream from
    training_design seed) + persistent adaptive buffer
  - adaptive candidates: separate seed stream
  - tuning-validation design: frozen, used for early stopping / checkpoint
    selection / stopping tests
  - certification design: frozen, evaluated ONLY after checkpoint freeze.
All PDE math in float64.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .config import ModelParams, ROOT
from .costs import RegimeProfile
from .dynamics import ou_step_exact, reflect, soc_step, correlated_normals
from .hamiltonian import minimize_hamiltonian
from .pinn_value import (DTYPE, ValueNet, greedy_action_np, hjb_residual,
                         policy_residual, value_and_derivatives,
                         value_rate_scale)
from .residual_validation import (boundary_points, residual_metrics,
                                  sobol_points,
                                  lipschitz_sensitivity_estimate)
from .safety import cont_bounds, deploy_bounds


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class PolicyFn:
    """pi_n: greedy w.r.t. a frozen value model (or zero for pi_0)."""

    def __init__(self, model: Optional[ValueNet], p: ModelParams,
                 prof: RegimeProfile):
        self.model = model
        self.p = p
        self.prof = prof

    def __call__(self, t: torch.Tensor, s: torch.Tensor, y: torch.Tensor,
                 pz: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return torch.zeros_like(s)
        from .pinn_value import greedy_action_torch
        return greedy_action_torch(self.model, t, s, y, pz, self.prof,
                                   bounds="deploy", dt=self.p.dt_ctrl)


class EvalResult(dict):
    pass


class BasePINNTrainer:
    def __init__(self, p: ModelParams, prof: RegimeProfile, cfg: Dict,
                 seeds: Dict, out_scale: float, use_p: bool = True,
                 device: Optional[str] = None, method_seed: int = 0,
                 adaptive: bool = True, width: Optional[int] = None,
                 depth: Optional[int] = None, hard_terminal: bool = True):
        self.p, self.prof, self.cfg = p, prof, cfg
        self.seeds = seeds
        self.use_p = use_p
        self.method_seed = method_seed
        self.adaptive = adaptive
        self.hard_terminal = hard_terminal
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net_cfg = cfg["network"]
        self.width = width or int(net_cfg["width"])
        self.depth = depth or int(net_cfg["hidden_layers"])
        self.reference_baseline = str(
            net_cfg.get("reference_baseline", "zero_action_conditional_mean"))
        self.reference_quadrature_points = int(
            net_cfg.get("reference_quadrature_points", 24))
        # `out_scale` arrives as B0/T (bill rate). The ansatz scale must
        # instead match the value RATE near t=T, which is dominated by
        # L^a Phi; otherwise the network
        # cannot reach the required output magnitude within budget.
        raw_rate_scale = value_rate_scale(p, prof)
        if self.reference_baseline == "terminal_only":
            self.rate_scale = max(float(out_scale), raw_rate_scale)
        else:
            # The network now approximates a correction to a no-storage
            # cost-to-go.  The geometric mean retains capacity for the sharp
            # near-terminal correction without making an O(1) network output
            # worth tens of thousands of currency units at early times.
            self.rate_scale = max(float(out_scale), float(p.c_step),
                                  float(np.sqrt(max(float(out_scale), 1e-12)
                                                * raw_rate_scale)))
        self.out_scale = self.rate_scale
        # value-magnitude scale for Neumann/terminal normalization
        self.V_scale = (p.lam_T * max((p.s_max - p.s_tar) ** 2,
                                      (p.s_min - p.s_tar) ** 2)
                        + float(out_scale) * p.T)
        pe = cfg["policy_evaluation"]
        self.adam_steps = int(pe["adam_max_steps"])
        self.batch = int(pe["adam_batch_size"])
        self.lr0 = float(pe["lr_start"])
        self.lr1 = float(pe["lr_end"])
        self.lbfgs_iter = int(pe["lbfgs_max_iter"])
        self.n_boundary = int(pe["boundary_points"])
        self.n_cert = int(pe["certification_points"])
        self.n_tuneval = int(pe["tuning_validation_points"])
        self.eval_every = int(pe["eval_every_steps"])
        self.patience = int(pe.get("early_stop_patience_evals", 10))
        ar = cfg["adaptive_refinement"]
        self.pool_size = int(ar["candidate_pool_size"])
        self.refresh_every = int(ar["refresh_every_steps"])
        self.top_q = float(ar["top_quantile"])
        self.max_buffer = int(ar.get("max_buffer_size", 131072))
        self.lam_neu = 1.0
        # frozen designs
        self.tuneval = sobol_points(self.n_tuneval, p,
                                    int(seeds["tuning_validation"]),
                                    use_p, self.device)
        self.tuneval_b = boundary_points(4096, p,
                                         int(seeds["tuning_validation"]) + 1,
                                         use_p, self.device)
        self._cert = None      # generated lazily at freeze time
        self._sobol_ctr = 0
        self.buffer: Optional[Dict[str, torch.Tensor]] = None
        # normalization scales for the composite loss
        self.r_scale = self.rate_scale               # currency / hour
        # v_y * (y-range) / V_scale is dimensionless O(1)
        self.vy_scale = (p.y_max - p.y_min) / self.V_scale
        self.vp_scale = (p.p_max - p.p_min) / self.V_scale

    # ----- point streams ------------------------------------------------
    def fresh_batch(self, n: int) -> Dict[str, torch.Tensor]:
        self._sobol_ctr += 1
        seed = int(self.seeds["training_design"]) + 1000 * self.method_seed \
            + self._sobol_ctr
        return sobol_points(n, self.p, seed, self.use_p, self.device)

    def candidate_pool(self) -> Dict[str, torch.Tensor]:
        self._sobol_ctr += 1
        seed = int(self.seeds["adaptive_candidate"]) + 1000 * self.method_seed \
            + self._sobol_ctr
        return sobol_points(self.pool_size, self.p, seed, self.use_p,
                            self.device)

    def cert_design(self) -> Dict[str, torch.Tensor]:
        if self._cert is None:
            self._cert = sobol_points(self.n_cert, self.p,
                                      int(self.seeds["final_certification"]),
                                      self.use_p, self.device)
        return self._cert

    # ----- residuals ----------------------------------------------------
    def residual_on(self, model: ValueNet, pts: Dict[str, torch.Tensor],
                    policy: Callable) -> torch.Tensor:
        a = policy(pts["t"], pts["s"], pts["y"], pts["pz"])
        r, _ = policy_residual(model, pts["t"], pts["s"], pts["y"],
                               pts["pz"], a, self.prof)
        return r

    def neumann_on(self, model: ValueNet, pts: Dict[str, torch.Tensor]):
        d = value_and_derivatives(model, pts["t"], pts["s"], pts["y"],
                                  pts["pz"], second_order=False)
        is_y = pts["is_y_edge"]
        vy = d["v_y"][is_y] * self.vy_scale
        vp = d["v_p"][~is_y] * self.vp_scale if self.use_p else vy[:0]
        return vy, vp

    def loss_terms(self, model: ValueNet, pts, bpts, policy):
        r = self.residual_on(model, pts, policy) / self.r_scale
        vy, vp = self.neumann_on(model, bpts)
        loss_pde = (r ** 2).mean()
        loss_neu = (vy ** 2).mean() + ((vp ** 2).mean() if vp.numel() else 0.0)
        loss = loss_pde + self.lam_neu * loss_neu
        if not getattr(model, "hard_terminal", True):
            loss = loss + self.terminal_penalty(model, pts)
        return loss, loss_pde, loss_neu

    def terminal_penalty(self, model: ValueNet, pts) -> torch.Tensor:
        """MSE terminal mismatch at t = T (ablation A2 only)."""
        p = self.p
        tT = torch.full_like(pts["s"], p.T)
        v = model(tT, pts["s"], pts["y"], pts["pz"])
        phi = p.lam_T * (pts["s"] - p.s_tar) ** 2
        return (((v - phi) / self.V_scale) ** 2).mean()

    # ----- evaluation on frozen tuning design ---------------------------
    def evaluate_tuning(self, model: ValueNet, policy,
                        cached_actions: Optional[torch.Tensor] = None) -> Dict:
        n = self.tuneval["t"].shape[0]
        chunks = []
        for i0 in range(0, n, 16384):
            pts = {k: v[i0:i0 + 16384] for k, v in self.tuneval.items()}
            if cached_actions is not None:
                a = cached_actions[i0:i0 + 16384]
                r, _ = policy_residual(model, pts["t"], pts["s"], pts["y"],
                                       pts["pz"], a, self.prof)
                chunks.append(r.detach())
            else:
                chunks.append(self.residual_on(model, pts, policy).detach())
        r = torch.cat(chunks)
        return residual_metrics(r)

    # ----- boundary sample for training ---------------------------------
    def fresh_boundary(self, n: int) -> Dict[str, torch.Tensor]:
        self._sobol_ctr += 1
        seed = int(self.seeds["training_design"]) + 500_000 \
            + 1000 * self.method_seed + self._sobol_ctr
        return boundary_points(n, self.p, seed, self.use_p, self.device)

    # ----- epoch pool with precomputed policy actions -------------------
    def _refresh_pool(self, policy):
        """Fresh Sobol epoch pool; policy actions precomputed once (the
        policy is frozen within a policy-evaluation stage)."""
        n_pool = max(8 * self.batch, 32768)
        pts = self.fresh_batch(n_pool)
        with torch.no_grad():
            a = torch.cat([policy(pts["t"][i0:i0 + 16384],
                                  pts["s"][i0:i0 + 16384],
                                  pts["y"][i0:i0 + 16384],
                                  pts["pz"][i0:i0 + 16384])
                           for i0 in range(0, n_pool, 16384)])
        self._pool = (pts, a)

    def _draw_batch(self):
        pts, a = self._pool
        n_pool = a.shape[0]
        if self.buffer is not None and self.adaptive \
                and self.buffer["t"].shape[0] > 0:
            n_half = self.batch // 2
            idx = torch.randint(n_pool, (self.batch - n_half,),
                                device=self.device)
            nb = self.buffer["t"].shape[0]
            jdx = torch.randint(nb, (min(n_half, nb),), device=self.device)
            batch_pts = {k: torch.cat([pts[k][idx], self.buffer[k][jdx]])
                         for k in ("t", "s", "y", "pz")}
            batch_a = torch.cat([a[idx], self.buffer["a"][jdx]])
        else:
            idx = torch.randint(n_pool, (self.batch,), device=self.device)
            batch_pts = {k: pts[k][idx] for k in ("t", "s", "y", "pz")}
            batch_a = a[idx]
        return batch_pts, batch_a

    # ----- adam + lbfgs -------------------------------------------------
    def train_value(self, model: ValueNet, policy, log_prefix: str = "",
                    history: Optional[List] = None) -> Dict:
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.adam_steps, 1), eta_min=self.lr1)
        best = {"metric": float("inf"), "state": None, "step": -1}
        bad_evals = 0
        t0 = time.time()
        # cache tuning-validation actions for this frozen policy
        with torch.no_grad():
            tv_a = torch.cat([policy(self.tuneval["t"][i0:i0 + 16384],
                                     self.tuneval["s"][i0:i0 + 16384],
                                     self.tuneval["y"][i0:i0 + 16384],
                                     self.tuneval["pz"][i0:i0 + 16384])
                              for i0 in range(0, self.n_tuneval, 16384)])
        self.buffer = None            # buffer actions are policy-specific
        self._refresh_pool(policy)
        for step in range(self.adam_steps):
            if step > 0 and step % self.refresh_every == 0:
                self._refresh_pool(policy)
                if self.adaptive:
                    self._refresh_buffer(model, policy)
            pts, a_const = self._draw_batch()
            bpts = self.fresh_boundary(max(256, self.batch // 8))
            r, _ = policy_residual(model, pts["t"], pts["s"], pts["y"],
                                   pts["pz"], a_const, self.prof)
            vy, vp = self.neumann_on(model, bpts)
            loss = ((r / self.r_scale) ** 2).mean() + self.lam_neu * (
                (vy ** 2).mean() + ((vp ** 2).mean() if vp.numel() else 0.0))
            if not getattr(model, "hard_terminal", True):
                loss = loss + self.terminal_penalty(model, pts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % self.eval_every == 0 or step == self.adam_steps - 1:
                m = self.evaluate_tuning(model, policy, cached_actions=tv_a)
                if history is not None:
                    history.append({"stage": "adam", "step": step + 1,
                                    "train_loss": float(loss.item()),
                                    **{f"tuneval_{k}": v for k, v in m.items()}})
                if m["rms"] < best["metric"]:
                    best = {"metric": m["rms"],
                            "state": {k: v.detach().clone()
                                      for k, v in model.state_dict().items()},
                            "step": step + 1}
                    bad_evals = 0
                else:
                    bad_evals += 1
                if bad_evals >= self.patience:
                    break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        # ---- L-BFGS polish on a fixed design with frozen policy actions ----
        if self.lbfgs_iter > 0:
            pts = self.fresh_batch(min(4 * self.batch, 32768))
            bpts = self.fresh_boundary(2048)
            a_fixed = policy(pts["t"], pts["s"], pts["y"], pts["pz"])
            lbfgs = torch.optim.LBFGS(model.parameters(),
                                      max_iter=self.lbfgs_iter,
                                      history_size=50,
                                      tolerance_grad=1e-12,
                                      tolerance_change=1e-14,
                                      line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                r, _ = policy_residual(model, pts["t"], pts["s"], pts["y"],
                                       pts["pz"], a_fixed, self.prof)
                vy, vp = self.neumann_on(model, bpts)
                loss = ((r / self.r_scale) ** 2).mean() \
                    + self.lam_neu * ((vy ** 2).mean()
                                      + ((vp ** 2).mean() if vp.numel() else 0.0))
                loss.backward()
                return loss

            pre = self.evaluate_tuning(model, policy,
                                       cached_actions=tv_a)["rms"]
            state_pre = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
            try:
                lbfgs.step(closure)
                post = self.evaluate_tuning(model, policy,
                                            cached_actions=tv_a)["rms"]
                if not np.isfinite(post) or post > pre:
                    model.load_state_dict(state_pre)     # reject bad polish
            except Exception:
                model.load_state_dict(state_pre)
        m = self.evaluate_tuning(model, policy, cached_actions=tv_a)
        return {"tuneval": m, "best_adam_step": best["step"],
                "wall_s": time.time() - t0,
                "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                if torch.cuda.is_available() else None)}

    def _refresh_buffer(self, model: ValueNet, policy):
        pool = self.candidate_pool()
        with torch.no_grad():
            a_pool = torch.cat([policy(pool["t"][i0:i0 + 16384],
                                       pool["s"][i0:i0 + 16384],
                                       pool["y"][i0:i0 + 16384],
                                       pool["pz"][i0:i0 + 16384])
                                for i0 in range(0, self.pool_size, 16384)])
        chunks = []
        for i0 in range(0, self.pool_size, 16384):
            pts = {k: v[i0:i0 + 16384] for k, v in pool.items()}
            r, _ = policy_residual(model, pts["t"], pts["s"], pts["y"],
                                   pts["pz"], a_pool[i0:i0 + 16384],
                                   self.prof)
            chunks.append(r.detach())
        r = torch.cat(chunks)
        thr = torch.quantile(r.abs(), self.top_q)
        sel = r.abs() >= thr
        new = {k: pool[k][sel] for k in ("t", "s", "y", "pz")}
        new["a"] = a_pool[sel]
        if self.buffer is None:
            self.buffer = new
        else:
            self.buffer = {k: torch.cat([self.buffer[k], new[k]])[-self.max_buffer:]
                           for k in ("t", "s", "y", "pz", "a")}

    # ----- common-random-number rollout cost -----------------------------
    def rollout_cost(self, model: Optional[ValueNet], n_paths: int = 128,
                     seed: int = 7777) -> float:
        p = self.p
        rng = np.random.default_rng(seed)
        dt_c, dt_s = p.dt_ctrl, p.dt_sim
        sub = int(round(dt_c / dt_s))
        n_ctrl = int(round(p.T / dt_c))
        s = np.full(n_paths, p.s0)
        y = np.zeros(n_paths)
        pz = np.zeros(n_paths)
        total = np.zeros(n_paths)
        for k in range(n_ctrl):
            t = k * dt_c
            tt = torch.full((n_paths,), t, dtype=DTYPE, device=self.device)
            st = torch.as_tensor(s, dtype=DTYPE, device=self.device)
            yt = torch.as_tensor(y, dtype=DTYPE, device=self.device)
            pt = torch.as_tensor(pz, dtype=DTYPE, device=self.device)
            if model is None:
                a = np.zeros(n_paths, dtype=np.float64)
            else:
                a = greedy_action_np(
                    model, tt, st, yt, pt, self.prof,
                    bounds="deploy", dt=dt_c, objective="hard_common")
            lo, hi = deploy_bounds(s, p, dt_c)
            a = np.minimum(np.maximum(a, lo), hi)
            for j in range(sub):
                ts = t + j * dt_s
                tarr = np.full(n_paths, ts)
                nbar = self.prof.n_bar(tarr)
                cbar = self.prof.c_bar(tarr)
                from .costs import common_running_cost_exact
                G = nbar + y - a
                C = cbar + pz
                ell = common_running_cost_exact(C, G, a, p)
                total += ell * dt_s
                s = soc_step(s, a, dt_s, p)
                e1, e2 = correlated_normals(rng, p.rho, n_paths)
                y = reflect(ou_step_exact(y, p.kappa_y, p.sigma_y, dt_s, e1),
                            p.y_min, p.y_max)
                pz = reflect(ou_step_exact(pz, p.kappa_p, p.sigma_p, dt_s, e2),
                             p.p_min, p.p_max)
        total += p.lam_T * (s - p.s_tar) ** 2
        return float(total.mean())


class PINNPITrainer(BasePINNTrainer):
    """Full RV-PINN-PI loop (Algorithm 1)."""

    def run(self, out_dir: str, max_iterations: Optional[int] = None,
            tau_eval: Optional[float] = None, tau_pi: Optional[float] = None
            ) -> Dict:
        os.makedirs(out_dir, exist_ok=True)
        cfg_pi = self.cfg["policy_iteration"]
        K = max_iterations or int(cfg_pi["max_iterations"])
        tau_eval = tau_eval or cfg_pi.get("tau_eval")
        tau_pi = tau_pi or cfg_pi.get("tau_pi")
        p = self.p
        prev_model: Optional[ValueNet] = None
        iters: List[Dict] = []
        prev_pol_actions = None
        model = None
        for n in range(K):
            policy = PolicyFn(prev_model, p, self.prof)
            if model is None:
                model = ValueNet(p, width=self.width, depth=self.depth,
                                 use_p=self.use_p, out_scale=self.out_scale,
                                 seed=1000 + self.method_seed,
                                 hard_terminal=self.hard_terminal,
                                 reference_baseline=self.reference_baseline,
                                 reference_quadrature_points=
                                     self.reference_quadrature_points)
                model.attach_profile(self.prof)
            history: List = []
            info = self.train_value(model, policy, history=history)
            # ---- diagnostics on the frozen tuning design ----
            tv = self.tuneval
            with torch.enable_grad():
                a_next_t = torch.as_tensor(
                    greedy_action_np(model, tv["t"], tv["s"], tv["y"],
                                     tv["pz"], self.prof),
                    dtype=DTYPE, device=self.device)
                a_cur_t = policy(tv["t"], tv["s"], tv["y"], tv["pz"])
                r_pi, d = policy_residual(model, tv["t"], tv["s"], tv["y"],
                                          tv["pz"], a_cur_t, self.prof)
                r_next, _ = policy_residual(model, tv["t"], tv["s"], tv["y"],
                                            tv["pz"], a_next_t, self.prof, d=None)
            hjb_max = float(r_next.detach().abs().max())
            greedy_err = float((r_pi - r_next).detach().abs().max())
            pol_change = float((a_next_t - a_cur_t).detach().abs().max()
                               / max(p.a_max, 1.0))
            cost = self.rollout_cost(model)
            ckpt = os.path.join(out_dir, f"iter{n:02d}.pt")
            torch.save({"state_dict": model.state_dict(),
                        "width": self.width, "depth": self.depth,
                        "use_p": self.use_p, "out_scale": self.out_scale,
                        "hard_terminal": self.hard_terminal,
                        "reference_baseline": self.reference_baseline,
                        "reference_quadrature_points":
                            self.reference_quadrature_points,
                        "profile_n_hour": self.prof.n_hour,
                        "profile_c_hour": self.prof.c_hour},
                       ckpt)
            rec = {"iteration": n,
                   "train_info": info,
                   "tuneval_policy_residual": info["tuneval"],
                   "hjb_residual_max_tuneval": hjb_max,
                   "greedy_error_tuneval": greedy_err,
                   "policy_change_norm": pol_change,
                   "rollout_cost_crn": cost,
                   "checkpoint": ckpt,
                   "history_len": len(history)}
            iters.append(rec)
            with open(os.path.join(out_dir, "iterations.json"), "w") as f:
                json.dump(iters, f, indent=2, default=str)
            print(f"  [PI n={n}] rms={info['tuneval']['rms']:.4g} "
                  f"max={info['tuneval']['max']:.4g} hjb_max={hjb_max:.4g} "
                  f"dpol={pol_change:.3g} cost={cost:.2f} "
                  f"({info['wall_s']:.0f}s)", flush=True)
            stop = (tau_eval is not None and tau_pi is not None
                    and info["tuneval"]["max"] <= tau_eval
                    and pol_change <= tau_pi)
            prev_model = ValueNet(p, width=self.width, depth=self.depth,
                                  use_p=self.use_p, out_scale=self.out_scale,
                                  seed=1000 + self.method_seed,
                                  hard_terminal=self.hard_terminal,
                                  reference_baseline=self.reference_baseline,
                                  reference_quadrature_points=
                                      self.reference_quadrature_points)
            prev_model.attach_profile(self.prof)
            prev_model.load_state_dict(model.state_dict())
            prev_model.to(self.device)
            if stop:
                break
        # ---- checkpoint freeze: sequential hard-cost incumbent rule ----
        # Residuals are diagnostics. A candidate replaces the incumbent only
        # when the CRN hard-objective mean improves by the frozen margin.
        margin = float(self.cfg.get("policy_selection", {}).get(
            "min_mean_improvement_eur_per_day", 0.10))
        best_n = 0
        for candidate in range(1, len(iters)):
            if (iters[candidate]["rollout_cost_crn"]
                    < iters[best_n]["rollout_cost_crn"] - margin):
                best_n = candidate
        result = {"iterations": iters, "selected_iteration": best_n,
                  "selected_checkpoint": iters[best_n]["checkpoint"],
                  "selection_criterion": "hard_common_crn_rollout_cost",
                  "selection_min_improvement_eur_per_day": margin}
        # ---- certification (frozen design, evaluated once, after freeze) ---
        model_sel = load_checkpoint(iters[best_n]["checkpoint"], p, self.device)
        prev_ck = iters[best_n - 1]["checkpoint"] if best_n >= 1 else None
        prev_sel = load_checkpoint(prev_ck, p, self.device) if prev_ck else None
        cert = self.certify(model_sel, PolicyFn(prev_sel, p, self.prof))
        result["certification"] = cert
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result

    def certify(self, model: ValueNet, policy) -> Dict:
        pts = self.cert_design()
        n = pts["t"].shape[0]
        rs, rh = [], []
        for i0 in range(0, n, 16384):
            sub = {k: v[i0:i0 + 16384] for k, v in pts.items()}
            rs.append(self.residual_on(model, sub, policy).detach())
            r_star, _, _ = hjb_residual(model, sub["t"], sub["s"], sub["y"],
                                        sub["pz"], self.prof)
            rh.append(r_star.detach())
        r_pi = torch.cat(rs)
        r_st = torch.cat(rh)
        bp = boundary_points(8192, self.p,
                             int(self.seeds["final_certification"]) + 1,
                             self.use_p, self.device)
        vy, vp = self.neumann_on(model, bp)
        lip = lipschitz_sensitivity_estimate(model, pts, self.prof)
        from .residual_validation import fill_distance_estimate
        dim = 4 if self.use_p else 3
        h_fill = fill_distance_estimate(n, dim)
        cert = {
            "policy_residual": residual_metrics(r_pi),
            "hjb_residual": residual_metrics(r_st),
            "neumann_vy_max_normalized": float(vy.detach().abs().max()),
            "neumann_vp_max_normalized": (float(vp.detach().abs().max())
                                          if vp.numel() else 0.0),
            "lipschitz_sensitivity_L": lip,
            "fill_distance_unit_cube": h_fill,
            # labeled per protocol: NOT a certified global bound
            "lipschitz_corrected_sensitivity_estimate":
                float(r_st.abs().max()) + lip * h_fill,
            "label": "independent validation maximum + "
                     "Lipschitz-corrected sensitivity estimate",
            "n_points": n,
        }
        return cert


class DirectHJBTrainer(BasePINNTrainer):
    """Direct nonlinear HJB-PINN with matched architecture and budget."""

    def hjb_residual_on(self, model, pts):
        r, _, _ = hjb_residual(model, pts["t"], pts["s"], pts["y"],
                               pts["pz"], self.prof)
        return r

    def run(self, out_dir: str, total_updates: Optional[int] = None) -> Dict:
        os.makedirs(out_dir, exist_ok=True)
        model = ValueNet(self.p, width=self.width, depth=self.depth,
                         use_p=self.use_p, out_scale=self.out_scale,
                         seed=2000 + self.method_seed,
                         reference_baseline=self.reference_baseline,
                         reference_quadrature_points=
                             self.reference_quadrature_points)
        model.attach_profile(self.prof)
        model.to(self.device)
        steps = total_updates or self.adam_steps
        opt = torch.optim.Adam(model.parameters(), lr=self.lr0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(steps, 1), eta_min=self.lr1)
        best = {"metric": float("inf"), "state": None}
        bad = 0
        t0 = time.time()
        history = []
        for step in range(steps):
            pts = self.fresh_batch(self.batch)
            bpts = self.fresh_boundary(max(256, self.batch // 8))
            r = self.hjb_residual_on(model, pts) / self.r_scale
            vy, vp = self.neumann_on(model, bpts)
            loss = (r ** 2).mean() + self.lam_neu * (
                (vy ** 2).mean() + ((vp ** 2).mean() if vp.numel() else 0.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % self.eval_every == 0 or step == steps - 1:
                rr = torch.cat([
                    self.hjb_residual_on(
                        model, {k: v[i0:i0 + 16384]
                                for k, v in self.tuneval.items()}).detach()
                    for i0 in range(0, self.n_tuneval, 16384)])
                m = residual_metrics(rr)
                history.append({"step": step + 1, **m})
                if m["rms"] < best["metric"]:
                    best = {"metric": m["rms"],
                            "state": {k: v.detach().clone()
                                      for k, v in model.state_dict().items()}}
                    bad = 0
                else:
                    bad += 1
                if bad >= self.patience:
                    break
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        ckpt = os.path.join(out_dir, "final.pt")
        torch.save({"state_dict": model.state_dict(), "width": self.width,
                    "depth": self.depth, "use_p": self.use_p,
                    "out_scale": self.out_scale,
                    "reference_baseline": self.reference_baseline,
                    "reference_quadrature_points":
                        self.reference_quadrature_points,
                    "profile_n_hour": self.prof.n_hour,
                    "profile_c_hour": self.prof.c_hour}, ckpt)
        rr = torch.cat([
            self.hjb_residual_on(
                model, {k: v[i0:i0 + 16384]
                        for k, v in self.tuneval.items()}).detach()
            for i0 in range(0, self.n_tuneval, 16384)])
        result = {"tuneval_hjb_residual": residual_metrics(rr),
                  "wall_s": time.time() - t0, "checkpoint": ckpt,
                  "history": history,
                  "rollout_cost_crn": self.rollout_cost(model)}
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        return result


def load_checkpoint(path: str, p: ModelParams, device: str = "cpu") -> ValueNet:
    # Resolve checkpoint records against this package so loading is portable
    # and cannot select a different copy elsewhere on the workstation.
    parts = str(path).replace("\\", "/").split("/")
    if "checkpoints" in parts:
        local = os.path.join(ROOT, *parts[parts.index("checkpoints"):])
        if os.path.exists(local):
            path = local
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ValueNet(p, width=ck["width"], depth=ck["depth"],
                     use_p=ck["use_p"], out_scale=ck["out_scale"],
                     hard_terminal=ck.get("hard_terminal", True),
                     reference_baseline=ck.get("reference_baseline",
                                               "terminal_only"),
                     reference_quadrature_points=ck.get(
                         "reference_quadrature_points", 24))
    if model.reference_baseline != "terminal_only":
        if "profile_n_hour" not in ck or "profile_c_hour" not in ck:
            raise ValueError("reference-baseline checkpoint lacks profile")
        model.attach_profile(RegimeProfile(ck["profile_n_hour"],
                                           ck["profile_c_hour"], T=p.T))
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    return model
