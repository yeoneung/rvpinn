"""Revision analysis for Reviewer 1, Comment 1.7 (and Reviewer 2.5).

(a) Adequacy of the reflected-OU disturbance model: histograms, Q-Q
    quantiles, moments, and tail rates of the standardized AR(1)
    innovations of the hourly net-load and price forecast errors,
    compared with the N(0,1) innovations the OU model assumes.
    Computed separately on the training and test periods, pooled over
    the eight regimes (per-regime numbers also stored).

(b) Historical-error resampling stress test: synthetic test days are
    built by 6-hour block resampling of the REAL forecast-error paths
    (jointly for load and price, within regime), which preserves the
    marginals and short-range dependence of the historical errors while
    breaking the within-day dependence structure assumed by the OU
    model (block joins insert abrupt changes). Every learned/rule
    method is evaluated on the same synthetic days; MPC on a stratified
    subset. Comparison values on the ORIGINAL days come from the same
    evaluation engine (results_master).

Output: results/rev/ou_adequacy.json, figS09_ou_adequacy.pdf/.png,
        resampling_daily.parquet, resampling_summary.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
from scripts.rev.rev_common import ROOT, OUT, remap, save_json  # noqa: E402

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import torch                                              # noqa: E402
from scipy import stats as sps                            # noqa: E402

from src.evaluation import (NeuralController,             # noqa: E402
                            evaluate_method, load_region_days)
from src.exp_common import (ckpt_dir, load_stack,         # noqa: E402
                            pick_regimes, pick_seeds, regime_bundle)
from src.mpc import MPCController                         # noqa: E402
from src.policy_iteration import load_checkpoint          # noqa: E402
from src.regime_profiles import load_calibration          # noqa: E402
from src.rules import NoStorage, PriceThreshold, SelfConsumption  # noqa: E402

torch.set_default_dtype(torch.float64)
BLOCK_H = 6
RNG_SEED = 20250825
MPC_DAYS_PER_REGIME = 6


def day_residuals(day, prof):
    """Hourly forecast-error paths (y, p) of one realized day."""
    hrs = np.arange(24, dtype=float)
    y = day["N"] - prof.n_bar(hrs)
    pz = day["C"] - prof.c_bar(hrs)
    return y, pz


# ---------------------------------------------------------------- (a) --
def adequacy() -> dict:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    regimes = pick_regimes(cfg, "primary")
    params, profs, _ = regime_bundle("primary", regimes, cfg)
    cal = load_calibration()
    out = {"per_regime": {}, "pooled": {}}
    pooled = {part: {"uy": [], "up": []} for part in ("train", "test")}
    for part in ("train", "test"):
        days = [d for d in load_region_days("primary", part, cal)
                if d["regime"] in params]
        for reg in regimes:
            p = params[reg]
            prof = profs[reg]
            phi_y = np.exp(-p.kappa_y * 1.0)
            phi_p = np.exp(-p.kappa_p * 1.0)
            q_y = p.sigma_y ** 2 * (1 - phi_y ** 2) / (2 * p.kappa_y)
            q_p = p.sigma_p ** 2 * (1 - phi_p ** 2) / (2 * p.kappa_p)
            uy, up = [], []
            for d in days:
                if d["regime"] != reg:
                    continue
                y, pz = day_residuals(d, prof)
                uy.append((y[1:] - phi_y * y[:-1]) / np.sqrt(q_y))
                up.append((pz[1:] - phi_p * pz[:-1]) / np.sqrt(q_p))
            if not uy:
                continue
            uy = np.concatenate(uy)
            up = np.concatenate(up)
            pooled[part]["uy"].append(uy)
            pooled[part]["up"].append(up)
            key = f"{part}:{reg}"
            out["per_regime"][key] = {
                "n": int(uy.size),
                "y": _moments(uy), "p": _moments(up)}
    for part in ("train", "test"):
        uy = np.concatenate(pooled[part]["uy"])
        up = np.concatenate(pooled[part]["up"])
        out["pooled"][part] = {"n": int(uy.size),
                               "y": _moments(uy), "p": _moments(up)}
    _figure(pooled)
    return out


def _moments(u: np.ndarray) -> dict:
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ks = sps.kstest(u, "norm")
    return {"mean": float(u.mean()), "std": float(u.std(ddof=1)),
            "skew": float(sps.skew(u)),
            "excess_kurtosis": float(sps.kurtosis(u)),
            "quantiles": {str(q): float(np.quantile(u, q)) for q in qs},
            "normal_quantiles": {str(q): float(sps.norm.ppf(q))
                                 for q in qs},
            "P_abs_gt2": float(np.mean(np.abs(u) > 2)),
            "P_abs_gt3": float(np.mean(np.abs(u) > 3)),
            "N01_P_abs_gt2": 0.0455, "N01_P_abs_gt3": 0.0027,
            "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue)}


def _figure(pooled) -> None:
    # Third revision (Reviewer 1, Comment 3): drawn with the shared
    # publication style (src/plotting.py) so figure width, fonts, colors,
    # and panel labels match every other figure exactly.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import panel_label, set_style
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    grid = np.linspace(-5, 5, 400)
    for col, key, label in ((0, "uy", "net-load innovation"),
                            (1, "up", "price innovation")):
        u = np.concatenate(pooled["test"][key])
        ax = axes[0][col]
        ax.hist(u, bins=80, density=True, alpha=0.65, color="#0072B2",
                label="test data")
        ax.plot(grid, sps.norm.pdf(grid), "k--", lw=1.0, label="N(0,1)")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlabel(f"standardized {label}")
        ax.set_ylabel("density (log)")
        ax.legend()
        panel_label(ax, chr(97 + col))
        ax = axes[1][col]
        qq = np.linspace(0.001, 0.999, 199)
        ax.plot(sps.norm.ppf(qq), np.quantile(u, qq), ".", ms=3,
                color="#0072B2")
        lim = [-4.5, 4.5]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_xlabel("N(0,1) quantile")
        ax.set_ylabel(f"empirical quantile ({label})")
        panel_label(ax, chr(99 + col))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figS09_ou_adequacy.{ext}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("figS09 saved", flush=True)


# ---------------------------------------------------------------- (b) --
def make_bootstrap_days(days_by_regime, profs, rng):
    """6-hour block resampling of (y, p) residual paths within regime."""
    synth = []
    for reg, days in days_by_regime.items():
        prof = profs[reg]
        res = [day_residuals(d, prof) for d in days]
        Y = np.stack([r[0] for r in res])       # (n_days, 24)
        P = np.stack([r[1] for r in res])
        n = len(days)
        hrs = np.arange(24, dtype=float)
        nb = prof.n_bar(hrs)
        cb = prof.c_bar(hrs)
        for i, d in enumerate(days):
            y_new = np.empty(24)
            p_new = np.empty(24)
            for b in range(0, 24, BLOCK_H):
                j = rng.integers(0, n)
                y_new[b:b + BLOCK_H] = Y[j, b:b + BLOCK_H]
                p_new[b:b + BLOCK_H] = P[j, b:b + BLOCK_H]
            N_new = nb + y_new
            C_new = cb + p_new
            synth.append({"date": d["date"], "regime": reg,
                          "N": N_new, "C": C_new,
                          "L": d["L"], "R": d["R"]})
    return synth


def resampling() -> None:
    cfg = load_stack(os.path.join(ROOT, "configs", "paper_full.yaml"))
    seeds_m = pick_seeds(cfg)
    regimes = pick_regimes(cfg, "primary")
    params, profs, _ = regime_bundle("primary", regimes, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal = load_calibration()
    days = [d for d in load_region_days("primary", "test", cal)
            if d["regime"] in params]
    days_by_regime = {r: [d for d in days if d["regime"] == r]
                      for r in regimes}
    rng = np.random.default_rng(RNG_SEED)
    synth = make_bootstrap_days(days_by_regime, profs, rng)
    print(f"{len(synth)} bootstrap days", flush=True)

    tuning = json.load(open(os.path.join(ROOT, "results", "raw", "expB",
                                         "threshold_tuning.json")))
    rows = []

    def record(res, method, seed):
        for r in res:
            rows.append({"method": method, "seed": seed,
                         "date": r["date"], "regime": r["regime"],
                         "bill": r["bill"], "objective": r["objective"],
                         "peak_import_kw": r["peak_import_kw"],
                         "soc_violations": r["soc_violations"],
                         "projection_events": r["projection_events"]})

    record(evaluate_method(lambda reg: NoStorage(), synth, params, profs),
           "no_storage", -1)
    record(evaluate_method(lambda reg: SelfConsumption(), synth, params,
                           profs), "self_consumption", -1)
    record(evaluate_method(
        lambda reg: PriceThreshold(params[reg], tuning["c_lo"],
                                   tuning["c_hi"]),
        synth, params, profs), "threshold_rule", -1)
    print("rules done", flush=True)

    for ms in seeds_m:
        def fac(reg, _ms=ms):
            res = json.load(open(os.path.join(
                ckpt_dir("pinn_pi", "primary", reg, f"seed{_ms}"),
                "result.json")))
            model = load_checkpoint(remap(res["selected_checkpoint"]),
                                    params[reg], device)
            return NeuralController(model, params[reg], profs[reg])
        record(evaluate_method(fac, synth, params, profs), "rvpinnpi", ms)
        print(f"rvpinnpi seed{ms} done", flush=True)

    # MPC on a stratified subset
    sub = []
    for reg in regimes:
        sub += [d for d in synth if d["regime"] == reg][:MPC_DAYS_PER_REGIME]
    record(evaluate_method(lambda reg: MPCController(params[reg]), sub,
                           params, profs), "mpc_deterministic", -1)
    print("mpc subset done", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "resampling_daily.parquet"))

    # original-day comparison from the immutable results
    master = pd.read_parquet(os.path.join(ROOT, "results", "immutable",
                                          "results_master.parquet"))
    b = master[(master.experiment == "expB_operation") & master.is_primary
               & (master.metric == "bill")]
    orig = b.groupby("method")["value"].mean().to_dict()
    mpc_dates = set(d["date"] for d in sub)
    orig_mpc_sub = float(b[(b.method == "mpc_deterministic")
                           & (b.date.isin(mpc_dates))]["value"].mean())

    summary = {}
    for meth, g in df.groupby("method"):
        entry = {"boot_mean_bill": float(g.bill.mean()),
                 "boot_total_violations": int(g.soc_violations.sum()),
                 "n_days": int(g.date.nunique())}
        if meth == "mpc_deterministic":
            entry["orig_mean_bill_same_days"] = orig_mpc_sub
        elif meth in orig:
            entry["orig_mean_bill"] = float(orig[meth])
        summary[meth] = entry
    save_json("resampling_summary.json", summary)
    print("resampling stress test complete", flush=True)


def main() -> int:
    save_json("ou_adequacy.json", adequacy())
    resampling()
    return 0


if __name__ == "__main__":
    sys.exit(main())
