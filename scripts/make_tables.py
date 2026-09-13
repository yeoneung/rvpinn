"""Generate Tables 2-6 (tex + csv), the statistics files, and the result
macros with traceability — all exclusively from results_master.parquet."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.latex_export import (MacroWriter, fmt, fmt_ci, fmt_p,  # noqa: E402
                              latex_table, load_master)
from src.statistics import (cvar, cvar_bootstrap_ci,     # noqa: E402
                            holm_correction, paired_comparison)

TAB = os.path.join(ROOT, "paper", "tables")
STATS = os.path.join(ROOT, "results", "statistics")

METHOD_LABEL = {
    "no_storage": "No storage", "self_consumption": "Self-consumption",
    "threshold_rule": "Threshold rule",
    "mpc_deterministic": "Deterministic MPC",
    "mpc_perfect_forecast": "Perfect-forecast MPC (oracle)",
    "sac": "SAC", "td3": "TD3", "direct_hjb": "Direct HJB--PINN",
    "rvpinnpi": "RV--PINN--PI",
    "rvpinnpi_noadapt": "RV--PINN--PI (no adaptive)",
}
ORDER = ["no_storage", "self_consumption", "threshold_rule",
         "mpc_deterministic", "mpc_perfect_forecast", "sac", "td3",
         "direct_hjb", "rvpinnpi"]


def pivot_daily(master: pd.DataFrame, metric: str,
                region="primary", split="test") -> pd.DataFrame:
    df = master[(master["experiment"] == "expB_operation")
                & (master["metric"] == metric)
                & (master["region"] == region)
                & (master["split"] == split)]
    return df.groupby(["method", "date"])["value"].mean().unstack("date")


def seed_daily(master, metric, method, region="primary", split="test"):
    df = master[(master["experiment"] == "expB_operation")
                & (master["metric"] == metric)
                & (master["region"] == region) & (master["split"] == split)
                & (master["method"] == method)]
    return df.groupby(["method_seed", "date"])["value"].mean()


def table3(master, mw, boot_cfg):
    bills = pivot_daily(master, "bill")
    if "rvpinnpi" not in bills.index:
        print("table3: rvpinnpi missing; skipping")
        return
    common = bills.columns[bills.notna().all(axis=0)]
    ref = bills.loc["rvpinnpi", common].values
    stats_rows = {}
    pvals = {}
    for m in bills.index:
        if m == "rvpinnpi":
            continue
        diff = bills.loc[m, common].values - ref
        st = paired_comparison(diff, **boot_cfg)
        stats_rows[m] = st
        if m != "mpc_perfect_forecast":     # oracle excluded from testing
            pvals[m] = st["wilcoxon_p"]
    holm = holm_correction(pvals)
    for m in stats_rows:
        stats_rows[m]["holm_p"] = holm.get(m, np.nan)
    os.makedirs(STATS, exist_ok=True)
    with open(os.path.join(STATS, "table3_paired_stats.json"), "w") as f:
        json.dump(stats_rows, f, indent=2, default=str)

    def col(metric):
        return pivot_daily(master, metric)

    peak = col("peak_import_kw")
    efc = col("efc")
    tdev = col("terminal_soc_dev")
    viol = col("soc_violations")
    lat = col("latency_mean_ms")
    obj = col("objective")

    header = ["Method", "Bill (GBP/day)", "Objective (GBP/day)",
              "Peak (kW)", "EFC", "$|S_T-s_{\\mathrm{tar}}|$",
              "CVaR$_{0.95}$", "Viol.", "Latency (ms)",
              "$\\Delta$Bill 95\\% CI", "Holm $p$"]
    rows_tex = []
    rows_csv = []
    feasible_bills = {}
    for m in ORDER:
        if m not in bills.index:
            continue
        b = bills.loc[m, common].values
        se = b.std(ddof=1) / np.sqrt(len(b))
        cv = cvar(b, 0.95)
        vtot = int(viol.loc[m].sum()) if m in viol.index else 0
        if vtot == 0 and m != "mpc_perfect_forecast":
            feasible_bills[m] = b.mean()
        row = {
            "method": m, "bill_mean": b.mean(), "bill_se": se,
            "objective_mean": obj.loc[m].mean() if m in obj.index else np.nan,
            "peak_mean": peak.loc[m].mean() if m in peak.index else np.nan,
            "efc_mean": efc.loc[m].mean() if m in efc.index else np.nan,
            "cvar95": cv,
            "violations_total": vtot,
            "latency_ms": lat.loc[m].mean() if m in lat.index else np.nan,
            "tdev_mean": tdev.loc[m].mean() if m in tdev.index else np.nan,
        }
        if m in stats_rows:
            row.update({"diff_mean": stats_rows[m]["mean_diff"],
                        "diff_ci_lo": stats_rows[m]["ci_lo"],
                        "diff_ci_hi": stats_rows[m]["ci_hi"],
                        "holm_p": stats_rows[m]["holm_p"]})
        rows_csv.append(row)
    best_bill = min(feasible_bills.values()) if feasible_bills else np.nan
    for row in rows_csv:
        m = row["method"]
        bold = (np.isfinite(best_bill)
                and abs(row["bill_mean"] - best_bill) < 1e-9
                and m in feasible_bills)
        bill_txt = f"{fmt(row['bill_mean'])} $\\pm$ {fmt(row['bill_se'])}"
        if bold:
            bill_txt = f"\\textbf{{{bill_txt}}}"
        ci = (fmt_ci(row["diff_ci_lo"], row["diff_ci_hi"])
              if "diff_ci_lo" in row else "--")
        hp = fmt_p(row.get("holm_p", np.nan)) if "holm_p" in row else "--"
        rows_tex.append([METHOD_LABEL[m], bill_txt,
                         fmt(row["objective_mean"], 1),
                         fmt(row["peak_mean"], 1),
                         fmt(row["efc_mean"], 3), fmt(row["tdev_mean"], 3),
                         fmt(row["cvar95"]),
                         str(row["violations_total"]),
                         fmt(row["latency_ms"], 2), ci, hp])
    tex = latex_table("lcccccccccc", header, rows_tex,
                      "Held-out test results (primary region). Mean $\\pm$ "
                      "standard error over test days; paired moving-block "
                      "bootstrap 95\\% CIs of the daily bill difference "
                      "versus RV--PINN--PI; Holm-adjusted Wilcoxon $p$. "
                      "Perfect-forecast MPC is an information-relaxed "
                      "oracle, not a deployable method. The daily bill "
                      "excludes degradation, state, power-smoothing, and "
                      "terminal penalties; the full objective is the "
                      "quantity optimized in the stochastic-control "
                      "formulation, and the two endpoints should not be "
                      "interpreted interchangeably.",
                      "tab:main_results", resize=True)
    with open(os.path.join(TAB, "table03_main_operational_results.tex"),
              "w") as f:
        f.write(tex)
    pd.DataFrame(rows_csv).to_csv(
        os.path.join(TAB, "table03_main_operational_results.csv"),
        index=False)

    # macros
    rv_bill = bills.loc["rvpinnpi", common].mean()
    for ref_m, macro in (("self_consumption", "MainBillReductionVsRule"),
                         ("mpc_deterministic", "MainBillReductionVsMPC"),
                         ("sac", "MainBillReductionVsSAC")):
        if ref_m in bills.index:
            other = bills.loc[ref_m, common].mean()
            red = 100.0 * (other - rv_bill) / abs(other) if other else np.nan
            mw.add(macro, f"{red:.1f}\\%", "bill",
                   f"expB_operation/bill/{ref_m},rvpinnpi",
                   "mean over common test days, pct difference")
    rv_viol = int(viol.loc["rvpinnpi"].sum()) if "rvpinnpi" in viol.index else -1
    mw.add("MainViolationCount", str(rv_viol), "soc_violations",
           "expB_operation/soc_violations/rvpinnpi", "sum over test days")
    if "self_consumption" in peak.index and "rvpinnpi" in peak.index:
        pk_red = 100.0 * (peak.loc["self_consumption"].mean()
                          - peak.loc["rvpinnpi"].mean()) \
            / peak.loc["self_consumption"].mean()
        mw.add("MainPeakReductionVsRule", f"{pk_red:.1f}\\%",
               "peak_import_kw", "expB_operation/peak_import_kw",
               "mean over test days, pct difference vs self-consumption")


def table2(master, mw):
    df = master[master["experiment"] == "expA_fd_verification"]
    if len(df) == 0:
        print("table2: no expA data")
        return
    final = df[df["policy_iteration"] == -1]
    piv = final.pivot_table(index=["method", "method_seed"],
                            columns="metric", values="value",
                            aggfunc="first")
    rows_tex, rows_csv = [], []
    header = ["Method", "$e_V^\\infty$", "$e_V^2$", "$e_\\pi^2$ (kW)",
              "$e_J$", "$\\widehat R_\\infty^{\\pi}$ (val)",
              "$\\widehat R_\\infty^{*}$", "Time (s)"]
    best = {}
    for m in ("direct_hjb", "rvpinnpi_noadapt", "rvpinnpi"):
        if m not in piv.index.get_level_values(0):
            continue
        sub = piv.loc[m].copy()
        # PI methods: final-checkpoint residuals live in the certification
        # columns (frozen independent design); coalesce so the proposed
        # method's residual columns are never blank (reviewer item R3c)
        for tgt, src in (("val_max_residual", "cert_policy_max"),
                         ("hjb_max_residual", "cert_hjb_max")):
            if src in sub.columns:
                if tgt not in sub.columns:
                    sub[tgt] = np.nan
                sub[tgt] = sub[tgt].fillna(sub[src])
        agg = sub.mean()
        sd = sub.std()
        if not np.isfinite(agg.get("wall_s", np.nan)):
            # PI methods: total training time = sum over policy iterations
            it = df[(df["method"] == m) & (df["policy_iteration"] >= 0)
                    & (df["metric"] == "wall_s")]
            if len(it):
                agg["wall_s"] = float(
                    it.groupby("method_seed")["value"].sum().mean())
        best[m] = agg.get("value_rmse", np.nan)
        rows_csv.append({"method": m, "n_seeds": len(sub),
                         **{f"{c}_mean": agg.get(c, np.nan)
                            for c in ("value_linf", "value_rmse",
                                      "policy_rmse", "cost_gap_eJ",
                                      "val_max_residual", "hjb_max_residual",
                                      "wall_s", "peak_mem_mb")},
                         **{f"{c}_std": sd.get(c, np.nan)
                            for c in ("value_linf", "value_rmse",
                                      "policy_rmse", "cost_gap_eJ")}})
    bmin = min(best.values()) if best else np.nan
    for row in rows_csv:
        m = row["method"]
        def pm(c, d=3):
            v, s = row.get(f"{c}_mean", np.nan), row.get(f"{c}_std", np.nan)
            txt = f"{fmt(v, d)}"
            if np.isfinite(s):
                txt += f" $\\pm$ {fmt(s, d)}"
            return txt
        v_rmse = pm("value_rmse")
        if np.isfinite(bmin) and abs(row["value_rmse_mean"] - bmin) < 1e-12:
            v_rmse = f"\\textbf{{{v_rmse}}}"
        rows_tex.append([METHOD_LABEL.get(m, m), pm("value_linf"), v_rmse,
                         pm("policy_rmse"), pm("cost_gap_eJ"),
                         fmt(row.get("val_max_residual_mean", np.nan), 3),
                         fmt(row.get("hjb_max_residual_mean", np.nan), 3),
                         fmt(row.get("wall_s_mean", np.nan), 0)])
    conv_path = os.path.join(ROOT, "results", "raw", "expA",
                             "fd_convergence.json")
    caption = ("Finite-difference verification (2-state problem, five "
               "seeds, mean $\\pm$ std). Value errors against the finest "
               "grid; $e_J$ from the FD linear policy-evaluation solve. "
               "The reported residual quantities are maxima over a frozen "
               "independent validation design; they are empirical "
               "finite-sample estimates and are not labeled global "
               "certificates unless accompanied by a valid "
               "fill-distance/Lipschitz correction.")
    if os.path.exists(conv_path):
        conv = json.load(open(conv_path))
        for c in conv:
            rows_tex.append([f"FD {c['grid']} (vs finest)",
                             fmt(c["linf_vs_finest"], 3),
                             fmt(c["rmse_vs_finest"], 3),
                             "--", "--", "--", "--", "--"])
    tex = latex_table("lccccccc", header, rows_tex, caption,
                      "tab:fd_verification", resize=True)
    with open(os.path.join(TAB, "table02_fd_verification.tex"), "w") as f:
        f.write(tex)
    pd.DataFrame(rows_csv).to_csv(
        os.path.join(TAB, "table02_fd_verification.csv"), index=False)
    if rows_csv:
        rv = [r for r in rows_csv if r["method"] == "rvpinnpi"]
        dh = [r for r in rows_csv if r["method"] == "direct_hjb"]
        if rv and dh:
            mw.add("FDValueErrorPI", fmt(rv[0]["value_rmse_mean"], 3),
                   "value_rmse", "expA/rvpinnpi", "mean over seeds")
            mw.add("FDValueErrorDirect", fmt(dh[0]["value_rmse_mean"], 3),
                   "value_rmse", "expA/direct_hjb", "mean over seeds")


def table4(master, mw):
    df = master[master["experiment"] == "expC_robustness"]
    if len(df) == 0:
        print("table4: no expC data")
        return
    piv = df[df["metric"].isin(["bill", "objective", "soc_violations",
                                "peak_import_kw", "max_violation"])]
    g = piv.groupby(["scenario", "scenario_value", "method", "metric"])[
        "value"].agg(["mean", "count"]).reset_index()
    g.to_csv(os.path.join(TAB, "table04_robustness_results.csv"),
             index=False)
    rows_tex = []
    header = ["Scenario", "Value", "Method", "Mean bill", "Violations"]
    for (scen, sval), sub in g.groupby(["scenario", "scenario_value"]):
        for m in sub["method"].unique():
            ss = sub[sub["method"] == m]
            bill = ss[ss["metric"] == "bill"]["mean"]
            vio = ss[ss["metric"] == "soc_violations"]["mean"]
            rows_tex.append([str(scen).replace("_", "\\_"), fmt(sval, 2),
                             METHOD_LABEL.get(m, m).replace("_", "\\_"),
                             fmt(bill.iloc[0]) if len(bill) else "--",
                             fmt(vio.iloc[0] if len(vio) else 0, 2)])
    tex = latex_table("lllcc", header, rows_tex[:48],
                      "Robustness scenarios (mean over test days; "
                      "'known'/'nominal' denote the safety-parameter mode). "
                      "Full matrix in the CSV companion and Fig.~7/S5.",
                      "tab:robustness")
    with open(os.path.join(TAB, "table04_robustness_results.tex"), "w") as f:
        f.write(tex)
    kn = df[(df["scenario"].str.contains("capacity_pct_known", na=False))
            & (df["metric"] == "soc_violations")
            & (df["method"] == "rvpinnpi")]
    mw.add("RobustKnownViolations",
           fmt(kn["value"].sum() if len(kn) else np.nan, 0),
           "soc_violations", "expC/capacity_known/rvpinnpi", "sum")


def table5(master, mw):
    df = master[master["experiment"] == "expD_ablations"]
    if len(df) == 0:
        print("table5: no expD data")
        return
    keep = df[df["metric"].isin(
        ["value_rmse", "policy_rmse", "cost_gap_eJ", "bill", "efc",
         "rollout_cost", "soc_violations", "e_T", "delta_greedy_max",
         "action_rmse", "throughput_kwh"])]
    g = keep.groupby(["scenario", "method", "metric"])["value"].agg(
        ["mean", "std", "count"]).reset_index()
    g.to_csv(os.path.join(TAB, "table05_ablation_results.csv"), index=False)
    rows_tex = []
    header = ["Ablation", "Variant", "Metric", "Mean", "Std", "$n$"]
    for _, r in g.iterrows():
        rows_tex.append([str(r["scenario"]).split(":")[0].replace("_", "\\_"),
                         str(r["scenario"]).split(":")[-1].replace("_", "\\_"),
                         str(r["metric"]).replace("_", "\\_"),
                         fmt(r["mean"], 3), fmt(r["std"], 3),
                         str(int(r["count"]))])
    tex = latex_table("lllccc", header, rows_tex[:70],
                      "Ablation summary (Experiment D). Full detail in the "
                      "CSV companion and Fig.~S6.", "tab:ablations")
    with open(os.path.join(TAB, "table05_ablation_results.tex"), "w") as f:
        f.write(tex)


def table6(master, mw):
    rows_csv = []
    df = master[(master["experiment"] == "expA_fd_verification")
                & (master["metric"].isin(["wall_s", "peak_mem_mb"]))]
    for m, sub in df.groupby("method"):
        w = sub[sub["metric"] == "wall_s"].groupby("method_seed")[
            "value"].sum()
        mem = sub[sub["metric"] == "peak_mem_mb"]["value"].max()
        rows_csv.append({"stage": "expA", "method": m,
                         "train_wall_s_mean": w.mean(),
                         "peak_mem_mb": mem})
    lat = master[(master["experiment"] == "expB_operation")
                 & (master["metric"] == "latency_mean_ms")]
    for m, sub in lat.groupby("method"):
        rows_csv.append({"stage": "expB", "method": m,
                         "latency_ms_mean": sub["value"].mean()})
    env = {}
    man = os.path.join(ROOT, "results", "immutable", "run_manifest.json")
    if os.path.exists(man):
        env = json.load(open(man)).get("environment", {})
    rows_csv.append({"stage": "hardware",
                     "method": env.get("gpu", "unknown"),
                     "notes": f"torch {env.get('torch','')}, "
                              f"CUDA {env.get('cuda','')}, "
                              f"RAM {env.get('ram_gb','')} GB"})
    dfout = pd.DataFrame(rows_csv)
    dfout.to_csv(os.path.join(TAB, "table06_compute_costs.csv"), index=False)
    rows_tex = []
    for _, r in dfout.iterrows():
        rows_tex.append([str(r.get("stage", "")),
                         str(r.get("method", "")).replace("_", "\\_"),
                         fmt(r.get("train_wall_s_mean", np.nan), 0),
                         fmt(r.get("peak_mem_mb", np.nan), 0),
                         fmt(r.get("latency_ms_mean", np.nan), 2)])
    tex = latex_table("llccc",
                      ["Stage", "Method/HW", "Train (s)", "Mem (MB)",
                       "Latency (ms)"],
                      rows_tex,
                      "Training and inference compute. Hardware: single "
                      "workstation GPU; see reproducibility report for the "
                      "full environment.", "tab:compute")
    with open(os.path.join(TAB, "table06_compute_costs.tex"), "w") as f:
        f.write(tex)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-new-results", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(TAB, exist_ok=True)
    master = load_master(args.allow_new_results)
    mw = MacroWriter()
    boot_cfg = {"n_boot": args.bootstrap, "block": 7}
    table2(master, mw)
    table3(master, mw, boot_cfg)
    table4(master, mw)
    table5(master, mw)
    table6(master, mw)
    gen = os.path.join(ROOT, "paper", "generated")
    mw.write(os.path.join(gen, "results_macros.tex"),
             os.path.join(gen, "result_traceability.csv"))
    print("tables + macros written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
