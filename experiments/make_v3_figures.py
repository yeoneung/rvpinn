"""Create the cost--computation and mechanism figures for version 3."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
FIGURES = ROOT / "manuscript" / "figures"


def make_factorial():
    data = pd.read_csv(RESULTS / "sensitivity_cells.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    widths = (5, 10, 20)
    thresholds = (250, 300, 350)
    fees = (20, 40, 80)
    values = data["learned_minus_deterministic"].to_numpy(float)
    vmax = max(float(np.max(np.abs(values))), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.2), sharey=True,
                             constrained_layout=True)
    image = None
    for ax, width in zip(axes, widths):
        sub = data[data["smoothing_width_kw"] == width]
        matrix = np.full((len(fees), len(thresholds)), np.nan)
        for iy, fee in enumerate(fees):
            for ix, threshold in enumerate(thresholds):
                row = sub[(sub["fee_per_hour"] == fee)
                          & (sub["threshold_kw"] == threshold)]
                matrix[iy, ix] = row["learned_minus_deterministic"].iloc[0]
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        for iy in range(len(fees)):
            for ix in range(len(thresholds)):
                value = matrix[iy, ix]
                color = "white" if abs(value) > 0.55 * vmax else "black"
                ax.text(ix, iy, f"{value:.0f}", ha="center", va="center",
                        color=color, fontsize=14)
        ax.set_title(f"Training width {width} kW", fontsize=13)
        ax.set_xticks(range(len(thresholds)), thresholds)
        ax.set_xlabel("Threshold (kW)", fontsize=13)
        ax.set_yticks(range(len(fees)), fees)
        ax.tick_params(labelsize=12)
    axes[0].set_ylabel("Fee (EUR/h)", fontsize=13)
    cbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("Cost difference (EUR/day)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_factorial.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def make_cost_latency():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fees = (20, 40, 80)
    summary = pd.read_csv(RESULTS / "confirmatory_method_summary.csv")
    latencies = summary['mean_latency_mean_ms'].to_numpy(float)
    if not np.isfinite(latencies).all() or not (latencies > 0.).all():
        raise ValueError('logarithmic cost plot requires finite positive latencies')
    method_order = (
        "no_storage", "validation_tuned_price_rule", "convex_envelope_mpc",
        "deterministic_exact_band_miqp",
        "stochastic_two_stage_exact_band_miqp", "offline_value_policy")
    labels = {
        "no_storage": "No storage",
        "validation_tuned_price_rule": "Price rule",
        "convex_envelope_mpc": "Envelope MPC",
        "deterministic_exact_band_miqp": "CM MIQP",
        "stochastic_two_stage_exact_band_miqp": "Stochastic MIQP",
        "offline_value_policy": "Offline procedure",
    }
    colors = plt.get_cmap("tab10")(np.arange(len(method_order)))
    markers = ("o", "s", "^", "D", "P", "X")
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.2), sharey=True,
                             sharex=True, constrained_layout=True)
    handles = []
    for ax, fee in zip(axes, fees):
        sub = summary[summary["fee"] == fee].set_index("method")
        baseline = float(sub.loc["no_storage", "mean_common_cost"])
        for idx, method in enumerate(method_order):
            x = float(sub.loc[method, "mean_latency_mean_ms"])
            y = 100.0 * (float(sub.loc[method, "mean_common_cost"])
                         / baseline - 1.0)
            point = ax.scatter(x, y, s=36, color=colors[idx],
                               marker=markers[idx], edgecolor="black",
                               linewidth=0.35, label=labels[method], zorder=3)
            if fee == fees[0]:
                handles.append(point)
        ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(f"Fee {fee} EUR/h", fontsize=13)
        ax.set_xlabel("Mean call latency (ms)\n(log scale)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    axes[0].set_ylabel("Cost vs no storage (%)", fontsize=13)
    fig.legend(handles=handles, labels=[labels[x] for x in method_order],
               loc="outside lower center", ncol=3, frameon=False, fontsize=12)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_cost_latency.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(FIGURES / "fig_cost_latency.pdf")


def main() -> int:
    make_factorial()
    make_cost_latency()
    print(FIGURES / "fig_factorial.pdf")
    print(FIGURES / "fig_cost_latency.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
