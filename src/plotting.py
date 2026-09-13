"""Publication figure style and helpers."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from .config import ROOT                  # noqa: E402

FIG = os.path.join(ROOT, "paper", "figures")
FIGDATA = os.path.join(ROOT, "paper", "figure_data")

# Okabe-Ito colorblind-safe palette; fixed method encoding across figures
METHOD_STYLE = {
    "rvpinnpi":            dict(color="#D55E00", ls="-",  marker="o",
                                label="RV-PINN-PI"),
    "rvpinnpi_noadapt":    dict(color="#E69F00", ls="--", marker="s",
                                label="RV-PINN-PI (uniform)"),
    "direct_hjb":          dict(color="#0072B2", ls="-.", marker="^",
                                label="Direct HJB-PINN"),
    "mpc_deterministic":   dict(color="#009E73", ls="-",  marker="v",
                                label="Deterministic MPC"),
    "mpc_perfect_forecast": dict(color="#009E73", ls=":", marker="x",
                                 label="Perfect-forecast MPC"),
    "sac":                 dict(color="#CC79A7", ls="-",  marker="D",
                                label="SAC"),
    "td3":                 dict(color="#882255", ls="--", marker="P",
                                label="TD3"),
    "self_consumption":    dict(color="#56B4E9", ls="-",  marker="<",
                                label="Self-consumption"),
    "threshold_rule":      dict(color="#F0E442", ls="--", marker=">",
                                label="Threshold rule"),
    "no_storage":          dict(color="#999999", ls=":",  marker=".",
                                label="No storage"),
    "fd":                  dict(color="#000000", ls="-",  marker="",
                                label="Finite differences"),
    "zero_action":         dict(color="#333333", ls="-",  marker="",
                                label="zero action"),
}


def set_style():
    plt.rcParams.update({
        # third revision (Reviewer 1, Comment 3): one uniform type scale --
        # titles 8.5, axis labels 8.5, ticks 7.5 (category ticks 6.5),
        # legends 7, panel labels 9 bold
        "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "figure.dpi": 120, "savefig.dpi": 600,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "lines.linewidth": 1.3, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "font.family": "DejaVu Sans",
        "figure.constrained_layout.use": True,
        # extra inter-panel padding so the below-axis panel labels
        # (journal style, first revision) never collide with the next row
        "figure.constrained_layout.h_pad": 0.30,
    })


def panel_label(ax, letter: str, dy: float = -30.0):
    """Panel label centered BELOW the subplot (outside the plotting
    area), per the journal's multipanel-figure convention. `dy` (points)
    moves the label further down for axes whose rotated tick labels
    extend below the default offset (third revision, Reviewer 1)."""
    ax.annotate(f"({letter})", xy=(0.5, 0.0), xycoords="axes fraction",
                xytext=(0.0, dy), textcoords="offset points",
                ha="center", va="top", fontsize=9,
                fontweight="bold", annotation_clip=False)


def save_figure(fig, name: str, data=None):
    """Save PDF + 600 dpi PNG + figure-source data."""
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(FIGDATA, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"),
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600,
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    if data is not None:
        if isinstance(data, pd.DataFrame):
            data.to_parquet(os.path.join(FIGDATA, name + ".parquet"))
        else:
            with open(os.path.join(FIGDATA, name + ".json"), "w") as f:
                json.dump(data, f, indent=2, default=str)
    print(f"  saved {name}")


def qa_check() -> List[str]:
    """Automated figure QA: existence, PNG size, no zero-byte files."""
    problems = []
    for f in sorted(os.listdir(FIG)):
        p = os.path.join(FIG, f)
        if f.startswith("_") or not f.lower().endswith((".pdf", ".png")):
            continue
        if os.path.getsize(p) < 2000:
            problems.append(f"{f}: suspiciously small ({os.path.getsize(p)}B)")
    mains = [f"fig{i:02d}" for i in range(1, 9)]
    for m in mains:
        pdfs = [f for f in os.listdir(FIG)
                if f.startswith(m) and f.endswith(".pdf")]
        pngs = [f for f in os.listdir(FIG)
                if f.startswith(m) and f.endswith(".png")]
        if not pdfs:
            problems.append(f"missing PDF for {m}")
        if not pngs:
            problems.append(f"missing PNG for {m}")
    return problems
