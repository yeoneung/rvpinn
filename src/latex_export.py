"""Master-results IO with hash guard, LaTeX helpers, macros, traceability."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ROOT
from .reproducibility import file_sha256

IMM = os.path.join(ROOT, "results", "immutable")


def load_master(allow_new: bool = False) -> pd.DataFrame:
    path = os.path.join(IMM, "results_master.parquet")
    man_path = os.path.join(IMM, "run_manifest.json")
    with open(man_path) as f:
        man = json.load(f)
    sha = file_sha256(path)
    if sha != man["results_master_sha256"] and not allow_new:
        raise RuntimeError(
            "results_master.parquet hash mismatch with run_manifest.json; "
            "rerun consolidate_results.py or pass --allow-new-results")
    return pd.read_parquet(path)


def fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    ax = abs(x)
    if ax != 0 and (ax < 10 ** (-digits) or ax >= 10 ** 6):
        return f"{x:.{digits}e}"
    return f"{x:,.{digits}f}"


def fmt_ci(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{fmt(lo, digits)}, {fmt(hi, digits)}]"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.4f}"


def latex_table(colspec: str, header: List[str], rows: List[List[str]],
                caption: str, label: str, resize: bool = False) -> str:
    """`resize=True` scales the tabular to \\linewidth so wide tables never
    overflow the (narrower) MDPI text block."""
    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small"]
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines += [
        f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if resize:
        lines.append("}")
    lines += ["\\end{table}", ""]
    return "\n".join(lines)


class MacroWriter:
    def __init__(self):
        self.macros: List[str] = []
        self.trace: List[Dict] = []

    def add(self, name: str, value: str, metric: str, source_rows: str,
            aggregation: str, config_hash: str = ""):
        self.macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
        self.trace.append({"macro": name, "value": value,
                           "source_metric": metric,
                           "source_rows": source_rows,
                           "aggregation": aggregation,
                           "script": "make_tables.py/make_figures.py",
                           "config_hash": config_hash})

    def write(self, macro_path: str, trace_path: str):
        os.makedirs(os.path.dirname(macro_path), exist_ok=True)
        with open(macro_path, "w", encoding="utf-8") as f:
            f.write("% auto-generated result macros — do not edit by hand\n")
            f.write("\n".join(self.macros) + "\n")
        pd.DataFrame(self.trace).to_csv(trace_path, index=False)
