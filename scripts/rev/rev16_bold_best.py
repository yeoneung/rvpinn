"""Third revision (Reviewer 1, Comment 2): mark the best result per
performance metric in Tables 2-5 in bold and append a blue caption
sentence stating the convention. Pure text transformation of the
archived table fragments; no number is changed. Idempotent: existing
\textbf{} wrappers in data cells are stripped before re-marking.

Usage: python scripts/rev/rev16_bold_best.py <manuscript/tables dir>
"""
from __future__ import annotations

import os
import re
import sys

BOLD = "\\textbf{%s}"
PROVIDE = "\\providecommand{\\revthree}[1]{#1}\n"


def num(cell: str):
    """Leading numeric value of a cell like '1,156.495 $\\pm$ 548.783'."""
    m = re.match(r"\s*([0-9,]*\.?[0-9]+(?:e-?[0-9]+)?)", cell.strip())
    return float(m.group(1).replace(",", "")) if m else None


def split_rows(body: str):
    return [r for r in body.split("\\\\") if r.strip()]


def strip_bold(s: str) -> str:
    return re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)


def load(path):
    return open(path, encoding="utf-8").read()


def save(path, tex):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(tex)
    print("updated", os.path.basename(path))


def add_caption_note(tex: str, note: str) -> str:
    if "\\revthree{" in tex:          # already annotated
        return tex
    out, n = re.subn(r"(\})(\s*\n\\label\{)",
                     " \\\\revthree{" + note + "}\\1\\2", tex, count=1)
    assert n == 1
    if "\\providecommand{\\revthree}" not in out:
        out = PROVIDE + out
    return out


def bold_cells(rows, decide):
    """decide(rows) -> set of (row_index, col_index) to bold."""
    winners = decide(rows)
    out = []
    for i, row in enumerate(rows):
        cells = row.split("&")
        for j in range(len(cells)):
            if (i, j) in winners:
                lead = len(cells[j]) - len(cells[j].lstrip())
                tail = len(cells[j]) - len(cells[j].rstrip())
                core = cells[j].strip()
                cells[j] = (cells[j][:lead] + BOLD % core
                            + (cells[j][len(cells[j]) - tail:] if tail else ""))
        out.append("&".join(cells))
    return out


def apply(path, decide, note):
    tex = load(path)
    m = re.search(r"(\\midrule\n)(.*?)(\n?\\bottomrule)", tex, re.S)
    body = strip_bold(m.group(2))
    rows = split_rows(body)
    new_body = " \\\\\n".join(bold_cells(rows, decide)) + " \\\\"
    tex = tex[:m.start()] + m.group(1) + new_body + m.group(3) + tex[m.end():]
    tex = add_caption_note(tex, note)
    save(path, tex)


def argmin_set(vals):
    """Indices attaining the minimum among not-None vals (ties all win)."""
    ok = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(ok) < 2:
        return set()
    lo = min(v for _, v in ok)
    return {i for i, v in ok if v == lo}


# ---- Table 2: best among the three neural methods per column ----------
def decide_t2(rows):
    neural = [i for i, r in enumerate(rows) if "PINN" in r]
    win = set()
    ncol = len(rows[0].split("&"))
    for j in range(1, ncol):
        vals = [num(rows[i].split("&")[j]) if i in neural else None
                for i in range(len(rows))]
        win |= {(i, j) for i in argmin_set(vals)}
    return win


# ---- Table 3: best per metric column (1..8); skip all-equal columns ---
def decide_t3(rows):
    win = set()
    for j in range(1, 9):                    # Bill..Latency
        vals = [num(r.split("&")[j]) for r in rows]
        s = argmin_set(vals)
        if len({v for v in vals if v is not None}) > 1:
            win |= {(i, j) for i in s}
    return win


# ---- Table 4: lowest mean bill within each (scenario, value) group ----
def decide_t4(rows):
    groups = {}
    for i, r in enumerate(rows):
        c = r.split("&")
        groups.setdefault((c[0].strip(), c[1].strip()), []).append(i)
    win = set()
    for idx in groups.values():
        vals = [num(rows[i].split("&")[3]) for i in idx]
        win |= {(idx[k], 3) for k in argmin_set(vals)}
    return win


# ---- Table 5: best variant per (ablation, error metric) ---------------
T5_METRICS = {"cost\\_gap\\_eJ", "policy\\_rmse", "value\\_rmse", "e\\_T"}


def decide_t5(rows):
    groups = {}
    for i, r in enumerate(rows):
        c = r.split("&")
        met = c[2].strip()
        if met in T5_METRICS:
            groups.setdefault((c[0].strip(), met), []).append(i)
    win = set()
    for idx in groups.values():
        if len(idx) < 2:
            continue
        vals = [num(rows[i].split("&")[3]) for i in idx]
        win |= {(idx[k], 3) for k in argmin_set(vals)}
    return win


def main():
    d = sys.argv[1]
    apply(os.path.join(d, "table02_fd_verification.tex"), decide_t2,
          "The best value among the neural methods in each column is "
          "shown in bold (the FD rows are grid-convergence references).")
    apply(os.path.join(d, "table03_main_operational_results.tex"), decide_t3,
          "The best value in each metric column is shown in bold; the "
          "violation counts are uniformly zero, and the lowest bill is "
          "attained by a battery-depleting rule (see the caveat above).")
    apply(os.path.join(d, "table04_robustness_results.tex"), decide_t4,
          "The lowest mean bill within each scenario--value group is "
          "shown in bold.")
    apply(os.path.join(d, "table05_ablation_results.tex"), decide_t5,
          "Within each ablation, the best variant for each error metric "
          "is shown in bold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
