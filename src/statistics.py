"""Paired statistics: moving-block bootstrap, Wilcoxon, Holm, CVaR,
effect sizes (main.tex §7.10)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as sps


def moving_block_bootstrap_ci(x: np.ndarray, block: int = 7,
                              n_boot: int = 10000, alpha: float = 0.05,
                              seed: int = 6101,
                              stat=np.mean) -> Dict[str, float]:
    """95% percentile CI of stat(x) under a moving-block bootstrap."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        boots[b] = stat(x[idx])
    return {"stat": float(stat(x)),
            "ci_lo": float(np.quantile(boots, alpha / 2)),
            "ci_hi": float(np.quantile(boots, 1 - alpha / 2))}


def cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    x = np.asarray(x, dtype=np.float64)
    var = np.quantile(x, alpha)
    tail = x[x >= var]
    return float(tail.mean()) if len(tail) else float(var)


def cvar_bootstrap_ci(x: np.ndarray, alpha: float = 0.95, block: int = 7,
                      n_boot: int = 10000, seed: int = 6101) -> Dict:
    return moving_block_bootstrap_ci(x, block, n_boot, seed=seed,
                                     stat=lambda z: cvar(z, alpha))


def paired_comparison(diff: np.ndarray, block: int = 7, n_boot: int = 10000,
                      seed: int = 6101) -> Dict[str, float]:
    """diff = method - reference (positive => reference better)."""
    diff = np.asarray(diff, dtype=np.float64)
    out = moving_block_bootstrap_ci(diff, block, n_boot, seed=seed)
    res: Dict[str, float] = {
        "mean_diff": out["stat"], "ci_lo": out["ci_lo"],
        "ci_hi": out["ci_hi"], "median_diff": float(np.median(diff)),
        "n_days": int(len(diff)),
    }
    nz = diff[diff != 0.0]
    if len(nz) >= 10:
        w = sps.wilcoxon(nz, alternative="two-sided")
        res["wilcoxon_p"] = float(w.pvalue)
        # rank-biserial correlation from the signed-rank statistic
        n = len(nz)
        total = n * (n + 1) / 2.0
        res["rank_biserial"] = float(2.0 * w.statistic / total - 1.0)
    else:
        res["wilcoxon_p"] = np.nan
        res["rank_biserial"] = np.nan
    sd = diff.std(ddof=1)
    res["cohens_dz"] = float(diff.mean() / sd) if sd > 0 else np.nan
    res["se_mean"] = float(sd / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    return res


def holm_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values across prespecified comparisons."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (k, pv) in enumerate(items):
        val = min(1.0, (m - i) * pv)
        running = max(running, val)
        adj[k] = running
    for k, v in pvals.items():
        if k not in adj:
            adj[k] = np.nan
    return adj
