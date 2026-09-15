"""Calibrate regime profiles, OU parameters, domain bounds, and objective
coefficients from the TRAINING period only (main.tex §5.5, §6.4-6.7).

Outputs data/calibrated_parameters.json (per region) with:
  - microgrid scaling stats, B0, Qpk0
  - lambda_pk, lambda_1, lambda_2, lambda_T, eps_g, eps_a
  - per regime: hourly n_bar/c_bar profiles, OU parameters, domain bounds
  - out-of-domain frequencies per split
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config                               # noqa: E402
from src.data_preprocess import assign_regime, scale_microgrid   # noqa: E402
from src.ou_calibration import domain_bounds, fit_pair_from_pairs  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "data", "splits")

COUNTRY_HOLIDAYS = {"GB": "GB", "IT": "IT", "DE": "DE", "AT": "AT",
                    "FR": "FR", "ES": "ES", "NL": "NL", "BE": "BE",
                    "DK": "DK", "SE": "SE", "NO": "NO", "PL": "PL",
                    "CH": "CH", "CZ": "CZ", "HU": "HU", "PT": "PT"}


def holiday_dates(country: str, years) -> set:
    try:
        import holidays as hol
        code = COUNTRY_HOLIDAYS.get(country)
        if code is None:
            return set()
        return set(hol.country_holidays(code, years=list(years)).keys())
    except Exception:
        return set()


def within_day_pairs(values: pd.Series):
    """Consecutive (z_k, z_{k+1}) pairs restricted to the same day."""
    z0_list, z1_list = [], []
    for _, day in values.groupby(values.index.normalize()):
        v = day.values
        if len(v) >= 2:
            z0_list.append(v[:-1])
            z1_list.append(v[1:])
    return np.concatenate(z0_list), np.concatenate(z1_list)


def calibrate_region(role: str, cfg: dict) -> dict:
    with open(os.path.join(SPLITS, f"{role}_split.json")) as f:
        meta = json.load(f)
    zone, tz, split = meta["zone"], meta["tz"], meta["split"]
    df = pd.read_parquet(os.path.join(PROC, f"{role}_{zone}.parquet"))
    df.index = pd.DatetimeIndex(df.index).tz_convert(tz)

    mg = cfg["microgrid"]
    P_base = float(mg["P_base_kw"])
    gamma_R = float(mg["gamma_R"])
    T = float(cfg["horizon"]["T_hours"])
    alpha_s = float(cfg["cost"]["alpha_s"])
    a_max = max(float(cfg["battery"]["a_c_max_kw"]),
                float(cfg["battery"]["a_d_max_kw"]))

    # chronological masks on local time (split dates are date strings)
    ts_naive = df.index.tz_localize(None)
    m_train = (ts_naive >= split["train"][0]) & (ts_naive < split["train"][1])
    m_val = (ts_naive >= split["val"][0]) & (ts_naive < split["val"][1])
    m_test = (ts_naive >= split["test"][0]) & (ts_naive < split["test"][1])

    L, R, scale_stats = scale_microgrid(df["load"], df["renew"],
                                        m_train.values
                                        if hasattr(m_train, "values")
                                        else m_train, P_base, gamma_R)
    N = L - R                                   # kW
    C = df["price"] / 1000.0                    # currency/kWh
    data = pd.DataFrame({"N": N, "C": C})

    country = zone.split("_")[0]
    hdates = holiday_dates(country, sorted(set(df.index.year)))
    regimes = assign_regime(data.index, hdates)

    train = data[m_train]
    reg_train = regimes[m_train]

    out_regimes = {}
    residual_frames = {}
    for reg in sorted(reg_train.unique()):
        sel = train[reg_train == reg]
        hours = sel.index.hour
        n_bar = np.array([np.median(sel["N"][hours == h]) for h in range(24)])
        c_bar = np.array([np.median(sel["C"][hours == h]) for h in range(24)])
        y_res = sel["N"] - n_bar[hours]
        p_res = sel["C"] - c_bar[hours]
        y0, y1 = within_day_pairs(y_res)
        p0, p1 = within_day_pairs(p_res)
        ou = fit_pair_from_pairs(y0, y1, p0, p1, dt=1.0)
        y_lo, y_hi = domain_bounds(y_res.values)
        p_lo, p_hi = domain_bounds(p_res.values)
        out_regimes[reg] = {
            "n_bar": n_bar.tolist(), "c_bar": c_bar.tolist(),
            **ou,
            "y_min": y_lo, "y_max": y_hi, "p_min": p_lo, "p_max": p_hi,
            "n_train_hours": int(len(sel)),
        }
        residual_frames[reg] = (n_bar, c_bar)

    # out-of-domain frequencies per split
    ood = {}
    for name, mask in (("train", m_train), ("val", m_val), ("test", m_test)):
        n_out = 0
        n_tot = 0
        d = data[mask]
        rg = regimes[mask]
        for reg, (n_bar, c_bar) in residual_frames.items():
            if reg not in out_regimes:
                continue
            sel = d[rg == reg]
            hh = sel.index.hour
            y_res = sel["N"] - n_bar[hh]
            p_res = sel["C"] - c_bar[hh]
            b = out_regimes[reg]
            n_out += int(((y_res < b["y_min"]) | (y_res > b["y_max"])
                          | (p_res < b["p_min"]) | (p_res > b["p_max"])).sum())
            n_tot += len(sel)
        ood[name] = n_out / max(n_tot, 1)

    # --- method-independent objective coefficients (train only) ---
    day = train.index.normalize()
    Np = np.maximum(train["N"], 0.0)
    Nm = np.maximum(-train["N"], 0.0)
    daily_bill = (train["C"] * Np - alpha_s * train["C"] * Nm).groupby(day).sum()
    daily_qpk = (Np ** 2).groupby(day).sum()
    # only complete days
    counts = train["N"].groupby(day).count()
    full = counts[counts == 24].index
    B0 = float(np.median(daily_bill[full]))
    Qpk0 = float(np.median(daily_qpk[full]))

    rc = float(cfg["cost"]["replacement_cost_per_kwh"])
    cl = float(cfg["cost"]["cycle_life_efc"])
    coeffs = {
        "B0": B0, "Qpk0": Qpk0,
        "lam_pk": 0.10 * B0 / Qpk0,
        "lam1": rc / (2.0 * cl),
        "lam2": 0.01 * B0 / (T * a_max ** 2),
        "lam_T": 0.25 * B0 / (0.10 ** 2),
        "eps_g": 1e-3 * P_base,
        "eps_a": 1e-3 * a_max,
        "lam_s": float(cfg["cost"].get("lambda_s", 0.0)),
        "alpha_s": alpha_s,
    }
    return {"zone": zone, "tz": tz, "split": split,
            "scaling": scale_stats, "coefficients": coeffs,
            "regimes": out_regimes, "ood_fraction": ood,
            "currency": "GBP" if country == "GB" else "EUR"}


def main() -> int:
    cfg = load_config()
    out = {}
    for role in ("primary", "external", "confirmatory"):
        if os.path.exists(os.path.join(SPLITS, f"{role}_split.json")):
            out[role] = calibrate_region(role, cfg)
            c = out[role]["coefficients"]
            print(f"{role}: B0={c['B0']:.2f} lam_pk={c['lam_pk']:.3e} "
                  f"lam1={c['lam1']:.4f} lam2={c['lam2']:.3e} "
                  f"lam_T={c['lam_T']:.1f}")
            print(f"  OOD fractions: {out[role]['ood_fraction']}")
    path = os.path.join(ROOT, "data", "calibrated_parameters.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
