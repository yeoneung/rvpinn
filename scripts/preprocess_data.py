"""Zone discovery, deterministic zone selection, cleaning, and splits.

Selection rule (preregistered, main.tex §7.2): among bidding zones with
simultaneous load, renewable (solar + onshore wind + offshore wind, at least
one component) and day-ahead price coverage, keep years with <= 1% missing
in every series, then choose by (1) maximum complete overlapping hours,
(2) minimum missingness, (3) lexicographic zone code. External region =
highest-ranked remaining zone from a different country.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config                              # noqa: E402
from src.data_preprocess import (chronological_split,           # noqa: E402
                                 drop_incomplete_days,
                                 interpolate_short_gaps)

RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
SPLITS = os.path.join(ROOT, "data", "splits")

LOAD_PAT = re.compile(r"^(?P<zone>[A-Z_]+?)_load_actual_entsoe_transparency$")
PRICE_PAT = re.compile(r"^(?P<zone>[A-Z_]+?)_price_day_ahead$")
RENEW_PATS = {
    "solar": "_solar_generation_actual",
    "wind_onshore": "_wind_onshore_generation_actual",
    "wind_offshore": "_wind_offshore_generation_actual",
    "wind": "_wind_generation_actual",
}

# documented local-timezone mapping for calendar regimes
TZ_OF_COUNTRY = {
    "DE": "Europe/Berlin", "AT": "Europe/Vienna", "BE": "Europe/Brussels",
    "CH": "Europe/Zurich", "CZ": "Europe/Prague", "DK": "Europe/Copenhagen",
    "ES": "Europe/Madrid", "FR": "Europe/Paris", "GB": "Europe/London",
    "HU": "Europe/Budapest", "IT": "Europe/Rome", "NL": "Europe/Amsterdam",
    "NO": "Europe/Oslo", "PL": "Europe/Warsaw", "PT": "Europe/Lisbon",
    "RO": "Europe/Bucharest", "SE": "Europe/Stockholm", "SI": "Europe/Ljubljana",
    "SK": "Europe/Bratislava", "GR": "Europe/Athens", "FI": "Europe/Helsinki",
    "IE": "Europe/Dublin", "LT": "Europe/Vilnius", "LV": "Europe/Riga",
    "EE": "Europe/Tallinn", "BG": "Europe/Sofia", "HR": "Europe/Zagreb",
    "LU": "Europe/Luxembourg", "RS": "Europe/Belgrade",
}


def country_of_zone(zone: str) -> str:
    return zone.split("_")[0]


def discover_zones(columns) -> dict:
    zones = {}
    for c in columns:
        m = LOAD_PAT.match(c)
        if m:
            zones.setdefault(m.group("zone"), {})["load"] = c
        m = PRICE_PAT.match(c)
        if m:
            zones.setdefault(m.group("zone"), {})["price"] = c
    for c in columns:
        for key, suffix in RENEW_PATS.items():
            if c.endswith(suffix):
                zone = c[: -len(suffix)]
                if zone in zones:
                    zones[zone].setdefault("renew", []).append(c)
    return {z: v for z, v in zones.items()
            if "load" in v and "price" in v and v.get("renew")}


def yearly_quality(df: pd.DataFrame, max_missing: float):
    """Per-year missing fraction across the three series; complete years."""
    out = {}
    for y in sorted(set(df.index.year)):
        sl = df[df.index.year == y]
        hours_in_year = pd.Timestamp(f"{y+1}-01-01") - pd.Timestamp(f"{y}-01-01")
        n_expected = int(hours_in_year.total_seconds() // 3600)
        if len(sl) < 0.99 * n_expected:
            frac = 1.0
        else:
            frac = float(sl.isna().any(axis=1).mean())
        out[y] = frac
    complete = [y for y, f in out.items() if f <= max_missing]
    # keep only the longest consecutive run ending at the newest year to
    # guarantee a contiguous chronological interval
    complete_sorted = sorted(complete)
    runs = []
    for y in complete_sorted:
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    best = max(runs, key=len) if runs else []
    return out, best


def main() -> int:
    cfg = load_config()
    release = cfg["source"]["release"]
    csv_path = os.path.join(RAW, f"opsd_time_series_60min_{release}.csv")
    if not os.path.exists(csv_path):
        print("raw file missing; run scripts/download_data.py", file=sys.stderr)
        return 1
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(SPLITS, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0)
    zones = discover_zones(header.columns)
    print(f"{len(zones)} candidate zones with load+price+renewable")

    max_missing = float(cfg["zone_selection"]["max_missing_fraction_per_year"])
    rows = []
    zone_frames = {}
    usecols_all = ["utc_timestamp"]
    for z, v in zones.items():
        usecols_all += [v["load"], v["price"]] + v["renew"]
    df_all = pd.read_csv(csv_path, usecols=sorted(set(usecols_all)),
                         parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    df_all.index = pd.DatetimeIndex(df_all.index).tz_convert("UTC")

    for z, v in sorted(zones.items()):
        country = country_of_zone(z)
        if country not in TZ_OF_COUNTRY:
            continue
        sub = pd.DataFrame({
            "load": df_all[v["load"]],
            "price": df_all[v["price"]],
            "renew": df_all[v["renew"]].sum(axis=1, min_count=1),
        })
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        quality, complete_years = yearly_quality(sub, max_missing)
        n_hours = sum(int((sub.index.year == y).sum()) for y in complete_years)
        overall_missing = (float(np.mean([quality[y] for y in complete_years]))
                           if complete_years else 1.0)
        rows.append({"zone": z, "country": country,
                     "complete_years": ",".join(map(str, complete_years)),
                     "n_complete_years": len(complete_years),
                     "complete_hours": n_hours,
                     "mean_missing_in_complete_years": overall_missing})
        if complete_years:
            zone_frames[z] = (sub, complete_years)

    rank = pd.DataFrame(rows).sort_values(
        ["complete_hours", "mean_missing_in_complete_years", "zone"],
        ascending=[False, True, True]).reset_index(drop=True)
    rank.to_csv(os.path.join(ROOT, "data", "zone_selection.csv"), index=False)
    print(rank.head(10).to_string())

    eligible = rank[rank["complete_hours"] > 0]
    primary = eligible.iloc[0]["zone"]
    primary_country = eligible.iloc[0]["country"]
    external = None
    for _, r in eligible.iloc[1:].iterrows():
        if r["country"] != primary_country:
            external = r["zone"]
            break
    if external is None and len(eligible) > 1:
        external = eligible.iloc[1]["zone"]
    print(f"primary zone: {primary}; external zone: {external}")

    max_gap = int(cfg["zone_selection"]["max_interp_gap_hours"])
    selection = {"primary": primary, "external": external}
    for role, zone in selection.items():
        if zone is None:
            continue
        sub, years = zone_frames[zone]
        sub = sub[(sub.index.year >= years[0]) & (sub.index.year <= years[-1])]
        missing_before = sub.isna().mean().to_dict()
        for c in sub.columns:
            sub[c] = interpolate_short_gaps(sub[c], max_gap)
        sub = drop_incomplete_days(sub)
        missing_after = sub.isna().mean().to_dict()
        split = chronological_split(sub.index)
        with open(os.path.join(SPLITS, f"{role}_split.json"), "w") as f:
            json.dump({"zone": zone, "years": years, "split": split,
                       "tz": TZ_OF_COUNTRY[country_of_zone(zone)],
                       "missing_before": missing_before,
                       "missing_after": missing_after,
                       "n_hours": len(sub)}, f, indent=2, default=str)
        out = os.path.join(PROC, f"{role}_{zone}.parquet")
        sub.to_parquet(out)
        print(f"{role}: {zone} -> {out} ({len(sub)} h, "
              f"{split['train']} / {split['val']} / {split['test']})")

    with open(os.path.join(ROOT, "data", "data_manifest.json")) as f:
        manifest = json.load(f)
    manifest["zones"] = selection
    manifest["columns_used"] = {z: zones[z] for z in selection.values()
                                if z is not None}
    with open(os.path.join(ROOT, "data", "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
