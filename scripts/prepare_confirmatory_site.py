"""Prepare the prespecified untouched IT_NORD confirmatory site.

The site is selected from field availability alone. Selection provenance is
recorded in data/splits/confirmatory_split.json. This script reports coverage
and creates chronological train/validation/test files;
it does not compute any controller cost or compare any method.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data_preprocess import (chronological_split, drop_incomplete_days,
                                 interpolate_short_gaps)  # noqa: E402
from scripts.preprocess_data import (TZ_OF_COUNTRY, discover_zones,
                                     yearly_quality)  # noqa: E402

ZONE = "IT_NORD"
ROLE = "confirmatory"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_raw(release: str) -> Path:
    name = f"opsd_time_series_60min_{release}.csv"
    candidates = [ROOT / "data" / "raw" / name,
                  ROOT.parent / "data" / "raw" / name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{name} not found; run scripts/download_data.py or place it under "
        f"{ROOT / 'data' / 'raw'}")


def main() -> int:
    cfg = load_config()
    release = str(cfg["source"]["release"])
    raw = locate_raw(release)
    header = pd.read_csv(raw, nrows=0)
    zones = discover_zones(header.columns)
    if ZONE not in zones:
        raise RuntimeError(f"prespecified zone {ZONE} lacks required fields")
    fields = zones[ZONE]
    usecols = ["utc_timestamp", fields["load"], fields["price"],
               *fields["renew"]]
    raw_frame = pd.read_csv(raw, usecols=usecols,
                            parse_dates=["utc_timestamp"],
                            index_col="utc_timestamp")
    raw_frame.index = pd.DatetimeIndex(raw_frame.index).tz_convert("UTC")
    frame = pd.DataFrame({
        "load": raw_frame[fields["load"]],
        "price": raw_frame[fields["price"]],
        "renew": raw_frame[fields["renew"]].sum(axis=1, min_count=1),
    }).dropna(how="all")

    max_missing = float(
        cfg["zone_selection"]["max_missing_fraction_per_year"])
    quality, years = yearly_quality(frame, max_missing)
    if len(years) < 4:
        raise RuntimeError(
            f"{ZONE} has only {len(years)} consecutive eligible years: {years}")
    frame = frame[(frame.index.year >= years[0])
                  & (frame.index.year <= years[-1])].copy()
    missing_before = frame.isna().mean().to_dict()
    max_gap = int(cfg["zone_selection"]["max_interp_gap_hours"])
    for column in frame.columns:
        frame[column] = interpolate_short_gaps(frame[column], max_gap)
    frame = drop_incomplete_days(frame)
    missing_after = frame.isna().mean().to_dict()
    # Four complete years: first two training, third validation, last test.
    # This explicit calendar split was frozen before any cost outcome.
    split = {
        "train": (f"{years[0]}-01-01", f"{years[-2]}-01-01"),
        "val": (f"{years[-2]}-01-01", f"{years[-1]}-01-01"),
        "test": (f"{years[-1]}-01-01", f"{years[-1] + 1}-01-01"),
    }
    # The final eligible calendar year must be the only test year.  Abort
    # instead of silently accepting a broader fractional holdout.
    test_start = pd.Timestamp(split["test"][0]).year
    test_end = pd.Timestamp(split["test"][1]).year
    if test_start != years[-1] or test_end != years[-1] + 1:
        raise RuntimeError(
            f"expected final-year test, obtained {split['test']} from {years}")

    proc = ROOT / "data" / "processed"
    splits = ROOT / "data" / "splits"
    proc.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    data_path = proc / f"{ROLE}_{ZONE}.parquet"
    split_path = splits / f"{ROLE}_split.json"
    frame.to_parquet(data_path)
    meta = {
        "zone": ZONE,
        "selection_basis": ("four consecutive complete years and a major "
                            "previously unused bidding zone"),
        "years": years,
        "per_year_missing_fraction": quality,
        "split": split,
        "tz": TZ_OF_COUNTRY["IT"],
        "missing_before": missing_before,
        "missing_after": missing_after,
        "n_hours": len(frame),
        "source_sha256": sha256(raw),
        "outcome_inspected_before_freeze": False,
        "protocol_amendment": ("DE_LU failed the coverage-only eligibility "
                               "check before any controller outcome"),
    }
    split_path.write_text(json.dumps(meta, indent=2, default=str),
                          encoding="utf-8")

    manifest_path = ROOT / "data" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("zones", {})[ROLE] = ZONE
    manifest.setdefault("columns_used", {})[ZONE] = fields
    manifest["confirmatory_protocol"] = {
        "frozen_on": "2026-09-06 Asia/Seoul",
        "site": ZONE,
        "test": split["test"],
        "selection_basis": meta["selection_basis"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str),
                             encoding="utf-8")
    print(json.dumps({
        "zone": ZONE, "eligible_years": years, "split": split,
        "n_hours_after_cleaning": len(frame),
        "missing_before": missing_before, "missing_after": missing_after,
        "data_path": str(data_path), "split_path": str(split_path),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
