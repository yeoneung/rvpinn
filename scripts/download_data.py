"""Download the OPSD time-series package and record provenance.

Saves the original archive unchanged under data/raw/ and records release,
URL, date, SHA-256, license, and columns used in data/data_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config                     # noqa: E402
from src.reproducibility import file_sha256            # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    src = cfg["source"]
    url = src["url"]
    release = src["release"]
    local_path = src.get("local_path")
    dest = os.path.join(RAW, f"opsd_time_series_60min_{release}.csv")
    os.makedirs(RAW, exist_ok=True)

    if local_path:
        import shutil
        shutil.copy(local_path, dest)
        origin = f"local:{local_path}"
    elif os.path.exists(dest) and not args.force:
        print(f"already downloaded: {dest}")
        origin = url
    else:
        print(f"downloading {url} ...")
        tmp = dest + ".part"
        try:
            with urllib.request.urlopen(url, timeout=120) as r, \
                    open(tmp, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (1 << 26) < (1 << 22):
                        print(f"  {total/2**20:.0f} MB", flush=True)
            os.replace(tmp, dest)
        except Exception as e:
            print(f"DOWNLOAD FAILED: {e}", file=sys.stderr)
            with open(os.path.join(ROOT, "artifacts", "failure_log.md"),
                      "a", encoding="utf-8") as f:
                f.write(f"\n## {time.strftime('%Y-%m-%d')} — OPSD download "
                        f"failure\n`{url}`: {e}\n")
            return 1
        origin = url

    sha = file_sha256(dest)
    size = os.path.getsize(dest)
    manifest_path = os.path.join(ROOT, "data", "data_manifest.json")
    manifest = {
        "source": "Open Power System Data time_series package",
        "release": release,
        "url": origin,
        "download_date": time.strftime("%Y-%m-%d"),
        "file": os.path.relpath(dest, ROOT),
        "sha256": sha,
        "size_bytes": size,
        "license": "OPSD time series: mixed sources; see "
                   "https://open-power-system-data.org (primarily "
                   "ENTSO-E Transparency, national TSOs); attribution "
                   "required per OPSD terms",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(ROOT, "data", "checksums.txt"), "a",
              encoding="utf-8") as f:
        f.write(f"{sha}  {os.path.basename(dest)}\n")
    print(f"saved {dest} ({size/2**20:.1f} MB), sha256={sha[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
