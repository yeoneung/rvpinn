"""Validate all rule settings, freeze selection, then evaluate historical tests."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from src.evaluation import load_region_days, run_day
from src.exp_common import regime_bundle
from src.rules import NoStorage
from heuristic import TariffPeakShaving

OUT = HERE / "results/heuristic"
_CACHE = {}


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def init():
    for part in ("train", "val", "test"):
        _CACHE[part] = load_region_days("confirmatory", part)
    _CACHE["params"], _CACHE["profiles"], _ = regime_bundle("confirmatory")


def task(payload):
    regime, fee, part, cid, setting = payload
    p = deepcopy(_CACHE["params"][regime])
    p.c_step, p.g_thr, p.w_step = fee, 300.0, 10.0
    controller = NoStorage() if setting is None else TariffPeakShaving(p, **setting)
    days = [d for d in _CACHE[part] if d["regime"] == regime][:30]
    assert len(days) == 30
    started = time.perf_counter()
    rows = [run_day(controller, day, p, _CACHE["profiles"][regime]) for day in days]
    return dict(regime=regime, fee=fee, partition=part, candidate_id=cid,
                setting=setting, days=rows, mean_cost=float(np.mean([r["common_cost"] for r in rows])),
                elapsed_s=time.perf_counter() - started)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "protocol.json").read_text())
    init()
    h = cfg["heuristic"]
    prices = np.concatenate([d["C"] for d in _CACHE["train"]])
    keys = ["reserve_soc", "charge_target_soc", "charge_price_quantile", "charge_when_already_above_band", "allow_partial_shaving"]
    settings = [None]
    for values in product(*(h[k] for k in keys)):
        setting = dict(zip(keys, values))
        quantile = setting.pop("charge_price_quantile")
        setting["charge_price"] = float(np.quantile(prices, quantile))
        settings.append(setting)
    cells = [("winter_weekday", fee) for fee in cfg["fees"]]
    cells += [(season + "_weekday", 40) for season in ("spring", "summer", "autumn")]
    manifest = dict(protocol_sha256=hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                    heuristic_sha256=hashlib.sha256((HERE / "heuristic.py").read_bytes()).hexdigest(),
                    candidates=settings, cells=cells,
                    dates={part: {regime: [d["date"] for d in _CACHE[part] if d["regime"] == regime][:30]
                                  for regime, fee in cells} for part in ("val", "test")})
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == json.loads(json.dumps(manifest)), "Frozen experiment changed"
    else:
        save(manifest_path, manifest)
    jobs = [(r, f, "val", i, setting) for r, f in cells for i, setting in enumerate(settings)]
    with ProcessPoolExecutor(max_workers=4, initializer=init) as pool:
        futures = {}
        for job in jobs:
            path = OUT / f"val_{job[0]}_fee{job[1]}_candidate{job[3]:03d}.json"
            if not path.exists():
                futures[pool.submit(task, job)] = path
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            save(futures[future], result)
            if n % 12 == 0 or n == len(futures):
                print(f"validation {n}/{len(futures)} {result['regime']} fee={result['fee']} mean={result['mean_cost']:.3f}", flush=True)
        # Every validation setting is complete before any test evaluation starts.
        selected = []
        for regime, fee in cells:
            records = [json.loads((OUT / f"val_{regime}_fee{fee}_candidate{i:03d}.json").read_text()) for i in range(len(settings))]
            incumbent = records[0]
            for record in records[1:]:
                if incumbent["mean_cost"] - record["mean_cost"] >= 0.10:
                    incumbent = record
            selected.append(dict(regime=regime, fee=fee, candidate_id=incumbent["candidate_id"],
                                 setting=incumbent["setting"], validation_cost=incumbent["mean_cost"]))
        selection_path = OUT / "selection.json"
        if selection_path.exists():
            assert json.loads(selection_path.read_text()) == selected
        else:
            save(selection_path, selected)
        futures = {}
        for s in selected:
            path = OUT / f"test_{s['regime']}_fee{s['fee']}.json"
            if not path.exists():
                job = (s["regime"], s["fee"], "test", s["candidate_id"], s["setting"])
                futures[pool.submit(task, job)] = path
        for future in as_completed(futures):
            result = future.result()
            save(futures[future], result)
            print(f"TEST {result['regime']} fee={result['fee']} candidate={result['candidate_id']} cost={result['mean_cost']:.6f}", flush=True)
    print("HEURISTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
