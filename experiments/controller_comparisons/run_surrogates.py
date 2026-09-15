"""Validation-selected hinge slopes with common hard-cost test evaluation."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exact_miqp import ExactMIPController
from src.evaluation import load_region_days, run_day
from src.exp_common import regime_bundle
from src.safety import deploy_bounds
from src.deployment_numerics import recover_solver_action, BAND_SEPARATION_KW
from run_solver_budget import native_log, save

OUT = HERE / "results/surrogate"
_CACHE = {}


class ThresholdPolishedSurrogate(ExactMIPController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polish = []

    def __call__(self, t, s, y, pz, ctx):
        raw = super().__call__(t, s, y, pz, ctx)
        lo, hi = (float(x[0]) for x in deploy_bounds(np.array([s]), self.p, self.p.dt_ctrl))
        overshoot = ctx["N_now"] - raw - self.p.g_thr
        action = recover_solver_action(raw, ctx["N_now"], self.p.g_thr, lo, hi,
                                       not (0. < overshoot <= BAND_SEPARATION_KW))
        self.polish.append(dict(raw_deployed_action=raw, final_action=action,
                                threshold_polish_kw=action - raw))
        return action


def init():
    params, profiles, _ = regime_bundle("confirmatory", ["winter_weekday"])
    _CACHE["p"], _CACHE["prof"] = params["winter_weekday"], profiles["winter_weekday"]
    for part in ("val", "test"):
        _CACHE[part] = {d["date"]: d for d in load_region_days("confirmatory", part)
                        if d["regime"] == "winter_weekday"}


def task(job):
    with native_log(OUT / (job["id"] + ".log")):
        assert "torch" not in sys.modules
        p, prof = deepcopy(_CACHE["p"]), _CACHE["prof"]
        p.c_step, p.g_thr, p.w_step = job["fee"], 300., 10.
        if job["width"] is None:
            cap = float(np.max(prof.n_bar(np.arange(0., p.T, p.dt_ctrl)))) + p.y_max + p.a_c
            slope, mode = p.c_step / (cap - p.g_thr), "envelope"
        else:
            slope, mode = p.c_step / job["width"], "softcap"
        controller = ThresholdPolishedSurrogate(
            p, reopt_every_hours=p.dt_ctrl, time_limit_s=5., mip_gap=.01,
            random_seed=0, formulation="miqp", shrinking_horizon=True,
            band_mode=mode, step_slope=slope)
        started = time.perf_counter()
        result = run_day(controller, _CACHE[job["part"]][job["date"]], p, prof, collect_traj=True)
        records = [dict(**asdict(r), **polish) for r, polish in zip(controller.solve_records, controller.polish)]
        assert len(records) == 96
        payload = dict(job=job, slope=slope, mode=mode, daily=result, solves=records,
                       elapsed_s=time.perf_counter() - started)
        save(OUT / (job["id"] + ".json"), payload)
        return dict(id=job["id"], elapsed_s=payload["elapsed_s"], cost=result["common_cost"])


def make_jobs(part, choices):
    dates = sorted(_CACHE[part])[:30]
    jobs = []
    for fee, widths in choices.items():
        for width in widths:
            for date in dates:
                name = "envelope" if width is None else f"width{width}"
                identity = f"{part}_fee{fee}_{name}_{date}"
                jobs.append(dict(id=identity, part=part, fee=fee, width=width, date=date))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "protocol.json").read_text())
    init()
    widths = [None] + cfg["surrogate"]["softcap_effective_width_kw"]
    val_jobs = make_jobs("val", {fee: widths for fee in cfg["fees"]})
    manifest = dict(jobs=val_jobs, hashes={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                    for name in ("protocol.json", "surrogate_implementation.json", "run_surrogates.py")})
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        save(path, manifest)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
        def execute(jobs):
            jobs = [job for job in jobs if not (OUT / (job["id"] + ".json")).exists()]
            futures = {pool.submit(task, job): job for job in jobs}
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                print(f"surrogate {i}/{len(jobs)} {r}", flush=True)
        execute(val_jobs)
        selections = {}
        for fee in cfg["fees"]:
            results = []
            for width in widths:
                jobs = [j for j in val_jobs if j["fee"] == fee and j["width"] == width]
                costs = [json.loads((OUT / (j["id"] + ".json")).read_text())["daily"]["common_cost"] for j in jobs]
                results.append(dict(width=width, mean_cost=float(np.mean(costs))))
            best = results[0]
            for candidate in results[1:]:
                if best["mean_cost"] - candidate["mean_cost"] >= .10:
                    best = candidate
            selections[str(fee)] = dict(selected=best, candidates=results)
        path = OUT / "selection.json"
        if path.exists():
            assert json.loads(path.read_text()) == selections
        else:
            save(path, selections)
        execute(make_jobs("test", {int(fee): [r["selected"]["width"]] for fee, r in selections.items()}))
    print("SURROGATE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
