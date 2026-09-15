"""Matched-state and closed-loop budget sweeps; retain every solve and log."""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from stochastic_miqp import ScenarioExactMIPController
from src.evaluation import load_region_days, run_day
from src.exp_common import regime_bundle
from src.dynamics import reflect

OUT = HERE / "results/solver_budget"
_CACHE = {}


def save(path, data):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=convert), encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def native_log(path):
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(i) for i in (1, 2)]
    with path.open("ab", buffering=0) as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for descriptor, backup in zip((1, 2), saved):
                os.dup2(backup, descriptor)
                os.close(backup)


def init():
    params, profiles, _ = regime_bundle("confirmatory", ["winter_weekday"])
    _CACHE["p"], _CACHE["prof"] = params["winter_weekday"], profiles["winter_weekday"]
    for part in ("val", "test"):
        _CACHE[part] = {d["date"]: d for d in load_region_days("confirmatory", part)
                        if d["regime"] == "winter_weekday"}


def compute(job):
    p = deepcopy(_CACHE["p"])
    prof = _CACHE["prof"]
    p.c_step, p.g_thr, p.w_step = job["fee"], 300., 10.
    controller = ScenarioExactMIPController(
        p, n_scenarios=job["scenarios"], scenario_pool_size=16,
        random_seed=job["stream"], time_limit_s=job["limit_s"], mip_gap=.01,
        reopt_every_hours=p.dt_ctrl, shrinking_horizon=True)
    started = time.perf_counter()
    if job["kind"] == "fixed":
        day = _CACHE["val"][job["date"]]
        t, s = job["hour"], job["soc"]
        net, price = float(day["N"][int(t)]), float(day["C"][int(t)])
        y = float(reflect(np.array([net - prof.n_bar(np.array([t]))[0]]), p.y_min, p.y_max)[0])
        pz = float(reflect(np.array([price - prof.c_bar(np.array([t]))[0]]), p.p_min, p.p_max)[0])
        ctx = dict(date=job["date"], N_now=net, C_now=price,
                   N_forecast=lambda clock: prof.n_bar(np.asarray(clock)),
                   C_forecast=lambda clock: prof.c_bar(np.asarray(clock)))
        n, c = controller._forecast_scenarios(t, int(round((p.T-t)/p.dt_ctrl)), y, pz, ctx)
        scenario_hash = hashlib.sha256(n.tobytes() + c.tobytes()).hexdigest()
        action = controller(t, s, y, pz, ctx)
        payload = dict(action=action, current_net=net, current_price=price,
                       scenario_sha256=scenario_hash,
                       solves=[asdict(r) for r in controller.solve_records])
    else:
        day = _CACHE["test"][job["date"]]
        payload = dict(daily=run_day(controller, day, p, prof, collect_traj=True),
                       solves=[asdict(r) for r in controller.solve_records])
    return dict(job=job, elapsed_s=time.perf_counter()-started, **payload)


def task(job):
    with native_log(OUT / (job["id"] + ".log")):
        try:
            assert "torch" not in sys.modules, "SCIP worker must stay Torch-free"
            result = compute(job)
            save(OUT / (job["id"] + ".json"), result)
            return dict(id=job["id"], elapsed_s=result["elapsed_s"],
                        n_solves=len(result["solves"]),
                        cost=result.get("daily", {}).get("common_cost"))
        except Exception:
            error = dict(job=job, error=traceback.format_exc())
            save(OUT / (job["id"] + ".error.json"), error)
            raise


def jobs_for(kind, cfg):
    init()
    jobs = []
    if kind == "fixed":
        dates = sorted(_CACHE["val"])[:2]
        for limit, fee, m, stream, date, hour, soc in product(
                cfg["limits_s"], cfg["fees"], cfg["scenarios"], cfg["streams"], dates, (8,16), (.2,.5,.8)):
            identity = f"fixed_fee{fee}_m{m}_seed{stream}_{date}_h{hour}_s{soc:.1f}_b{limit}"
            jobs.append(dict(id=identity, kind=kind, fee=fee, scenarios=m, stream=stream,
                             date=date, hour=hour, soc=soc, limit_s=limit))
    else:
        dates = sorted(_CACHE["test"])[:3]
        for limit, fee, m, date in product((5,60,300), cfg["fees"], cfg["scenarios"], dates):
            identity = f"closed_fee{fee}_m{m}_seed7301_{date}_b{limit}"
            jobs.append(dict(id=identity, kind=kind, fee=fee, scenarios=m, stream=7301,
                             date=date, limit_s=limit))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["fixed", "closed"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((HERE / "protocol.json").read_text())["solver_budget"]
    jobs = jobs_for(args.kind, cfg)
    manifest = dict(protocol_sha256=hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), jobs=jobs)
    path = OUT / f"{args.kind}_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Frozen budget experiment changed"
    else:
        save(path, manifest)
    pending = [job for job in jobs if not (OUT / (job["id"] + ".json")).exists()]
    print(f"{args.kind}: {len(pending)}/{len(jobs)} jobs remaining; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
        futures = {pool.submit(task, job): job for job in pending}
        for n, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"{args.kind} {n}/{len(pending)} {result}", flush=True)
    print(f"BUDGET {args.kind} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
