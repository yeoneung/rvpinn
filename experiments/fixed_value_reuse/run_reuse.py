"""Run the fixed-model reuse benchmark with its specified checkpoint."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
import time

from common import HERE, PUBLIC, RESULTS, protocol, digest, make_cases, bundle, gpu_status
from simulation import simulate_miqp, simulate_neural, warm_scip_worker


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare():
    RESULTS.mkdir(parents=True, exist_ok=True)
    cfg = protocol()
    if digest(PUBLIC / cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise RuntimeError('Checkpoint differs from the reuse protocol')
    target = RESULTS / 'manifest.json'
    if target.exists():
        manifest = json.loads(target.read_text())
        assert manifest['protocol_sha256'] == digest(HERE / 'protocol.json')
        assert manifest['checkpoint_sha256'] == digest(PUBLIC / cfg['checkpoint'])
        return manifest
    import pandas as pd
    offline = pd.read_csv(PUBLIC / 'experiments/results/confirmatory_offline_compute.csv')
    historical = float(offline[(offline.fee == 40) & (offline.batch == 0)].total_measured_offline_s.iloc[0])
    started = time.perf_counter()
    cases = make_cases()
    manifest = dict(protocol_sha256=digest(HERE / 'protocol.json'),
                    checkpoint_sha256=digest(PUBLIC / cfg['checkpoint']), cases=cases,
                    path_preparation_s=time.perf_counter()-started,
                    historical_training_and_selection_s=historical,
                    historical_timing_source='confirmatory_offline_compute.csv; fee40 batch0',
                    python=os.sys.version, background_jobs_not_stopped=True,
                    implementation_files=sorted(f.name for f in HERE.glob('*.py')))
    save(target, manifest)
    return manifest


def run_miqp(manifest):
    cfg = protocol()
    for n in cfg['benchmark_sizes']:
        target = RESULTS / f'miqp_n{n}.json'
        if target.exists():
            continue
        partial = RESULTS / f'miqp_n{n}_partial.jsonl'
        if partial.exists():
            raise RuntimeError(f'Interrupted timed group {n}; partial records preserved. Do not silently treat resumed timing as uninterrupted.')
        workers = min(n, cfg['miqp_workers'])
        setup_started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=warm_scip_worker) as pool:
            [f.result() for f in [pool.submit(bundle) for _ in range(workers)]]
            setup_s = time.perf_counter() - setup_started
            started = time.perf_counter()
            futures = {pool.submit(simulate_miqp, c): c['id'] for c in manifest['cases'][:n]}
            outputs = []
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(case_id=case_id, error=repr(exc))
                outputs.append(result)
                with partial.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(result) + '\n')
                print(f'MIQP n={n} completed={len(outputs)}/{n} case={case_id} '
                      f'cost={result.get("row", {}).get("cost")} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            wall_s = time.perf_counter() - started
        save(target, dict(method='miqp', n=n, workers=workers, process_setup_s=setup_s,
                          wall_s=wall_s, results=outputs))
        print(f'MIQP GROUP COMPLETE n={n} wall={wall_s:.3f}s', flush=True)


def run_neural(manifest):
    import numpy as np
    import torch
    from src.policy_iteration import load_checkpoint
    from batched_policy import batched_actions
    cfg = protocol()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    p, prof = bundle()
    models, loading = {}, {}
    jobs = [(n, rep) for n in cfg['benchmark_sizes'] for rep in range(cfg['neural_repetitions'])]
    random.Random(821).shuffle(jobs)
    for n, rep in jobs:
        target = RESULTS / f'neural_n{n}_r{rep}.json'
        if target.exists():
            continue
        gpu = gpu_status()
        device = ('cuda' if gpu['free_mib'] is not None and gpu['free_mib'] > 2048
                  and gpu['utilization_percent'] < 85 and torch.cuda.is_available() else 'cpu')
        if device not in models:
            started = time.perf_counter()
            models[device] = load_checkpoint(str(PUBLIC / cfg['checkpoint']), p, device).eval()
            c = manifest['cases'][0]
            for _ in range(2):
                batched_actions(models[device], 0., [c['initial_soc']], [c['y'][0]],
                                [c['pz'][0]], [c['net'][0]], [c['price'][0]])
            loading[device] = time.perf_counter() - started
        modes = ['scalar', 'batched', 'reference']
        random.Random(900+n*3+rep).shuffle(modes)
        output = []
        for mode in modes:
            started = time.perf_counter()
            result = simulate_neural(manifest['cases'][:n], models[device], mode)
            result.update(method=mode, n=n, repetition=rep, device=device if mode != 'reference' else 'cpu',
                          full_call_wall_s=time.perf_counter()-started, gpu_background_before=gpu)
            output.append(result)
            print(f'NEURAL n={n} repetition={rep} mode={mode} device={result["device"]} '
                  f'wall={result["full_call_wall_s"]:.3f}s', flush=True)
        by_mode = {r['method']: r for r in output}
        scalar = np.array([r['actions'] for r in by_mode['scalar']['rows']])
        batch = np.array([r['actions'] for r in by_mode['batched']['rows']])
        costs_scalar = np.array([r['cost'] for r in by_mode['scalar']['rows']])
        costs_batch = np.array([r['cost'] for r in by_mode['batched']['rows']])
        check = dict(max_action_difference_kw=float(np.max(np.abs(scalar-batch))),
                     max_cost_difference_eur=float(np.max(np.abs(costs_scalar-costs_batch))))
        save(target, dict(results=output, scalar_batch_check=check))
        if check['max_action_difference_kw'] > 1e-5 or check['max_cost_difference_eur'] > 1e-6:
            raise RuntimeError('Batching changed the deployed policy; retained output is diagnostic only')
    save(RESULTS / 'neural_setup.json', dict(model_loading_and_warmup_s=loading,
                                            torch=torch.__version__, cuda=torch.version.cuda,
                                            gpu_status_end=gpu_status()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'neural', 'miqp', 'analyze'])
    args = parser.parse_args()
    manifest = prepare()
    if args.phase == 'neural':
        run_neural(manifest)
    elif args.phase == 'miqp':
        run_miqp(manifest)
    elif args.phase == 'analyze':
        from analyze_fixed_value_reuse import main as analyze
        analyze()


if __name__ == '__main__':
    main()
