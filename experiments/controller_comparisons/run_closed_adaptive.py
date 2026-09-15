"""Execute unchanged closed-loop jobs, expanding only when the fixed pool finishes.

Do not run concurrently with another closed-loop executor. This changes worker
allocation, not the controller, scenario stream, sample, budget, or stop rule.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from run_solver_budget import HERE, OUT, init, task, save


def fixed_complete():
    manifest=json.loads((OUT/'fixed_manifest.json').read_text())
    return all((OUT/(j['id']+'.json')).exists() for j in manifest['jobs'])


def main():
    manifest=json.loads((OUT/'closed_manifest.json').read_text())
    assert manifest['protocol_sha256']==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    assert manifest['script_sha256']==hashlib.sha256((HERE/'run_solver_budget.py').read_bytes()).hexdigest()
    # The original fixed and surrogate pools must not both still be active:
    # four closed workers plus four fixed workers keep at most eight solve jobs.
    surrogate=json.loads((HERE/'results/surrogate/selection.json').read_text())
    for fee,choice in surrogate.items():
        width=choice['selected']['width']; name='envelope' if width is None else f'width{width}'
        assert len(list((HERE/'results/surrogate').glob(f'test_fee{fee}_{name}_*.json')))==30
    pending=[j for j in manifest['jobs'] if not (OUT/(j['id']+'.json')).exists()]
    execution=dict(start_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   original_runner_sha256=manifest['script_sha256'],
                   completed_at_start=len(manifest['jobs'])-len(pending),
                   allocation_rule='4 concurrent closed jobs until all fixed jobs finish, then 8',events=[])
    log_path=OUT/'adaptive_execution.json'
    history=json.loads(log_path.read_text()) if log_path.exists() else []
    history.append(execution)
    active={}; completed=0; last_cap=None; last_log=time.monotonic()
    with ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        while pending or active:
            cap=8 if fixed_complete() else 4
            if cap!=last_cap:
                execution['events'].append(dict(utc=datetime.now(timezone.utc).isoformat(),workers=cap))
                print(f'Closed-loop concurrent jobs: {cap}',flush=True)
                last_cap=cap
            while pending and len(active)<cap:
                job=pending.pop(0); active[pool.submit(task,job)]=job
            done,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in done:
                result=future.result(); del active[future]; completed+=1
                print(f'closed adaptive {completed}: {result}',flush=True)
            if done or time.monotonic()-last_log>=30:
                execution.update(completed_new=completed,pending=len(pending),active=[j['id'] for j in active.values()],
                                 updated_utc=datetime.now(timezone.utc).isoformat())
                save(log_path,history); last_log=time.monotonic()
    execution.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    save(log_path,history)
    assert all((OUT/(j['id']+'.json')).exists() for j in manifest['jobs'])
    print('BUDGET closed COMPLETE',flush=True)


if __name__=='__main__':
    main()
