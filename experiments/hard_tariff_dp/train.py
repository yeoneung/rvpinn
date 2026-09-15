"""Train matched two-state PINN-PI models in a separate result directory."""
from pathlib import Path
import dataclasses
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from src.exp_common import load_stack, regime_bundle, seeds_of, trainer_cfg
from src.policy_iteration import PINNPITrainer
from src.reproducibility import seed_everything

HERE = Path(__file__).resolve().parent


def portable_checkpoint_fields(value):
    """Keep training metrics intact and store checkpoint paths relative to ROOT."""
    if isinstance(value, list):
        return [portable_checkpoint_fields(item) for item in value]
    if isinstance(value, dict):
        record = {}
        for key, item in value.items():
            if key in ('checkpoint', 'selected_checkpoint') and isinstance(item, str):
                path = Path(item)
                resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
                record[key] = resolved.relative_to(ROOT).as_posix()
            else:
                record[key] = portable_checkpoint_fields(item)
        return record
    return value


def normalize_training_records(target):
    for name in ('iterations.json', 'result.json'):
        path = target / name
        record = json.loads(path.read_text())
        path.write_text(json.dumps(portable_checkpoint_fields(record), indent=2), encoding='utf-8')


def bundle():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    cfg = load_stack(str(ROOT / protocol['training_override']))
    params, profiles, b0 = regime_bundle(protocol['region'], [protocol['regime']], cfg)
    p, prof = params[protocol['regime']], profiles[protocol['regime']]
    p.c_step, p.g_thr, p.w_step = protocol['fee'], protocol['threshold'], protocol['smoothing_width']
    p.sigma_p, p.rho = 0.0, 0.0
    return protocol, cfg, p, prof, b0


def main():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol, cfg, p, prof, b0 = bundle()
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    manifest_path = out / 'training_manifest.json'
    protocol_sha = hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['protocol_sha256'] == protocol_sha
    else:
        manifest = dict(protocol_sha256=protocol_sha, parameters=dataclasses.asdict(p),
                        config=trainer_cfg(cfg), device='cuda' if torch.cuda.is_available() else 'cpu', seeds=[])
    for seed in protocol['training_seeds']:
        target = out / f'seed{seed}'
        if any(row['seed'] == seed for row in manifest['seeds']):
            continue
        if target.exists():
            raise RuntimeError(f'Incomplete training directory needs inspection: {target}')
        seed_everything(seed)
        trainer = PINNPITrainer(p, prof, trainer_cfg(cfg), seeds_of(cfg), b0 / p.T,
                               use_p=False, device=manifest['device'], method_seed=seed,
                               adaptive=False)
        start = time.perf_counter()
        print(f'Training reduced hard-tariff model, seed {seed}', flush=True)
        trainer.run(str(target), max_iterations=protocol['training_rounds'])
        elapsed = time.perf_counter()-start
        normalize_training_records(target)
        manifest['seeds'].append(dict(seed=seed, wall_s=elapsed,
            checkpoints={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in target.glob('*.pt')}))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('All reduced-model training seeds complete.', flush=True)


if __name__ == '__main__':
    main()
