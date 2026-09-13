"""Export lightweight checkpoint metadata needed by table regeneration.

This is an author-side export, not a prerequisite for readers regenerating
tables from the committed manifest. Neural weights remain optional downloads.
"""
from pathlib import Path
import hashlib
import json

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = {}
    for path in sorted((ROOT / "checkpoints").glob("v3_*/confirmatory/*/seed*/iter*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = path.relative_to(ROOT / "checkpoints").as_posix()
        records[key] = {
            "out_scale": float(payload["out_scale"]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if not records:
        raise RuntimeError("no author-side checkpoints found")
    target = ROOT / "data" / "checkpoint_metadata.json"
    target.write_text(json.dumps({"checkpoints": records}, indent=2), encoding="utf-8")
    print(f"exported {len(records)} checkpoint metadata records to {target}")


if __name__ == "__main__":
    main()
