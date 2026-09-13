"""Portable release metadata must retain valid numerical and timing digests."""
from pathlib import Path
import hashlib
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_numerical_artifact_hashes_match():
    audit = json.loads((ROOT / "experiments/results/corrected_input_audit.json")
                       .read_text(encoding="utf-8"))
    assert audit["status"] == "input_accounting_and_coverage_passed"
    for name, expected in audit["artifact_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_training_time_manifest_references_match_portable_records():
    rows = pd.read_csv(ROOT / "experiments/results/confirmatory_restart_summary.csv")
    assert len(rows) == 35
    for row in rows.itertuples():
        record = ROOT / row.training_manifest
        assert hashlib.sha256(record.read_bytes()).hexdigest() == row.training_manifest_sha256


def test_checkpoint_and_run_metadata_do_not_require_machine_directories():
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert not re.search(r"[A-Za-z]:[\\/]Users[\\/]", value)

    for name in ("checkpoints", "results/logs", "experiments/results"):
        for path in (ROOT / name).rglob("*.json"):
            visit(json.loads(path.read_text(encoding="utf-8")))
