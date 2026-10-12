"""Check paired inference against the distributed daily outcomes."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rule_miqp_paired", ROOT / "experiments/controller_comparisons/analyze_rule_miqp.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_pairing_and_circular_blocks():
    analysis.self_test()


def test_stored_paired_intervals():
    result = json.loads((ROOT / "experiments/controller_comparisons/results/heuristic/rule_miqp_paired.json").read_text())
    assert result["n_boot"] == analysis.N_BOOT == 10000
    assert result["block_eligible_days"] == analysis.BLOCK == 7
    for item in result["inputs"]:
        assert analysis.hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    assert [row["fee"] for row in result["results"]] == [20, 40, 80]
    for row in result["results"]:
        pairs = row["daily_pairs"]
        assert row["n_days"] == len(pairs) == len({p["date"] for p in pairs}) == 30
        assert [p["date"] for p in pairs] == sorted(p["date"] for p in pairs)
        differences = np.array([p["common_cost_rule"] - p["common_cost_miqp"] for p in pairs])
        np.testing.assert_allclose(differences.mean(), row["difference"], rtol=0, atol=1e-10)
        assert row["seed"] == analysis.BASE_SEED + row["fee"]
        interval = np.percentile(analysis.moving_block_means(differences, row["seed"]), [2.5, 97.5])
        np.testing.assert_allclose(interval, row["ci95"], rtol=0, atol=1e-10)
