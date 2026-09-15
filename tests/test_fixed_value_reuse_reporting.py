"""Stored reuse results must support both timing and operating-cost statements."""
import copy
import hashlib
import json

import pytest

from experiments import analyze_fixed_value_reuse as report
from experiments.manuscript_checks import enabled


@pytest.fixture(scope="module")
def data():
    return report.collect()


def test_saved_metrics_and_sample_units(data):
    raw, timing, quality, m = data
    assert len(raw) == 30 and len(timing) == 12 and len(quality) == 48
    assert m["independent_paths"] == 4 and m["n_cases"] == 12
    assert m["wall_ratio"] == pytest.approx(15.6987160306)
    assert m["cost_premium_percent"] == pytest.approx(3.8128834115)
    assert m["learned_total_with_offline_s"] > m["miqp_wall_s"]
    assert m["projected_compute_only_crossover_cases"] == 29
    assert m["max_scalar_batch_action_difference_kw"] < 1e-12
    assert m["max_scalar_batch_cost_difference_eur"] == 0
    assert m["primary_miqp_calls"] == 1152


def test_rendered_manuscript_matches_records(data):
    _, timing, quality, m = data
    texts = report.render(timing, quality, m)
    if enabled():
        for name, text in texts.items():
            assert text == (report.GENERATED / name).read_text(encoding="utf-8")
    assert "excluding setup and training" in texts["fixed_value_reuse_abstract.tex"]
    assert "3.8\\% higher" in texts["fixed_value_reuse_abstract.tex"]
    assert "unamortized" in texts["fixed_value_reuse_main.tex"]
    assert "throughput-based projection concerns computation alone" in texts["fixed_value_reuse_supplement.tex"]
    assert "-0.33\\% to 8.71\\%" in texts["fixed_value_reuse_supplement.tex"]


@pytest.mark.skipif(not enabled(), reason="Current manuscript is maintained separately")
def test_supplement_retains_dependence_and_cpu_scope():
    text = (report.ROOT / "manuscript/supplement.tex").read_text(encoding="utf-8")
    assert "only four independent disturbance paths" in text
    assert "All reported reuse runs use the CPU" in text
    assert "not the realized future path" in text
    assert "reconstruction checks realized cost and state feasibility" in " ".join(text.split())


def test_timing_and_cost_text_follow_values(data):
    _, timing, quality, metrics = data
    m = copy.deepcopy(metrics)
    m.update(scalar_wall_s=12.5, miqp_wall_s=400.0, cost_premium_percent=7.2)
    abstract = report.render(timing, quality, m)["fixed_value_reuse_abstract.tex"]
    assert "12.5 s versus 400.0 s" in abstract and "7.2\\% higher" in abstract
    assert "39.5 s" not in abstract


def test_manifest_and_protocol_hash():
    path = report.HERE / "protocol.json"
    manifest = json.loads((report.RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["protocol_sha256"]
