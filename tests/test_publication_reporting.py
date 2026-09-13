"""Publication renderers must preserve both estimates and their interpretation."""
from pathlib import Path
import contextlib
import io
import shutil

import pandas as pd
import pytest

from experiments import analyze_round2_reference as reference
from experiments import analyze_round2_scenarios as scenarios
from experiments import analyze_seasonal_sensitivity as seasons

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
GENERATED = ROOT / "manuscript" / "generated"


@pytest.fixture
def scenario_frames():
    return [pd.read_csv(RESULTS / f"round2_scenario_{name}.csv") for name in
            ("setting_summary", "stream_summary", "solver_summary", "pairwise")]


def test_scenario_export_matches_distributed_manuscript(tmp_path, monkeypatch,
                                                       scenario_frames):
    monkeypatch.setattr(scenarios, "GENERATED", tmp_path)
    scenarios._write_latex(*scenario_frames)
    for name in ("round2_scenarios.tex", "round2_scenario_supplement.tex"):
        assert (tmp_path / name).read_text(encoding="utf-8").strip() == (
            GENERATED / name).read_text(encoding="utf-8").strip()


def test_scenario_gap_prose_tracks_availability_not_only_mean(tmp_path, monkeypatch,
                                                            scenario_frames):
    settings, streams, solvers, pairs = scenario_frames
    solvers.loc[(solvers.fee == 80) & (solvers.time_limit_s == 5),
                "finite_gap_fraction"] = 0.125
    solvers.loc[(solvers.fee == 80) & (solvers.time_limit_s == 15),
                "finite_gap_fraction"] = 0.875
    monkeypatch.setattr(scenarios, "GENERATED", tmp_path)
    scenarios._write_latex(settings, streams, solvers, pairs)
    text = (tmp_path / "round2_scenarios.tex").read_text(encoding="utf-8")
    assert r"12.5\% and 87.5\% of calls" in text
    assert "conditional means" in text
    assert "do not measure matched solver progress" in text
    assert "availability and closed-loop states differ" in text


def test_reference_export_matches_distributed_manuscript(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "GENERATED", tmp_path)
    reference._write_latex(
        pd.read_csv(RESULTS / "round2_reference_method_summary.csv"),
        pd.read_csv(RESULTS / "round2_reference_pairwise.csv"))
    for name in ("round2_reference.tex", "round2_reference_supplement.tex"):
        assert (tmp_path / name).read_text(encoding="utf-8").strip() == (
            GENERATED / name).read_text(encoding="utf-8").strip()


def test_training_search_does_not_claim_global_optimization():
    text = (ROOT / "src" / "hamiltonian.py").read_text(encoding="utf-8")
    assert '"""Global min on' not in text
    assert "not a certificate of" in text
    assert "discrete hard-cost" in text


def test_seasonal_export_matches_distributed_manuscript(tmp_path, monkeypatch):
    # Only aggregate stored outputs; do not train or invoke an online solver.
    result_copy = tmp_path / "results"
    shutil.copytree(RESULTS, result_copy)
    generated = tmp_path / "generated"
    generated.mkdir()
    monkeypatch.setattr(seasons, "RESULTS", result_copy)
    monkeypatch.setattr(seasons, "GENERATED", generated)
    with contextlib.redirect_stdout(io.StringIO()):
        assert seasons.main() == 0
    for path in generated.glob("*.tex"):
        assert path.read_text(encoding="utf-8").strip() == (
            GENERATED / path.name).read_text(encoding="utf-8").strip()
