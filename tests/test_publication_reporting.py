"""Publication renderers must preserve both estimates and their interpretation."""
from pathlib import Path
import contextlib
import io
import shutil

import pandas as pd
import pytest

from experiments import analyze_comparative_reference as reference
from experiments import analyze_comparative_scenarios as scenarios
from experiments import analyze_seasonal_sensitivity as seasons
from experiments import analyze_confirmatory as confirmatory
from experiments.manuscript_checks import enabled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
GENERATED = ROOT / "manuscript" / "generated"


def test_abstract_export_matches_distributed_manuscript():
    text = confirmatory.render_abstract_result(
        pd.read_csv(RESULTS / "confirmatory_method_summary.csv"),
        pd.read_csv(RESULTS / "confirmatory_pairwise.csv"))
    assert r"1.2\% higher, 2.3\% higher, and 1.4\% higher" in text
    assert 'primary two-stage stochastic MIQP' in text
    if enabled():
        assert text.strip() == (GENERATED / "abstract_result.tex").read_text(
            encoding="utf-8").strip()


def test_abstract_cost_and_latency_use_same_comparator():
    methods = pd.read_csv(RESULTS / "confirmatory_method_summary.csv")
    pairs = pd.read_csv(RESULTS / "confirmatory_pairwise.csv")
    cm = methods.method == "deterministic_exact_band_miqp"
    methods.loc[cm, "mean_common_cost"] = 1.0
    methods.loc[cm, "mean_latency_mean_ms"] = 9876.5
    stochastic = methods.method == "stochastic_two_stage_exact_band_miqp"
    methods.loc[stochastic, "mean_latency_mean_ms"] = 4321.0
    text = confirmatory.render_abstract_result(methods, pairs)
    assert r"1.2\% higher, 2.3\% higher, and 1.4\% higher" in text
    assert "primary two-stage stochastic MIQP" in text
    assert "4321.0 ms for this single-threaded stochastic MIQP" in text
    assert "9876.5" not in text and "conditional-mean" not in text


def test_exploratory_reference_reports_bounds_not_established_superiority(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "GENERATED", tmp_path)
    methods = pd.read_csv(RESULTS / "comparative_reference_method_summary.csv")
    pairs = pd.read_csv(RESULTS / "comparative_reference_pairwise.csv")
    pairs["one_sided_95_upper"] = 1.0
    reference._write_latex(methods, pairs)
    for name in ("comparative_reference.tex", "comparative_reference_supplement.tex"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "Exploratory" in text
        assert "established" not in text and "Superiority" not in text
    assert "No reference-comparison anchor had a one-sided upper bound below" in (
        tmp_path / "comparative_reference.tex").read_text(encoding="utf-8")


def test_compute_time_payback_is_not_economic_break_even(seasonal_generated):
    text = (seasonal_generated / "supplement_results.tex").read_text(encoding="utf-8")
    assert "Compute-time payback" in text
    assert "not an economic break-even period" in text
    assert "Break-even days" not in text


@pytest.fixture
def scenario_frames():
    return [pd.read_csv(RESULTS / f"comparative_scenario_{name}.csv") for name in
            ("setting_summary", "stream_summary", "solver_summary", "pairwise")]


def test_scenario_export_matches_distributed_manuscript(tmp_path, monkeypatch,
                                                       scenario_frames):
    monkeypatch.setattr(scenarios, "GENERATED", tmp_path)
    scenarios._write_latex(*scenario_frames)
    for name in ("comparative_scenarios.tex", "comparative_scenario_supplement.tex"):
        text = (tmp_path / name).read_text(encoding="utf-8").strip()
        assert r'\begin{table}' in text
        if enabled():
            assert text == (GENERATED / name).read_text(encoding="utf-8").strip()


def test_scenario_gap_prose_tracks_availability_not_only_mean(tmp_path, monkeypatch,
                                                            scenario_frames):
    settings, streams, solvers, pairs = scenario_frames
    solvers.loc[(solvers.fee == 80) & (solvers.time_limit_s == 5),
                "finite_gap_fraction"] = 0.125
    solvers.loc[(solvers.fee == 80) & (solvers.time_limit_s == 15),
                "finite_gap_fraction"] = 0.875
    monkeypatch.setattr(scenarios, "GENERATED", tmp_path)
    scenarios._write_latex(settings, streams, solvers, pairs)
    text = (tmp_path / "comparative_scenarios.tex").read_text(encoding="utf-8")
    assert r"12.5\% and 87.5\% of calls" in text
    assert "conditional means" in text
    assert "do not measure matched solver progress" in text
    assert "availability and closed-loop states differ" in text


def test_reference_export_matches_distributed_manuscript(tmp_path, monkeypatch):
    monkeypatch.setattr(reference, "GENERATED", tmp_path)
    reference._write_latex(
        pd.read_csv(RESULTS / "comparative_reference_method_summary.csv"),
        pd.read_csv(RESULTS / "comparative_reference_pairwise.csv"))
    for name in ("comparative_reference.tex", "comparative_reference_supplement.tex"):
        text = (tmp_path / name).read_text(encoding="utf-8").strip()
        assert r'\begin{table}' in text
        if enabled():
            assert text == (GENERATED / name).read_text(encoding="utf-8").strip()


def test_training_search_does_not_claim_global_optimization():
    text = (ROOT / "src" / "hamiltonian.py").read_text(encoding="utf-8")
    assert '"""Global min on' not in text
    assert "not a certificate of" in text
    assert "discrete hard-cost" in text


@pytest.fixture
def seasonal_generated(tmp_path, monkeypatch):
    # Only aggregate stored outputs; do not train or invoke an online solver.
    result_copy = tmp_path / "results"
    shutil.copytree(RESULTS, result_copy)
    generated = tmp_path / "generated"
    generated.mkdir()
    monkeypatch.setattr(seasons, "RESULTS", result_copy)
    monkeypatch.setattr(seasons, "GENERATED", generated)
    with contextlib.redirect_stdout(io.StringIO()):
        assert seasons.main() == 0
    return generated


def test_seasonal_export_matches_distributed_manuscript(seasonal_generated):
    generated = seasonal_generated
    assert (generated / 'factorial_results.tex').is_file()
    assert (generated / 'supplement_results.tex').is_file()
    for path in generated.glob("*.tex"):
        text = path.read_text(encoding='utf-8').strip()
        assert text
        if enabled():
            assert text == (GENERATED / path.name).read_text(encoding="utf-8").strip()
