"""Abstract count expands the actual result text, excluding only the draft banner."""
import pytest
from pathlib import Path
import re

from experiments.audit_manuscript_format import (
    abstract_text, highlight_texts, latex_dependencies, pending_result_sources)
from experiments.make_submission_materials import make_docx


def test_abstract_count_expands_result_text_and_preserves_percentages(tmp_path):
    (tmp_path / 'generated').mkdir()
    (tmp_path / 'generated' / 'abstract_result.tex').write_text(
        r'Costs were 1.2\% higher.', encoding='utf-8')
    (tmp_path / 'main.tex').write_text(
        r'\begin{abstract}'
        r'\IfFileExists{alignment_pending.tex}{\input{alignment_pending}}{}'
        r'A hard-cost policy. \input{generated/abstract_result}'
        r'\end{abstract}', encoding='utf-8')
    assert abstract_text(tmp_path) == 'A hard-cost policy. Costs were 1.2% higher.'


def test_abstract_cannot_read_outside_manuscript(tmp_path):
    (tmp_path / 'main.tex').write_text(
        r'\begin{abstract}\input{../outside}\end{abstract}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='unsafe'):
        abstract_text(tmp_path)


def test_dependencies_include_generated_text_figures_and_bibliography(tmp_path):
    (tmp_path / 'piece.tex').write_text(r'\includegraphics[width=2cm]{plot.pdf}', encoding='utf-8')
    (tmp_path / 'main.tex').write_text(
        r'\input{piece}\bibliography{refs}', encoding='utf-8')
    assert {p.name for p in latex_dependencies(tmp_path / 'main.tex', tmp_path.resolve())} == {
        'main.tex', 'piece.tex', 'plot.pdf', 'refs.bib'}


def test_release_check_finds_a_nested_pending_result_placeholder(tmp_path):
    (tmp_path / 'generated').mkdir()
    (tmp_path / 'main.tex').write_text(r'\input{generated/additional}')
    (tmp_path / 'supplement.tex').write_text('Supplement.')
    (tmp_path / 'generated/additional.tex').write_text('% REASSESSMENT_PENDING\nPending.')
    assert pending_result_sources(tmp_path) == ['generated/additional.tex']


def test_highlight_check_ignores_only_title_and_bullet_markers(tmp_path):
    lines = ['A matched tariff.', 'A hard-cost decision.']
    (tmp_path / 'highlights.txt').write_text('\n'.join('- ' + line for line in lines), encoding='utf-8')
    make_docx(lines, tmp_path / 'highlights.docx')
    assert highlight_texts(tmp_path) == lines


def test_highlight_check_detects_outdated_docx(tmp_path):
    (tmp_path / 'highlights.txt').write_text('- Corrected result.', encoding='utf-8')
    make_docx(['Old result.'], tmp_path / 'highlights.docx')
    with pytest.raises(RuntimeError, match='disagree'):
        highlight_texts(tmp_path)


def test_materials_export_preserves_current_highlight_text(tmp_path, monkeypatch):
    from experiments import make_submission_materials as materials
    source = 'A matched simulation benchmark.\n- Stored policies support replay.\nA common hard cost.\n'
    (tmp_path / 'highlights.txt').write_text(source, encoding='utf-8')
    monkeypatch.setattr(materials, 'ROOT', tmp_path)
    assert materials.main() == 0
    assert (tmp_path / 'highlights.txt').read_text(encoding='utf-8') == source
    assert highlight_texts(tmp_path) == [
        'A matched simulation benchmark.', 'Stored policies support replay.', 'A common hard cost.']


def test_submission_manifest_and_flattening_cover_all_generated_inputs():
    root = Path(__file__).resolve().parents[1]
    packager = (root / 'make_submission_package.ps1').read_text(encoding='utf-8-sig')
    required = packager.split('$Required = @(', 1)[1].split('\n)', 1)[0]
    copied = packager.split("foreach ($Name in @('abstract_result.tex'", 1)[1].split(')) {', 1)[0]
    archived = packager.split('$ArchiveFiles = @(', 1)[1].split('\n)', 1)[0]
    for style in ('bibliography_setup.tex', 'elsarticle.cls', 'elsarticle-num-names.bst'):
        assert f"'{style}'" in required
        assert f"'{style}'" in archived
        assert (root / 'manuscript' / style).is_file()
    pending = [root / 'manuscript' / f'{paper}.tex' for paper in ('main', 'supplement')]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        source = path.read_text(encoding='utf-8-sig')
        # Include currently commented pending sections: final packaging must
        # also support their explicit restoration after reconciliation.
        for name in re.findall(r'\\input\{generated/([^}]+)\}', source):
            stem = name.removesuffix('.tex')
            assert f"'generated\\{stem}.tex'" in required, stem
            assert stem == 'abstract_result' or f"'{stem}.tex'" in copied, stem
            assert f"'{stem}.tex'" in archived, stem
            assert f".Replace('generated/{stem}','{stem}')" in packager, stem
            pending.append(root / 'manuscript' / 'generated' / f'{stem}.tex')
    assert 'foreach ($RelativeInclude in $Required)' in packager
    assert '$GeneratedText.Replace("generated/$IncludeStem", $IncludeStem)' in packager
