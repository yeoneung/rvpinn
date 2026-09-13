"""Check local submission constraints, not certify the live journal guide.

The 30-page cap is the author's explicit requirement. Abstract/highlight
limits retain this project's submission targets. Source/PDF and numerical
review must still be completed separately.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def abstract_text(manuscript):
    root = Path(manuscript).resolve()
    source = (root / 'main.tex').read_text(encoding='utf-8-sig')
    text = source.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    # Count the intended abstract separately from the conspicuous draft banner.
    text = text.replace(r'\IfFileExists{alignment_pending.tex}{\input{alignment_pending}}{}', '')

    def expand(value, stack):
        def include(match):
            path = (root / match[1]).with_suffix('.tex').resolve()
            if root not in path.parents or path in stack:
                raise RuntimeError('unsafe or recursive abstract include')
            return expand(path.read_text(encoding='utf-8-sig'), stack + (path,))
        return re.sub(r'\\input\{([^}]+)\}', include, value)

    text = expand(text, ())
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    text = re.sub(r'\\([%&#_])', r'\1', text)
    text = re.sub(r'\\[A-Za-z@]+\*?(?:\[[^]]*\])?', '', text)
    return ' '.join(text.replace('{', '').replace('}', '').replace('~', ' ').split())


def highlight_texts(root):
    plain = [line.strip().removeprefix('- ').removeprefix('* ')
             for line in (Path(root) / 'highlights.txt').read_text(encoding='utf-8-sig').splitlines()
             if line.strip()]
    with zipfile.ZipFile(Path(root) / 'highlights.docx') as archive:
        document = ET.fromstring(archive.read('word/document.xml'))
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    docx = []
    for paragraph in document.findall('.//w:p', ns):
        value = ''.join(node.text or '' for node in paragraph.findall('.//w:t', ns)).strip()
        if value and value != 'Highlights':
            docx.append(value.removeprefix('- ').removeprefix('* ').removeprefix('\u2022 '))
    if plain != docx:
        raise RuntimeError('plain-text and DOCX highlights disagree')
    return plain


def latex_dependencies(path, root, seen=None):
    seen = set() if seen is None else seen
    path = path.resolve()
    if root not in path.parents:
        raise RuntimeError('manuscript dependency leaves its source directory')
    if path in seen:
        return seen
    seen.add(path)
    source = path.read_text(encoding='utf-8-sig')
    source = re.sub(r'(?<!\\)%[^\n]*', '', source)
    for include in re.findall(r'\\input\{([^}]+)\}', source):
        dependency = (root / include).with_suffix('.tex')
        if include == 'alignment_pending' and not dependency.exists():
            continue
        latex_dependencies(dependency, root, seen)
    for include in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source):
        seen.add((root / include).with_suffix('.pdf').resolve())
    for include in re.findall(r'\\bibliography\{([^}]+)\}', source):
        seen.add((root / include).with_suffix('.bib').resolve())
    return seen


def pending_result_sources(manuscript):
    """A generated placeholder must remain visible to the release gate."""
    root = Path(manuscript).resolve()
    sources = set()
    for name in ('main', 'supplement'):
        sources.update(latex_dependencies(root / f'{name}.tex', root))
    return sorted(path.relative_to(root).as_posix() for path in sources
                  if path.suffix == '.tex'
                  and 'REASSESSMENT_PENDING' in path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manuscript', type=Path, default=ROOT / 'manuscript')
    parser.add_argument('--materials', type=Path, default=ROOT)
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manuscript = args.manuscript.resolve()
    errors, warnings = [], []
    text = abstract_text(manuscript)
    count = len(text.split())
    if count > 250:
        errors.append(f'abstract has {count} whitespace-delimited words; target is at most 250')
    source = (manuscript / 'main.tex').read_text(encoding='utf-8-sig')
    keywords = source.split(r'\begin{keyword}', 1)[1].split(r'\end{keyword}', 1)[0].split(r'\sep')
    if not 1 <= len(keywords) <= 7:
        errors.append('expected one to seven keywords (SMPT author guide)')
    highlights = highlight_texts(args.materials)
    if not 3 <= len(highlights) <= 5 or any(len(value) > 85 for value in highlights):
        errors.append('expected three to five highlights of at most 85 characters each')
    pdfs = {}
    for name in ('main', 'supplement'):
        pdf = manuscript / f'{name}.pdf'
        info = subprocess.run(['pdfinfo', str(pdf)], check=True, capture_output=True,
                              text=True, encoding='utf-8', errors='replace')
        pages = int(re.search(r'^Pages:\s+(\d+)', info.stdout, re.MULTILINE)[1])
        fonts = subprocess.run(['pdffonts', str(pdf)], check=True, capture_output=True,
                               text=True, encoding='utf-8', errors='replace')
        type3 = bool(re.search(r'\bType\s+3\b', fonts.stdout))
        if type3:
            warnings.append(f'{name}.pdf contains Type 3 fonts; regenerate remaining old figures')
        if name == 'main' and pages > 30:
            errors.append(f'main.pdf has {pages} pages; author requires at most 30')
        log = (manuscript / f'{name}.log').read_text(encoding='utf-8', errors='replace')
        if re.search(r'Overfull|undefined references|undefined citations|multiply defined|LaTeX Error'
                     r'|Citation .* undefined|Reference .* undefined', log):
            errors.append(f'{name}.log has unresolved layout/reference diagnostics')
        dependencies = latex_dependencies(manuscript / f'{name}.tex', manuscript)
        stale = [str(path.relative_to(manuscript)) for path in dependencies
                 if not path.exists() or pdf.stat().st_mtime < path.stat().st_mtime]
        if stale:
            errors.append(f'{name}.pdf has newer or missing source dependencies: {stale}')
        pdfs[name] = {'pages': pages, 'type3_fonts': type3,
                      'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()}
    pending = (manuscript / 'alignment_pending.tex').exists()
    if pending:
        warnings.append('numerical reconciliation banner is present; not a release')
    for name in pending_result_sources(manuscript):
        warnings.append(f'{name} contains incomplete-result or release placeholders')
    cover = (args.materials / 'cover_letter.tex').read_text(encoding='utf-8-sig')
    if 'Working draft:' in cover or 'not ready for submission' in cover:
        warnings.append('the cover letter still identifies an incomplete release')
    if args.release:
        errors.extend(warnings)
    report = {'status': 'format_check_failed' if errors else 'format_targets_checked_not_submission_certified',
              'document_checks_passed': not errors and not warnings,
              'journal_submission_status': 'not assessed by this document checker',
              'abstract_words_excluding_draft_banner': count,
              'word_count_convention': 'whitespace-delimited words after expanding the abstract source',
              'keywords': len(keywords), 'highlight_characters': list(map(len, highlights)),
              'pdfs': pdfs, 'errors': errors, 'warnings': warnings}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
