"""Create the editable Elsevier highlights files from final results."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"


def make_docx(lines, path: Path) -> None:
    paragraphs = ["Highlights"] + [f"• {line}" for line in lines]
    body = "".join(
        f"<w:p><w:r><w:t>{escape(text)}</w:t></w:r></w:p>"
        for text in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>' + body
        + '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
          '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
          'w:left="1440"/></w:sectPr></w:body></w:document>')
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            '</Types>'),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            '</Relationships>'),
        "word/document.xml": document,
        "word/_rels/document.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>Manuscript highlights</dc:title>'
            '<dc:creator>Yeoneung Kim</dc:creator>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            '</cp:coreProperties>'),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            '<Application>Microsoft Office Word</Application></Properties>'),
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content.encode("utf-8"))


def main() -> int:
    lines = [line.strip().removeprefix('- ').strip()
             for line in (ROOT / "highlights.txt").read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if not 3 <= len(lines) <= 5:
        raise ValueError("Elsevier highlights require three to five entries")
    too_long = [line for line in lines if len(line) > 85]
    if too_long:
        raise ValueError(f"Elsevier highlight exceeds 85 characters: {too_long}")
    make_docx(lines, ROOT / "highlights.docx")
    print(json.dumps({"highlights": lines,
                      "character_counts": [len(x) for x in lines]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
