import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from utils import ROOT

_PAGE_BREAK_RE = re.compile(r'w:br[^>]*w:type="page"')


@dataclass
class DocumentChunk:
    source_file_id: str
    section: str
    page: int | None
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentChunk":
        return cls(d["source_file_id"], d["section"], d.get("page"), d["text"])


def _is_heading(para: Paragraph, text: str) -> bool:
    if not text:
        return False
    if len(text) > 120:
        return False
    return any(r.bold for r in para.runs)


def _blocks(doc: DocxDocument):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _page_breaks(block) -> int:
    xml = block._element.xml if hasattr(block, "_element") else block._p.xml
    return xml.count("lastRenderedPageBreak") + len(_PAGE_BREAK_RE.findall(xml))


def _dedupe_cells(cells: list[str]) -> list[str]:
    out: list[str] = []
    for c in cells:
        if not out or c != out[-1]:
            out.append(c)
    return out


def _table_rows(table: Table) -> list[tuple[str | None, str]]:
    rows = [_dedupe_cells([c.text.strip() for c in r.cells]) for r in table.rows]
    rows = [r for r in rows if any(r)]

    if not rows:
        return []

    header_idx = None
    for i, cells in enumerate(rows):
        joined = " ".join(cells).lower()
        if "раздел" in joined or "описание" in joined:
            header_idx = i
            break

    out: list[tuple[str | None, str]] = []
    if header_idx is not None:
        header = rows[header_idx]
        lowered = [c.lower() for c in header]

        def col(name: str) -> int | None:
            for j, h in enumerate(lowered):
                if name in h:
                    return j
            return None

        col_razdel = col("раздел")
        col_opis = col("описание")

        for cells in rows[header_idx + 1:]:
            if not any(cells):
                continue
            label = cells[col_razdel] if (col_razdel is not None and col_razdel < len(cells) and cells[col_razdel]) else None
            if col_opis is not None and col_opis < len(cells) and cells[col_opis]:
                content = cells[col_opis]
            else:
                content = " ".join(c for c in cells if c)
            if not content:
                continue
            out.append((label, content))
    else:
        for cells in rows:
            if not any(cells):
                continue
            if len(cells) >= 2 and cells[0] and len(cells[0]) <= 60:
                out.append((cells[0], " ".join(c for c in cells[1:] if c)))
            else:
                out.append((None, " ".join(c for c in cells if c)))

    return out


def chunk_document(doc: DocxDocument, source_file_id: str, legacy: bool = False) -> list[DocumentChunk]:
    chunks: list[tuple[str, int, str]] = []  # (section, page, text)
    current_section: str | None = None
    page = 1
    total_breaks = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = "\n".join(x for x in buffer if x.strip()).strip()
        buffer = []
        if text:
            chunks.append((current_section or "document", page, text))

    def new_section(label: str) -> None:
        nonlocal current_section
        flush()
        current_section = label
        buffer.append(label)  # keep the heading text inside its own section

    for block in _blocks(doc):
        breaks = _page_breaks(block)
        page += breaks
        total_breaks += breaks
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if _is_heading(block, text):
                new_section(text)
            elif text:
                buffer.append(text)
        else:  # Table
            for label, content in _table_rows(block):
                if label:
                    new_section(label)
                if content:
                    buffer.append(content)

    flush()

    out: list[DocumentChunk] = []
    for section, p, text in chunks:
        if total_breaks == 0 and legacy:
            p = None  # no reliable pagination for a legacy .doc
        out.append(DocumentChunk(source_file_id, section, p, text))

    return out


_SOFFICE_CANDIDATES = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)


def _find_soffice() -> str | None:
    env = os.getenv("SOFFICE")
    if env and os.path.exists(env):
        return env
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    return next((p for p in _SOFFICE_CANDIDATES if os.path.exists(p)), None)


def _file_uri(path: str) -> str:
    return "file:///" + os.path.abspath(path).replace("\\", "/")


def _convert_doc_to_docx(path: str, out_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, stem + ".docx")

    soffice = _find_soffice()
    if soffice:
        last = "no output"
        for attempt in range(3):
            started = time.time()
            with tempfile.TemporaryDirectory(prefix="lo_profile_") as profile:
                cmd = [soffice, f"-env:UserInstallation={_file_uri(profile)}",
                       "--headless", "--norestore", "--convert-to", "docx",
                       "--outdir", out_dir, path]
                result = subprocess.run(cmd, capture_output=True)
            written = os.path.exists(out_path) and os.path.getmtime(out_path) >= started - 1
            if result.returncode == 0 and written:
                return out_path
            detail = (result.stderr or result.stdout or b"").decode("utf-8", "replace").strip()
            last = detail or f"exit {result.returncode}"
            if written:
                return out_path
            time.sleep(1.0 + attempt)
        raise RuntimeError(f"LibreOffice failed to convert {os.path.basename(path)}: {last}")

    textutil = shutil.which("textutil")
    if textutil:
        subprocess.run([textutil, "-convert", "docx", "-output", out_path, path],
                       check=True, capture_output=True)
        return out_path

    raise RuntimeError(
        "Cannot convert .doc: neither LibreOffice (soffice) nor textutil is available. "
        "Install LibreOffice or run on macOS."
    )


def extract_document(path: str, source_file_id: str | None = None, work_dir: str | None = None) -> list[DocumentChunk]:
    source_file_id = source_file_id or os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    legacy = ext == ".doc"

    if ext == ".docx":
        doc = docx.Document(path)
    elif legacy:
        if work_dir is None:
            work_dir = os.path.join(ROOT, "out", "converted")
        os.makedirs(work_dir, exist_ok=True)
        converted = _convert_doc_to_docx(path, work_dir)
        doc = docx.Document(converted)
    else:
        raise ValueError(f"Unsupported file type: {ext} (expected .doc or .docx)")

    return chunk_document(doc, source_file_id, legacy=legacy)


def prepare_chunks(chunks: list[DocumentChunk], guard: bool = True) -> tuple[list[DocumentChunk], list[dict]]:
    if not guard:
        return list(chunks), []
    from guardrails import sanitize_chunks
    clean, hits = sanitize_chunks([c.to_dict() for c in chunks])
    return [DocumentChunk.from_dict(d) for d in clean], hits
