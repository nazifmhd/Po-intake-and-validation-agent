"""Stage 2 - ingestion & normalization.

Detect the file type (text PDF / scanned PDF / image / CSV), extract text and/or page images,
and produce one common `RawDocument`. Every failure raises an `IngestionError` with a
machine-readable code so the graph can route the item to review with a precise reason.
"""
from __future__ import annotations

import csv
import io
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MIN_TEXT_CHARS_PER_PAGE = 40   # below this a PDF page is treated as a scan
MAX_IMAGE_SIDE = 2000          # vision models don't need more; keeps payloads small
RENDER_DPI = 200
MAX_PAGES = 20


class IngestionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class RawDocument:
    doc_type: str                      # pdf_text | pdf_scanned | image | csv
    raw_text: str = ""                 # text layer, OCR text, or CSV rendered as text
    text_source: str = "none"          # text_layer | ocr | csv | none
    page_images: list[str] = field(default_factory=list)
    page_count: int = 0
    csv_rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ocr_available() -> bool:
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def _ocr_image(path: str) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as img:
        try:
            # Orientation & script detection handles rotated scans
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
            rotate = int(osd.get("rotate", 0))
            if rotate:
                img = img.rotate(-rotate, expand=True)
        except Exception:  # OSD fails on sparse pages; plain OCR still works
            pass
        return pytesseract.image_to_string(img, config="--psm 6")


def _normalize_image(src: Path | io.BytesIO, out_path: Path) -> None:
    from PIL import Image, ImageOps

    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)  # phone photos carry their rotation in EXIF
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        img.save(out_path, format="PNG", optimize=True)


def load_pdf(path: Path, pages_dir: Path) -> RawDocument:
    import pymupdf as fitz

    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise IngestionError("corrupt_file", f"Cannot open PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:  # owner-password-only PDFs open fine; user-password ones don't
            raise IngestionError("locked_file", "PDF is password-protected")
        if doc.page_count == 0:
            raise IngestionError("corrupt_file", "PDF has no pages")
        warnings: list[str] = []
        page_count = doc.page_count
        if page_count > MAX_PAGES:
            warnings.append(f"PDF has {page_count} pages; only the first {MAX_PAGES} were processed")

        # Text layer: pdfplumber keeps row layout (one table row per line), PyMuPDF is the fallback
        text_pages: list[str] = []
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:MAX_PAGES]:
                    text_pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("pdfplumber failed, falling back to PyMuPDF text", extra={"error": str(exc)})
            text_pages = [doc[i].get_text("text") for i in range(min(page_count, MAX_PAGES))]

        scanned = [len(t.strip()) < MIN_TEXT_CHARS_PER_PAGE for t in text_pages]
        is_scan = sum(scanned) > len(scanned) / 2

        page_images: list[str] = []
        pages_dir.mkdir(parents=True, exist_ok=True)
        # Always render pages: scans need them for vision; text PDFs use them in the review UI
        for i in range(min(page_count, MAX_PAGES)):
            pix = doc[i].get_pixmap(dpi=RENDER_DPI)
            out = pages_dir / f"page-{i + 1}.png"
            pix.save(out)
            if max(pix.width, pix.height) > MAX_IMAGE_SIDE:
                _normalize_image(out, out)
            page_images.append(str(out))

    if not is_scan:
        text = "\n\n".join(f"--- page {i + 1} ---\n{t}" for i, t in enumerate(text_pages))
        return RawDocument("pdf_text", text, "text_layer", page_images, page_count, warnings=warnings)

    raw_text, source = "", "none"
    if ocr_available():
        raw_text = "\n\n".join(f"--- page {i + 1} ---\n{_ocr_image(p)}" for i, p in enumerate(page_images))
        source = "ocr"
    else:
        warnings.append("Scanned PDF and no local OCR engine; relying on the vision model only")
    return RawDocument("pdf_scanned", raw_text, source, page_images, page_count, warnings=warnings)


def load_image(path: Path, pages_dir: Path) -> RawDocument:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.verify()  # detects truncated / corrupt files
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise IngestionError("corrupt_file", f"Cannot read image: {exc}") from exc

    pages_dir.mkdir(parents=True, exist_ok=True)
    out = pages_dir / "page-1.png"
    _normalize_image(path, out)

    raw_text, source, warnings = "", "none", []
    if ocr_available():
        raw_text = _ocr_image(str(out))
        source = "ocr"
    else:
        warnings.append("Image input and no local OCR engine; relying on the vision model only")
    return RawDocument("image", raw_text, source, [str(out)], 1, warnings=warnings)


def load_csv(path: Path) -> RawDocument:
    try:
        data = path.read_bytes()
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None or not text.strip():
            raise IngestionError("empty_file", "CSV is empty")
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("corrupt_file", f"Cannot parse CSV: {exc}") from exc
    if not rows:
        raise IngestionError("empty_file", "CSV has a header but no rows")
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]
    return RawDocument("csv", text, "csv", [], 1, csv_rows=rows)


def load_document(path: str | Path, pages_dir: str | Path) -> RawDocument:
    path, pages_dir = Path(path), Path(pages_dir)
    if not path.exists():
        raise IngestionError("file_missing", f"File not found: {path}")
    if path.stat().st_size == 0:
        raise IngestionError("empty_file", "File is empty (0 bytes)")

    ext = path.suffix.lower()
    head = path.read_bytes()[:8]
    if ext == ".pdf" or head.startswith(b"%PDF"):
        return load_pdf(path, pages_dir)
    if ext in IMAGE_EXTS:
        return load_image(path, pages_dir)
    if ext == ".csv":
        return load_csv(path)
    raise IngestionError("unsupported_type", f"Unsupported file type: {ext or 'unknown'}")
