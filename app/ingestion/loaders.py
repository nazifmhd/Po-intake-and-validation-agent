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
    doc_type: str                      # pdf_text | pdf_mixed | pdf_scanned | image | csv
    raw_text: str = ""                 # text layer, OCR text, or CSV rendered as text
    text_source: str = "none"          # text_layer | mixed | ocr | csv | none
    page_images: list[str] = field(default_factory=list)
    page_count: int = 0
    csv_rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vision_images: list[str] = field(default_factory=list)  # pages the model must look at (subset of page_images)
    flags: dict[str, str] = field(default_factory=dict)     # ingestion problems that force review: code -> reason


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


def _save_normalized(img, out_path: Path) -> None:
    from PIL import ImageOps

    img = ImageOps.exif_transpose(img)  # phone photos carry their rotation in EXIF
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    img.save(out_path, format="JPEG", quality=92)  # a noisy photo as PNG would approach the API's size limit


def _is_scanned_page(page, text: str) -> bool:
    """Little text but something drawn on it, or a picture covering most of the page (a scan that carries
    the scanner's own OCR layer). A blank trailing page is not a scan."""
    images = page.get_image_info()
    if len(text.strip()) < MIN_TEXT_CHARS_PER_PAGE:
        return bool(images) or len(page.get_drawings()) > 20
    area = abs(page.rect.width * page.rect.height) or 1
    return any(abs((i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])) / area > 0.6 for i in images)


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
        flags: dict[str, str] = {}
        page_count = doc.page_count
        if page_count > MAX_PAGES:
            flags["truncated_document"] = f"PDF has {page_count} pages; only the first {MAX_PAGES} were processed"
            warnings.append(flags["truncated_document"])

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

        scanned = [_is_scanned_page(doc[i], t) for i, t in enumerate(text_pages)]

        page_images: list[str] = []
        pages_dir.mkdir(parents=True, exist_ok=True)
        # Always render pages: scans need them for vision; text PDFs use them in the review UI
        for i in range(min(page_count, MAX_PAGES)):
            # JPEG for scans (photographic, PNG would be huge), PNG for crisp text pages
            rect = doc[i].rect
            dpi = max(72, min(RENDER_DPI, int(MAX_IMAGE_SIDE * 72 / max(rect.width, rect.height, 1))))
            out = pages_dir / f"page-{i + 1}.{'jpg' if scanned[i] else 'png'}"
            doc[i].get_pixmap(dpi=dpi).save(out)
            page_images.append(str(out))

    vision = [p for p, s in zip(page_images, scanned) if s]
    if not vision:
        if not any(t.strip() for t in text_pages):
            raise IngestionError("empty_document", "PDF has no text and no images on any page")
        text = "\n\n".join(f"--- page {i + 1} ---\n{t}" for i, t in enumerate(text_pages))
        return RawDocument("pdf_text", text, "text_layer", page_images, page_count, warnings=warnings, flags=flags)

    # At least one scanned page. A PO that is half text, half scan must not lose the scanned half.
    ocr = ocr_available()
    if ocr:
        text_pages = [_ocr_image(p) if s and len(t.strip()) < MIN_TEXT_CHARS_PER_PAGE else t
                      for p, s, t in zip(page_images, scanned, text_pages)]
    else:
        warnings.append("Scanned page(s) and no local OCR engine; relying on the vision model only")
    mixed = any(t.strip() and not s for s, t in zip(scanned, text_pages))
    text = ""
    if any(t.strip() for t in text_pages):
        text = "\n\n".join(f"--- page {i + 1} ---\n{t}" for i, t in enumerate(text_pages))
    source = "mixed" if mixed else ("ocr" if text else "none")
    return RawDocument("pdf_mixed" if mixed else "pdf_scanned", text, source, page_images, page_count,
                       warnings=warnings, vision_images=vision, flags=flags)


def load_image(path: Path, pages_dir: Path) -> RawDocument:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.verify()  # detects corrupt files
        pages_dir.mkdir(parents=True, exist_ok=True)
        outs = []
        with Image.open(path) as img:
            frames = getattr(img, "n_frames", 1)  # multi-page TIFF scans
            for i in range(min(frames, MAX_PAGES)):
                img.seek(i)
                out = pages_dir / f"page-{i + 1}.jpg"
                _save_normalized(img, out)  # decoding here also catches truncated files
                outs.append(str(out))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise IngestionError("corrupt_file", f"Cannot read image: {exc}") from exc

    raw_text, source, warnings = "", "none", []
    flags: dict[str, str] = {}
    if frames > MAX_PAGES:
        flags["truncated_document"] = f"Image has {frames} pages; only the first {MAX_PAGES} were processed"
    if ocr_available():
        raw_text = "\n\n".join(_ocr_image(p) for p in outs)
        source = "ocr"
    else:
        warnings.append("Image input and no local OCR engine; relying on the vision model only")
    return RawDocument("image", raw_text, source, outs, frames, warnings=warnings, vision_images=list(outs),
                       flags=flags)


def decode_csv(data: bytes) -> tuple[str, type[csv.Dialect]]:
    """Encoding + delimiter detection, shared by the loader and the batch splitter."""
    text = ""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return text, dialect


def load_csv(path: Path) -> RawDocument:
    try:
        text, dialect = decode_csv(path.read_bytes())
        if not text.strip():
            raise IngestionError("empty_file", "CSV is empty")
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("corrupt_file", f"Cannot parse CSV: {exc}") from exc
    if not rows:
        raise IngestionError("empty_file", "CSV has a header but no rows")
    flags = {}
    ragged = [i for i, r in enumerate(rows, start=2) if None in r]
    if ragged:  # e.g. an unquoted "1,540.00" - the columns after it are shifted and can't be trusted
        flags["malformed_csv"] = f"Row(s) {', '.join(map(str, ragged[:5]))} have more cells than the header"
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in rows]
    return RawDocument("csv", text, "csv", [], 1, csv_rows=rows, flags=flags)


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
