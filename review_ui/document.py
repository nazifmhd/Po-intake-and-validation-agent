"""The source document, with the values that matter marked on it.

Text PDFs are searched directly; scans are located through OCR word boxes when Tesseract is available. Marking
is a reading aid only - if a value can't be located the page is simply shown without the mark.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import streamlit as st

MAX_PREVIEW_PAGES = 5
FLAGGED, NORMAL = (217, 83, 30), (15, 120, 130)


def _amount(v) -> list[str]:
    return [] if v is None else [f"{float(v):,.2f}", f"{float(v):.2f}"]


def marks_for(po: dict, flags: list[str], checks: list[dict]) -> tuple[list[str], list[str]]:
    """-> (values behind a flag, other extracted values)"""
    val = {f: (po.get(f) or {}).get("value") for f in
           ("po_number", "vendor_name", "vendor_tax_id", "payment_terms", "currency")}
    hot: list[str] = []
    if any(f.startswith("vendor_") for f in flags):
        hot += [val["vendor_name"], val["vendor_tax_id"]] if "vendor_tax_id_mismatch" in flags else [val["vendor_name"]]
    for c in checks:
        if c.get("status") == "price_anomaly":
            hot += _amount(c.get("unit_price"))
        elif c.get("status") == "unknown_item" or c.get("sku_mismatch"):
            hot.append(c.get("description"))
    if {"over_threshold", "exceeds_po_limit", "math_mismatch", "math_unverified", "possible_po_split"} & set(flags):
        hot += _amount(po.get("total"))
    if "noncompliant_terms" in flags:
        hot.append(val["payment_terms"])
    if "po_number_conflict" in flags:
        hot.append(val["po_number"])
    rest = [val["po_number"], val["vendor_name"], val["payment_terms"], *_amount(po.get("total")),
            *_amount(po.get("subtotal"))]
    for li in po.get("line_items") or []:
        rest += [li.get("sku"), *_amount(li.get("line_total"))]
    hot = [h for h in dict.fromkeys(hot) if h]
    return hot, [r for r in dict.fromkeys(rest) if r and r not in hot]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _ocr_boxes(img, terms: list[str]) -> dict[str, list[tuple]]:
    import pytesseract
    from rapidfuzz import fuzz

    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 11")
    words = [(_norm(t), d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])
             for i, t in enumerate(d["text"]) if _norm(t)]
    out: dict[str, list[tuple]] = {}
    for term in terms:
        want, n = _norm(term), len(term.split())
        if len(want) < 4:
            continue
        for i in range(len(words) - n + 1):
            run = words[i:i + n]
            got = "".join(w[0] for w in run)
            exact = want == got or (any(ch.isdigit() for ch in want) and want in got and len(got) <= len(want) + 2)
            if exact or (not any(ch.isdigit() for ch in want) and len(want) > 6 and fuzz.ratio(want, got) >= 88):
                out.setdefault(term, []).append((min(w[1] for w in run), min(w[2] for w in run),
                                                 max(w[3] for w in run), max(w[4] for w in run)))
    return out


def _draw(img, boxes: list[tuple], color: tuple, pad: int = 5):
    from PIL import Image, ImageDraw

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x0, y0, x1, y1 in boxes:
        d.rounded_rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), 5, fill=(*color, 46), outline=(*color, 255), width=2)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=64)
def render(path: str, mtime: float, hot: tuple[str, ...], rest: tuple[str, ...]) -> tuple[list[bytes], int, str]:
    """-> (page images, how many flagged values were located, note)"""
    from PIL import Image, ImageOps

    from app.ingestion.loaders import ocr_available

    p, pages, found = Path(path), [], 0

    def finish(img, hot_boxes, rest_boxes):
        img = _draw(_draw(img, rest_boxes, NORMAL), hot_boxes, FLAGGED)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        pages.append(buf.getvalue())

    def mark_by_ocr(img):
        if not (ocr_available() and (hot or rest)):
            return [], [], 0
        try:
            boxes = _ocr_boxes(img, [*hot, *rest])
        except Exception:
            return [], [], 0
        h = [b for t in hot for b in boxes.get(t, [])]
        return h, [b for t in rest for b in boxes.get(t, []) if b not in h], sum(t in boxes for t in hot)

    if p.suffix.lower() == ".pdf":
        import pymupdf as fitz

        with fitz.open(p) as doc:
            if doc.needs_pass:
                return [], 0, "This PDF is password-protected, so it can't be shown."
            for i in range(min(doc.page_count, MAX_PREVIEW_PAGES)):
                page, scale = doc[i], 2.0
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                if page.get_text().strip():
                    def boxes(terms):
                        return [tuple(v * scale for v in r) for t in terms for r in page.search_for(t)]
                    found += sum(bool(page.search_for(t)) for t in hot)
                    finish(img, boxes(hot), boxes(rest))
                else:
                    h, r, n = mark_by_ocr(img)
                    found += n
                    finish(img, h, r)
            more = doc.page_count - MAX_PREVIEW_PAGES
            return pages, found, f"{more} more page(s) in the original" if more > 0 else ""
    with Image.open(p) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) < 1800:
            k = 1800 / max(img.size)
            img = img.resize((int(img.width * k), int(img.height * k)), Image.LANCZOS)
        h, r, found = mark_by_ocr(img)
        finish(img, h, r)
    return pages, found, ""


def show(source_file: str, doc_type: str | None, po: dict, flags: list[str], checks: list[dict], key: str) -> None:
    import pandas as pd

    path = Path(source_file)
    if not path.exists():
        st.warning("The original file is no longer in storage.")
        return
    try:
        if path.suffix.lower() == ".csv":
            st.dataframe(pd.read_csv(path, dtype=str, sep=None, engine="python").fillna(""), width="stretch",
                         hide_index=True)
        else:
            hot, rest = marks_for(po, flags, checks)
            mode = st.segmented_control("Marks on the document", ["Flagged values", "All values", "None"],
                                        default="Flagged values", key=f"marks-{key}", label_visibility="collapsed")
            pages, found, note = render(str(path), path.stat().st_mtime, tuple(hot) if mode != "None" else (),
                                        tuple(rest) if mode == "All values" else ())
            for data in pages:
                st.image(data, width="stretch")
            if note:
                st.caption(note)
            if mode == "Flagged values" and hot and not found:
                st.caption("The flagged values could not be located on the page.")
    except Exception as exc:  # a broken file is exactly what lands here - don't crash the page
        st.error(f"The document can't be shown: {exc}")
    st.download_button("Download original", path.read_bytes(), file_name=path.name.split("_", 1)[-1], key=f"dl-{key}",
                       icon=":material/download:")
