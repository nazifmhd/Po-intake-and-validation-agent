"""Stage 3 - extraction: RawDocument -> validated PurchaseOrder with per-field confidence + evidence.

Three paths, chosen per document:
  * CSV with a recognised header  -> deterministic column mapping (no LLM needed, confidence 0.95)
  * LLM (Anthropic / OpenAI)      -> text and/or page images, native structured output + Pydantic
  * heuristic (offline fallback)  -> regex over the text layer / OCR text; lower confidence

All paths finish in `finalize()`, which normalizes values and - critically - verifies each
field's evidence actually occurs in the source text. Evidence that cannot be found is treated
as a likely hallucination and the field's confidence is cut, which pushes the PO to review.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional

from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.extraction.prompts import EXTRACTION_SYSTEM, extraction_user_prompt
from app.extraction.schema import Confidenced, LineItem, PurchaseOrder, PurchaseOrderExtraction
from app.ingestion.loaders import RawDocument

logger = logging.getLogger(__name__)

CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "C$": "CAD", "A$": "AUD"}
DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
                "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y")


class ExtractionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- helpers
def parse_number(raw) -> Optional[float]:
    """'$1,234.50' -> 1234.5 ; '1.234,50' -> 1234.5 ; '(12.00)' -> -12.0 ; '' -> None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None
    negative = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s or s in {"-", ".", ","}:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):  # European: 1.234,50
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    try:
        value = float(s)
    except ValueError:
        return None
    return -value if negative else value


def normalize_date(value: Optional[str]) -> tuple[Optional[str], bool]:
    """Returns (normalized_value, parsed_ok)."""
    if not value:
        return value, False
    v = value.strip().rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date().isoformat(), True
        except ValueError:
            continue
    return value, False


def normalize_terms(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    m = re.search(r"net\s*[-]?\s*(\d{1,3})", value, re.I)
    if m:
        return f"Net {int(m.group(1))}"
    if re.search(r"due on receipt|immediate|cod|cash on delivery|prepaid|advance", value, re.I):
        return value.strip()
    return value.strip()


def normalize_currency(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    v = value.strip()
    if v in CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOLS[v]
    m = re.search(r"\b([A-Z]{3})\b", v.upper())
    return m.group(1) if m else v.upper()


# --------------------------------------------------------------------------- CSV path
CSV_HEADER_ALIASES = {
    "po_number": ["po_number", "po number", "po_no", "po #", "po#", "purchase_order", "order_number"],
    "po_date": ["po_date", "po date", "order_date", "date"],
    "vendor_name": ["vendor_name", "vendor", "supplier", "supplier_name"],
    "vendor_tax_id": ["vendor_tax_id", "tax_id", "vat", "tin", "ein"],
    "buyer_department": ["buyer_department", "department", "dept", "cost_center"],
    "currency": ["currency", "ccy"],
    "payment_terms": ["payment_terms", "terms"],
    "delivery_date": ["delivery_date", "deliver_by", "required_by", "need_by"],
    "line_no": ["line_no", "line", "line_number", "#"],
    "sku": ["sku", "item_code", "part_number", "product_code", "item"],
    "description": ["description", "item_description", "desc"],
    "quantity": ["quantity", "qty"],
    "unit": ["unit", "uom", "unit_of_measure"],
    "unit_price": ["unit_price", "price", "unit cost", "unit_cost"],
    "line_total": ["line_total", "amount", "extended_price", "total_price"],
    "subtotal": ["subtotal", "sub_total"],
    "tax": ["tax", "sales_tax", "vat_amount"],
    "shipping": ["shipping", "freight", "delivery_charge"],
    "total": ["total", "grand_total", "po_total", "order_total"],
}


def _map_csv_columns(columns: list[str]) -> dict[str, str]:
    norm = {c.strip().lower().replace("-", "_"): c for c in columns}
    mapping = {}
    for field, aliases in CSV_HEADER_ALIASES.items():
        for a in aliases:
            if a in norm:
                mapping[field] = norm[a]
                break
    return mapping


def csv_is_structured(rows: list[dict]) -> bool:
    if not rows:
        return False
    m = _map_csv_columns(list(rows[0].keys()))
    return {"quantity", "unit_price"} <= m.keys() and ("description" in m or "sku" in m)


def extract_from_csv(rows: list[dict]) -> PurchaseOrderExtraction:
    m = _map_csv_columns(list(rows[0].keys()))
    first = rows[0]

    def header(field: str) -> Confidenced:
        col = m.get(field)
        if not col:
            return Confidenced.missing()
        values = [r.get(col, "").strip() for r in rows if r.get(col, "").strip()]
        if not values:
            return Confidenced.missing()
        distinct = set(values)
        conf = 0.95 if len(distinct) == 1 else 0.5  # conflicting header values across rows
        return Confidenced(value=values[0], confidence=conf, evidence=f"{col}={values[0]}")

    def num(row: dict, field: str) -> Optional[float]:
        col = m.get(field)
        return parse_number(row.get(col)) if col else None

    items = []
    for i, r in enumerate(rows, start=1):
        if not any(r.get(m[f], "") for f in ("description", "sku", "quantity") if f in m):
            continue
        ln = num(r, "line_no")
        items.append(LineItem(
            line_no=int(ln) if ln is not None else i,
            sku=r.get(m["sku"]) or None if "sku" in m else None,
            description=r.get(m["description"]) or None if "description" in m else None,
            quantity=num(r, "quantity"), unit=(r.get(m["unit"]) or None) if "unit" in m else None,
            unit_price=num(r, "unit_price"), line_total=num(r, "line_total"),
            confidence=0.95, evidence=", ".join(f"{k}={v}" for k, v in r.items() if v)[:300],
        ))

    def total_field(field: str) -> Optional[float]:
        col = m.get(field)
        if not col:
            return None
        vals = [parse_number(r.get(col)) for r in rows if (r.get(col) or "").strip()]
        return vals[0] if vals else None

    total = total_field("total")
    return PurchaseOrderExtraction(
        po_number=header("po_number"), po_date=header("po_date"), vendor_name=header("vendor_name"),
        vendor_tax_id=header("vendor_tax_id"), buyer_department=header("buyer_department"),
        currency=header("currency"), line_items=items,
        subtotal=total_field("subtotal"), tax=total_field("tax"), shipping=total_field("shipping"),
        total=total, totals_confidence=0.95 if total is not None else 0.0,
        totals_evidence=f"{m['total']}={first.get(m['total'])}" if total is not None and "total" in m else None,
        payment_terms=header("payment_terms"), delivery_date=header("delivery_date"),
    )


# --------------------------------------------------------------------------- heuristic path
_MONEY = r"[$€£]?\s?-?[\d,]+\.\d{2}"
_HEADER_PATTERNS = {
    "po_number": r"(?:P\.?O\.?|Purchase\s+Order)\s*(?:Number|No\.|No\b|#)\s*[:#]?\s*([A-Z0-9\-/]*\d[A-Z0-9\-/]*)",
    "po_date": r"(?:PO\s+Date|Order\s+Date|Date\s+Issued|Issue\s+Date)\s*[:]\s*([^\n]+)",
    "vendor_name": r"(?:Vendor|Supplier)(?:\s+Name)?\s*[:]\s*([^\n]+)",
    "vendor_tax_id": r"(?:Tax\s*ID|VAT(?:\s*No\.?)?|TIN|EIN)\s*[:#]?\s*([A-Z]{0,3}[\-\s]?[0-9][0-9\-]{4,})",
    "buyer_department": r"(?:Buyer\s+Department|Department|Dept\.?)\s*[:]\s*([^\n]+)",
    "currency": r"Currency\s*[:]\s*([A-Z]{3})",
    "payment_terms": r"(?:Payment\s+Terms|Terms)\s*[:]\s*([^\n]+)",
    "delivery_date": r"(?:Delivery\s+Date|Deliver\s+By|Required\s+By|Need\s+By)\s*[:]\s*([^\n]+)",
}
_TOTAL_PATTERNS = {
    "subtotal": rf"^\s*Sub\s*-?\s*total\s*[:]?\s*({_MONEY})",
    "tax": rf"^\s*(?:Sales\s+)?Tax(?:\s*\([^)]*\))?\s*[:]?\s*({_MONEY})",
    "shipping": rf"^\s*(?:Shipping|Freight|Delivery)(?:\s*&\s*Handling)?\s*[:]?\s*({_MONEY})",
    "total": rf"^\s*(?:Grand\s+Total|Total\s+Due|PO\s+Total|Total)\s*(?:\([A-Z]{{3}}\))?\s*[:]?\s*({_MONEY})",
}
_LINE_RE = re.compile(
    rf"^\s*(\d{{1,3}})\s+([A-Z0-9][A-Z0-9\-]{{2,}})\s+(.+?)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]{{1,5}})\s+({_MONEY})\s+({_MONEY})\s*$"
)


def extract_heuristic(doc: RawDocument) -> PurchaseOrderExtraction:
    text = doc.raw_text or ""
    base = 0.88 if doc.text_source == "text_layer" else 0.6  # OCR text is noisier

    def stop_at_next_label(v: str) -> str:
        # two-column layouts put "Label: value  Other Label: value" on one line
        return re.split(r"\s{2,}|\s(?=[A-Z][A-Za-z ]{2,20}:\s)", v.strip())[0].strip()

    header: dict[str, Confidenced] = {}
    for field, pattern in _HEADER_PATTERNS.items():
        m = re.search(pattern, text, re.I)
        if m:
            value = stop_at_next_label(m.group(1))
            header[field] = Confidenced(value=value or None, confidence=base if value else 0.0,
                                        evidence=m.group(0).strip()[:200])
        else:
            header[field] = Confidenced.missing()

    if not header["currency"].present:
        for sym, code in CURRENCY_SYMBOLS.items():
            if sym in text:
                header["currency"] = Confidenced(value=code, confidence=0.6, evidence=sym)
                break

    items = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            items.append(LineItem(
                line_no=int(m.group(1)), sku=m.group(2), description=m.group(3).strip(),
                quantity=float(m.group(4)), unit=m.group(5).upper(),
                unit_price=parse_number(m.group(6)), line_total=parse_number(m.group(7)),
                confidence=base, evidence=line.strip(),
            ))

    totals: dict[str, Optional[float]] = {}
    total_ev = None
    for field, pattern in _TOTAL_PATTERNS.items():
        m = re.search(pattern, text, re.I | re.M)
        totals[field] = parse_number(m.group(1)) if m else None
        if field == "total" and m:
            total_ev = m.group(0).strip()

    return PurchaseOrderExtraction(
        **header, line_items=items, **totals,
        totals_confidence=base if totals["total"] is not None else 0.0, totals_evidence=total_ev,
    )


# --------------------------------------------------------------------------- LLM path
def extract_llm(doc: RawDocument, settings: Settings) -> PurchaseOrderExtraction:
    from app.llm import LLMClient

    # Text PDFs: the text layer is exact, images add little. Scans/images: vision is primary,
    # OCR text (if any) is passed as secondary evidence.
    images = doc.page_images if doc.doc_type in ("pdf_scanned", "image") else []
    if not images and not doc.raw_text.strip():
        raise ExtractionError("empty_document", "Nothing to extract: no text and no page images")
    client = LLMClient(settings)
    return client.structured(
        system=EXTRACTION_SYSTEM,
        text=extraction_user_prompt(doc.doc_type, doc.raw_text, doc.text_source, bool(images)),
        response_model=PurchaseOrderExtraction,
        image_paths=images,
    )


# --------------------------------------------------------------------------- finalize
def _evidence_found(evidence: str, source: str) -> bool:
    if not evidence or not source:
        return True
    ev = " ".join(evidence.split()).lower()
    src = " ".join(source.split()).lower()
    if ev in src:
        return True
    return fuzz.partial_ratio(ev, src) >= 85


def finalize(ext: PurchaseOrderExtraction, doc: RawDocument, method: str) -> PurchaseOrder:
    po = PurchaseOrder(**ext.model_dump(), extraction_method=method)
    warnings: list[str] = list(doc.warnings)

    # --- normalization (values only; vendor name is deliberately NOT normalized) ---
    for f in ("po_date", "delivery_date"):
        c: Confidenced = getattr(po, f)
        if c.present:
            norm, ok = normalize_date(c.value)
            c.value = norm
            if not ok:
                warnings.append(f"{f} '{c.value}' is not an unambiguous date")
                c.confidence = min(c.confidence, 0.6)
    if po.payment_terms.present:
        po.payment_terms.value = normalize_terms(po.payment_terms.value)
    if po.currency.present:
        po.currency.value = normalize_currency(po.currency.value)
    for c in (po.po_number, po.vendor_name, po.vendor_tax_id, po.buyer_department):
        if c.present:
            c.value = " ".join(str(c.value).split())
    for li in po.line_items:
        if li.sku:
            li.sku = li.sku.strip().upper()
        if li.unit:
            li.unit = li.unit.strip().upper()

    # --- anti-hallucination: evidence must exist in the source text ---
    # (deterministic CSV mapping can't hallucinate; its evidence is a column reference, not a quote)
    can_verify = (doc.text_source in ("text_layer", "csv") and len(doc.raw_text) > 50
                  and method != "csv_columns")
    if can_verify:
        for f in PurchaseOrder.HEADER_FIELDS:
            c = getattr(po, f)
            if c.present and c.evidence and not _evidence_found(c.evidence, doc.raw_text):
                c.confidence = round(c.confidence * 0.5, 3)
                warnings.append(f"{f}: evidence not found in source text (possible hallucination)")
        for li in po.line_items:
            if li.evidence and not _evidence_found(li.evidence, doc.raw_text):
                li.confidence = round(li.confidence * 0.5, 3)
                warnings.append(f"line {li.line_no}: evidence not found in source text")
        if po.total is not None and po.totals_evidence and not _evidence_found(po.totals_evidence, doc.raw_text):
            po.totals_confidence = round(po.totals_confidence * 0.5, 3)
            warnings.append("total: evidence not found in source text")

    # --- a present value with zero confidence / a null value with high confidence are both suspect ---
    for f in PurchaseOrder.HEADER_FIELDS:
        c = getattr(po, f)
        if not c.present and c.confidence > 0.2:
            c.confidence = 0.0
    if not po.line_items:
        warnings.append("No line items were extracted")
    if po.total is None:
        po.totals_confidence = 0.0

    confs = [c for c in po.field_confidences().values()]
    po.overall_confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
    po.extraction_warnings = warnings
    return po


def extract(doc: RawDocument, settings: Settings | None = None) -> PurchaseOrder:
    settings = settings or get_settings()
    if doc.doc_type == "csv" and csv_is_structured(doc.csv_rows):
        return finalize(extract_from_csv(doc.csv_rows), doc, "csv_columns")
    if settings.llm_provider in ("anthropic", "openai"):
        return finalize(extract_llm(doc, settings), doc, f"llm:{settings.llm_provider}")
    if not doc.raw_text.strip():
        raise ExtractionError(
            "no_text_for_offline_extraction",
            "Offline (heuristic) extraction needs a text layer or OCR text; this document has neither. "
            "Configure an LLM provider with vision to process scans.",
        )
    return finalize(extract_heuristic(doc), doc, "heuristic")
