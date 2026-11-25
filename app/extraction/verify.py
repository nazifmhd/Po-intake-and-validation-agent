"""Anti-hallucination: can every extracted value be traced back to the document?

Two links are checked, because a model can break either one:
    value -> evidence    the snippet it quotes must actually contain the value it reports
    evidence -> source   the snippet must actually occur in the document

Text documents are checked strictly. Scans are checked against OCR text (when OCR is available) only for the
fields a wrong auto-approval would hinge on - vendor, PO number, total - and tolerantly, because OCR is noisy.
A field that fails keeps its value but loses half its confidence, which puts the PO in review.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from app.extraction.schema import Confidenced, LineItem, PurchaseOrder
from app.ingestion.loaders import RawDocument

NO_EVIDENCE_CAP = 0.75  # a value the model offers no evidence for is never good enough to auto-approve
_SYMBOLS = {"USD": ("$", "us dollar", "u.s. dollar"), "EUR": ("€", "euro"), "GBP": ("£", "pound", "sterling"),
            "JPY": ("¥", "yen"), "INR": ("₹", "rupee"), "CAD": ("c$", "ca$"), "AUD": ("a$", "au$")}
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def _flat(s: str) -> str:
    return " ".join(s.split()).lower()


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def numbers_in(text: str) -> set[float]:
    from app.extraction.extractor import parse_number

    out = set()
    for token in re.findall(r"\d[\d,.]*", text):
        token = token.rstrip(".,")
        value = parse_number(token)
        if value is not None:
            out.add(round(value, 4))
    return out


def has_number(value: float, numbers: set[float]) -> bool:
    return any(abs(abs(value) - n) < 0.005 for n in numbers)


def snippet_in(snippet: str, source: str, threshold: float = 85) -> bool:
    ev, src = _flat(snippet), _flat(source)
    return ev in src or fuzz.partial_ratio(ev, src) >= threshold


def _date_in(iso: str, text: str) -> bool:
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", iso)
    if not m:
        return _flat(iso) in _flat(text)
    year, month, day = (int(g) for g in m.groups())
    ints = [int(n) for n in re.findall(r"\d+", text)]
    month_ok = month in ints or _MONTHS[month - 1] in text.lower()
    return day in ints and month_ok and (year in ints or year % 100 in ints)


def value_in(field: str, value: str, text: str, threshold: float = 88) -> bool:
    if field in ("po_date", "delivery_date"):
        return _date_in(value, text)
    if field in ("po_number", "vendor_tax_id"):
        return _alnum(value) in _alnum(text)
    if field == "currency":
        low = text.lower()
        return value.lower() in low or any(s in low for s in _SYMBOLS.get(value.upper(), ()))
    if field == "payment_terms":
        m = re.fullmatch(r"net (\d+)", value.lower())
        if m:
            return m.group(1) in re.findall(r"\d+", text)
    return fuzz.partial_ratio(_flat(value), _flat(text)) >= threshold if len(_flat(value)) <= len(_flat(text)) \
        else fuzz.ratio(_flat(value), _flat(text)) >= threshold


def verify_grounding(po: PurchaseOrder, doc: RawDocument) -> list[str]:
    source = doc.raw_text if len(doc.raw_text.strip()) > 50 else ""
    if doc.text_source in ("text_layer", "csv") and source:
        return _verify_text(po, source)
    return _verify_scan(po, source if doc.text_source in ("ocr", "mixed") else "")


def _halve(c) -> None:
    c.confidence = round(c.confidence * 0.5, 3)


def _verify_text(po: PurchaseOrder, source: str) -> list[str]:
    warnings = []
    for f in PurchaseOrder.HEADER_FIELDS:
        c: Confidenced = getattr(po, f)
        if not c.present:
            continue
        if not c.evidence:
            c.confidence = min(c.confidence, NO_EVIDENCE_CAP) if value_in(f, c.value, source) else 0.3
            warnings.append(f"{f}: no evidence was given for '{c.value}'")
        elif not snippet_in(c.evidence, source):
            _halve(c)
            warnings.append(f"{f}: evidence not found in source text (possible hallucination)")
        elif not value_in(f, c.value, c.evidence):
            _halve(c)
            warnings.append(f"{f}: '{c.value}' does not appear in its own evidence '{c.evidence[:80]}' "
                            "(possible hallucination)")

    in_source = numbers_in(source)
    for i, li in enumerate(po.line_items, start=1):
        problem = _line_problem(li, source, in_source)
        if problem:
            _halve(li)
            warnings.append(f"line {li.line_no or i}: {problem} (possible hallucination)")

    if po.total is not None:
        ev = po.totals_evidence
        if not has_number(po.total, in_source):
            problem = f"total {po.total:,.2f} does not appear in the document"
        elif ev and not snippet_in(ev, source):
            problem = "total: evidence not found in source text"
        elif ev and not has_number(po.total, numbers_in(ev)):
            problem = f"total {po.total:,.2f} does not appear in its own evidence '{ev[:80]}'"
        else:
            problem = None
        if problem:
            po.totals_confidence = round(po.totals_confidence * 0.5, 3)
            warnings.append(f"{problem} (possible hallucination)")
    for name in ("subtotal", "tax", "shipping"):
        v = getattr(po, name)
        if v and not has_number(v, in_source):  # 0 / None are often implied, not printed
            po.totals_confidence = round(po.totals_confidence * 0.5, 3)
            warnings.append(f"{name} {v:,.2f} does not appear in the document (possible hallucination)")
    return warnings


def _line_problem(li: LineItem, source: str, in_source: set[float]) -> str | None:
    if li.evidence and not snippet_in(li.evidence, source, threshold=75):  # rows are often re-spaced
        return "evidence not found in source text"
    for name in ("quantity", "unit_price", "line_total"):
        v = getattr(li, name)
        if v is not None and not has_number(v, in_source):
            return f"{name} {v:g} does not appear in the document"
    if li.sku and _alnum(li.sku) not in _alnum(source):
        return f"SKU {li.sku} does not appear in the document"
    return None


def _verify_scan(po: PurchaseOrder, ocr: str) -> list[str]:
    """Vision output: evidence must contain the value; OCR, if readable, must not contradict it."""
    warnings = []
    for f in ("vendor_name", "po_number"):
        c: Confidenced = getattr(po, f)
        if not c.present:
            continue
        if not c.evidence:
            c.confidence = min(c.confidence, NO_EVIDENCE_CAP)
            warnings.append(f"{f}: no evidence was given for '{c.value}'")
        elif not value_in(f, c.value, c.evidence):
            _halve(c)
            warnings.append(f"{f}: '{c.value}' does not appear in its own evidence (possible hallucination)")
        elif ocr and fuzz.partial_ratio(_flat(c.value), _flat(ocr)) < 75:
            _halve(c)
            warnings.append(f"{f}: '{c.value}' is not corroborated by the OCR text")
    if po.total is not None:
        if po.totals_evidence and not has_number(po.total, numbers_in(po.totals_evidence)):
            po.totals_confidence = round(po.totals_confidence * 0.5, 3)
            warnings.append(f"total {po.total:,.2f} does not appear in its own evidence (possible hallucination)")
        elif ocr and not has_number(po.total, numbers_in(ocr)):
            po.totals_confidence = round(po.totals_confidence * 0.5, 3)
            warnings.append(f"total {po.total:,.2f} is not corroborated by the OCR text")
    return warnings
