"""Stage 7 - persistence: approved POs + line items, the review queue, duplicate and related-order lookup."""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date, timedelta
from typing import Optional

from sqlmodel import Session, select

from app.config import Settings, get_settings
from app.db import session_scope
from app.models import LineItemRecord, PurchaseOrderRecord, ReviewQueueItem
from app.rag.store import base_name

logger = logging.getLogger(__name__)

# Serializes "check for duplicates, then file" inside a process; UNIQUE(po_key) covers other processes.
FILING_LOCK = threading.RLock()
LIVE_STATUSES = ("approved", "approved_by_review")


class FilingConflict(Exception):
    """The PO stopped being fileable between scoring and filing."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def po_key(po_number: Optional[str]) -> Optional[str]:
    """'PO-2026-0412', 'po 2026 0412' and 'PO2026/0412' are the same PO."""
    key = re.sub(r"[^A-Z0-9]", "", (po_number or "").upper())
    return key or None


# ------------------------------------------------------------------ duplicates
def _val(po: dict, field: str):
    return (po.get(field) or {}).get("value")


def _content(po: dict) -> dict:
    def num(v):
        return None if v is None else round(float(v), 2)

    lines = sorted((str(li.get("sku") or li.get("description") or "").strip().lower(), num(li.get("quantity")) or 0,
                    num(li.get("unit_price")) or 0) for li in po.get("line_items") or [])
    return {"vendor": base_name(_val(po, "vendor_name") or ""), "currency": _val(po, "currency"),
            "total": num(po.get("total")), "payment_terms": _val(po, "payment_terms"),
            "delivery_date": _val(po, "delivery_date"), "line items": lines}


def po_differences(old: dict, new: dict) -> list[str]:
    a, b = _content(old), _content(new)
    out = []
    for k in a:
        if a[k] != b[k]:
            out.append("line items changed" if k == "line items" else f"{k} {a[k]!r} -> {b[k]!r}")
    return out


def check_duplicate(po: dict, settings: Settings | None = None, exclude_review_id: Optional[int] = None,
                    ) -> Optional[dict]:
    """PO numbers are issued by *our* company, so one number = one PO.

    Same number and same content as a PO already filed or queued -> kind "duplicate" (a re-send; skipped).
    Same number but different content -> kind "conflict" (an amendment, a misread number, or fraud; a human
    decides). A previously *rejected* PO can be resubmitted.
    """
    key = po_key(_val(po, "po_number"))
    if not key:
        return None
    existing = []
    with session_scope(settings) as s:
        for rec in s.exec(select(PurchaseOrderRecord).where(PurchaseOrderRecord.status.in_(LIVE_STATUSES))).all():
            if po_key(rec.po_number) == key:
                existing.append(("purchase_orders", rec, [rec.extracted_json]))
        for item in s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status != "rejected")).all():
            if item.id != exclude_review_id and po_key(item.po_number) == key:
                existing.append(("review_queue", item, [item.extracted_json, item.original_json]))
    if not existing:
        return None
    found = []
    for table, rec, payloads in existing:
        ref = {"table": table, "id": rec.id, "status": rec.status, "job_id": rec.job_id}
        diffs = [po_differences(json.loads(p), po) for p in payloads if p and p not in ("{}", "null")]
        if any(not d for d in diffs):
            return {"kind": "duplicate", **ref, "differences": []}
        found.append({"kind": "conflict", **ref, "differences": diffs[0] if diffs else ["content unknown"]})
    return found[0]


# ------------------------------------------------------------------ related orders (spend splitting)
def _parse_date(value: Optional[str], fallback) -> date:
    try:
        return date.fromisoformat(value or "")
    except ValueError:
        return fallback.date() if fallback is not None else date.today()


def related_orders(vendor_id: str, skus: set[str], po_number: Optional[str], po_date: Optional[str], window_days: int,
                   settings: Settings | None = None, exclude_review_id: Optional[int] = None) -> list[dict]:
    """Filed or pending POs to the same vendor, sharing an item, dated within the window."""
    centre, window, key = _parse_date(po_date, None), timedelta(days=window_days), po_key(po_number)
    out = []
    with session_scope(settings) as s:
        recs = s.exec(select(PurchaseOrderRecord).where(PurchaseOrderRecord.vendor_id == vendor_id,
                                                        PurchaseOrderRecord.status.in_(LIVE_STATUSES))).all()
        for rec in recs:
            lines = s.exec(select(LineItemRecord).where(LineItemRecord.po_id == rec.id)).all()
            if po_key(rec.po_number) != key and skus & {li.matched_sku for li in lines} and \
                    abs(_parse_date(rec.po_date, rec.created_at) - centre) <= window:
                out.append({"po_number": rec.po_number, "total": rec.total, "table": "purchase_orders", "id": rec.id})
        for item in s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status == "pending_review")).all():
            if item.id == exclude_review_id or po_key(item.po_number) == key:
                continue
            if (json.loads(item.vendor_match_json or "{}") or {}).get("vendor_id") != vendor_id:
                continue
            matched = {c.get("matched_sku") for c in json.loads(item.price_checks_json or "[]")}
            po = json.loads(item.extracted_json or "{}")
            if skus & matched and abs(_parse_date(_val(po, "po_date"), item.created_at) - centre) <= window:
                out.append({"po_number": item.po_number, "total": item.total, "table": "review_queue", "id": item.id})
    return out


# ------------------------------------------------------------------ writes
def insert_purchase_order(s: Session, po: dict, *, job_id: str, vendor_match: dict, price_checks: list[dict],
                          signals: dict, source_file: str, message_id: Optional[str], status: str,
                          approved_by: str) -> int:
    checks = {c.get("index", i): c for i, c in enumerate(price_checks or [])}
    rec = PurchaseOrderRecord(
        job_id=job_id, po_number=_val(po, "po_number"), po_key=po_key(_val(po, "po_number")),
        po_date=_val(po, "po_date"), vendor_name_raw=_val(po, "vendor_name"), vendor_id=vendor_match.get("vendor_id"),
        vendor_canonical_name=vendor_match.get("canonical_name"),
        buyer_department=_val(po, "buyer_department"), currency=_val(po, "currency"),
        subtotal=po.get("subtotal"), tax=po.get("tax"), shipping=po.get("shipping"), total=po.get("total"),
        payment_terms=_val(po, "payment_terms"), delivery_date=_val(po, "delivery_date"),
        status=status, overall_confidence=po.get("overall_confidence", 0.0), source_file=source_file,
        source_message_id=message_id, extracted_json=json.dumps(po, default=str),
        signals_json=json.dumps(signals or {}, default=str), approved_by=approved_by,
    )
    s.add(rec)
    s.flush()
    for i, li in enumerate(po.get("line_items", [])):
        chk = checks.get(i, {})
        s.add(LineItemRecord(
            po_id=rec.id, line_no=li.get("line_no"), sku=li.get("sku"), matched_sku=chk.get("matched_sku"),
            description=li.get("description"), quantity=li.get("quantity"), unit=li.get("unit"),
            unit_price=li.get("unit_price"), line_total=li.get("line_total"),
            catalog_unit_price=chk.get("catalog_unit_price"), price_deviation_pct=chk.get("deviation_pct"),
        ))
    return rec.id


def store_approved(state: dict, settings: Settings | None = None) -> int:
    with session_scope(settings) as s:
        return insert_purchase_order(
            s, state["po"], job_id=state["job_id"], vendor_match=state.get("vendor_match") or {},
            price_checks=state.get("price_checks") or [], signals=state.get("signals") or {},
            source_file=state["file_path"], message_id=state.get("source_message_id"),
            status="approved", approved_by="agent",
        )


def review_fields(state: dict) -> dict:
    """The review_queue columns that (re)evaluating a PO produces."""
    po = state.get("po") or {}
    confidences = {}
    if po:
        confidences = {f: (po.get(f) or {}).get("confidence") for f in
                       ("po_number", "po_date", "vendor_name", "vendor_tax_id", "buyer_department", "currency",
                        "payment_terms", "delivery_date")}
        confidences["totals"] = po.get("totals_confidence")
        confidences["line_items"] = min((li.get("confidence", 0) for li in po.get("line_items", [])), default=0.0)
    return dict(
        po_number=_val(po, "po_number"), vendor_name_raw=_val(po, "vendor_name"),
        vendor_match_json=json.dumps(state.get("vendor_match") or {}, default=str),
        total=po.get("total"), currency=_val(po, "currency"),
        flags_json=json.dumps(state.get("flags") or []),
        reasons_json=json.dumps(state.get("reasons") or {}, default=str),
        field_confidences_json=json.dumps(confidences),
        extracted_json=json.dumps(po, default=str),
        signals_json=json.dumps(state.get("signals") or {}, default=str),
        price_checks_json=json.dumps(state.get("price_checks") or [], default=str),
        policy_findings_json=json.dumps({"findings": state.get("policy_findings") or [],
                                         "clauses": state.get("policy_clauses") or []}, default=str),
        errors_json=json.dumps(state.get("errors") or [], default=str),
        ingest_flags_json=json.dumps(state.get("ingest_flags") or {}),
        duplicate_json=json.dumps(state.get("duplicate_of") or {}, default=str),
        doc_type=state.get("doc_type"),
    )


def enqueue_review(state: dict, settings: Settings | None = None) -> int:
    fields = review_fields(state)
    with session_scope(settings) as s:
        # a retried job must not queue the same document twice
        item = s.exec(select(ReviewQueueItem).where(ReviewQueueItem.job_id == state["job_id"],
                                                    ReviewQueueItem.status == "pending_review")).first()
        if item is None:
            item = ReviewQueueItem(job_id=state["job_id"], status="pending_review", source_file=state["file_path"],
                                   source_message_id=state.get("source_message_id"),
                                   original_json=fields["extracted_json"])
        for k, v in fields.items():
            setattr(item, k, v)
        s.add(item)
        s.flush()
        return item.id


def write_dead_letter(state: dict, error: str, settings: Settings | None = None) -> str:
    """Last resort when the review queue itself can't be written: never lose the item."""
    settings = settings or get_settings()
    folder = settings.storage_path / "dead_letter"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{state.get('job_id')}.json"
    keep = ("job_id", "file_path", "source_message_id", "decision", "flags", "reasons", "po", "errors")
    path.write_text(json.dumps({"error": error, **{k: state.get(k) for k in keep}}, default=str, indent=2),
                    encoding="utf-8")
    return str(path)
