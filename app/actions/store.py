"""Stage 7 - persistence: approved POs + line items, the review queue, duplicate lookup."""
from __future__ import annotations

import json
from typing import Optional

from sqlmodel import select

from app.config import Settings, get_settings
from app.db import session_scope
from app.models import LineItemRecord, PurchaseOrderRecord, ReviewQueueItem


def _norm_po(po_number: Optional[str]) -> Optional[str]:
    return "".join(po_number.upper().split()) if po_number else None


def find_duplicate(po_number: Optional[str], settings: Settings | None = None) -> Optional[dict]:
    """PO numbers are issued by *our* company, so one number = one PO. A PO already filed or
    already waiting in the review queue is a duplicate; a previously *rejected* one is not."""
    key = _norm_po(po_number)
    if not key:
        return None
    with session_scope(settings) as s:
        for rec in s.exec(select(PurchaseOrderRecord)).all():
            if _norm_po(rec.po_number) == key:
                return {"table": "purchase_orders", "id": rec.id, "status": rec.status, "job_id": rec.job_id}
        for item in s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status != "rejected")).all():
            if _norm_po(item.po_number) == key:
                return {"table": "review_queue", "id": item.id, "status": item.status, "job_id": item.job_id}
    return None


def write_purchase_order(po: dict, *, job_id: str, vendor_match: dict, price_checks: list[dict], signals: dict,
                         source_file: str, message_id: Optional[str], status: str, approved_by: str,
                         settings: Settings | None = None) -> int:
    checks_by_line = {c.get("line_no"): c for c in price_checks or []}
    with session_scope(settings) as s:
        rec = PurchaseOrderRecord(
            job_id=job_id, po_number=po["po_number"]["value"], po_date=po["po_date"]["value"],
            vendor_name_raw=po["vendor_name"]["value"], vendor_id=vendor_match.get("vendor_id"),
            vendor_canonical_name=vendor_match.get("canonical_name"),
            buyer_department=po["buyer_department"]["value"], currency=po["currency"]["value"],
            subtotal=po.get("subtotal"), tax=po.get("tax"), shipping=po.get("shipping"), total=po.get("total"),
            payment_terms=po["payment_terms"]["value"], delivery_date=po["delivery_date"]["value"],
            status=status, overall_confidence=po.get("overall_confidence", 0.0), source_file=source_file,
            source_message_id=message_id, extracted_json=json.dumps(po, default=str),
            signals_json=json.dumps(signals or {}, default=str), approved_by=approved_by,
        )
        s.add(rec)
        s.flush()
        for li in po.get("line_items", []):
            chk = checks_by_line.get(li.get("line_no"), {})
            s.add(LineItemRecord(
                po_id=rec.id, line_no=li.get("line_no"), sku=li.get("sku"), matched_sku=chk.get("matched_sku"),
                description=li.get("description"), quantity=li.get("quantity"), unit=li.get("unit"),
                unit_price=li.get("unit_price"), line_total=li.get("line_total"),
                catalog_unit_price=chk.get("catalog_unit_price"), price_deviation_pct=chk.get("deviation_pct"),
            ))
        return rec.id


def store_approved(state: dict, settings: Settings | None = None) -> int:
    return write_purchase_order(
        state["po"], job_id=state["job_id"], vendor_match=state.get("vendor_match") or {},
        price_checks=state.get("price_checks") or [], signals=state.get("signals") or {},
        source_file=state["file_path"], message_id=state.get("source_message_id"),
        status="approved", approved_by="agent", settings=settings,
    )


def enqueue_review(state: dict, settings: Settings | None = None) -> int:
    po = state.get("po") or {}
    get = lambda f: (po.get(f) or {}).get("value")  # noqa: E731
    confidences = {}
    if po:
        confidences = {f: (po.get(f) or {}).get("confidence") for f in
                       ("po_number", "po_date", "vendor_name", "vendor_tax_id", "buyer_department", "currency",
                        "payment_terms", "delivery_date")}
        confidences["totals"] = po.get("totals_confidence")
        confidences["line_items"] = min((li.get("confidence", 0) for li in po.get("line_items", [])), default=0.0)
    with session_scope(settings) as s:
        item = ReviewQueueItem(
            job_id=state["job_id"], po_number=get("po_number"), vendor_name_raw=get("vendor_name"),
            vendor_match_json=json.dumps(state.get("vendor_match") or {}, default=str),
            total=po.get("total"), currency=get("currency"), status="pending_review",
            flags_json=json.dumps(state.get("flags") or []),
            reasons_json=json.dumps(state.get("reasons") or {}, default=str),
            field_confidences_json=json.dumps(confidences),
            extracted_json=json.dumps(po, default=str),
            signals_json=json.dumps(state.get("signals") or {}, default=str),
            price_checks_json=json.dumps(state.get("price_checks") or [], default=str),
            policy_findings_json=json.dumps({"findings": state.get("policy_findings") or [],
                                             "clauses": state.get("policy_clauses") or []}, default=str),
            errors_json=json.dumps(state.get("errors") or [], default=str),
            source_file=state["file_path"], doc_type=state.get("doc_type"),
            source_message_id=state.get("source_message_id"),
        )
        s.add(item)
        s.flush()
        return item.id
