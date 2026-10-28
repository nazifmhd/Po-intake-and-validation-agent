"""Human-in-the-loop: the reviewer closes the loop on a queued PO (used by the Streamlit UI and the API).

approve -> promote to purchase_orders (status approved_by_review)
reject  -> close with a reason
Optional vendor alias learning: when the reviewer confirms that an "unknown" vendor is really a
registry vendor under a new spelling, the alias is saved so the next PO matches automatically.
"""
from __future__ import annotations

import json
from typing import Optional

from sqlmodel import select

from app.actions.audit import audit
from app.actions.notify import send_slack
from app.actions.sheets import mirror_row
from app.actions.store import write_purchase_order
from app.config import Settings, get_settings
from app.db import session_scope
from app.models import ReviewQueueItem, VendorAlias, utcnow
from app.rag.store import load_vendors


class ReviewError(Exception):
    pass


def _load_pending(item_id: int, settings: Settings) -> ReviewQueueItem:
    with session_scope(settings) as s:
        item = s.get(ReviewQueueItem, item_id)
        if item is None:
            raise ReviewError(f"Review item {item_id} not found")
        if item.status != "pending_review":
            raise ReviewError(f"Review item {item_id} is already {item.status}")
        return item


def approve(item_id: int, reviewer: str, note: str = "", vendor_override_id: Optional[str] = None,
            learn_alias: bool = False, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    if not reviewer.strip():
        raise ReviewError("Reviewer name is required")
    item = _load_pending(item_id, settings)
    po = json.loads(item.extracted_json or "{}")
    if not po:
        raise ReviewError("Nothing was extracted from this document; it can only be rejected")

    vendor_match = json.loads(item.vendor_match_json or "{}")
    if vendor_override_id:
        vendor = next((v for v in load_vendors(settings.seed_path) if v.vendor_id == vendor_override_id), None)
        if vendor is None:
            raise ReviewError(f"Unknown vendor id {vendor_override_id}")
        vendor_match = {**vendor_match, "vendor_id": vendor.vendor_id, "canonical_name": vendor.name,
                        "vendor_status": vendor.status, "status": "reviewer_confirmed"}
        raw_name = (po.get("vendor_name") or {}).get("value")
        if learn_alias and raw_name and raw_name not in vendor.all_names:
            with session_scope(settings) as s:
                s.add(VendorAlias(vendor_id=vendor.vendor_id, alias=raw_name, created_by=reviewer))
            audit("vendor_alias_learned", job_id=item.job_id, actor=reviewer,
                  details={"vendor_id": vendor.vendor_id, "alias": raw_name}, settings=settings)

    po_id = write_purchase_order(
        po, job_id=item.job_id, vendor_match=vendor_match,
        price_checks=json.loads(item.price_checks_json or "[]"), signals=json.loads(item.signals_json or "{}"),
        source_file=item.source_file, message_id=item.source_message_id,
        status="approved_by_review", approved_by=reviewer, settings=settings,
    )
    with session_scope(settings) as s:
        db_item = s.get(ReviewQueueItem, item_id)
        db_item.status, db_item.reviewer, db_item.review_note = "approved", reviewer, note
        db_item.reviewed_at, db_item.promoted_po_id = utcnow(), po_id
        s.add(db_item)

    flags = json.loads(item.flags_json or "[]")
    audit("review_approved", job_id=item.job_id, decision="approved_by_review", flags=flags, actor=reviewer,
          details={"review_id": item_id, "po_id": po_id, "note": note, "vendor_override": vendor_override_id},
          settings=settings)
    send_slack(f":ballot_box_with_check: PO {item.po_number or '(unknown)'} approved by {reviewer} after review "
               f"(flags: {', '.join(flags) or 'none'})", settings)
    mirror_row("Approved", [po_id, item.po_number, po.get("po_date", {}).get("value"),
                            vendor_match.get("canonical_name") or item.vendor_name_raw, vendor_match.get("vendor_id"),
                            item.total, item.currency, "approved_by_review", utcnow().isoformat()], settings)
    return po_id


def reject(item_id: int, reviewer: str, reason: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not reviewer.strip():
        raise ReviewError("Reviewer name is required")
    if not reason.strip():
        raise ReviewError("A rejection reason is required")
    item = _load_pending(item_id, settings)
    with session_scope(settings) as s:
        db_item = s.get(ReviewQueueItem, item_id)
        db_item.status, db_item.reviewer, db_item.review_note, db_item.reviewed_at = "rejected", reviewer, reason, utcnow()
        s.add(db_item)
    audit("review_rejected", job_id=item.job_id, decision="rejected", flags=json.loads(item.flags_json or "[]"),
          actor=reviewer, details={"review_id": item_id, "reason": reason}, settings=settings)
    send_slack(f":x: PO {item.po_number or '(unknown)'} rejected by {reviewer}: {reason}", settings)


def pending_items(settings: Settings | None = None) -> list[ReviewQueueItem]:
    with session_scope(settings) as s:
        return list(s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status == "pending_review")
                           .order_by(ReviewQueueItem.created_at)).all())
