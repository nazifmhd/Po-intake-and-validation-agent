"""Human-in-the-loop: the reviewer closes the loop on a queued PO (used by the Streamlit UI and the API).

correct    -> fix misread values; every check runs again on the corrected PO
reprocess  -> extract the original file again
approve    -> promote to purchase_orders (status approved_by_review)
reject     -> close with a reason

The rules a reviewer can't bypass live here, not in the UI, so the API enforces them too:
the vendor must be an *active* registry record, critical fields must be present, and every flag needs a
written justification (e.g. the director's sign-off reference).

Optional vendor alias learning: when the reviewer confirms that an "unknown" vendor is really a
registry vendor under a new spelling, the alias is saved so the next PO matches automatically.
"""
from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.actions.audit import audit
from app.actions.notify import send_slack
from app.actions.sheets import mirror_row
from app.actions.store import FILING_LOCK, LIVE_STATUSES, insert_purchase_order, po_key, review_fields
from app.config import Settings, get_settings
from app.db import session_scope
from app.extraction.schema import Confidenced, LineItem, PurchaseOrder
from app.models import PurchaseOrderRecord, ReviewQueueItem, VendorAlias, utcnow
from app.rag.store import base_name
from app.rag.vendors import load_registry

HEADER_FIELDS = PurchaseOrder.HEADER_FIELDS
AMOUNT_FIELDS = ("subtotal", "tax", "shipping", "total")
LINE_FIELDS = ("line_no", "sku", "description", "quantity", "unit", "unit_price", "line_total")


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


def _require_reviewer(reviewer: str) -> str:
    if not (reviewer or "").strip():
        raise ReviewError("Reviewer name is required")
    return reviewer.strip()


def _claim(s: Session, item_id: int, status: str, reviewer: str, note: str) -> None:
    """Atomic pending -> closed; of two simultaneous clicks exactly one wins."""
    done = s.exec(update(ReviewQueueItem)
                  .where(ReviewQueueItem.id == item_id, ReviewQueueItem.status == "pending_review")
                  .values(status=status, reviewer=reviewer, review_note=note, reviewed_at=utcnow()))
    if done.rowcount != 1:
        raise ReviewError(f"Review item {item_id} has already been handled")


def _resolve_vendor(vendor_match: dict, override_id: Optional[str], settings: Settings):
    """The registry record this PO will be booked against - looked up now, not when it was queued."""
    by_id = {v.vendor_id: v for v in load_registry(settings)}
    if vendor_match.get("status") == "matched" and vendor_match.get("vendor_id") in by_id and \
            by_id[vendor_match["vendor_id"]].status == "blocked":
        raise ReviewError(f"{vendor_match.get('canonical_name')} is BLOCKED: this PO can only be rejected. If the "
                          "vendor name was misread, correct it first.")
    # an uncertain or ambiguous match is a suggestion: the reviewer has to name the record themselves
    vendor_id = override_id or (vendor_match.get("vendor_id") if vendor_match.get("status") == "matched" else None)
    if not vendor_id and vendor_match.get("vendor_id"):
        raise ReviewError(f"'{vendor_match.get('canonical_name')}' is only a possible match. Choose the registry record "
                          "yourself if you have confirmed it is the same company, or reject the PO.")
    if not vendor_id:
        raise ReviewError("The vendor is not in the approved registry. Choose the registry record it belongs to, "
                          "or reject the PO until the vendor is onboarded.")
    vendor = by_id.get(vendor_id)
    if vendor is None:
        raise ReviewError(f"Unknown vendor id {vendor_id}")
    if vendor.status != "active":
        raise ReviewError(f"Vendor {vendor.name} is {vendor.status.upper()} - it cannot be approved")
    return vendor


def approve(item_id: int, reviewer: str, note: str = "", vendor_override_id: Optional[str] = None,
            learn_alias: bool = False, supersede: bool = False, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    reviewer = _require_reviewer(reviewer)
    item = _load_pending(item_id, settings)
    po = json.loads(item.extracted_json or "{}")
    flags = json.loads(item.flags_json or "[]")
    if not po:
        raise ReviewError("Nothing was extracted from this document; reprocess or reject it")
    gaps = [f for f, ok in (("vendor", (po.get("vendor_name") or {}).get("value")), ("total", po.get("total")),
                            ("line items", po.get("line_items")),
                            ("PO number", (po.get("po_number") or {}).get("value"))) if not ok]
    if gaps:
        raise ReviewError(f"Cannot file a PO without: {', '.join(gaps)}. Correct the values first.")
    if flags and not (note or "").strip():
        raise ReviewError(f"An approval note is required to override: {', '.join(flags)}")

    vendor_match = json.loads(item.vendor_match_json or "{}")
    vendor = _resolve_vendor(vendor_match, vendor_override_id, settings)
    if vendor.vendor_id != vendor_match.get("vendor_id") or vendor_match.get("status") != "matched":
        vendor_match = {**vendor_match, "status": "reviewer_confirmed"}
    vendor_match.update(vendor_id=vendor.vendor_id, canonical_name=vendor.name, vendor_status=vendor.status)

    raw_name = (po.get("vendor_name") or {}).get("value")
    if learn_alias:
        if not vendor_override_id:
            raise ReviewError("Choose the vendor record the spelling belongs to before remembering it")
        owner = next((v for v in load_registry(settings) if v.vendor_id != vendor.vendor_id
                      and base_name(raw_name) in {base_name(n) for n in v.all_names}), None)
        if owner:
            raise ReviewError(f"'{raw_name}' is already a name of {owner.name}; it can't also mean {vendor.name}")
        learn_alias = raw_name not in vendor.all_names

    key = po_key(po["po_number"]["value"])
    superseded: list[int] = []
    try:
        # One transaction: the claim, the filed PO, the alias and the superseded record stand or fall together.
        with FILING_LOCK, session_scope(settings) as s:
            _claim(s, item_id, "approved", reviewer, note)
            for old in s.exec(select(PurchaseOrderRecord)
                              .where(PurchaseOrderRecord.status.in_(LIVE_STATUSES))).all():
                if po_key(old.po_number) == key:
                    if not supersede:
                        raise ReviewError(f"{old.po_number} is already filed as record #{old.id}. Approve with "
                                          "'replace the filed PO' if this is an amendment, otherwise reject it.")
                    old.status, old.po_key = "superseded", None
                    s.add(old)
                    superseded.append(old.id)
            s.flush()
            po_id = insert_purchase_order(
                s, po, job_id=item.job_id, vendor_match=vendor_match,
                price_checks=json.loads(item.price_checks_json or "[]"),
                signals=json.loads(item.signals_json or "{}"), source_file=item.source_file,
                message_id=item.source_message_id, status="approved_by_review", approved_by=reviewer,
            )
            s.get(ReviewQueueItem, item_id).promoted_po_id = po_id
            if learn_alias:
                s.add(VendorAlias(vendor_id=vendor.vendor_id, alias=raw_name, created_by=reviewer))
    except IntegrityError as exc:
        raise ReviewError(f"{item.po_number} was filed by someone else in the meantime") from exc

    if learn_alias:
        audit("vendor_alias_learned", job_id=item.job_id, actor=reviewer,
              details={"vendor_id": vendor.vendor_id, "alias": raw_name}, settings=settings)
    audit("review_approved", job_id=item.job_id, decision="approved_by_review", flags=flags, actor=reviewer,
          details={"review_id": item_id, "po_id": po_id, "note": note, "vendor_id": vendor.vendor_id,
                   "vendor_override": vendor_override_id, "superseded_po_ids": superseded}, settings=settings)
    send_slack(f":ballot_box_with_check: PO {item.po_number or '(unknown)'} approved by {reviewer} after review "
               f"(flags: {', '.join(flags) or 'none'}){' - note: ' + note if note else ''}", settings)
    mirror_row("Approved", [po_id, item.po_number, (po.get("po_date") or {}).get("value"), vendor.name,
                            vendor.vendor_id, item.total, item.currency, "approved_by_review",
                            utcnow().isoformat()], settings)
    return po_id


def reject(item_id: int, reviewer: str, reason: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    reviewer = _require_reviewer(reviewer)
    if not (reason or "").strip():
        raise ReviewError("A rejection reason is required")
    item = _load_pending(item_id, settings)
    with session_scope(settings) as s:
        _claim(s, item_id, "rejected", reviewer, reason)
    audit("review_rejected", job_id=item.job_id, decision="rejected", flags=json.loads(item.flags_json or "[]"),
          actor=reviewer, details={"review_id": item_id, "reason": reason}, settings=settings)
    send_slack(f":x: PO {item.po_number or '(unknown)'} rejected by {reviewer}: {reason}", settings)


# ------------------------------------------------------------------ corrections
def _number(name: str, value) -> Optional[float]:
    from app.extraction.extractor import parse_number

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    out = parse_number(value)
    if out is None:
        raise ReviewError(f"{name}: '{value}' is not a number")
    return out


def _corrected(po: PurchaseOrder, header: dict, amounts: dict, line_items: Optional[list[dict]],
               reviewer: str) -> tuple[PurchaseOrder, dict]:
    from app.extraction.extractor import normalize_currency, normalize_date, normalize_terms

    po = po.model_copy(deep=True)
    changes: dict[str, list] = {}
    source = f"corrected by {reviewer}"
    unknown = (set(header) - set(HEADER_FIELDS)) | (set(amounts) - set(AMOUNT_FIELDS))
    if unknown:
        raise ReviewError(f"Unknown field(s): {', '.join(sorted(unknown))}")
    for f, value in header.items():
        new = (str(value).strip() or None) if value is not None else None
        if f in ("po_date", "delivery_date") and new:
            new, ok = normalize_date(new)
            if not ok:
                raise ReviewError(f"{f}: '{value}' is not a date (use YYYY-MM-DD)")
        elif f == "payment_terms":
            new = normalize_terms(new)
        elif f == "currency":
            new = normalize_currency(new)
        old = getattr(po, f).value
        if new != old:
            changes[f] = [old, new]
            setattr(po, f, Confidenced(value=new, confidence=1.0 if new else 0.0, evidence=source))
    for f, value in amounts.items():
        new, old = _number(f, value), getattr(po, f)
        if new != old:
            changes[f] = [old, new]
            setattr(po, f, new)
            if f == "total":
                po.totals_confidence, po.totals_evidence = (1.0 if new is not None else 0.0), source
    if line_items is not None:
        new_lines = []
        for i, row in enumerate(line_items, start=1):
            data = {k: row.get(k) for k in LINE_FIELDS}
            for k in ("quantity", "unit_price", "line_total"):
                data[k] = _number(f"line {i} {k}", data[k])
            data["line_no"] = int(_number(f"line {i} line_no", data["line_no"]) or i)
            for k in ("sku", "description", "unit"):
                data[k] = (str(data[k]).strip() or None) if data[k] is not None else None
            data["sku"] = data["sku"].upper() if data["sku"] else None
            data["unit"] = data["unit"].upper() if data["unit"] else None
            if not any(data[k] for k in ("sku", "description", "quantity", "unit_price")):
                continue  # an empty row in the editor
            before = next((li for li in po.line_items if {k: getattr(li, k) for k in LINE_FIELDS} == data), None)
            new_lines.append(before or LineItem(**data, confidence=1.0, evidence=source))
        old_rows = [{k: getattr(li, k) for k in LINE_FIELDS} for li in po.line_items]
        new_rows = [{k: getattr(li, k) for k in LINE_FIELDS} for li in new_lines]
        if old_rows != new_rows:
            changes["line_items"] = [old_rows, new_rows]
            po.line_items = new_lines
    return po, changes


def _reevaluate(item: ReviewQueueItem, state: dict, settings: Settings) -> dict:
    from app.graph.nodes import evaluate

    state = {**state, "job_id": item.job_id, "file_path": item.source_file, "doc_type": state.get("doc_type")
             or item.doc_type, "source_message_id": item.source_message_id, "exclude_review_id": item.id,
             "options": {}}
    result = evaluate(state)
    with session_scope(settings) as s:
        row = s.get(ReviewQueueItem, item.id)
        if row.status != "pending_review":
            raise ReviewError(f"Review item {item.id} is already {row.status}")
        for k, v in review_fields(result).items():
            setattr(row, k, v)
        s.add(row)
    return {"flags": result.get("flags") or [], "reasons": result.get("reasons") or {}}


def correct(item_id: int, reviewer: str, header: Optional[dict] = None, amounts: Optional[dict] = None,
            line_items: Optional[list[dict]] = None, settings: Settings | None = None) -> dict:
    """Apply the reviewer's corrections and run every check again. The item stays in the queue - a correction
    can clear flags, but approving is still a separate, deliberate action."""
    settings = settings or get_settings()
    reviewer = _require_reviewer(reviewer)
    item = _load_pending(item_id, settings)
    raw = json.loads(item.extracted_json or "{}")
    blank = {f: Confidenced.missing() for f in HEADER_FIELDS}
    po = PurchaseOrder.model_validate(raw) if raw else PurchaseOrder(**blank, line_items=[])
    po, changes = _corrected(po, header or {}, amounts or {}, line_items, reviewer)
    if changes:
        note = f"corrected by {reviewer}: {', '.join(changes)}"
        po.extraction_warnings = [*po.extraction_warnings, note]
        confs = po.field_confidences()
        po.overall_confidence = round(sum(confs.values()) / len(confs), 3)
    out = _reevaluate(item, {"po": po.model_dump(), "ingest_flags": json.loads(item.ingest_flags_json or "{}")},
                      settings)
    audit("review_corrected", job_id=item.job_id, flags=out["flags"], actor=reviewer,
          details={"review_id": item_id, "changes": changes, "flags_before": json.loads(item.flags_json or "[]")},
          settings=settings)
    return {**out, "changes": changes}


def reprocess(item_id: int, reviewer: str, settings: Settings | None = None) -> dict:
    """Run ingestion + extraction + every check again on the original file (e.g. after an outage)."""
    from app.graph.nodes import extract_again

    settings = settings or get_settings()
    reviewer = _require_reviewer(reviewer)
    item = _load_pending(item_id, settings)
    state = extract_again(item.source_file, item.job_id)
    out = _reevaluate(item, state, settings)
    audit("review_reprocessed", job_id=item.job_id, flags=out["flags"], actor=reviewer,
          details={"review_id": item_id, "errors": state.get("errors"),
                   "flags_before": json.loads(item.flags_json or "[]")}, settings=settings)
    return out


def pending_items(settings: Settings | None = None) -> list[ReviewQueueItem]:
    with session_scope(settings) as s:
        return list(s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status == "pending_review")
                           .order_by(ReviewQueueItem.created_at)).all())
