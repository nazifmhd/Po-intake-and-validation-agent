"""Read-only queries behind the pages."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from sqlmodel import select

from app.db import session_scope
from app.models import AuditLog, Job, LineItemRecord, ProcessedEmail, PurchaseOrderRecord, ReviewQueueItem, VendorAlias


def jl(raw: Optional[str], default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except json.JSONDecodeError:
        return default


def pending() -> list[ReviewQueueItem]:
    with session_scope() as s:
        return list(s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status == "pending_review")
                           .order_by(ReviewQueueItem.created_at)).all())


def closed_reviews() -> list[ReviewQueueItem]:
    with session_scope() as s:
        return list(s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status != "pending_review")
                           .order_by(ReviewQueueItem.reviewed_at.desc())).all())


def review_item(item_id: int) -> Optional[ReviewQueueItem]:
    with session_scope() as s:
        return s.get(ReviewQueueItem, item_id)


def purchase_orders() -> list[PurchaseOrderRecord]:
    with session_scope() as s:
        return list(s.exec(select(PurchaseOrderRecord).order_by(PurchaseOrderRecord.created_at.desc())
                           .limit(500)).all())


def line_items(po_id: int) -> list[LineItemRecord]:
    with session_scope() as s:
        return list(s.exec(select(LineItemRecord).where(LineItemRecord.po_id == po_id)
                           .order_by(LineItemRecord.id)).all())


def aliases() -> list[VendorAlias]:
    with session_scope() as s:
        return list(s.exec(select(VendorAlias).order_by(VendorAlias.created_at.desc())).all())


def audit(limit: int = 300, job_id: Optional[str] = None) -> list[AuditLog]:
    with session_scope() as s:
        q = select(AuditLog)
        if job_id:
            q = q.where(AuditLog.job_id == job_id)
        return list(s.exec(q.order_by(AuditLog.id.desc()).limit(limit)).all())


def trace(job_id: str) -> list[dict]:
    for row in audit(50, job_id):
        if row.event == "pipeline_completed":
            return jl(row.details_json, {}).get("trace") or []
    return []


def activity(limit: int = 12) -> list[dict]:
    """Newest documents first, with whatever is known about each of them so far."""
    with session_scope() as s:
        jobs = s.exec(select(Job).order_by(Job.created_at.desc(), Job.id.desc()).limit(limit)).all()
        ids = [j.job_id for j in jobs]
        pos = {p.job_id: p for p in s.exec(select(PurchaseOrderRecord).where(PurchaseOrderRecord.job_id.in_(ids))).all()}
        items = {r.job_id: r for r in s.exec(select(ReviewQueueItem).where(ReviewQueueItem.job_id.in_(ids))).all()}
        mails = {m.message_id: m for m in s.exec(select(ProcessedEmail).where(
            ProcessedEmail.message_id.in_([j.message_id for j in jobs if j.message_id]))).all()}
    out = []
    for j in jobs:
        po, item = pos.get(j.job_id), items.get(j.job_id)
        rec = po or item
        if j.status in ("queued", "processing", "failed"):
            status = j.status
        elif j.status == "rejected":
            status = "refused"
        elif item is not None and item.status == "rejected":
            status = "rejected"
        elif po is not None:
            status = po.status
        else:
            status = j.decision or j.status
        mail = mails.get(j.message_id)
        out.append({
            "job_id": j.job_id, "file": j.filename, "when": j.created_at, "status": status, "note": j.note,
            "po_number": rec.po_number if rec else None,
            "vendor": (po.vendor_canonical_name or po.vendor_name_raw) if po else (item.vendor_name_raw if item else None),
            "total": rec.total if rec else None, "currency": rec.currency if rec else None,
            "flags": jl(item.flags_json, []) if item else [], "item_id": item.id if item else None,
            "from": mail.sender if mail else None,
        })
    return out


def summary() -> dict:
    with session_scope() as s:
        pos = s.exec(select(PurchaseOrderRecord)).all()
        items = s.exec(select(ReviewQueueItem)).all()
        jobs = s.exec(select(Job)).all()
    live = [p for p in pos if p.status != "superseded"]
    waiting = [i for i in items if i.status == "pending_review"]
    flags = Counter(f for i in items for f in jl(i.flags_json, []))
    auto = sum(p.status == "approved" for p in live)
    # every item that reached a person counts as reviewed, also after a correction cleared its flags
    return {
        "processed": auto + len(items), "auto": auto, "waiting": len(waiting),
        "by_person": sum(p.status == "approved_by_review" for p in live),
        "rejected": sum(i.status == "rejected" for i in items),
        "value_filed": sum(p.total or 0 for p in live if (p.currency or "USD") == "USD"),
        "value_waiting": sum(i.total or 0 for i in waiting if (i.currency or "USD") == "USD"),
        "flags": flags.most_common(6), "duplicates": sum(j.decision == "duplicate" for j in jobs),
        "refused": sum(j.status == "rejected" for j in jobs),
        "running": sum(j.status in ("queued", "processing") for j in jobs),
    }
