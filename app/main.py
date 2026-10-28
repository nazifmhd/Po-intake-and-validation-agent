"""FastAPI app: the inbound-email webhook plus a small read/act API over the stores."""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlmodel import select

from app.actions import review as review_actions
from app.config import get_settings
from app.db import get_engine, session_scope
from app.intake.common import InboundAttachment, InboundEmail, accept_email, process_email, run_job
from app.intake.webhook import WebhookAuthError, parse_inbound
from app.logging_setup import configure_logging
from app.models import AuditLog, Job, LineItemRecord, PurchaseOrderRecord, ReviewQueueItem

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    get_engine(settings)
    try:
        from app.rag.ingest_kb import build_knowledge_base
        build_knowledge_base(settings, if_missing=True)
    except Exception:
        logger.exception("Knowledge base build failed at startup; retrieval degrades to lexical")
    logger.info("PO agent API ready", extra={"llm": settings.llm_provider, "embeddings": settings.embedding_provider,
                                             "rag_enabled": settings.rag_enabled})
    yield


app = FastAPI(title="PO Intake & Validation Agent", version="1.0.0", lifespan=lifespan)


def _run_jobs(jobs: list[dict], message_id: str) -> None:
    for job in jobs:
        run_job(job, message_id)


# ------------------------------------------------------------------ intake
@app.post("/webhook/inbound-email")
async def inbound_email(request: Request, background: BackgroundTasks, provider: Optional[str] = None):
    """Verify -> persist -> dedup -> enqueue -> return 200 fast (so the provider doesn't retry)."""
    settings = get_settings()
    provider = provider or settings.webhook_provider
    form = await request.form()
    try:
        email = await parse_inbound(provider, form, request.headers, request.query_params, settings)
    except WebhookAuthError as exc:
        logger.warning("Rejected inbound webhook", extra={"provider": provider, "reason": str(exc)})
        from app.actions.audit import audit
        audit("webhook_rejected", details={"provider": provider, "reason": str(exc),
                                           "client": request.client.host if request.client else None})
        raise HTTPException(status_code=401, detail=f"webhook verification failed: {exc}")
    result = accept_email(email, settings)
    if result.jobs:
        background.add_task(_run_jobs, result.jobs, result.message_id)
    return result.as_dict()


@app.post("/process")
async def process_upload(files: list[UploadFile] = File(...), sender: str = Form("upload@local"),
                         subject: str = Form("manual upload"), message_id: Optional[str] = Form(None)):
    """Direct upload (demo / manual): runs synchronously and returns the decisions."""
    attachments = [InboundAttachment(f.filename or "upload", await f.read(), f.content_type or "") for f in files]
    email = InboundEmail(message_id=message_id or f"<upload-{uuid.uuid4().hex}@po-agent>", sender=sender,
                         subject=subject, attachments=attachments, source="upload")
    return process_email(email)


# ------------------------------------------------------------------ read API
def _rows(model, order_by, limit: int, where=None) -> list[dict]:
    with session_scope() as s:
        q = select(model)
        if where is not None:
            q = q.where(where)
        return [r.model_dump() for r in s.exec(q.order_by(order_by.desc()).limit(limit)).all()]


def _decode(row: dict) -> dict:
    for k in list(row):
        if k.endswith("_json") and isinstance(row[k], str):
            try:
                row[k[:-5]] = json.loads(row.pop(k))
            except json.JSONDecodeError:
                pass
    return row


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "llm_provider": s.llm_provider, "embedding_provider": s.embedding_provider,
            "rag_enabled": s.rag_enabled, "time": datetime.utcnow().isoformat()}


@app.get("/jobs")
def list_jobs(limit: int = Query(50, le=500)):
    return _rows(Job, Job.created_at, limit)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with session_scope() as s:
        job = s.exec(select(Job).where(Job.job_id == job_id)).first()
        if not job:
            raise HTTPException(404, "job not found")
        out = job.model_dump()
        po = s.exec(select(PurchaseOrderRecord).where(PurchaseOrderRecord.job_id == job_id)).first()
        rq = s.exec(select(ReviewQueueItem).where(ReviewQueueItem.job_id == job_id)).first()
        out["purchase_order_id"] = po.id if po else None
        out["review_queue_id"] = rq.id if rq else None
        return out


@app.get("/purchase-orders")
def list_pos(limit: int = Query(50, le=500)):
    return [_decode(r) for r in _rows(PurchaseOrderRecord, PurchaseOrderRecord.created_at, limit)]


@app.get("/purchase-orders/{po_id}")
def get_po(po_id: int):
    with session_scope() as s:
        rec = s.get(PurchaseOrderRecord, po_id)
        if not rec:
            raise HTTPException(404, "purchase order not found")
        lines = s.exec(select(LineItemRecord).where(LineItemRecord.po_id == po_id)).all()
        return {**_decode(rec.model_dump()), "line_items": [li.model_dump() for li in lines]}


@app.get("/review-queue")
def list_review(status: str = "pending_review", limit: int = Query(50, le=500)):
    where = None if status == "all" else ReviewQueueItem.status == status
    return [_decode(r) for r in _rows(ReviewQueueItem, ReviewQueueItem.created_at, limit, where)]


class ApproveBody(BaseModel):
    reviewer: str
    note: str = ""
    vendor_override_id: Optional[str] = None
    learn_alias: bool = False


class RejectBody(BaseModel):
    reviewer: str
    reason: str


@app.post("/review-queue/{item_id}/approve")
def approve_item(item_id: int, body: ApproveBody):
    try:
        po_id = review_actions.approve(item_id, body.reviewer, body.note, body.vendor_override_id, body.learn_alias)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))
    return {"status": "approved", "purchase_order_id": po_id}


@app.post("/review-queue/{item_id}/reject")
def reject_item(item_id: int, body: RejectBody):
    try:
        review_actions.reject(item_id, body.reviewer, body.reason)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))
    return {"status": "rejected"}


@app.get("/audit")
def audit_log(limit: int = Query(100, le=1000), job_id: Optional[str] = None):
    where = AuditLog.job_id == job_id if job_id else None
    return [_decode(r) for r in _rows(AuditLog, AuditLog.id, limit, where)]


@app.post("/kb/rebuild")
def rebuild_kb():
    from app.rag.ingest_kb import build_knowledge_base
    return build_knowledge_base(get_settings(), if_missing=False)
