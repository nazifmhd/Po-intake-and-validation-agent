"""FastAPI app: the inbound-email webhook plus a small read/act API over the stores."""
from __future__ import annotations

import hmac
import json
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlmodel import select

from app.actions import review as review_actions
from app.config import get_settings
from app.db import get_engine, session_scope
from app.intake.common import (
    InboundAttachment, InboundEmail, accept_email, process_email, requeue_own_jobs, run_job, sweep_jobs,
)
from app.intake.webhook import WebhookAuthError, parse_inbound, replay_cache
from app.logging_setup import configure_logging
from app.models import AuditLog, Job, LineItemRecord, PurchaseOrderRecord, ReviewQueueItem

logger = logging.getLogger(__name__)


WORKER = "api"


def _sweeper(stop: threading.Event, interval: int) -> None:
    while not stop.wait(interval):
        try:
            sweep_jobs(WORKER)
        except Exception:
            logger.exception("Job sweep failed")


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
    if not settings.real_secret(settings.api_key):
        logger.warning("API_KEY is not set: every endpoint except the webhook and /health is disabled")
    recovered = requeue_own_jobs(WORKER, settings)
    if recovered:
        logger.warning("Re-queued jobs interrupted by the last shutdown", extra={"jobs": recovered})
    stop = threading.Event()
    threading.Thread(target=_sweeper, args=(stop, settings.job_sweep_seconds), daemon=True, name="sweeper").start()
    yield
    stop.set()


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Fail closed: the webhook is meant to be exposed through a tunnel, and everything else shares its port."""
    expected = get_settings().real_secret(get_settings().api_key)
    if not expected:
        raise HTTPException(503, "API_KEY is not configured; this endpoint is disabled")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key")


app = FastAPI(title="PO Intake & Validation Agent", version="1.0.0", lifespan=lifespan)
api = APIRouter(dependencies=[Depends(require_api_key)])


def _run_jobs(jobs: list[dict], message_id: str) -> None:
    for job in jobs:
        run_job(job, message_id, worker=WORKER)


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
    token = str(form.get("token", ""))
    try:
        result = await run_in_threadpool(accept_email, email, settings)
    except Exception:
        replay_cache.forget(token)  # let the provider's retry through
        raise HTTPException(503, "intake failed, please retry", headers={"Retry-After": "60"})
    if result.status == "in_progress":
        replay_cache.forget(token)
        raise HTTPException(503, "this message is still being accepted, please retry", headers={"Retry-After": "60"})
    if result.jobs:
        background.add_task(_run_jobs, result.jobs, result.message_id)
    return result.as_dict()


@api.post("/process")
async def process_upload(files: list[UploadFile] = File(...), sender: str = Form("upload@local"),
                         subject: str = Form("manual upload"), message_id: Optional[str] = Form(None)):
    """Direct upload (demo / manual): runs synchronously and returns the decisions."""
    attachments = [InboundAttachment(f.filename or "upload", await f.read(), f.content_type or "") for f in files]
    email = InboundEmail(message_id=message_id or f"<upload-{uuid.uuid4().hex}@po-agent>", sender=sender,
                         subject=subject, attachments=attachments, source="upload")
    return await run_in_threadpool(process_email, email, None, WORKER)


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
            "rag_enabled": s.rag_enabled, "time": datetime.now(timezone.utc).isoformat()}


@api.get("/jobs")
def list_jobs(limit: int = Query(50, le=500)):
    return _rows(Job, Job.created_at, limit)


@api.get("/jobs/{job_id}")
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


@api.get("/purchase-orders")
def list_pos(limit: int = Query(50, le=500)):
    return [_decode(r) for r in _rows(PurchaseOrderRecord, PurchaseOrderRecord.created_at, limit)]


@api.get("/purchase-orders/{po_id}")
def get_po(po_id: int):
    with session_scope() as s:
        rec = s.get(PurchaseOrderRecord, po_id)
        if not rec:
            raise HTTPException(404, "purchase order not found")
        lines = s.exec(select(LineItemRecord).where(LineItemRecord.po_id == po_id)).all()
        return {**_decode(rec.model_dump()), "line_items": [li.model_dump() for li in lines]}


@api.get("/review-queue")
def list_review(status: str = "pending_review", limit: int = Query(50, le=500)):
    where = None if status == "all" else ReviewQueueItem.status == status
    return [_decode(r) for r in _rows(ReviewQueueItem, ReviewQueueItem.created_at, limit, where)]


class ApproveBody(BaseModel):
    reviewer: str
    note: str = ""
    vendor_override_id: Optional[str] = None
    learn_alias: bool = False
    supersede: bool = False


class CorrectBody(BaseModel):
    reviewer: str
    header: dict[str, Optional[str]] = {}
    amounts: dict[str, Optional[float]] = {}
    line_items: Optional[list[dict]] = None


class ReprocessBody(BaseModel):
    reviewer: str


class RejectBody(BaseModel):
    reviewer: str
    reason: str


@api.post("/review-queue/{item_id}/approve")
def approve_item(item_id: int, body: ApproveBody):
    try:
        po_id = review_actions.approve(item_id, body.reviewer, body.note, body.vendor_override_id, body.learn_alias,
                                       body.supersede)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))
    return {"status": "approved", "purchase_order_id": po_id}


@api.post("/review-queue/{item_id}/reject")
def reject_item(item_id: int, body: RejectBody):
    try:
        review_actions.reject(item_id, body.reviewer, body.reason)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))
    return {"status": "rejected"}


@api.post("/review-queue/{item_id}/correct")
def correct_item(item_id: int, body: CorrectBody):
    """Fix misread values; every check runs again. The item stays in the queue until approved or rejected."""
    try:
        return review_actions.correct(item_id, body.reviewer, body.header, body.amounts, body.line_items)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))


@api.post("/review-queue/{item_id}/reprocess")
def reprocess_item(item_id: int, body: ReprocessBody):
    try:
        return review_actions.reprocess(item_id, body.reviewer)
    except review_actions.ReviewError as exc:
        raise HTTPException(409, str(exc))


@api.get("/audit")
def audit_log(limit: int = Query(100, le=1000), job_id: Optional[str] = None):
    where = AuditLog.job_id == job_id if job_id else None
    return [_decode(r) for r in _rows(AuditLog, AuditLog.id, limit, where)]


@api.post("/kb/rebuild")
def rebuild_kb():
    from app.rag.ingest_kb import build_knowledge_base
    return build_knowledge_base(get_settings(), if_missing=False)


app.include_router(api)
