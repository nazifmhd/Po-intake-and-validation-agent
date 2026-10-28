"""Stage 1 core - shared by the webhook, the IMAP poller, direct upload and the CLI.

accept_email():  verify sender -> dedup on Message-ID -> type/size check per attachment ->
                 persist raw file -> split CSV batches -> one Job per PO document.
run_job():       run the LangGraph pipeline for one job and record its outcome.

Idempotent: the same Message-ID is never processed twice, and the pipeline itself dedups on
PO number, so a retried webhook or a re-sent email can't create a second record.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.actions.audit import audit
from app.actions.notify import notify_rejected_attachment
from app.config import Settings, get_settings
from app.db import session_scope
from app.models import Job, ProcessedEmail

logger = logging.getLogger(__name__)


@dataclass
class InboundAttachment:
    filename: str
    content: bytes
    content_type: str = ""


@dataclass
class InboundEmail:
    message_id: Optional[str]
    sender: Optional[str]
    subject: Optional[str]
    attachments: list[InboundAttachment] = field(default_factory=list)
    source: str = "webhook"


@dataclass
class IntakeResult:
    message_id: str
    status: str                                   # accepted | duplicate | no_attachment | rejected_sender | all_rejected
    jobs: list[dict] = field(default_factory=list)  # {job_id, path, filename}
    rejected: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"message_id": self.message_id, "status": self.status, "jobs": self.jobs, "rejected": self.rejected}


def _safe_name(name: str) -> str:
    name = Path(name or "attachment").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or "attachment"


def _synthetic_message_id(email: InboundEmail) -> str:
    h = hashlib.sha256()
    for part in (email.sender or "", email.subject or ""):
        h.update(part.encode())
    for a in email.attachments:
        h.update(hashlib.sha256(a.content).digest())
    return f"<synthetic-{h.hexdigest()[:24]}@po-agent>"


def split_csv_batch(content: bytes) -> list[tuple[str, bytes]]:
    """A CSV batch can hold several POs (rows grouped by a PO-number column) -> one CSV per PO."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return [("", content)]
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return [("", content)]
    po_col = next((c for c in reader.fieldnames
                   if c and c.strip().lower().replace(" ", "_") in {"po_number", "po_no", "po#", "purchase_order"}), None)
    if po_col is None:
        return [("", content)]
    groups: dict[str, list[dict]] = {}
    for row in reader:
        groups.setdefault((row.get(po_col) or "").strip(), []).append(row)
    if len(groups) <= 1:
        return [("", content)]
    out = []
    for po_no, rows in groups.items():
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=reader.fieldnames)
        w.writeheader()
        w.writerows(rows)
        out.append((po_no or "no-po-number", buf.getvalue().encode("utf-8")))
    return out


def accept_email(email: InboundEmail, settings: Settings | None = None) -> IntakeResult:
    settings = settings or get_settings()
    message_id = (email.message_id or "").strip() or _synthetic_message_id(email)
    sender_addr = parseaddr(email.sender or "")[1].lower()

    # Atomic claim: the UNIQUE constraint on message_id makes concurrent retries of the same
    # webhook (or a re-sent email) lose the race cleanly instead of being processed twice.
    try:
        with session_scope(settings) as s:
            s.add(ProcessedEmail(message_id=message_id, sender=sender_addr, subject=email.subject,
                                 source=email.source, attachment_count=len(email.attachments),
                                 status="received"))
    except IntegrityError:
        audit("email_duplicate_skipped", details={"message_id": message_id, "sender": sender_addr},
              settings=settings)
        return IntakeResult(message_id, "duplicate")

    def record(status: str, note: str = "") -> None:
        with session_scope(settings) as s:
            row = s.exec(select(ProcessedEmail).where(ProcessedEmail.message_id == message_id)).one()
            row.status, row.note = status, note
            s.add(row)

    # --- verify source ---
    allowed = settings.allowed_sender_domain_list
    if allowed and sender_addr.rsplit("@", 1)[-1] not in allowed:
        record("rejected_sender", f"sender domain not in allow-list: {sender_addr}")
        audit("email_rejected_sender", details={"message_id": message_id, "sender": sender_addr}, settings=settings)
        return IntakeResult(message_id, "rejected_sender")

    if not email.attachments:
        record("no_attachment", "acknowledged and skipped")
        audit("email_no_attachment", details={"message_id": message_id, "sender": sender_addr,
                                              "subject": email.subject}, settings=settings)
        return IntakeResult(message_id, "no_attachment")

    inbox = settings.storage_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    result = IntakeResult(message_id, "accepted")
    max_bytes = int(settings.max_attachment_mb * 1024 * 1024)

    for att in email.attachments:
        fname = _safe_name(att.filename)
        ext = Path(fname).suffix.lower()
        reason = None
        if ext not in settings.supported_extensions:
            reason = f"unsupported attachment type '{ext or 'none'}' (supported: {', '.join(settings.supported_extensions)})"
        elif not att.content:
            reason = "attachment is empty"
        elif len(att.content) > max_bytes:
            reason = f"attachment is {len(att.content) / 1e6:.1f} MB (limit {settings.max_attachment_mb:g} MB)"
        if reason:
            result.rejected.append({"filename": fname, "reason": reason})
            with session_scope(settings) as s:
                s.add(Job(job_id=uuid.uuid4().hex[:12], message_id=message_id, filename=fname, status="rejected",
                          note=reason, sha256=hashlib.sha256(att.content).hexdigest()))
            audit("attachment_rejected", details={"message_id": message_id, "filename": fname, "reason": reason},
                  settings=settings)
            notify_rejected_attachment(fname, reason, sender_addr, settings)
            continue

        parts = split_csv_batch(att.content) if ext == ".csv" else [("", att.content)]
        for po_hint, content in parts:
            job_id = uuid.uuid4().hex[:12]
            stem, suffix = Path(fname).stem, Path(fname).suffix
            name = f"{job_id}_{stem}{('_' + _safe_name(po_hint)) if po_hint else ''}{suffix}"
            path = inbox / name
            path.write_bytes(content)
            with session_scope(settings) as s:
                s.add(Job(job_id=job_id, message_id=message_id, filename=fname, stored_path=str(path),
                          sha256=hashlib.sha256(content).hexdigest(), status="queued",
                          note=f"CSV batch part for PO {po_hint}" if po_hint else None))
            result.jobs.append({"job_id": job_id, "path": str(path), "filename": fname})

    if not result.jobs:
        result.status = "all_rejected"
    record(result.status, f"{len(result.jobs)} job(s), {len(result.rejected)} rejected")
    audit("email_accepted", details={"message_id": message_id, "sender": sender_addr, "subject": email.subject,
                                     "source": email.source, "jobs": [j["job_id"] for j in result.jobs],
                                     "rejected": result.rejected}, settings=settings)
    return result


def run_job(job: dict, message_id: Optional[str] = None, options: Optional[dict] = None,
            settings: Settings | None = None) -> dict:
    """Run the pipeline for one job. Never raises: the graph is fail-safe, this guards the rest."""
    from app.graph.build import run_pipeline, summarize

    settings = settings or get_settings()
    _set_job(job["job_id"], status="processing", settings=settings)
    try:
        final = run_pipeline(job["path"], job_id=job["job_id"], message_id=message_id, options=options)
        summary = summarize(final)
        _set_job(job["job_id"], status="done", decision=summary["decision"], settings=settings)
        return summary
    except Exception as exc:  # the graph itself could not run (e.g. import / DB outage)
        logger.exception("Job failed outside the graph", extra={"job_id": job["job_id"]})
        _set_job(job["job_id"], status="failed", note=str(exc)[:500], settings=settings)
        audit("job_failed", job_id=job["job_id"], details={"error": str(exc)}, settings=settings)
        return {"job_id": job["job_id"], "decision": "failed", "error": str(exc)}


def _set_job(job_id: str, settings: Settings, **fields) -> None:
    try:
        with session_scope(settings) as s:
            job = s.exec(select(Job).where(Job.job_id == job_id)).first()
            if job:
                for k, v in fields.items():
                    setattr(job, k, v)
                if fields.get("status") in ("done", "failed"):
                    job.finished_at = datetime.now(timezone.utc)
                s.add(job)
    except Exception as exc:
        logger.error("Could not update job", extra={"job_id": job_id, "error": str(exc)})


def process_email(email: InboundEmail, settings: Settings | None = None) -> dict:
    """Synchronous accept + run (IMAP poller, CLI, direct upload)."""
    result = accept_email(email, settings)
    out = result.as_dict()
    out["results"] = [run_job(j, result.message_id, settings=settings) for j in result.jobs]
    return out
