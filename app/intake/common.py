"""Stage 1 core - shared by the webhook, the IMAP poller, direct upload and the CLI.

accept_email():  verify sender -> dedup on Message-ID -> type/size check per attachment ->
                 persist raw file -> split CSV batches -> one Job per PO document.
run_job():       claim the job, run the LangGraph pipeline, record its outcome.
sweep_jobs():    pick up work a crashed process left behind.

Idempotent *and* lossless: the same Message-ID is never processed twice, but a claim only counts as
"processed" once its jobs are safely stored. If intake dies half-way, the redelivered email takes the
claim over; if a worker dies, the sweeper re-queues its job (bounded, then it goes to a human).
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Optional

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.actions.audit import audit
from app.actions.notify import notify_rejected_attachment, send_slack
from app.actions.store import enqueue_review, write_dead_letter
from app.config import Settings, get_settings
from app.db import session_scope
from app.ingestion.loaders import IMAGE_EXTS, decode_csv
from app.models import Job, ProcessedEmail, utcnow

logger = logging.getLogger(__name__)

MIN_DOCUMENT_PIXELS = 250  # a legible PO can't be smaller than this; signature logos and icons are
_MAGIC = ((b"%PDF", ".pdf"), (b"\xff\xd8\xff", ".jpg"), (b"\x89PNG", ".png"), (b"II*\x00", ".tif"),
          (b"MM\x00*", ".tif"))


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
    sender_verified: bool = True  # False when the mail server reported a forged From address


@dataclass
class IntakeResult:
    message_id: str
    status: str                                   # accepted | duplicate | in_progress | no_attachment | rejected_sender | all_rejected
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
    """A CSV batch can hold several POs (rows grouped by a PO-number column) -> one CSV per PO.
    Continuation rows may leave the PO number blank; they belong to the PO above them."""
    text, dialect = decode_csv(content)
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        return [("", content)]
    po_col = next((c for c in reader.fieldnames
                   if c and c.strip().lower().replace(" ", "_") in {"po_number", "po_no", "po#", "po_#",
                                                                    "purchase_order", "order_number"}), None)
    if po_col is None:
        return [("", content)]
    groups: dict[str, list[dict]] = {}
    current = ""
    for row in reader:
        if None in row:  # ragged rows: keep the file whole, the loader flags it
            return [("", content)]
        current = (row.get(po_col) or "").strip() or current
        groups.setdefault(current, []).append(row)
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


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)  # SQLite drops the timezone


def _claim_email(message_id: str, sender: str, email: InboundEmail, settings: Settings) -> str:
    """-> new | takeover | in_progress | duplicate. Atomic: the UNIQUE constraint on message_id makes
    concurrent retries of the same webhook (or a re-sent email) lose the race cleanly."""
    try:
        with session_scope(settings) as s:
            s.add(ProcessedEmail(message_id=message_id, sender=sender, subject=email.subject, source=email.source,
                                 attachment_count=len(email.attachments), status="received"))
        return "new"
    except IntegrityError:
        pass
    with session_scope(settings) as s:
        row = s.exec(select(ProcessedEmail).where(ProcessedEmail.message_id == message_id)).one()
        if row.status not in ("received", "intake_failed"):
            return "duplicate"
        stale = utcnow() - _aware(row.received_at) > timedelta(seconds=settings.intake_stale_seconds)
        if row.status == "received" and not stale:
            return "in_progress"
        won = s.exec(update(ProcessedEmail)
                     .where(ProcessedEmail.id == row.id, ProcessedEmail.status == row.status,
                            ProcessedEmail.received_at == row.received_at)
                     .values(status="received", received_at=utcnow()))
        return "takeover" if won.rowcount == 1 else "in_progress"


def _effective_name(att: InboundAttachment, settings: Settings) -> str:
    """Mail clients sometimes drop the extension; trust the file signature then."""
    name = _safe_name(att.filename)
    if Path(name).suffix.lower() not in settings.supported_extensions and not Path(name).suffix:
        for magic, ext in _MAGIC:
            if att.content.startswith(magic):
                return name + ext
    return name


def _too_small_image(content: bytes) -> Optional[str]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as img:
            if min(img.size) < MIN_DOCUMENT_PIXELS:
                return f"image is only {img.width}x{img.height} px (signature logo or icon, not a document)"
    except Exception:  # unreadable: let the pipeline report it to a human
        pass
    return None


def accept_email(email: InboundEmail, settings: Settings | None = None) -> IntakeResult:
    settings = settings or get_settings()
    message_id = (email.message_id or "").strip() or _synthetic_message_id(email)
    sender_addr = parseaddr(email.sender or "")[1].lower()

    claim = _claim_email(message_id, sender_addr, email, settings)
    if claim == "duplicate":
        audit("email_duplicate_skipped", details={"message_id": message_id, "sender": sender_addr},
              settings=settings)
        return IntakeResult(message_id, "duplicate")
    if claim == "in_progress":
        return IntakeResult(message_id, "in_progress")
    try:
        return _accept_claimed(email, message_id, sender_addr, settings)
    except Exception as exc:  # e.g. disk full: release the claim so a redelivery is not mistaken for a duplicate
        logger.exception("Intake failed; releasing the claim", extra={"message_id": message_id})
        _record_email(message_id, "intake_failed", str(exc)[:500], settings)
        audit("intake_failed", details={"message_id": message_id, "error": str(exc)}, settings=settings)
        raise


def _record_email(message_id: str, status: str, note: str, settings: Settings) -> None:
    with session_scope(settings) as s:
        row = s.exec(select(ProcessedEmail).where(ProcessedEmail.message_id == message_id)).one()
        row.status, row.note = status, note
        s.add(row)


def sender_allowed(sender: Optional[str], settings: Settings) -> bool:
    """No list configured = everyone (the webhook signature is then the only gate)."""
    domains, addresses = settings.allowed_sender_domain_list, settings.allowed_sender_list
    addr = parseaddr(sender or "")[1].lower()
    return not (domains or addresses) or addr in addresses or addr.rsplit("@", 1)[-1] in domains


def _accept_claimed(email: InboundEmail, message_id: str, sender_addr: str, settings: Settings) -> IntakeResult:
    # --- verify source ---
    reason = None
    if not email.sender_verified:
        reason = f"sender address looks forged (SPF/DKIM/DMARC failed): {sender_addr}"
    elif not sender_allowed(sender_addr, settings):
        reason = f"sender not in allow-list: {sender_addr}"
    if reason:
        _record_email(message_id, "rejected_sender", reason, settings)
        audit("email_rejected_sender", details={"message_id": message_id, "sender": sender_addr, "reason": reason},
              settings=settings)
        return IntakeResult(message_id, "rejected_sender")

    if not email.attachments:
        _record_email(message_id, "no_attachment", "acknowledged and skipped", settings)
        audit("email_no_attachment", details={"message_id": message_id, "sender": sender_addr,
                                              "subject": email.subject}, settings=settings)
        return IntakeResult(message_id, "no_attachment")

    with session_scope(settings) as s:  # jobs an interrupted earlier attempt already stored
        existing = {(j.filename, j.sha256): j for j in s.exec(select(Job).where(Job.message_id == message_id)).all()}

    inbox = settings.storage_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    result = IntakeResult(message_id, "accepted")
    max_bytes = int(settings.max_attachment_mb * 1024 * 1024)

    for att in email.attachments:
        fname = _effective_name(att, settings)
        ext = Path(fname).suffix.lower()
        reason, alert = None, True
        if ext not in settings.supported_extensions:
            reason = f"unsupported attachment type '{ext or 'none'}' (supported: {', '.join(settings.supported_extensions)})"
        elif not att.content:
            reason = "attachment is empty"
        elif len(att.content) > max_bytes:
            reason = f"attachment is {len(att.content) / 1e6:.1f} MB (limit {settings.max_attachment_mb:g} MB)"
        elif ext in IMAGE_EXTS:
            reason, alert = _too_small_image(att.content), False
        if reason:
            result.rejected.append({"filename": fname, "reason": reason})
            with session_scope(settings) as s:
                s.add(Job(job_id=uuid.uuid4().hex[:12], message_id=message_id, filename=fname, status="rejected",
                          note=reason, sha256=hashlib.sha256(att.content).hexdigest()))
            audit("attachment_rejected", details={"message_id": message_id, "filename": fname, "reason": reason},
                  settings=settings)
            if alert:
                notify_rejected_attachment(fname, reason, sender_addr, settings)
            continue

        parts = split_csv_batch(att.content) if ext == ".csv" else [("", att.content)]
        for po_hint, content in parts:
            sha = hashlib.sha256(content).hexdigest()
            known = existing.get((fname, sha))
            if known is not None and known.status != "rejected":
                result.jobs.append({"job_id": known.job_id, "path": known.stored_path, "filename": fname})
                continue
            job_id = uuid.uuid4().hex[:12]
            stem, suffix = Path(fname).stem, Path(fname).suffix
            name = f"{job_id}_{stem}{('_' + _safe_name(po_hint)) if po_hint else ''}{suffix}"
            path = inbox / name
            path.write_bytes(content)
            with session_scope(settings) as s:
                s.add(Job(job_id=job_id, message_id=message_id, filename=fname, stored_path=str(path),
                          sha256=sha, status="queued",
                          note=f"CSV batch part for PO {po_hint}" if po_hint else None))
            result.jobs.append({"job_id": job_id, "path": str(path), "filename": fname})

    if not result.jobs:
        result.status = "all_rejected"
    _record_email(message_id, result.status, f"{len(result.jobs)} job(s), {len(result.rejected)} rejected", settings)
    audit("email_accepted", details={"message_id": message_id, "sender": sender_addr, "subject": email.subject,
                                     "source": email.source, "jobs": [j["job_id"] for j in result.jobs],
                                     "rejected": result.rejected}, settings=settings)
    return result


# ------------------------------------------------------------------ jobs
def _claim_job(job_id: str, worker: str, settings: Settings) -> bool:
    """Atomic queued -> processing, so the webhook task and the sweeper never run the same job twice."""
    with session_scope(settings) as s:
        done = s.exec(update(Job).where(Job.job_id == job_id, Job.status == "queued")
                      .values(status="processing", worker=worker, started_at=utcnow(), attempts=Job.attempts + 1))
        return done.rowcount == 1


def run_job(job: dict, message_id: Optional[str] = None, options: Optional[dict] = None,
            settings: Settings | None = None, worker: str = "local") -> dict:
    """Run the pipeline for one job. Never raises: the graph is fail-safe, this guards the rest."""
    from app.graph.build import run_pipeline, summarize

    settings = settings or get_settings()
    job_id = job["job_id"]
    try:
        if not _claim_job(job_id, worker, settings):
            return {"job_id": job_id, "decision": "already_claimed", "flags": [], "errors": []}
        final = run_pipeline(job["path"], job_id=job_id, message_id=message_id, options=options)
        summary = summarize(final)
        if summary["decision"] == "needs_review" and not summary["record"]["id"]:
            raise RuntimeError("decided needs_review, but nothing could be written to the review queue")
        _set_job(job_id, status="done", decision=summary["decision"], settings=settings)
        return summary
    except Exception as exc:
        logger.exception("Job did not complete", extra={"job_id": job_id})
        _retry_or_hand_over(job, message_id, str(exc), settings)
        return {"job_id": job_id, "decision": "failed", "flags": [], "errors": [], "error": str(exc)}


def _retry_or_hand_over(job: dict, message_id: Optional[str], error: str, settings: Settings) -> None:
    """Bounded retries by the sweeper; after that the document itself goes to a human."""
    job_id = job["job_id"]
    audit("job_failed", job_id=job_id, details={"error": error}, settings=settings)
    try:
        with session_scope(settings) as s:
            row = s.exec(select(Job).where(Job.job_id == job_id)).first()
            attempts = row.attempts if row else settings.job_max_attempts
        if attempts < settings.job_max_attempts:
            _set_job(job_id, status="queued", note=f"attempt {attempts} failed: {error[:300]}", settings=settings)
            return
    except Exception as exc:
        logger.error("Could not read job", extra={"job_id": job_id, "error": str(exc)})
    state = {"job_id": job_id, "file_path": job["path"], "source_message_id": message_id, "decision": "needs_review",
             "flags": ["processing_error"], "reasons": {"processing_error": error[:500]}, "errors":
             [{"node": "job", "code": "processing_error", "message": error[:500]}]}
    try:
        review_id = enqueue_review(state, settings)
        send_slack(f":warning: Job `{job_id}` ({job.get('filename')}) failed {settings.job_max_attempts} times and "
                   f"needs review (queue #{review_id}): {error[:200]}", settings)
        _set_job(job_id, status="done", decision="needs_review", note=error[:500], settings=settings)
    except Exception as exc:
        path = write_dead_letter(state, f"{error} / {exc}", settings)
        send_slack(f":rotating_light: Job `{job_id}` ({job.get('filename')}) failed and could not be queued. "
                   f"Saved to {path}", settings)
        _set_job(job_id, status="failed", note=error[:500], settings=settings)


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


def requeue_own_jobs(worker: str, settings: Settings | None = None) -> int:
    """At startup: anything this worker was processing when it died is certainly not running any more."""
    with session_scope(settings) as s:
        return s.exec(update(Job).where(Job.status == "processing", Job.worker == worker)
                      .values(status="queued")).rowcount


def sweep_jobs(worker: str, settings: Settings | None = None, grace_seconds: Optional[int] = None) -> list[dict]:
    """Re-queue jobs whose worker died and run jobs nobody picked up."""
    settings = settings or get_settings()
    grace = settings.job_queued_grace_seconds if grace_seconds is None else grace_seconds
    now = utcnow()
    with session_scope(settings) as s:
        s.exec(update(Job).where(Job.status == "processing",
                                 Job.started_at < now - timedelta(seconds=settings.job_stale_seconds))
               .values(status="queued"))
    with session_scope(settings) as s:
        cutoff = now - timedelta(seconds=grace)
        todo = [({"job_id": j.job_id, "path": j.stored_path, "filename": j.filename}, j.message_id, j.attempts)
                for j in s.exec(select(Job).where(Job.status == "queued", Job.created_at <= cutoff,
                                                  or_(Job.started_at.is_(None), Job.started_at <= cutoff))).all()]
    results = []
    for job, message_id, attempts in todo:
        if attempts >= settings.job_max_attempts:  # it keeps killing its worker: stop retrying
            _retry_or_hand_over(job, message_id, f"gave up after {attempts} attempts", settings)
            continue
        logger.warning("Recovering job", extra={"job_id": job["job_id"], "attempts": attempts})
        results.append(run_job(job, message_id, settings=settings, worker=worker))
    return results


def process_email(email: InboundEmail, settings: Settings | None = None, worker: str = "local") -> dict:
    """Synchronous accept + run (IMAP poller, CLI, direct upload)."""
    result = accept_email(email, settings)
    out = result.as_dict()
    out["results"] = [run_job(j, result.message_id, settings=settings, worker=worker) for j in result.jobs]
    return out
