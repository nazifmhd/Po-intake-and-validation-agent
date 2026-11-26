"""Fallback trigger: poll a mailbox (e.g. Gmail with an App Password) over IMAP - no public URL, no DNS, no signup.

    python -m app.intake.imap_poller          # poll every IMAP_POLL_SECONDS
    python -m app.intake.imap_poller --once   # single pass

Safe to point at a mailbox that also receives other mail: it refuses to start unless it knows which mail is meant
for it (ALLOWED_SENDERS / ALLOWED_SENDER_DOMAINS, or a dedicated folder or Gmail label), looks only at recent unread
messages, reads their headers first, and leaves everything else untouched - not downloaded, not marked as read.

Processed messages are marked \\Seen. Dedup on Message-ID still applies, so a crash between processing and marking
seen can't create duplicates on the next pass - and the job sweeper finishes whatever the crash interrupted.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime, timedelta

from app.config import Settings, get_settings
from app.intake.common import InboundAttachment, InboundEmail, process_email, sender_allowed
from app.logging_setup import configure_logging

logger = logging.getLogger(__name__)


class ImapConfigError(RuntimeError):
    pass


def require_safe_config(settings: Settings) -> None:
    if not (settings.imap_user and settings.imap_password):
        raise ImapConfigError("IMAP_USER / IMAP_PASSWORD are not configured")
    scoped = settings.allowed_sender_list or settings.allowed_sender_domain_list
    if not scoped and settings.imap_folder.strip().upper() == "INBOX":
        raise ImapConfigError("Refusing to read a whole inbox: set ALLOWED_SENDERS to the addresses that send POs, or "
                              "IMAP_FOLDER to a folder / Gmail label that only contains POs")


def open_mailbox(settings: Settings):
    from imap_tools import MailBox

    password = "".join(settings.imap_password.split())  # Gmail shows App Passwords in groups of four
    return MailBox(settings.imap_host, port=settings.imap_port).login(
        settings.imap_user, password, initial_folder=settings.imap_folder)


def spoofed(headers: dict) -> bool:
    """The receiving server's own verdict (topmost header). Only an explicit failure counts: the From address
    alone is trivial to forge, so an allow-list would be worthless without this."""
    verdict = " ".join((headers.get("authentication-results") or [""])[0].lower().split())
    if not verdict:
        return False
    if re.search(r"\bdmarc=fail\b", verdict):
        return True
    return bool(re.search(r"\bspf=(fail|softfail)\b", verdict)) and not re.search(r"\bdkim=pass\b", verdict)


def candidates(mailbox, settings: Settings) -> list:
    """Headers only: recent, unread, from a sender that is allowed to send POs."""
    from imap_tools import AND

    since = date.today() - timedelta(days=settings.imap_since_days)
    found = mailbox.fetch(AND(seen=False, date_gte=since), mark_seen=False, headers_only=True, bulk=True)
    return [m for m in found if sender_allowed(m.from_, settings)][: settings.imap_max_per_poll]


def poll_once(settings: Settings | None = None) -> list[dict]:
    from imap_tools import AND

    settings = settings or get_settings()
    require_safe_config(settings)
    results = []
    with open_mailbox(settings) as mailbox:
        for head in candidates(mailbox, settings):
            msg = next(iter(mailbox.fetch(AND(uid=head.uid), mark_seen=False)), None)
            if msg is None:  # moved or deleted meanwhile
                continue
            message_id = (msg.headers.get("message-id") or [None])[0]
            email = InboundEmail(
                message_id=message_id.strip() if message_id else None, sender=msg.from_, subject=msg.subject,
                attachments=[InboundAttachment(a.filename or "attachment", a.payload, a.content_type)
                             for a in msg.attachments],
                source="imap", sender_verified=not spoofed(msg.headers),
            )
            try:
                out = process_email(email, settings, worker="imap")
                results.append(out)
                logger.info("IMAP message processed", extra={"message_id": out["message_id"], "status": out["status"],
                                                             "jobs": len(out["jobs"])})
            except Exception:
                logger.exception("IMAP message failed; leaving it unread for the next pass",
                                 extra={"uid": msg.uid})
                continue
            if out["status"] == "in_progress":  # another process holds the claim; decide on the next pass
                continue
            mailbox.flag(msg.uid, "\\Seen", True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="IMAP polling intake")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        require_safe_config(settings)
    except ImapConfigError as exc:
        sys.exit(f"IMAP poller not started: {exc}")

    from app.rag.ingest_kb import build_knowledge_base
    build_knowledge_base(settings, if_missing=True)

    if args.once:
        poll_once(settings)
        return

    from app.intake.common import requeue_own_jobs, sweep_jobs
    requeue_own_jobs("imap", settings)

    from apscheduler.schedulers.blocking import BlockingScheduler

    def job():
        try:
            poll_once(settings)
        except Exception:
            logger.exception("IMAP poll failed")
        try:
            sweep_jobs("imap", settings)
        except Exception:
            logger.exception("Job sweep failed")

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", seconds=settings.imap_poll_seconds, max_instances=1, coalesce=True,
                      next_run_time=datetime.now())
    logger.info("IMAP poller started", extra={"host": settings.imap_host, "folder": settings.imap_folder,
                                              "every_s": settings.imap_poll_seconds})
    scheduler.start()


if __name__ == "__main__":
    main()
