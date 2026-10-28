"""Fallback trigger: poll a dedicated mailbox over IMAP (no public URL, no DNS, no signup).

    python -m app.intake.imap_poller          # poll every IMAP_POLL_SECONDS
    python -m app.intake.imap_poller --once   # single pass

Unread messages are processed and then marked \\Seen. Dedup on Message-ID still applies, so a
crash between processing and marking seen can't create duplicates on the next pass.
"""
from __future__ import annotations

import argparse
import logging

from app.config import Settings, get_settings
from app.intake.common import InboundAttachment, InboundEmail, process_email
from app.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def poll_once(settings: Settings | None = None) -> list[dict]:
    from imap_tools import AND, MailBox

    settings = settings or get_settings()
    if not (settings.imap_user and settings.imap_password):
        raise RuntimeError("IMAP_USER / IMAP_PASSWORD are not configured")
    results = []
    with MailBox(settings.imap_host, port=settings.imap_port).login(
            settings.imap_user, settings.imap_password, initial_folder=settings.imap_folder) as mailbox:
        for msg in mailbox.fetch(AND(seen=False), mark_seen=False, bulk=False):
            message_id = (msg.headers.get("message-id") or [None])[0]
            email = InboundEmail(
                message_id=message_id.strip() if message_id else None, sender=msg.from_, subject=msg.subject,
                attachments=[InboundAttachment(a.filename or "attachment", a.payload, a.content_type)
                             for a in msg.attachments],
                source="imap",
            )
            try:
                out = process_email(email, settings)
                results.append(out)
                logger.info("IMAP message processed", extra={"message_id": out["message_id"], "status": out["status"],
                                                             "jobs": len(out["jobs"])})
            except Exception:
                logger.exception("IMAP message failed; leaving it unread for the next pass",
                                 extra={"uid": msg.uid})
                continue
            mailbox.flag(msg.uid, "\\Seen", True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="IMAP polling intake")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    from app.rag.ingest_kb import build_knowledge_base
    build_knowledge_base(settings, if_missing=True)

    if args.once:
        poll_once(settings)
        return

    from apscheduler.schedulers.blocking import BlockingScheduler

    def job():
        try:
            poll_once(settings)
        except Exception:
            logger.exception("IMAP poll failed")

    scheduler = BlockingScheduler()
    scheduler.add_job(job, "interval", seconds=settings.imap_poll_seconds, max_instances=1, coalesce=True,
                      next_run_time=__import__("datetime").datetime.now())
    logger.info("IMAP poller started", extra={"host": settings.imap_host, "every_s": settings.imap_poll_seconds})
    scheduler.start()


if __name__ == "__main__":
    main()
