"""Send a real email with PO attachments (SMTP) - used by scripts/send_test_email.py and the review UI."""
from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Optional

from app.config import Settings


def smtp_ready(settings: Settings) -> bool:
    return bool(settings.smtp_user and settings.smtp_password and (settings.test_email_to or settings.imap_user))


def connect(settings: Settings) -> smtplib.SMTP:
    smtp = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
    try:
        smtp.starttls()
        smtp.login(settings.smtp_user, "".join(settings.smtp_password.split()))
    except Exception:
        smtp.close()
        raise
    return smtp


def send_email(files: list[tuple[str, bytes]], settings: Settings, *, subject: str, to: Optional[str] = None) -> str:
    """-> Message-ID of the sent mail"""
    msg = EmailMessage()
    msg["From"], msg["To"] = settings.smtp_user, to or settings.test_email_to or settings.imap_user
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid(domain="po-agent.test")
    msg.set_content("Hello,\n\nPlease find our purchase order(s) attached.\n\nRegards,\nProcurement")
    for name, content in files:
        maintype, subtype = (mimetypes.guess_type(name)[0] or "application/octet-stream").split("/", 1)
        msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=name)
    with connect(settings) as smtp:
        smtp.send_message(msg)
    return msg["Message-ID"]
