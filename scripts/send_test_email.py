"""Send a real email with PO attachments (for the IMAP intake path, or a Mailgun/SendGrid route).

    python scripts/send_test_email.py data/samples/01_clean_northwind.pdf [more files] [--to inbox@example.com]

Uses SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD and TEST_EMAIL_TO from .env
(for Gmail use an App Password).
"""
from __future__ import annotations

import argparse
import mimetypes
import smtplib
import sys
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class SmtpSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    test_email_to: str = ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--to", default=None)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()
    s = SmtpSettings()
    to = args.to or s.test_email_to
    if not (s.smtp_user and s.smtp_password and to):
        sys.exit("Set SMTP_USER, SMTP_PASSWORD and TEST_EMAIL_TO (or --to) in .env")

    msg = EmailMessage()
    msg["From"], msg["To"] = s.smtp_user, to
    msg["Subject"] = args.subject or f"Purchase order: {', '.join(Path(f).stem for f in args.files)}"
    msg["Message-ID"] = make_msgid(domain="po-agent.test")
    msg.set_content("Hello,\n\nPlease find our purchase order(s) attached.\n\nRegards,\nProcurement")
    for f in args.files:
        p = Path(f)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(p.read_bytes(), maintype=maintype, subtype=subtype, filename=p.name)

    with smtplib.SMTP(s.smtp_host, s.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(s.smtp_user, s.smtp_password)
        smtp.send_message(msg)
    print(f"Sent {msg['Message-ID']} to {to} with {len(args.files)} attachment(s)")


if __name__ == "__main__":
    main()
