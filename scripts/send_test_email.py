"""Send a real email with PO attachments (for the IMAP intake path, or a Mailgun/SendGrid route).

    python scripts/send_test_email.py data/samples/01_clean_northwind.pdf [more files] [--to inbox@example.com]

Uses SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD and TEST_EMAIL_TO from .env
(for Gmail use an App Password).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.config import get_settings
    from app.intake.mailer import send_email, smtp_ready

    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--to", default=None)
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()
    s = get_settings()
    if not smtp_ready(s) and not (args.to and s.smtp_user and s.smtp_password):
        sys.exit("Set SMTP_USER, SMTP_PASSWORD and TEST_EMAIL_TO (or --to) in .env")
    files = [(Path(f).name, Path(f).read_bytes()) for f in args.files]
    subject = args.subject or f"Purchase order: {', '.join(Path(f).stem for f in args.files)}"
    print(f"Sent {send_email(files, s, subject=subject, to=args.to)} with {len(files)} attachment(s)")


if __name__ == "__main__":
    main()
