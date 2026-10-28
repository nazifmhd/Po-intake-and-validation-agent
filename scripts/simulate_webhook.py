"""Post a correctly *signed* Mailgun-style inbound email to the running API - no domain or tunnel needed.

    python scripts/simulate_webhook.py data/samples/01_clean_northwind.pdf [more files] \
        [--url http://localhost:8000] [--provider mailgun|generic] [--message-id <id>] [--bad-signature]

Mailgun signs: HMAC-SHA256(MAILGUN_SIGNING_KEY, timestamp + token). --bad-signature shows the
spoofed-post rejection (HTTP 401). Re-sending with the same --message-id shows dedup.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import mimetypes
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--provider", default="mailgun", choices=["mailgun", "generic"])
    ap.add_argument("--message-id", default=None)
    ap.add_argument("--sender", default="Dana Buyer <dana@northbridge.example>")
    ap.add_argument("--subject", default="Purchase order attached")
    ap.add_argument("--bad-signature", action="store_true")
    args = ap.parse_args()
    s = get_settings()

    message_id = args.message_id or f"<{uuid.uuid4().hex}@mail.northbridge.example>"
    files = []
    for i, f in enumerate(args.files, start=1):
        p = Path(f)
        key = f"attachment-{i}" if args.provider == "mailgun" else f"file{i}"
        files.append((key, (p.name, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")))

    headers = {}
    if args.provider == "mailgun":
        ts, token = str(int(time.time())), uuid.uuid4().hex
        sig = hmac.new(s.mailgun_signing_key.encode(), f"{ts}{token}".encode(), hashlib.sha256).hexdigest()
        data = {"timestamp": ts, "token": token, "signature": "0" * 64 if args.bad_signature else sig,
                "Message-Id": message_id, "sender": args.sender, "from": args.sender, "subject": args.subject,
                "attachment-count": str(len(files))}
    else:
        data = {"message_id": message_id, "sender": args.sender, "subject": args.subject}
        headers["X-Webhook-Secret"] = "wrong" if args.bad_signature else s.webhook_shared_secret

    r = httpx.post(f"{args.url}/webhook/inbound-email?provider={args.provider}", data=data,
                   files=files or None, headers=headers, timeout=60)
    print(r.status_code, r.text)


if __name__ == "__main__":
    main()
