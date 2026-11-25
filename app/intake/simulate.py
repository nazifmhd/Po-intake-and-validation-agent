"""Post a PO to the webhook exactly as the email provider would - signed, multipart, with a Message-ID.

Used by scripts/simulate_webhook.py and by the "Send a PO" page of the review UI. It exercises the real
endpoint (signature check, dedup, background job); only the mail transport in front of it is simulated.
"""
from __future__ import annotations

import hashlib
import hmac
import mimetypes
import time
import uuid
from typing import Optional

import httpx

from app.config import Settings


class NotConfigured(Exception):
    pass


def post_email(files: list[tuple[str, bytes]], settings: Settings, *, url: str, sender: str, subject: str,
               provider: Optional[str] = None, message_id: Optional[str] = None,
               bad_signature: bool = False) -> httpx.Response:
    provider = provider or settings.webhook_provider
    secret = settings.real_secret(settings.mailgun_signing_key if provider == "mailgun"
                                  else settings.webhook_shared_secret)
    if not secret:
        raise NotConfigured("No webhook secret configured - run: python scripts/init_env.py")
    message_id = message_id or f"<{uuid.uuid4().hex}@mail.northbridge.example>"
    parts = [(f"attachment-{i}", (name, content, mimetypes.guess_type(name)[0] or "application/octet-stream"))
             for i, (name, content) in enumerate(files, start=1)]
    headers: dict[str, str] = {}
    params = {"provider": provider}
    if provider == "mailgun":  # HMAC-SHA256(signing key, timestamp + token)
        ts, token = str(int(time.time())), uuid.uuid4().hex
        sig = hmac.new(secret.encode(), f"{ts}{token}".encode(), hashlib.sha256).hexdigest()
        data = {"timestamp": ts, "token": token, "signature": "0" * 64 if bad_signature else sig,
                "Message-Id": message_id, "sender": sender, "from": sender, "subject": subject,
                "attachment-count": str(len(files))}
    elif provider == "sendgrid":
        params["token"] = "wrong" if bad_signature else secret
        data = {"headers": f"Message-ID: {message_id}\n", "from": sender, "subject": subject}
    else:
        headers["X-Webhook-Secret"] = "wrong" if bad_signature else secret
        data = {"message_id": message_id, "sender": sender, "subject": subject}
    return httpx.post(f"{url}/webhook/inbound-email", params=params, data=data, files=parts or None, headers=headers,
                      timeout=60)
