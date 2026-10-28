"""Inbound-email webhook parsing + source verification.

mailgun   HMAC-SHA256(signing_key, timestamp + token) == signature, timestamp fresh, token never reused
sendgrid  Inbound Parse has no payload signature -> shared secret in the configured URL (?token=...)
generic   X-Webhook-Secret header (used by the n8n workflow and scripts/simulate_webhook.py)
"""
from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict

from starlette.datastructures import UploadFile

from app.config import Settings
from app.intake.common import InboundAttachment, InboundEmail


class WebhookAuthError(Exception):
    pass


class _ReplayCache:
    """Remembers recently seen Mailgun tokens (in-process; use Redis when running >1 replica)."""

    def __init__(self, max_items: int = 10_000):
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._max = max_items
        self._lock = threading.Lock()

    def seen_before(self, token: str, ttl: int) -> bool:
        now = time.time()
        with self._lock:
            for k in [k for k, t in self._seen.items() if now - t > ttl]:
                self._seen.pop(k, None)
            if token in self._seen:
                return True
            self._seen[token] = now
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)
            return False


replay_cache = _ReplayCache()


def verify_mailgun(timestamp: str, token: str, signature: str, settings: Settings) -> None:
    if not settings.mailgun_signing_key:
        raise WebhookAuthError("MAILGUN_SIGNING_KEY is not configured")
    if not (timestamp and token and signature):
        raise WebhookAuthError("missing Mailgun signature fields")
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError as exc:
        raise WebhookAuthError("invalid timestamp") from exc
    if age > settings.webhook_max_age_seconds:
        raise WebhookAuthError(f"stale webhook (timestamp {age:.0f}s old)")
    expected = hmac.new(settings.mailgun_signing_key.encode(), f"{timestamp}{token}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise WebhookAuthError("invalid Mailgun signature")
    if replay_cache.seen_before(token, settings.webhook_max_age_seconds):
        raise WebhookAuthError("replayed webhook token")


def verify_shared_secret(provided: str | None, settings: Settings) -> None:
    if not settings.webhook_shared_secret:
        raise WebhookAuthError("WEBHOOK_SHARED_SECRET is not configured")
    if not provided or not hmac.compare_digest(provided, settings.webhook_shared_secret):
        raise WebhookAuthError("invalid webhook secret")


def _message_id_from_headers(raw_headers: str) -> str | None:
    m = re.search(r"^Message-I[dD]:\s*(\S+)", raw_headers or "", re.M)
    return m.group(1) if m else None


async def _attachments(form) -> list[InboundAttachment]:
    out = []
    for _, value in form.multi_items():
        if isinstance(value, UploadFile):
            out.append(InboundAttachment(filename=value.filename or "attachment", content=await value.read(),
                                         content_type=value.content_type or ""))
    return out


async def parse_inbound(provider: str, form, headers, query, settings: Settings) -> InboundEmail:
    if provider == "mailgun":
        verify_mailgun(form.get("timestamp", ""), form.get("token", ""), form.get("signature", ""), settings)
        return InboundEmail(
            message_id=form.get("Message-Id") or form.get("message-id") or _message_id_from_headers(
                form.get("message-headers", "")),
            sender=form.get("sender") or form.get("from"), subject=form.get("subject"),
            attachments=await _attachments(form), source="webhook:mailgun",
        )
    if provider == "sendgrid":
        verify_shared_secret(query.get("token"), settings)
        return InboundEmail(
            message_id=_message_id_from_headers(form.get("headers", "")), sender=form.get("from"),
            subject=form.get("subject"), attachments=await _attachments(form), source="webhook:sendgrid",
        )
    if provider == "generic":
        verify_shared_secret(headers.get("x-webhook-secret"), settings)
        return InboundEmail(
            message_id=form.get("message_id"), sender=form.get("sender"), subject=form.get("subject"),
            attachments=await _attachments(form), source="webhook:generic",
        )
    raise WebhookAuthError(f"unknown provider '{provider}'")
