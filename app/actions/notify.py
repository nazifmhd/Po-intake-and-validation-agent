"""Slack notifications. A notification failure is logged, never allowed to change a decision."""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def send_slack(text: str, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if not settings.slack_webhook_url:
        logger.info("Slack (not configured) would post", extra={"slack_text": text})
        return False
    try:
        r = httpx.post(settings.slack_webhook_url, json={"text": text}, timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Slack notification failed", extra={"error": str(exc)})
        return False


def _money(total: Optional[float], currency: Optional[str]) -> str:
    return f"{currency or ''} {total:,.2f}".strip() if total is not None else "n/a"


def notify_approved(po: dict, vendor_match: dict, record_id: int, settings: Settings | None = None) -> bool:
    vendor = vendor_match.get("canonical_name") or po["vendor_name"]["value"]
    vid = vendor_match.get("vendor_id") or "-"
    return send_slack(
        f":white_check_mark: *PO {po['po_number']['value']}* auto-approved and filed (record #{record_id})\n"
        f"Vendor: {vendor} ({vid}) · Total: {_money(po.get('total'), po['currency']['value'])}",
        settings,
    )


def notify_review(po: Optional[dict], flags: list[str], reasons: dict, review_id: int,
                  vendor_match: Optional[dict], settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    po_no = (po or {}).get("po_number", {}).get("value") or "(unknown PO)"
    vendor = (po or {}).get("vendor_name", {}).get("value") or "unknown vendor"
    lines = "\n".join(f"• `{f}` - {reasons.get(f, '')}"[:300] for f in flags)
    return send_slack(
        f":warning: *PO {po_no}* from _{vendor}_ needs review (queue #{review_id})\n{lines}\n"
        f"<{settings.review_ui_url}|Open review queue>",
        settings,
    )


def notify_rejected_attachment(filename: str, reason: str, sender: Optional[str],
                               settings: Settings | None = None) -> bool:
    return send_slack(f":no_entry: Couldn't process attachment *{filename}* from {sender or 'unknown sender'}: {reason}",
                      settings)
