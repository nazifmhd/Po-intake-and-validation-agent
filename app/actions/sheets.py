"""Optional Google Sheets mirror: "Approved" and "Needs Review" tabs.

Disabled unless GOOGLE_SHEETS_ENABLED=true and gspread + a service-account file are present.
Purely a visual mirror for demos; the database stays the system of record.
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
APPROVED_HEADER = ["Record #", "PO Number", "PO Date", "Vendor", "Vendor ID", "Total", "Currency", "Status", "Filed At"]
REVIEW_HEADER = ["Queue #", "PO Number", "Vendor (as printed)", "Total", "Currency", "Flags", "Status", "Queued At"]


def _worksheet(title: str, header: list[str], settings: Settings):
    import gspread

    gc = gspread.service_account(filename=settings.google_service_account_file)
    sh = gc.open_by_key(settings.google_sheets_spreadsheet_id)
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=len(header))
        ws.append_row(header)
    return ws


def mirror_row(tab: str, row: list, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if not settings.google_sheets_enabled:
        return False
    try:
        header = APPROVED_HEADER if tab == "Approved" else REVIEW_HEADER
        _worksheet(tab, header, settings).append_row([("" if v is None else str(v)) for v in row])
        return True
    except Exception as exc:
        logger.error("Google Sheets mirror failed", extra={"error": str(exc), "tab": tab})
        return False
