"""Append-only audit trail: every run, every rejection, every reviewer decision.

Written twice - to the `audit_log` table (UPDATE/DELETE blocked by triggers) and to
storage/audit.jsonl - so the trail survives even if the DB write fails.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()


def audit(event: str, *, job_id: Optional[str] = None, decision: Optional[str] = None,
          flags: Optional[list[str]] = None, details: Optional[dict[str, Any]] = None,
          actor: str = "agent", settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(), "job_id": job_id, "event": event, "actor": actor,
        "decision": decision, "flags": flags or [], "details": details or {},
    }
    try:
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        with _lock, open(settings.storage_path / "audit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.error("Audit file write failed", extra={"error": str(exc)})
    try:
        from app.db import session_scope
        from app.models import AuditLog

        with session_scope(settings) as s:
            s.add(AuditLog(job_id=job_id, event=event, actor=actor, decision=decision,
                           flags_json=json.dumps(flags or []), details_json=json.dumps(details or {}, default=str)))
    except Exception as exc:
        logger.error("Audit DB write failed", extra={"error": str(exc), "event": event})
    logger.info("audit", extra={"event": event, "job_id": job_id, "decision": decision, "flags": flags or []})
