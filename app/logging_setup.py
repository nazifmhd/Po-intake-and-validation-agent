"""Structured JSON logging. Every node logs its job_id, duration and outcome."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = True, stream=None) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "pdfminer", "apscheduler", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
