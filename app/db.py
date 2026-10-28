"""Engine, session helper, and schema creation (incl. append-only triggers for the audit log)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import event, text
from sqlmodel import Session, SQLModel, create_engine

from app import models  # noqa: F401  (registers tables)
from app.config import Settings, get_settings

_engines: dict[str, object] = {}

_AUDIT_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
       BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;""",
    """CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
       BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;""",
]


def get_engine(settings: Settings | None = None):
    settings = settings or get_settings()
    url = settings.resolved_database_url
    if url not in _engines:
        if url.startswith("sqlite:///"):
            Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _):  # WAL: API, poller and review UI share the file
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        else:
            engine = create_engine(url, pool_pre_ping=True)
        SQLModel.metadata.create_all(engine)
        if url.startswith("sqlite"):
            with engine.begin() as conn:
                for ddl in _AUDIT_TRIGGERS:
                    conn.execute(text(ddl))
        _engines[url] = engine
    return _engines[url]


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    with Session(get_engine(settings), expire_on_commit=False) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def reset_engines() -> None:
    for eng in _engines.values():
        eng.dispose()
    _engines.clear()
