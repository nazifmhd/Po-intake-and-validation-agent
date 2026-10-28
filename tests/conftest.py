"""Tests run fully offline: heuristic extractor, lexical retrieval, temp DB + storage per test."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def samples(tmp_path_factory) -> dict[str, Path]:
    from make_samples import make_samples

    return make_samples(tmp_path_factory.mktemp("samples"))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    env = {
        "LLM_PROVIDER": "heuristic",
        "EMBEDDING_PROVIDER": "none",
        "RAG_ENABLED": "true",
        "POLICY_LLM_REVIEW": "false",
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "STORAGE_DIR": str(tmp_path / "storage"),
        "CHROMA_PATH": str(tmp_path / "chroma"),
        "SLACK_WEBHOOK_URL": "",
        "GOOGLE_SHEETS_ENABLED": "false",
        "ALLOWED_SENDER_DOMAINS": "",
        "MAILGUN_SIGNING_KEY": "test-signing-key",
        "WEBHOOK_SHARED_SECRET": "test-shared-secret",
        "LOG_JSON": "false",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from app.config import reset_settings_cache
    from app.db import reset_engines
    from app.rag.store import reset_index_cache

    reset_settings_cache()
    reset_engines()
    reset_index_cache()
    yield
    reset_engines()
    reset_settings_cache()
    reset_index_cache()
