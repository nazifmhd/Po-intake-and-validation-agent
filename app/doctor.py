"""Is everything that is configured actually working?   python -m app.cli check

Costs nothing (no model is called) and never prints a secret.
"""
from __future__ import annotations

from typing import Callable

from app.config import Settings

OK, OFF, BAD = "ok", "off", "FAIL"


def _llm(s: Settings) -> tuple[str, str]:
    if s.llm_provider == "heuristic":
        return OFF, "offline parser; scans go to review"
    if s.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(**({"api_key": s.anthropic_api_key} if s.anthropic_api_key else {}), timeout=20)
        return OK, f"key accepted, {client.models.retrieve(s.anthropic_model).id}"
    import openai

    return OK, f"key accepted, {openai.OpenAI(api_key=s.openai_api_key, timeout=20).models.retrieve(s.openai_model).id}"


def _knowledge_base(s: Settings) -> tuple[str, str]:
    from app.rag.store import get_index

    if s.embedding_provider == "none":
        return OFF, "embeddings disabled: only exact vendor names are accepted"
    index = get_index(s)
    return (OK, "built and current") if index is not None and index.is_built() else \
        (BAD, "not built: python -m app.cli kb build")


def _webhook(s: Settings) -> tuple[str, str]:
    key = s.mailgun_signing_key if s.webhook_provider == "mailgun" else s.webhook_shared_secret
    return (OK, f"{s.webhook_provider} secret set") if s.real_secret(key) else (BAD, "run python scripts/init_env.py")


def _api(s: Settings) -> tuple[str, str]:
    return (OK, "key set") if s.real_secret(s.api_key) else (OFF, "API_KEY empty: API closed except webhook and /health")


def _imap(s: Settings) -> tuple[str, str]:
    if not (s.imap_user or s.imap_password):
        return OFF, "IMAP_USER / IMAP_PASSWORD empty"
    from app.intake.imap_poller import candidates, open_mailbox, require_safe_config

    require_safe_config(s)
    with open_mailbox(s) as mailbox:
        waiting = candidates(mailbox, s)
    return OK, f"signed in, folder {s.imap_folder}, {len(waiting)} unread message(s) from allowed senders"


def _smtp(s: Settings) -> tuple[str, str]:
    from app.intake.mailer import connect, smtp_ready

    if not (s.smtp_user or s.smtp_password):
        return OFF, "SMTP_USER / SMTP_PASSWORD empty (only needed to send test emails)"
    if not smtp_ready(s):
        return BAD, "SMTP_USER, SMTP_PASSWORD and TEST_EMAIL_TO (or IMAP_USER) are all needed"
    connect(s).quit()
    return OK, "signed in"


def _slack(s: Settings) -> tuple[str, str]:
    return (OK, "webhook set (not posted to)") if s.slack_webhook_url else (OFF, "messages are only logged")


def _sheets(s: Settings) -> tuple[str, str]:
    return (OK, "enabled") if s.google_sheets_enabled else (OFF, "optional mirror disabled")


CHECKS: list[tuple[str, Callable[[Settings], tuple[str, str]]]] = [
    ("Model", _llm), ("Knowledge base", _knowledge_base), ("Webhook", _webhook), ("API", _api),
    ("Gmail in (IMAP)", _imap), ("Gmail out (SMTP)", _smtp), ("Slack", _slack), ("Google Sheets", _sheets),
]


def _explain(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    if "AUTHENTICATIONFAILED" in text or "Invalid credentials" in text or "Username and Password not accepted" in text:
        return "sign-in refused: use a 16-letter Google App Password, not your normal password"
    if "Application-specific password required" in text:
        return "Google requires an App Password for this account"
    return f"{type(exc).__name__}: {text[:200]}"


def run(settings: Settings) -> list[tuple[str, str, str]]:
    out = []
    for name, check in CHECKS:
        try:
            status, detail = check(settings)
        except Exception as exc:
            status, detail = BAD, _explain(exc)
        out.append((name, status, detail))
    return out
