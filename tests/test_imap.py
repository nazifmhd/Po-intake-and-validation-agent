"""The IMAP fallback, against a fake mailbox."""
from types import SimpleNamespace

import imap_tools
import pytest
from sqlmodel import select

from app.config import get_settings
from app.db import session_scope
from app.intake.imap_poller import poll_once
from app.models import ProcessedEmail, PurchaseOrderRecord


@pytest.fixture
def mailbox(monkeypatch):
    class FakeMailBox:
        messages, seen = [], []

        def __init__(self, host, port):
            pass

        def login(self, *a, **k):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def fetch(self, *a, **k):
            return [m for m in self.messages if m.uid not in self.seen]

        def flag(self, uid, flag, value):
            self.seen.append(uid)

    FakeMailBox.messages, FakeMailBox.seen = [], []
    monkeypatch.setattr(imap_tools, "MailBox", FakeMailBox)
    return FakeMailBox


def message(samples, uid, *keys):
    return SimpleNamespace(
        uid=uid, headers={"message-id": (f"<imap-{uid}@test>",)}, from_="buyer@northbridge.example", subject="PO",
        attachments=[SimpleNamespace(filename=samples[k].name, payload=samples[k].read_bytes(), content_type="")
                     for k in keys])


@pytest.fixture
def settings():
    return get_settings().model_copy(update={"imap_user": "u", "imap_password": "p"})


def test_unread_mail_is_processed_once_and_marked_seen(mailbox, samples, settings):
    mailbox.messages = [message(samples, "1", "01_clean_northwind", "03_unknown_vendor"), message(samples, "2")]
    out = poll_once(settings)
    assert [r["decision"] for r in out[0]["results"]] == ["auto_approve", "needs_review"]
    assert out[1]["status"] == "no_attachment" and mailbox.seen == ["1", "2"]
    assert poll_once(settings) == []


def test_crash_before_mark_seen_does_not_file_twice(mailbox, samples, settings):
    mailbox.messages = [message(samples, "1", "01_clean_northwind")]
    poll_once(settings)
    mailbox.seen.clear()                                     # the flag never reached the server
    assert poll_once(settings)[0]["status"] == "duplicate"
    with session_scope() as s:
        assert len(s.exec(select(PurchaseOrderRecord)).all()) == 1


def test_mail_still_being_accepted_elsewhere_stays_unread(mailbox, samples, settings):
    mailbox.messages = [message(samples, "1", "01_clean_northwind")]
    with session_scope() as s:
        s.add(ProcessedEmail(message_id="<imap-1@test>", status="received"))
    assert poll_once(settings)[0]["status"] == "in_progress" and mailbox.seen == []
