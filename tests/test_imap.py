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
        messages, seen, downloaded, login_args = [], [], [], []

        def __init__(self, host, port):
            pass

        def login(self, *a, **k):
            self.login_args[:] = a
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def fetch(self, criteria, **k):
            if k.get("headers_only"):
                return [m for m in self.messages if m.uid not in self.seen]
            self.downloaded.append(str(criteria))
            return [m for m in self.messages if f"UID {m.uid}" in str(criteria)]

        def flag(self, uid, flag, value):
            self.seen.append(uid)

    FakeMailBox.messages, FakeMailBox.seen, FakeMailBox.downloaded, FakeMailBox.login_args = [], [], [], []
    monkeypatch.setattr(imap_tools, "MailBox", FakeMailBox)
    return FakeMailBox


def message(samples, uid, *keys, sender="buyer@northbridge.example", auth=None):
    headers = {"message-id": (f"<imap-{uid}@test>",)}
    if auth:
        headers["authentication-results"] = (auth,)
    return SimpleNamespace(
        uid=uid, headers=headers, from_=sender, subject="PO",
        attachments=[SimpleNamespace(filename=samples[k].name, payload=samples[k].read_bytes(), content_type="")
                     for k in keys])


@pytest.fixture
def settings():
    return get_settings().model_copy(update={"imap_user": "u", "imap_password": "abcd efgh ijkl mnop",
                                               "allowed_senders": "buyer@northbridge.example"})


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


def test_other_peoples_mail_is_never_downloaded_or_marked_read(mailbox, samples, settings):
    mailbox.messages = [message(samples, "7", "01_clean_northwind", sender="friend@gmail.com"),
                        message(samples, "8", "01_clean_northwind", sender="Buyer <BUYER@northbridge.example>")]
    assert len(poll_once(settings)) == 1
    assert mailbox.seen == ["8"] and all("UID 7" not in d for d in mailbox.downloaded)
    with session_scope() as s:
        assert [e.sender for e in s.exec(select(ProcessedEmail)).all()] == ["buyer@northbridge.example"]


def test_whole_inbox_is_refused_without_knowing_the_senders(mailbox, settings):
    from app.intake.imap_poller import ImapConfigError
    open_inbox = settings.model_copy(update={"allowed_senders": ""})
    with pytest.raises(ImapConfigError, match="Refusing to read a whole inbox"):
        poll_once(open_inbox)
    assert poll_once(open_inbox.model_copy(update={"imap_folder": "PO"})) == []  # a dedicated label is fine


def test_number_of_messages_per_poll_is_capped(mailbox, samples, settings):
    mailbox.messages = [message(samples, str(i)) for i in range(1, 8)]
    assert len(poll_once(settings.model_copy(update={"imap_max_per_poll": 3}))) == 3


def test_app_password_groups_are_joined(mailbox, settings):
    poll_once(settings)
    assert mailbox.login_args == ["u", "abcdefghijklmnop"]


@pytest.mark.parametrize("verdict,refused", [
    ("mx.google.com; dkim=pass; spf=pass; dmarc=pass", False),
    ("mx.google.com; dkim=pass; spf=softfail; dmarc=pass", False),   # forwarded mail
    ("mx.google.com; spf=fail smtp.mailfrom=x; dkim=none", True),
    ("mx.google.com; dkim=fail; spf=pass; dmarc=fail (p=REJECT)", True), (None, False)])
def test_forged_sender_is_refused(mailbox, samples, settings, verdict, refused):
    mailbox.messages = [message(samples, "1", "01_clean_northwind", auth=verdict)]
    out = poll_once(settings)[0]
    assert (out["status"] == "rejected_sender") is refused
    with session_scope() as s:
        assert len(s.exec(select(PurchaseOrderRecord)).all()) == (0 if refused else 1)


def test_sender_list_takes_addresses_and_domains():
    from app.intake.common import sender_allowed
    s = get_settings().model_copy(update={"allowed_senders": "a@gmail.com", "allowed_sender_domains": "corp.example"})
    assert sender_allowed("A <A@gmail.com>", s) and sender_allowed("x@corp.example", s)
    assert not sender_allowed("b@gmail.com", s) and not sender_allowed("x@evil-corp.example", s)
    assert not sender_allowed("a@gmail.com.evil.example", s) and not sender_allowed(None, s)


# ------------------------------------------------------------------ configuration check
def test_check_reports_without_leaking_secrets(mailbox, settings, capsys, monkeypatch):
    from app import doctor
    s = settings.model_copy(update={"mailgun_signing_key": "very-secret", "api_key": "other-secret"})
    report = {n: (st, d) for n, st, d in doctor.run(s)}
    assert report["Gmail in (IMAP)"][0] == "ok" and report["Slack"][0] == "off" and report["Model"][0] == "off"
    assert "secret" not in str(report).replace("mailgun secret set", "") and "mnop" not in str(report)


def test_check_explains_a_wrong_password(mailbox, settings, monkeypatch):
    from app import doctor

    def refuse(s):
        raise RuntimeError("b'[AUTHENTICATIONFAILED] Invalid credentials (Failure)'")
    monkeypatch.setattr("app.intake.imap_poller.open_mailbox", refuse)
    report = {n: (st, d) for n, st, d in doctor.run(settings)}
    assert report["Gmail in (IMAP)"][0] == "FAIL" and "App Password" in report["Gmail in (IMAP)"][1]


def test_real_email_carries_the_attachment(samples, monkeypatch, settings):
    from app.intake import mailer
    sent = []

    class FakeSMTP:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def send_message(self, msg):
            sent.append(msg)
    monkeypatch.setattr(mailer, "connect", lambda s: FakeSMTP())
    s = settings.model_copy(update={"smtp_user": "me@gmail.com", "smtp_password": "x", "imap_user": "po@gmail.com"})
    assert mailer.smtp_ready(s)
    p = samples["01_clean_northwind"]
    mailer.send_email([(p.name, p.read_bytes())], s, subject="PO")
    assert sent[0]["To"] == "po@gmail.com" and [a.get_filename() for a in sent[0].iter_attachments()] == [p.name]
