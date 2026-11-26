"""The review desk, page by page, through Streamlit's own test runner."""
import hashlib
import hmac
import re
from types import SimpleNamespace

import pytest
from sqlmodel import select

from app.config import PROJECT_ROOT, reset_settings_cache
from app.db import session_scope
from app.graph.build import run_pipeline
from app.intake.common import InboundAttachment, InboundEmail, process_email
from app.models import PurchaseOrderRecord, ReviewQueueItem

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def open_page(page="overview", **params) -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "review_ui" / "app.py"), default_timeout=60)
    at.query_params["page"] = page
    for k, v in params.items():
        at.query_params[k] = str(v)
    at.run()
    assert not at.exception
    return at


def html(at) -> str:
    return "\n".join(e.proto.body for e in at.get("html"))


def field(at, label):
    return next(t for t in at.text_input if t.label == label)


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception


@pytest.fixture
def mailbox(samples):
    def send(*keys):
        return process_email(InboundEmail(f"<{'-'.join(keys)}@test>", "dana@northbridge.example", "PO", [
            InboundAttachment(samples[k].name, samples[k].read_bytes()) for k in keys]))
    return send


@pytest.mark.parametrize("page", ["overview", "send", "queue", "filed", "knowledge", "audit", "no-such-page"])
def test_every_page_opens_on_an_empty_system(page):
    open_page(page)


def test_every_page_opens_with_data(mailbox):
    mailbox("01_clean_northwind", "03_unknown_vendor", "05_blocked_vendor", "edge_locked", "edge_unsupported")
    for page in ("overview", "send", "queue", "filed", "knowledge", "audit"):
        open_page(page)


def test_overview_counts_every_outcome(mailbox):
    from review_ui import data
    mailbox("01_clean_northwind", "03_unknown_vendor", "05_blocked_vendor", "edge_unsupported")
    s = data.summary()
    assert (s["processed"], s["auto"], s["waiting"], s["refused"]) == (3, 1, 2, 1)
    assert s["value_filed"] == 4200.0 and dict(s["flags"])["vendor_blocked"] == 1
    statuses = {r["file"]: r["status"] for r in data.activity()}
    assert statuses["01_clean_northwind.pdf"] == "approved" and statuses["edge_unsupported.zip"] == "refused"
    assert "Blocked vendor" in html(open_page("overview"))


def test_queue_explains_the_flags_in_plain_words(samples):
    run_pipeline(str(samples["05_blocked_vendor"]))
    body = html(open_page("queue"))
    assert "Blocked vendor" in body and "vendor_blocked" in body and "Reject and report to Compliance" in body


def test_a_check_that_had_nothing_to_check_is_not_shown_as_passed(samples, monkeypatch):
    from app.extraction.extractor import extract

    def without_lines(doc):
        po = extract(doc)
        po.line_items = []
        return po
    monkeypatch.setattr("app.graph.nodes.extract", without_lines)
    run_pipeline(str(samples["01_clean_northwind"]))
    body = html(open_page("queue"))
    for label in ("Items and prices", "Arithmetic"):
        assert f'<div class="check na"><em>not possible</em><b>{label}</b>' in body


def test_correct_then_approve(samples):
    run_pipeline(str(samples["06_math_error"]))
    at = open_page("queue")
    field(at, "Reviewer name").set_value("dana")
    field(at, "Total").set_value("2021.40")
    click(at, "Apply corrections and re-check")
    assert any("no flags left" in e.value for e in at.success)
    click(at, "Approve and file")
    with session_scope() as s:
        rec = s.exec(select(PurchaseOrderRecord)).one()
    assert rec.total == 2021.40 and rec.approved_by == "dana"


def test_rules_are_enforced_and_explained(samples):
    run_pipeline(str(samples["05_blocked_vendor"]))
    at = open_page("queue")
    click(at, "Approve and file")
    assert any("Reviewer name is required" in e.value for e in at.error)
    field(at, "Reviewer name").set_value("dana")
    at.text_area[0].set_value("please")
    click(at, "Approve and file")
    assert any("BLOCKED" in e.value for e in at.error)
    at.text_area[0].set_value("reported to Compliance")
    click(at, "Reject")
    with session_scope() as s:
        item = s.exec(select(ReviewQueueItem)).one()
    assert item.status == "rejected" and item.review_note == "reported to Compliance"


def test_document_text_cannot_inject_markup(samples, monkeypatch):
    from app.extraction.extractor import extract
    evil = "<img src=x onerror=alert(1)><script>bad()</script>"

    def injected(doc):
        po = extract(doc)
        po.vendor_name.value = po.vendor_name.evidence = po.line_items[0].description = evil
        return po
    monkeypatch.setattr("app.graph.nodes.extract", injected)
    run_pipeline(str(samples["01_clean_northwind"]))
    for page in ("queue", "overview", "audit"):
        body = html(open_page(page))
        assert "<script>bad" not in body and "<img src=x" not in body
    assert "&lt;script&gt;" in html(open_page("queue"))


def test_send_page_posts_a_signed_email(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append((url, kw))
        return SimpleNamespace(status_code=200, json=lambda: {"jobs": [{}], "rejected": [], "status": "accepted"})
    monkeypatch.setattr("app.intake.simulate.httpx.post", fake_post)
    at = open_page("send")
    click(at, "Send")
    url, kw = calls[0]
    assert url.endswith("/webhook/inbound-email") and kw["files"][0][1][0] == "01_clean_northwind.pdf"
    d = kw["data"]
    assert d["signature"] == hmac.new(b"test-signing-key", (d["timestamp"] + d["token"]).encode(),
                                      hashlib.sha256).hexdigest()
    assert any("Accepted: 1 document" in e.value for e in at.success)
    click(at, "Send with a forged signature")
    assert calls[1][1]["data"]["signature"] == "0" * 64


def test_send_page_survives_an_unreachable_api(monkeypatch):
    import httpx

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("app.intake.simulate.httpx.post", refuse)
    at = open_page("send")
    click(at, "Send")
    assert any("did not answer" in e.value for e in at.error)


def test_flagged_values_are_located_on_the_document(samples):
    from review_ui import document
    out = run_pipeline(str(samples["06_math_error"]))
    hot, rest = document.marks_for(out["po"], out["flags"], out["price_checks"])
    assert "2,121.40" in hot and "PO-2026-0430" in rest
    path = samples["06_math_error"]
    pages, found, _ = document.render(str(path), path.stat().st_mtime, tuple(hot), ())
    plain, none, _ = document.render(str(path), path.stat().st_mtime, (), ())
    assert found >= 1 and none == 0 and pages[0] != plain[0]


def test_every_flag_the_agent_can_raise_has_a_plain_title():
    from app.scoring import POLICY_FLAG_CODES
    from review_ui.theme import FLAGS
    raised = set(POLICY_FLAG_CODES)
    for f in ("app/scoring.py", "app/graph/nodes.py", "app/ingestion/loaders.py", "app/extraction/extractor.py"):
        text = (PROJECT_ROOT / f).read_text(encoding="utf-8")
        raised |= set(re.findall(r'flag\("([a-z_]+)"', text)) | set(re.findall(r'flags\["([a-z_]+)"\]', text))
        raised |= set(re.findall(r'"code": "([a-z_]+)"', text)) | set(re.findall(r'IngestionError\("([a-z_]+)"', text))
    assert len(raised) > 25 and raised - set(FLAGS) - {"queue_error"} == set()


def test_password_hides_everything(monkeypatch, samples):
    monkeypatch.setenv("REVIEW_UI_PASSWORD", "hunter2")
    reset_settings_cache()
    run_pipeline(str(samples["06_math_error"]))
    at = open_page("queue")
    assert not at.button and "PO-2026-0430" not in html(at)
    at.text_input[0].set_value("hunter2").run()
    assert "PO-2026-0430" in html(at)


def test_uncertain_vendor_is_not_preselected(samples):
    run_pipeline(str(samples["10_lookalike_vendor"]))
    at = open_page("queue")
    assert at.selectbox[0].value == "No record chosen"
    assert "resembles Acme Industrial Supplies Inc., not confirmed" in html(at)


def test_send_page_can_use_real_email(monkeypatch):
    sent = []
    monkeypatch.setenv("SMTP_USER", "me@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "app-password")
    monkeypatch.setenv("IMAP_USER", "po@gmail.com")
    reset_settings_cache()
    monkeypatch.setattr("app.intake.mailer.connect", lambda s: pytest.fail("must not connect"))
    monkeypatch.setattr("review_ui.pages.send_email", lambda files, s, subject: sent.append(files[0][0]))
    at = open_page("send")
    click(at, "Send")
    assert sent == ["01_clean_northwind.pdf"] and any("po@gmail.com" in e.value for e in at.success)
