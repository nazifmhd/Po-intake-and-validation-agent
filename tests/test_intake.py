import hashlib
import hmac
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.config import get_settings
from app.db import session_scope
from app.intake.common import InboundAttachment, InboundEmail, accept_email, process_email, split_csv_batch
from app.intake.webhook import WebhookAuthError, verify_mailgun
from app.models import Job, ProcessedEmail


def sign(ts: str, token: str, key="test-signing-key") -> str:
    return hmac.new(key.encode(), f"{ts}{token}".encode(), hashlib.sha256).hexdigest()


def test_mailgun_signature_valid_then_replay_rejected():
    ts, tok = str(int(time.time())), "tok-" + str(time.time())
    verify_mailgun(ts, tok, sign(ts, tok), get_settings())
    with pytest.raises(WebhookAuthError, match="replayed"):
        verify_mailgun(ts, tok, sign(ts, tok), get_settings())


@pytest.mark.parametrize("ts_offset,bad_sig,msg", [(0, True, "invalid"), (-3600, False, "stale")])
def test_mailgun_signature_rejections(ts_offset, bad_sig, msg):
    ts, tok = str(int(time.time()) + ts_offset), f"tok-{ts_offset}-{bad_sig}"
    sig = "0" * 64 if bad_sig else sign(ts, tok)
    with pytest.raises(WebhookAuthError, match=msg):
        verify_mailgun(ts, tok, sig, get_settings())


def _email(samples, *keys, mid="<m1@test>"):
    return InboundEmail(message_id=mid, sender="Buyer <buyer@northbridge.example>", subject="POs",
                        attachments=[InboundAttachment(samples[k].name, samples[k].read_bytes()) for k in keys])


def test_dedup_on_message_id(samples):
    assert accept_email(_email(samples, "01_clean_northwind")).status == "accepted"
    again = accept_email(_email(samples, "01_clean_northwind"))
    assert again.status == "duplicate" and not again.jobs


def test_unsupported_attachment_rejected_others_processed(samples):
    res = accept_email(_email(samples, "edge_unsupported", "01_clean_northwind"))
    assert len(res.jobs) == 1 and res.rejected[0]["filename"] == "edge_unsupported.zip"
    with session_scope() as s:
        statuses = sorted(j.status for j in s.exec(select(Job)).all())
    assert statuses == ["queued", "rejected"]


def test_no_attachment_acknowledged():
    res = accept_email(InboundEmail(message_id="<empty@test>", sender="a@b.c", subject="hi"))
    assert res.status == "no_attachment"
    with session_scope() as s:
        assert s.exec(select(ProcessedEmail)).one().status == "no_attachment"


def test_sender_allow_list(samples, monkeypatch):
    from app.config import reset_settings_cache
    monkeypatch.setenv("ALLOWED_SENDER_DOMAINS", "trusted.example")
    reset_settings_cache()
    assert accept_email(_email(samples, "01_clean_northwind")).status == "rejected_sender"


def test_csv_batch_split_into_one_job_per_po(samples):
    parts = split_csv_batch(samples["08_csv_batch"].read_bytes())
    assert [p for p, _ in parts] == ["PO-2026-0450", "PO-2026-0451"]
    out = process_email(_email(samples, "08_csv_batch", mid="<batch@test>"))
    assert [r["decision"] for r in out["results"]] == ["auto_approve", "auto_approve"]


def test_mixed_email_multiple_attachments(samples):
    out = process_email(_email(samples, "01_clean_northwind", "03_unknown_vendor", mid="<mixed@test>"))
    assert sorted(r["decision"] for r in out["results"]) == ["auto_approve", "needs_review"]


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_webhook_mailgun_end_to_end(client, samples):
    ts, tok = str(int(time.time())), "tok-e2e"
    files = {"attachment-1": (samples["01_clean_northwind"].name, samples["01_clean_northwind"].read_bytes(),
                              "application/pdf")}
    data = {"timestamp": ts, "token": tok, "signature": sign(ts, tok), "Message-Id": "<e2e@test>",
            "sender": "buyer@northbridge.example", "subject": "PO-2026-0412"}
    r = client.post("/webhook/inbound-email?provider=mailgun", data=data, files=files)
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    job_id = r.json()["jobs"][0]["job_id"]
    job = client.get(f"/jobs/{job_id}").json()  # BackgroundTasks have run by now in TestClient
    assert job["status"] == "done" and job["decision"] == "auto_approve" and job["purchase_order_id"]
    assert client.get("/purchase-orders").json()[0]["po_number"] == "PO-2026-0412"


def test_webhook_rejects_bad_signature(client, samples):
    data = {"timestamp": str(int(time.time())), "token": "t", "signature": "bad", "Message-Id": "<x@test>"}
    r = client.post("/webhook/inbound-email?provider=mailgun", data=data)
    assert r.status_code == 401


def test_webhook_generic_secret(client, samples):
    files = {"file": (samples["05_blocked_vendor"].name, samples["05_blocked_vendor"].read_bytes(), "application/pdf")}
    bad = client.post("/webhook/inbound-email?provider=generic", data={"message_id": "<g@test>"}, files=files,
                      headers={"X-Webhook-Secret": "nope"})
    assert bad.status_code == 401
    ok = client.post("/webhook/inbound-email?provider=generic", data={"message_id": "<g@test>"}, files=files,
                     headers={"X-Webhook-Secret": "test-shared-secret"})
    assert ok.status_code == 200
    queue = client.get("/review-queue").json()
    assert queue[0]["flags"] == ["vendor_blocked"]
    r = client.post(f"/review-queue/{queue[0]['id']}/reject", json={"reviewer": "dana", "reason": "blocked"})
    assert r.json()["status"] == "rejected"


def test_upload_endpoint(client, samples):
    r = client.post("/process", files=[("files", (samples["06_math_error"].name,
                                                  samples["06_math_error"].read_bytes(), "application/pdf"))])
    assert r.json()["results"][0]["flags"] == ["math_mismatch"]
