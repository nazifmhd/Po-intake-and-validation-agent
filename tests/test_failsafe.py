"""Regression tests for the two promises: nothing questionable is auto-approved, nothing is lost.

Each test reproduces a way the system used to break one of them.
"""
import io
import json
import threading
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from test_scoring import C, make_po, run

from app.actions import review
from app.config import get_settings, reset_settings_cache
from app.db import session_scope
from app.extraction.extractor import extract, extract_heuristic, finalize
from app.extraction.schema import Confidenced, LineItem, PurchaseOrder, PurchaseOrderExtraction
from app.graph.build import run_pipeline
from app.ingestion.loaders import RawDocument, load_csv, load_document
from app.intake.common import (
    InboundAttachment, InboundEmail, accept_email, process_email, run_job, split_csv_batch, sweep_jobs,
)
from app.models import Job, ProcessedEmail, PurchaseOrderRecord, ReviewQueueItem, utcnow

KEY = {"X-API-Key": "test-api-key"}


def rows(model):
    with session_scope() as s:
        return s.exec(select(model)).all()


@pytest.fixture
def slack(monkeypatch):
    sent = []
    for module in ("app.actions.notify", "app.graph.nodes", "app.intake.common", "app.actions.review"):
        monkeypatch.setattr(f"{module}.send_slack", lambda text, settings=None: sent.append(text) or True)
    return sent


@pytest.fixture
def pipeline(samples, monkeypatch):
    def go(po: PurchaseOrder, **kw):
        monkeypatch.setattr("app.graph.nodes.extract", lambda doc: po)
        return run_pipeline(str(samples["01_clean_northwind"]), **kw)
    return go


# ================================================================== nothing questionable is auto-approved
def mutate(**fields):
    po = make_po()
    for path, value in fields.items():
        if path.startswith("line_"):
            setattr(po.line_items[1], path[5:], value)
        elif isinstance(getattr(po, path), Confidenced):
            setattr(po, path, Confidenced.missing() if value is None else C(value))
        else:
            setattr(po, path, value)
    return po


@pytest.mark.parametrize("po,flag", [
    (mutate(line_unit_price=None), "incomplete_lines"),
    (mutate(line_quantity=None), "incomplete_lines"),
    (mutate(line_sku=None, line_description=None), "incomplete_lines"),
    (mutate(line_quantity=0.0), "invalid_amount"),
    (mutate(line_unit="PALLET"), "unit_mismatch"),
    (mutate(line_description="Gold plated espresso machine"), "sku_mismatch"),
    (mutate(currency=None), "missing_fields"),
    (mutate(payment_terms=None), "missing_fields"),
    (mutate(additional_po_numbers=["PO-2"]), "multiple_pos_in_document"),
    (make_po(lines=[("IT-LAP-14P", "laptop", -2, 1850.0)], tax=0, shipping=0), "invalid_amount"),
    (make_po(lines=[("IT-LAP-14P", "laptop", 10, 1850.0), ("IT-MON-27Q", "monitor", -30, 310.0)], tax=0, shipping=0),
     "invalid_amount"),
    (make_po(lines=[("IT-LAP-14P", "laptop", 6, 1850.0)], tax=0, shipping=-2000.0), "invalid_amount"),
])
def test_questionable_content_is_flagged(po, flag):
    decision, flags, reasons = run(po)
    assert decision == "needs_review" and flag in flags and reasons[flag]


@pytest.mark.parametrize("field", ["po_number", "currency"])
def test_low_confidence_identity_fields_are_critical(field):
    po = make_po()
    getattr(po, field).confidence = 0.1
    assert "low_extraction_confidence" in run(po)[1]


def test_total_is_verified_even_without_printed_line_totals():
    po = make_po()
    po.subtotal = None
    for li in po.line_items:
        li.line_total = None
    assert run(po)[0] == "auto_approve"           # quantity x price adds up to the total
    po.total = 9.0
    assert "math_mismatch" in run(po)[1]           # it used to pass: "nothing to check" counted as "correct"


def test_unverifiable_total_is_a_flag():
    po = mutate(line_line_total=None, line_unit_price=None)
    assert "math_unverified" in run(po)[1]


def test_unit_synonyms_are_not_mismatches():
    po = make_po()
    po.line_items[0].unit, po.line_items[1].unit = "EACH", "PCS"
    assert run(po)[0] == "auto_approve"


def test_ambiguous_description_matches_nothing():
    from app.rag.catalog import validate_line_items
    po = make_po(lines=[(None, "Copy paper", 10, 38.5)])
    assert validate_line_items(po)[0]["status"] == "unknown_item"


def test_unverified_vendor_status_is_never_accepted():
    from app.scoring import arithmetic_check, build_signals, decide
    po = make_po()
    signals = build_signals(po, {"status": "error", "vendor_status": "unknown"}, [], [], arithmetic_check(po), True)
    assert "vendor_not_verified" in decide(po, signals)[1]


# ------------------------------------------------------------------ vendors: lookalikes
class FakeIndex:
    """Stands in for Chroma: similarity per vendor id."""

    def __init__(self, sims):
        self.sims = sims

    def query(self, name, text, k):
        return [{"id": f"{vid}-n0", "metadata": {"vendor_id": vid}, "similarity": sim} for vid, sim in self.sims.items()]


@pytest.mark.parametrize("name,semantic,status", [
    ("Apex Industrial Supplies Inc", 0.757, "uncertain"),   # 92% string-similar to Acme; the embedding vetoes it
    ("Acme Industrial Supply", 0.976, "matched"),           # a real misspelling: both signals agree
])
def test_lookalike_vendor_needs_both_signals(monkeypatch, name, semantic, status):
    from app.rag import vendors
    monkeypatch.setattr(vendors, "get_index", lambda s: FakeIndex({"V001": semantic}))
    monkeypatch.setattr(vendors, "sync_learned_aliases", lambda s: 0)
    m = vendors.match_vendor(name)
    assert m["status"] == status and m["vendor_id"] == "V001"


def test_semantic_suggestion_when_the_string_match_is_weak(monkeypatch):
    from app.rag import vendors
    monkeypatch.setattr(vendors, "get_index", lambda s: FakeIndex({"V015": 0.93}))
    monkeypatch.setattr(vendors, "sync_learned_aliases", lambda s: 0)
    m = vendors.match_vendor("SST Shadow Goods")            # would be "unknown" on spelling alone
    assert m["status"] == "uncertain" and m["vendor_status"] == "blocked"


def test_without_embeddings_only_exact_names_are_accepted():
    from app.rag.vendors import match_vendor
    assert match_vendor("Apex Industrial Supplies Inc")["status"] == "uncertain"
    assert match_vendor("ACME CORP.")["status"] == "matched"


def test_paraphrased_item_is_matched_semantically(monkeypatch):
    from app.rag import catalog
    sims = {"SAF-GOG-AF": 0.92, "SAF-HLM-HRD": 0.66}

    class Index:
        def query(self, name, text, k):
            return [{"id": i, "similarity": s} for i, s in sims.items()]
    monkeypatch.setattr(catalog, "get_index", lambda s: Index())
    po = make_po(lines=[(None, "Fog-resistant protective eyewear", 10, 9.75)])
    chk = catalog.validate_line_items(po)[0]
    assert chk["matched_sku"] == "SAF-GOG-AF" and chk["match_method"] == "description_semantic"
    sims["SAF-HLM-HRD"] = 0.89                              # a close runner-up makes it ambiguous
    assert catalog.validate_line_items(po)[0]["status"] == "unknown_item"


# ------------------------------------------------------------------ policy
@pytest.fixture
def seed(tmp_path, monkeypatch):
    import shutil
    target = tmp_path / "seed"
    shutil.copytree(get_settings().seed_path, target)
    monkeypatch.setenv("SEED_DIR", str(target))
    reset_settings_cache()

    def edit(old, new):
        p = target / "policy.md"
        text = p.read_text(encoding="utf-8")
        assert old in text
        p.write_text(text.replace(old, new), encoding="utf-8")
    return edit


def test_editing_the_policy_text_changes_the_decision(seed):
    assert run(make_po())[0] == "auto_approve"
    seed("10,000.00", "2,000.00")
    assert "over_threshold" in run(make_po())[1]


def test_policy_can_tighten_terms_and_tolerance(seed):
    seed("Net 30, Net 45 and Net 60", "Net 45 and Net 60")
    seed("more than 10% above", "more than 1% above")
    po = make_po(lines=[("IT-LAP-14P", "14-inch business laptop", 2, 1890.0)])
    assert {"noncompliant_terms", "price_anomaly"} <= set(run(po)[1])


def test_policy_cannot_silently_loosen_the_configured_cap(seed):
    seed("10,000.00", "90,000.00")
    assert "over_threshold" in run(make_po(lines=[("IT-LAP-14P", "laptop", 6, 1850.0)]))[1]


def test_deleted_policy_rule_sends_everything_to_review(seed):
    seed("No single PO may exceed USD 50,000.00", "Very large POs need care")
    decision, flags, reasons = run(make_po())
    assert "policy_rule_missing" in flags and "single_po_hard_limit" in reasons["policy_rule_missing"]


def test_policy_llm_outage_is_a_flag_and_bad_citations_are_dropped(monkeypatch):
    import app.rag.policy as pol
    s = get_settings().model_copy(update={"llm_provider": "anthropic", "policy_llm_review": True})
    monkeypatch.setattr("app.llm.LLMClient.__init__", lambda self, settings=None: None)

    def boom(self, **kw):
        raise RuntimeError("overloaded")
    monkeypatch.setattr("app.llm.LLMClient.structured", boom)
    finding = pol.check_policy(make_po(), None, s)["findings"][-1]
    assert finding["code"] == "policy_review_unavailable" and finding["severity"] == "violation"

    monkeypatch.setattr("app.llm.LLMClient.structured", lambda self, **kw: pol.PolicyReviewLLM(findings=[
        {"clause": c, "finding": f"finding {c}", "severity": "violation"}
        for c in ("", "a", "Clause that does not exist", "## Currency")]))
    llm = [f for f in pol.check_policy(make_po(), None, s)["findings"] if f["source"] == "llm"]
    assert [f["clause"] for f in llm] == ["## Currency"]


def test_spend_splitting_is_detected(pipeline):
    def order(n, date="2026-10-20", sku="IT-LAP-14P"):
        po = make_po(lines=[(sku, "", 5, 1850.0 if sku == "IT-LAP-14P" else 310.0)], vendor="Globex Technology",
                     tax=0, shipping=0)
        for li in po.line_items:
            li.description = None
        po.po_number, po.po_date = C(f"PO-S{n}"), C(date)
        return pipeline(po)
    assert order(1)["decision"] == "auto_approve"                              # 9,250
    second = order(2)
    assert second["decision"] == "needs_review" and second["flags"] == ["possible_po_split"]
    assert "PO-S1" in second["reasons"]["possible_po_split"]
    assert order(3, sku="IT-MON-27Q")["decision"] == "auto_approve"            # different item
    assert order(4, date="2026-12-20")["decision"] == "auto_approve"           # outside the window


# ------------------------------------------------------------------ hallucination checks
TEXT = "PURCHASE ORDER\nPO Number: PO-1\nVendor: Shady Vendor Inc\nBuyer: Northwind\nTotal: $100.00\n" + "x" * 60


def extraction(vendor, evidence, total=100.0, totals_evidence="Total: $100.00", lines=()):
    miss = {k: Confidenced.missing() for k in ("po_date", "vendor_tax_id", "buyer_department", "currency",
                                               "payment_terms", "delivery_date")}
    return PurchaseOrderExtraction(
        po_number=Confidenced(value="PO-1", confidence=0.95, evidence="PO Number: PO-1"),
        vendor_name=Confidenced(value=vendor, confidence=0.95, evidence=evidence), line_items=list(lines),
        total=total, totals_confidence=0.95, totals_evidence=totals_evidence, **miss)


def ground(*a, **k):
    return finalize(extraction(*a, **k), RawDocument("pdf_text", TEXT, "text_layer"), "llm:test")


def test_true_value_keeps_its_confidence():
    po = ground("Shady Vendor Inc", "Vendor: Shady Vendor Inc")
    assert po.vendor_name.confidence == 0.95 and po.totals_confidence == 0.95


@pytest.mark.parametrize("vendor,evidence", [
    ("Northwind Office Products LLC", "Vendor: Shady Vendor Inc"),  # real snippet, different value
    ("Northwind Office Products LLC", None),                        # no evidence, not in the document
    ("Shady Vendor Inc", None),                                     # no evidence at all
])
def test_untraceable_vendor_loses_confidence(vendor, evidence):
    assert ground(vendor, evidence).vendor_name.confidence < 0.8


def test_untraceable_amounts_lose_confidence():
    assert ground("Shady Vendor Inc", "Vendor: Shady Vendor Inc", total=90.0).totals_confidence < 0.8
    po = ground("Shady Vendor Inc", "Vendor: Shady Vendor Inc",
                lines=[LineItem(line_no=1, sku="ABC", quantity=3, unit_price=77.0, confidence=0.95)])
    assert po.line_items[0].confidence < 0.8


@pytest.mark.parametrize("value,evidence,ok", [
    ("2026-10-20", "Date: Oct 20, 2026", True), ("2026-10-20", "20/10/26", True), ("2026-10-21", "Oct 20, 2026", False)])
def test_dates_are_compared_by_meaning(value, evidence, ok):
    from app.extraction.verify import value_in
    assert value_in("po_date", value, evidence) is ok


def test_scan_contradicted_by_ocr_loses_confidence():
    doc = RawDocument("image", TEXT, "ocr", vision_images=["p.jpg"])
    po = finalize(extraction("Shady Vendor Inc", "Vendor: Shady Vendor Inc", 900.0, "Total: $900.00"), doc, "llm:test")
    assert po.vendor_name.confidence == 0.95 and po.totals_confidence < 0.8


# ------------------------------------------------------------------ LLM contract
def test_schema_stays_within_claudes_union_limit():
    from app.llm import strict_json_schema
    schema = strict_json_schema(PurchaseOrderExtraction)

    def unions(node):  # counted the strict way: every $ref expanded at every use
        if isinstance(node, list):
            return sum(map(unions, node))
        if not isinstance(node, dict):
            return 0
        if "$ref" in node:
            return unions(schema["$defs"][node["$ref"].split("/")[-1]])
        return ("anyOf" in node) + sum(unions(v) for k, v in node.items() if k != "$defs")
    assert unions(schema) <= 16


def test_empty_string_means_not_present():
    c = Confidenced.model_validate({"value": " ", "confidence": 0.1, "evidence": ""})
    assert c.value is None and c.evidence is None and not c.present


def test_missing_api_key_is_a_clear_reason(samples, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    s = get_settings().model_copy(update={"llm_provider": "openai", "openai_api_key": ""})
    monkeypatch.setattr("app.extraction.extractor.get_settings", lambda: s)
    out = run_pipeline(str(samples["01_clean_northwind"]))
    assert out["decision"] == "needs_review" and "llm_not_configured" in out["flags"]


# ------------------------------------------------------------------ documents
def test_scanned_page_inside_a_text_pdf_is_not_ignored(samples, tmp_path, monkeypatch):
    import pymupdf
    d = pymupdf.open(str(samples["01_clean_northwind"]))
    page = d.new_page()
    page.insert_image(page.rect, filename=str(samples["02_scanned_globex"]))
    d.new_page()                                            # a blank page is not a scan
    path = tmp_path / "mixed.pdf"
    d.save(str(path))
    monkeypatch.setattr("app.ingestion.loaders.ocr_available", lambda: False)
    monkeypatch.setattr("app.extraction.extractor.ocr_available", lambda: False)
    doc = load_document(path, tmp_path / "pages")
    assert doc.doc_type == "pdf_mixed" and len(doc.vision_images) == 1 and "PO-2026-0412" in doc.raw_text
    assert "unread_pages" in run_pipeline(str(path))["flags"]


def test_long_document_is_flagged(samples, tmp_path):
    import pymupdf
    d = pymupdf.open(str(samples["01_clean_northwind"]))
    for _ in range(22):
        d.new_page().insert_text((50, 50), "Standard terms and conditions apply to this order. " * 2)
    path = tmp_path / "long.pdf"
    d.save(str(path))
    assert "truncated_document" in run_pipeline(str(path))["flags"]


def test_two_pos_in_one_document_are_noticed():
    text = "PO Number: PO-11\nVendor: A\n\nPO Number: PO-12\nVendor: B\n"
    assert extract_heuristic(RawDocument("pdf_text", text, "text_layer")).additional_po_numbers == ["PO-12"]


CSV = ("po_number;po_date;vendor_name;currency;payment_terms;delivery_date;sku;description;quantity;unit;unit_price;total\n"
       "PO-A;2026-10-20;Litware;USD;Net 30;2026-11-01;IT-KBM-WLS;Wireless keyboard and mouse combo;2;EA;59,00;247,00\n"
       ";;;;;;IT-HDS-ANC;Noise-cancelling USB headset;1;EA;129,00;\n"
       "PO-B;2026-10-20;Litware;USD;Net 30;2026-11-01;IT-KBM-WLS;Wireless keyboard and mouse combo;1;EA;59,00;59,00\n")


def test_semicolon_csv_batch_with_continuation_rows_is_split():
    parts = split_csv_batch(CSV.encode("cp1252"))
    assert [p for p, _ in parts] == ["PO-A", "PO-B"]
    assert parts[0][1].decode().count("\n") == 3            # header + 2 lines


def test_shifted_csv_columns_are_flagged(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("po_number,vendor,quantity,unit_price,total\nPO-1,Litware,1,1,540.00,1540.00\n")
    assert "malformed_csv" in load_csv(path).flags


def test_conflicting_csv_totals_lower_confidence(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("po_number,vendor,sku,description,quantity,unit_price,total\n"
                    "PO-1,Litware,A,a,1,10,10\nPO-1,Litware,B,b,1,10,20\n")
    assert extract(load_csv(path)).totals_confidence < 0.8


# ================================================================== nothing is lost
def email(samples, key="01_clean_northwind", mid="<m@test>"):
    p = samples[key]
    return InboundEmail(mid, "buyer@northbridge.example", "PO", [InboundAttachment(p.name, p.read_bytes())])


def test_job_survives_a_crash_after_the_email_was_accepted(samples):
    accept_email(email(samples))                            # ...and the process dies before running the job
    assert process_email(email(samples))["status"] == "duplicate"
    assert [r["decision"] for r in sweep_jobs("api", grace_seconds=0)] == ["auto_approve"]
    assert rows(Job)[0].status == "done" and len(rows(PurchaseOrderRecord)) == 1


def test_job_survives_a_worker_that_died_mid_run(samples):
    accept_email(email(samples))
    with session_scope() as s:
        job = s.exec(select(Job)).one()
        job.status, job.worker, job.attempts = "processing", "imap", 1
        job.started_at = utcnow() - timedelta(hours=1)
        s.add(job)
    assert [r["decision"] for r in sweep_jobs("api", grace_seconds=0)] == ["auto_approve"]


def test_job_that_keeps_killing_its_worker_goes_to_a_human(samples, slack):
    accept_email(email(samples))
    with session_scope() as s:
        job = s.exec(select(Job)).one()
        job.attempts = 3
        s.add(job)
    sweep_jobs("api", grace_seconds=0)
    assert json.loads(rows(ReviewQueueItem)[0].flags_json) == ["processing_error"] and slack


def test_a_job_is_never_run_twice(samples):
    res = accept_email(email(samples))
    assert run_job(res.jobs[0])["decision"] == "auto_approve"
    assert run_job(res.jobs[0])["decision"] == "already_claimed"


def test_failed_intake_is_taken_over_by_the_redelivery(samples, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("app.intake.common.split_csv_batch", lambda c: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            accept_email(email(samples, "03_unknown_vendor"))
    assert rows(ProcessedEmail)[0].status == "intake_failed"
    assert accept_email(email(samples, "03_unknown_vendor")).status == "accepted"
    assert len(rows(Job)) == 1


def test_half_accepted_email_is_in_progress_then_taken_over_without_duplicating_jobs(samples):
    accept_email(email(samples))
    with session_scope() as s:
        row = s.exec(select(ProcessedEmail)).one()
        row.status = "received"                             # died before the claim was completed
        s.add(row)
    assert accept_email(email(samples)).status == "in_progress"
    with session_scope() as s:
        row = s.exec(select(ProcessedEmail)).one()
        row.received_at = utcnow() - timedelta(hours=1)
        s.add(row)
    res = accept_email(email(samples))
    assert res.status == "accepted" and len(res.jobs) == 1 and len(rows(Job)) == 1


def test_unwritable_review_queue_leaves_a_dead_letter_and_retries(samples, monkeypatch, slack):
    def boom(*a, **k):
        raise RuntimeError("db locked")
    with monkeypatch.context() as m:
        m.setattr("app.graph.nodes.enqueue_review", boom)
        res = accept_email(email(samples, "05_blocked_vendor"))
        assert run_job(res.jobs[0])["decision"] == "failed"
    assert any("Could not queue" in t for t in slack)
    assert list((get_settings().storage_path / "dead_letter").glob("*.json"))
    assert rows(Job)[0].status == "queued"
    assert [r["decision"] for r in sweep_jobs("api", grace_seconds=0)] == ["needs_review"]  # recovered
    assert len(rows(ReviewQueueItem)) == 1


def test_filing_failure_falls_back_to_the_queue(samples, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr("app.graph.nodes.store_approved", boom)
    out = run_pipeline(str(samples["01_clean_northwind"]))
    assert out["decision"] == "needs_review" and out["flags"] == ["storage_error"] and out["record_id"]


def test_identical_resend_is_skipped_and_announced(samples, slack):
    run_pipeline(str(samples["01_clean_northwind"]))
    assert run_pipeline(str(samples["01_clean_northwind"]))["decision"] == "duplicate"
    assert any("received again" in t for t in slack)


def test_same_po_number_with_different_content_goes_to_review(pipeline):
    assert pipeline(make_po())["decision"] == "auto_approve"
    out = pipeline(make_po(vendor="Umbra Safety", shipping=304.0))
    assert out["decision"] == "needs_review" and "po_number_conflict" in out["flags"]
    assert "total" in out["reasons"]["po_number_conflict"]


def test_simultaneous_pos_with_the_same_number_are_filed_once(pipeline, monkeypatch):
    """Both were scored before either was filed."""
    monkeypatch.setattr("app.graph.nodes.check_duplicate", lambda *a, **k: None)
    assert pipeline(make_po())["decision"] == "auto_approve"
    out = pipeline(make_po())
    assert out["decision"] == "needs_review" and len(rows(PurchaseOrderRecord)) == 1


def test_signature_logo_is_not_treated_as_a_po():
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (120, 40), "navy").save(b, "PNG")
    res = accept_email(InboundEmail("<logo>", "a@b.c", "s", [InboundAttachment("image001.png", b.getvalue())]))
    assert res.status == "all_rejected" and "logo" in res.rejected[0]["reason"]


def test_attachment_without_extension_is_recognised(samples):
    p = samples["01_clean_northwind"]
    res = accept_email(InboundEmail("<x>", "a@b.c", "s", [InboundAttachment("purchase order", p.read_bytes())]))
    assert res.jobs[0]["filename"] == "purchase_order.pdf"


def test_n8n_workflow_sends_every_attachment():
    from app.config import PROJECT_ROOT
    flow = json.loads((PROJECT_ROOT / "workflows" / "n8n_intake.json").read_text(encoding="utf-8"))
    names = {n["name"] for n in flow["nodes"]}
    targets = {c["node"] for outs in flow["connections"].values() for branch in outs["main"] for c in branch}
    assert targets <= names and set(flow["connections"]) <= names and "Split attachments" in targets


# ================================================================== approvals can't be abused
@pytest.fixture
def blocked(samples):
    return run_pipeline(str(samples["05_blocked_vendor"]))["record_id"]


@pytest.fixture
def unknown(samples):
    return run_pipeline(str(samples["03_unknown_vendor"]))["record_id"]


def test_blocked_vendor_can_never_be_approved(blocked, unknown):
    with pytest.raises(review.ReviewError, match="BLOCKED"):
        review.approve(blocked, "dana", "please")
    with pytest.raises(review.ReviewError, match="BLOCKED"):
        review.approve(blocked, "dana", "please", vendor_override_id="V002")  # relabelling it doesn't help
    with pytest.raises(review.ReviewError, match="BLOCKED"):
        review.approve(unknown, "dana", "please", vendor_override_id="V015")
    assert not rows(PurchaseOrderRecord)


def test_unregistered_vendor_needs_a_registry_record(unknown):
    with pytest.raises(review.ReviewError, match="not in the approved registry"):
        review.approve(unknown, "dana", "fine by me")


def test_overriding_flags_needs_a_note(pipeline):
    out = pipeline(make_po(lines=[("IT-LAP-14P", "14-inch business laptop", 30, 1850.0)]))
    with pytest.raises(review.ReviewError, match="note is required"):
        review.approve(out["record_id"], "dana")
    assert review.approve(out["record_id"], "dana", "CFO approval #88")


def test_simultaneous_approvals_file_once(pipeline, monkeypatch):
    out = pipeline(make_po(terms="Net 90"))
    original, barrier, errors = review._load_pending, threading.Barrier(2), []

    def slow(item_id, settings):
        item = original(item_id, settings)
        barrier.wait(5)
        return item
    monkeypatch.setattr(review, "_load_pending", slow)

    def click():
        try:
            review.approve(out["record_id"], "dana", "ok")
        except review.ReviewError as exc:
            errors.append(str(exc))
    threads = [threading.Thread(target=click) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(rows(PurchaseOrderRecord)) == 1 and len(errors) == 1


def test_alias_of_another_vendor_cannot_be_learned(pipeline):
    out = pipeline(make_po(vendor="Globex Technology", terms="Net 90"))
    with pytest.raises(review.ReviewError, match="already a name"):
        review.approve(out["record_id"], "dana", "ok", "V002", learn_alias=True)


def test_amendment_replaces_the_filed_po_only_when_asked(pipeline):
    pipeline(make_po())
    out = pipeline(make_po(shipping=304.0))
    with pytest.raises(review.ReviewError, match="already filed"):
        review.approve(out["record_id"], "dana", "amended")
    new = review.approve(out["record_id"], "dana", "amended", supersede=True)
    assert {r.id: r.status for r in rows(PurchaseOrderRecord)} == {1: "superseded", new: "approved_by_review"}


# ------------------------------------------------------------------ corrections
def test_correction_reruns_every_check(samples):
    out = run_pipeline(str(samples["06_math_error"]))
    item = out["record_id"]
    res = review.correct(item, "dana", amounts={"total": 2021.40})
    assert res["flags"] == [] and res["changes"] == {"total": [2121.4, 2021.4]}
    assert review.approve(item, "dana")                     # no flags left, so no note needed
    res = [r for r in rows(ReviewQueueItem) if r.id == item][0]
    assert json.loads(res.original_json)["total"] == 2121.4  # what was extracted is kept
    assert run_pipeline(str(samples["06_math_error"]))["decision"] == "duplicate"


def test_correction_can_add_flags_too(samples):
    out = run_pipeline(str(samples["06_math_error"]))
    res = review.correct(out["record_id"], "dana", header={"vendor_name": "Shadow Supply Traders"},
                         line_items=[{"sku": "IT-LAP-14P", "description": "laptop", "quantity": "10",
                                      "unit": "EA", "unit_price": "1,850.00", "line_total": "18500"}])
    assert {"vendor_blocked", "math_mismatch"} <= set(res["flags"])


@pytest.mark.parametrize("kwargs,msg", [
    ({"header": {"po_date": "next week"}}, "not a date"), ({"amounts": {"total": "lots"}}, "not a number"),
    ({"header": {"approved_by": "me"}}, "Unknown field")])
def test_bad_corrections_are_refused(samples, kwargs, msg):
    out = run_pipeline(str(samples["06_math_error"]))
    with pytest.raises(review.ReviewError, match=msg):
        review.correct(out["record_id"], "dana", **kwargs)


def test_unreadable_document_can_be_keyed_in_by_hand(samples):
    out = run_pipeline(str(samples["edge_locked"]))
    with pytest.raises(review.ReviewError, match="Nothing was extracted"):
        review.approve(out["record_id"], "dana", "x")
    res = review.correct(out["record_id"], "dana", header={
        "po_number": "PO-2026-0412", "po_date": "2026-10-20", "vendor_name": "Northwind Office", "currency": "USD",
        "payment_terms": "net 30 days", "delivery_date": "2026-11-05"}, amounts={"total": 570},
        line_items=[{"sku": "OFF-CHR-ERG", "quantity": 2, "unit": "EA", "unit_price": 285, "line_total": 570}])
    assert res["flags"] == []


def test_reprocess_picks_up_a_fixed_outage(samples, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr("app.graph.nodes.extract", lambda doc: (_ for _ in ()).throw(RuntimeError("LLM down")))
        out = run_pipeline(str(samples["01_clean_northwind"]))
    assert "processing_error" in out["flags"]
    assert review.reprocess(out["record_id"], "dana")["flags"] == []


# ------------------------------------------------------------------ API + webhook security
@pytest.fixture
def anonymous():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("method,path", [
    ("get", "/audit"), ("get", "/review-queue"), ("get", "/purchase-orders"), ("get", "/jobs"),
    ("post", "/kb/rebuild"), ("post", "/process"), ("post", "/review-queue/1/approve"),
    ("post", "/review-queue/1/reject"), ("post", "/review-queue/1/correct"), ("post", "/review-queue/1/reprocess")])
def test_everything_but_webhook_and_health_needs_the_key(anonymous, method, path):
    assert getattr(anonymous, method)(path).status_code == 401
    assert getattr(anonymous, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert anonymous.get("/health").status_code == 200


def test_without_a_configured_key_the_api_is_closed(monkeypatch, anonymous):
    for value in ("", "change-me"):
        monkeypatch.setenv("API_KEY", value)
        reset_settings_cache()
        assert anonymous.get("/audit", headers={"X-API-Key": value}).status_code == 503


def test_placeholder_secrets_are_not_secrets(monkeypatch, anonymous):
    monkeypatch.setenv("WEBHOOK_SHARED_SECRET", "change-me-shared-secret")
    reset_settings_cache()
    r = anonymous.post("/webhook/inbound-email?provider=generic", data={"message_id": "<z>"},
                       headers={"X-Webhook-Secret": "change-me-shared-secret"})
    assert r.status_code == 401


def test_caller_cannot_pick_a_provider_that_is_not_enabled(monkeypatch, anonymous):
    monkeypatch.setenv("WEBHOOK_EXTRA_PROVIDERS", "")
    reset_settings_cache()
    r = anonymous.post("/webhook/inbound-email?provider=generic", data={"message_id": "<z>"},
                       headers={"X-Webhook-Secret": "test-shared-secret"})
    assert r.status_code == 401 and "not enabled" in r.json()["detail"]


def test_failed_intake_asks_the_provider_to_retry_with_the_same_token(anonymous, samples, monkeypatch):
    import time

    from test_intake import sign
    ts, tok = str(int(time.time())), "tok-retry"
    data = {"timestamp": ts, "token": tok, "signature": sign(ts, tok), "Message-Id": "<retry@test>", "sender": "a@b.c"}
    p = samples["01_clean_northwind"]
    files = {"attachment-1": (p.name, p.read_bytes(), "application/pdf")}
    url = "/webhook/inbound-email?provider=mailgun"
    with monkeypatch.context() as m:
        m.setattr("app.intake.common._effective_name", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
        assert anonymous.post(url, data=data, files=files).status_code == 503
    assert anonymous.post(url, data=data, files=files).json()["status"] == "accepted"


def test_review_api_end_to_end(anonymous, samples):
    run_pipeline(str(samples["06_math_error"]))
    item = anonymous.get("/review-queue", headers=KEY).json()[0]
    r = anonymous.post(f"/review-queue/{item['id']}/correct", headers=KEY,
                       json={"reviewer": "dana", "amounts": {"total": 2021.40}})
    assert r.json()["flags"] == []
    r = anonymous.post(f"/review-queue/{item['id']}/approve", headers=KEY, json={"reviewer": "dana"})
    assert r.json()["status"] == "approved"
    assert anonymous.post(f"/review-queue/{item['id']}/approve", headers=KEY,
                          json={"reviewer": "dana"}).status_code == 409


# ------------------------------------------------------------------ storage
def test_old_database_is_upgraded_in_place(tmp_path):
    import sqlite3

    from app.db import get_engine, reset_engines
    path = tmp_path / "old.db"
    get_engine(get_settings())
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, job_id VARCHAR NOT NULL, message_id VARCHAR, filename VARCHAR "
                "NOT NULL, stored_path VARCHAR NOT NULL, sha256 VARCHAR NOT NULL, status VARCHAR NOT NULL, decision "
                "VARCHAR, note VARCHAR, created_at DATETIME NOT NULL, finished_at DATETIME)")
    con.execute("INSERT INTO jobs VALUES (1,'j1',NULL,'f','p','s','done',NULL,NULL,'2026-10-28 00:00:00',NULL)")
    con.execute("CREATE TABLE purchase_orders (id INTEGER PRIMARY KEY, job_id VARCHAR NOT NULL, po_number VARCHAR)")
    con.commit()
    con.close()
    settings = get_settings().model_copy(update={"database_url": f"sqlite:///{path.as_posix()}"})
    with session_scope(settings) as s:
        assert s.exec(select(Job)).one().attempts == 0
        assert s.exec(select(PurchaseOrderRecord)).all() == []
    reset_engines()


# ------------------------------------------------------------------ first lines of defence, on their own
def test_split_is_found_by_the_policy_check_not_only_at_filing(pipeline):
    from app.rag.catalog import validate_line_items
    from app.rag.policy import check_policy
    from app.rag.vendors import match_vendor
    first = make_po(lines=[("IT-LAP-14P", "14-inch business laptop", 5, 1850.0)], vendor="Globex", tax=0, shipping=0)
    assert pipeline(first)["decision"] == "auto_approve"
    second = first.model_copy(deep=True)
    second.po_number = C("PO-2")
    findings = check_policy(second, match_vendor("Globex"), price_checks=validate_line_items(second))["findings"]
    assert "possible_po_split" in {f["code"] for f in findings}


def test_item_rejected_in_the_meantime_cannot_be_approved(pipeline, monkeypatch):
    out = pipeline(make_po(terms="Net 90"))
    original = review._load_pending

    def rejected_right_after_loading(item_id, settings):
        item = original(item_id, settings)
        with session_scope() as s:
            s.get(ReviewQueueItem, item_id).status = "rejected"
        return item
    monkeypatch.setattr(review, "_load_pending", rejected_right_after_loading)
    with pytest.raises(review.ReviewError, match="already been handled"):
        review.approve(out["record_id"], "dana", "ok")
    assert not rows(PurchaseOrderRecord)


def test_policy_llm_is_told_about_every_check_not_only_the_policy_ones(monkeypatch):
    """Otherwise it reports the blocked vendor again as its own violation (seen in a live run)."""
    import app.rag.policy as pol
    from app.rag.catalog import validate_line_items
    from app.rag.vendors import match_vendor
    s = get_settings().model_copy(update={"llm_provider": "anthropic", "policy_llm_review": True})
    prompts = []
    monkeypatch.setattr("app.llm.LLMClient.__init__", lambda self, settings=None: None)
    monkeypatch.setattr("app.llm.LLMClient.structured",
                        lambda self, **kw: prompts.append(kw["text"]) or pol.PolicyReviewLLM(findings=[]))
    po = make_po(vendor="Shadow Supply Traders", lines=[("IT-LAP-14P", "laptop", 2, 2368.0)], total=1.0)
    pol.check_policy(po, match_vendor("Shadow Supply Traders"), s, price_checks=validate_line_items(po))
    raised = prompts[0].split("already raised:")[1].split("Retrieved policy")[0]
    assert all(code in raised for code in ("vendor_blocked", "price_anomaly", "math_mismatch"))


def test_lookalike_vendor_is_never_filed_under_the_real_one_by_default(samples):
    """An uncertain match is a suggestion; approving must not silently adopt it."""
    out = run_pipeline(str(samples["10_lookalike_vendor"]))
    assert out["flags"] == ["vendor_low_match"] and out["vendor_match"]["vendor_id"] == "V001"
    with pytest.raises(review.ReviewError, match="only a possible match"):
        review.approve(out["record_id"], "dana", "looks fine")
    assert not rows(PurchaseOrderRecord)
    assert review.approve(out["record_id"], "dana", "called them, same company", vendor_override_id="V001")
