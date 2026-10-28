"""End-to-end through the LangGraph pipeline (offline)."""
import json

import pytest
from sqlalchemy import text
from sqlmodel import select

from app.db import get_engine, session_scope
from app.extraction.schema import Confidenced, LineItem, PurchaseOrder
from app.graph.build import run_pipeline
from app.models import AuditLog, LineItemRecord, PurchaseOrderRecord, ReviewQueueItem


def test_clean_pdf_is_auto_approved_and_stored(samples):
    out = run_pipeline(str(samples["01_clean_northwind"]))
    assert out["decision"] == "auto_approve" and out["flags"] == [] and not out["errors"]
    assert out["record_table"] == "purchase_orders"
    with session_scope() as s:
        rec = s.get(PurchaseOrderRecord, out["record_id"])
        assert rec.status == "approved" and rec.vendor_id == "V002" and rec.total == 4200.0
        assert len(s.exec(select(LineItemRecord).where(LineItemRecord.po_id == rec.id)).all()) == 2
        events = [a.event for a in s.exec(select(AuditLog)).all()]
    assert "pipeline_completed" in events
    assert [t["node"] for t in out["trace"]] == ["ingest", "extract", "enrich", "score", "route", "store"]


def test_alias_vendor_two_column_multipage_auto_approves(samples):
    out = run_pipeline(str(samples["04_acme_alias_twocol"]))
    assert out["decision"] == "auto_approve", out["reasons"]
    assert out["vendor_match"]["vendor_id"] == "V001"


@pytest.mark.parametrize("key,expected", [
    ("03_unknown_vendor", {"vendor_unknown", "missing_fields"}),
    ("05_blocked_vendor", {"vendor_blocked"}),
    ("06_math_error", {"math_mismatch"}),
    ("07_bad_terms", {"noncompliant_terms"}),
])
def test_flagged_documents_go_to_review(samples, key, expected):
    out = run_pipeline(str(samples[key]))
    assert out["decision"] == "needs_review"
    assert set(out["flags"]) == expected, out["reasons"]
    with session_scope() as s:
        item = s.get(ReviewQueueItem, out["record_id"])
        assert item.status == "pending_review" and set(json.loads(item.flags_json)) == expected


def _true_scanned_po() -> PurchaseOrder:
    """What a vision model reads off 02_scanned_globex.jpg."""
    c = lambda v, k=0.9: Confidenced(value=v, confidence=k, evidence=v)  # noqa: E731
    return PurchaseOrder(
        po_number=c("PO-2026-0419"), po_date=c("2026-10-22"), vendor_name=c("GLOBEX TECH SOLUTIONS INC"),
        vendor_tax_id=c("US-34-5678901"), buyer_department=c("IT"), currency=c("USD"), payment_terms=c("Net 30"),
        delivery_date=c("2026-11-12"),
        line_items=[LineItem(line_no=1, sku="IT-LAP-14P", description="14-inch business laptop", quantity=5, unit="EA",
                             unit_price=2368.0, line_total=11840.0, confidence=0.88),
                    LineItem(line_no=2, sku="IT-DCK-USC", description="USB-C docking station", quantity=5, unit="EA",
                             unit_price=245.0, line_total=1225.0, confidence=0.88)],
        subtotal=13065.0, tax=0.0, shipping=435.0, total=13500.0, totals_confidence=0.9,
        extraction_method="llm:mock",
    )


def test_scanned_po_star_demo_flags(samples, monkeypatch):
    monkeypatch.setattr("app.graph.nodes.extract", lambda doc: _true_scanned_po())
    out = run_pipeline(str(samples["02_scanned_globex"]))
    assert out["doc_type"] == "image"
    assert set(out["flags"]) == {"price_anomaly", "over_threshold"}, out["reasons"]
    assert out["vendor_match"]["vendor_id"] == "V003"


def test_ablation_rag_off_lets_bad_po_through(samples, monkeypatch):
    monkeypatch.setattr("app.graph.nodes.extract", lambda doc: _true_scanned_po())
    out = run_pipeline(str(samples["02_scanned_globex"]), options={"rag_enabled": False})
    assert out["decision"] == "auto_approve"  # exactly why the KB is not decoration


@pytest.mark.parametrize("key,flag", [("edge_corrupt", "corrupt_file"), ("edge_locked", "locked_file")])
def test_bad_files_fail_toward_review_without_crashing(samples, key, flag):
    out = run_pipeline(str(samples[key]))
    assert out["decision"] == "needs_review" and flag in out["flags"]
    assert out["record_table"] == "review_queue"


def test_scan_without_vision_or_ocr_goes_to_review(samples, monkeypatch):
    monkeypatch.setattr("app.ingestion.loaders.ocr_available", lambda: False)
    out = run_pipeline(str(samples["02_scanned_globex"]))
    assert out["decision"] == "needs_review"
    assert "no_text_for_offline_extraction" in out["flags"]


def test_node_exception_routes_to_review_not_crash(samples, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("vector store down")
    monkeypatch.setattr("app.graph.nodes.match_vendor", boom)
    out = run_pipeline(str(samples["01_clean_northwind"]))
    assert out["decision"] == "needs_review" and "vendor_check_failed" in out["flags"]


def test_scorer_exception_never_auto_approves(samples, monkeypatch):
    def boom(*a, **k):
        raise ValueError("scorer bug")
    monkeypatch.setattr("app.graph.nodes.decide", boom)
    out = run_pipeline(str(samples["01_clean_northwind"]))
    assert out["decision"] == "needs_review" and "processing_error" in out["flags"]


def test_duplicate_po_is_skipped(samples):
    first = run_pipeline(str(samples["01_clean_northwind"]))
    second = run_pipeline(str(samples["01_clean_northwind"]))
    assert first["decision"] == "auto_approve" and second["decision"] == "duplicate"
    with session_scope() as s:
        assert len(s.exec(select(PurchaseOrderRecord)).all()) == 1


def test_review_approve_promotes_and_learns_alias(samples):
    from app.actions import review
    from app.rag.vendors import match_vendor

    out = run_pipeline(str(samples["03_unknown_vendor"]))
    assert match_vendor("Pinnacle Promo Goods Ltd")["status"] == "unknown"
    with pytest.raises(review.ReviewError):
        review.approve(out["record_id"], reviewer="")
    po_id = review.approve(out["record_id"], "dana", "onboarded as Northwind subsidiary", "V002", learn_alias=True)
    with session_scope() as s:
        assert s.get(PurchaseOrderRecord, po_id).status == "approved_by_review"
        assert s.get(ReviewQueueItem, out["record_id"]).status == "approved"
    m = match_vendor("Pinnacle Promo Goods Ltd")  # next PO matches automatically
    assert m["status"] == "matched" and m["vendor_id"] == "V002"
    with pytest.raises(review.ReviewError):
        review.approve(out["record_id"], "dana")  # can't act twice


def test_review_reject(samples):
    from app.actions import review

    out = run_pipeline(str(samples["05_blocked_vendor"]))
    review.reject(out["record_id"], "dana", "blocked vendor - reported to compliance")
    with session_scope() as s:
        assert s.get(ReviewQueueItem, out["record_id"]).status == "rejected"


def test_audit_log_is_append_only(samples):
    run_pipeline(str(samples["01_clean_northwind"]))
    with get_engine().connect() as conn:
        with pytest.raises(Exception, match="append-only"):
            conn.execute(text("UPDATE audit_log SET event='tampered'"))
        with pytest.raises(Exception, match="append-only"):
            conn.execute(text("DELETE FROM audit_log"))
