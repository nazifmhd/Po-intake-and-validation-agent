import pytest

from app.config import get_settings
from app.extraction.schema import Confidenced, LineItem, PurchaseOrder
from app.rag.catalog import validate_line_items
from app.rag.policy import check_policy
from app.rag.vendors import match_vendor
from app.scoring import arithmetic_check, build_signals, decide


def C(v, c=0.95):
    return Confidenced(value=v, confidence=c, evidence=v)


def make_po(lines=None, vendor="Northwind Office Products LLC", tax=296.0, shipping=204.0, total=None,
            terms="Net 30", delivery="2026-11-05", currency="USD", tax_id=None, line_conf=0.95, total_conf=0.95):
    lines = lines or [("OFF-CHR-ERG", "Ergonomic mesh office chair with lumbar support", 10, 285.0),
                      ("OFF-LMP-LED", "LED desk lamp with dimmer", 20, 42.5)]
    items = [LineItem(line_no=i, sku=s, description=d, quantity=q, unit="EA", unit_price=p,
                      line_total=round(q * p, 2), confidence=line_conf) for i, (s, d, q, p) in enumerate(lines, 1)]
    subtotal = round(sum(li.line_total for li in items), 2)
    return PurchaseOrder(
        po_number=C("PO-1"), po_date=C("2026-10-20"), vendor_name=C(vendor) if vendor else Confidenced.missing(),
        vendor_tax_id=C(tax_id) if tax_id else Confidenced.missing(), buyer_department=C("Ops"),
        currency=C(currency), payment_terms=C(terms), delivery_date=C(delivery) if delivery else Confidenced.missing(),
        line_items=items, subtotal=subtotal, tax=tax, shipping=shipping,
        total=total if total is not None else round(subtotal + tax + shipping, 2), totals_confidence=total_conf,
    )


def run(po, rag=True):
    vm = match_vendor(po.vendor_name.value, po.vendor_tax_id.value) if rag else None
    checks = validate_line_items(po) if rag else []
    policy = check_policy(po, vm)["findings"] if rag else []
    signals = build_signals(po, vm, checks, policy, arithmetic_check(po), rag)
    return decide(po, signals)


# ---------------------------------------------------------------- arithmetic
def test_arithmetic_ok():
    assert arithmetic_check(make_po())["ok"]


def test_arithmetic_catches_wrong_total_and_line():
    po = make_po(total=4300.0)
    po.line_items[0].line_total = 2800.0
    res = arithmetic_check(po)
    assert not res["ok"] and len(res["issues"]) >= 2


# ---------------------------------------------------------------- vendor RAG
@pytest.mark.parametrize("name,vid", [
    ("Acme Corp", "V001"), ("ACME Corporation (Pvt) Ltd", "V001"), ("acme industrial supplies, inc", "V001"),
    ("GLOBEX TECH SOLUTIONS INC", "V003"), ("Northwind Office Products", "V002"),
])
def test_vendor_spelling_drift_matches(name, vid):
    m = match_vendor(name)
    assert m["status"] == "matched" and m["vendor_id"] == vid


def test_unknown_vendor():
    m = match_vendor("Pinnacle Promo Goods Ltd")
    assert m["status"] == "unknown" and m["vendor_id"] is None


def test_blocked_vendor_is_matched_but_blocked():
    m = match_vendor("Shadow Supply Traders")
    assert m["vendor_id"] == "V015" and m["vendor_status"] == "blocked"


def test_tax_id_mismatch_detected():
    m = match_vendor("Northwind Office Products LLC", "US-99-0000000")
    assert m["tax_id_match"] is False


def test_tax_id_rescues_odd_spelling():
    m = match_vendor("NW Office Prod.", "US-23-4567890")
    assert m["vendor_id"] == "V002" and m["tax_id_match"] is True


def test_missing_vendor():
    assert match_vendor(None)["status"] == "missing"


# ---------------------------------------------------------------- catalog RAG
def test_price_anomaly_28_percent():
    po = make_po(lines=[("IT-LAP-14P", "14-inch business laptop", 5, 2368.0)])
    chk = validate_line_items(po)[0]
    assert chk["status"] == "price_anomaly" and chk["deviation_pct"] == pytest.approx(28.0)


def test_price_within_tolerance_and_description_match():
    po = make_po(lines=[(None, "Cordless drill driver kit 18V", 2, 165.0)])
    chk = validate_line_items(po)[0]
    assert chk["matched_sku"] == "TLS-DRL-18V" and chk["status"] == "ok"


def test_unknown_item():
    po = make_po(lines=[("ZZZ-999", "Gold plated espresso machine", 1, 9000.0)])
    assert validate_line_items(po)[0]["status"] == "unknown_item"


# ---------------------------------------------------------------- policy RAG
def test_policy_threshold_and_citation():
    po = make_po(lines=[("IT-LAP-14P", "laptop", 6, 1850.0)])
    findings = check_policy(po, match_vendor(po.vendor_name.value))["findings"]
    f = next(f for f in findings if f["code"] == "over_threshold")
    assert f["clause"] == "Auto-approval threshold and sign-off"


def test_policy_hard_limit_and_terms():
    po = make_po(lines=[("IT-LAP-14P", "laptop", 30, 1850.0)], terms="Net 90")
    codes = {f["code"] for f in check_policy(po, None)["findings"]}
    assert {"exceeds_po_limit", "noncompliant_terms"} <= codes and "over_threshold" not in codes


def test_policy_clauses_are_retrieved():
    clauses = check_policy(make_po(), None)["clauses"]
    headings = {c["heading"] for c in clauses}
    assert "Allowed payment terms" in headings and "Auto-approval threshold and sign-off" in headings


# ---------------------------------------------------------------- decision policy
def test_clean_po_auto_approves():
    decision, flags, _ = run(make_po())
    assert decision == "auto_approve" and flags == []


@pytest.mark.parametrize("kwargs,flag", [
    ({"vendor": "Pinnacle Promo Goods Ltd"}, "vendor_unknown"),
    ({"vendor": "Shadow Supply Traders"}, "vendor_blocked"),
    ({"total": 9999.0}, "math_mismatch"),
    ({"terms": "Net 90"}, "noncompliant_terms"),
    ({"currency": "EUR"}, "non_base_currency"),
    ({"delivery": None}, "missing_fields"),
    ({"vendor": None}, "missing_fields"),
    ({"line_conf": 0.55}, "low_extraction_confidence"),
    ({"total_conf": 0.4}, "low_extraction_confidence"),
    ({"tax_id": "US-00-0000000"}, "vendor_tax_id_mismatch"),
    ({"lines": [("IT-LAP-14P", "laptop", 5, 2368.0)]}, "price_anomaly"),
    ({"lines": [("IT-LAP-14P", "laptop", 6, 1850.0)]}, "over_threshold"),
])
def test_each_signal_forces_review(kwargs, flag):
    decision, flags, reasons = run(make_po(**kwargs))
    assert decision == "needs_review" and flag in flags and reasons[flag]


def test_ablation_without_rag_misses_vendor_price_and_threshold():
    """The point of RAG: with the KB off, a blocked vendor, +28% price and $13.5k total sail through."""
    po = make_po(vendor="Shadow Supply Traders", lines=[("IT-LAP-14P", "laptop", 5, 2368.0)], shipping=0, tax=0)
    assert run(po, rag=True)[0] == "needs_review"
    decision, flags, _ = run(po, rag=False)
    assert decision == "auto_approve" and flags == []


def test_thresholds_come_from_config(monkeypatch):
    from app.config import reset_settings_cache
    monkeypatch.setenv("AUTO_APPROVAL_LIMIT", "1000")
    reset_settings_cache()
    assert get_settings().auto_approval_limit == 1000
    assert "over_threshold" in run(make_po())[1]
