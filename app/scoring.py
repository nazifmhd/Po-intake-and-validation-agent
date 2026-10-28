"""Stage 5 - scoring: fuse independent signals into one auditable decision + a list of flags.

Deterministic on purpose - never "ask the LLM if it's confident". Every flag carries a
human-readable reason, so every routing decision can be explained to a reviewer.
Auto-approve only when the flag list is empty.
"""
from __future__ import annotations

from typing import Optional

from app.config import Settings, get_settings
from app.extraction.schema import REQUIRED_HEADER_FIELDS, PurchaseOrder

# Flags that come from a policy finding of severity "violation"
POLICY_FLAG_CODES = {"over_threshold", "exceeds_po_limit", "noncompliant_terms", "non_base_currency",
                     "policy_violation"}


def _tol(value: float, settings: Settings) -> float:
    return max(settings.arithmetic_abs_tolerance, settings.arithmetic_rel_tolerance * abs(value))


def arithmetic_check(po: PurchaseOrder, settings: Settings | None = None) -> dict:
    """Pure Python, no LLM: catches OCR / extraction errors the model is blind to."""
    settings = settings or get_settings()
    issues: list[str] = []
    checked = 0

    for li in po.line_items:
        if li.quantity is not None and li.unit_price is not None and li.line_total is not None:
            checked += 1
            expected = round(li.quantity * li.unit_price, 2)
            if abs(expected - li.line_total) > _tol(li.line_total, settings):
                issues.append(f"Line {li.line_no}: {li.quantity:g} x {li.unit_price:,.2f} = {expected:,.2f}, "
                              f"but line total reads {li.line_total:,.2f}")

    line_totals = [li.line_total for li in po.line_items]
    lines_sum = round(sum(line_totals), 2) if line_totals and all(t is not None for t in line_totals) else None

    if lines_sum is not None and po.subtotal is not None:
        checked += 1
        if abs(lines_sum - po.subtotal) > _tol(po.subtotal, settings):
            issues.append(f"Sum of line totals {lines_sum:,.2f} != subtotal {po.subtotal:,.2f}")

    base = po.subtotal if po.subtotal is not None else lines_sum
    if po.total is not None and base is not None:
        checked += 1
        expected_total = round(base + (po.tax or 0) + (po.shipping or 0), 2)
        if abs(expected_total - po.total) > _tol(po.total, settings):
            issues.append(f"Subtotal {base:,.2f} + tax {po.tax or 0:,.2f} + shipping {po.shipping or 0:,.2f} "
                          f"= {expected_total:,.2f}, but total reads {po.total:,.2f}")

    return {"ok": not issues, "checks_performed": checked, "issues": issues, "lines_sum": lines_sum}


def build_signals(po: PurchaseOrder, vendor_match: Optional[dict], price_checks: list[dict],
                  policy_findings: list[dict], arithmetic: dict, rag_enabled: bool,
                  settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    confs = []
    if po.vendor_name.present:
        confs.append(("vendor_name", po.vendor_name.confidence))
    if po.total is not None:
        confs.append(("total", po.totals_confidence))
    for li in po.line_items:
        confs.append((f"line_{li.line_no}", li.confidence))
    weakest = min(confs, key=lambda c: c[1]) if confs else (None, 0.0)

    missing = []
    if not po.vendor_name.present:
        missing.append("vendor_name")
    if po.total is None:
        missing.append("total")
    if not po.line_items:
        missing.append("line_items")
    missing += [f for f in REQUIRED_HEADER_FIELDS if f != "vendor_name" and not getattr(po, f).present]

    vm = vendor_match or {}
    return {
        "rag_enabled": rag_enabled,
        "critical_field_confidence": round(weakest[1], 3),
        "weakest_critical_field": weakest[0],
        "overall_confidence": po.overall_confidence,
        "missing_critical_fields": missing,
        "vendor_match_status": vm.get("status", "not_checked"),
        "vendor_status": vm.get("vendor_status", "not_checked"),
        "vendor_match_score": vm.get("score"),
        "vendor_tax_id_match": vm.get("tax_id_match"),
        "price_violations": [c for c in price_checks if c["status"] == "price_anomaly"],
        "unknown_items": [c for c in price_checks if c["status"] == "unknown_item"],
        "policy_violations": [f for f in policy_findings if f.get("severity") == "violation"],
        "arithmetic_ok": arithmetic.get("ok", True),
        "arithmetic_issues": arithmetic.get("issues", []),
        "total": po.total,
        "currency": po.currency.value,
    }


def decide(po: PurchaseOrder, signals: dict, settings: Settings | None = None) -> tuple[str, list[str], dict]:
    """Returns (decision, flags, reasons). decision is 'auto_approve' only if flags is empty."""
    settings = settings or get_settings()
    flags: list[str] = []
    reasons: dict[str, str] = {}

    def flag(code: str, reason: str) -> None:
        if code not in flags:
            flags.append(code)
            reasons[code] = reason
        else:
            reasons[code] += f"; {reason}"

    # 1. vendor (RAG: approved_vendors)
    if signals["rag_enabled"]:
        vms = signals["vendor_match_status"]
        if signals["vendor_status"] == "blocked":
            flag("vendor_blocked", "Vendor is on the registry but BLOCKED - do not transact")
        if vms == "unknown":
            flag("vendor_unknown", f"Vendor '{po.vendor_name.value}' is not in the approved-vendor registry")
        elif vms == "ambiguous":
            flag("vendor_ambiguous", "Vendor name matches more than one registry vendor")
        elif vms == "uncertain":
            flag("vendor_low_match", f"Vendor match score {signals['vendor_match_score']} is below the "
                                     f"{settings.vendor_match_high:g} acceptance band")
        if signals["vendor_tax_id_match"] is False:
            flag("vendor_tax_id_mismatch", "Tax ID on the PO does not match the registry record for this vendor")

    # 2. price validation (RAG: product_catalog)
    for c in signals["price_violations"]:
        flag("price_anomaly", c["reason"])
    for c in signals["unknown_items"]:
        flag("unknown_item", c["reason"])

    # 3. policy compliance (RAG: procurement_policy)
    for f in signals["policy_violations"]:
        code = f["code"] if f["code"] in POLICY_FLAG_CODES else "policy_violation"
        cite = f" [policy: {f['clause']}]" if f.get("clause") else ""
        flag(code, f["message"] + cite)

    # 4. arithmetic (pure Python)
    if not signals["arithmetic_ok"]:
        flag("math_mismatch", "; ".join(signals["arithmetic_issues"]))

    # 5. extraction confidence + completeness
    if signals["critical_field_confidence"] < settings.critical_field_confidence_min and \
            signals["weakest_critical_field"] is not None:
        flag("low_extraction_confidence",
             f"Critical field '{signals['weakest_critical_field']}' confidence "
             f"{signals['critical_field_confidence']:.2f} < {settings.critical_field_confidence_min:.2f}")
    if signals["missing_critical_fields"]:
        flag("missing_fields", f"Missing: {', '.join(signals['missing_critical_fields'])}")

    return ("auto_approve" if not flags else "needs_review"), flags, reasons
