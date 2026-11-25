"""Stage 5 - scoring: fuse independent signals into one auditable decision + a list of flags.

Deterministic on purpose - never "ask the LLM if it's confident". Every flag carries a
human-readable reason, so every routing decision can be explained to a reviewer.
Auto-approve only when the flag list is empty - and a check that could not be *performed* is a flag too:
"nothing to check" is not the same as "checked and fine".
"""
from __future__ import annotations

import math
from typing import Optional

from app.config import Settings, get_settings
from app.extraction.schema import REQUIRED_HEADER_FIELDS, LineItem, PurchaseOrder

# Flags that come from a policy finding of severity "violation"
POLICY_FLAG_CODES = {"over_threshold", "exceeds_po_limit", "noncompliant_terms", "non_base_currency",
                     "possible_po_split", "policy_rule_missing", "policy_review_unavailable", "policy_violation"}


def _tol(value: float, settings: Settings) -> float:
    return max(settings.arithmetic_abs_tolerance, settings.arithmetic_rel_tolerance * abs(value))


def _line_amount(li: LineItem) -> Optional[float]:
    if li.line_total is not None:
        return li.line_total
    if li.quantity is not None and li.unit_price is not None:
        return round(li.quantity * li.unit_price, 2)
    return None


def arithmetic_check(po: PurchaseOrder, settings: Settings | None = None) -> dict:
    """Pure Python, no LLM: catches OCR / extraction errors the model is blind to."""
    settings = settings or get_settings()
    issues: list[str] = []
    checked = 0

    for i, li in enumerate(po.line_items, start=1):
        if li.quantity is not None and li.unit_price is not None and li.line_total is not None:
            checked += 1
            expected = round(li.quantity * li.unit_price, 2)
            if abs(expected - li.line_total) > _tol(li.line_total, settings):
                issues.append(f"Line {li.line_no or i}: {li.quantity:g} x {li.unit_price:,.2f} = {expected:,.2f}, "
                              f"but line total reads {li.line_total:,.2f}")

    amounts = [_line_amount(li) for li in po.line_items]
    lines_sum = round(sum(amounts), 2) if amounts and all(a is not None for a in amounts) else None

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

    # verified = the grand total was traced all the way back to the individual lines
    return {"ok": not issues, "verified": lines_sum is not None and po.total is not None,
            "checks_performed": checked, "issues": issues, "lines_sum": lines_sum}


def _bad(value: Optional[float], minimum: float = 0.0, strict: bool = False) -> bool:
    if value is None:
        return False
    return not math.isfinite(value) or value < minimum or (strict and value == minimum)


def amount_issues(po: PurchaseOrder) -> list[str]:
    """Negative lines or credits can pull a large order under the approval limit, so none are booked unseen."""
    issues = []
    for i, li in enumerate(po.line_items, start=1):
        no = li.line_no or i
        if _bad(li.quantity, strict=True):
            issues.append(f"line {no} quantity {li.quantity:g}")
        if _bad(li.unit_price, strict=True):
            issues.append(f"line {no} unit price {li.unit_price:,.2f}")
        if _bad(li.line_total):
            issues.append(f"line {no} total {li.line_total:,.2f}")
    for name in ("subtotal", "tax", "shipping"):
        if _bad(getattr(po, name)):
            issues.append(f"{name} {getattr(po, name):,.2f}")
    if _bad(po.total, strict=True):
        issues.append(f"total {po.total:,.2f}")
    return issues


def incomplete_lines(po: PurchaseOrder) -> list[str]:
    out = []
    for i, li in enumerate(po.line_items, start=1):
        gaps = [n for n in ("quantity", "unit_price") if getattr(li, n) is None]
        if not li.sku and not li.description:
            gaps.append("sku/description")
        if gaps:
            out.append(f"line {li.line_no or i}: no {', '.join(gaps)}")
    return out


def build_signals(po: PurchaseOrder, vendor_match: Optional[dict], price_checks: list[dict],
                  policy_findings: list[dict], arithmetic: dict, rag_enabled: bool,
                  settings: Settings | None = None, ingest_flags: Optional[dict] = None) -> dict:
    settings = settings or get_settings()
    confs = []
    for name in ("vendor_name", "po_number", "currency"):
        if getattr(po, name).present:
            confs.append((name, getattr(po, name).confidence))
    if po.total is not None:
        confs.append(("total", po.totals_confidence))
    for i, li in enumerate(po.line_items, start=1):
        confs.append((f"line_{li.line_no or i}", li.confidence))
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
        "incomplete_lines": incomplete_lines(po),
        "amount_issues": amount_issues(po),
        "vendor_match_status": vm.get("status", "not_checked"),
        "vendor_status": vm.get("vendor_status", "not_checked"),
        "vendor_match_score": vm.get("score"),
        "vendor_match_reason": vm.get("reason"),
        "vendor_tax_id_match": vm.get("tax_id_match"),
        "price_violations": [c for c in price_checks if c["status"] == "price_anomaly"],
        "unknown_items": [c for c in price_checks if c["status"] == "unknown_item"],
        "unit_mismatches": [c for c in price_checks if c.get("uom_mismatch")],
        "sku_mismatches": [c for c in price_checks if c.get("sku_mismatch")],
        "policy_violations": [f for f in policy_findings if f.get("severity") == "violation"],
        "arithmetic_ok": arithmetic.get("ok", False),
        "arithmetic_verified": arithmetic.get("verified", False),
        "arithmetic_issues": arithmetic.get("issues", []),
        "ingest_flags": ingest_flags or {},
        "additional_po_numbers": po.additional_po_numbers,
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
            flag("vendor_ambiguous", signals.get("vendor_match_reason") or "Vendor name matches several records")
        elif vms == "uncertain":
            flag("vendor_low_match", signals.get("vendor_match_reason") or "Vendor match is weak")
        elif vms != "matched" and vms != "missing":  # "missing" is reported as a missing field below
            flag("vendor_not_verified", f"Vendor could not be verified (status: {vms})")
        if signals["vendor_tax_id_match"] is False:
            flag("vendor_tax_id_mismatch", "Tax ID on the PO does not match the registry record for this vendor")

    # 2. price validation (RAG: product_catalog)
    for c in signals["price_violations"]:
        flag("price_anomaly", c["reason"])
    for c in signals["unknown_items"]:
        flag("unknown_item", c["reason"])
    for c in signals["unit_mismatches"]:
        flag("unit_mismatch", c["reason"])
    for c in signals["sku_mismatches"]:
        flag("sku_mismatch", c["reason"])

    # 3. policy compliance (RAG: procurement_policy)
    for f in signals["policy_violations"]:
        code = f["code"] if f["code"] in POLICY_FLAG_CODES else "policy_violation"
        cite = f" [policy: {f['clause']}]" if f.get("clause") else ""
        flag(code, f["message"] + cite)

    # 4. arithmetic (pure Python)
    if not signals["arithmetic_ok"]:
        flag("math_mismatch", "; ".join(signals["arithmetic_issues"]))
    elif not signals["arithmetic_verified"] and po.line_items and po.total is not None:
        flag("math_unverified", "The total could not be checked against the line items")
    if signals["amount_issues"]:
        flag("invalid_amount", "Zero, negative or invalid amounts: " + ", ".join(signals["amount_issues"]))

    # 5. extraction confidence + completeness
    if signals["critical_field_confidence"] < settings.critical_field_confidence_min and \
            signals["weakest_critical_field"] is not None:
        flag("low_extraction_confidence",
             f"Critical field '{signals['weakest_critical_field']}' confidence "
             f"{signals['critical_field_confidence']:.2f} < {settings.critical_field_confidence_min:.2f}")
    if signals["missing_critical_fields"]:
        flag("missing_fields", f"Missing: {', '.join(signals['missing_critical_fields'])}")
    if signals["incomplete_lines"]:
        flag("incomplete_lines", "; ".join(signals["incomplete_lines"]))

    # 6. the document itself
    for code, reason in signals["ingest_flags"].items():
        flag(code, reason)
    if signals["additional_po_numbers"]:
        flag("multiple_pos_in_document",
             f"The document also contains {', '.join(signals['additional_po_numbers'])}, which was not processed")

    return ("auto_approve" if not flags else "needs_review"), flags, reasons
