"""Policy compliance: retrieval-augmented decision-making, not lookup.

1. Build retrieval queries from the PO's actual situation (its total, terms, vendor status...).
2. Retrieve the relevant clauses from the procurement_policy collection.
3. Deterministic checks for the rules that have hard numbers (threshold, hard limit, allowed
   terms, required fields) - every finding cites the clause it is grounded in.
4. Optional LLM review: the model reads the retrieved clauses + PO facts and may ADD findings
   the rules missed (it can never remove a deterministic flag).
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel
from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.extraction.schema import REQUIRED_HEADER_FIELDS, PurchaseOrder
from app.rag.store import POLICY, PolicyChunk, get_index, load_policy_chunks

logger = logging.getLogger(__name__)


class PolicyFindingLLM(BaseModel):
    clause: str
    finding: str
    severity: Literal["violation", "info"]


class PolicyReviewLLM(BaseModel):
    findings: list[PolicyFindingLLM]


def retrieve_clauses(queries: list[str], settings: Settings, k: int = 2) -> list[dict]:
    chunks = load_policy_chunks(settings.seed_path)
    by_id = {c.chunk_id: c for c in chunks}
    found: dict[str, dict] = {}
    index = get_index(settings)
    for q in queries:
        hits: list[tuple[PolicyChunk, float]] = []
        if index is not None:
            try:
                hits = [(by_id[h["id"]], h["similarity"]) for h in index.query(POLICY, q, k=k) if h["id"] in by_id]
            except Exception as exc:
                logger.warning("Policy vector retrieval failed; lexical fallback", extra={"error": str(exc)})
        if not hits:  # lexical fallback
            ranked = sorted(chunks, key=lambda c: fuzz.token_set_ratio(q.lower(), f"{c.heading} {c.text}".lower()),
                            reverse=True)
            hits = [(c, fuzz.token_set_ratio(q.lower(), f"{c.heading} {c.text}".lower()) / 100) for c in ranked[:k]]
        for chunk, score in hits:
            prev = found.get(chunk.chunk_id)
            if prev is None or score > prev["score"]:
                found[chunk.chunk_id] = {"chunk_id": chunk.chunk_id, "heading": chunk.heading,
                                         "text": chunk.text, "score": round(float(score), 3), "query": q}
    return sorted(found.values(), key=lambda c: c["score"], reverse=True)


def _cite(clauses: list[dict], *keywords: str) -> Optional[str]:
    for c in clauses:
        hay = (c["heading"] + " " + c["text"]).lower()
        if all(k.lower() in hay for k in keywords):
            return c["heading"]
    return None


def check_policy(po: PurchaseOrder, vendor_match: Optional[dict], settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    total = po.total
    terms = po.payment_terms.value
    currency = po.currency.value or settings.base_currency
    vstatus = (vendor_match or {}).get("vendor_status", "unknown")

    queries = [
        f"approval threshold and sign-off required for a purchase order total of {total or 'unknown'} {currency}",
        f"maximum single purchase order spend limit {total or ''}",
        f"allowed payment terms; is '{terms or 'missing'}' acceptable",
        "required fields on every purchase order: PO number, date, vendor, delivery date, line items, total",
        f"vendor approval requirements; vendor status {vstatus}; unregistered or blocked suppliers",
        f"currency of purchase orders {currency}; price variance against catalog",
    ]
    clauses = retrieve_clauses(queries, settings)
    findings: list[dict] = []

    def add(code: str, severity: str, message: str, *cite_kw: str) -> None:
        findings.append({"code": code, "severity": severity, "message": message,
                         "clause": _cite(clauses, *cite_kw) if cite_kw else None, "source": "rule"})

    if total is not None:
        if total > settings.single_po_hard_limit:
            add("exceeds_po_limit", "violation",
                f"Total {total:,.2f} exceeds the single-PO hard limit of {settings.single_po_hard_limit:,.0f}; "
                "requires CFO approval and three competitive quotes", "limit")
        elif total > settings.auto_approval_limit:
            add("over_threshold", "violation",
                f"Total {total:,.2f} is above the {settings.auto_approval_limit:,.0f} auto-approval limit; "
                "requires department director sign-off", "approval")
    if terms:
        if terms not in settings.allowed_terms_list:
            add("noncompliant_terms", "violation",
                f"Payment terms '{terms}' are not in the allowed list ({', '.join(settings.allowed_terms_list)})",
                "payment terms")
        negotiated = (vendor_match or {}).get("negotiated_terms")
        if negotiated and terms != negotiated and terms in settings.allowed_terms_list:
            add("terms_differ_from_contract", "info",
                f"Payment terms '{terms}' differ from the vendor's negotiated terms '{negotiated}'", "payment terms")
    if currency and currency != settings.base_currency:
        add("non_base_currency", "violation",
            f"PO currency {currency} is not the base currency {settings.base_currency}; "
            "threshold checks require conversion and finance review", "currency")
    missing = [f for f in REQUIRED_HEADER_FIELDS if not getattr(po, f).present]
    if missing:  # the flag itself is raised by the scorer; this finding attaches the policy citation
        add("required_fields_clause", "info", f"Required fields missing: {', '.join(missing)}", "required")

    # --- LLM reasoning over the retrieved clauses (can only add findings) ---
    if settings.policy_llm_review and settings.llm_provider in ("anthropic", "openai") and clauses:
        try:
            findings.extend(_llm_policy_review(po, vendor_match, clauses, findings, settings))
        except Exception as exc:
            findings.append({"code": "policy_review_unavailable", "severity": "info", "clause": None,
                             "message": f"LLM policy review failed: {exc}", "source": "llm"})
    return {"clauses": clauses, "findings": findings}


def _llm_policy_review(po, vendor_match, clauses, rule_findings, settings) -> list[dict]:
    from app.extraction.prompts import POLICY_REVIEW_SYSTEM
    from app.llm import LLMClient, to_json

    facts = {
        "po_number": po.po_number.value, "po_date": po.po_date.value, "vendor": po.vendor_name.value,
        "vendor_registry_status": (vendor_match or {}).get("vendor_status"),
        "buyer_department": po.buyer_department.value, "currency": po.currency.value,
        "payment_terms": po.payment_terms.value, "delivery_date": po.delivery_date.value,
        "subtotal": po.subtotal, "tax": po.tax, "shipping": po.shipping, "total": po.total,
        "line_items": [li.model_dump(exclude={"evidence", "confidence"}) for li in po.line_items],
    }
    policy_text = "\n\n".join(f"## {c['heading']}\n{c['text']}" for c in clauses)
    text = (f"PO facts:\n{to_json(facts)}\n\nDeterministic checks already raised:\n"
            f"{to_json([{k: f[k] for k in ('code', 'message')} for f in rule_findings])}\n\n"
            f"Retrieved policy clauses:\n<policy>\n{policy_text}\n</policy>")
    review = LLMClient(settings).structured(system=POLICY_REVIEW_SYSTEM, text=text,
                                            response_model=PolicyReviewLLM, max_tokens=4000)
    headings = {c["heading"].lower() for c in clauses}
    out = []
    for f in review.findings:
        grounded = any(h in f.clause.lower() or f.clause.lower() in h for h in headings)
        if not grounded:  # ungrounded citations are discarded, not trusted
            logger.info("Dropping ungrounded LLM policy finding", extra={"clause": f.clause})
            continue
        out.append({"code": "policy_violation" if f.severity == "violation" else "policy_note",
                    "severity": f.severity, "message": f.finding, "clause": f.clause, "source": "llm"})
    return out
