"""Graph nodes. Each is a function POState -> partial POState update.

Fail-safe by construction: `@safe_node` catches any exception, records it in state["errors"]
with a machine-readable code, and lets the graph continue. The route node turns any error into
a review flag - the default direction of failure is toward a human, never toward auto-approve.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from app.actions.audit import audit
from app.actions.notify import notify_approved, notify_duplicate, notify_review, send_slack
from app.actions.sheets import mirror_row
from app.actions.store import (
    FILING_LOCK, FilingConflict, check_duplicate, enqueue_review, store_approved, write_dead_letter,
)
from app.config import get_settings
from app.extraction.extractor import extract
from app.extraction.schema import PurchaseOrder
from app.graph.state import POState
from app.ingestion.loaders import RawDocument, load_document
from app.rag.catalog import validate_line_items
from app.rag.policy import check_policy, find_split
from app.rag.rules import load_rules
from app.rag.vendors import match_vendor
from app.scoring import arithmetic_check, build_signals, decide

logger = logging.getLogger(__name__)


def safe_node(name: str) -> Callable:
    def wrap(fn: Callable[[POState], dict]) -> Callable[[POState], dict]:
        @functools.wraps(fn)
        def inner(state: POState) -> dict:
            started = time.perf_counter()
            try:
                update = fn(state) or {}
                ok, err = True, None
            except Exception as exc:  # noqa: BLE001 - this IS the safety net
                code = getattr(exc, "code", None) or "processing_error"
                err = {"node": name, "code": code, "type": type(exc).__name__, "message": str(exc)[:1000]}
                logger.exception("Node failed; item will be routed to review",
                                 extra={"node": name, "job_id": state.get("job_id"), "code": code})
                update, ok = {"errors": [err]}, False
            ms = round((time.perf_counter() - started) * 1000, 1)
            update["trace"] = [{"node": name, "ms": ms, "ok": ok}]
            logger.info("node finished", extra={"node": name, "job_id": state.get("job_id"), "ms": ms, "ok": ok})
            return update
        return inner
    return wrap


def _rag_enabled(state: POState) -> bool:
    return (state.get("options") or {}).get("rag_enabled", get_settings().rag_enabled)


def _raw_doc(state: POState) -> RawDocument:
    return RawDocument(
        doc_type=state.get("doc_type") or "unknown", raw_text=state.get("raw_text") or "",
        text_source=state.get("text_source") or "none", page_images=state.get("page_images") or [],
        csv_rows=state.get("csv_rows") or [], warnings=state.get("ingest_warnings") or [],
        vision_images=state.get("vision_images") or [], flags=dict(state.get("ingest_flags") or {}),
    )


# ----------------------------------------------------------------------------- nodes
@safe_node("ingest")
def ingest(state: POState) -> dict:
    pages_dir = get_settings().storage_path / "pages" / state["job_id"]
    doc = load_document(state["file_path"], pages_dir)
    return {"doc_type": doc.doc_type, "raw_text": doc.raw_text, "text_source": doc.text_source,
            "page_images": doc.page_images, "vision_images": doc.vision_images, "csv_rows": doc.csv_rows,
            "ingest_warnings": doc.warnings, "ingest_flags": doc.flags}


@safe_node("extract")
def extract_node(state: POState) -> dict:
    if not state.get("doc_type"):
        return {}  # ingest failed - its error is already recorded
    doc = _raw_doc(state)
    po = extract(doc)
    return {"po": po.model_dump(), "ingest_flags": doc.flags}


@safe_node("enrich")
def enrich(state: POState) -> dict:
    if not state.get("po"):
        return {}
    if not _rag_enabled(state):
        return {"vendor_match": {"status": "not_checked", "vendor_status": "not_checked",
                                 "reason": "RAG disabled (ablation run)"},
                "price_checks": [], "policy_findings": [], "policy_clauses": []}

    po = PurchaseOrder.model_validate(state["po"])
    update: dict = {"errors": []}
    # Each validation is independent: one failing must not stop the others from running.
    try:
        update["vendor_match"] = match_vendor(po.vendor_name.value, po.vendor_tax_id.value)
    except Exception as exc:
        update["errors"].append({"node": "enrich.vendor", "code": "vendor_check_failed", "message": str(exc)})
        update["vendor_match"] = {"status": "error", "vendor_status": "unknown", "reason": str(exc)}
    try:
        update["price_checks"] = validate_line_items(po)
    except Exception as exc:
        update["errors"].append({"node": "enrich.catalog", "code": "catalog_check_failed", "message": str(exc)})
        update["price_checks"] = []
    try:
        policy = check_policy(po, update.get("vendor_match"), price_checks=update.get("price_checks"),
                              exclude_review_id=state.get("exclude_review_id"))
        update["policy_findings"], update["policy_clauses"] = policy["findings"], policy["clauses"]
    except Exception as exc:
        update["errors"].append({"node": "enrich.policy", "code": "policy_check_failed", "message": str(exc)})
        update["policy_findings"], update["policy_clauses"] = [], []
    if not update["errors"]:
        del update["errors"]
    return update


@safe_node("score")
def score(state: POState) -> dict:
    if not state.get("po"):
        return {"decision": "needs_review", "flags": ["extraction_failed"],
                "reasons": {"extraction_failed": "No purchase order could be extracted from the document"},
                "signals": None}
    po = PurchaseOrder.model_validate(state["po"])
    arithmetic = arithmetic_check(po)
    signals = build_signals(po, state.get("vendor_match"), state.get("price_checks") or [],
                            state.get("policy_findings") or [], arithmetic, _rag_enabled(state),
                            ingest_flags=state.get("ingest_flags"))
    decision, flags, reasons = decide(po, signals)

    duplicate = check_duplicate(state["po"], exclude_review_id=state.get("exclude_review_id"))
    if duplicate:
        where = f"{duplicate['table']} #{duplicate['id']} ({duplicate['status']})"
        if duplicate["kind"] == "duplicate" and not state.get("exclude_review_id"):
            decision = "duplicate"
            reasons = {**reasons, "duplicate_po": f"PO {po.po_number.value} already exists in {where}"}
        else:
            decision = "needs_review"
            flags = [*flags, "po_number_conflict"]
            reasons = {**reasons, "po_number_conflict": f"PO {po.po_number.value} already exists in {where}: "
                       + ("; ".join(duplicate["differences"]) or "same content")}
    return {"arithmetic": arithmetic, "signals": signals, "decision": decision, "flags": flags,
            "reasons": reasons, "duplicate_of": duplicate}


@safe_node("route")
def route(state: POState) -> dict:
    """Final say on the decision: any recorded error forces review, with its reason attached."""
    errors = state.get("errors") or []
    decision = state.get("decision") or "needs_review"
    flags = list(state.get("flags") or [])
    reasons = dict(state.get("reasons") or {})
    if decision == "duplicate" and not errors:
        return {"decision": "duplicate"}
    for err in errors:
        code = err.get("code") or "processing_error"
        if code not in flags:
            flags.append(code)
        reasons[code] = f"{err.get('node')}: {err.get('message')}"
    if flags and decision == "auto_approve":
        decision = "needs_review"
    if not flags and decision != "auto_approve":
        flags, reasons = ["unexplained_review"], {"unexplained_review": "Pipeline did not reach a decision"}
    if decision == "duplicate":  # duplicate *and* errors: let a human look
        decision = "needs_review"
        flags.append("possible_duplicate")
        reasons["possible_duplicate"] = str(state.get("duplicate_of"))
    return {"decision": decision, "flags": flags, "reasons": reasons}


def route_decision(state: POState) -> str:
    decision = state.get("decision")
    if state.get("errors") and decision == "auto_approve":  # belt and braces if `route` itself failed
        return "review"
    return {"auto_approve": "store", "duplicate": "skip"}.get(decision, "review")


def _file(state: POState) -> int:
    """Re-check, then file, as one step: two POs scored at the same moment can each look clean on their own."""
    with FILING_LOCK:
        duplicate = check_duplicate(state["po"])
        if duplicate:
            raise FilingConflict("po_number_conflict", f"PO number was filed meanwhile in {duplicate['table']} "
                                                       f"#{duplicate['id']}")
        if _rag_enabled(state):
            po = PurchaseOrder.model_validate(state["po"])
            settings = get_settings()
            split = find_split(po, state.get("vendor_match"), state.get("price_checks") or [], load_rules(settings),
                               settings)
            if split:
                raise FilingConflict("possible_po_split", split)
        return store_approved(state)


def _queue(state: POState) -> int:
    """Put the item in the review queue. If even that fails it goes to the dead-letter folder, with an alert."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            return enqueue_review(state)
        except Exception as exc:
            last = exc
            logger.warning("Review queue write failed", extra={"job_id": state.get("job_id"), "attempt": attempt + 1,
                                                              "error": str(exc)[:200]})
            time.sleep(0.2 * (attempt + 1))
    path = write_dead_letter(state, str(last))
    send_slack(f":rotating_light: Could not queue job `{state.get('job_id')}` for review ({str(last)[:200]}). "
               f"Saved to {path}; the job will be retried.")
    raise RuntimeError(f"review queue write failed: {last}") from last


@safe_node("store")
def store(state: POState) -> dict:
    try:
        po_id = _file(state)
    except Exception as exc:
        # Could not file it - fall back to the review queue rather than lose it.
        code = getattr(exc, "code", None) if isinstance(exc, FilingConflict) else "storage_error"
        err = {"node": "store", "code": code, "message": str(exc)[:500]}
        fallback = {**state, "decision": "needs_review", "flags": [code], "reasons": {code: err["message"]},
                    "errors": [*(state.get("errors") or []), err]}
        update = {"errors": [err], "decision": "needs_review", "flags": [code], "reasons": fallback["reasons"]}
        try:
            review_id = _queue(fallback)
        except Exception as queue_exc:
            return {**update, "errors": [err, {"node": "store", "code": "queue_error", "message": str(queue_exc)}]}
        notify_review(state["po"], [code], fallback["reasons"], review_id, state.get("vendor_match"))
        return {**update, "record_id": review_id, "record_table": "review_queue"}
    po = state["po"]
    vm = state.get("vendor_match") or {}
    notify_approved(po, vm, po_id)
    mirror_row("Approved", [po_id, po["po_number"]["value"], po["po_date"]["value"],
                            vm.get("canonical_name") or po["vendor_name"]["value"], vm.get("vendor_id"),
                            po.get("total"), po["currency"]["value"], "approved", ""])
    return {"record_id": po_id, "record_table": "purchase_orders"}


@safe_node("review")
def review(state: POState) -> dict:
    review_id = _queue(state)
    po = state.get("po") or {}
    notify_review(po, state.get("flags") or [], state.get("reasons") or {}, review_id, state.get("vendor_match"))
    mirror_row("Needs Review", [review_id, (po.get("po_number") or {}).get("value"),
                                (po.get("vendor_name") or {}).get("value"), po.get("total"),
                                (po.get("currency") or {}).get("value"), ", ".join(state.get("flags") or []),
                                "pending_review", ""])
    return {"record_id": review_id, "record_table": "review_queue"}


@safe_node("skip")
def skip(state: POState) -> dict:
    notify_duplicate(state.get("po") or {}, state.get("duplicate_of") or {})
    return {"record_id": (state.get("duplicate_of") or {}).get("id"),
            "record_table": (state.get("duplicate_of") or {}).get("table")}


# ----------------------------------------------------------------------------- re-evaluation (reviewer actions)
def _apply(state: dict, update: dict) -> dict:
    merged = {**state, **update}
    for key in ("errors", "trace"):  # same reducers the graph uses
        merged[key] = [*(state.get(key) or []), *(update.get(key) or [])]
    return merged


def evaluate(state: dict) -> dict:
    """enrich -> score -> route on an already-extracted PO, through the same fail-safe nodes as a live run."""
    for node in (enrich, score, route):
        state = _apply(state, node(state))
    return state


def extract_again(file_path: str, job_id: str) -> dict:
    state: dict = {"job_id": job_id, "file_path": file_path, "errors": [], "trace": []}
    for node in (ingest, extract_node):
        state = _apply(state, node(state))
    return state


def audit_node(state: POState) -> dict:
    """Always runs last, on every path. Not wrapped: audit() itself never raises."""
    po = state.get("po") or {}
    audit(
        "pipeline_completed", job_id=state.get("job_id"), decision=state.get("decision"), flags=state.get("flags"),
        details={
            "file": state.get("file_path"), "doc_type": state.get("doc_type"),
            "message_id": state.get("source_message_id"),
            "po_number": (po.get("po_number") or {}).get("value"),
            "vendor_raw": (po.get("vendor_name") or {}).get("value"), "total": po.get("total"),
            "extraction_method": po.get("extraction_method"),
            "record": {"table": state.get("record_table"), "id": state.get("record_id")},
            "reasons": state.get("reasons"), "signals": state.get("signals"),
            "vendor_match": {k: (state.get("vendor_match") or {}).get(k) for k in
                             ("status", "vendor_id", "canonical_name", "vendor_status", "score", "method")},
            "errors": state.get("errors"), "trace": state.get("trace"),
            "options": state.get("options"),
        },
    )
    return {}
