"""LangGraph wiring:  ingest -> extract -> enrich -> score -> route -> {store | review | skip} -> audit."""
from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Optional

from langgraph.graph import END, StateGraph

from app.graph import nodes
from app.graph.state import POState

logger = logging.getLogger(__name__)


def build_graph():
    g = StateGraph(POState)
    g.add_node("ingest", nodes.ingest)
    g.add_node("extract", nodes.extract_node)
    g.add_node("enrich", nodes.enrich)
    g.add_node("score", nodes.score)
    g.add_node("route", nodes.route)
    g.add_node("store", nodes.store)
    g.add_node("review", nodes.review)
    g.add_node("skip", nodes.skip)
    g.add_node("audit", nodes.audit_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "enrich")
    g.add_edge("enrich", "score")
    g.add_edge("score", "route")
    g.add_conditional_edges("route", nodes.route_decision,
                            {"store": "store", "review": "review", "skip": "skip"})
    for terminal in ("store", "review", "skip"):
        g.add_edge(terminal, "audit")
    g.add_edge("audit", END)
    return g.compile()


@lru_cache
def get_graph():
    return build_graph()


def run_pipeline(file_path: str, *, job_id: Optional[str] = None, message_id: Optional[str] = None,
                 options: Optional[dict] = None) -> dict:
    job_id = job_id or uuid.uuid4().hex[:12]
    initial: POState = {"job_id": job_id, "file_path": str(file_path), "source_message_id": message_id,
                        "options": options or {}, "errors": [], "trace": [], "flags": [], "reasons": {}}
    logger.info("pipeline started", extra={"job_id": job_id, "file": str(file_path)})
    final = get_graph().invoke(initial)
    logger.info("pipeline finished", extra={"job_id": job_id, "decision": final.get("decision"),
                                            "flags": final.get("flags")})
    return final


def summarize(state: dict) -> dict:
    """Compact, JSON-safe view of a finished run (API responses, CLI output)."""
    po = state.get("po") or {}
    vm = state.get("vendor_match") or {}
    return {
        "job_id": state.get("job_id"), "file": state.get("file_path"), "doc_type": state.get("doc_type"),
        "decision": state.get("decision"), "flags": state.get("flags"), "reasons": state.get("reasons"),
        "record": {"table": state.get("record_table"), "id": state.get("record_id")},
        "po_number": (po.get("po_number") or {}).get("value"),
        "vendor_as_printed": (po.get("vendor_name") or {}).get("value"),
        "vendor_match": {k: vm.get(k) for k in ("status", "vendor_id", "canonical_name", "vendor_status", "score")},
        "total": po.get("total"), "currency": (po.get("currency") or {}).get("value"),
        "extraction_method": po.get("extraction_method"),
        "overall_confidence": po.get("overall_confidence"),
        "errors": state.get("errors"), "trace": state.get("trace"),
    }
