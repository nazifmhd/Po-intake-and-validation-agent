"""The single shared state object that flows through every node."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class POState(TypedDict, total=False):
    # --- input ---
    job_id: str
    file_path: str
    source_message_id: Optional[str]
    options: dict                         # per-run overrides, e.g. {"rag_enabled": False} for the ablation

    # --- ingest ---
    doc_type: Optional[str]               # pdf_text | pdf_scanned | image | csv
    raw_text: Optional[str]
    text_source: Optional[str]            # text_layer | ocr | csv | none
    page_images: list[str]
    csv_rows: list[dict]
    ingest_warnings: list[str]

    # --- extract ---
    po: Optional[dict]                    # PurchaseOrder.model_dump()

    # --- enrich (RAG) ---
    vendor_match: Optional[dict]
    price_checks: list[dict]
    policy_findings: list[dict]
    policy_clauses: list[dict]

    # --- score / route ---
    arithmetic: Optional[dict]
    signals: Optional[dict]
    duplicate_of: Optional[dict]
    decision: Optional[str]               # auto_approve | needs_review | duplicate
    flags: list[str]
    reasons: dict

    # --- action ---
    record_id: Optional[int]              # purchase_orders.id or review_queue.id
    record_table: Optional[str]

    # --- accumulated across nodes (reducers append) ---
    errors: Annotated[list[dict], operator.add]
    trace: Annotated[list[dict], operator.add]
