"""Database tables (SQLModel -> SQLite by default, Postgres-ready via DATABASE_URL)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PurchaseOrderRecord(SQLModel, table=True):
    __tablename__ = "purchase_orders"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    po_number: Optional[str] = Field(default=None, index=True)
    po_date: Optional[str] = None
    vendor_name_raw: Optional[str] = None
    vendor_id: Optional[str] = Field(default=None, index=True)
    vendor_canonical_name: Optional[str] = None
    buyer_department: Optional[str] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    shipping: Optional[float] = None
    total: Optional[float] = None
    payment_terms: Optional[str] = None
    delivery_date: Optional[str] = None
    status: str = "approved"  # approved | approved_by_review
    overall_confidence: float = 0.0
    source_file: str = ""
    source_message_id: Optional[str] = None
    extracted_json: str = "{}"
    signals_json: str = "{}"
    approved_by: str = "agent"
    created_at: datetime = Field(default_factory=utcnow)


class LineItemRecord(SQLModel, table=True):
    __tablename__ = "line_items"

    id: Optional[int] = Field(default=None, primary_key=True)
    po_id: int = Field(foreign_key="purchase_orders.id", index=True)
    line_no: Optional[int] = None
    sku: Optional[str] = None
    matched_sku: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    catalog_unit_price: Optional[float] = None
    price_deviation_pct: Optional[float] = None


class ReviewQueueItem(SQLModel, table=True):
    __tablename__ = "review_queue"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    po_number: Optional[str] = Field(default=None, index=True)
    vendor_name_raw: Optional[str] = None
    vendor_match_json: str = "{}"
    total: Optional[float] = None
    currency: Optional[str] = None
    status: str = Field(default="pending_review", index=True)  # pending_review | approved | rejected
    flags_json: str = "[]"
    reasons_json: str = "{}"
    field_confidences_json: str = "{}"
    extracted_json: str = "{}"
    signals_json: str = "{}"
    price_checks_json: str = "[]"
    policy_findings_json: str = "[]"
    errors_json: str = "[]"
    source_file: str = ""
    doc_type: Optional[str] = None
    source_message_id: Optional[str] = None
    reviewer: Optional[str] = None
    review_note: Optional[str] = None
    promoted_po_id: Optional[int] = None
    created_at: datetime = Field(default_factory=utcnow)
    reviewed_at: Optional[datetime] = None


class AuditLog(SQLModel, table=True):
    """Append-only. UPDATE/DELETE are blocked by triggers on SQLite (see app/db.py)."""

    __tablename__ = "audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    job_id: Optional[str] = Field(default=None, index=True)
    event: str
    actor: str = "agent"
    decision: Optional[str] = None
    flags_json: str = "[]"
    details_json: str = "{}"


class ProcessedEmail(SQLModel, table=True):
    __tablename__ = "processed_emails"

    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: str = Field(unique=True, index=True)
    sender: Optional[str] = None
    subject: Optional[str] = None
    source: str = "webhook"  # webhook | imap | upload | cli
    attachment_count: int = 0
    status: str = "accepted"  # accepted | no_attachment | rejected_sender | all_rejected
    note: Optional[str] = None
    received_at: datetime = Field(default_factory=utcnow)


class Job(SQLModel, table=True):
    """One job per (supported) attachment - or per PO inside a CSV batch."""

    __tablename__ = "jobs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(unique=True, index=True)
    message_id: Optional[str] = Field(default=None, index=True)
    filename: str = ""
    stored_path: str = ""
    sha256: str = ""
    status: str = "queued"  # queued | processing | done | rejected | failed
    decision: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None


class VendorAlias(SQLModel, table=True):
    """Aliases learned from reviewer decisions (stretch: vendor alias learning)."""

    __tablename__ = "vendor_aliases"

    id: Optional[int] = Field(default=None, primary_key=True)
    vendor_id: str = Field(index=True)
    alias: str
    created_by: str = "reviewer"
    created_at: datetime = Field(default_factory=utcnow)
