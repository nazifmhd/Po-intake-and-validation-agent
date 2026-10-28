"""The extraction output contract.

`PurchaseOrderExtraction` is what the model fills. `PurchaseOrder` adds the fields the
agent fills afterwards (overall confidence, warnings) - the model never sees those.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field, field_validator


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


class Confidenced(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: Optional[str] = None  # the text/region this came from

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return _clamp01(v)

    @classmethod
    def missing(cls) -> "Confidenced":
        return cls(value=None, confidence=0.0, evidence=None)

    @property
    def present(self) -> bool:
        return self.value is not None and str(self.value).strip() != ""


class LineItem(BaseModel):
    line_no: Optional[int] = None
    sku: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return _clamp01(v)


class PurchaseOrderExtraction(BaseModel):
    """Exactly what the LLM is asked to produce."""

    po_number: Confidenced
    po_date: Confidenced
    vendor_name: Confidenced
    vendor_tax_id: Confidenced
    buyer_department: Confidenced
    currency: Confidenced
    line_items: list[LineItem]
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    shipping: Optional[float] = None
    total: Optional[float] = None
    totals_confidence: float = Field(default=0.0, ge=0, le=1)
    totals_evidence: Optional[str] = None
    payment_terms: Confidenced
    delivery_date: Confidenced

    @field_validator("totals_confidence", mode="before")
    @classmethod
    def _clamp_totals_confidence(cls, v: Any) -> float:
        return _clamp01(v)


class PurchaseOrder(PurchaseOrderExtraction):
    # filled by the agent, not the model:
    overall_confidence: float = 0.0
    extraction_warnings: list[str] = Field(default_factory=list)
    extraction_method: str = ""

    HEADER_FIELDS: ClassVar[tuple[str, ...]] = (
        "po_number", "po_date", "vendor_name", "vendor_tax_id", "buyer_department",
        "currency", "payment_terms", "delivery_date",
    )

    def field_confidences(self) -> dict[str, float]:
        out = {name: getattr(self, name).confidence for name in self.HEADER_FIELDS}
        out["totals"] = self.totals_confidence
        out["line_items"] = min((li.confidence for li in self.line_items), default=0.0)
        return out


# Fields the policy requires on every PO (see data/seed/policy.md, "Required fields")
REQUIRED_HEADER_FIELDS = ("po_number", "po_date", "vendor_name", "delivery_date")
# Critical fields used by the decision policy (blueprint section 4)
CRITICAL_FIELDS = ("vendor_name", "total", "line_items")
