"""Line-item validation: does each line map to a known product, and is its price within tolerance?

SKU match is exact first (precision); if the SKU is absent or unknown, the description is
retrieved semantically and re-ranked with rapidfuzz before a match is accepted.
"""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.extraction.schema import LineItem, PurchaseOrder
from app.rag.store import CATALOG, Product, get_index, load_catalog

logger = logging.getLogger(__name__)


def _desc_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    return float(max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b) * 0.95))


def _match_product(li: LineItem, catalog: list[Product], settings: Settings) -> tuple[Optional[Product], str, float]:
    by_sku = {p.sku: p for p in catalog}
    if li.sku:
        sku = li.sku.strip().upper()
        if sku in by_sku:
            return by_sku[sku], "sku_exact", 100.0
    if not li.description:
        return None, "none", 0.0

    candidates: list[Product] = []
    index = get_index(settings)
    if index is not None:
        try:
            hits = index.query(CATALOG, f"{li.sku or ''} {li.description}".strip(), k=5)
            candidates = [by_sku[h["id"]] for h in hits if h["id"] in by_sku]
            method = "semantic+fuzzy"
        except Exception as exc:
            logger.warning("Catalog vector retrieval failed; lexical fallback", extra={"error": str(exc)})
    if not candidates:
        candidates, method = catalog, "lexical"

    best, best_score = None, 0.0
    for p in candidates:
        s = _desc_similarity(li.description, p.description)
        if s > best_score:
            best, best_score = p, s
    if best is not None and best_score >= settings.catalog_match_min:
        return best, f"description_{method}", round(best_score, 1)
    return None, method, round(best_score, 1)


def validate_line_items(po: PurchaseOrder, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    catalog = load_catalog(settings.seed_path)
    tol = settings.price_tolerance_pct
    checks = []
    for li in po.line_items:
        product, method, score = _match_product(li, catalog, settings)
        check = {
            "line_no": li.line_no, "sku": li.sku, "description": li.description,
            "quantity": li.quantity, "unit": li.unit, "unit_price": li.unit_price,
            "matched_sku": product.sku if product else None, "match_method": method, "match_score": score,
            "catalog_description": product.description if product else None,
            "catalog_unit_price": product.standard_unit_price if product else None,
            "catalog_unit": product.unit if product else None,
            "deviation_pct": None, "uom_mismatch": False, "status": "ok", "reason": "",
        }
        if product is None:
            check["status"] = "unknown_item"
            label = " ".join(x for x in (li.sku, li.description) if x)
            check["reason"] = f"No catalog product matches line {li.line_no} ('{label}', best score {score})"
        elif li.unit_price is None:
            check["status"] = "no_price"
            check["reason"] = f"Line {li.line_no} has no unit price to validate"
        else:
            dev = (li.unit_price - product.standard_unit_price) / product.standard_unit_price * 100
            check["deviation_pct"] = round(dev, 2)
            if abs(dev) > tol:
                check["status"] = "price_anomaly"
                check["reason"] = (f"Line {li.line_no} {product.sku}: unit price {li.unit_price:,.2f} is "
                                   f"{dev:+.1f}% vs catalog {product.standard_unit_price:,.2f} (tolerance ±{tol:g}%)")
            else:
                check["reason"] = f"Line {li.line_no} {product.sku}: {dev:+.1f}% vs catalog - within tolerance"
        if product and li.unit and product.unit and li.unit != product.unit:
            check["uom_mismatch"] = True
            check["reason"] += f" [unit {li.unit} vs catalog {product.unit}]"
        checks.append(check)
    return checks
