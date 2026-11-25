"""Line-item validation: does each line map to a known product, and is its price within tolerance?

SKU match is exact first (precision) and cross-checked against the description. If the SKU is absent or
unknown, the description is matched two ways: lexically (rapidfuzz) and semantically (embeddings). Either
may accept, each behind its own threshold *and* a margin over the runner-up - "Copy paper" alone fits two
SKUs, so it fits neither.
"""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.extraction.schema import LineItem, PurchaseOrder
from app.rag.rules import load_rules
from app.rag.store import CATALOG, Product, get_index, load_catalog

logger = logging.getLogger(__name__)

_UNIT_SYNONYMS = {
    "EA": "EA EACH PC PCS PCE PIECE PIECES UNIT UNITS NO NOS",
    "BOX": "BOX BOXES BX", "CS": "CS CASE CASES", "SPL": "SPL SPOOL SPOOLS", "BDL": "BDL BUNDLE BUNDLES",
    "PK": "PK PACK PACKS PKT", "RL": "RL ROLL ROLLS", "SET": "SET SETS", "RK": "RK RACK RACKS",
}
_UNIT = {syn: unit for unit, syns in _UNIT_SYNONYMS.items() for syn in syns.split()}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.strip().upper().rstrip(".")
    return _UNIT.get(u, u)


def desc_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    return float(max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b) * 0.95))


def _semantic_scores(text: str, settings: Settings) -> dict[str, float]:
    index = get_index(settings)
    if index is None:
        return {}
    try:
        return {h["id"]: h["similarity"] for h in index.query(CATALOG, text, k=5)}
    except Exception as exc:
        logger.warning("Catalog vector retrieval failed; lexical fallback", extra={"error": str(exc)})
        return {}


def _top_two(scores: dict[str, float]) -> tuple[Optional[str], float, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    return ranked[0][0], ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0


def match_product(li: LineItem, catalog: list[Product], settings: Settings) -> dict:
    """-> {product, method, score, semantic, sku_mismatch, note}"""
    out = {"product": None, "method": "none", "score": 0.0, "semantic": None, "sku_mismatch": False, "note": ""}
    by_sku = {p.sku: p for p in catalog}
    semantic = _semantic_scores(li.description, settings) if li.description else {}

    sku = li.sku.strip().upper() if li.sku else None
    if sku in by_sku:
        product = by_sku[sku]
        out.update(product=product, method="sku_exact", score=100.0, semantic=semantic.get(sku))
        if li.description:
            lexical = desc_similarity(li.description, product.description)
            if lexical < 45 and (semantic.get(sku) or 0.0) < 0.65:
                out["sku_mismatch"] = True
                out["note"] = (f"SKU {sku} is '{product.description}' in the catalog, but the line reads "
                               f"'{li.description}'")
        return out
    if not li.description:
        return out

    lexical = {p.sku: desc_similarity(li.description, p.description) for p in catalog}
    best, score, second = _top_two(lexical)
    out["score"] = round(score, 1)
    if score >= settings.catalog_match_min and score - second >= settings.catalog_ambiguity_margin:
        out.update(product=by_sku[best], method="description_lexical", semantic=semantic.get(best))
        return out

    s_best, s_score, s_second = _top_two(semantic)
    out["semantic"] = round(s_score, 3) if semantic else None
    if s_best in by_sku and s_score >= settings.catalog_semantic_min and \
            s_score - s_second >= settings.catalog_semantic_margin:
        out.update(product=by_sku[s_best], method="description_semantic", score=round(lexical[s_best], 1))
        return out

    guess = s_best if s_best in by_sku else best
    if guess:
        out["note"] = f"closest is {guess} '{by_sku[guess].description}'"
    return out


def validate_line_items(po: PurchaseOrder, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    catalog = load_catalog(settings.seed_path)
    tol = load_rules(settings).price_tolerance_pct
    checks = []
    for i, li in enumerate(po.line_items):
        m = match_product(li, catalog, settings)
        product = m["product"]
        no = li.line_no if li.line_no is not None else i + 1
        check = {
            "index": i, "line_no": li.line_no, "sku": li.sku, "description": li.description,
            "quantity": li.quantity, "unit": li.unit, "unit_price": li.unit_price,
            "matched_sku": product.sku if product else None, "match_method": m["method"], "match_score": m["score"],
            "match_semantic": m["semantic"],
            "catalog_description": product.description if product else None,
            "catalog_unit_price": product.standard_unit_price if product else None,
            "catalog_unit": product.unit if product else None, "category": product.category if product else None,
            "deviation_pct": None, "uom_mismatch": False, "sku_mismatch": m["sku_mismatch"],
            "status": "ok", "reason": "",
        }
        if product is None:
            check["status"] = "unknown_item"
            label = " ".join(x for x in (li.sku, li.description) if x)
            extra = f", semantic {m['semantic']:.2f}" if m["semantic"] is not None else ""
            extra += f"; {m['note']}" if m["note"] else ""
            check["reason"] = f"No catalog product matches line {no} ('{label}', best string score {m['score']:g}{extra})"
        elif li.unit_price is None:
            check["status"] = "no_price"
            check["reason"] = f"Line {no} has no unit price to validate"
        else:
            dev = (li.unit_price - product.standard_unit_price) / product.standard_unit_price * 100
            check["deviation_pct"] = round(dev, 2)
            if abs(dev) > tol:
                check["status"] = "price_anomaly"
                check["reason"] = (f"Line {no} {product.sku}: unit price {li.unit_price:,.2f} is "
                                   f"{dev:+.1f}% vs catalog {product.standard_unit_price:,.2f} (tolerance ±{tol:g}%)")
            else:
                check["reason"] = f"Line {no} {product.sku}: {dev:+.1f}% vs catalog - within tolerance"
        if m["sku_mismatch"]:
            check["reason"] += f" [{m['note']}]"
        if product and li.unit and product.unit and normalize_unit(li.unit) != normalize_unit(product.unit):
            check["uom_mismatch"] = True
            check["reason"] += f" [unit {li.unit} vs catalog {product.unit}]"
        checks.append(check)
    return checks
