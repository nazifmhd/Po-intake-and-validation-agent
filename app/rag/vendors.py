"""Vendor validation: is this a real, active vendor - and which record is it, despite spelling drift?

Hybrid matching (blueprint section 5):
  1. semantic retrieval over every vendor name + alias  -> candidate vendors
     (+ exact tax-ID lookup, which is identity evidence independent of spelling)
  2. fuzzy re-rank (rapidfuzz) on normalized names       -> precise similarity score
  3. confidence band: high -> accept, medium -> accept but flag, low/none -> unknown
     plus an ambiguity check when two different vendors score within a small margin.
"""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz

from app.config import Settings, get_settings
from app.rag.store import VENDORS, Vendor, get_index, load_vendors, normalize_name, normalize_tax_id

logger = logging.getLogger(__name__)


def load_registry(settings: Settings | None = None) -> list[Vendor]:
    """Seed vendors + aliases learned from reviewer decisions."""
    settings = settings or get_settings()
    vendors = load_vendors(settings.seed_path)
    try:
        from sqlmodel import select

        from app.db import session_scope
        from app.models import VendorAlias

        with session_scope(settings) as s:
            learned = s.exec(select(VendorAlias)).all()
        by_id = {v.vendor_id: v for v in vendors}
        for a in learned:
            if a.vendor_id in by_id and a.alias not in by_id[a.vendor_id].all_names:
                by_id[a.vendor_id].learned_aliases.append(a.alias)
    except Exception as exc:  # registry must still work if the DB is unavailable
        logger.warning("Could not load learned vendor aliases", extra={"error": str(exc)})
    return vendors


def sync_learned_aliases(settings: Settings | None = None, force: bool = False) -> int:
    """Index learned aliases into Chroma. Only the API/worker process writes to the index."""
    settings = settings or get_settings()
    index = get_index(settings)
    if index is None:
        return 0
    try:
        from sqlmodel import select

        from app.db import session_scope
        from app.models import VendorAlias

        with session_scope(settings) as s:
            learned = s.exec(select(VendorAlias)).all()
        if not learned:
            return 0
        ids = [f"learned-{a.id}" for a in learned]
        have = set() if force else index.existing_ids(VENDORS, ids)
        new = [(i, a) for i, a in zip(ids, learned) if i not in have]
        if new:
            index.upsert(VENDORS, [i for i, _ in new], [a.alias for _, a in new],
                         [{"vendor_id": a.vendor_id, "canonical_name": "", "status": "", "is_alias": True}
                          for _, a in new])
        return len(new)
    except Exception as exc:
        logger.warning("Learned alias sync failed", extra={"error": str(exc)})
        return 0


def name_similarity(a: str, b: str) -> float:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:  # name was nothing but legal suffixes - compare raw
        na, nb = a.lower().strip(), b.lower().strip()
    # ratio + token_sort only: partial/token_set ratios over-reward a single shared word ("Supplies")
    return float(max(fuzz.ratio(na, nb), fuzz.token_sort_ratio(na, nb)))


def match_vendor(name: Optional[str], tax_id: Optional[str] = None,
                 settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    if not name and not tax_id:
        return {"status": "missing", "vendor_id": None, "canonical_name": None, "vendor_status": "unknown",
                "score": 0.0, "method": "none", "tax_id_match": None, "candidates": [],
                "reason": "No vendor name or tax ID was extracted"}

    registry = load_registry(settings)
    by_id = {v.vendor_id: v for v in registry}
    candidate_ids: dict[str, dict] = {}

    tid = normalize_tax_id(tax_id or "")
    if tid:
        for v in registry:
            if normalize_tax_id(v.tax_id) == tid:
                candidate_ids.setdefault(v.vendor_id, {})["tax_id"] = True

    method = "lexical"
    index = get_index(settings)
    if index is not None and name:
        try:
            sync_learned_aliases(settings)
            for hit in index.query(VENDORS, name, k=8):
                vid = hit["metadata"]["vendor_id"]
                entry = candidate_ids.setdefault(vid, {})
                entry["semantic"] = max(entry.get("semantic", 0.0), hit["similarity"])
            method = "semantic+fuzzy"
        except Exception as exc:
            logger.warning("Vector retrieval failed; lexical fallback", extra={"error": str(exc)})
            index = None
    if index is None and name:
        for v in registry:  # small registry: score everything
            candidate_ids.setdefault(v.vendor_id, {})

    scored = []
    for vid, info in candidate_ids.items():
        v = by_id.get(vid)
        if v is None:
            continue
        fuzzy, best_alias = 0.0, v.name
        if name:
            for alias in v.all_names:
                s = name_similarity(name, alias)
                if s > fuzzy:
                    fuzzy, best_alias = s, alias
        score = fuzzy
        if info.get("tax_id"):
            # tax ID is identity evidence; a very different name still needs a human look
            score = max(fuzzy, 95.0) if fuzzy >= 50 else max(fuzzy, 80.0)
        scored.append({"vendor_id": vid, "canonical_name": v.name, "matched_alias": best_alias,
                       "vendor_status": v.status, "score": round(score, 1), "fuzzy": round(fuzzy, 1),
                       "semantic": info.get("semantic"), "tax_id_match": bool(info.get("tax_id"))})
    scored.sort(key=lambda c: c["score"], reverse=True)
    top = scored[:3]

    if not scored or scored[0]["score"] < settings.vendor_match_medium:
        return {"status": "unknown", "vendor_id": None, "canonical_name": None, "vendor_status": "unknown",
                "score": scored[0]["score"] if scored else 0.0, "method": method, "tax_id_match": None,
                "candidates": top, "reason": f"No registry vendor matches '{name}' "
                f"(best score {scored[0]['score'] if scored else 0} < {settings.vendor_match_medium})"}

    best = scored[0]
    vendor = by_id[best["vendor_id"]]
    tax_id_match = None
    if tid and vendor.tax_id:
        tax_id_match = normalize_tax_id(vendor.tax_id) == tid

    second = next((c for c in scored[1:] if c["vendor_id"] != best["vendor_id"]), None)
    if second and second["score"] >= settings.vendor_match_medium and \
            best["score"] - second["score"] < settings.vendor_ambiguity_margin and not best["tax_id_match"]:
        status = "ambiguous"
        reason = (f"'{name}' matches both {best['canonical_name']} ({best['score']}) and "
                  f"{second['canonical_name']} ({second['score']})")
    elif best["score"] >= settings.vendor_match_high:
        status, reason = "matched", f"Matched '{name}' to {vendor.name} via '{best['matched_alias']}' ({best['score']})"
    else:
        status = "uncertain"
        reason = f"Weak match '{name}' -> {vendor.name} ({best['score']}, below {settings.vendor_match_high})"

    return {"status": status, "vendor_id": vendor.vendor_id, "canonical_name": vendor.name,
            "vendor_status": vendor.status, "score": best["score"], "method": method,
            "tax_id_match": tax_id_match, "negotiated_terms": vendor.payment_terms,
            "vendor_notes": vendor.notes, "candidates": top, "reason": reason}
