"""Build the three Chroma collections from data/seed.

    python -m app.cli kb build            # (re)build everything
    python -m app.cli kb build --if-missing

approved_vendors    one document per vendor *name or alias* (so "Acme Corp" and
                    "ACME Corporation (Pvt) Ltd" are both directly retrievable), metadata -> vendor_id
product_catalog     one document per SKU: "SKU - description (category, unit)"
procurement_policy  one document per policy clause (## heading)
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.rag.store import (
    CATALOG, POLICY, VENDORS, get_index, load_catalog, load_policy_chunks, load_vendors,
)

logger = logging.getLogger(__name__)


def build_knowledge_base(settings: Settings | None = None, if_missing: bool = False) -> dict:
    settings = settings or get_settings()
    index = get_index(settings)
    if index is None:
        logger.info("EMBEDDING_PROVIDER=none - no vector index to build (lexical retrieval)")
        return {"status": "skipped", "reason": "embedding_provider=none"}
    if if_missing and index.is_built():
        logger.info("Knowledge base already built", extra={"embedder": index.embedder.name})
        return {"status": "exists"}

    seed = settings.seed_path
    vendors, catalog, policy = load_vendors(seed), load_catalog(seed), load_policy_chunks(seed)

    for name in (VENDORS, CATALOG, POLICY):
        index.reset(name)

    ids, docs, metas = [], [], []
    for v in vendors:
        for j, alias in enumerate(v.all_names):
            ids.append(f"{v.vendor_id}-n{j}")
            docs.append(alias)
            metas.append({"vendor_id": v.vendor_id, "canonical_name": v.name, "status": v.status,
                          "is_alias": j > 0})
    index.upsert(VENDORS, ids, docs, metas)

    index.upsert(
        CATALOG,
        [p.sku for p in catalog],
        [f"{p.sku} - {p.description} ({p.category}, per {p.unit})" for p in catalog],
        [{"sku": p.sku, "standard_unit_price": p.standard_unit_price, "unit": p.unit} for p in catalog],
    )
    index.upsert(
        POLICY,
        [c.chunk_id for c in policy],
        [f"{c.heading}\n{c.text}" for c in policy],
        [{"heading": c.heading} for c in policy],
    )

    # learned aliases (from reviewer feedback) survive a rebuild
    from app.rag.vendors import sync_learned_aliases
    synced = sync_learned_aliases(settings, force=True)

    summary = {"status": "built", "embedder": index.embedder.name, "vendor_names": len(ids),
               "skus": len(catalog), "policy_chunks": len(policy), "learned_aliases": synced}
    logger.info("Knowledge base built", extra=summary)
    return summary
