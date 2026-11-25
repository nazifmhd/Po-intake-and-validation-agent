"""Ablation: run the same POs with the knowledge base off, with lexical matching only, and with full hybrid RAG.

    python scripts/ablation.py [files ...]      (default: demo + eval samples)

off      the agent still extracts and checks arithmetic and confidence - but unknown/blocked vendors,
         off-contract prices, non-compliant terms and over-threshold spend are no longer caught
lexical  knowledge base on, embeddings off (EMBEDDING_PROVIDER=none)
hybrid   embeddings + fuzzy matching (the default)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT = ["01_clean_northwind.pdf", "02_scanned_globex.jpg", "03_unknown_vendor.csv", "04_acme_alias_twocol.pdf",
           "05_blocked_vendor.pdf", "06_math_error.pdf", "07_bad_terms.pdf", "09_no_sku_paraphrased.csv",
           "10_lookalike_vendor.pdf"]


def main() -> None:
    files = [Path(f) for f in sys.argv[1:]] or [ROOT / "data" / "samples" / f for f in DEFAULT]
    tmp = Path(tempfile.mkdtemp(prefix="po-ablation-"))
    os.environ["STORAGE_DIR"] = str(tmp / "storage")
    os.environ["SLACK_WEBHOOK_URL"] = ""
    os.environ["GOOGLE_SHEETS_ENABLED"] = "false"

    import logging
    logging.disable(logging.ERROR)
    from app.config import get_settings, reset_settings_cache
    from app.db import reset_engines
    from app.graph.build import run_pipeline
    from app.rag.ingest_kb import build_knowledge_base
    from app.rag.store import reset_index_cache

    if get_settings().embedding_provider == "none":
        sys.exit("The hybrid column needs EMBEDDING_PROVIDER=local or openai")
    hybrid = get_settings().embedding_provider

    results: dict[str, dict] = {}
    for mode, rag, embeddings in (("off", False, "none"), ("lexical", True, "none"), ("hybrid", True, hybrid)):
        os.environ["EMBEDDING_PROVIDER"] = embeddings
        os.environ["DATABASE_URL"] = f"sqlite:///{(tmp / f'{mode}.db').as_posix()}"  # fresh DB: no dedup
        reset_settings_cache()
        reset_engines()
        reset_index_cache()
        build_knowledge_base(get_settings(), if_missing=True)
        for f in files:
            results.setdefault(f.name, {})[mode] = run_pipeline(str(f), options={"rag_enabled": rag})

    def cell(o: dict) -> str:
        return "AUTO-APPROVE" if o["decision"] == "auto_approve" else "review: " + ", ".join(o["flags"])

    print(f"{'document':27s} | {'KB off':45s} | {'lexical only':45s} | hybrid RAG")
    print("-" * 165)
    caught = better = 0
    for name, r in results.items():
        print(f"{name:27s} | {cell(r['off']):45s} | {cell(r['lexical']):45s} | {cell(r['hybrid'])}")
        caught += r["off"]["decision"] == "auto_approve" and r["hybrid"]["decision"] == "needs_review"
        better += (r["lexical"]["decision"], r["lexical"]["flags"]) != (r["hybrid"]["decision"], r["hybrid"]["flags"])
    print(f"\n{caught} PO(s) would be auto-filed without the knowledge base; "
          f"{better} PO(s) get a different outcome because of the embeddings.")


if __name__ == "__main__":
    main()
