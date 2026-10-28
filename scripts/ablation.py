"""Ablation: run the same POs with the knowledge base ON and OFF and show what RAG catches.

    python scripts/ablation.py [files ...]      (default: demo + eval samples)

With RAG off, the agent still extracts, checks arithmetic and confidence - but unknown/blocked
vendors, off-contract prices and over-threshold spend are no longer caught.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT = ["01_clean_northwind.pdf", "02_scanned_globex.jpg", "03_unknown_vendor.csv", "04_acme_alias_twocol.pdf",
           "05_blocked_vendor.pdf", "06_math_error.pdf", "07_bad_terms.pdf"]


def main() -> None:
    files = [Path(f) for f in sys.argv[1:]] or [ROOT / "data" / "samples" / f for f in DEFAULT]
    tmp = Path(tempfile.mkdtemp(prefix="po-ablation-"))
    os.environ["STORAGE_DIR"] = str(tmp / "storage")
    os.environ["SLACK_WEBHOOK_URL"] = ""
    os.environ["GOOGLE_SHEETS_ENABLED"] = "false"

    import logging
    logging.disable(logging.ERROR)
    from app.config import reset_settings_cache
    from app.db import reset_engines
    from app.graph.build import run_pipeline

    results = {}
    for mode, rag in (("RAG on", True), ("RAG off", False)):
        os.environ["DATABASE_URL"] = f"sqlite:///{(tmp / f'{mode[-2:].strip()}.db').as_posix()}"  # fresh DB: no dedup
        reset_settings_cache()
        reset_engines()
        for f in files:
            out = run_pipeline(str(f), options={"rag_enabled": rag})
            results.setdefault(f.name, {})[mode] = out

    print(f"{'document':28s} | {'RAG on':52s} | RAG off")
    print("-" * 130)
    caught = 0
    for name, r in results.items():
        on, off = r["RAG on"], r["RAG off"]
        cell = lambda o: f"{o['decision']} {o['flags']}"  # noqa: E731
        marker = ""
        if on["decision"] == "needs_review" and off["decision"] == "auto_approve":
            marker, caught = "  <-- only RAG catches this", caught + 1
        print(f"{name:28s} | {cell(on):52s} | {cell(off)}{marker}")
    print(f"\n{caught} PO(s) would have been auto-filed without the knowledge base.")


if __name__ == "__main__":
    main()
