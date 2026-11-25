"""Does the vector store earn its place? Measure matching with and without embeddings.

    python scripts/eval_retrieval.py [-v]

data/eval/retrieval_cases.csv holds paraphrased items, misspelled vendors, look-alikes that must NOT match, and
policy questions. Thresholds were chosen on the `dev` rows only; the `test` rows are the honest number.
    fuzzy         plain string similarity with the same acceptance threshold - the obvious alternative
    lexical       what the agent does when the vector store is down: vendors need an exact name (stricter on purpose)
    hybrid        embeddings + fuzzy matching (the default)

    correct       right record accepted, or a look-alike correctly not accepted
    false accept  a wrong record accepted - the only kind of error that could auto-approve a bad PO
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="po-retrieval-"))
    os.environ["DATABASE_URL"] = f"sqlite:///{(tmp / 'eval.db').as_posix()}"
    logging.disable(logging.ERROR)

    from app.config import get_settings, reset_settings_cache
    from app.extraction.schema import LineItem
    from app.rag.catalog import desc_similarity, match_product
    from app.rag.ingest_kb import build_knowledge_base
    from app.rag.policy import retrieve_clauses
    from app.rag.store import load_catalog, load_vendors, reset_index_cache
    from app.rag.vendors import match_vendor, name_similarity

    with open(ROOT / "data" / "eval" / "retrieval_cases.csv", newline="", encoding="utf-8") as fh:
        cases = list(csv.DictReader(fh))
    hybrid = get_settings().embedding_provider
    if hybrid == "none":
        sys.exit("Set EMBEDDING_PROVIDER=local or openai")

    def fuzzy_only(kind: str, query: str, s) -> str:
        if kind == "vendor":
            best = max(load_vendors(s.seed_path), key=lambda v: max(name_similarity(query, n) for n in v.all_names))
            return best.vendor_id if max(name_similarity(query, n) for n in best.all_names) >= s.vendor_match_high else ""
        best = max(load_catalog(s.seed_path), key=lambda p: desc_similarity(query, p.description))
        return best.sku if desc_similarity(query, best.description) >= s.catalog_match_min else ""

    def answer(kind: str, query: str, s) -> str:
        if kind == "product":
            m = match_product(LineItem(description=query), load_catalog(s.seed_path), s)
            return m["product"].sku if m["product"] else ""
        if kind == "vendor":
            m = match_vendor(query, settings=s)
            return m["vendor_id"] if m["status"] == "matched" else ""
        return retrieve_clauses([query], s, k=1)[0]["heading"]

    print(f"{'kind':8s} {'split':5s} {'mode':8s} {'correct':>9s} {'false accepts':>14s}")
    for kind in ("product", "vendor", "policy"):
        for split in ("dev", "test"):
            rows = [c for c in cases if c["kind"] == kind and c["split"] == split]
            modes = [("lexical", "none"), ("hybrid", hybrid)]
            for mode, provider in ([("fuzzy", "none")] if kind != "policy" else []) + modes:
                os.environ["EMBEDDING_PROVIDER"] = provider
                reset_settings_cache()
                reset_index_cache()
                s = get_settings()
                build_knowledge_base(s, if_missing=True)
                fn = fuzzy_only if mode == "fuzzy" else answer
                got = [(c, fn(kind, c["query"], s)) for c in rows]
                ok = sum(a == c["expected"] for c, a in got)
                wrong = [(c, a) for c, a in got if a and a != c["expected"]]
                print(f"{kind:8s} {split:5s} {mode:8s} {ok:>4d}/{len(rows):<4d} {len(wrong):>14d}")
                if args.verbose:
                    for c, a in got:
                        if a != c["expected"]:
                            print(f"      {c['query']!r}: expected {c['expected'] or '(no match)'}, "
                                  f"got {a or '(no match)'}")


if __name__ == "__main__":
    main()
