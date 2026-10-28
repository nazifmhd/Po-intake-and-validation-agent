"""Tiny eval harness: measure extraction field accuracy and routing accuracy against labelled POs.

    python scripts/eval_extraction.py [--samples data/samples] [--provider anthropic|openai|heuristic]

Field accuracy  - every header field, totals, and per-line sku/qty/unit_price/line_total vs ground truth
Routing accuracy - full pipeline decision + flag set vs the expected outcome in each label
Runs against a throwaway database so it never touches real records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def norm(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    return " ".join(str(v).split()).lower()


def compare(expected, got) -> bool:
    e, g = norm(expected), norm(got)
    if isinstance(e, float) and isinstance(g, float):
        return abs(e - g) <= 0.011
    return e == g


def score_po(label: dict, po) -> tuple[int, int, list[str]]:
    misses, total, ok = [], 0, 0
    for field, expected in label["fields"].items():
        got = getattr(po, field)
        got = got.value if hasattr(got, "value") else got
        total += 1
        if compare(expected, got):
            ok += 1
        else:
            misses.append(f"{field}: expected {expected!r}, got {got!r}")
    exp_lines = label["line_items"]
    for i, el in enumerate(exp_lines):
        li = po.line_items[i] if i < len(po.line_items) else None
        for k, ev in el.items():
            total += 1
            gv = getattr(li, k) if li else None
            if compare(ev, gv):
                ok += 1
            else:
                misses.append(f"line {i + 1}.{k}: expected {ev!r}, got {gv!r}")
    if len(po.line_items) != len(exp_lines):
        misses.append(f"line count: expected {len(exp_lines)}, got {len(po.line_items)}")
    return ok, total, misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--provider", default=None, help="override LLM_PROVIDER")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="po-eval-"))
    os.environ["DATABASE_URL"] = f"sqlite:///{(tmp / 'eval.db').as_posix()}"
    os.environ["STORAGE_DIR"] = str(tmp / "storage")
    os.environ["SLACK_WEBHOOK_URL"] = ""
    os.environ["GOOGLE_SHEETS_ENABLED"] = "false"
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    import logging
    logging.disable(logging.ERROR)
    from app.config import get_settings
    from app.extraction.extractor import extract
    from app.graph.build import run_pipeline
    from app.ingestion.loaders import load_document
    from app.intake.common import split_csv_batch

    settings = get_settings()
    samples = Path(args.samples)
    print(f"LLM provider: {settings.llm_provider} | embeddings: {settings.embedding_provider}\n")
    print(f"{'document':34s} {'fields':>9s} {'acc':>6s}  {'routing':8s} decision / flags")
    print("-" * 100)

    grand_ok = grand_total = route_ok = route_total = 0
    for label_path in sorted((samples / "labels").glob("*.json")):
        label = json.loads(label_path.read_text())
        src = samples / label["file"]
        units = []
        if "batch" in label:
            parts = split_csv_batch(src.read_bytes())
            for (po_no, content), sub in zip(parts, label["batch"]):
                p = tmp / f"{label_path.stem}_{po_no}.csv"
                p.write_bytes(content)
                units.append((f"{label_path.stem}[{po_no}]", p, sub))
        else:
            units.append((label_path.stem, src, label))

        for name, path, lab in units:
            try:
                po = extract(load_document(path, tmp / "pages" / name))
                ok, total, misses = score_po(lab, po)
            except Exception as exc:
                ok, total, misses = 0, len(lab["fields"]) + 4 * len(lab["line_items"]), [f"extraction failed: {exc}"]
            grand_ok += ok
            grand_total += total

            out = run_pipeline(str(path))
            exp = lab["expected"]
            routed = out["decision"] == exp["decision"] and set(out["flags"] or []) == set(exp["flags"])
            route_ok += routed
            route_total += 1
            print(f"{name:34s} {ok:>4d}/{total:<4d} {ok / total:6.0%}  {'PASS' if routed else 'FAIL':8s} "
                  f"{out['decision']} {out['flags']}" + ("" if routed else f"  (expected {exp['decision']} {exp['flags']})"))
            if args.verbose:
                for m in misses:
                    print(f"      - {m}")

    print("-" * 100)
    print(f"Field accuracy:   {grand_ok}/{grand_total} = {grand_ok / grand_total:.1%}")
    print(f"Routing accuracy: {route_ok}/{route_total} = {route_ok / route_total:.1%}")


if __name__ == "__main__":
    main()
