"""Streamlit review UI - where uncertain POs live until a person acts on them.

    streamlit run review_ui/app.py

Source document on the left, extracted fields on the right with low-confidence fields and flags
highlighted, the RAG evidence (vendor candidates, catalog price checks, cited policy clauses), and
approve / reject actions that close the loop (and are audit-logged).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# This file is itself named app.py: drop its folder from sys.path (and any shadowing module)
# so `import app` resolves to the project package, not to this script.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sqlmodel import select  # noqa: E402

from app.actions import review as review_actions  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import session_scope  # noqa: E402
from app.models import AuditLog, LineItemRecord, PurchaseOrderRecord, ReviewQueueItem  # noqa: E402
from app.rag.store import load_vendors  # noqa: E402

settings = get_settings()
st.set_page_config(page_title="PO Review Queue", page_icon=":clipboard:", layout="wide")

FLAG_HELP = {
    "vendor_unknown": "Vendor is not in the approved-vendor registry",
    "vendor_blocked": "Vendor is BLOCKED in the registry",
    "vendor_ambiguous": "Vendor matches more than one registry record",
    "vendor_low_match": "Vendor match is weak",
    "vendor_tax_id_mismatch": "Tax ID differs from the registry",
    "price_anomaly": "A line price is outside catalog tolerance",
    "unknown_item": "A line item is not in the catalog",
    "over_threshold": "Above the auto-approval limit - director sign-off",
    "exceeds_po_limit": "Above the single-PO hard limit",
    "noncompliant_terms": "Payment terms not allowed by policy",
    "non_base_currency": "Not in the base currency",
    "policy_violation": "Policy reviewer found a violation",
    "math_mismatch": "Numbers don't add up",
    "low_extraction_confidence": "A critical field was read with low confidence",
    "missing_fields": "Required fields are missing",
}


def jl(s: str, default):
    try:
        return json.loads(s) if s else default
    except json.JSONDecodeError:
        return default


def conf_badge(c) -> str:
    if c is None:
        return "-"
    color = "red" if c < settings.critical_field_confidence_min else ("orange" if c < 0.9 else "green")
    return f":{color}[{c:.2f}]"


def show_source(item: ReviewQueueItem) -> None:
    path = Path(item.source_file)
    if not path.exists():
        st.warning(f"Source file not found: {path}")
        return
    st.caption(f"Source: `{path.name}` ({item.doc_type or 'unknown type'})")
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            import pymupdf as fitz

            with fitz.open(path) as doc:
                if doc.needs_pass:
                    st.error("Password-protected PDF - cannot preview")
                    return
                for i in range(min(doc.page_count, 5)):
                    st.image(doc[i].get_pixmap(dpi=110).tobytes("png"), caption=f"page {i + 1}",
                             width="stretch")
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            st.image(str(path), width="stretch")
        elif suffix == ".csv":
            st.dataframe(pd.read_csv(path, dtype=str).fillna(""), width="stretch")
        else:
            st.info("No preview for this file type")
    except Exception as exc:  # a broken file is exactly what lands here - don't crash the UI
        st.error(f"Preview failed: {exc}")
    with open(path, "rb") as fh:
        st.download_button("Download original", fh.read(), file_name=path.name, key=f"dl-{item.id}")


def show_item(item: ReviewQueueItem) -> None:
    flags = jl(item.flags_json, [])
    reasons = jl(item.reasons_json, {})
    po = jl(item.extracted_json, {})
    vm = jl(item.vendor_match_json, {})
    checks = jl(item.price_checks_json, [])
    policy = jl(item.policy_findings_json, {})
    errors = jl(item.errors_json, [])
    signals = jl(item.signals_json, {})

    st.subheader(f"#{item.id} · {item.po_number or '(no PO number)'} · {item.vendor_name_raw or 'unknown vendor'}")
    st.markdown("**Why this needs review**")
    for f in flags:
        st.error(f"**{f}** - {reasons.get(f) or FLAG_HELP.get(f, '')}", icon="⚠️")

    left, right = st.columns([5, 6], gap="large")
    with left:
        show_source(item)
    with right:
        if po:
            st.markdown("**Extracted header** (confidence: red < threshold)")
            rows = []
            for f in ("po_number", "po_date", "vendor_name", "vendor_tax_id", "buyer_department", "currency",
                      "payment_terms", "delivery_date"):
                c = po.get(f) or {}
                rows.append(f"| {f} | {c.get('value') if c.get('value') is not None else '**MISSING**'} | "
                            f"{conf_badge(c.get('confidence'))} | {(c.get('evidence') or '')[:60]} |")
            st.markdown("| field | value | conf. | evidence |\n|---|---|---|---|\n" + "\n".join(rows))
            cur = (po.get("currency") or {}).get("value") or ""
            st.markdown(
                f"Subtotal **{po.get('subtotal')}** · Tax **{po.get('tax')}** · Shipping **{po.get('shipping')}** · "
                f"Total **{cur} {po.get('total')}** {conf_badge(po.get('totals_confidence'))} · "
                f"method `{po.get('extraction_method')}`")
            if po.get("extraction_warnings"):
                with st.expander("Extraction warnings"):
                    for w in po["extraction_warnings"]:
                        st.write(f"- {w}")

            st.markdown("**Line items vs catalog**")
            by_line = {c.get("line_no"): c for c in checks}
            table = []
            for li in po.get("line_items", []):
                c = by_line.get(li.get("line_no"), {})
                table.append({"#": li.get("line_no"), "sku": li.get("sku"), "description": li.get("description"),
                              "qty": li.get("quantity"), "unit": li.get("unit"), "unit_price": li.get("unit_price"),
                              "line_total": li.get("line_total"), "catalog_price": c.get("catalog_unit_price"),
                              "dev_%": c.get("deviation_pct"), "check": c.get("status", "-"),
                              "conf": li.get("confidence")})
            if table:
                df = pd.DataFrame(table)
                st.dataframe(df.style.apply(
                    lambda r: ["background-color: rgba(255,80,80,.25)" if r["check"] in
                               ("price_anomaly", "unknown_item") else "" for _ in r], axis=1),
                    width="stretch", hide_index=True)
        else:
            st.warning("Nothing could be extracted from this document.")

        with st.expander("Vendor match (RAG)", expanded="vendor_unknown" in flags or "vendor_ambiguous" in flags):
            st.write(vm.get("reason", ""))
            st.json({k: vm.get(k) for k in ("status", "vendor_id", "canonical_name", "vendor_status", "score",
                                            "method", "tax_id_match")})
            if vm.get("candidates"):
                st.dataframe(pd.DataFrame(vm["candidates"]), width="stretch", hide_index=True)
        with st.expander("Policy findings & retrieved clauses (RAG)"):
            for f in policy.get("findings", []):
                st.write(f"- **{f.get('code')}** ({f.get('severity')}, {f.get('source')}): {f.get('message')}"
                         + (f" - _clause: {f['clause']}_" if f.get("clause") else ""))
            for c in policy.get("clauses", []):
                st.caption(f"{c['heading']} (score {c['score']})")
                st.text(c["text"][:600])
        with st.expander("Signals"):
            st.json(signals)
        if errors:
            with st.expander("Pipeline errors", expanded=True):
                st.json(errors)

    st.divider()
    st.markdown("**Decision**")
    a, b = st.columns(2)
    reviewer = st.text_input("Reviewer name", key=f"rev-{item.id}", value=st.session_state.get("reviewer", ""))
    if reviewer:
        st.session_state["reviewer"] = reviewer
    with a:
        vendors = load_vendors(settings.seed_path)
        options = ["(keep as extracted)"] + [f"{v.vendor_id} · {v.name}" + (" [BLOCKED]" if v.status == "blocked" else "")
                                             for v in vendors]
        default_idx = 0
        if vm.get("vendor_id"):
            default_idx = next((i for i, o in enumerate(options) if o.startswith(vm["vendor_id"])), 0)
        choice = st.selectbox("Vendor record", options, index=default_idx, key=f"vend-{item.id}")
        learn = st.checkbox("Remember this spelling as an alias of the chosen vendor", key=f"learn-{item.id}",
                            help="Vendor alias learning: the next PO with this spelling matches automatically")
        note = st.text_input("Approval note (e.g. director sign-off reference)", key=f"note-{item.id}")
        if st.button("✅ Approve & file", key=f"approve-{item.id}", type="primary", disabled=not po):
            override = None if choice.startswith("(") else choice.split(" · ")[0]
            if override and "[BLOCKED]" in choice:
                st.error("Cannot approve against a BLOCKED vendor.")
            else:
                try:
                    po_id = review_actions.approve(item.id, reviewer, note, override, learn)
                    st.success(f"Approved - filed as purchase order #{po_id}")
                    st.rerun()
                except review_actions.ReviewError as exc:
                    st.error(str(exc))
    with b:
        reason = st.text_area("Rejection reason", key=f"reason-{item.id}")
        if st.button("❌ Reject", key=f"reject-{item.id}"):
            try:
                review_actions.reject(item.id, reviewer, reason)
                st.warning("Rejected")
                st.rerun()
            except review_actions.ReviewError as exc:
                st.error(str(exc))


# ------------------------------------------------------------------ page
st.title("📋 PO Review Queue")
st.caption(f"Auto-approval limit {settings.auto_approval_limit:,.0f} {settings.base_currency} · "
           f"price tolerance ±{settings.price_tolerance_pct:g}% · confidence ≥ {settings.critical_field_confidence_min}")

tab_queue, tab_approved, tab_audit = st.tabs(["Needs review", "Filed purchase orders", "Audit log"])

with tab_queue:
    pending = review_actions.pending_items(settings)
    if not pending:
        st.success("Queue is empty - nothing needs a human right now.")
    else:
        labels = [f"#{i.id} · {i.po_number or '?'} · {i.vendor_name_raw or '?'} · "
                  f"{', '.join(jl(i.flags_json, []))}" for i in pending]
        idx = st.radio("Pending items", range(len(pending)), format_func=lambda k: labels[k], label_visibility="collapsed")
        st.divider()
        show_item(pending[idx])
    if st.button("↻ Refresh"):
        st.rerun()

with tab_approved:
    with session_scope(settings) as s:
        pos = s.exec(select(PurchaseOrderRecord).order_by(PurchaseOrderRecord.created_at.desc()).limit(200)).all()
        st.dataframe(pd.DataFrame([{
            "id": p.id, "po_number": p.po_number, "vendor": p.vendor_canonical_name or p.vendor_name_raw,
            "vendor_id": p.vendor_id, "total": p.total, "currency": p.currency, "status": p.status,
            "approved_by": p.approved_by, "filed_at": p.created_at} for p in pos]),
            width="stretch", hide_index=True)
        if pos:
            pick = st.selectbox("Line items for", [p.id for p in pos],
                                format_func=lambda i: next(f"#{p.id} {p.po_number}" for p in pos if p.id == i))
            lines = s.exec(select(LineItemRecord).where(LineItemRecord.po_id == pick)).all()
            st.dataframe(pd.DataFrame([li.model_dump() for li in lines]), width="stretch", hide_index=True)
        reviewed = s.exec(select(ReviewQueueItem).where(ReviewQueueItem.status != "pending_review")
                          .order_by(ReviewQueueItem.reviewed_at.desc())).all()
        if reviewed:
            st.markdown("**Closed review items**")
            st.dataframe(pd.DataFrame([{"id": r.id, "po_number": r.po_number, "status": r.status,
                                        "reviewer": r.reviewer, "note": r.review_note,
                                        "flags": ", ".join(jl(r.flags_json, [])), "reviewed_at": r.reviewed_at}
                                       for r in reviewed]), width="stretch", hide_index=True)

with tab_audit:
    with session_scope(settings) as s:
        logs = s.exec(select(AuditLog).order_by(AuditLog.id.desc()).limit(300)).all()
        st.dataframe(pd.DataFrame([{"id": a.id, "ts": a.ts, "event": a.event, "actor": a.actor, "job": a.job_id,
                                    "decision": a.decision, "flags": ", ".join(jl(a.flags_json, []))}
                                   for a in logs]), width="stretch", hide_index=True)
        if logs:
            pick = st.selectbox("Details for audit entry", [a.id for a in logs])
            st.json(jl(next(a for a in logs if a.id == pick).details_json, {}))
